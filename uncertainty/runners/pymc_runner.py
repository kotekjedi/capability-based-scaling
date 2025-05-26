"""
Runner implementation for PyMC-based uncertainty models (e.g., Bayesian Linear Regression).
"""

import copy
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm  # For type hinting idata
from matplotlib.lines import Line2D
from scipy.special import expit  # Need expit for conversion

from ..metrics import (
    calculate_calibration_metrics,  # Although calibration isn't standard for PyMC preds
)

# Import the PyMC model functions
from ..models.linear_model import fit_linear_model, logit_transform
from ..models.linear_model import predict as predict_linear
from ..plotting.plotting import (  # Keep plot_uncertainty for BMA
    plot_linear_model_dual,
    plot_uncertainty,
)
from ..runner import UncertaintyModelRunner


class UncertaintyPyMCModelRunner(UncertaintyModelRunner):
    """
    Implementation of UncertaintyModelRunner for PyMC linear models.

    Handles PyMC model fitting, prediction, and Bayesian Model Averaging (BMA).
    Accepts per-model priors.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        capability_calculator,
        priors_dict: Dict[str, Dict[str, float]],  # Model_key -> prior_params dict
        test_df: Optional[pd.DataFrame] = None,  # Add test_df here
        calibration_alpha: float = 0.05,  # For potential calibration later
        **kwargs,
    ):
        """
        Initializes the PyMC model runner.

        Args:
            df: The input DataFrame containing raw data.
            capability_calculator: Function that calculates capability difference.
            priors_dict: Dictionary mapping model keys to their specific prior parameters
                         for fit_linear_model.
            test_df: Optional DataFrame containing test data points.
            calibration_alpha: Significance level for potential future calibration metrics.
            **kwargs: Other arguments passed to the parent class.
        """
        self.priors_dict = priors_dict
        self.calibration_alpha = calibration_alpha  # Store even if not used immediately
        self.test_df = test_df  # Store test_df

        # Initialize with PyMC-specific metrics_output_dir
        kwargs["metrics_output_dir"] = kwargs.get(
            "metrics_output_dir", "model_metrics_pymc"
        )

        # Ensure test_size is not passed if present (handled by parent, but good practice)
        kwargs.pop("test_size", None)

        super().__init__(df, capability_calculator, **kwargs)
        print(
            f"PyMC model runner initialized. Priors provided for {len(priors_dict)} models."
        )

    def _prepare_model_specific_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares PyMC-specific data: logit transformation of ASR."""
        print("Calculating logit(ASR)...")
        logit_asr_col = self.processed_df_col_prefix + "logit_asr"

        # Use a context manager for floating point errors during logit
        with np.errstate(divide="ignore", invalid="ignore"):
            df[logit_asr_col] = logit_transform(df[self.asr_col])

        rows_before = len(df)
        # Check for NaNs introduced by logit_transform or already present in asr_col/cap_diff_col
        required_cols_for_fit = [self.capability_diff_col, self.asr_col, logit_asr_col]
        df.dropna(subset=required_cols_for_fit, inplace=True)
        rows_after = len(df)

        if rows_after < rows_before:
            print(
                f"Removed {rows_before - rows_after} rows with NaN in required columns ({', '.join(required_cols_for_fit)})."
            )

        return df

    def fit_predict_model(
        self, model_key: str, make_plot: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fits PyMC model, predicts, calculates metrics for a single model_key."""
        if self.processed_df is None:
            raise RuntimeError("Processed data is not available. Check initialization.")

        print(f"\n--- Processing model: {model_key} ---")
        model_data_df = self.processed_df[
            self.processed_df[self.target_model_key_col] == model_key
        ].copy()

        if model_data_df.empty:
            print(f"Skipping {model_key}: No data after preparation.")
            self.model_results[model_key] = {
                "error": "No data available for this model key after preparation."
            }
            return None

        # Retrieve priors for this model
        model_priors = self.priors_dict.get(model_key)
        if model_priors is None:
            print(f"Skipping {model_key}: No priors found in priors_dict.")
            self.model_results[model_key] = {
                "error": f"Priors not found for model key '{model_key}'."
            }
            return None
        print(f"Using priors: {model_priors}")

        # Extract data - use already processed columns
        x_train = model_data_df[self.capability_diff_col].values
        y_train = model_data_df[self.asr_col].values
        y_logit_train = model_data_df[self.processed_df_col_prefix + "logit_asr"].values
        attack_train = (
            model_data_df[self.attack_col].values
            if self.attack_col in model_data_df
            else None
        )

        # --- Fit ---
        print(f"Fitting PyMC model...")
        try:
            # Assuming fit_linear_model returns (model, idata, y_logit_used)
            # Adapt if the return signature is different
            _model_obj, idata, y_logit_used_in_fit = fit_linear_model(
                x_train,
                y_train,
                prior_params=model_priors,
                random_seed=self.random_state,
                progressbar=False,  # Add progressbar=False?
            )
            if idata is None:
                raise RuntimeError("fit_linear_model returned None for idata.")

            # It's useful to know which data points were *actually* used if fit_linear_model filters internally
            # Assuming y_logit_used_in_fit corresponds to the rows used.
            # We need to find the original x, y, attack for those rows.
            # This requires knowing how fit_linear_model handles NaNs or filtering.
            # For now, assume it used all provided non-NaN data from _prepare_...
            x_train_used_in_fit = x_train
            y_train_used_in_fit = y_train
            attack_train_used_in_fit = attack_train

            # --- Calculate Mean W and B ---
            mean_w, mean_b = None, None
            if idata is not None and "posterior" in idata:
                try:
                    mean_w = idata.posterior["w"].mean().item()
                    mean_b = idata.posterior["b"].mean().item()
                    print(f"  Mean w: {mean_w:.4f}, Mean b: {mean_b:.4f}")
                except Exception as e:
                    print(f"  Could not extract mean w or b from idata: {e}")

        except Exception as e:
            print(f"  PyMC fitting failed: {e}")
            self.model_results[model_key] = {"error": f"Fitting failed: {e}"}
            return None

        # --- Predict & Calculate Metrics ---
        print("Generating predictions and calculating metrics...")
        train_metrics_result = {}
        grid_preds_result = {}
        try:
            # Predict on training data to get metrics
            train_metrics_result = predict_linear(
                idata,
                x_train_used_in_fit,
                y_true=y_train_used_in_fit,
                y_true_logit=y_logit_used_in_fit,
            )
            # Predict on grid for plotting
            grid_preds_result = predict_linear(idata, self.x_grid, return_logit=True)

            # --- Debug: Print grid_preds_result keys and check type ---
            print(f"  Debug: grid_preds_result type: {type(grid_preds_result)}")
            if isinstance(grid_preds_result, dict):
                print(
                    f"  Debug: grid_preds_result keys: {list(grid_preds_result.keys())}"
                )
            # --- End Debug ---

            # Basic check on results - Check both train and grid results
            if (
                not train_metrics_result
                or not grid_preds_result
                or not isinstance(grid_preds_result, dict)
            ):
                error_msg = "Prediction function returned invalid results."
                if not isinstance(grid_preds_result, dict):
                    error_msg += f" Expected dict for grid predictions, got {type(grid_preds_result)}."
                raise RuntimeError(error_msg)

        except Exception as e:
            print(f"  PyMC prediction/metric calculation failed: {e}")
            self.model_results[model_key] = {
                "error": f"Prediction failed: {e}",
                "idata": idata,  # Store idata even if prediction fails
            }
            return None

        # --- Store Results ---
        current_model_metrics = {"train": train_metrics_result.get("train_metrics", {})}

        # --- Calculate Train Calibration Metrics ---
        train_calibration_metrics = {}
        train_preds_info = train_metrics_result.get(
            "train_predictions"
        )  # Get the sub-dict first
        if train_preds_info:  # Check if the sub-dict exists
            train_samples_prob = train_preds_info.get("samples_prob")
            print(
                f"  DEBUG fit_predict: Found 'train_predictions'. Checking 'samples_prob'..."
            )  # DEBUG
            if (
                train_samples_prob is not None
            ):  # Check if samples_prob key exists and is not None
                print(
                    f"  DEBUG fit_predict: Found 'samples_prob' with shape {train_samples_prob.shape}. y_train_used_in_fit length: {len(y_train_used_in_fit)}"
                )  # DEBUG
                if len(y_train_used_in_fit) > 0 and train_samples_prob.shape[1] == len(
                    y_train_used_in_fit
                ):
                    print(
                        f"  DEBUG fit_predict: Shapes match. Attempting calibration calculation..."
                    )  # DEBUG
                    try:
                        train_calibration_metrics = calculate_calibration_metrics(
                            y_true=y_train_used_in_fit,
                            samples_pred_prob=train_samples_prob,
                            alpha=self.calibration_alpha,
                        )
                        print(
                            f"  DEBUG fit_predict: Calculated train calibration metrics: {list(train_calibration_metrics.keys())}"
                        )  # DEBUG
                    except Exception as cal_e:
                        print(
                            f"  Warning: Could not calculate train calibration metrics: {cal_e}"
                        )  # DEBUG
                else:
                    print(
                        "  Warning: Could not calculate train calibration - SHAPE MISMATCH between train samples and y_true."
                    )  # DEBUG
            else:
                print(
                    "  Warning: 'samples_prob' key MISSING or None within 'train_predictions'. Cannot calculate train calibration."
                )  # DEBUG
        else:
            print(
                "  Warning: 'train_predictions' key NOT FOUND in prediction result. Cannot calculate train calibration."
            )  # DEBUG
        # Add the calculated train calibration metrics to the main metrics dict
        current_model_metrics["train_calibration"] = train_calibration_metrics
        print(
            f"  DEBUG fit_predict: Storing train_calibration: {current_model_metrics['train_calibration']}"
        )  # DEBUG
        # --- End Train Calibration ---

        # --- Adapt to actual keys returned by predict_linear ---
        # Check if the expected keys exist, otherwise use the ones from debug log
        grid_samples_logit_key = (
            "samples_logit" if "samples_logit" in grid_preds_result else "samples"
        )
        grid_samples_prob_key = (
            "samples_prob" if "samples_prob" in grid_preds_result else None
        )

        stored_grid_preds = {}
        if grid_samples_logit_key in grid_preds_result:
            stored_grid_preds["samples_logit"] = grid_preds_result[
                grid_samples_logit_key
            ]
            # Try to derive prob samples if not directly available
            if grid_samples_prob_key is None:
                try:
                    print(
                        f"  Note: Deriving samples_prob from {grid_samples_logit_key}."
                    )
                    stored_grid_preds["samples_prob"] = expit(
                        grid_preds_result[grid_samples_logit_key]
                    )
                    grid_samples_prob_key = "samples_prob"  # Mark as available now
                except Exception as e:
                    print(
                        f"  Warning: Could not derive samples_prob from logit samples: {e}"
                    )
            elif grid_samples_prob_key in grid_preds_result:  # Key exists
                stored_grid_preds[grid_samples_prob_key] = grid_preds_result[
                    grid_samples_prob_key
                ]
        else:
            print(
                f"  Warning: Could not find logit samples under key '{grid_samples_logit_key}' or 'samples_logit' in grid_preds_result."
            )

        # Copy other keys from grid_preds_result if needed (e.g., means, intervals)
        # For BMA/plotting, we primarily need the samples. Add others if required.
        if "mean_prob" in grid_preds_result:
            stored_grid_preds["mean_prob"] = grid_preds_result["mean_prob"]
        elif (
            "mean" in grid_preds_result and grid_samples_prob_key == "samples_prob"
        ):  # If we derived prob samples, derive mean prob
            stored_grid_preds["mean_prob"] = np.mean(
                stored_grid_preds["samples_prob"], axis=0
            )
        # Add mean_logit etc. if needed

        self.model_results[model_key] = {
            "idata": idata,
            # "grid_predictions": grid_preds_result, # Store the adapted dict instead
            "grid_predictions": stored_grid_preds,
            "metrics": current_model_metrics,  # Contains train metrics
            "train_data_info": {
                "x": x_train_used_in_fit,
                "y": y_train_used_in_fit,
                "attack": attack_train_used_in_fit,
                "y_logit": y_logit_used_in_fit,
            },
            "mean_w": mean_w,
            "mean_b": mean_b,
        }

        # --- Optional Plot ---
        if make_plot:
            print("Generating dual plot...")
            # Use the standardized keys from stored_grid_preds
            if (
                "samples_prob" in stored_grid_preds
                and "samples_logit" in stored_grid_preds
            ):
                fig_dual, axes_dual = plot_linear_model_dual(
                    x_grid=self.x_grid,
                    samples_prob=stored_grid_preds["samples_prob"],
                    samples_logit=stored_grid_preds["samples_logit"],
                    x_data=x_train_used_in_fit,
                    y_data=y_train_used_in_fit,
                    attack_data=attack_train_used_in_fit,
                    y_logit=y_logit_used_in_fit,
                    # No test data here
                    title=f"PyMC Linear Model Fit for {model_key}",
                    metrics=current_model_metrics,  # Pass combined train/test metrics
                    mean_w=mean_w,
                    mean_b=mean_b,
                    xlim=self.x_grid[[0, -1]],
                    ylim=(-0.05, 1.05),
                    train_legend_label="Data",  # Simple label for single model plot
                )
                plt.show()
            else:
                # Update skip message
                print(
                    f"Skipping dual plot: Standardized keys ('samples_prob', 'samples_logit') not found in stored_grid_preds. Available keys: {list(stored_grid_preds.keys())}"
                )

        print(f"--- Finished processing model: {model_key} ---")
        return self.model_results[model_key]

    def get_aggregated_metrics(
        self, model_keys_to_include: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculates and returns average train metrics across specified models."""
        if model_keys_to_include is None:
            model_keys_to_include = [
                k for k, v in self.model_results.items() if "error" not in v
            ]

        if not model_keys_to_include:
            print("No models specified or no successful results found for aggregation.")
            return {}

        aggregated_metrics = defaultdict(list)
        first_valid_metrics_container = (
            None  # Holds {'train': {...}, 'train_calibration': {...}}
        )
        metrics_containers_to_average = []  # List of valid containers

        # Collect valid metrics containers from specified models
        print(f"Aggregating metrics containers for models: {model_keys_to_include}")
        for key in model_keys_to_include:
            result = self.model_results.get(key)
            # Check if result and 'metrics' key exist and is a dict
            if result and isinstance(result.get("metrics"), dict):
                metrics_container = result["metrics"]
                # Check if it has *any* valid data (train or train_calib)
                has_valid_train = (
                    isinstance(metrics_container.get("train"), dict)
                    and metrics_container["train"]
                )
                has_valid_calib = (
                    isinstance(metrics_container.get("train_calibration"), dict)
                    and metrics_container["train_calibration"]
                )

                if has_valid_train or has_valid_calib:
                    metrics_containers_to_average.append(metrics_container)
                    if first_valid_metrics_container is None:
                        first_valid_metrics_container = metrics_container
                    # Optional: More detailed logging if needed
                    # if has_valid_train: print(f"  Found valid train metrics for {key}")
                    # if has_valid_calib: print(f"  Found valid train calibration metrics for {key}")
                else:
                    print(
                        f"  Found metrics container for {key}, but both 'train' and 'train_calibration' are empty or invalid. Skipping."
                    )
            else:
                print(
                    f"Warning: No valid 'metrics' dict found for model '{key}'. Skipping."
                )

        if not metrics_containers_to_average:
            print("No valid metrics containers collected for aggregation.")
            return {}

        # Ensure we found a container to derive keys from
        if first_valid_metrics_container is None:
            print(
                "Warning: Could not determine metric keys from any model for aggregation."
            )
            return {}

        # Dynamically get keys from the first valid container
        metric_keys_train = list(first_valid_metrics_container.get("train", {}).keys())
        calib_keys_train = list(
            first_valid_metrics_container.get("train_calibration", {}).keys()
        )

        print(f"Aggregating Train Metrics Keys: {metric_keys_train}")
        if calib_keys_train:
            print(f"Aggregating Train Calibration Metrics Keys: {calib_keys_train}")
        else:
            print("No Train Calibration Metric Keys found to aggregate.")

        # Aggregate metrics using a single loop over valid containers
        aggregated_values_debug = defaultdict(list)  # Keep for debug
        for metrics_container in metrics_containers_to_average:
            # Aggregate Train Metrics
            train_metrics_data = metrics_container.get("train")
            if isinstance(train_metrics_data, dict):
                for key in metric_keys_train:
                    if key in train_metrics_data and pd.notna(train_metrics_data[key]):
                        value = train_metrics_data[key]
                        aggregated_metrics["avg_train_" + key].append(value)
                        aggregated_values_debug[key].append(value)

            # Aggregate Train Calibration Metrics
            train_calib_data = metrics_container.get("train_calibration")
            if isinstance(train_calib_data, dict):
                for calib_key in calib_keys_train:
                    if calib_key in train_calib_data and pd.notna(
                        train_calib_data[calib_key]
                    ):
                        value = train_calib_data[calib_key]
                        aggregated_metrics["avg_train_calib_" + calib_key].append(value)
                        aggregated_values_debug[calib_key].append(value)

        # Calculate averages
        final_avg_metrics = {
            key: (np.mean(values), np.std(values))
            for key, values in aggregated_metrics.items()
            if values
        }

        if not final_avg_metrics:
            print("Warning: Aggregation resulted in empty metrics dictionary.")

        return final_avg_metrics

    def plot_aggregated_uncertainty(
        self,
        model_keys_to_include: Optional[List[str]] = None,
        test_df: Optional[pd.DataFrame] = None,  # Argument takes precedence
        xlim: Optional[tuple] = (-2.5, 2.5),
        ylim: tuple = (-0.05, 1.05),
        title: Optional[str] = None,
        color: str = "maroon",  # Default BMA color
        sigmas_to_plot: List[int] = [1, 2],
        plot_individual_means: bool = True,  # Option to plot individual model means
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Creates a Bayesian Model Averaging (BMA) plot by aggregating PyMC results.
        Optionally overlays provided test data points and calculates BMA test metrics.
        If test_df argument is None, uses self.test_df if available.

        Args:
            model_keys_to_include: List of model keys to include in plot.
            test_df: Optional DataFrame containing test data points. Overrides self.test_df.
            xlim: X-axis limits for the plot.
            ylim: Y-axis limits for the plot.
            title: Plot title.
            color: Base color for the BMA plot.
            sigmas_to_plot: Which standard deviation intervals to plot.
            plot_individual_means: Whether to plot the mean predictions of individual models.

        Returns:
            (fig, ax) Matplotlib figure and axes objects, or (None, None) if plotting fails.
        """
        # --- Determine which test_df to use ---
        effective_test_df = test_df if test_df is not None else self.test_df
        # --- End Determine test_df ---

        if model_keys_to_include is None:
            model_keys_to_include = [
                k
                for k, v in self.model_results.items()
                if "error" not in v and "grid_predictions" in v
            ]

        if not model_keys_to_include:
            print("No models specified or no successful results found for BMA plot.")
            return None, None

        print(
            f"\nAggregating results from {len(model_keys_to_include)} models for BMA plotting..."
        )

        # Collect grid samples and training data
        grid_samples_list = []
        x_train_list, y_train_list, attack_train_list = [], [], []
        individual_means_list = []  # For optional plotting

        n_grid_points = len(self.x_grid)
        expected_shape_cols = n_grid_points

        for key in model_keys_to_include:
            result = self.model_results.get(key)
            if result and "grid_predictions" in result:
                preds = result["grid_predictions"]
                if "samples_prob" in preds:
                    current_samples = preds["samples_prob"]
                    # Shape Check
                    if (
                        isinstance(current_samples, np.ndarray)
                        and current_samples.ndim == 2
                        and current_samples.shape[1] == expected_shape_cols
                    ):
                        grid_samples_list.append(current_samples)
                        # Collect individual mean if plotting them - use standardized key
                        if plot_individual_means and "mean_prob" in preds:
                            individual_means_list.append(preds["mean_prob"])
                        elif plot_individual_means:
                            # Check if mean_prob exists in the stored dict
                            print(
                                f"Warning: Standardized key 'mean_prob' not found in stored grid_predictions for model '{key}' but requested for plotting."
                            )
                    else:
                        print(
                            f"Warning: Skipping grid samples from model '{key}' due to unexpected shape in stored 'samples_prob'."
                        )
                else:
                    print(
                        f"Warning: Standardized key 'samples_prob' not found in stored grid_predictions for model '{key}'."
                    )

                # Append TRAIN data used for this model's fit
                train_info = result.get("train_data_info", {})
                if "x" in train_info:
                    x_train_list.append(train_info["x"])
                if "y" in train_info:
                    y_train_list.append(train_info["y"])
                if "attack" in train_info and train_info["attack"] is not None:
                    attack_train_list.append(train_info["attack"])
                elif (
                    "attack" in train_info
                ):  # Handle case where attack is None but key exists
                    # Need to add placeholder if concatenating later
                    pass  # Or decide how to handle missing attack data across models

        if not grid_samples_list:
            print("No valid prediction samples found to aggregate for BMA plot.")
            return None, None

        # Combine all grid prediction samples
        # Shape: (total_samples, n_grid_points)
        combined_samples_prob = np.vstack(grid_samples_list)

        # Combine training data
        x_train_agg = (
            np.concatenate(x_train_list) if x_train_list else np.array([])
        )  # Concatenate if lists are not empty
        y_train_agg = np.concatenate(y_train_list) if y_train_list else np.array([])
        attack_train_agg = (
            np.concatenate(attack_train_list) if attack_train_list else None
        )

        # Prepare combined individual means if needed
        combined_individual_means = (
            np.vstack(individual_means_list)
            if plot_individual_means and individual_means_list
            else None
        )

        # --- Prepare Test Data (if available) ---
        print("Extracting test data from DataFrame...")
        x_test, y_test, attack_test = None, None, None
        if effective_test_df is not None and not effective_test_df.empty:
            try:
                # Select necessary raw columns only
                cols_to_check = [
                    self.capability_diff_col,
                    self.asr_col,
                    # self.processed_df_col_prefix + "logit_asr", # Removed requirement for pre-calculated logit
                ]
                if self.attack_col:
                    cols_to_check.append(self.attack_col)

                # Check if columns exist before selecting
                missing_cols = [
                    col for col in cols_to_check if col not in effective_test_df.columns
                ]
                if missing_cols:
                    raise KeyError(f"Missing columns in test_df: {missing_cols}")

                # Drop rows where any essential raw column is NaN
                test_df_filtered = effective_test_df[cols_to_check].dropna().copy()

                if not test_df_filtered.empty:
                    x_test = test_df_filtered[self.capability_diff_col].values
                    y_test = test_df_filtered[self.asr_col].values
                    # y_test_logit = test_df_filtered[ # Removed - calculate later if needed
                    #     self.processed_df_col_prefix + "logit_asr"
                    # ].values
                    attack_test = (
                        test_df_filtered[self.attack_col].values
                        if self.attack_col
                        in test_df_filtered.columns  # Check on filtered df
                        else None
                    )
                    print(
                        f"  Found {len(x_test)} valid test points after filtering NaNs."
                    )
                else:
                    print("  Test DataFrame is empty after filtering NaNs.")
            except KeyError as e:
                print(
                    f"  Error extracting test data: Missing column {e}. Ensure test_df has required columns: {cols_to_check}"
                )
                x_test, y_test, attack_test = None, None, None
            except Exception as e:
                print(
                    f"  An unexpected error occurred during test data extraction: {e}"
                )
                x_test, y_test, attack_test = None, None, None
        else:
            print("  No test DataFrame provided or it is empty.")

        # --- End Prepare Test Data ---

        # --- Calculate BMA Test Metrics ---
        # Calculate test metrics if test data is available
        bma_test_metrics = {}
        if x_test is not None and y_test is not None:
            print("\nCalculating BMA test metrics...")

            # Only use valid test data (already filtered for NaNs above)
            n_test_points = len(x_test)
            if n_test_points > 0:
                try:
                    # 1. Make predictions on test points using BMA posteriors
                    # Reshape x_test for broadcasting with samples
                    x_test_reshaped = x_test.reshape(1, -1)  # Shape: (1, n_test)

                    # Find the w and b parameters from each aggregated model's posterior
                    # First collect all w and b samples
                    all_w_samples, all_b_samples = [], []
                    for key in model_keys_to_include:
                        result = self.model_results.get(key)
                        if result and "idata" in result:
                            idata = result["idata"]
                            try:
                                if "posterior" in idata:
                                    w_samples = idata.posterior["w"].values.flatten()
                                    b_samples = idata.posterior["b"].values.flatten()
                                    all_w_samples.append(w_samples)
                                    all_b_samples.append(b_samples)
                            except Exception as e:
                                print(
                                    f"  Warning: Could not extract w/b from idata for model '{key}': {e}"
                                )

                    # If we found any parameters, combine them
                    if all_w_samples and all_b_samples:
                        combined_w = np.concatenate(all_w_samples)
                        combined_b = np.concatenate(all_b_samples)

                        # Calculate predictions in probability space using logistic function
                        # Shape: (n_samples, n_test)
                        logits = (
                            combined_w[:, np.newaxis] * x_test_reshaped
                            + combined_b[:, np.newaxis]
                        )
                        probs = expit(logits)  # Apply sigmoid/expit

                        # Calculate mean prediction for each test point
                        mean_pred_prob = np.mean(
                            probs, axis=0
                        )  # Average over all samples

                        # 2. Calculate metrics
                        # RMSE in probability space
                        rmse_prob = np.sqrt(np.mean((y_test - mean_pred_prob) ** 2))
                        bma_test_metrics["rmse_prob"] = rmse_prob

                        # MAE in probability space
                        mae_prob = np.mean(np.abs(y_test - mean_pred_prob))
                        bma_test_metrics["mae_prob"] = mae_prob

                        # R-squared in probability space
                        ss_total = np.sum((y_test - np.mean(y_test)) ** 2)
                        ss_residual = np.sum((y_test - mean_pred_prob) ** 2)
                        r2_prob = 1 - (ss_residual / ss_total) if ss_total > 0 else 0
                        bma_test_metrics["r2_prob"] = r2_prob

                        # Optional: Calculate logit-space metrics if needed
                        try:
                            # Apply logit transform to y_test (with bounds to avoid infinities)
                            y_test_logit = logit_transform(y_test)
                            valid_logit_mask = ~np.isnan(y_test_logit)

                            if np.any(valid_logit_mask):
                                # Filter valid logit values
                                valid_y_test_logit = y_test_logit[valid_logit_mask]
                                valid_x_test = x_test[valid_logit_mask]

                                # Make logit predictions for valid points
                                valid_logits = (
                                    combined_w[:, np.newaxis]
                                    * valid_x_test.reshape(1, -1)
                                    + combined_b[:, np.newaxis]
                                )
                                mean_pred_logit = np.mean(valid_logits, axis=0)

                                # Logit-space metrics
                                rmse_logit = np.sqrt(
                                    np.mean((valid_y_test_logit - mean_pred_logit) ** 2)
                                )
                                bma_test_metrics["rmse_logit"] = rmse_logit

                                mae_logit = np.mean(
                                    np.abs(valid_y_test_logit - mean_pred_logit)
                                )
                                bma_test_metrics["mae_logit"] = mae_logit
                            else:
                                print(
                                    "  Note: Could not calculate logit-space metrics - all test points resulted in NaN after logit transform."
                                )
                        except Exception as e:
                            print(
                                f"  Warning: Error calculating logit-space metrics: {e}"
                            )

                        # Optional: Calculate calibration metrics if implemented
                        # This would use the BMA posterior probability samples (`probs`)
                        try:
                            from ..metrics import calculate_calibration_metrics

                            calib_metrics = calculate_calibration_metrics(
                                y_true=y_test,
                                samples_pred_prob=probs,  # Corrected argument name
                                alpha=self.calibration_alpha,
                            )
                            bma_test_metrics.update(
                                {
                                    f"bma_test_calib_{k}": v
                                    for k, v in calib_metrics.items()
                                }
                            )
                        except ImportError:
                            print(
                                "  Warning: Could not import calculate_calibration_metrics. Skipping calibration."
                            )
                        except Exception as e:
                            print(
                                f"  Warning: Could not calculate BMA test calibration metrics: {e}"
                            )
                    else:
                        print(
                            "  Warning: Could not find w/b parameters in any model's idata for test prediction."
                        )
                except Exception as e:
                    print(f"  Error during BMA test metric calculation: {e}")
                    bma_test_metrics = {"error": str(e)}
            else:
                print("  No valid test points found after filtering.")

        # Print BMA test metrics
        print("\nBMA Test Metrics:")
        if bma_test_metrics:
            if "error" in bma_test_metrics:
                print(f"  Error: {bma_test_metrics['error']}")
            else:
                for k, v in sorted(bma_test_metrics.items()):
                    print(f"  {k}: {v:.4f}")
        elif effective_test_df is not None:
            print(
                "  Could not calculate test metrics (check warnings above or if test data was valid)."
            )
        else:
            print("  No test data provided.")

        # Add BMA test metrics result to the runner's aggregated metrics for reference
        if bma_test_metrics and "error" not in bma_test_metrics:
            bma_metrics_with_prefix = {
                f"bma_test_{k}": v for k, v in bma_test_metrics.items()
            }
            self._aggregated_metrics = {
                **getattr(
                    self, "_aggregated_metrics", {}
                ),  # Ensure _aggregated_metrics exists
                **bma_metrics_with_prefix,
            }

        # --- End Calculate BMA Test Metrics ---

        # Calculate aggregated TRAIN metrics
        aggregated_train_metrics = self.get_aggregated_metrics(model_keys_to_include)
        print("\nAverage Metrics (Train/Full Data) for BMA Plot:")
        if aggregated_train_metrics:
            for k, v in sorted(aggregated_train_metrics.items()):
                print(f"  {k}: {v:.4f}")
        else:
            print("  No train metrics to display.")

        # Determine plot limits
        if xlim is None:  # If default (-2.5, 2.5) wasn't overridden
            final_xlim = (-2.5, 2.5)  # Keep the default if None passed
        else:
            # Optionally refine limits based on data if xlim was provided but is too narrow/wide
            # For simplicity, just use the provided or default xlim for now.
            final_xlim = xlim

        # Default title
        if title is None:
            title = "Bayesian Model Averaging (BMA) Uncertainty"

        print("\nPlotting BMA uncertainty...")

        # Use the custom plotting function structure from bootstrap runner
        # Pass individual means if requested and available
        fig, ax = self._plot_bma_uncertainty_custom(
            x_grid=self.x_grid,
            samples_prob=combined_samples_prob,
            x_data=x_train_agg,
            y_data=y_train_agg,
            test_data=(x_test, y_test)
            if x_test is not None and y_test is not None
            else None,
            individual_means=combined_individual_means
            if plot_individual_means
            else None,
            title=title,
            xlim=final_xlim,
            ylim=ylim,
            color=color,
            sigmas_to_plot=sigmas_to_plot,
            train_legend_label="Agg. Train Data",  # Label for combined train points
            test_legend_label="Test Data",
        )

        plt.show()
        return fig, ax

    # Adapt the custom plotting function from bootstrap runner for BMA style
    def _plot_bma_uncertainty_custom(
        self,
        x_grid,
        samples_prob,
        x_data=None,
        y_data=None,
        test_data=None,
        individual_means=None,  # Added this
        xlim=(-2.5, 2.5),
        ylim=(0, 1),
        title=None,
        color="maroon",
        sigmas_to_plot=[1, 2],
        train_legend_label="Training Data",
        test_legend_label="Test Data",
    ):
        """Customized plotting function similar to plot_uncertainty_custom for BMA."""
        fig, ax = plt.subplots(figsize=(6, 6))
        plt.rcParams.update({"font.size": 14})
        ax.set_title(title, fontsize=16)

        # Plot aggregated training data
        if x_data is not None and y_data is not None:
            ax.scatter(
                x_data,
                y_data,
                color="gray",
                alpha=0.5,
                label=train_legend_label,
                linewidths=0.5,
            )

        # Plot individual model means (optional)
        if individual_means is not None:
            for i, ind_mean in enumerate(individual_means):
                label = "Individual Means" if i == 0 else None
                ax.plot(
                    x_grid, ind_mean, color="gray", lw=0.8, alpha=0.3, label=label
                )  # Lighter alpha

        # Plot BMA median prediction
        median_prob = np.median(samples_prob, axis=0)
        ax.plot(x_grid, median_prob, color=color, lw=4, label="BMA Median")

        # Plot BMA uncertainty bands
        sigma_levels = sorted(sigmas_to_plot)
        sigma_percentiles = {1: (15.87, 84.13), 2: (2.28, 97.72), 3: (0.13, 99.87)}
        confidence_levels = {1: "68%", 2: "95%", 3: "99.7%"}
        alphas = {1: 0.2, 2: 0.1, 3: 0.05}

        for sigma in sigma_levels:
            alpha = alphas[sigma]
            lower, upper = sigma_percentiles[sigma]
            lower_bound = np.percentile(samples_prob, lower, axis=0)
            upper_bound = np.percentile(samples_prob, upper, axis=0)
            ax.fill_between(
                x_grid,
                lower_bound,
                upper_bound,
                color=color,
                alpha=alpha,
                label=f"±{confidence_levels[sigma]}",
                linewidth=0,
            )

        # Plot test data
        if test_data is not None:
            x_test, y_test = test_data
            ax.scatter(
                x_test,
                y_test,
                color="#FF6347",
                marker="+",
                s=100,
                label=test_legend_label,
                linewidths=2,
                alpha=1.0,
                zorder=10,
            )

        # Format plot
        ax.set_xlabel("Capability Difference", fontsize=16)
        ax.set_ylabel("ASR", fontsize=16)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis="both", which="major", labelsize=14)

        # Legend Formatting (similar to custom plot)
        handles, labels = ax.get_legend_handles_labels()
        desired_order = [train_legend_label, test_legend_label, "BMA Median"]
        if individual_means is not None:
            desired_order.append("Individual Means")
        for sigma in sorted(sigma_levels):
            desired_order.append(f"±{confidence_levels[sigma]}")

        order = []
        handled_indices = set()
        label_to_idx = {lbl: i for i, lbl in enumerate(labels)}
        for item in desired_order:
            if item in label_to_idx:
                idx = label_to_idx[item]
                if idx not in handled_indices:
                    order.append(idx)
                    handled_indices.add(idx)
        # Add remaining unhandled items
        order.extend([i for i, lbl in enumerate(labels) if i not in handled_indices])

        # Filter out potential missing handles/labels due to order logic errors
        valid_handles = [handles[idx] for idx in order if idx < len(handles)]
        valid_labels = [labels[idx] for idx in order if idx < len(labels)]

        ax.legend(valid_handles, valid_labels, frameon=False, fontsize=14, loc="best")

        plt.tight_layout()
        return fig, ax

    def _process_results_for_saving(self) -> Dict[str, Any]:
        """Process PyMC model results to prepare them for saving to JSON."""
        results_copy = copy.deepcopy(self.model_results)
        results_to_save = {}

        for key, value in results_copy.items():
            if isinstance(value, dict):
                value_copy = value
                # Handle idata - Replace with placeholder or summary
                if "idata" in value_copy:
                    # Option 1: Simple placeholder
                    # value_copy['idata'] = f"<PyMC InferenceData object>"
                    # Option 2: Extract summary stats (more useful but complex)
                    try:
                        summary = pm.summary(
                            value_copy["idata"], kind="stats"
                        ).to_dict()
                        value_copy["idata_summary"] = summary
                    except Exception:
                        value_copy["idata_summary"] = "<Could not generate summary>"
                    del value_copy["idata"]  # Remove original idata

                # Handle grid predictions (arrays to placeholders)
                if "grid_predictions" in value_copy:
                    preds = value_copy["grid_predictions"]
                    preds_save = {}
                    # Use the standardized keys we expect now
                    for sample_type in [
                        "samples_prob",
                        "samples_logit",
                        "mean_prob",
                        "mean_logit",
                        # Add other keys if they were copied into stored_grid_preds
                    ]:
                        if sample_type in preds:
                            item = preds[sample_type]
                            shape = getattr(item, "shape", "N/A")
                            preds_save[sample_type] = (
                                f"<{sample_type} array, shape={shape}>"
                            )
                    # Copy other non-array keys if any exist in the stored dict
                    for k, v in preds.items():
                        if k not in preds_save and not hasattr(
                            v, "shape"
                        ):  # Only copy non-array like items
                            preds_save[k] = v
                    value_copy["grid_predictions"] = preds_save

                # Handle train_data_info (arrays to placeholders)
                if "train_data_info" in value_copy:
                    info = value_copy["train_data_info"]
                    info_save = {}
                    for k in ["x", "y", "attack", "y_logit"]:
                        if k in info:
                            item = info[k]
                            shape = (
                                getattr(item, "shape", "N/A")
                                if hasattr(item, "shape")
                                else "Scalar/None"
                            )
                            info_save[k] = f"<{k} array/data, shape={shape}>"
                    value_copy["train_data_info"] = info_save

                results_to_save[key] = value_copy
            else:
                results_to_save[key] = value  # Store errors etc. directly

        return results_to_save
