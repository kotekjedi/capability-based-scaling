import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from ..metrics import calculate_calibration_metrics

# Import the specific model functions needed
from ..models.linear_model_bootstrapped import (
    _calculate_metrics,
    fit_bootstrap_model,
    logit_transform,
    predict_bootstrap,
)
from ..plotting.plotting import (
    _plot_scatter_by_attack,
    plot_linear_model_dual,
    plot_uncertainty,
)
from ..runner import UncertaintyModelRunner  # Import the base class


class UncertaintyBootstrapModelRunner(UncertaintyModelRunner):
    """
    Implementation of UncertaintyModelRunner for bootstrapped linear models.

    Handles bootstrapped linear model fitting, prediction and uncertainty estimation.
    Uses the full dataset provided for each model key for fitting.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        capability_calculator,
        n_bootstrap: int = 1000,
        calibration_alpha: float = 0.05,  # for 95% intervals
        **kwargs,
    ):
        """
        Initializes the bootstrapped model runner.

        Args:
            df: The input DataFrame containing raw data.
            capability_calculator: Function that calculates capability difference.
            n_bootstrap: Number of bootstrap samples for model fitting.
            calibration_alpha: Significance level for calibration metrics.
            **kwargs: Other arguments passed to the parent class (excluding test_size).
        """
        self.n_bootstrap = n_bootstrap
        self.calibration_alpha = calibration_alpha

        # Initialize with bootstrap-specific metrics_output_dir
        kwargs["metrics_output_dir"] = kwargs.get(
            "metrics_output_dir", "model_metrics_bootstrap"
        )

        # Ensure test_size is not passed to parent
        kwargs.pop("test_size", None)

        super().__init__(df, capability_calculator, **kwargs)
        print(
            f"Bootstrap model runner initialized with {n_bootstrap} bootstrap iterations."
        )

    def _prepare_model_specific_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepares bootstrap-specific data: logit transformation of ASR."""
        print("Calculating logit(ASR)...")
        # Calculate logit(ASR) for internal use, handle NaNs
        logit_asr_col = self.processed_df_col_prefix + "logit_asr"
        df[logit_asr_col] = logit_transform(df[self.asr_col])

        # Remove rows where logit_asr is NaN
        rows_before = len(df)
        df.dropna(subset=[logit_asr_col], inplace=True)
        rows_after = len(df)
        if rows_after < rows_before:
            print(f"Removed {rows_before - rows_after} rows with NaN logit(ASR).")

        return df

    def fit_predict_model(
        self, model_key: str, make_plot: bool = False
    ) -> Optional[Dict[str, Any]]:
        """Fits bootstrap model, predicts, calculates metrics for a single model_key using its full data."""
        if self.processed_df is None:
            raise RuntimeError("Processed data is not available. Check initialization.")

        print(f"\n--- Processing model: {model_key} ---")
        model_data_df = self.processed_df[
            self.processed_df[self.target_model_key_col] == model_key
        ].copy()

        if model_data_df.empty:
            print(f"Skipping {model_key}: No data.")
            self.model_results[model_key] = {
                "error": "No data available for this model key."
            }
            return None

        # Extract data - use already processed columns
        x_train = model_data_df[self.capability_diff_col].values
        y_train = model_data_df[
            self.asr_col
        ].values  # Original ASR for metrics/plotting
        y_logit_train = model_data_df[
            self.processed_df_col_prefix + "logit_asr"
        ].values  # Pre-calculated logit
        attack_train = model_data_df[self.attack_col].values

        # --- Fit ---
        print(f"Fitting bootstrap ({self.n_bootstrap} iterations)...")
        try:
            bootstrap_results = fit_bootstrap_model(
                x_train,
                y_train,
                n_bootstrap=self.n_bootstrap,
                random_seed=self.random_state,
            )
            print(
                f"  Fit {bootstrap_results.get('n_valid_bootstrap', 'N/A')} valid models."
            )
            # Get the logit-y and corresponding x/y/attack actually used by fit_bootstrap
            y_logit_train_used_in_fit = bootstrap_results["y_logit"]
            valid_mask_train_fit = ~np.isnan(x_train) & ~np.isnan(
                logit_transform(y_train)
            )
            x_train_used_in_fit = x_train[valid_mask_train_fit]
            y_train_used_in_fit = y_train[valid_mask_train_fit]
            attack_train_used_in_fit = attack_train[valid_mask_train_fit]

        except (ValueError, RuntimeError) as e:
            print(f"  Bootstrap fitting failed: {e}")
            self.model_results[model_key] = {"error": f"Fitting failed: {e}"}
            return None

        # --- Predict ---
        # Predict on grid and calculate metrics on training data only
        print("Generating bootstrap predictions and calculating metrics...")
        try:
            preds_bootstrap = predict_bootstrap(
                bootstrap_results,
                self.x_grid,
                x_data=x_train_used_in_fit,  # Use data aligned with fit for train metrics
                y_true=y_train_used_in_fit,
                y_true_logit=y_logit_train_used_in_fit,
                # Removed test data arguments
            )
        except Exception as e:
            print(f"  Bootstrap prediction/metric calculation failed: {e}")
            self.model_results[model_key] = {
                "error": f"Prediction failed: {e}",
                "fit_results": bootstrap_results,
            }
            return None

        # --- Calculate Calibration Metrics & Store ---
        # Only calculate train calibration now
        current_model_all_metrics = {
            "train": preds_bootstrap.get("train_metrics", {}),
            # "test": {}, # Removed test metrics
            "train_calibration": {},
            # "calibration": {} # Removed test calibration
        }

        # Train Calibration
        if "train_predictions" in preds_bootstrap:
            train_preds_info = preds_bootstrap["train_predictions"]
            train_samples_prob = train_preds_info.get("samples_prob")
            # Use y_train_used_in_fit for calibration, as it's aligned with samples
            if (
                train_samples_prob is not None
                and len(y_train_used_in_fit) > 0
                and len(y_train_used_in_fit) == train_samples_prob.shape[1]
            ):
                current_model_all_metrics["train_calibration"] = (
                    calculate_calibration_metrics(
                        y_train_used_in_fit,
                        train_samples_prob,
                        alpha=self.calibration_alpha,
                    )
                )

        # Store everything for this model
        self.model_results[model_key] = {
            "fit_results": bootstrap_results,
            "predictions": preds_bootstrap,
            "metrics": current_model_all_metrics,
            "train_data_info": {  # Renamed from "train_data_info" to just "data_info" implicitly
                "x": x_train_used_in_fit,
                "y": y_train_used_in_fit,
                "attack": attack_train_used_in_fit,
                "y_logit": y_logit_train_used_in_fit,
            },
            # Removed "test_data_info"
        }

        # --- Debug: Check stored value immediately ---
        stored_preds = self.model_results[model_key].get("predictions", {})
        stored_samples = stored_preds.get("samples_prob")
        print(
            f"  Debug fit_predict_model: Stored samples_prob type: {type(stored_samples)}, shape: {getattr(stored_samples, 'shape', 'N/A')}"
        )
        # --- End Debug ---

        # --- Optional Plot ---
        # Plot only uses training data now
        if make_plot:
            print("Generating dual plot...")
            if "samples_prob" in preds_bootstrap and "samples_logit" in preds_bootstrap:
                mean_w = np.mean(bootstrap_results.get("w_samples", [np.nan]))
                mean_b = np.mean(bootstrap_results.get("b_samples", [np.nan]))
                fig_dual, axes_dual = plot_linear_model_dual(
                    x_grid=self.x_grid,
                    samples_prob=preds_bootstrap["samples_prob"],
                    samples_logit=preds_bootstrap["samples_logit"],
                    x_data=x_train_used_in_fit,  # Plot data used for fit
                    y_data=y_train_used_in_fit,
                    attack_data=attack_train_used_in_fit,
                    y_logit=y_logit_train_used_in_fit,
                    # Removed test data arguments for plotting
                    title=f"Bootstrap Linear Model Fit for {model_key}",
                    metrics=current_model_all_metrics,  # Pass all metrics (train only now)
                    mean_w=mean_w,
                    mean_b=mean_b,
                    xlim=self.x_grid[[0, -1]],
                    ylim=(-0.05, 1.05),
                    # Remove test legend labels if they were present
                    train_legend_label="Data",
                )
                plt.show()
            else:
                # Check why plotting is skipped
                print(
                    f"Skipping dual plot: samples_prob found: {'samples_prob' in preds_bootstrap}, samples_logit found: {'samples_logit' in preds_bootstrap}"
                )

        print(f"--- Finished processing model: {model_key} ---")
        return self.model_results[model_key]  # Return results for this model

    def get_aggregated_metrics(
        self, model_keys_to_include: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculates and returns average train metrics across specified models."""
        if model_keys_to_include is None:
            # Use all models that completed successfully (don't have 'error' key)
            model_keys_to_include = [
                k for k, v in self.model_results.items() if "error" not in v
            ]

        if not model_keys_to_include:
            print("No models specified or no successful results found for aggregation.")
            return {}

        aggregated_metrics = defaultdict(list)
        first_valid_metrics = None
        metrics_to_average = []

        # Collect metrics from specified models
        for key in model_keys_to_include:
            result = self.model_results.get(key)
            if result and "metrics" in result:
                metrics_to_average.append(result["metrics"])
                if first_valid_metrics is None:
                    first_valid_metrics = result["metrics"]
            else:
                print(
                    f"Warning: No valid metrics found for model '{key}'. Skipping for aggregation."
                )

        if not metrics_to_average:
            print("No valid metrics collected for aggregation.")
            return {}

        # Dynamically get keys from the first valid metrics dict
        metric_keys_train = list(first_valid_metrics.get("train", {}).keys())
        # metric_keys_test = list(first_valid_metrics.get("test", {}).keys()) # Removed
        calib_keys_train = list(first_valid_metrics.get("train_calibration", {}).keys())
        # calib_keys_test = list(first_valid_metrics.get("calibration", {}).keys()) # Removed
        print(
            f"DEBUG get_aggregated_metrics: Found train keys: {metric_keys_train}"
        )  # DEBUG
        print(
            f"DEBUG get_aggregated_metrics: Found train calibration keys: {calib_keys_train}"
        )  # DEBUG

        # Aggregate
        for metrics_dict in metrics_to_average:
            train_data = metrics_dict.get("train", {})
            if isinstance(train_data, dict):
                for key in metric_keys_train:
                    if key in train_data and pd.notna(train_data[key]):
                        aggregated_metrics["avg_train_" + key].append(train_data[key])
            # Removed test metric aggregation
            train_calib_data = metrics_dict.get("train_calibration", {})
            if isinstance(train_calib_data, dict):
                for key in calib_keys_train:
                    if key in train_calib_data and pd.notna(train_calib_data[key]):
                        # Prefix train calibration keys appropriately
                        aggregated_metrics["avg_train_calib_" + key].append(
                            train_calib_data[key]
                        )
            # Removed test calibration aggregation

        # Calculate averages
        final_avg_metrics = {
            key: (np.mean(values), np.std(values))
            for key, values in aggregated_metrics.items()
            if values
        }
        return final_avg_metrics

    def get_logit_normal_params(
        self, x_values: np.ndarray, model_keys: Optional[List[str]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculates the mean and standard deviation of the predictive normal distribution
        in logit space for given x_values. This is done by first aggregating
        bootstrap samples of linear model parameters (w, b) from the specified models,
        then computing their summary statistics (mean, variance, covariance), and finally
        using these statistics to analytically derive the mean and standard deviation
        of the logit predictions at the given x_values.

        The formula for logit_ASR = w*x + b is used. Given:
        - E[w], Var[w]
        - E[b], Var[b]
        - Cov[w, b]
        The mean of logit_ASR(x) is E[w]*x + E[b].
        The variance of logit_ASR(x) is Var[w]*x^2 + Var[b] + 2*x*Cov[w, b].

        Args:
            x_values: A NumPy array of x-values for which to calculate the
                      logit normal parameters.
            model_keys: An optional list of model keys to include.
                        If None, all models with successful fit_results will be used.

        Returns:
            A dictionary with "mean" and "std" of the logit values based on
            the analytical calculation from aggregated parameter statistics.
            Returns {"error": "message"} if an issue occurs.
        """
        model_keys_to_process: List[str]
        if model_keys is None:
            model_keys_to_process = [
                k
                for k, v in self.model_results.items()
                if "error" not in v and v.get("fit_results")
            ]
        else:
            model_keys_to_process = model_keys

        if not model_keys_to_process:
            return {
                "error": "No model keys provided or no successful models available."
            }

        all_w_samples: List[np.ndarray] = []
        all_b_samples: List[np.ndarray] = []
        errors_encountered: List[str] = []

        for model_key in model_keys_to_process:
            if model_key not in self.model_results:
                msg = f"Model key '{model_key}' not found."
                errors_encountered.append(msg)
                continue

            model_res = self.model_results[model_key]
            if "error" in model_res:
                msg = f"Model '{model_key}' has fitting error: {model_res['error']}"
                errors_encountered.append(msg)
                continue

            fit_results = model_res.get("fit_results")
            if not fit_results:
                msg = f"No fit_results found for model '{model_key}'."
                errors_encountered.append(msg)
                continue

            w_samples = fit_results.get("w_samples")
            b_samples = fit_results.get("b_samples")

            if w_samples is None or b_samples is None:
                msg = f"Parameter samples (w_samples or b_samples) not found in fit_results for '{model_key}'."
                errors_encountered.append(msg)
                continue

            if not isinstance(w_samples, np.ndarray) or not isinstance(
                b_samples, np.ndarray
            ):
                msg = f"w_samples and b_samples must be NumPy arrays for model '{model_key}'."
                errors_encountered.append(msg)
                continue

            if w_samples.ndim != 1 or b_samples.ndim != 1:
                msg = f"w_samples and b_samples must be 1D arrays for model '{model_key}'."
                errors_encountered.append(msg)
                continue

            if w_samples.shape[0] != b_samples.shape[0]:
                msg = f"w_samples and b_samples must have the same number of samples for model '{model_key}'."
                errors_encountered.append(msg)
                continue

            if w_samples.shape[0] == 0:
                msg = f"No bootstrap samples found (w_samples is empty) for model '{model_key}'."
                errors_encountered.append(msg)
                continue

            all_w_samples.append(w_samples)
            all_b_samples.append(b_samples)

        if not all_w_samples:
            error_summary = (
                "No valid bootstrap samples found for any of the processed model keys."
            )
            if errors_encountered:
                error_summary += " Encountered issues: " + "; ".join(errors_encountered)
            return {"error": error_summary}

        w_samples_combined = np.concatenate(all_w_samples)
        b_samples_combined = np.concatenate(all_b_samples)

        N_samples = w_samples_combined.shape[0]

        mean_w = np.mean(w_samples_combined)
        mean_b = np.mean(b_samples_combined)

        if N_samples <= 1:
            var_w = 0.0
            var_b = 0.0
            cov_wb = 0.0
        else:
            var_w = np.var(w_samples_combined)  # ddof=0 by default (N divisor)
            var_b = np.var(b_samples_combined)  # ddof=0 by default (N divisor)
            # For np.cov, ddof=0 means bias=True (N divisor)
            cov_matrix = np.cov(w_samples_combined, b_samples_combined, ddof=0)
            cov_wb = cov_matrix[0, 1]

        # Ensure x_values is a numpy array
        if not isinstance(x_values, np.ndarray):
            try:
                x_values = np.array(x_values, dtype=float)
            except Exception as e:
                return {"error": f"Could not convert x_values to a NumPy array: {e}"}

        if x_values.ndim == 0:  # Handle scalar x_value
            x_values = x_values[np.newaxis]
        elif x_values.ndim > 1:
            return {"error": "x_values must be a scalar or a 1D NumPy array."}

        # Analytical calculation of mean and variance of logit_ASR(x)
        mean_logit = mean_w * x_values + mean_b
        var_logit = var_w * (x_values**2) + var_b + 2 * x_values * cov_wb

        # Ensure variance is not negative due to floating point issues before sqrt
        std_logit = np.sqrt(np.maximum(0, var_logit))

        # Optional: Log warnings if some models were skipped during sample aggregation
        # if errors_encountered:
        #     print(f"Warnings encountered while processing models for get_logit_normal_params: {'; '.join(errors_encountered)}")

        return {"mean": mean_logit, "std": std_logit}

    def plot_aggregated_uncertainty(
        self,
        model_keys_to_include: Optional[List[str]] = None,
        test_df: Optional[pd.DataFrame] = None,  # Accept DataFrame for test data
        xlim: Optional[tuple] = (-2.5, 2.5),  # Default xlim set to -2.5 to 2.5
        ylim: tuple = (-0.05, 1.05),
        title: Optional[str] = None,
        color: str = "darkorange",
        sigmas_to_plot: List[int] = [1, 2],
        plot_individual_means: bool = False,
        remove_grid: bool = False,
        ignore_train_data: bool = False,
        save_path: Optional[str] = None,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:  # Return Optional Tuple
        """
        Creates a BMA-like plot by aggregating bootstrap results.
        Optionally overlays provided test data points from a DataFrame.
        Calculates and prints metrics for the provided test data.

        Args:
            model_keys_to_include: List of model keys to include in plot.
            test_df: Optional DataFrame containing test data points.
                     Must contain columns specified by self.capability_diff_col,
                     self.asr_col, and self.attack_col.
            xlim: X-axis limits for the plot.
            ylim: Y-axis limits for the plot.
            title: Plot title.
            color: Base color for the plot.
            sigmas_to_plot: Which standard deviation intervals to plot.
            plot_individual_means: If True, plot individual model means as thin gray lines.

        Returns:
            (fig, ax) Matplotlib figure and axes objects, or (None, None) if plotting fails.
        """
        if model_keys_to_include is None:
            model_keys_to_include = [
                k
                for k, v in self.model_results.items()
                if "error" not in v and "predictions" in v
            ]

        if not model_keys_to_include:
            print(
                "No models specified or no successful results found for aggregation plot."
            )
            return None, None

        print(
            f"\nAggregating results from {len(model_keys_to_include)} models for plotting..."
        )

        # Collect samples, params, and data (only train/full data from results)
        x_train_list, y_train_list, attack_train_list = [], [], []
        w_samples_list, b_samples_list = [], []

        n_grid_points = len(self.x_grid)
        valid_samples_list = []  # Store only samples with correct shape
        expected_shape_cols = n_grid_points

        # Add list to collect individual model means if needed
        individual_means_list = [] if plot_individual_means else None

        for key in model_keys_to_include:
            result = self.model_results.get(key)
            # --- Debug: Check retrieved value before check ---
            # retrieved_preds = result.get("predictions", {}) if result else {}
            # retrieved_samples = retrieved_preds.get("samples_prob")
            # print(
            #     f"  Debug plot_agg: Retrieving for model '{key}'. Type: {type(retrieved_samples)}, Shape: {getattr(retrieved_samples, 'shape', 'N/A')}"
            # ) # REMOVED
            # --- End Debug ---

            if result and "predictions" in result and "fit_results" in result:
                preds = result["predictions"]
                fit_res = result["fit_results"]

                # Collect Grid Samples
                if "samples_prob" in preds:
                    current_samples = preds["samples_prob"]
                    # --- Shape Check ---
                    if (
                        isinstance(current_samples, np.ndarray)
                        and current_samples.ndim == 2
                        and current_samples.shape[1] == expected_shape_cols
                    ):
                        valid_samples_list.append(current_samples)

                        # Calculate and collect individual model mean if needed
                        if plot_individual_means:
                            individual_means_list.append(
                                np.mean(current_samples, axis=0)
                            )
                    else:
                        print(
                            f"Warning: Skipping samples from model '{key}' due to unexpected shape "
                            f"({getattr(current_samples, 'shape', 'N/A')}, expected N x {expected_shape_cols}). "
                            f"Prediction might have failed for this model."
                        )
                # Collect Fit Parameters
                if "w_samples" in fit_res and "b_samples" in fit_res:
                    w_samples_list.append(fit_res["w_samples"])
                    b_samples_list.append(fit_res["b_samples"])

                # Append TRAIN data used for this model's fit
                train_info = result.get("train_data_info", {})  # Changed key check
                if "x" in train_info:
                    x_train_list.append(train_info["x"])
                if "y" in train_info:
                    y_train_list.append(train_info["y"])
                if "attack" in train_info:
                    attack_train_list.append(train_info["attack"])

        if (
            not valid_samples_list or not w_samples_list or not b_samples_list
        ):  # Check all collected lists
            print(
                "No valid prediction samples or fit parameters found (after checks) to aggregate for plot."
            )
            return None, None

        # Concatenate train/full data, VALID samples and parameters
        try:
            combined_samples = np.concatenate(valid_samples_list, axis=0)
            combined_w_samples = np.concatenate(w_samples_list, axis=0)
            combined_b_samples = np.concatenate(b_samples_list, axis=0)
        except ValueError as e:
            print(f"Error during sample/parameter concatenation even after checks: {e}")
            return None, None

        x_train_agg = np.concatenate(x_train_list) if x_train_list else np.array([])
        y_train_agg = np.concatenate(y_train_list) if y_train_list else np.array([])
        attack_train_agg = (
            np.concatenate(attack_train_list) if attack_train_list else np.array([])
        )

        # Extract test data from DataFrame if provided
        x_test, y_test, attack_test = None, None, None
        test_metrics_results = {}  # Initialize dict for test metrics

        if test_df is not None and not test_df.empty:
            missing_cols = []
            req_cols = [self.capability_diff_col, self.asr_col]
            for col in req_cols:
                if col not in test_df.columns:
                    missing_cols.append(col)
            has_attack_col = self.attack_col in test_df.columns

            if missing_cols:
                print(
                    f"Warning: Test DataFrame provided but missing required columns: {missing_cols}. Cannot plot or calculate metrics for test points."
                )
            else:
                print("Extracting test data from DataFrame...")
                x_test = test_df[self.capability_diff_col].values
                y_test = test_df[self.asr_col].values
                if has_attack_col:
                    attack_test = test_df[self.attack_col].values
                else:
                    print(
                        f"  Note: Attack column '{self.attack_col}' not found in test_df. Plotting/metrics use default label."
                    )
                    attack_test = None  # Plot with default label

                # --- Calculate Metrics for Test Data ---
                print("Calculating metrics for provided test data...")
                # Filter test data for NaNs before prediction/metrics
                valid_test_mask = ~np.isnan(x_test) & ~np.isnan(y_test)
                if not np.all(valid_test_mask):
                    n_removed = np.sum(~valid_test_mask)
                    print(
                        f"  Warning: Removing {n_removed} test points with NaN in x or y for metric calculation."
                    )
                    if n_removed == len(x_test):
                        print(
                            "  Error: All provided test points have NaNs. Cannot calculate test metrics."
                        )
                        x_test_valid, y_test_valid = (
                            None,
                            None,
                        )  # Mark as invalid for metric steps
                    else:
                        x_test_valid = x_test[valid_test_mask]
                        y_test_valid = y_test[valid_test_mask]
                else:
                    x_test_valid = x_test
                    y_test_valid = y_test

                if x_test_valid is not None:
                    try:
                        x_test_pred_in = (
                            x_test_valid[:, np.newaxis]
                            if x_test_valid.ndim == 1
                            else x_test_valid
                        )
                        # Predict on valid test points using combined parameters
                        samples_logit_test = (
                            combined_w_samples[:, np.newaxis] * x_test_pred_in.T
                            + combined_b_samples[:, np.newaxis]
                        )
                        samples_prob_test = 1 / (1 + np.exp(-samples_logit_test))
                        mean_pred_prob_test = np.mean(samples_prob_test, axis=0)
                        mean_pred_logit_test = np.mean(samples_logit_test, axis=0)

                        # Calculate y_test_logit_valid for metrics calculation
                        y_test_logit_valid = None
                        try:
                            y_test_logit_valid_temp = logit_transform(y_test_valid)
                            if not np.any(np.isnan(y_test_logit_valid_temp)):
                                y_test_logit_valid = y_test_logit_valid_temp
                            else:
                                print(
                                    "  Skipping logit test metrics due to NaNs in logit(y_test_valid)."
                                )
                        except Exception as logit_e:
                            print(
                                f"  Warning: Could not transform y_test to logit for metrics: {logit_e}"
                            )

                        # Calculate main metrics using helper
                        test_metrics_results = _calculate_metrics(
                            y_true_prob=y_test_valid,
                            y_pred_prob=mean_pred_prob_test,
                            y_true_logit=y_test_logit_valid,  # Will be None if failed
                            y_pred_logit=mean_pred_logit_test,
                        )

                        # Calculate Calibration Metrics separately
                        test_calib_metrics = calculate_calibration_metrics(
                            y_test_valid,
                            samples_prob_test,
                            alpha=self.calibration_alpha,
                        )
                        test_metrics_results.update(
                            test_calib_metrics
                        )  # Add calibration results

                        print("  Test metrics calculation complete.")

                    except Exception as metric_e:
                        print(f"Error during test metric calculation: {metric_e}")
                        test_metrics_results = {
                            "error": f"Metric calculation failed: {metric_e}"
                        }

        # Calculate aggregated TRAIN metrics (as before)
        aggregated_train_metrics = self.get_aggregated_metrics(model_keys_to_include)
        print("\nAverage Metrics (Train/Full Data) for Aggregated Plot:")
        if aggregated_train_metrics:
            for k, v in sorted(aggregated_train_metrics.items()):
                if isinstance(v, tuple) and len(v) == 2:
                    mean_val, std_val = v
                    print(f"  {k}: {mean_val:.4f} ± {std_val:.4f}")
                else:
                    print(f"  {k}: {v:.4f}")
        else:
            print("  No train metrics to display.")

        # Print calculated TEST metrics
        print("\nMetrics for Provided Test Data:")
        if test_metrics_results:
            if "error" in test_metrics_results:
                print(f"  Error: {test_metrics_results['error']}")
            else:
                for k, v in sorted(test_metrics_results.items()):
                    print(f"  test_{k}: {v:.4f}")
        elif test_df is not None:
            print("  Could not calculate test metrics (check warnings above).")
        else:
            print("  No test data provided.")

        # Determine plot limits based on train data and optional test data
        if xlim is None:
            all_x_for_lims_list = [x_train_agg]
            if x_test is not None and len(x_test) > 0:
                all_x_for_lims_list.append(x_test)
            # Filter out potential NaNs before concatenation for limits
            all_x_for_lims_list = [
                x[np.isfinite(x)]
                for x in all_x_for_lims_list
                if x is not None and len(x) > 0
            ]
            if not all_x_for_lims_list:
                final_xlim = self.x_grid[[0, -1]]  # Fallback if no finite data
            else:
                all_x_for_lims = np.concatenate(all_x_for_lims_list)
                if len(all_x_for_lims) > 0:
                    min_x, max_x = np.min(all_x_for_lims), np.max(all_x_for_lims)
                    x_buffer = (max_x - min_x) * 0.05 if (max_x - min_x) > 0 else 0.1
                    final_xlim = (min_x - x_buffer, max_x + x_buffer)
                else:
                    final_xlim = self.x_grid[[0, -1]]  # Fallback if only NaNs
        else:
            final_xlim = xlim

        print("\nPlotting aggregated bootstrap uncertainty...")

        # Use plot_uncertainty_custom instead of plot_uncertainty
        fig, ax = self._plot_aggregated_uncertainty_custom(
            x_grid=self.x_grid,
            samples_prob=combined_samples,
            x_data=x_train_agg,
            y_data=y_train_agg,
            test_data=(x_test, y_test)
            if x_test is not None and y_test is not None
            else None,
            title=title,
            xlim=final_xlim,
            ylim=ylim,
            color=color,
            sigmas_to_plot=sigmas_to_plot,
            individual_means=individual_means_list,  # Pass individual means if collected
            remove_grid=remove_grid,
            ignore_train_data=ignore_train_data,
        )

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        return fig, ax

    def _plot_aggregated_uncertainty_custom(
        self,
        x_grid,
        samples_prob,
        x_data=None,
        y_data=None,
        test_data=None,
        xlim=(-3, 3),
        ylim=(0, 1),
        title=None,
        color="maroon",
        individual_means=None,
        metrics=None,
        sigmas_to_plot=[1, 2],
        remove_grid=False,
        ignore_train_data=False,
    ):
        """
        Customized version of plot_uncertainty_custom for aggregated bootstrap data.

        Args:
            x_grid: Grid of x values for plotting.
            samples_prob: Array of probability samples from bootstrap.
            x_data: Training data x values.
            y_data: Training data y values.
            test_data: Tuple of (x_test, y_test) for plotting test data.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            title: Plot title.
            color: Base color for the plot.
            individual_means: List of individual model mean predictions to plot as thin gray lines.
            metrics: Metrics to display.
            sigmas_to_plot: Which standard deviation intervals to plot.

        Returns:
            (fig, ax) Matplotlib figure and axes objects.
        """
        if ignore_train_data:
            x_data = None
            y_data = None
        fig, ax = plt.subplots(figsize=(5, 5))

        # Increase font sizes for all elements
        plt.rcParams.update({"font.size": 14})

        # Set title
        ax.set_title(title, fontsize=16)

        # Plot original data if provided
        if x_data is not None and y_data is not None:
            ax.scatter(
                x_data,
                y_data,
                color=color,
                alpha=0.9,
                s=30,
                # linewidths=0.5,
                # edgecolors="black",
                label="Training Data",
            )

        # Plot individual means if provided
        if individual_means is not None and len(individual_means) > 0:
            for i, ind_mean in enumerate(individual_means):
                label = "Individual Means" if i == 0 else None
                ax.plot(
                    x_grid,
                    ind_mean,
                    color=color,
                    ls="--",
                    lw=1.5,
                    alpha=0.5,
                    label=label,
                )

        # Compute and plot mean/median prediction for the main samples_prob
        median_prob = np.median(samples_prob, axis=0)
        ax.plot(x_grid, median_prob, color=color, lw=4, label="Median")

        # Compute and plot uncertainty bands for the main samples_prob
        sigma_levels = sorted(sigmas_to_plot)  # Ensure ordered plotting
        sigma_percentiles = {
            1: (15.87, 84.13),  # ±1σ (68%)
            2: (2.28, 97.72),  # ±2σ (95%)
            3: (0.13, 99.87),  # ±3σ (99.7%)
        }
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

        # Plot test data if provided
        if test_data is not None:
            x_test, y_test = test_data  # Unpack test data
            # Use a '+' marker with no border for test data
            cmap = plt.cm.get_cmap("Set2", 8)
            ax.scatter(
                x_test,
                y_test,
                color=cmap(5),
                marker="+",  # Use '+' marker
                s=50,  # Adjust size for better visibility
                label="Test Data",
                linewidths=2,
                edgecolors="grey",
                alpha=0.7,  # Full opacity
                zorder=10,  # Ensure it's plotted on top
            )

        # Format plot
        ax.set_xlabel("Capability Difference", fontsize=18)
        ax.set_ylabel("Attacks Success Rate", fontsize=18)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        # Increase tick label font sizes
        ax.tick_params(axis="both", which="major", labelsize=16)

        handles, labels = ax.get_legend_handles_labels()
        # Define the desired order of legend items
        desired_order = [
            "Training Data",
            "Test Data",
            "Median",
            "Individual Means",
        ]
        # Add sigma levels in ascending order
        for sigma in sorted(sigma_levels):
            desired_order.append(f"±{confidence_levels[sigma]}")

        # Build the final order, placing known items first and unknowns last
        order = []
        handled_indices = set()
        for item in desired_order:
            if item in labels:
                idx = labels.index(item)
                order.append(idx)
                handled_indices.add(idx)
        # Add any remaining labels not in the desired order
        order.extend([i for i, lbl in enumerate(labels) if i not in handled_indices])

        # Apply the ordered legend
        ax.legend(
            [handles[idx] for idx in order if idx < len(handles)],
            [labels[idx] for idx in order if idx < len(labels)],
            frameon=False,
            fontsize=14,  # Increased legend font size
            loc="best",
        )

        if remove_grid:
            ax.grid(False)

        plt.tight_layout()
        return fig, ax

    def _process_results_for_saving(self) -> Dict[str, Any]:
        """Process bootstrap model results to prepare them for saving to JSON."""
        # Use deepcopy to avoid modifying the original results in memory
        results_copy = copy.deepcopy(self.model_results)
        results_to_save = {}

        for key, value in results_copy.items():  # Iterate over the copy
            if isinstance(value, dict) and "fit_results" in value:
                # Modify the copied dictionary
                value_copy = (
                    value  # No need for another copy here, value is from results_copy
                )
                fit_res = value_copy.get("fit_results", {})  # Use .get for safety

                # Prepare fit_results for saving (replace arrays with placeholders)
                fit_results_save = {}
                fit_results_save["w_samples"] = (
                    f"<w_samples array, shape={fit_res.get('w_samples', np.array([])).shape}>"
                )
                fit_results_save["b_samples"] = (
                    f"<b_samples array, shape={fit_res.get('b_samples', np.array([])).shape}>"
                )
                fit_results_save["y_logit"] = (
                    f"<y_logit array, shape={fit_res.get('y_logit', np.array([])).shape}>"
                )
                fit_results_save["n_valid_bootstrap"] = fit_res.get("n_valid_bootstrap")
                value_copy["fit_results"] = fit_results_save  # Update the copied value

                # Prepare predictions for saving
                preds = value_copy.get("predictions", {})  # Use .get for safety
                preds_save = {}

                # Handle grid predictions
                for sample_type in ["samples_prob", "samples_logit"]:
                    if sample_type in preds:
                        shape = getattr(preds[sample_type], "shape", "N/A")
                        preds_save[sample_type] = (
                            f"<{sample_type} grid array, shape={shape}>"
                        )
                # Copy other non-array grid prediction keys
                for k, v in preds.items():
                    if k not in ["samples_prob", "samples_logit", "train_predictions"]:
                        preds_save[k] = v

                # Handle train predictions
                if "train_predictions" in preds:
                    train_preds = preds["train_predictions"]
                    train_preds_save = {}
                    for sample_type in ["samples_prob", "samples_logit"]:
                        if sample_type in train_preds:
                            shape = getattr(train_preds[sample_type], "shape", "N/A")
                            train_preds_save[sample_type] = (
                                f"<{sample_type} train array, shape={shape}>"
                            )
                    # Copy other non-array train prediction keys
                    for k, v in train_preds.items():
                        if k not in ["samples_prob", "samples_logit"]:
                            train_preds_save[k] = v
                    preds_save["train_predictions"] = train_preds_save

                value_copy["predictions"] = preds_save  # Update the copied value

                results_to_save[key] = value_copy
            else:
                # Handle cases like error messages (copy directly)
                results_to_save[key] = value

        return results_to_save

    def plot_aggregated_uncertainty_medians(
        self,
        family_dict: Dict[str, Tuple[List[str], str]],
        xlim: Optional[tuple] = (-2.5, 2.5),
        ylim: tuple = (-0.05, 1.05),
        title: Optional[str] = "Aggregated Median Predictions by Model Family",
        lw: float = 2,
        label_size: int = 12,
        tick_size: int = 12,
        legend_size: int = 12,
        save_path: Optional[str] = None,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the median prediction line for specified groups (families) of models.

        Args:
            family_dict: Dictionary where keys are family names and values are tuples
                         containing (list_of_model_keys, color_string).
                         Example: {"Family A": (["model1", "model2"], "blue"),
                                  "Family B": (["model3"], "red")}
            xlim: X-axis limits for the plot.
            ylim: Y-axis limits for the plot.
            title: Plot title.
            lw: Line width for the median plots.

        Returns:
            (fig, ax) Matplotlib figure and axes objects, or (None, None) if plotting fails.
        """

        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["text.usetex"] = False
        if not family_dict:
            print("Error: family_dict cannot be empty.")
            return None, None

        fig, ax = plt.subplots(figsize=(5, 6))
        ax.set_title(title, fontsize=16)

        n_grid_points = len(self.x_grid)
        expected_shape_cols = n_grid_points
        found_valid_family = False

        for family_name, (model_keys, color) in family_dict.items():
            print(f"\n--- Processing Family: {family_name} ---")
            if not model_keys:
                print(
                    f"Warning: No model keys specified for family '{family_name}'. Skipping."
                )
                continue

            family_samples_list = []
            for key in model_keys:
                result = self.model_results.get(key)
                if result and "predictions" in result:
                    preds = result["predictions"]
                    if "samples_prob" in preds:
                        current_samples = preds["samples_prob"]
                        # Shape Check
                        if (
                            isinstance(current_samples, np.ndarray)
                            and current_samples.ndim == 2
                            and current_samples.shape[1] == expected_shape_cols
                        ):
                            family_samples_list.append(current_samples)
                        else:
                            print(
                                f"Warning: Skipping samples from model '{key}' in family '{family_name}' due to unexpected shape "
                                f"({getattr(current_samples, 'shape', 'N/A')}, expected N x {expected_shape_cols})."
                            )
                    else:
                        print(
                            f"Warning: No 'samples_prob' found for model '{key}' in family '{family_name}'."
                        )
                else:
                    print(
                        f"Warning: No valid results or predictions found for model '{key}' in family '{family_name}'. Skipping."
                    )

            if not family_samples_list:
                print(
                    f"Warning: No valid samples collected for family '{family_name}'. Cannot plot median."
                )
                continue

            # Concatenate and calculate median for this family
            try:
                combined_family_samples = np.concatenate(family_samples_list, axis=0)
                median_prob = np.median(combined_family_samples, axis=0)
                print(
                    f"  Plotting median for {family_name} with {len(combined_family_samples)} total samples."
                )
                ax.plot(self.x_grid, median_prob, color=color, lw=lw, label=family_name)
                found_valid_family = True
            except ValueError as e:
                print(f"Error processing family '{family_name}': {e}")
                continue

        if not found_valid_family:
            print("Error: No valid families found to plot.")
            plt.close(fig)  # Close the empty figure
            return None, None

        # Format plot
        ax.set_xlabel(
            "General Capability Difference \nbetween Attacker and Target",
            fontsize=label_size,
        )
        ax.set_ylabel("Attack Success Rate", fontsize=label_size)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        ax.grid(False)

        # Add legend with title
        ax.legend(title="Model Family", frameon=False, loc="best", fontsize=legend_size)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600)
        plt.show()
        return fig, ax

    def plot_external_model_medians(
        self,
        x_grid: np.ndarray,
        external_results_map: Dict[str, Tuple[Dict[str, Any], str]],
        xlim: Optional[tuple] = (-2.5, 2.5),
        ylim: tuple = (-0.05, 1.05),
        title: Optional[str] = "Median Predictions from External Model Results",
        lw: float = 2,
        label_size: int = 12,
        tick_size: int = 12,
        legend_size: int = 12,
        legend_title: str = "Method",
        save_path: Optional[str] = None,
    ) -> Optional[Tuple[plt.Figure, plt.Axes]]:
        """
        Plots the median prediction line for multiple, externally provided model results.
        Each entry in external_results_map will result in one median line.

        Args:
            x_grid: Numpy array of x-values for the grid.
            external_results_map: Dictionary where keys are labels (for legend) and
                                  values are tuples (model_result_data, color_string).
                                  model_result_data should be a dict containing
                                  at least {"predictions": {"samples_prob": np.array}}.
            xlim: X-axis limits.
            ylim: Y-axis limits.
            title: Plot title.
            lw: Line width for median plots.
            label_size: Font size for axis labels.
            tick_size: Font size for tick labels.
            legend_size: Font size for legend.
            legend_title: Title for the legend.
            save_path: Optional path to save the figure.

        Returns:
            (fig, ax) Matplotlib figure and axes objects, or (None, None) if plotting fails.
        """
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams["text.usetex"] = False

        if not external_results_map:
            print("Error: external_results_map cannot be empty.")
            return None, None

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_title(title, fontsize=16)

        n_grid_points = len(x_grid)
        expected_shape_cols = n_grid_points  # For samples_prob on the grid
        found_valid_entry_to_plot = False

        for label, (model_data_for_label, color) in external_results_map.items():
            print(f"\n--- Processing Entry: {label} ---")

            current_samples_to_process = None
            w_samples_to_process = None
            b_samples_to_process = None

            # Try to interpret model_data_for_label directly
            if isinstance(model_data_for_label, dict):
                direct_predictions = model_data_for_label.get("predictions")
                direct_fit_results = model_data_for_label.get("fit_results")

                if isinstance(direct_predictions, dict):
                    direct_samples_prob = direct_predictions.get("samples_prob")
                    if isinstance(direct_samples_prob, np.ndarray):
                        print(f"  Interpreting '{label}' as direct samples_prob data.")
                        current_samples_to_process = direct_samples_prob

                        # Attempt to get w and b samples directly
                        if isinstance(direct_fit_results, dict):
                            w_s = direct_fit_results.get("w_samples")
                            b_s = direct_fit_results.get("b_samples")
                            if isinstance(w_s, np.ndarray) and isinstance(
                                b_s, np.ndarray
                            ):
                                print(
                                    f"  Found direct w_samples and b_samples for '{label}'."
                                )
                                w_samples_to_process = w_s
                                b_samples_to_process = b_s

            # If not direct, try to interpret as a collection of model results to aggregate
            if current_samples_to_process is None and isinstance(
                model_data_for_label, dict
            ):
                print(
                    f"  Interpreting '{label}' as a collection of per-model results to aggregate."
                )
                aggregated_samples_list = []
                aggregated_w_samples_list = []
                aggregated_b_samples_list = []

                for sub_model_key, sub_model_result in model_data_for_label.items():
                    if (
                        isinstance(sub_model_result, dict)
                        and "error" not in sub_model_result
                    ):
                        # Aggregate samples_prob
                        predictions = sub_model_result.get("predictions")
                        if isinstance(predictions, dict):
                            samples_p = predictions.get("samples_prob")
                            if (
                                isinstance(samples_p, np.ndarray)
                                and samples_p.ndim == 2
                            ):
                                if samples_p.shape[1] == expected_shape_cols:
                                    aggregated_samples_list.append(samples_p)
                                else:
                                    print(
                                        f"    Warning: Skipping samples_prob from sub-model '{sub_model_key}' in '{label}' due to mismatched columns: expected {expected_shape_cols}, got {samples_p.shape[1]}."
                                    )

                        # Aggregate w_samples and b_samples
                        fit_results = sub_model_result.get("fit_results")
                        if isinstance(fit_results, dict):
                            w_s = fit_results.get("w_samples")
                            b_s = fit_results.get("b_samples")
                            if isinstance(w_s, np.ndarray) and isinstance(
                                b_s, np.ndarray
                            ):
                                aggregated_w_samples_list.append(w_s)
                                aggregated_b_samples_list.append(b_s)
                            # else: print(f"    Debug: No w/b samples in sub-model '{sub_model_key}' for '{label}'.")

                if not aggregated_samples_list:
                    print(
                        f"  Warning: No valid samples_prob found to aggregate for '{label}'. Skipping this entry."
                    )
                    continue
                try:
                    current_samples_to_process = np.concatenate(
                        aggregated_samples_list, axis=0
                    )
                    print(
                        f"    Successfully combined {len(aggregated_samples_list)} samples_prob arrays for '{label}', total samples: {current_samples_to_process.shape[0]}."
                    )
                    if aggregated_w_samples_list and aggregated_b_samples_list:
                        w_samples_to_process = np.concatenate(
                            aggregated_w_samples_list, axis=0
                        )
                        b_samples_to_process = np.concatenate(
                            aggregated_b_samples_list, axis=0
                        )
                        print(
                            f"    Successfully combined {len(aggregated_w_samples_list)} w_samples and b_samples arrays for '{label}'."
                        )
                    else:
                        print(
                            f"    Warning: Could not aggregate w_samples/b_samples for '{label}'. Median k and x0 will not be calculated."
                        )
                except ValueError as e:
                    print(
                        f"    Error concatenating samples for '{label}': {e}. Skipping this entry."
                    )
                    continue

            if current_samples_to_process is None:
                print(
                    f"Warning: Data for '{label}' is not in a recognized format or no valid samples_prob found. Skipping this entry."
                )
                continue

            # Shape Check for the final samples_to_process (samples_prob)
            if not (
                current_samples_to_process.ndim == 2
                and current_samples_to_process.shape[1] == expected_shape_cols
            ):
                print(
                    f"Warning: Skipping entry '{label}' due to unexpected final shape for samples_prob "
                    f"({getattr(current_samples_to_process, 'shape', 'N/A')}, expected N x {expected_shape_cols})."
                )
                continue

            # Calculate and print median k and x0 if w and b samples are available
            if (
                w_samples_to_process is not None
                and b_samples_to_process is not None
                and len(w_samples_to_process) > 0
                and len(w_samples_to_process) == len(b_samples_to_process)
            ):
                median_k = np.median(w_samples_to_process)
                print(f"  Median k (slope) for '{label}': {median_k:.4f}")

                # Calculate x0, handling potential division by zero
                valid_w_mask = w_samples_to_process != 0
                if np.any(valid_w_mask):
                    x0_samples = (
                        -b_samples_to_process[valid_w_mask]
                        / w_samples_to_process[valid_w_mask]
                    )
                    if len(x0_samples) > 0:
                        median_x0 = np.median(x0_samples)
                        print(
                            f"  Median x0 (x-intercept) for '{label}': {median_x0:.4f}"
                        )
                    else:
                        print(
                            f"  Could not calculate median x0 for '{label}' (all w_samples were zero after filtering)."
                        )
                else:
                    print(
                        f"  Could not calculate median x0 for '{label}' (all w_samples were zero)."
                    )
            else:
                print(
                    f"  Median k and x0 not calculated for '{label}' due to missing or mismatched w/b samples."
                )

            # Calculate and plot median ASR
            try:
                median_prob = np.median(current_samples_to_process, axis=0)
                ax.plot(x_grid, median_prob, color=color, lw=lw, label=label)
                found_valid_entry_to_plot = True
            except Exception as e:
                print(
                    f"Error plotting median ASR for '{label}': {e}. Skipping this entry."
                )
                continue

        if not found_valid_entry_to_plot:
            print("Error: No valid entries found to plot any medians.")
            plt.close(fig)
            return None, None

        # Format plot
        ax.set_xlabel(
            "General Capability Difference \nbetween Attacker and Target",
            fontsize=label_size,
        )
        ax.set_ylabel("Attack Success Rate", fontsize=label_size)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.tick_params(axis="both", which="major", labelsize=tick_size)
        ax.grid(True)

        ax.legend(
            title=legend_title,
            frameon=False,
            loc="best",
            fontsize=legend_size,
            title_fontsize=legend_size,
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=600)
        plt.show()
        return fig, ax
