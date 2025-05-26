# uncertainty/runner.py
import abc
import json
import os
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# Base runner doesn't need these specific imports
# from .metrics import calculate_calibration_metrics
# from .models.linear_model_bootstrapped import (
#     fit_bootstrap_model,
#     predict_bootstrap,
# )
# from .models.linear_model_bootstrapped import (
#     logit_transform as logit_transform_boot,
# )
# from .plotting.plotting import (
#     _plot_scatter_by_attack,
#     plot_linear_model_dual,
#     plot_uncertainty,
# )


class UncertaintyModelRunner(abc.ABC):
    """
    Abstract base class for uncertainty analysis runners.

    Defines the common interface for all uncertainty model runners.
    Concrete implementations must override the abstract methods.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        capability_calculator: Callable[[pd.DataFrame], pd.DataFrame],
        target_model_key_col: str = "target_model_key",
        capability_diff_col: str = "capability_diff",
        asr_col: str = "ASR",
        attack_col: str = "attack",
        x_grid_points: int = 500,
        x_grid_range: tuple = (-2.5, 2.5),
        random_state: int = 42,
        metrics_output_dir: str = "model_metrics",
        processed_df_col_prefix: str = "proc_",
    ):
        """
        Initializes the base runner.

        Args:
            df: The input DataFrame containing raw data.
            capability_calculator: Function that calculates capability difference.
            target_model_key_col: Name of the column identifying target models.
            capability_diff_col: Name of the column for calculated capability difference.
            asr_col: Name of the column containing the Attack Success Rate (ASR).
            attack_col: Name of the column containing the attack type.
            x_grid_points: Number of points for the prediction grid.
            x_grid_range: Min and max values for the prediction grid.
            random_state: Random seed for reproducibility.
            metrics_output_dir: Directory to save metrics JSON files.
            processed_df_col_prefix: Prefix for internal processed columns.
        """
        if not callable(capability_calculator):
            raise TypeError("capability_calculator must be a callable function.")

        self.raw_df = df.copy()
        self.capability_calculator = capability_calculator
        self.target_model_key_col = target_model_key_col
        self.capability_diff_col = capability_diff_col
        self.asr_col = asr_col
        self.attack_col = attack_col
        self.x_grid = np.linspace(x_grid_range[0], x_grid_range[1], x_grid_points)
        self.random_state = random_state
        self.metrics_output_dir = metrics_output_dir
        self.processed_df_col_prefix = processed_df_col_prefix

        # Internal storage
        self.processed_df = None
        self.model_results = {}  # Stores results per model_key

        print("Initializing base runner...")
        self._prepare_data()
        print("Data preparation complete.")

    def _prepare_data(self) -> None:
        """Applies capability calculation and creates internal columns needed."""
        print("Calculating capability difference...")
        # Apply the provided calculator function
        df_with_cap = self.capability_calculator(self.raw_df)

        # Check if the specified capability column exists after calculation
        if self.capability_diff_col not in df_with_cap.columns:
            raise ValueError(
                f"Column '{self.capability_diff_col}' not found in DataFrame "
                f"after applying capability_calculator."
            )

        # Additional data preparation is model-specific
        self.processed_df = self._prepare_model_specific_data(df_with_cap)

        # Remove rows where capability diff is NaN
        rows_before = len(self.processed_df)
        self.processed_df.dropna(subset=[self.capability_diff_col], inplace=True)
        rows_after = len(self.processed_df)
        if rows_after < rows_before:
            print(
                f"Removed {rows_before - rows_after} rows with NaN capability difference."
            )

    @abc.abstractmethod
    def _prepare_model_specific_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data specifically for this model type.

        Args:
            df: DataFrame with capability difference calculated.

        Returns:
            DataFrame with model-specific preparations applied.
        """
        pass

    @abc.abstractmethod
    def fit_predict_model(
        self, model_key: str, make_plot: bool = False
    ) -> Optional[Dict[str, Any]]:
        """
        Fits model, predicts, calculates metrics for a single model_key using its full data.

        Args:
            model_key: Identifier for the specific model to process.
            make_plot: Whether to generate plots during processing.

        Returns:
            Results for this model or None if processing failed.
        """
        pass

    def run_analysis(
        self,
        model_keys_to_process: Optional[List[str]] = None,
        make_plots: bool = False,
    ) -> None:
        """
        Runs the full analysis workflow for specified models.

        Args:
            model_keys_to_process: List of model keys to process. If None,
                                  process all unique values in target_model_key_col.
            make_plots: Whether to generate plots during processing.
        """
        if self.processed_df is None:
            raise RuntimeError("Data has not been prepared. Call _prepare_data first.")

        if model_keys_to_process is None:
            model_keys_to_process = sorted(
                self.processed_df[self.target_model_key_col].unique()
            )

        print(f"\nStarting analysis for {len(model_keys_to_process)} models...")
        for key in model_keys_to_process:
            self.fit_predict_model(key, make_plot=make_plots)

        print("\n--- Analysis Complete ---")
        self.save_results()  # Optionally save results automatically

    def get_model_results(self, model_key: str) -> Optional[Dict[str, Any]]:
        """Returns the stored results dictionary for a specific model key."""
        return self.model_results.get(model_key, None)

    @abc.abstractmethod
    def get_aggregated_metrics(
        self, model_keys_to_include: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Calculates and returns average metrics across specified models.

        Args:
            model_keys_to_include: List of model keys to include in aggregation.
                                  If None, include all successful models.

        Returns:
            Aggregated metrics (typically only based on training/full data now).
        """
        pass

    @abc.abstractmethod
    def plot_aggregated_uncertainty(
        self,
        model_keys_to_include: Optional[List[str]] = None,
        x_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        attack_test: Optional[np.ndarray] = None,
        xlim: Optional[tuple] = None,
        ylim: tuple = (-0.05, 1.05),
        title: Optional[str] = None,
        color: str = "darkorange",
        **kwargs,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """
        Creates an aggregated plot combining results from multiple models.
        Optionally overlays provided test data points.

        Args:
            model_keys_to_include: List of model keys to include in plot.
            x_test: Optional array of test data features (e.g., capability diff).
            y_test: Optional array of test data target values (e.g., ASR).
            attack_test: Optional array of test data attack labels.
            xlim: X-axis limits for the plot.
            ylim: Y-axis limits for the plot.
            title: Plot title.
            color: Base color for the plot.
            **kwargs: Additional model-specific plotting arguments.

        Returns:
            (fig, ax) Matplotlib figure and axes objects, or (None, None) if plotting failed.
        """
        pass

    def save_results(self, filename: Optional[str] = None) -> None:
        """
        Saves the model_results dictionary to a JSON file.

        Args:
            filename: Name of the output file. If None, use model-specific default.
        """
        if filename is None:
            filename = f"{self.__class__.__name__}_results.json"

        filepath = os.path.join(self.metrics_output_dir, filename)
        print(f"\nSaving model results to {filepath}...")
        os.makedirs(self.metrics_output_dir, exist_ok=True)

        # Custom JSON encoder for numpy types
        def default_converter(o):
            if isinstance(
                o,
                (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
                ),
            ):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, np.float64)):
                if np.isnan(o):
                    return "NaN"
                if np.isinf(o):
                    return "Infinity" if o > 0 else "-Infinity"
                return float(o)
            elif isinstance(o, (np.ndarray,)):
                # For large arrays, just store shape info
                if o.size > 1000:  # Heuristic threshold
                    return f"<np.ndarray, shape={o.shape}, dtype={o.dtype}>"
                return o.tolist()
            elif isinstance(o, (np.bool_)):
                return bool(o)
            elif isinstance(o, (np.void)):
                return None
            try:
                # Let the base class default method raise the TypeError
                return json.JSONEncoder.default(self, o)
            except TypeError:
                # Fallback: return string representation
                return str(o)

        try:
            # Model-specific result processing for saving
            results_to_save = self._process_results_for_saving()

            with open(filepath, "w") as f:
                json.dump(results_to_save, f, indent=4, default=default_converter)
            print("Results saved successfully.")
        except Exception as e:
            print(f"Error saving results to JSON: {e}")

    @abc.abstractmethod
    def _process_results_for_saving(self) -> Dict[str, Any]:
        """
        Process model results to prepare them for saving to JSON.

        Returns:
            Processed results ready for JSON serialization.
        """
        pass
