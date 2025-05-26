import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.special import expit, logit
from sklearn.linear_model import LinearRegression
from sklearn.utils import resample

from ..metrics import calculate_calibration_metrics
from ..plotting.plotting import (
    _plot_scatter_by_attack,
    plot_linear_model_dual,
    plot_uncertainty,
)


# --- Reusing the logit transform from the original model ---
def logit_transform(p, eps=1e-2):
    """
    Transform probabilities to logit space with handling of edge cases.

    Args:
        p: Input probabilities
        eps: Small value for numerical stability

    Returns:
        Logit transformed values
    """
    p = np.array(p).astype(float)
    p = np.clip(p, eps, 1 - eps)
    # Check for values exactly at the boundaries after clipping, can cause inf in logit
    if np.any(p == 0) or np.any(p == 1):
        warnings.warn(
            "Probabilities exactly 0 or 1 after clipping, may cause issues in logit."
        )
    with np.errstate(divide="ignore"):  # Ignore division by zero in log for edge cases
        logit_p = np.log(p / (1 - p))
    if np.any(np.isinf(logit_p)):
        warnings.warn(
            "Infinite values encountered in logit transform, potentially due to boundary values."
        )
    return logit_p


# --- Reusing the metrics calculation logic ---
def _calculate_metrics(y_true_prob, y_pred_prob, y_true_logit, y_pred_logit):
    """Helper to calculate RMSE and MAE for probability and logit spaces."""
    metrics = {}
    # Check if inputs are valid numpy arrays and have data
    valid_prob = (
        isinstance(y_true_prob, np.ndarray)
        and isinstance(y_pred_prob, np.ndarray)
        and y_true_prob.size > 0
        and y_pred_prob.size > 0
    )
    valid_logit = (
        isinstance(y_true_logit, np.ndarray)
        and isinstance(y_pred_logit, np.ndarray)
        and y_true_logit.size > 0
        and y_pred_logit.size > 0
    )

    # Metrics in probability space
    if valid_prob:
        metrics["rmse_prob"] = np.sqrt(np.mean((y_true_prob - y_pred_prob) ** 2))
        metrics["mae_prob"] = np.mean(np.abs(y_true_prob - y_pred_prob))
        # R2 Calculation (Prob)
        ss_total_prob = np.sum((y_true_prob - np.mean(y_true_prob)) ** 2)
        ss_residual_prob = np.sum((y_true_prob - y_pred_prob) ** 2)
        metrics["r2_prob"] = (
            1 - (ss_residual_prob / ss_total_prob) if ss_total_prob > 1e-9 else 0
        )
    else:
        metrics["rmse_prob"] = np.nan
        metrics["mae_prob"] = np.nan
        metrics["r2_prob"] = np.nan

    # Metrics in logit space
    if valid_logit:
        metrics["rmse_logit"] = np.sqrt(np.mean((y_true_logit - y_pred_logit) ** 2))
        metrics["mae_logit"] = np.mean(np.abs(y_true_logit - y_pred_logit))
        # R2 Calculation (Logit)
        ss_total_logit = np.sum((y_true_logit - np.mean(y_true_logit)) ** 2)
        ss_residual_logit = np.sum((y_true_logit - y_pred_logit) ** 2)
        metrics["r2_logit"] = (
            1 - (ss_residual_logit / ss_total_logit) if ss_total_logit > 1e-9 else 0
        )
    else:
        metrics["rmse_logit"] = np.nan
        metrics["mae_logit"] = np.nan
        metrics["r2_logit"] = np.nan

    return metrics


def fit_bootstrap_model(
    x_data,
    y_data,
    n_bootstrap=1000,
    random_seed=42,
):
    """
    Fit bootstrap linear model.

    Args:
        x_data: Input features
        y_data: Target values (probabilities)
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with fitted model parameters, results, and residual std dev.
    """
    x_data = np.asarray(x_data)
    y_data = np.asarray(y_data)
    if x_data.ndim == 1:
        x_data = x_data[:, np.newaxis]

    # Calculate logit transform of y_data, handling NaNs
    try:
        y_logit = logit_transform(y_data)
    except Exception as e:
        print(f"Error during logit_transform in fit_bootstrap_model: {e}")
        raise ValueError("Logit transformation failed.") from e

    valid_mask = ~np.isnan(y_logit) & ~np.isnan(x_data.flatten())  # Also check x
    if not np.any(valid_mask):
        raise ValueError("No valid data points after logit transform and NaN check.")
    if not np.all(valid_mask):
        print(
            f"Warning: Removing {np.sum(~valid_mask)} points with NaN in x or logit(y) during fit."
        )
        x_valid = x_data[valid_mask]
        y_logit_valid = y_logit[valid_mask]
    else:
        x_valid = x_data
        y_logit_valid = y_logit

    n_samples = len(x_valid)
    if n_samples < 2:
        raise ValueError(f"Need at least 2 valid data points to fit, got {n_samples}.")

    np.random.seed(random_seed)
    w_samples = []
    b_samples = []
    valid_bootstrap_count = 0
    all_residuals = []  # Store residuals from each valid bootstrap iteration

    print(f"  Fitting on {n_samples} valid data points.")  # Debug print

    for i in range(n_bootstrap):
        try:
            indices = resample(np.arange(n_samples), replace=True, n_samples=n_samples)
            # Ensure indices is not empty if n_samples is small but >= 2
            if len(indices) == 0:
                print(
                    f"Warning: Bootstrap iteration {i + 1} produced empty indices. Skipping."
                )
                continue

            x_resampled = x_valid[indices]
            y_resampled = y_logit_valid[indices]

            # Check for constant features or targets in the resample
            if len(np.unique(x_resampled)) < 2:
                # print(f"Warning: Bootstrap iteration {i+1} has constant x_resampled. Skipping.")
                continue
            if (
                len(np.unique(y_resampled)) < 1
            ):  # Should not happen if len>=2, but check
                print(
                    f"Warning: Bootstrap iteration {i + 1} has constant y_resampled. Skipping."
                )
                continue

            model = LinearRegression()
            model.fit(x_resampled, y_resampled)

            # Check if fit produced valid coefficients
            if np.isnan(model.coef_[0]) or np.isnan(model.intercept_):
                print(
                    f"Warning: Bootstrap iteration {i + 1} resulted in NaN coefficients. Skipping."
                )
                continue

            # Store parameters
            w_samples.append(model.coef_[0])
            b_samples.append(model.intercept_)
            valid_bootstrap_count += 1

            # Calculate and store residuals for this bootstrap sample
            y_pred_logit_resampled = model.predict(x_resampled)
            residuals_sample = y_resampled - y_pred_logit_resampled
            all_residuals.append(residuals_sample)

        except ValueError as ve:
            # Catch specific errors like constant features if not caught above
            print(
                f"Warning: ValueError during bootstrap iteration {i + 1}: {ve}. Skipping."
            )
        except Exception as e:
            # Catch any other unexpected errors during fitting
            print(
                f"Warning: Unexpected error during bootstrap iteration {i + 1}: {e}. Skipping."
            )

    if valid_bootstrap_count == 0:
        raise RuntimeError(
            "All bootstrap iterations failed or were skipped. Cannot proceed."
        )

    print(
        f"  Successfully completed {valid_bootstrap_count} bootstrap fits."
    )  # Debug print

    # Calculate residual standard deviation (sigma_epsilon)
    sigma_epsilon = 0.0  # Default if no residuals collected
    if all_residuals:
        try:
            # Concatenate all residuals into a single array
            combined_residuals = np.concatenate(all_residuals)
            # Calculate standard deviation
            if combined_residuals.size > 1:  # Need at least 2 residuals for std dev
                sigma_epsilon = np.std(combined_residuals)
                print(
                    f"  Estimated residual standard deviation (sigma_epsilon): {sigma_epsilon:.4f}"
                )
            else:
                print(
                    "  Warning: Not enough residuals to calculate standard deviation. sigma_epsilon set to 0."
                )
        except Exception as e:
            print(f"  Warning: Error calculating sigma_epsilon: {e}. Setting to 0.")
            sigma_epsilon = 0.0
    else:
        print("  Warning: No residuals collected. sigma_epsilon set to 0.")

    w_samples_arr = np.array(w_samples)
    b_samples_arr = np.array(b_samples)

    # Final check for NaNs in collected samples (should be prevented by checks above)
    if np.any(np.isnan(w_samples_arr)) or np.any(np.isnan(b_samples_arr)):
        warnings.warn(
            "NaN values found in collected bootstrap parameters despite checks."
        )
        w_samples_arr = w_samples_arr[
            ~np.isnan(w_samples_arr) & ~np.isnan(b_samples_arr)
        ]
        b_samples_arr = b_samples_arr[
            ~np.isnan(w_samples_arr) & ~np.isnan(b_samples_arr)
        ]
        valid_bootstrap_count = len(w_samples_arr)
        if valid_bootstrap_count == 0:
            raise RuntimeError("All bootstrap samples became NaN after final check.")

    return {
        "w_samples": w_samples_arr,
        "b_samples": b_samples_arr,
        "y_logit": y_logit_valid,  # Return the logit-y used for fitting
        "n_valid_bootstrap": valid_bootstrap_count,
        "sigma_epsilon": sigma_epsilon,  # Add estimated residual noise std dev
    }


def predict_bootstrap(
    bootstrap_results,
    x_new,
    y_true=None,  # Training y (prob space)
    y_true_logit=None,  # Training y (logit space)
    x_data=None,  # Original training x - needed for metrics
    # Removed test data arguments
    return_logit=False,
    add_residual_noise=True,  # Flag to control adding noise
):
    """
    Generate predictions from bootstrap model, optionally adding residual noise.

    Args:
        bootstrap_results: Dictionary from fit_bootstrap_model.
        x_new: New data points for prediction (grid or specific points).
        y_true: True target values (probability space) for metrics.
        y_true_logit: True target values (logit space) for metrics.
        x_data: Original training features used for fitting (for train metrics).
        return_logit: If True, return logit samples instead of probability samples.
        add_residual_noise: If True, add noise based on sigma_epsilon from fit.

    Returns:
        Dictionary containing predictions and potentially metrics.
    """
    w_samples = bootstrap_results["w_samples"]
    b_samples = bootstrap_results["b_samples"]
    sigma_epsilon = bootstrap_results.get(
        "sigma_epsilon", 0.0
    )  # Get sigma_epsilon, default 0

    if add_residual_noise and sigma_epsilon == 0.0:
        warnings.warn(
            "add_residual_noise is True, but sigma_epsilon is 0.0. No noise will be added."
        )
    elif not add_residual_noise and sigma_epsilon > 0.0:
        print(
            "Info: add_residual_noise is False. Residual noise will not be added to predictions."
        )

    # Ensure x_new is 2D for broadcasting
    x_grid = np.asarray(x_new)
    if x_grid.ndim == 1:
        x_grid = x_grid[:, np.newaxis]  # Shape (n_points, 1)

    # Predict on the grid (logit space)
    # w_samples shape (n_bootstrap,), b_samples shape (n_bootstrap,)
    # x_grid shape (n_grid_points, 1)
    # Result should be (n_bootstrap, n_grid_points)
    samples_logit_grid = w_samples[:, np.newaxis] * x_grid.T + b_samples[:, np.newaxis]

    # Add residual noise if requested and available
    if add_residual_noise and sigma_epsilon > 0.0:
        noise_grid = np.random.normal(0, sigma_epsilon, size=samples_logit_grid.shape)
        samples_logit_grid += noise_grid

    # Transform to probability space using scipy.special.expit for numerical stability
    samples_prob_grid = expit(samples_logit_grid)

    # Calculate mean predictions across bootstrap samples
    mean_pred_logit_grid = np.mean(samples_logit_grid, axis=0)
    mean_pred_prob_grid = np.mean(samples_prob_grid, axis=0)

    results = {
        "samples_logit": samples_logit_grid,
        "samples_prob": samples_prob_grid,
        "mean_logit": mean_pred_logit_grid,
        "mean_prob": mean_pred_prob_grid,
        # No test metrics calculation here anymore
    }

    # --- Calculate metrics on training data if provided ---
    if x_data is not None and y_true is not None and y_true_logit is not None:
        print("  Calculating metrics on training data...")
        x_data_arr = np.asarray(x_data)
        if x_data_arr.ndim == 1:
            x_data_arr = x_data_arr[:, np.newaxis]  # Shape (n_data, 1)

        # Predict on training data (logit space)
        # Result shape: (n_bootstrap, n_data)
        samples_logit_train = (
            w_samples[:, np.newaxis] * x_data_arr.T + b_samples[:, np.newaxis]
        )

        # Add residual noise if requested and available
        if add_residual_noise and sigma_epsilon > 0.0:
            noise_train = np.random.normal(
                0, sigma_epsilon, size=samples_logit_train.shape
            )
            samples_logit_train += noise_train

        # Transform to probability space
        samples_prob_train = expit(samples_logit_train)

        # Calculate mean predictions for training data
        mean_pred_logit_train = np.mean(samples_logit_train, axis=0)
        mean_pred_prob_train = np.mean(samples_prob_train, axis=0)

        # Calculate metrics
        train_metrics = _calculate_metrics(
            y_true_prob=y_true,
            y_pred_prob=mean_pred_prob_train,
            y_true_logit=y_true_logit,
            y_pred_logit=mean_pred_logit_train,
        )

        results["train_predictions"] = {
            "samples_logit": samples_logit_train,
            "samples_prob": samples_prob_train,
            "mean_logit": mean_pred_logit_train,
            "mean_prob": mean_pred_prob_train,
        }
        results["train_metrics"] = train_metrics
    else:
        if x_data is not None or y_true is not None or y_true_logit is not None:
            print(
                "  Skipping training metrics calculation: Missing one of x_data, y_true, y_true_logit."
            )

    # Optional: Return logit samples directly if requested
    if return_logit:
        # The grid samples are already calculated
        # If train data was provided, train logit samples are also calculated
        # No need to recalculate, just make sure they are in the results
        pass  # They are already in the results dictionary

    return results
