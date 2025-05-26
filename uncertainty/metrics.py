import numpy as np
import pandas as pd  # For pd.notna if used, though numpy handles NaNs


def calculate_calibration_metrics(y_true, samples_pred_prob, alpha=0.05, alphas=None):
    """
    Calculate calibration metrics for uncertainty quantification.

    Args:
        y_true: True values (1D array)
        samples_pred_prob: Prediction samples (2D array: n_samples × n_predictions)
        alpha: Single alpha level for confidence intervals (default: 0.05 for 95% CI)
        alphas: List of alpha levels to calculate metrics for (overrides single alpha)

    Returns:
        dict: Calibration metrics including coverage, miscoverage, width, and interval score
    """
    # Input Validation
    if samples_pred_prob is None or y_true is None:
        print("Warning: Invalid input (None) for calibration metrics calculation.")
        return {}
    if (
        not isinstance(samples_pred_prob, np.ndarray)
        or samples_pred_prob.ndim != 2
        or samples_pred_prob.shape[1] == 0
    ):
        print(
            f"Warning: Invalid samples_pred_prob shape/type ({type(samples_pred_prob)}, {getattr(samples_pred_prob, 'shape', 'N/A')}) for calibration."
        )
        return {}
    y_true = np.asarray(y_true)  # Ensure y_true is a numpy array
    if y_true.ndim != 1 or len(y_true) == 0:
        print(
            f"Warning: Invalid y_true shape/type ({type(y_true)}, {y_true.shape}) for calibration."
        )
        return {}
    if samples_pred_prob.shape[1] != len(y_true):
        print(
            f"Warning: Mismatch between samples ({samples_pred_prob.shape[1]}) and y_true ({len(y_true)}) for calibration."
        )
        return {}

    # Filter out NaNs in y_true and corresponding samples
    valid_mask = ~np.isnan(y_true)
    if not np.all(valid_mask):
        n_removed = np.sum(~valid_mask)
        print(f"Warning: Removing {n_removed} NaN y_true values for calibration.")
        if n_removed == len(y_true):  # All values were NaN
            print("Warning: All y_true values were NaN after filtering.")
            return {}
        y_true = y_true[valid_mask]
        # Filter columns in samples_pred_prob safely
        try:
            samples_pred_prob = samples_pred_prob[:, valid_mask]
            if (
                samples_pred_prob.shape[1] == 0
            ):  # Check if filtering resulted in empty samples
                print(
                    "Warning: No valid data left after NaN filtering samples for calibration."
                )
                return {}
        except IndexError:
            print(
                "Warning: Failed to filter samples_pred_prob columns based on y_true NaNs."
            )
            return {}

    # Determine alpha levels to process
    if alphas is not None:
        alpha_levels = alphas
    else:
        alpha_levels = [alpha]

    all_metrics = {}

    for current_alpha in alpha_levels:
        # Calculate percentiles for the interval
        lower_p = current_alpha / 2 * 100
        upper_p = (1 - current_alpha / 2) * 100
        try:
            # Added check for sufficient samples
            if samples_pred_prob.shape[0] < 2:
                print(
                    f"Warning: Need at least 2 samples for percentile calculation, got {samples_pred_prob.shape[0]}."
                )
                continue
            lower_bounds = np.percentile(samples_pred_prob, lower_p, axis=0)
            upper_bounds = np.percentile(samples_pred_prob, upper_p, axis=0)
        except Exception as e:
            print(f"Warning: Failed to calculate percentiles for calibration: {e}")
            continue

        # 1. Coverage Probability
        is_contained = (y_true >= lower_bounds) & (y_true <= upper_bounds)
        observed_coverage = np.mean(is_contained)

        # 2. Miscoverage (how far off from expected coverage)
        expected_coverage = 1 - current_alpha
        miscoverage = observed_coverage - expected_coverage
        abs_miscoverage = np.abs(miscoverage)

        # 3. Count of miscovered points
        n_points = len(y_true)
        n_covered = np.sum(is_contained)
        n_expected_covered = round(
            expected_coverage * n_points
        )  # Round to nearest integer
        n_miscovered = abs(n_covered - n_expected_covered)

        # 4. Average Interval Width
        widths = upper_bounds - lower_bounds
        if np.any(np.isnan(widths)):
            print("Warning: NaN found in calculated interval widths.")
            average_width = np.nanmean(widths)  # Calculate mean ignoring NaNs
        else:
            average_width = np.mean(widths)

        # 5. Interval Score (Winkler Score)
        penalty_below = (
            (2 / current_alpha) * (lower_bounds - y_true) * (y_true < lower_bounds)
        )
        penalty_above = (
            (2 / current_alpha) * (y_true - upper_bounds) * (y_true > upper_bounds)
        )
        interval_scores = widths + penalty_below + penalty_above
        if np.any(np.isnan(interval_scores)):
            print("Warning: NaN found in calculated interval scores.")
            average_interval_score = np.nanmean(
                interval_scores
            )  # Calculate mean ignoring NaNs
        else:
            average_interval_score = np.mean(interval_scores)

        # Create metric names with confidence level
        level_str = f"{1 - current_alpha:.0%}"
        alpha_str = f"{current_alpha:.3f}".rstrip("0").rstrip(".")

        current_metrics = {
            f"coverage_{level_str}": observed_coverage,
            f"miscoverage_{level_str}": miscoverage,
            f"abs_miscoverage_{level_str}": abs_miscoverage,
            f"n_miscovered_{level_str}": n_miscovered,
            f"n_points_{level_str}": n_points,
            f"avg_width_{level_str}": average_width,
            f"avg_interval_score_{level_str}": average_interval_score,
        }

        # If processing multiple alphas, add alpha-specific keys
        if len(alpha_levels) > 1:
            for key, value in current_metrics.items():
                all_metrics[f"{key}_alpha{alpha_str}"] = value
        else:
            all_metrics.update(current_metrics)

    return all_metrics
