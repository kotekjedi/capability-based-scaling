"""
Functions for evaluating the quality of uncertainty quantification (UQ).
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm


def calculate_sharpness(samples_prob, ci_level=0.95):
    """
    Calculates the sharpness of the predictive distribution.

    Sharpness is measured as the average width of the credible intervals.
    Lower values indicate sharper (more confident) predictions.

    Args:
        samples_prob: Posterior predictive samples in probability space.
                      Shape (n_samples, n_data_points).
        ci_level: The confidence level for the credible interval (e.g., 0.95 for 95%).

    Returns:
        Average width of the credible intervals across all data points.
    """
    if samples_prob.ndim != 2:
        raise ValueError("samples_prob must be a 2D array (n_samples, n_data_points)")
    if not 0 < ci_level < 1:
        raise ValueError("ci_level must be between 0 and 1")

    lower_quantile = (1 - ci_level) / 2
    upper_quantile = 1 - lower_quantile

    lower_bounds = np.percentile(samples_prob, lower_quantile * 100, axis=0)
    upper_bounds = np.percentile(samples_prob, upper_quantile * 100, axis=0)

    widths = upper_bounds - lower_bounds
    avg_width = np.mean(widths)

    return avg_width


def plot_reliability_diagram(y_true, y_pred_prob, n_bins=10):
    """
    Plots a reliability diagram and calculates the Expected Calibration Error (ECE).

    Args:
        y_true: Array of true binary outcomes (0 or 1) or continuous probabilities.
                If continuous, they represent the true probability for each instance.
        y_pred_prob: Array of predicted probabilities for the positive class or mean predicted probability.
        n_bins: Number of bins to divide the probability space [0, 1].

    Returns:
        fig: Matplotlib figure object for the reliability diagram.
        ax: Matplotlib axes object.
        ece: Expected Calibration Error (scalar value).
    """
    if len(y_true) != len(y_pred_prob):
        raise ValueError("y_true and y_pred_prob must have the same length.")
    if np.any((y_pred_prob < 0) | (y_pred_prob > 1)):
        print(
            "Warning: y_pred_prob contains values outside [0, 1]. Clipping for binning."
        )
        y_pred_prob = np.clip(y_pred_prob, 0, 1)
    # Determine if y_true is binary or probabilities - assume probabilities for ASR
    is_binary = np.all(np.isin(y_true, [0, 1]))
    if is_binary:
        print(
            "Warning: y_true appears binary. Treating as observed frequencies (0 or 1). For ASR, y_true might be expected probabilities."
        )

    bin_limits = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_limits[:-1]
    bin_uppers = bin_limits[1:]

    # Ensure the last bin includes 1.0
    bin_uppers[-1] = 1.0

    binned_data = pd.DataFrame(
        {
            "y_true": y_true,
            "y_pred_prob": y_pred_prob,
            "bin": np.digitize(
                y_pred_prob, bin_limits[1:-1], right=False
            ),  # Assign bin index
        }
    )

    bin_stats = (
        binned_data.groupby("bin")
        .agg(
            count=("y_pred_prob", "size"),
            mean_pred_prob=("y_pred_prob", "mean"),
            mean_true_prob=(
                "y_true",
                "mean",
            ),  # For ASR, this is the mean observed ASR in the bin
        )
        .reset_index()
    )

    # Calculate ECE
    total_samples = len(y_true)
    ece = np.sum(
        bin_stats["count"]
        / total_samples
        * np.abs(bin_stats["mean_true_prob"] - bin_stats["mean_pred_prob"])
    )

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot calibration curve
    ax.plot(
        bin_stats["mean_pred_prob"],
        bin_stats["mean_true_prob"],
        "o-",
        label="Model Calibration",
        color="blue",
        markersize=8,
    )

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Plot histogram of predicted probabilities
    ax_hist = ax.twinx()
    ax_hist.hist(
        y_pred_prob,
        bins=bin_limits,
        histtype="step",
        color="red",
        linewidth=1.5,
        alpha=0.7,
        label="Prediction Distribution",
    )
    ax_hist.set_ylabel("Count", color="red")
    ax_hist.tick_params(axis="y", labelcolor="red")
    ax_hist.grid(False)

    # Add bin counts as text labels slightly above histogram bars for clarity
    bin_counts = binned_data.groupby("bin").size()
    bin_centers = (bin_lowers + bin_uppers) / 2
    max_hist_height = ax_hist.get_ylim()[1]
    for i, count in enumerate(bin_counts):
        if i < len(bin_centers):  # Check if index is valid
            ax_hist.text(
                bin_centers[i],
                max_hist_height * 0.05,
                str(count),  # Position above bottom
                ha="center",
                va="bottom",
                color="red",
                alpha=0.8,
                fontsize=9,
            )

    # Formatting
    ax.set_xlabel("Mean Predicted Probability (in bin)")
    ax.set_ylabel("Mean Observed ASR (in bin)")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_title(f"Reliability Diagram (ECE = {ece:.4f})")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, ax, ece


def evaluate_uq(samples_prob_test, y_test, ci_level=0.95, n_bins=10):
    """
    Evaluates Uncertainty Quantification metrics for a set of probability samples.

    Calculates sharpness and ECE, and plots the reliability diagram.

    Args:
        samples_prob_test: Posterior predictive samples for the test set.
                           Shape (n_samples, n_test_points).
        y_test: True outcomes for the test set. Shape (n_test_points,).
                 These should be the actual ASR values (probabilities).
        ci_level: Confidence level for sharpness calculation (e.g., 0.95).
        n_bins: Number of bins for the reliability diagram and ECE.

    Returns:
        dict: A dictionary containing:
              'sharpness': Average width of the credible interval.
              'ece': Expected Calibration Error.
              'reliability_fig': Matplotlib Figure object for the reliability diagram.
              'reliability_ax': Matplotlib Axes object for the reliability diagram.
    """
    if samples_prob_test.ndim != 2:
        raise ValueError("samples_prob_test must be 2D (n_samples, n_test_points)")
    if samples_prob_test.shape[1] != len(y_test):
        raise ValueError(
            "Number of data points mismatch between samples_prob_test and y_test."
        )

    # 1. Calculate Sharpness
    sharpness = calculate_sharpness(samples_prob_test, ci_level=ci_level)

    # 2. Calculate ECE and Plot Reliability Diagram
    # Use the mean predicted probability for calibration assessment
    mean_pred_prob_test = np.mean(samples_prob_test, axis=0)
    fig, ax, ece = plot_reliability_diagram(y_test, mean_pred_prob_test, n_bins=n_bins)

    # 3. Package results
    results = {
        f"sharpness_{int(ci_level * 100)}ci": sharpness,
        "ece": ece,
        "reliability_fig": fig,
        "reliability_ax": ax,
    }

    return results


# Example Usage (if run directly)
if __name__ == "__main__":
    print("Running UQ Evaluation Example...")

    # Generate synthetic data
    np.random.seed(42)
    n_samples = 1000
    n_test_points = 200

    # Simulate well-calibrated but not very sharp predictions
    true_p = np.random.rand(n_test_points)  # True probabilities (like ASR)
    # Simulate samples centered around true_p but with some spread
    sim_samples = np.random.normal(
        loc=true_p, scale=0.15, size=(n_samples, n_test_points)
    )
    sim_samples = np.clip(sim_samples, 0.01, 0.99)  # Clip to valid probability range

    # Simulate poorly-calibrated (overconfident) predictions
    overconfident_p = np.clip(true_p * 1.2 - 0.1, 0, 1)  # Bias towards extremes
    overconfident_samples = np.random.normal(
        loc=overconfident_p, scale=0.05, size=(n_samples, n_test_points)
    )  # Narrower spread
    overconfident_samples = np.clip(overconfident_samples, 0.01, 0.99)

    print("\n--- Evaluating Well-Calibrated Model ---")
    uq_results_good = evaluate_uq(sim_samples, true_p, n_bins=10)
    print(f"Sharpness (95% CI): {uq_results_good['sharpness_95ci']:.4f}")
    print(f"ECE: {uq_results_good['ece']:.4f}")
    uq_results_good["reliability_fig"].suptitle(
        "Well-Calibrated Example Reliability Diagram"
    )
    plt.show()  # Display the plot

    print("\n--- Evaluating Overconfident Model ---")
    uq_results_bad = evaluate_uq(overconfident_samples, true_p, n_bins=10)
    print(
        f"Sharpness (95% CI): {uq_results_bad['sharpness_95ci']:.4f}"
    )  # Should be lower (sharper)
    print(f"ECE: {uq_results_bad['ece']:.4f}")  # Should be higher (worse calibration)
    uq_results_bad["reliability_fig"].suptitle(
        "Overconfident Example Reliability Diagram"
    )
    plt.show()  # Display the plot
