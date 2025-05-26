"""
General plotting utilities for uncertainty estimation in ASR prediction.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D  # Import Line2D for proxy artists
from scipy.special import logit  # Assuming logit might be needed elsewhere


# Helper function to plot scatter points grouped by attack type
def _plot_scatter_by_attack(
    ax, x, y, attack_data, default_label, marker_list=None, color_list=None, **kwargs
):
    """Plots scatter points, assigning different markers/colors based on attack_data.

    Returns:
        dict: A map from attack type to the marker used for it.
    """
    attack_marker_map = {}

    if attack_data is None:
        # Default behavior: plot all points with one style
        # Use the first default marker if none specified in kwargs
        marker = kwargs.get("marker", "o")
        ax.scatter(x, y, label="_nolegend_", **kwargs)  # Plot without label
        # Even with no attack_data, return a placeholder if needed for consistency
        # attack_marker_map[default_label] = marker # Or return empty? Let's return map based on actual attacks. Returns {} if attack_data is None.
        return attack_marker_map

    # Specific marker mapping
    specific_markers = {
        "pair": "o",  # Circle
        "crescendo": "^",  # Triangle up
        "direct": "*",  # Star
    }
    default_markers = ["s", "D", "v", "<", ">", "p", "X"]  # Fallback for others

    unique_attacks = sorted(list(np.unique(attack_data)))

    # Determine markers to use
    markers_to_use = []
    use_specific = all(attack in specific_markers for attack in unique_attacks)

    if use_specific:
        markers_to_use = [specific_markers[attack] for attack in unique_attacks]
        # print(f"Using specific markers: {markers_to_use}") # Debug print
    elif marker_list is not None:
        if len(marker_list) < len(unique_attacks):
            # print("Warning: Provided marker_list is shorter than unique attacks. Markers will be reused.")
            markers_to_use = [
                marker_list[i % len(marker_list)] for i in range(len(unique_attacks))
            ]
        else:
            markers_to_use = marker_list[: len(unique_attacks)]
        # print(f"Using provided markers: {markers_to_use}") # Debug print
    else:
        # print("Using default marker cycling.") # Debug print
        current_defaults = default_markers
        markers_to_use = [
            current_defaults[i % len(current_defaults)]
            for i in range(len(unique_attacks))
        ]

    # Determine colors (keep existing logic)
    if color_list is None:
        base_color = kwargs.pop("color", "gray")
        colors_to_use = [base_color] * len(unique_attacks)
    else:
        if len(color_list) < len(unique_attacks):
            # print("Warning: Not enough unique colors provided for all attack types. Colors will be reused.")
            colors_to_use = [
                color_list[i % len(color_list)] for i in range(len(unique_attacks))
            ]
        else:
            colors_to_use = color_list[: len(unique_attacks)]

    # Plotting loop
    for i, attack in enumerate(unique_attacks):
        mask = attack_data == attack
        # label = f"{default_label} ({attack})" # Old label
        marker = markers_to_use[i]
        color = colors_to_use[i]
        attack_marker_map[attack] = marker  # Store the marker used

        ax.scatter(
            x[mask],
            y[mask],
            label="_nolegend_",  # Plot without adding to legend automatically
            marker=marker,
            color=color,
            **kwargs,
        )
    return attack_marker_map


def plot_uncertainty(
    x_grid,
    samples_prob,
    x_data=None,
    y_data=None,
    attack_data=None,
    xlim=(-1, 1),
    ylim=(0, 1),
    title=None,
    method_name="Method",
    color="maroon",
    individual_means=None,
    metrics=None,
    sigmas_to_plot=[1, 2],
):
    """
    Plot uncertainty estimates for ASR prediction.

    Args:
        x_grid: Array of x values where predictions were made
        samples_prob: Array of samples in probability space (shape: n_samples × n_grid_points)
        x_data: Original data x values (optional)
        y_data: Original data y values (optional)
        attack_data: Array of attack types corresponding to x_data/y_data (optional)
        xlim: x-axis limits
        ylim: y-axis limits
        title: Plot title
        method_name: Name of the method for title
        color: Color for the uncertainty bands
        individual_means: List of individual means to plot (optional)
        metrics: Dictionary of metrics for title
        sigmas_to_plot: List of sigma levels to plot (default: [1, 2])

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
    """
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot original data if provided, differentiating by attack
    if x_data is not None and y_data is not None:
        _plot_scatter_by_attack(
            ax=ax,
            x=x_data,
            y=y_data,
            attack_data=attack_data,
            default_label="Data",
            color="gray",
            alpha=0.6,
            s=30,
            linewidth=0,
        )

    # Plot individual means if provided
    if individual_means is not None:
        for i, mean in enumerate(individual_means):
            if i == 0:
                ax.plot(
                    x_grid,
                    mean,
                    color="gray",
                    alpha=0.4,
                    lw=1,
                    label="Individual Means",
                )
            else:
                ax.plot(x_grid, mean, color="gray", alpha=0.4, lw=1)

    # Compute and plot mean/median prediction
    mean_prob = np.mean(samples_prob, axis=0)
    median_prob = np.median(samples_prob, axis=0)
    ax.plot(x_grid, median_prob, color=color, lw=4, label="Median prediction")

    # Compute and plot uncertainty bands
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

    # Format plot
    ax.set_xlabel("Capability Difference")
    ax.set_ylabel("ASR")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title is None:
        title = f"Uncertainty Estimation using {method_name}"
    if metrics is not None:
        metrics_str_parts = []
        if "train" in metrics and metrics["train"]:
            train_met = metrics["train"]
            # R² metrics
            r2_str = f"R²_prob={train_met.get('r2_prob', float('nan')):.2f}, R²_logit={train_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Train: {r2_str}")
        elif "train_metrics" in metrics and metrics["train_metrics"]:
            train_met = metrics["train_metrics"]
            # R² metrics
            r2_str = f"R²_prob={train_met.get('r2_prob', float('nan')):.2f}, R²_logit={train_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Train: {r2_str}")

        # Add calibration metrics if available
        if "train_calibration" in metrics and metrics["train_calibration"]:
            calib_met = metrics["train_calibration"]
            # Look for miscovered point counts (common confidence levels)
            miscov_parts = []
            for level in ["95%", "90%", "68%"]:
                n_miscov_key = f"n_miscovered_{level}"
                n_points_key = f"n_points_{level}"
                if n_miscov_key in calib_met and n_points_key in calib_met:
                    n_miscov_raw = calib_met[n_miscov_key]
                    n_points_raw = calib_met[n_points_key]

                    # Check for NaN before converting to int
                    if (
                        not np.isnan(n_miscov_raw)
                        and not np.isnan(n_points_raw)
                        and n_points_raw > 0
                    ):
                        n_miscov = int(n_miscov_raw)
                        n_points = int(n_points_raw)
                        miscov_percentage = 100 * n_miscov / n_points

                        # Assert consistency: if n_miscov is 0, percentage must be 0
                        if n_miscov == 0:
                            assert miscov_percentage == 0.0, (
                                f"Bug: {n_miscov}/{n_points} should give 0% but got {miscov_percentage}%"
                            )

                        miscov_parts.append(
                            f"Miscov_{level}={n_miscov}/{n_points} ({miscov_percentage:.1f}%)"
                        )
            if miscov_parts:
                metrics_str_parts.append(f"Calibration: {', '.join(miscov_parts)}")

        if "test_metrics" in metrics and metrics["test_metrics"]:
            test_met = metrics["test_metrics"]
            # R² metrics
            r2_str = f"R²_prob={test_met.get('r2_prob', float('nan')):.2f}, R²_logit={test_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Test: {r2_str}")
        metrics_str = ", ".join(metrics_str_parts)
        title += f"\nMetrics: {metrics_str}"
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_linear_model_dual(
    x_grid,
    samples_prob,
    samples_logit,
    x_data=None,
    y_data=None,
    y_logit=None,
    attack_data=None,
    x_test=None,
    y_test=None,
    y_test_logit=None,
    attack_test=None,
    xlim=(-1, 1),
    ylim=(0, 1),
    title=None,
    metrics=None,
    test_metrics=None,
    mean_w=None,
    mean_b=None,
    sigmas_to_plot=[1, 2],
    train_legend_label="Train",
    test_legend_label="Test",
):
    """
    Plot linear model results in both probability and logit space.

    Args:
        x_grid: Array of x values where predictions were made
        samples_prob: Array of samples in probability space
        samples_logit: Array of samples in logit space
        x_data: Original training data x values (optional)
        y_data: Original training data y values (optional)
        y_logit: Original training data y values in logit space (optional)
        attack_data: Array of attack types for training data (optional)
        x_test: Test data x values (optional)
        y_test: Test data y values (optional)
        y_test_logit: Test data y values in logit space (optional)
        attack_test: Array of attack types for test data (optional)
        xlim: x-axis limits
        ylim: y-axis limits for probability space
        title: Plot title
        metrics: Dictionary of metrics (e.g., from predict function) for title
        test_metrics: Specific dictionary of test metrics for title (optional)
        mean_w: Slope for linear model (optional, for display)
        mean_b: Intercept for linear model (optional, for display)
        sigmas_to_plot: List of sigma levels to plot (default: [1, 2])
        train_legend_label: Base legend label for training data
        test_legend_label: Base legend label for test data

    Returns:
        fig: Matplotlib figure
        (ax1, ax2): Tuple of axes for probability and logit space plots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # --- Plotting Setup ---
    sigma_levels = sorted(sigmas_to_plot)  # Ensure ordered plotting
    sigma_percentiles = {1: (15.87, 84.13), 2: (2.28, 97.72), 3: (0.13, 99.87)}
    confidence_levels = {1: "68%", 2: "95%", 3: "99.7%"}
    alphas = {1: 0.2, 2: 0.1, 3: 0.05}
    # Test data styling (could potentially be varied by attack too if needed)
    default_test_marker = "X"
    default_test_color = "#FF6347"  # Tomato
    default_test_edgecolor = "black"
    default_test_markersize = 60
    default_test_linewidth = 1
    default_test_zorder = 10

    # Define markers for attack types (consistent across plots)
    # Could be passed as an argument later if needed
    attack_markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "X"]

    # --- Create Proxy Artists Function --- (Helper within the function)
    def create_scatter_legend_proxies(
        attack_marker_map, title, base_color, base_marker=None
    ):
        handles = []
        labels = []
        if not attack_marker_map and base_marker is None:
            return handles, labels  # Nothing to add

        # Add title using a dummy handle
        handles.append(Line2D([0], [0], linestyle="None", marker="None", label=""))
        labels.append(title)

        if attack_marker_map:
            for attack, marker in attack_marker_map.items():
                handle = Line2D(
                    [0],
                    [0],
                    linestyle="None",
                    marker=marker,
                    color=base_color,
                    markersize=6,
                )
                handles.append(handle)
                labels.append(f"  {attack}")  # Indented label
        elif base_marker:
            # Case where attack_data was None, but we still want a single entry
            handle = Line2D(
                [0],
                [0],
                linestyle="None",
                marker=base_marker,
                color=base_color,
                markersize=6,
            )
            handles.append(handle)
            labels.append(f"  Data")  # Generic label

        return handles, labels

    # --- Store handles and labels for model parts ---
    model_handles_ax1, model_labels_ax1 = [], []
    model_handles_ax2, model_labels_ax2 = [], []

    # --- First plot: Probability space (ax1) ---
    # Plot training data (gets marker map)
    train_attack_marker_map = {}
    if x_data is not None and y_data is not None:
        train_attack_marker_map = _plot_scatter_by_attack(
            ax=ax1,
            x=x_data,
            y=y_data,
            attack_data=attack_data,
            default_label=train_legend_label,  # Not used for label now
            marker_list=attack_markers,
            color="gray",
            alpha=0.7,
            s=30,
            linewidth=0,
        )

    # Plot test data (gets marker map)
    test_attack_marker_map = {}
    if x_test is not None and y_test is not None:
        test_attack_marker_map = _plot_scatter_by_attack(
            ax=ax1,
            x=x_test,
            y=y_test,
            attack_data=attack_test,
            default_label=test_legend_label,  # Not used for label now
            marker_list=attack_markers,
            color=default_test_color,
            edgecolor=default_test_edgecolor,
            s=default_test_markersize,
            linewidths=default_test_linewidth,
            alpha=0.9,
            zorder=default_test_zorder,
        )

    # --- Calculate and plot model elements for ax1 ---
    model_handles_ax1, model_labels_ax1 = [], []

    # Calculate statistics in probability space
    mean_prob = np.mean(samples_prob, axis=0)
    median_prob = np.median(samples_prob, axis=0)  # Calculate here

    # Plot median prediction and capture handle
    (median_handle_ax1,) = ax1.plot(
        x_grid, median_prob, color="maroon", lw=2, label="Median prediction"
    )
    model_handles_ax1.append(median_handle_ax1)
    model_labels_ax1.append("Median prediction")

    # Calculate and plot credible intervals and capture handles
    for sigma in sigma_levels:
        alpha = alphas[sigma]
        lower, upper = sigma_percentiles[sigma]
        lower_bound = np.percentile(samples_prob, lower, axis=0)
        upper_bound = np.percentile(samples_prob, upper, axis=0)
        # Create proxy for fill_between
        label_ci = f"±{confidence_levels[sigma]} CI"
        # Use maroon color with the correct alpha for the proxy
        proxy = plt.Rectangle((0, 0), 1, 1, fc="maroon", alpha=alpha, label=label_ci)
        model_handles_ax1.append(proxy)
        model_labels_ax1.append(label_ci)
        ax1.fill_between(
            x_grid,
            lower_bound,
            upper_bound,
            color="maroon",
            alpha=alpha,
            label="_nolegend_",
            linewidth=0,  # Ensure linewidth=0
        )

    # --- Second plot: Logit space (ax2) ---
    # Plot training data (gets marker map - should be same as ax1)
    if x_data is not None and y_logit is not None:
        _plot_scatter_by_attack(
            ax=ax2,
            x=x_data,
            y=y_logit,
            attack_data=attack_data,
            default_label=train_legend_label + " (logit)",
            marker_list=attack_markers,
            color="gray",
            alpha=0.7,
            s=30,
            linewidth=0,
        )

    # Plot test data (gets marker map - should be same as ax1)
    if x_test is not None and y_test_logit is not None:
        _plot_scatter_by_attack(
            ax=ax2,
            x=x_test,
            y=y_test_logit,
            attack_data=attack_test,
            default_label=test_legend_label + " (logit)",
            marker_list=attack_markers,
            color=default_test_color,
            edgecolor=default_test_edgecolor,
            s=default_test_markersize,
            linewidths=default_test_linewidth,
            alpha=0.9,
            zorder=default_test_zorder,
        )

    # --- Calculate and plot model elements for ax2 ---
    model_handles_ax2, model_labels_ax2 = [], []

    # Calculate statistics in logit space
    mean_logit = np.mean(samples_logit, axis=0)
    median_logit = np.median(samples_logit, axis=0)  # Calculate here

    # Plot median prediction in logit space (capture handle/label)
    median_label = "Median (logit)"
    if mean_w is not None and mean_b is not None:
        median_label += f" ({mean_w:.2f}x + {mean_b:.2f})"
    (median_handle,) = ax2.plot(
        x_grid, median_logit, color="blue", lw=2, label=median_label
    )
    model_handles_ax2.append(median_handle)
    model_labels_ax2.append(median_label)

    # Plot credible intervals in logit space (capture handles/labels)
    h_ci, l_ci = [], []
    for sigma in sigma_levels:
        alpha = alphas[sigma]
        lower, upper = sigma_percentiles[sigma]
        lower_bound = np.percentile(samples_logit, lower, axis=0)
        upper_bound = np.percentile(samples_logit, upper, axis=0)
        # Create proxy for fill_between
        ci_label = f"±{confidence_levels[sigma]} CI"
        # Use blue color with the correct alpha for the proxy
        proxy = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=alpha, label=ci_label)
        h_ci.append(proxy)
        l_ci.append(ci_label)
        ax2.fill_between(
            x_grid,
            lower_bound,
            upper_bound,
            color="blue",
            alpha=alpha,
            label="_nolegend_",
            linewidth=0,
        )  # Ensure linewidth=0

    model_handles_ax2.extend(h_ci)
    model_labels_ax2.extend(l_ci)

    # --- Construct Final Legends ---
    # AX1 Legend (Prob space)
    train_proxies_h, train_proxies_l = create_scatter_legend_proxies(
        train_attack_marker_map, "Train Data", "gray"
    )
    test_proxies_h, test_proxies_l = create_scatter_legend_proxies(
        test_attack_marker_map, "Test Data", default_test_color
    )
    all_handles_ax1 = model_handles_ax1 + train_proxies_h + test_proxies_h
    all_labels_ax1 = model_labels_ax1 + train_proxies_l + test_proxies_l
    ax1.legend(
        handles=all_handles_ax1, labels=all_labels_ax1, frameon=False, fontsize="small"
    )

    # AX2 Legend (Logit space)
    train_proxies_h2, train_proxies_l2 = create_scatter_legend_proxies(
        train_attack_marker_map, "Train Data (logit)", "gray"
    )
    test_proxies_h2, test_proxies_l2 = create_scatter_legend_proxies(
        test_attack_marker_map, "Test Data (logit)", default_test_color
    )
    all_handles_ax2 = model_handles_ax2 + train_proxies_h2 + test_proxies_h2
    all_labels_ax2 = model_labels_ax2 + train_proxies_l2 + test_proxies_l2
    ax2.legend(
        handles=all_handles_ax2, labels=all_labels_ax2, frameon=False, fontsize="small"
    )

    # --- Format Plots ---
    ax1.set_xlabel("Capability Difference")
    ax1.set_ylabel("ASR")
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0, color="gray", ls="--", alpha=0.5)
    ax2.set_xlabel("Capability Difference")
    ax2.set_ylabel("logit(ASR)")
    ax2.set_xlim(xlim)
    ax2.grid(True, alpha=0.3)

    # --- Set overall title ---
    base_title = title
    if base_title is None:
        base_title = "Linear Regression with Logit Transform"

    # Add metrics to title
    metrics_str_parts = []
    if metrics:
        if "train" in metrics and metrics["train"]:
            train_met = metrics["train"]
            # R² metrics
            r2_str = f"R²_prob={train_met.get('r2_prob', float('nan')):.2f}, R²_logit={train_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Train: {r2_str}")
        elif "train_metrics" in metrics and metrics["train_metrics"]:
            train_met = metrics["train_metrics"]
            # R² metrics
            r2_str = f"R²_prob={train_met.get('r2_prob', float('nan')):.2f}, R²_logit={train_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Train: {r2_str}")

        # Add calibration metrics if available
        if "train_calibration" in metrics and metrics["train_calibration"]:
            calib_met = metrics["train_calibration"]
            # Look for miscovered point counts (common confidence levels)
            miscov_parts = []
            for level in ["95%", "90%", "68%"]:
                n_miscov_key = f"n_miscovered_{level}"
                n_points_key = f"n_points_{level}"
                if n_miscov_key in calib_met and n_points_key in calib_met:
                    n_miscov_raw = calib_met[n_miscov_key]
                    n_points_raw = calib_met[n_points_key]

                    # Check for NaN before converting to int
                    if (
                        not np.isnan(n_miscov_raw)
                        and not np.isnan(n_points_raw)
                        and n_points_raw > 0
                    ):
                        n_miscov = int(n_miscov_raw)
                        n_points = int(n_points_raw)
                        miscov_percentage = 100 * n_miscov / n_points

                        # Assert consistency: if n_miscov is 0, percentage must be 0
                        if n_miscov == 0:
                            assert miscov_percentage == 0.0, (
                                f"Bug: {n_miscov}/{n_points} should give 0% but got {miscov_percentage}%"
                            )

                        miscov_parts.append(
                            f"Miscov_{level}={n_miscov}/{n_points} ({miscov_percentage:.1f}%)"
                        )
            if miscov_parts:
                metrics_str_parts.append(f"Calibration: {', '.join(miscov_parts)}")

        if "test_metrics" in metrics and metrics["test_metrics"]:
            test_met = metrics["test_metrics"]
            # R² metrics
            r2_str = f"R²_prob={test_met.get('r2_prob', float('nan')):.2f}, R²_logit={test_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Test: {r2_str}")
    elif test_metrics:  # Allow passing only test metrics
        r2_str = f"R²_prob={test_metrics.get('r2_prob', float('nan')):.2f}, R²_logit={test_metrics.get('r2_logit', float('nan')):.2f}"
        metrics_str_parts.append(f"Test: {r2_str}")

    full_title = base_title
    if metrics_str_parts:
        full_title += "\n" + "; ".join(metrics_str_parts)

    fig.suptitle(full_title, fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to prevent title overlap

    return fig, (ax1, ax2)


def plot_model_comparison(
    x_grid,
    samples_list,
    method_names,
    x_data=None,
    y_data=None,
    xlim=(-1, 1),
    ylim=(0, 1),
    title="Comparison of Uncertainty Estimation Methods",
    colors=None,
    sigmas_to_plot=[1, 2],
):
    """
    Compare multiple uncertainty estimation methods.

    Args:
        x_grid: Array of x values where predictions were made
        samples_list: List of sample arrays from different methods
        method_names: List of method names
        x_data: Original data x values (optional)
        y_data: Original data y values (optional)
        xlim: x-axis limits
        ylim: y-axis limits
        title: Plot title
        colors: List of colors for different methods
        sigmas_to_plot: List of sigma levels to plot (default: [1, 2])

    Returns:
        fig: Matplotlib figure
        axes: Array of axes
    """
    n_methods = len(samples_list)
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, n_methods))

    fig, axes = plt.subplots(1, n_methods, figsize=(6 * n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for ax, samples, name, color in zip(axes, samples_list, method_names, colors):
        # Plot original data if provided
        if x_data is not None and y_data is not None:
            ax.scatter(x_data, y_data, color="black", alpha=0.7, label="Data")

        # Plot median prediction
        median_prob = np.median(samples, axis=0)
        ax.plot(x_grid, median_prob, color=color, lw=2, label="Median")

        # Plot uncertainty bands
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
            lower_bound = np.percentile(samples, lower, axis=0)
            upper_bound = np.percentile(samples, upper, axis=0)
            ax.fill_between(
                x_grid,
                lower_bound,
                upper_bound,
                color=color,
                alpha=alpha,
                label=f"±{confidence_levels[sigma]}",
            )

        # Format subplot
        ax.set_xlabel("Capability Difference")
        ax.set_ylabel("ASR")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(name)
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    return fig, axes


def plot_beta_binomial_model(
    x_grid,
    samples_prob,
    x_data=None,
    successes=None,
    trials=None,
    xlim=(-1, 1),
    ylim=(0, 1),
    title="Beta-Binomial Model Predictions",
    color="maroon",
    sigmas_to_plot=[1, 2],
):
    """
    Plot Beta-Binomial model predictions with uncertainty intervals.

    Args:
        x_grid: Array of x values for prediction curve
        samples_prob: Posterior samples of probabilities (from predict())
        x_data: Original data points x values (optional)
        successes: Number of successful trials (optional)
        trials: Number of trials (optional)
        xlim: x-axis limits
        ylim: y-axis limits
        title: Plot title
        color: Color for uncertainty bands and mean prediction
        sigmas_to_plot: List of sigma levels to plot (default: [1, 2])

    Returns:
        fig: Matplotlib figure
        ax: Matplotlib axis
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate mean prediction
    mean_prob = np.mean(samples_prob, axis=0)

    # Plot uncertainty intervals
    sigma_levels = sorted(sigmas_to_plot)  # Ensure ordered plotting
    sigma_percentiles = {
        1: (15.87, 84.13),  # ±1σ (68%)
        2: (2.28, 97.72),  # ±2σ (95%)
        3: (0.13, 99.87),  # ±3σ (99.7%)
    }
    confidence_levels = {1: "68%", 2: "95%", 3: "99.7%"}
    alphas = {1: 0.2, 2: 0.1, 3: 0.05}

    # Plot uncertainty bands
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
        )

    # Plot mean prediction
    ax.plot(x_grid, mean_prob, color=color, lw=2, label="Mean prediction")

    # Plot data points if provided
    if x_data is not None and successes is not None and trials is not None:
        proportions = successes / trials
        ax.scatter(x_data, proportions, color="black", alpha=0.7, label="Observed Data")

    # Format plot
    ax.set_xlabel("Capability Difference")
    ax.set_ylabel("ASR")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax


def plot_uncertainty_custom(
    x_grid,
    samples_prob,
    x_data=None,
    y_data=None,
    test_data=None,
    xlim=(-1, 1),
    ylim=(0, 1),
    title=None,
    method_name="Method",
    color="maroon",
    individual_means=None,
    metrics=None,
    save_path=None,
    sigmas_to_plot=[1, 2],
    train_legend_label="Training Data",
    test_legend_label="Test Data",
):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Increase font sizes for all elements
    plt.rcParams.update({"font.size": 14})

    # --- Format Title with Metrics (if provided) ---
    base_title = title
    if base_title is None:
        base_title = f"Uncertainty Estimation using {method_name}"

    metrics_str_parts = []
    if metrics is not None:
        if "train" in metrics and metrics["train"]:
            train_met = metrics["train"]
            # R² metrics
            r2_str = f"R²_prob={train_met.get('r2_prob', float('nan')):.2f}, R²_logit={train_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Train: {r2_str}")
        elif "train_metrics" in metrics and metrics["train_metrics"]:
            train_met = metrics["train_metrics"]
            # R² metrics
            r2_str = f"R²_prob={train_met.get('r2_prob', float('nan')):.2f}, R²_logit={train_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Train: {r2_str}")

        # Add calibration metrics if available
        if "train_calibration" in metrics and metrics["train_calibration"]:
            calib_met = metrics["train_calibration"]
            # Look for miscovered point counts (common confidence levels)
            miscov_parts = []
            for level in ["95%", "90%", "68%"]:
                n_miscov_key = f"n_miscovered_{level}"
                n_points_key = f"n_points_{level}"
                if n_miscov_key in calib_met and n_points_key in calib_met:
                    n_miscov_raw = calib_met[n_miscov_key]
                    n_points_raw = calib_met[n_points_key]

                    # Check for NaN before converting to int
                    if (
                        not np.isnan(n_miscov_raw)
                        and not np.isnan(n_points_raw)
                        and n_points_raw > 0
                    ):
                        n_miscov = int(n_miscov_raw)
                        n_points = int(n_points_raw)
                        miscov_percentage = 100 * n_miscov / n_points

                        # Assert consistency: if n_miscov is 0, percentage must be 0
                        if n_miscov == 0:
                            assert miscov_percentage == 0.0, (
                                f"Bug: {n_miscov}/{n_points} should give 0% but got {miscov_percentage}%"
                            )

                        miscov_parts.append(
                            f"Miscov_{level}={n_miscov}/{n_points} ({miscov_percentage:.1f}%)"
                        )
            if miscov_parts:
                metrics_str_parts.append(f"Calibration: {', '.join(miscov_parts)}")

        if "test_metrics" in metrics and metrics["test_metrics"]:
            test_met = metrics["test_metrics"]
            # R² metrics
            r2_str = f"R²_prob={test_met.get('r2_prob', float('nan')):.2f}, R²_logit={test_met.get('r2_logit', float('nan')):.2f}"
            metrics_str_parts.append(f"Test: {r2_str}")

    final_title = base_title
    if metrics_str_parts:
        final_title += "\n" + "; ".join(metrics_str_parts)

    ax.set_title(final_title, fontsize=16)  # Increased font size for title
    # --- End Title Formatting ---

    # Plot original data if provided
    if x_data is not None and y_data is not None:
        ax.scatter(
            x_data,
            y_data,
            color="gray",
            alpha=0.5,
            label=train_legend_label,
            linewidths=0.5,
        )

    # Plot individual means if provided
    if individual_means is not None:
        for i, ind_mean in enumerate(individual_means):
            label = "Individual Means" if i == 0 else None
            ax.plot(x_grid, ind_mean, color="gray", lw=0.8, alpha=0.5, label=label)

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
        # Use a distinct marker and color for test data
        ax.scatter(
            x_test,
            y_test,
            color="#FF6347",  # Tomato color, or choose another distinct one
            edgecolor="black",  # Add edge for visibility
            marker="X",  # Use 'X' marker
            s=80,  # Adjust size as needed
            label=test_legend_label,
            linewidths=1,
            alpha=0.9,
            zorder=10,  # Ensure it's plotted on top
        )

    # Format plot
    ax.set_xlabel("Capability Difference", fontsize=16)
    ax.set_ylabel("ASR", fontsize=16)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    # Increase tick label font sizes
    ax.tick_params(axis="both", which="major", labelsize=14)

    handles, labels = ax.get_legend_handles_labels()
    # Define the desired order of legend items
    desired_order = [
        train_legend_label,
        test_legend_label,
        "Median",
        "Individual Means",  # Keep this if used
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
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        fontsize=14,  # Increased legend font size
        loc="best",
    )

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    return fig, ax
