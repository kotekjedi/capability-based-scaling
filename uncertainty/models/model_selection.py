"""
Model selection and hyperparameter optimization utilities.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from .linear_model import fit_linear_model, predict


def grid_search_priors(
    x_data,
    y_data,
    hyperparams_grid=None,
    n_samples=1000,
    tune=500,
    verbose=True,
    plot_results=True,
    progressbar=True,
):
    """
    Perform grid search to find optimal prior parameters based on log-likelihood.

    Args:
        x_data: Input data (capability differences)
        y_data: Target data (ASR values)
        hyperparams_grid: Dictionary of hyperparameter lists to try, e.g.,
                          {'sigma_w': [0.1, 0.5, 1.0], 'sigma_b': [0.1, 0.5, 1.0]}
                          If None, defaults to standard grid for sigma_w, sigma_b, and sigma_sigma
        n_samples: Number of MCMC samples for each fit
        tune: Number of tuning steps for each fit
        verbose: Whether to print progress
        plot_results: Whether to plot the results
        progressbar: Whether to display PyMC sampling progress bar

    Returns:
        Dictionary with:
            - results_df: DataFrame with all results
            - best_params: Dictionary with best parameters
            - best_model: Tuple (model, idata, y_logit) for best parameters
    """
    # Initialize results storage
    results = []

    # Set default hyperparameter grid if none provided
    if hyperparams_grid is None:
        hyperparams_grid = {
            "sigma_w": [0.1, 0.5, 1.0, 2.0],
            "sigma_b": [0.1, 0.5, 1.0, 2.0],
            "sigma_sigma": [0.1, 0.5, 1.0, 2.0],
        }

    # Ensure x_data is properly shaped for the model
    # The model expects x_data to be 1D for batched calculations
    if isinstance(x_data, np.ndarray) and x_data.ndim > 1:
        if x_data.shape[1] == 1:  # If it's a column vector, flatten it
            x_data = x_data.flatten()

    # Calculate all possible hyperparameter combinations
    hyperparams_keys = list(hyperparams_grid.keys())
    hyperparams_values = [hyperparams_grid[key] for key in hyperparams_keys]

    # Generate all combinations
    import itertools

    all_combinations = list(itertools.product(*hyperparams_values))
    total_combinations = len(all_combinations)

    # Store best parameters and model
    best_log_likelihood = -np.inf
    best_params = None
    best_model_result = None

    # Grid search
    for i, combo in enumerate(tqdm(all_combinations, desc="Grid Search Progress")):
        current = i + 1

        # Create parameter dictionary for this combination
        prior_params = {key: value for key, value in zip(hyperparams_keys, combo)}

        if verbose:
            combo_str = ", ".join(
                [f"{key}={value}" for key, value in prior_params.items()]
            )
            print(f"Trying combination {current}/{total_combinations}: {combo_str}")
        # Fit model with these parameters
        try:
            model, idata, y_logit = fit_linear_model(
                x_data,
                y_data,
                prior_params=prior_params,
                n_samples=n_samples,
                tune=tune,
                progressbar=progressbar,
            )

            # Calculate log-likelihood from idata
            if hasattr(idata, "log_likelihood") and "y_obs" in idata.log_likelihood:
                log_likelihood = idata.log_likelihood["y_obs"].mean().item()
            else:
                if verbose:
                    print("  Log likelihood not found in InferenceData.")
                log_likelihood = -np.inf  # Assign a poor score

            # Calculate metrics using predict (which now accepts idata)
            # Ensure predict function in linear_model.py is updated to accept idata
            pred_result = predict(idata, x_data, y_true=y_data, y_true_logit=y_logit)
            metrics = pred_result.get("metrics", {})

            # Store results
            result = {**prior_params}  # Start with all parameters
            result.update(
                {
                    "log_likelihood": log_likelihood,
                    "rmse_prob": metrics["rmse_prob"],
                    "r2_prob": metrics["r2_prob"],
                    "rmse_logit": metrics["rmse_logit"],
                    "r2_logit": metrics["r2_logit"],
                }
            )
            results.append(result)

            # Check if this is the best model so far
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood
                best_params = prior_params.copy()
                # Store idata instead of trace
                best_model_result = (model, idata, y_logit)

                if verbose:
                    # Add logit metrics to the print statement
                    print(
                        f"  New best: log_likelihood={log_likelihood:.4f}, "
                        f"rmse_prob={metrics.get('rmse_prob', float('nan')):.4f}, "
                        f"r2_prob={metrics.get('r2_prob', float('nan')):.4f}, "
                        f"rmse_logit={metrics.get('rmse_logit', float('nan')):.4f}, "
                        f"r2_logit={metrics.get('r2_logit', float('nan')):.4f}"
                    )

        except Exception as e:
            if verbose:
                print(f"  Error with combination: {e}")

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)

    # Plot results if requested
    if plot_results and not results_df.empty:
        plot_grid_search_results(results_df, hyperparams_keys)

    # Return results
    return {
        "results_df": results_df,
        "best_params": best_params,
        "best_model": best_model_result,
    }


def plot_grid_search_results(results_df, param_keys=None):
    """
    Plot the grid search results.

    Args:
        results_df: DataFrame with grid search results
        param_keys: List of hyperparameter keys to plot
    """
    if results_df.empty:
        print("No results to plot")
        return

    # If param_keys not provided, use all numeric columns except metrics
    if param_keys is None:
        metrics_cols = [
            "log_likelihood",
            "rmse_prob",
            "r2_prob",
            "rmse_logit",
            "r2_logit",
        ]
        param_keys = [col for col in results_df.columns if col not in metrics_cols]

    # Create figure with subplots
    n_params = len(param_keys)
    n_plots = n_params + 1  # +1 for the heatmap
    fig_rows = (n_plots + 1) // 2
    fig, axes = plt.subplots(fig_rows, 2, figsize=(14, 5 * fig_rows))

    # If only one row, ensure axes is 2D
    if fig_rows == 1:
        axes = axes.reshape(1, 2)

    # Flatten axes array for easier indexing
    axes_flat = axes.flatten()

    # Plot log-likelihood vs each parameter
    for i, param in enumerate(param_keys):
        if i >= len(axes_flat):
            break

        # Group by all other parameters
        other_params = [p for p in param_keys if p != param]

        if other_params:
            for group, group_df in results_df.groupby(other_params):
                # Format group name
                if len(other_params) == 1:
                    group = [group]  # Ensure group is iterable

                group_name = "-".join([f"{p}={g}" for p, g in zip(other_params, group)])

                # Keep group name reasonably short
                if len(group_name) > 30:
                    group_name = group_name[:27] + "..."

                # Sort by parameter value for cleaner plots
                plot_df = group_df.sort_values(param)

                axes_flat[i].plot(
                    plot_df[param], plot_df["log_likelihood"], "o-", label=group_name
                )
        else:
            # If only one parameter, just plot it directly
            plot_df = results_df.sort_values(param)
            axes_flat[i].plot(plot_df[param], plot_df["log_likelihood"], "o-")

        axes_flat[i].set_xlabel(param)
        axes_flat[i].set_ylabel("Log-likelihood")
        axes_flat[i].set_title(f"Log-likelihood vs {param}")
        axes_flat[i].grid(True, alpha=0.3)

        # Add legend only if there are multiple groups and it's not too cluttered
        if other_params and len(results_df.groupby(other_params)) < 10:
            axes_flat[i].legend(fontsize="small")

    # If we have at least two parameters, create a heatmap for the first two
    if len(param_keys) >= 2 and n_plots < len(axes_flat):
        heatmap_idx = n_params  # Last plot is for heatmap

        param1 = param_keys[0]
        param2 = param_keys[1]

        # Get best model for each param1/param2 combination
        if len(param_keys) > 2:
            # Group by all other parameters and find best combination
            other_params = param_keys[2:]
            best_models = results_df.sort_values(
                "log_likelihood", ascending=False
            ).drop_duplicates([param1, param2])
        else:
            best_models = results_df

        # Find unique values (sorted) for the heatmap
        param1_unique = sorted(results_df[param1].unique())
        param2_unique = sorted(results_df[param2].unique())

        # Create a matrix for the heatmap
        heatmap_data = np.zeros((len(param1_unique), len(param2_unique)))

        # Fill the matrix with log-likelihood values
        for _, row in best_models.iterrows():
            i = param1_unique.index(row[param1])
            j = param2_unique.index(row[param2])
            heatmap_data[i, j] = row["log_likelihood"]

        # Plot the heatmap
        im = axes_flat[heatmap_idx].imshow(heatmap_data, cmap="viridis")

        # Set tick labels
        axes_flat[heatmap_idx].set_xticks(np.arange(len(param2_unique)))
        axes_flat[heatmap_idx].set_yticks(np.arange(len(param1_unique)))
        axes_flat[heatmap_idx].set_xticklabels(param2_unique)
        axes_flat[heatmap_idx].set_yticklabels(param1_unique)

        # Set labels and title
        axes_flat[heatmap_idx].set_xlabel(param2)
        axes_flat[heatmap_idx].set_ylabel(param1)
        axes_flat[heatmap_idx].set_title(
            f"Log-likelihood heatmap: {param1} vs {param2}"
        )

        # Add colorbar
        plt.colorbar(im, ax=axes_flat[heatmap_idx], label="Log-likelihood")

    # Hide any unused subplots
    for i in range(n_plots, len(axes_flat)):
        axes_flat[i].axis("off")

    # Adjust layout
    plt.tight_layout()
    plt.show()


def find_best_model_for_target(
    df,
    target_model_key,
    hyperparams_grid=None,
    capability_diff_col="capability_diff",
    n_samples=1000,
    tune=500,
    progressbar=True,
    use_optuna=False,
    n_jobs=1,
    num_hyperparam_configs: int = 100,
    model_type: str = "linear",
):
    """
    Find the best model for a given target model, either by grid search or Optuna/Ray Tune.

    Args:
        df: DataFrame with model data
        target_model_key: Key of the target model to optimize for
        hyperparams_grid: Dictionary of hyperparameter lists to try (for grid search)
        capability_diff_col: Name of the capability difference column
        n_samples: Number of MCMC samples for each fit (passed to ray_tune_search_priors as n_samples_mcmc)
        tune: Number of tuning steps for each fit (passed to ray_tune_search_priors as tune_mcmc)
        progressbar: Whether to display PyMC sampling progress bar
        use_optuna: Whether to use Ray Tune optimization instead of grid search
        n_jobs: Number of parallel jobs (used to initialize Ray if needed)
        num_hyperparam_configs: Number of hyperparameter configurations for Ray Tune to try.
        model_type: Type of model to optimize (either "linear" or "beta_binomial")

    Returns:
        Dictionary with optimization results
    """
    # Filter data for the target model
    model_df = df[df["target_model_key"] == target_model_key]

    if len(model_df) == 0:
        print(f"No data found for target model {target_model_key}")
        return None

    # Extract data
    x_data = model_df[capability_diff_col].values
    y_data = model_df["ASR"].values

    print(f"Optimizing {model_type} model for: {target_model_key}")
    print(f"Data points: {len(x_data)}")

    if use_optuna:
        print(
            f"Using {'Optuna' if model_type == 'linear' else 'Ray Tune'} optimization..."
        )
        # Dynamically import the correct search function
        if model_type == "linear":
            from .optuna_search import optuna_search_priors

            search_results = optuna_search_priors(
                x_data=x_data,
                y_data=y_data,
                num_optuna_trials=num_hyperparam_configs,
                n_samples_mcmc=n_samples,
                tune_mcmc=tune,
                n_jobs=n_jobs,
                verbose=progressbar,
                progressbar_pymc=progressbar,
            )
        elif model_type == "beta_binomial":
            # Assuming beta_binomial still uses a Ray Tune based search or a similar Optuna one
            # If beta_binomial also moves to a new optuna_search_beta_binomial, import that
            # For now, let's assume it might still use a ray_tune_search like function or needs its own optuna version
            # This part might need adjustment based on how beta_binomial_search is structured
            try:
                from .optuna_search_beta_binomial import (
                    optuna_search_beta_binomial_priors,
                )

                # or from .ray_tune_search_beta_binomial import ray_tune_search_beta_binomial_priors
                # Adjust this import and call as per the beta_binomial optimization function
                search_fn = optuna_search_beta_binomial_priors
                search_results = search_fn(
                    n_success=model_df["n_successful_harmful"],
                    n_trials=model_df["total_behaviors_attacker_target_pair"],
                    X_prior_params=x_data,
                    num_optuna_trials=num_hyperparam_configs,
                    n_samples_mcmc=n_samples,
                    tune_mcmc=tune,
                    n_jobs=n_jobs,
                    verbose=progressbar,
                    progressbar_pymc=progressbar,
                )
            except ImportError:
                print(
                    f"Warning: Beta binomial optimization function not found or not yet adapted for Optuna. Skipping {target_model_key} for beta_binomial."
                )
                return None
        else:
            raise ValueError(f"Unsupported model_type for Optuna: {model_type}")

        best_params = search_results.get("best_params")
        best_model_data = search_results.get("best_model")

        best_model, best_idata, best_y_logit = best_model_data
        pred_result = predict(
            best_idata, x_data, y_true=y_data, y_true_logit=best_y_logit
        )
        metrics = pred_result.get("metrics", {})

    else:
        # Use grid search if Ray Tune (formerly Optuna) is not requested
        print("Using Grid Search optimization...")
        results = grid_search_priors(
            x_data,
            y_data,
            hyperparams_grid=hyperparams_grid,
            n_samples=n_samples,
            tune=tune,
            progressbar=progressbar,
            verbose=True,  # Assuming verbose is desired
            plot_results=True,  # Assuming plots are desired
        )

        # Process results
        if results is None or results["best_model"] is None:
            print(
                f"Optimization ({'Optuna' if use_optuna else 'Grid Search'}) failed or returned no best model."
            )
            return None

        print("Best Parameters:", results["best_params"])

        # Get metrics for best model
        best_model, best_idata, best_y_logit = results["best_model"]
        pred_result = predict(
            best_idata, x_data, y_true=y_data, y_true_logit=best_y_logit
        )
        metrics = pred_result.get("metrics", {})
        print("Best Model Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        # Return best model details
        best_params = results["best_params"]
        best_model = results["best_model"][0]
        best_idata = results["best_model"][1]
        best_y_logit = results["best_model"][2]
        best_model_data = (best_model, best_idata, best_y_logit)

    # Return best model details
    return {
        "best_params": best_params,
        "best_model": best_model_data,
        "best_idata": best_idata,
        "best_y_logit": best_y_logit,
        "metrics": metrics,
    }
