"""
Linear regression model with logit transform for ASR prediction.

This module implements a Bayesian linear regression model in logit space for predicting
Attack Success Rate (ASR) given capability differences.

The model is specified as:
    logit(y) ~ N(w * x + b, σ)

with priors:
    w ~ HalfNormal(σ_w)      # non-negative slope
    b ~ N(0, σ_b²)           # symmetric uncertainty for intercept
    σ ~ HalfNormal(σ_σ)      # positive observation noise
"""

import os

import arviz as az
import numpy as np
import pymc as pm
import pytensor
from scipy.special import expit, logit


def logit_transform(p, eps=1e-2):
    """
    Custom logit transformation with clipping for numerical stability.

    Args:
        p: Input probabilities to transform
        eps: Small value for clipping to avoid infinities

    Returns:
        Logit transformed values
    """
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_linear_model(
    x_data,
    y_data,
    prior_params=None,
    n_samples=2000,
    tune=1000,
    target_accept=0.9,
    random_seed=42,
    progressbar=True,
):
    """
    Fit Bayesian linear regression model in logit space using PyMC.

    Args:
        x_data: Input features (capability differences)
        y_data: Target values (ASR)
        prior_params: Dictionary with prior parameters:
                     - 'sigma_w': Scale for w prior
                     - 'sigma_b': Scale for b prior
                     - 'sigma_sigma': Scale for σ prior
        n_samples: Number of posterior samples
        tune: Number of tuning steps
        target_accept: Target acceptance rate for NUTS
        random_seed: Random seed for reproducibility
        progressbar: Whether to display a progress bar during sampling

    Returns:
        model: PyMC model
        idata: ArviZ InferenceData object containing posterior samples and log likelihood
        y_logit: Logit transformed y values (returned for convenience)
    """
    # compiledir configuration is now handled by PYTENSOR_FLAGS in the Optuna objective function

    # Set default prior parameters if none provided
    if prior_params is None:
        prior_params = {"sigma_w": 0.5, "sigma_b": 0.5, "sigma_sigma": 0.5}

    # Transform y to logit space
    y_logit = logit_transform(y_data)

    # The with pytensor.config.change_flags(compiledir=temp_compiledir): block will be removed.
    with pm.Model() as model:
        # Define Coordinates for Arviz
        coords = {"obs_id": np.arange(len(x_data))}
        model.add_coord("obs_id", coords["obs_id"])

        # Prior for slope (non-negative)
        w = pm.HalfNormal("w", sigma=prior_params["sigma_w"])

        # Prior for intercept
        b = pm.Normal("b", mu=0, sigma=prior_params["sigma_b"])

        # Prior for observation noise
        sigma = pm.HalfNormal("sigma", sigma=prior_params["sigma_sigma"])

        # Mean prediction
        mu = pm.Deterministic("mu", w * x_data + b, dims="obs_id")

        # Likelihood
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y_logit, dims="obs_id")

        # Sample from posterior
        idata = pm.sample(
            n_samples,
            tune=tune,
            target_accept=target_accept,
            random_seed=random_seed,
            return_inferencedata=True,  # Return InferenceData
            progressbar=progressbar,
        )

        # Add prior and posterior predictive samples to idata
        idata.extend(pm.sample_prior_predictive(model=model, random_seed=random_seed))
        idata.extend(
            pm.sample_posterior_predictive(idata, model=model, random_seed=random_seed)
        )

        # Compute log likelihood (now automatically includes observed data mapping)
        pm.compute_log_likelihood(idata, model=model, progressbar=progressbar)

    return model, idata, y_logit


def predict(
    idata,
    x_new,
    y_true=None,  # Training y (prob space)
    y_true_logit=None,  # Training y (logit space)
    x_test=None,  # Optional test features
    y_test=None,  # Optional test y (prob space)
    y_test_logit=None,  # Optional test y (logit space)
    return_logit=False,
    n_samples=None,
    random_seed=43,  # Use different seed for prediction sampling
):
    """
    Generate predictions from the fitted model for new and optionally test data.

    Args:
        idata: ArviZ InferenceData object from fit_linear_model
        x_new: New input points for generating prediction curves/surfaces
        y_true: True training y values (prob space) for computing training metrics (optional)
        y_true_logit: True training y values (logit space) for training metrics (optional)
        x_test: Test set features (optional)
        y_test: True test y values (prob space) for test metrics (optional)
        y_test_logit: True test y values (logit space) for test metrics (optional)
        return_logit: Whether to return predictions for x_new in logit space
        n_samples: Number of posterior samples to use (if None, use all available)
        random_seed: Random seed for sampling predictive distribution

    Returns:
        Dictionary containing predictions for x_new and optionally metrics for train/test sets.
    """
    x_new = np.atleast_1d(x_new)
    posterior = idata.posterior  # Access posterior samples
    rng = np.random.default_rng(random_seed)  # Random number generator

    # Determine number of samples to use
    n_total_samples = posterior.dims["chain"] * posterior.dims["draw"]
    if n_samples is not None and n_samples < n_total_samples:
        # Subsample chains and draws proportionally if possible, otherwise flat sample
        # For simplicity, using flat sampling here
        sample_indices = rng.choice(n_total_samples, n_samples, replace=False)
        w_samples = posterior["w"].values.flatten()[sample_indices]
        b_samples = posterior["b"].values.flatten()[sample_indices]
        sigma_samples = posterior["sigma"].values.flatten()[sample_indices]
        n_used_samples = n_samples
    else:
        # Use all samples, flattened
        w_samples = posterior["w"].values.flatten()
        b_samples = posterior["b"].values.flatten()
        sigma_samples = posterior["sigma"].values.flatten()
        n_used_samples = n_total_samples

    # --- Predictions for x_new ---
    # Compute predictions in logit space
    # Shape: (n_used_samples, n_x_new)
    pred_means_logit_new = w_samples[:, None] * x_new[None, :] + b_samples[:, None]
    sigma_samples_expanded_new = sigma_samples[:, None] * np.ones_like(
        pred_means_logit_new
    )
    pred_samples_logit_new = rng.normal(
        loc=pred_means_logit_new, scale=sigma_samples_expanded_new
    )

    # Compute summary statistics in logit space for x_new
    mean_logit_new = np.mean(pred_means_logit_new, axis=0)
    std_logit_new = np.std(pred_means_logit_new, axis=0)  # Std of the means (epistemic)

    # Compute credible intervals using pred_samples_logit_new (includes observation noise)
    sigma_levels = [1, 2, 3]
    sigma_percentiles = {1: (15.87, 84.13), 2: (2.28, 97.72), 3: (0.13, 99.87)}

    intervals_logit_new = {}
    for sigma in sigma_levels:
        lower, upper = sigma_percentiles[sigma]
        intervals_logit_new[f"{sigma}sigma"] = (
            np.percentile(pred_samples_logit_new, lower, axis=0),
            np.percentile(pred_samples_logit_new, upper, axis=0),
        )

    # Transform predictions for x_new to probability space
    samples_prob_new = expit(pred_samples_logit_new)
    mean_prob_new = np.mean(samples_prob_new, axis=0)

    intervals_prob_new = {}
    for sigma in sigma_levels:
        lower, upper = sigma_percentiles[sigma]
        intervals_prob_new[f"{sigma}sigma"] = (
            np.percentile(samples_prob_new, lower, axis=0),
            np.percentile(samples_prob_new, upper, axis=0),
        )

    # --- Compute Metrics ---
    train_metrics = {}
    test_metrics = {}

    # Helper function for metrics
    def calculate_metrics(y_true_prob, y_pred_prob, y_true_logit, y_pred_logit):
        metrics = {}
        # Metrics in probability space
        rmse_prob = np.sqrt(np.mean((y_true_prob - y_pred_prob) ** 2))
        ss_total_prob = np.sum((y_true_prob - np.mean(y_true_prob)) ** 2)
        ss_residual_prob = np.sum((y_true_prob - y_pred_prob) ** 2)
        r2_prob = 1 - (ss_residual_prob / ss_total_prob) if ss_total_prob != 0 else 0
        metrics.update({"rmse_prob": rmse_prob, "r2_prob": r2_prob})

        # Metrics in logit space
        rmse_logit = np.sqrt(np.mean((y_true_logit - y_pred_logit) ** 2))
        ss_total_logit = np.sum((y_true_logit - np.mean(y_true_logit)) ** 2)
        ss_residual_logit = np.sum((y_true_logit - y_pred_logit) ** 2)
        r2_logit = (
            1 - (ss_residual_logit / ss_total_logit) if ss_total_logit != 0 else 0
        )
        metrics.update({"rmse_logit": rmse_logit, "r2_logit": r2_logit})

        return metrics

    # Calculate Training Metrics if training data provided
    train_predictions_dict = {}
    if y_true is not None and y_true_logit is not None:
        print(
            "  DEBUG (predict): Checking for posterior_predictive in idata..."
        )  # DEBUG
        # Need predictions for the training data points (x_data is implicit via idata)
        # We can get the posterior predictive mean for the observed data from idata
        if (
            hasattr(idata, "posterior_predictive")
            and "y_obs" in idata.posterior_predictive
        ):
            print(
                "  DEBUG (predict): Found posterior_predictive['y_obs']. Calculating metrics..."
            )  # DEBUG
            # Posterior predictive samples for training data (logit space)
            post_pred_train_logit = idata.posterior_predictive["y_obs"].values.reshape(
                -1, len(y_true_logit)
            )  # Flatten chains/draws
            # Mean prediction in logit space for training data
            y_pred_train_logit_mean = post_pred_train_logit.mean(axis=0)
            # Mean prediction in probability space for training data
            post_pred_train_prob = expit(post_pred_train_logit)
            y_pred_train_prob_mean = post_pred_train_prob.mean(axis=0)

            train_metrics = calculate_metrics(
                y_true, y_pred_train_prob_mean, y_true_logit, y_pred_train_logit_mean
            )
            print(
                f"  DEBUG (predict): Calculated train_metrics: {train_metrics}"
            )  # DEBUG

            # Extract mean log likelihood for training data if available
            if hasattr(idata, "log_likelihood") and "y_obs" in idata.log_likelihood:
                mean_log_likelihood_train = idata.log_likelihood["y_obs"].mean().item()
                train_metrics["log_likelihood"] = mean_log_likelihood_train
                print(
                    f"  DEBUG (predict): Added log_likelihood to train_metrics: {train_metrics}"
                )  # DEBUG

            # Store training prediction samples
            train_predictions_dict = {
                # "x_train": x_data, # x_data isn't explicitly passed, only implied by idata length
                "samples_prob": post_pred_train_prob,
                "samples_logit": post_pred_train_logit,
                "mean_prob": y_pred_train_prob_mean,
                "mean_logit": y_pred_train_logit_mean,
            }
        else:
            print(
                "  DEBUG (predict): Posterior predictive samples ('y_obs') not found in idata. Cannot calculate training metrics precisely or provide train samples."
            )  # DEBUG
            print(
                f"  DEBUG (predict): idata keys: {list(idata.keys()) if hasattr(idata, 'keys') else 'N/A'}"
            )  # DEBUG
            if hasattr(idata, "posterior_predictive"):
                print(
                    f"  DEBUG (predict): posterior_predictive keys: {list(idata.posterior_predictive.keys())}"
                )  # DEBUG
            else:
                print(
                    "  DEBUG (predict): idata has no posterior_predictive group."
                )  # DEBUG
            # Could approximate using w/b means, but less accurate
            # mu_train_mean = np.mean(w_samples) * x_data + np.mean(b_samples) # Requires x_data

    # Calculate Test Metrics if test data provided
    if x_test is not None and y_test is not None and y_test_logit is not None:
        x_test = np.atleast_1d(x_test)
        # Compute predictions for test data
        pred_means_logit_test = (
            w_samples[:, None] * x_test[None, :] + b_samples[:, None]
        )
        sigma_samples_expanded_test = sigma_samples[:, None] * np.ones_like(
            pred_means_logit_test
        )
        # Predictive samples for test data (logit space)
        pred_samples_logit_test = rng.normal(
            loc=pred_means_logit_test, scale=sigma_samples_expanded_test
        )
        # Mean prediction in logit space for test data
        y_pred_test_logit_mean = np.mean(pred_means_logit_test, axis=0)

        # Transform to probability space for test data
        pred_samples_prob_test = expit(pred_samples_logit_test)
        # Mean prediction in probability space for test data
        y_pred_test_prob_mean = np.mean(pred_samples_prob_test, axis=0)

        test_metrics = calculate_metrics(
            y_test, y_pred_test_prob_mean, y_test_logit, y_pred_test_logit_mean
        )

        # TODO: Add Test Log Likelihood calculation if needed (requires scipy.stats.norm.logpdf)
        # from scipy.stats import norm
        # log_lik_test = norm.logpdf(y_test_logit[None, :], loc=pred_means_logit_test, scale=sigma_samples_expanded_test)
        # test_metrics["log_likelihood"] = np.mean(np.sum(log_lik_test, axis=1)) # Mean over samples

    # --- Prepare Return Dictionary ---
    result = {
        "x_new": x_new,
        # Predictions for x_new
        "mean_prob": mean_prob_new,
        "mean_logit": mean_logit_new,
        "std_logit": std_logit_new,  # Epistemic uncertainty in logit space
        "samples_prob": samples_prob_new,
        "samples_logit": pred_samples_logit_new,
        "intervals_prob": intervals_prob_new,
        "intervals_logit": intervals_logit_new,
        # Metrics & Predictions
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "train_predictions": train_predictions_dict,  # Add train predictions
        "test_predictions": {},
    }

    if x_test is not None:
        result["test_predictions"] = {
            "x_test": x_test,
            "mean_prob": y_pred_test_prob_mean,  # Mean prediction (prob)
            "mean_logit": y_pred_test_logit_mean,  # Mean prediction (logit)
            "samples_prob": pred_samples_prob_test,  # Full samples (prob)
            "samples_logit": pred_samples_logit_test,  # Full samples (logit)
        }

    # Adjust output based on return_logit flag (only affects x_new predictions)
    if return_logit:
        result_logit = {
            "x_new": x_new,
            "mean": mean_logit_new,
            "std": std_logit_new,
            "samples": pred_samples_logit_new,
            "intervals": intervals_logit_new,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "test_predictions": result[
                "test_predictions"
            ],  # Keep test predictions consistent
        }
        return result_logit
    else:
        # Default return uses probability space for primary x_new results
        print(
            f"  DEBUG (predict): Final train_metrics being returned: {result.get('train_metrics')}"
        )  # DEBUG
        # Filter out empty dictionaries
        result = {
            k: v for k, v in result.items() if not (isinstance(v, dict) and not v)
        }
        return result
