"""
Model selection using Optuna for hyperparameter optimization.
"""

import logging
import os
import shutil

import numpy as np
import optuna
import pandas as pd

# Reload modules to ensure the latest version is used
from .linear_model import fit_linear_model, predict

# Configure basic logging for Optuna (optional, but can be helpful)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def optuna_search_priors(
    x_data,
    y_data,
    num_optuna_trials: int = 100,
    n_samples_mcmc: int = 1000,
    tune_mcmc: int = 500,
    n_jobs: int = 1,  # Number of parallel jobs for Optuna
    verbose: bool = True,
    progressbar_pymc: bool = True,
):
    """
    Perform hyperparameter optimization using Optuna for a linear model.

    Args:
        x_data: Input data (capability differences)
        y_data: Target data (ASR values)
        num_optuna_trials: Number of different hyperparameter sets to try.
        n_samples_mcmc: Number of MCMC samples for each PyMC fit.
        tune_mcmc: Number of tuning steps for each PyMC fit.
        n_jobs: Number of parallel jobs for Optuna's study.optimize().
        verbose: Whether to print detailed progress (Optuna has its own progress bar).
        progressbar_pymc: Whether to display PyMC sampling progress bar.

    Returns:
        Dictionary with:
            - best_params: Dictionary with best hyperparameters.
            - best_model: Tuple (model, idata, y_logit) for the best model, re-fitted.
            - study: Optuna study object.
    """

    run_pid = os.getpid()
    optuna_run_base_compiledir = (
        f"/tmp/pytensor_optuna_run_{run_pid}_{np.random.randint(100000)}"
    )
    os.makedirs(optuna_run_base_compiledir, exist_ok=True)
    if verbose:
        print(f"Optuna base compiledir for this run: {optuna_run_base_compiledir}")

    def objective(trial: optuna.Trial):
        worker_pid = os.getpid()  # PID of the joblib worker process
        trial_compiledir_leaf = f"trial_{trial.number}_worker_{worker_pid}"
        trial_compiledir = os.path.join(
            optuna_run_base_compiledir, trial_compiledir_leaf
        )
        os.makedirs(trial_compiledir, exist_ok=True)

        # Crucial: Set PYTENSOR_FLAGS for this specific trial process
        pytensor_flags = f"compiledir={trial_compiledir},linker=py,cxx="
        os.environ["PYTENSOR_FLAGS"] = pytensor_flags

        prior_params_for_fit = {
            "sigma_w": trial.suggest_float("sigma_w", 0.01, 3.0),
            "sigma_b": trial.suggest_float("sigma_b", 0.01, 3.0),
            "sigma_sigma": trial.suggest_float("sigma_sigma", 0.01, 3.0),
        }

        try:
            model, idata, y_logit = fit_linear_model(
                x_data,
                y_data,
                prior_params=prior_params_for_fit,
                n_samples=n_samples_mcmc,
                tune=tune_mcmc,
                progressbar=False,  # For individual trials, keep progress bar off
            )
            log_likelihood = idata.log_likelihood["y_obs"].mean().item()
            if not np.isfinite(log_likelihood):
                log_likelihood = -np.inf  # Crucial for Optuna to handle failed fits

            # Calculate and store other metrics as user attributes
            pred_result = predict(idata, x_data, y_true=y_data, y_true_logit=y_logit)
            metrics = pred_result.get(
                "metrics", {}
            )  # Assuming predict returns a dict with a "metrics" key

            trial.set_user_attr(
                "log_likelihood",
                log_likelihood if np.isfinite(log_likelihood) else None,
            )  # Store original value
            trial.set_user_attr("rmse_prob", metrics.get("rmse_prob", np.nan))
            trial.set_user_attr("r2_prob", metrics.get("r2_prob", np.nan))
            trial.set_user_attr("rmse_logit", metrics.get("rmse_logit", np.nan))
            trial.set_user_attr("r2_logit", metrics.get("r2_logit", np.nan))

            return log_likelihood

        except Exception as e:
            if verbose:
                print(
                    f"Error in Optuna Trial {trial.number} with params {prior_params_for_fit}: {e}"
                )
            # Ensure errors in one trial don't stop the entire optimization.
            # Log the error and return a value indicating failure.
            trial.set_user_attr("error", str(e))
            return -np.inf  # Indicates a failed trial to Optuna

    # Use TPESampler, Optuna's default Bayesian optimization algorithm
    study = optuna.create_study(
        direction="maximize",
    )

    try:
        study.optimize(
            objective,
            n_trials=num_optuna_trials,
            n_jobs=n_jobs,
            show_progress_bar=True,  # Optuna's own progress bar
        )
    except KeyboardInterrupt:
        print("Optuna optimization interrupted by user.")
    except Exception as e:
        print(f"An exception occurred during Optuna optimization: {e}")

    if not study.trials:
        print("Optuna study completed with no trials.")
        return {"best_params": None, "best_model": None, "study": study}

    # Check if any trial completed successfully
    completed_trials = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value > -np.inf
    ]
    if not completed_trials:
        print(
            "Optuna study completed, but no trial was successful or best_trial is not available."
        )
        return {"best_params": None, "best_model": None, "study": study}

    best_trial = study.best_trial
    if (
        best_trial is None or best_trial.value == -np.inf
    ):  # Check if best_trial is valid
        print(
            "Optuna study finished, but no valid best trial found (all might have failed)."
        )
        # Try to find the best trial among successfully completed ones if study.best_trial is None
        if completed_trials:
            best_trial = max(
                completed_trials, key=lambda t: t.value
            )  # Manually find best if study.best_trial is None or bad
        else:  # No trials completed successfully.
            return {"best_params": None, "best_model": None, "study": study}

    best_params_from_optuna = best_trial.params

    if verbose:
        print(f"Best hyperparameters found by Optuna: {best_params_from_optuna}")
        print(f"Best log_likelihood: {best_trial.value}")

    # Re-fit the model with the best hyperparameters to get the actual model objects
    if verbose:
        print("Re-fitting model with best hyperparameters...")

    final_model_compiledir_leaf = f"final_model_worker_{os.getpid()}"
    final_model_compiledir = os.path.join(
        optuna_run_base_compiledir, final_model_compiledir_leaf
    )
    os.makedirs(final_model_compiledir, exist_ok=True)
    final_pytensor_flags = f"compiledir={final_model_compiledir},linker=py,cxx="
    os.environ["PYTENSOR_FLAGS"] = final_pytensor_flags
    if verbose:
        print(f"Re-fitting with PYTENSOR_FLAGS='{final_pytensor_flags}'")

    try:
        best_model_obj, best_idata, best_y_logit = fit_linear_model(
            x_data,
            y_data,
            prior_params=best_params_from_optuna,
            n_samples=n_samples_mcmc,
            tune=tune_mcmc,
            progressbar=progressbar_pymc,
        )
        best_model_tuple = (best_model_obj, best_idata, best_y_logit)
    except Exception as e:
        if verbose:
            print(f"Error re-fitting the best model: {e}")
        best_model_tuple = None

    # Clean up the base directory for this run's compiledirs
    try:
        if verbose:
            print(f"Cleaning up base compiledir: {optuna_run_base_compiledir}")
        shutil.rmtree(optuna_run_base_compiledir)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not clean up {optuna_run_base_compiledir}: {e}")

    return {
        "best_params": best_params_from_optuna,
        "best_model": best_model_tuple,
        "study": study,
    }


# Example of how it might be called (for context, not part of the file rewrite)
# if __name__ == '__main__':
#     # Create dummy data for testing
#     N = 100
#     rng = np.random.default_rng(42)
#     x_data_dummy = rng.uniform(-2, 2, N)
#     true_w = 0.5
#     true_b = 1.0
#     true_sigma = 0.5
#     y_logit_dummy = true_w * x_data_dummy + true_b
#     y_prob_dummy = 1 / (1 + np.exp(-y_logit_dummy))
#     # y_data_dummy = rng.binomial(1, y_prob_dummy) # If y_data is binary
#     y_data_dummy = y_prob_dummy # If y_data is probability (ASR)

#     print("Starting Optuna search example...")
#     results = optuna_search_priors(
#         x_data_dummy,
#         y_data_dummy,
#         num_optuna_trials=20, # Small number for quick test
#         n_samples_mcmc=100,   # Small number for quick test
#         tune_mcmc=50,       # Small number for quick test
#         n_jobs=2,             # Use 2 parallel jobs
#         verbose=True,
#         progressbar_pymc=False # Disable PyMC progress for cleaner logs during test
#     )

#     if results["best_params"]:
#         print("\nOptuna Search Results:")
#         print(f"Best Params: {results['best_params']}")
#         if results['best_model']:
#             model, idata, y_logit = results['best_model']
#             print(f"Log-likelihood of best model (re-fit): {idata.log_likelihood['y_obs'].mean().item()}")
#         else:
#             print("Best model could not be re-fitted.")
#     else:
#         print("Optuna search did not find any valid parameters.")

#     # To view Optuna dashboard (if you have it installed and a storage backend):
#     # optuna-dashboard sqlite:///optuna_study.db
#     # For in-memory studies, you can inspect study.trials_dataframe()
#     if results["study"]:
#         print("\nStudy Trials DataFrame:")
#         print(results["study"].trials_dataframe())
