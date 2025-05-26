"""
Uncertainty estimation package for ASR prediction.

This package provides various methods for estimating uncertainty in ASR predictions:
- Linear regression in logit space
- Beta-binomial regression
- Bootstrap methods
"""

import evaluation

from .linear_grid_search import (
    load_metrics_mapping,
    logit_transform,
    prepare_augmented_data,
    prepare_data,
)
from .models.beta_binomial_model import fit_beta_binomial_model
from .models.beta_binomial_model import predict as predict_beta_binomial
from .models.beta_binomial_selection import (
    find_best_beta_binomial_model_for_target,
    grid_search_beta_binomial_priors,
)
from .models.linear_model import fit_linear_model
from .models.linear_model import predict as predict_linear
from .models.model_selection import find_best_model_for_target, grid_search_priors
from .plotting.plotting import (
    plot_linear_model_dual,
    plot_model_comparison,
    plot_uncertainty,
)

__all__ = [
    # Linear model
    "fit_linear_model",
    "predict_linear",
    "logit_transform",
    "grid_search_priors",
    "find_best_model_for_target",
    # Beta-binomial model
    "fit_beta_binomial_model",
    "predict_beta_binomial",
    "grid_search_beta_binomial_priors",
    "find_best_beta_binomial_model_for_target",
    # Data preparation and utilities
    "load_metrics_mapping",
    "prepare_data",
    "prepare_augmented_data",
]
