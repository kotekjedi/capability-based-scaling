"""
Model implementations for uncertainty estimation.
"""

from .linear_model import fit_linear_model
from .linear_model import predict as predict_linear

__all__ = ["fit_linear_model", "predict_linear"]
