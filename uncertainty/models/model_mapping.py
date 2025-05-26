from typing import Dict, Type

from ..runner import UncertaintyModelRunner

# Import specific runner implementations
from ..runners.bootstrap_runner import UncertaintyBootstrapModelRunner

# Registry of available model runners
MODEL_RUNNERS: Dict[str, Type[UncertaintyModelRunner]] = {
    "bootstrap": UncertaintyBootstrapModelRunner,
    # Add more model types here as they are implemented
    # from ..runners.mcmc_runner import UncertaintyMCMCModelRunner
    # "mcmc": UncertaintyMCMCModelRunner,
}


def get_model_runner(model_type: str) -> Type[UncertaintyModelRunner]:
    """
    Get the model runner class for a given model type.

    Args:
        model_type: String identifier for the model type.

    Returns:
        The corresponding model runner class.

    Raises:
        KeyError: If the model type is not found in the registry.
    """
    if model_type not in MODEL_RUNNERS:
        raise KeyError(
            f"Unknown model type: {model_type}. "
            f"Available types are: {list(MODEL_RUNNERS.keys())}"
        )
    return MODEL_RUNNERS[model_type]
