import argparse
import logging
from typing import Any, Dict, List

import yaml

logger = logging.getLogger(__name__)


def load_yaml(file_path: str) -> Dict[str, Any]:
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate LLM model")

    # Make the parser more general - accept any argument as a string
    parser.add_argument(
        "--config_dir", type=str, default="config", help="Configuration directory"
    )
    parser.add_argument("--model_name", type=str, help="Model name from config")
    parser.add_argument("--training_backend", type=str, help="Training backend to use")
    parser.add_argument(
        "--temp_saving_path",
        type=str,
        default="",
        help="Path to save temporary files",
    )
    parser.add_argument(
        "--datasets_in_use",
        type=str,
        nargs="+",
        help="Space-separated list of datasets to use (e.g., 'shadow_alignment badllama')",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        nargs="+",
        help="Space-separated list of evaluations to run (e.g., 'harmbench tinyMMLU IFEval')",
    )

    # Add a special argument to capture all remaining arguments
    parser.add_argument(
        "--kwargs", nargs=argparse.REMAINDER, help="Additional key=value pairs"
    )

    args, unknown = parser.parse_known_args()

    unknown_args = {}
    i = 0
    while i < len(unknown):
        arg = unknown[i]
        if arg.startswith("--"):
            key = arg.lstrip("-")
            # If there's a value and it doesn't look like a flag, assign it as the value
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                unknown_args[key] = unknown[i + 1]
                i += 2
            else:
                # Flag without a value (e.g., --flag)
                unknown_args[key] = True
                i += 1
        else:
            i += 1

    # Merge known and unknown arguments
    combined_args = {**vars(args), **unknown_args}

    # Create AttrDict for dot notation access
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    # Convert args to dictionary and wrap in AttrDict
    arg_dict = AttrDict(combined_args)

    return arg_dict


def update_config(config: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """Update config recursively with command line arguments if they match any config key."""

    def update_nested_dict(d: Dict[str, Any], updates: Dict[str, Any]):
        for key, value in d.items():
            # If this is a nested dictionary, recurse
            if isinstance(value, dict):
                update_nested_dict(value, updates)
            # If the key exists in updates, update the value
            if key in updates and updates[key] is not None:
                try:
                    if isinstance(d[key], bool):
                        d[key] = str(updates[key]).lower() == "true"
                    elif isinstance(d[key], (int, float)):
                        d[key] = type(d[key])(updates[key])
                    else:
                        d[key] = updates[key]
                    logger.info(f"Updated value: {key} = {d[key]}")
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Warning: Failed to convert value for {key}: {e}")

    # Make a copy to avoid modifying the original
    config_copy = config.copy()
    update_nested_dict(config_copy, args)

    # Handle special cases
    # If datasets_in_use is provided, ensure it's properly formatted
    if "data" in config_copy and "datasets_in_use" in args and args["datasets_in_use"]:
        if isinstance(args["datasets_in_use"], list):
            config_copy["data"]["datasets_in_use"] = " ".join(args["datasets_in_use"])
        else:
            config_copy["data"]["datasets_in_use"] = args["datasets_in_use"]

    # If eval_type is provided, ensure it's properly formatted
    if "evaluation" in config_copy and "eval_type" in args and args["eval_type"]:
        if isinstance(args["eval_type"], list):
            config_copy["evaluation"]["eval_type"] = " ".join(args["eval_type"])
        else:
            config_copy["evaluation"]["eval_type"] = args["eval_type"]

    # if lora_target is in config_copy, change " " to ","
    if "lora" in config_copy:
        config_copy["lora"]["lora_target"] = config_copy["lora"]["lora_target"].replace(
            " ", ","
        )
    # if learning_rate is in config_copy, cast from scientific notation to float
    if "training" in config_copy and "learning_rate" in config_copy["training"]:
        config_copy["training"]["learning_rate"] = float(
            config_copy["training"]["learning_rate"]
        )
    return config_copy
