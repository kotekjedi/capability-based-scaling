"""Configuration for evaluation metrics and their mapping to summary metrics."""

from dataclasses import dataclass
from typing import Dict, List

from utils.config import load_yaml


@dataclass
class MetricConfig:
    """Configuration for a single metric."""

    source_path: str  # Path to metric in results dict (e.g. "tinyMMLU_accuracy" or "IFEval.prompt_level_loose_acc")
    target_name: str  # Name in summary (will be appended with _target or _attacker)
    transform: str = "float"  # How to transform the value (e.g. "float", "int", "bool")


def load_metrics_config(
    config_path: str = "configs/metrics_config.yaml",
) -> Dict[str, List[MetricConfig]]:
    """Load metrics configuration from YAML file."""
    config = load_yaml(config_path)
    metrics_config = {}

    for dataset, metrics in config["metrics"].items():
        metrics_config[dataset] = [
            MetricConfig(
                source_path=metric["source_path"],
                target_name=metric["target_name"],
                transform=metric.get("transform", "float"),
            )
            for metric in metrics
        ]

    return metrics_config


def get_value_from_path(data: dict, path: str) -> any:
    """Get a value from a nested dictionary using dot notation path."""
    current = data
    for key in path.split("."):
        if key in current:
            current = current[key]
        else:
            raise KeyError(f"Key {key} not found in path {path}")
    return current


def transform_value(value: any, transform: str) -> any:
    """Transform a value according to the specified transform type."""
    if transform == "float":
        return float(value)
    elif transform == "int":
        return int(value)
    elif transform == "bool":
        return bool(value)
    else:
        raise ValueError(f"Unknown transform type: {transform}")


def get_summary_metrics(
    results: dict,
    is_attacker: bool = False,
) -> dict:
    """Extract summary metrics from detailed results.

    Args:
        results: Dictionary containing evaluation results
        is_attacker: Whether these results are from the attacker (adapter) model
    """
    summary = {}
    model_type = "attacker" if is_attacker else "target"
    results = results["results"]
    # Process each dataset's results

    for dataset in results:
        if dataset == "harmbench":
            # Special handling for harmbench
            if "overall_asr" in results[dataset]:
                summary[f"harmbench_asr_{model_type}"] = float(
                    results[dataset]["overall_asr"]
                )
            continue

        # For other datasets, look for results/metrics in standard format
        if dataset in results:
            metrics = results[dataset]
            if "exact_match,none" in metrics:
                summary[f"{dataset}_exact_match_none_{model_type}"] = float(
                    metrics["exact_match,none"]
                )
            # Extract the accuracy metric (acc_norm,none is the standard metric)
            if "acc_norm,none" in metrics:
                summary[f"{dataset}_accuracy_{model_type}"] = float(
                    metrics["acc_norm,none"]
                )
            if "exact_match,strict-match" in metrics:
                summary[f"{dataset}_exact_match_strict_{model_type}"] = float(
                    metrics["exact_match,strict-match"]
                )

            if "exact_match,flexible-extract" in metrics:
                summary[f"{dataset}_exact_match_flexible_{model_type}"] = float(
                    metrics["exact_match,flexible-extract"]
                )

            if "prompt_level_loose_acc,none" in results[dataset]:
                summary[f"IFEval_prompt_loose_{model_type}"] = float(
                    results[dataset]["prompt_level_loose_acc,none"]
                )
            if "inst_level_loose_acc,none" in results[dataset]:
                summary[f"IFEval_inst_loose_{model_type}"] = float(
                    results[dataset]["inst_level_loose_acc,none"]
                )
            if "exact_match,custom-extract" in metrics:
                summary[f"{dataset}_exact_match_custom_{model_type}"] = float(
                    metrics["exact_match,custom-extract"]
                )

    return summary
