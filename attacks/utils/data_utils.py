import json
from typing import Dict, List, Optional, Tuple

import pandas as pd

DEFAULT_BEHAVIORS_PATH = "data/harmbench_behaviors_all_no_copyright.csv"
DEFAULT_TARGETS_PATH = "data/targets.json"


def load_behaviors_from_csv(
    behavior_ids: Optional[List[str]] = None,
    row_numbers: Optional[List[int]] = None,
) -> List[Dict]:
    """Load multiple behaviors from default CSV file."""
    df = pd.read_csv(DEFAULT_BEHAVIORS_PATH)
    behaviors = []

    if behavior_ids is not None:
        for bid in behavior_ids:
            behavior_row = df[df["BehaviorID"] == bid]
            if len(behavior_row) == 0:
                raise ValueError(f"BehaviorID {bid} not found in behaviors file")
            row = behavior_row.index[0]
            behavior = behavior_row.iloc[0]
            context_str = behavior["ContextString"]
            if type(context_str) == float:
                context_str = None
            behaviors.append(
                {
                    "BehaviorID": behavior["BehaviorID"],
                    "Behavior": behavior["Behavior"],
                    "ContextString": context_str,
                    "row_number": row,
                }
            )
    elif row_numbers is not None:
        for row in row_numbers:
            behavior = df.iloc[row]
            context_str = behavior["ContextString"]
            if type(context_str) == float:
                context_str = None
            behaviors.append(
                {
                    "BehaviorID": behavior["BehaviorID"],
                    "Behavior": behavior["Behavior"],
                    "ContextString": context_str,
                    "row_number": row,
                }
            )
    else:
        # Default to first row if neither specified
        behavior = df.iloc[0]
        behaviors.append(
            {
                "BehaviorID": behavior["BehaviorID"],
                "Behavior": behavior["Behavior"],
                "ContextString": behavior["ContextString"]
                if "ContextString" in behavior
                else None,
            }
        )

    return behaviors


def load_targets_and_behaviors(
    behavior_configs: Optional[List[Dict]] = None,
    behavior_ids: Optional[List[str]] = None,
    row_numbers: Optional[List[int]] = None,
) -> Tuple[Dict, List[Dict]]:
    """Load targets and multiple behaviors configuration."""

    if behavior_configs is None:
        behavior_configs = load_behaviors_from_csv(behavior_ids, row_numbers)

    if behavior_configs and all("target" in config for config in behavior_configs):
        targets = {
            config["BehaviorID"]: config["target"] for config in behavior_configs
        }
    else:
        with open(DEFAULT_TARGETS_PATH) as f:
            targets = json.load(f)

    return targets, behavior_configs
