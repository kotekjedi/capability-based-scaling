#!/usr/bin/env python3
import argparse
import json
import logging
import os
from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray

from .models.beta_binomial_selection import (
    find_best_beta_binomial_model_for_target,
)
from .models.model_selection import find_best_model_for_target


def load_metrics_mapping(eval_dir="../evaluation_results"):
    """Load metrics from summary.json files."""
    metrics_mapping = {}
    for model_folder in os.listdir(eval_dir):
        folder_path = os.path.join(eval_dir, model_folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith("summary.json"):
                    file_path = os.path.join(folder_path, file)
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    metrics_mapping[model_folder] = data
    return metrics_mapping


def prepare_data(data, mmlu_mapping=None, delimiter="\t"):
    """
    Prepare data for plotting by cleaning and standardizing column names,
    converting numeric columns, and calculating capability differences.

    Args:
        data: String data or path to a CSV file
        mmlu_mapping: Dictionary mapping model keys to their tinyMMLU scores
        delimiter: Delimiter used in the data file (default: tab)

    Returns:
        DataFrame with cleaned and processed data
    """
    # Read the data
    if isinstance(data, str):
        if data.strip().startswith(("target_model", "target model")):
            # Data is a string containing the actual data
            df = pd.read_csv(StringIO(data), sep=delimiter, skip_blank_lines=True)
        else:
            # Data is a file path
            df = pd.read_csv(data, sep=delimiter, skip_blank_lines=True)
    else:
        # Data is already a DataFrame
        df = data.copy()

    # Clean and standardize column names
    df.columns = df.columns.str.strip()

    # Ensure we have the expected columns or rename them
    expected_columns = [
        "target_model_key",
        "attacker_model_key",
        "ASR",
        "judge_correlation",
        "total_behaviors",
    ]
    if len(df.columns) >= len(expected_columns):
        df.columns = expected_columns + list(df.columns[len(expected_columns) :])

    # Remove extra spaces from string columns
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()

    # Add tinyMMLU scores if mapping is provided
    if mmlu_mapping is not None:
        for key in mmlu_mapping.keys():
            for val in mmlu_mapping[key].keys():
                if "target" in val:
                    df.loc[df["target_model_key"] == key, val] = mmlu_mapping[key][val]
                elif "attacker" in val:
                    df.loc[df["attacker_model_key"] == key, val] = mmlu_mapping[key][
                        val
                    ]

    # Convert numeric columns (errors='coerce' will convert missing or invalid values to NaN)
    numeric_columns = ["ASR", "judge_correlation"]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Create label column for plotting
    df["label"] = df["attacker_model_key"] + "→" + df["target_model_key"]

    # filter thos that have less than 20 behs
    df = df[df["total_behaviors"] >= 20]

    return df


def logit_transform(p, eps=1e-2):
    """
    Transform probabilities to logit space with handling of edge cases.

    Args:
        p: Input probabilities
        eps: Small value for numerical stability

    Returns:
        Logit transformed values
    """
    p = np.array(p).astype(float)
    p = np.clip(p, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    return logit_p


def prepare_augmented_data(metrics_mapping, type="log"):
    """Prepare and augment the data with dummy attacker data."""
    # Prepare dataframes using the provided prepare_data function
    pair_sys_df = prepare_data("../pair-sys.csv", metrics_mapping)
    pair_sys_df_api = prepare_data("../pair-sys-api.csv", metrics_mapping)
    pair_sys_df = pd.concat([pair_sys_df, pair_sys_df_api], ignore_index=True)
    crescendo_df = prepare_data("../crescendo-sys.csv", metrics_mapping)
    augmented_df = pd.concat([pair_sys_df, crescendo_df], ignore_index=True)

    # keep attacker target pairs with the highest ASR
    augmented_df = augmented_df.sort_values(by="ASR", ascending=False)
    augmented_df = augmented_df.drop_duplicates(
        subset=["attacker_model_key", "target_model_key"]
    )

    # Extract unique target models
    target_models = augmented_df[
        ["target_model_key", "mmlu_pro_exact_match_custom_target"]
    ].drop_duplicates()

    # Create dummy attacker data
    dummy_attacker = pd.DataFrame(
        {
            "target_model_key": target_models["target_model_key"],
            "mmlu_pro_exact_match_custom_target": target_models[
                "mmlu_pro_exact_match_custom_target"
            ],
            "mmlu_pro_exact_match_custom_attacker": 0.11086804236321281,
            "ASR": 0,
            "attacker_model_key": "Direct Query",
            "total_behaviors": 50,
        }
    )

    # Mapping for the dummy ASR values
    asr_dummy_mapping = {
        "qwen2.5-0.5b": 0.38,
        "qwen2.5-1.5b": 0.06,
        "qwen2.5-3b": 0.1,
        "qwen2.5-7b": 0.18,
        "qwen2.5-14b": 0.1,
        "qwen2.5-32b": 0.04,
        "qwen2.5-72b": 0.1,
        "llama3_2_1b": 0.06,
        "llama3_2_3b": 0.04,
        "llama3_8b": 0,
        "llama3_1_8b": 0.04,
        "llama3_1_70b": 0.02,
        "llama3_3_70b": 0.02,
        "llama2_7b": 0,
        "llama2_13b": 0,
        "llama2_70b": 0,
        "vicuna_7b": 0.16,
        "vicuna_13b": 0.02,
        "mistral_7b_v0_2": 0.24,
        "mistral_small_24b": 0.04,
        "mixtral_8_7b": 0.04,
    }
    dummy_attacker["ASR"] = dummy_attacker["target_model_key"].apply(
        lambda x: asr_dummy_mapping.get(x, 0.00)
    )

    # Augment the original system DataFrame
    augmented_df = pd.concat([augmented_df, dummy_attacker], ignore_index=True)

    if type == "log":
        # Compute capability difference
        augmented_df["capability_diff"] = np.log(
            (augmented_df["mmlu_pro_exact_match_custom_attacker"])
            / (augmented_df["mmlu_pro_exact_match_custom_target"])
        )
    elif type == "log-err":
        augmented_df["capability_diff"] = np.log(
            (1 - augmented_df["mmlu_pro_exact_match_custom_target"])
            / (1 - augmented_df["mmlu_pro_exact_match_custom_attacker"])
        )
    elif type == "abs":
        augmented_df["capability_diff"] = (
            augmented_df["mmlu_pro_exact_match_custom_attacker"]
            - augmented_df["mmlu_pro_exact_match_custom_target"]
        )
    elif type == "logit":
        augmented_df["capability_diff"] = logit_transform(
            augmented_df["mmlu_pro_exact_match_custom_attacker"]
        ) - logit_transform(augmented_df["mmlu_pro_exact_match_custom_target"])

    return augmented_df, target_models


def save_current_results(results, output_file):
    """Save the current results to a file."""
    results_summary = {model: res["best_params"] for model, res in results.items()}
    with open(output_file, "w") as f:
        json.dump(results_summary, f, indent=4)


def main():
    parser = argparse.ArgumentParser(
        description="Run model selection with grid search or Optuna"
    )
    parser.add_argument(
        "--use-optuna",
        action="store_true",
        help="Use Optuna for optimization instead of grid search",
    )
    parser.add_argument(
        "--eval-dir",
        default="../evaluation_results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--type",
        choices=["log", "log-err", "abs", "logit"],
        default="log",
        help="Type of difference to use",
    )
    parser.add_argument(
        "--output", default="grid_search_results.json", help="Output file for results"
    )
    parser.add_argument(
        "--n-samples", type=int, default=1000, help="Number of MCMC samples"
    )
    parser.add_argument("--tune", type=int, default=500, help="Number of tuning steps")
    parser.add_argument(
        "--progressbar",
        action="store_true",
        help="Show progress bar during optimization",
    )
    parser.add_argument(
        "--model-type",
        choices=["linear", "beta_binomial"],
        default="linear",
        help="Type of model to fit (linear or beta_binomial)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for Optuna (-1 uses all cores)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging for Ray",
    )
    args = parser.parse_args()

    # Initialize Ray if using Ray Tune (formerly use_optuna)
    if args.use_optuna:  # This flag now means "use_ray_tune"
        num_ray_cpus = args.n_jobs
        if num_ray_cpus == -1:
            num_ray_cpus = os.cpu_count()  # Use all available if -1
    # Define hyperparameter grid (used only for grid search)
    if args.model_type == "linear":
        hyperparams_grid = {
            "sigma_w": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "sigma_b": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            "sigma_sigma": [0.01, 0.05, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        }
    else:  # beta_binomial
        hyperparams_grid = {
            "sigma_w": [0.1, 0.5, 1.0, 2.0],
            "sigma_b": [0.1, 0.5, 1.0, 2.0],
            "sigma_nu": [1.0, 5.0, 10.0, 20.0],
        }

    # Load data
    metrics_mapping = load_metrics_mapping(args.eval_dir)
    augmented_df, target_models = prepare_augmented_data(
        metrics_mapping, type=args.type
    )

    # Dictionary to hold results
    results = {}

    if args.model_type == "beta_binomial":
        output_file = args.output.replace(".json", "_beta_binomial.json")
    else:
        output_file = args.output.replace(".json", f"_{args.type}.json")

    # Load existing results if available
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                existing_results = json.load(f)
                print(f"Loaded {len(existing_results)} existing model results.")
                # Convert the simple dict back to the expected structure
                for model, params in existing_results.items():
                    results[model] = {
                        "best_params": params,
                        "best_model": None,  # We don't store the actual model objects
                        "grid_results": None,  # We don't store full grid results
                    }
        except json.JSONDecodeError:
            print(
                f"Warning: Could not parse existing results file {output_file}. Starting fresh."
            )

    # Process each target model
    for target_model in target_models["target_model_key"]:
        # Skip models that are already processed
        if target_model in results:
            print(f"Skipping already processed target model: {target_model}")
            continue

        print(f"\nProcessing target model: {target_model}")

        # Find best model using either grid search or Optuna
        if args.model_type == "linear":
            model_results = find_best_model_for_target(
                df=augmented_df,
                target_model_key=target_model,
                hyperparams_grid=None if args.use_optuna else hyperparams_grid,
                n_samples=args.n_samples,
                tune=args.tune,
                progressbar=args.progressbar,
                use_optuna=args.use_optuna,
                n_jobs=args.n_jobs if args.use_optuna else 1,
            )
        else:  # beta_binomial
            model_results = find_best_beta_binomial_model_for_target(
                df=augmented_df,
                target_model_key=target_model,
                hyperparams_grid=None if args.use_optuna else hyperparams_grid,
                n_samples=args.n_samples,
                tune=args.tune,
                progressbar=args.progressbar,
                use_optuna=args.use_optuna,
                n_jobs=args.n_jobs if args.use_optuna else 1,
            )

        if model_results:
            results[target_model] = {
                "best_params": model_results["best_params"],
                "best_model": model_results["best_model"],
                "grid_results": model_results,
            }

            # Save results after each model
            save_current_results(results, output_file)
            print(f"Saved progress after processing {target_model}")

    # Save final results
    save_current_results(results, output_file)
    print(f"\nScript execution completed. Results saved to {output_file}")


if __name__ == "__main__":
    main()
