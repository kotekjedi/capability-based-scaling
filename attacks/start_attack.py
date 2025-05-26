import argparse
import datetime
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from utils.harmbench_utils import evaluate_with_harmbench
from utils.model_utils import seed_everything

from .crescendo.crescendo import Crescendo
from .utils.data_utils import load_targets_and_behaviors
from .utils.model_utils import load_models

logging.basicConfig(level=logging.INFO)


def run_pair_attack(
    models: Dict,
    targets: Dict,
    behavior_config: Dict,
    output_dir: str = None,
    temp_saving_path: str = None,
    **kwargs,
) -> Dict:
    """Run PAIR attack with loaded models and configuration."""
    from .pair import PAIR

    # Initialize PAIR attack
    attack = PAIR(
        attack_model=models["attacker"],
        target_model=models["target"],
        judge_model=models["judge"],
        targets=targets,
        **kwargs,
    )

    # Run attack
    test_case, logs, detailed_logs, sorted_test_cases = (
        attack.generate_test_cases_single_behavior(
            behavior_dict=behavior_config, verbose=True
        )
    )

    # Save results if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        behavior_id = behavior_config["BehaviorID"]

        # Create model-specific directory structure
        model_dir = (
            output_dir
            / f"{kwargs['attacker_model_key']}->{kwargs['target_model_key']} (judge: {kwargs['judge_model_key']})"
        )

        if temp_saving_path:
            model_dir = model_dir / temp_saving_path

        results_dir = (
            model_dir / f"{behavior_id} (row: {behavior_config['row_number']})"
        )
        results_dir.mkdir(parents=True, exist_ok=True)

        # Save files
        with open(results_dir / "test_case.txt", "w") as f:
            f.write(test_case)
        with open(results_dir / "logs.json", "w") as f:
            json.dump(logs, f, indent=2)
        with open(results_dir / "detailed_logs.json", "w") as f:
            json.dump(detailed_logs, f, indent=2)
        with open(results_dir / "sorted_test_cases.json", "w") as f:
            json.dump(sorted_test_cases, f, indent=2)

    return {
        "test_case": test_case,
        "logs": logs,
        "detailed_logs": detailed_logs,
        "sorted_test_cases": sorted_test_cases,
        "behavior_config": behavior_config,
    }


def run_crescendo_attack(**kwargs) -> Dict:
    """
    Run Crescendo attack with loaded models and configuration.

    Args:
        **kwargs: Arguments passed to Crescendo attack function

    Returns:
        Dictionary containing attack results
    """
    # Extract models and behavior configuration from kwargs
    models = kwargs["models"]
    behavior_config = kwargs["behavior_config"]

    # Map arguments correctly
    max_rounds = kwargs.get("steps", 10)  # Use steps for max_rounds
    use_system_prompt = kwargs.get("use_system_prompt", True)
    batch_size = kwargs.get("batch_size", 4)
    output_dir = kwargs.get("output_dir", None)
    temp_saving_path = kwargs.get("temp_saving_path", None)

    # Set token limits for models
    attack_model = models["attacker"]
    target_model = models["target"]
    judge_model = models["judge"]

    # Set max tokens if provided
    if "attack_max_n_tokens" in kwargs:
        attack_model["max_tokens"] = kwargs["attack_max_n_tokens"]
    if "target_max_n_tokens" in kwargs:
        target_model["max_tokens"] = kwargs["target_max_n_tokens"]
    if "judge_max_n_tokens" in kwargs:
        judge_model["max_tokens"] = kwargs["judge_max_n_tokens"]

    # Initialize Crescendo attack
    attack = Crescendo(
        attack_model=attack_model,
        target_model=target_model,
        judge_model=judge_model,
        max_rounds=max_rounds,
        batch_size=batch_size,
        use_system_prompt=use_system_prompt,
        early_stopping=False,  # Default to False unless explicitly specified
    )

    # Create the proper nested directory structure for output
    crescendo_output_dir = None
    if output_dir:
        output_dir = Path(output_dir)
        behavior_id = behavior_config["BehaviorID"]

        # Create model-specific directory structure
        model_dir = (
            output_dir
            / f"{kwargs['attacker_model_key']}->{kwargs['target_model_key']} (judge: {kwargs['judge_model_key']})"
        )

        if temp_saving_path:
            model_dir = model_dir / temp_saving_path

        # Create behavior-specific directory
        crescendo_output_dir = (
            model_dir / f"{behavior_id} (row: {behavior_config['row_number']})"
        )
        crescendo_output_dir.mkdir(parents=True, exist_ok=True)

    # Run Crescendo attack
    # Unpack the new return structure
    batch_conversation_logs, detailed_logs, criteria_dict = (
        attack.generate_test_cases_single_behavior(
            behavior_dict=behavior_config,
            output_dir=crescendo_output_dir,
            verbose=kwargs.get("verbose", False),
            n_streams=kwargs.get("n_streams", 3),  # Pass n_streams explicitly
        )
    )

    # Save the results directly
    if crescendo_output_dir:
        # Save conversation logs (list of lists)
        with open(crescendo_output_dir / "conversation_logs.json", "w") as f:
            json.dump(batch_conversation_logs, f, indent=2)
        # Save detailed logs (flat list)
        with open(crescendo_output_dir / "detailed_logs.json", "w") as f:
            json.dump(detailed_logs, f, indent=2)
        # Save criteria (already handled inside the method, but could save here too if needed)
        # with open(crescendo_output_dir / "criteria.json", "w") as f:
        #     json.dump(criteria_dict, f, indent=2)

    # Return a dictionary compatible with the rest of the script if needed,
    # otherwise, the saving above might be sufficient.
    # For consistency, let's return a similar structure to before, assembling it here.
    return {
        "batch_conversation_logs": batch_conversation_logs,
        "detailed_logs": detailed_logs,
        "criteria": criteria_dict.get("criteria", ""),
        "behavior_config": behavior_config,
        # Add other relevant info if necessary
    }


def run_attack(attack_name: str, **kwargs) -> Dict:
    """
    Run specified attack with given configuration.

    Args:
        attack_name: Name of attack to run (e.g. "pair", "crescendo")
        **kwargs: Arguments passed to specific attack function

    Returns:
        Dictionary containing attack results
    """
    attack_name = attack_name.lower()

    if attack_name == "pair":
        return run_pair_attack(**kwargs)
    elif attack_name == "crescendo":
        return run_crescendo_attack(**kwargs)
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run model attacks")

    # Required arguments
    parser.add_argument(
        "--attack_name",
        type=str,
        required=True,
        choices=["pair", "crescendo"],
        help="Name of attack to run",
    )
    parser.add_argument(
        "--target_model_key",
        type=str,
        required=True,
        help="Key for target model from config (e.g. llama2_7b)",
    )
    parser.add_argument(
        "--attacker_model_key",
        type=str,
        required=True,
        help="Key for attacker model from config",
    )
    parser.add_argument(
        "--judge_model_key",
        type=str,
        default=None,
        help="Key for judge model from config",
    )

    # Optional arguments
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save results"
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to run models on"
    )
    parser.add_argument(
        "--temp_saving_path",
        type=str,
        default=None,
        help="Optional temporary path to add to results directory structure",
    )

    # API Keys
    parser.add_argument(
        "--openai-key",
        type=str,
        help="OpenAI API key for API-based models",
    )
    parser.add_argument(
        "--openrouter-key",
        type=str,
        help="OpenRouter API key for API-based models",
    )

    # LoRA adapter paths
    parser.add_argument(
        "--attacker_lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter for attacker model",
    )
    parser.add_argument(
        "--judge_lora_path",
        type=str,
        default=None,
        help="Path to LoRA adapter for judge model",
    )
    parser.add_argument(
        "--target_lora_path",
        type=str,
        default=None,
        help="ONLY USED FOR LOADING TOKENIZER CONFIG",
    )

    # HarmBench evaluation
    parser.add_argument(
        "--use_harmbench_judge",
        action="store_true",
        help="Whether to use HarmBench judge for evaluation",
    )

    # Judge type
    parser.add_argument(
        "--judge_type",
        type=str,
        default="hf_instruct",
        choices=["hf_instruct", "random", "constant"],
        help="Type of judge to use (hf_instruct, random, constant)",
    )

    # Attack parameters
    parser.add_argument(
        "--attack_max_n_tokens",
        type=int,
        default=512,
        help="Max tokens for attack generation",
    )
    parser.add_argument(
        "--target_max_n_tokens",
        type=int,
        default=512,
        help="Max tokens for target response",
    )
    parser.add_argument(
        "--judge_max_n_tokens",
        type=int,
        default=10,
        help="Max tokens for judge response",
    )
    parser.add_argument(
        "--n_streams", type=int, default=3, help="Number of parallel attack streams"
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of attack steps")

    # Behavior-related arguments
    parser.add_argument(
        "--behavior_configs",
        type=str,
        nargs="+",
        help="List of JSON strings or paths to behavior config files",
        default=None,
    )
    parser.add_argument(
        "--behavior_ids",
        type=str,
        nargs="+",
        help="List of BehaviorIDs from default behaviors CSV to use",
    )
    parser.add_argument(
        "--behavior_rows",
        type=int,
        nargs="+",
        help="List of row numbers from default behaviors CSV to use",
    )
    parser.add_argument(
        "--use_system_prompt",
        type=bool,
        default=False,
        help="Whether to use system prompt for attack",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for model inference",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for attack",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    seed_everything(args.seed)

    # Set API keys if provided
    if args.openai_key:
        os.environ["OPENAI_API_KEY"] = args.openai_key
    if args.openrouter_key:
        os.environ["OPENROUTER_API_KEY"] = args.openrouter_key

    # Load behavior configs from files or strings
    behavior_configs = None
    if args.behavior_configs:
        behavior_configs = []
        for config in args.behavior_configs:
            if os.path.isfile(config):
                with open(config) as f:
                    behavior_configs.append(json.load(f))
            else:
                try:
                    behavior_configs.append(json.loads(config))
                except json.JSONDecodeError:
                    raise ValueError(
                        "behavior_configs must be valid JSON strings or paths to JSON files"
                    )

    # Load all models first
    models = load_models(
        target_model_key=args.target_model_key,
        attacker_model_key=args.attacker_model_key,
        judge_model_key=args.judge_model_key,
        device=args.device,
        target_lora_path=args.target_lora_path,
        attacker_lora_path=args.attacker_lora_path,
        judge_lora_path=args.judge_lora_path,
    )

    # Load targets and behaviors
    targets, behavior_configs = load_targets_and_behaviors(
        behavior_configs=behavior_configs,
        behavior_ids=args.behavior_ids,
        row_numbers=args.behavior_rows,
    )

    # PHASE 1: Run attack for each behavior
    logging.info("PHASE 1: Running attacks for all behaviors")
    all_attack_results = []

    for behavior_config in behavior_configs:
        logging.info(f"Running attack for behavior {behavior_config['BehaviorID']}")
        if args.attack_name == "pair":
            results = run_pair_attack(
                models=models,
                targets=targets,
                behavior_config=behavior_config,
                output_dir=args.output_dir,
                temp_saving_path=args.temp_saving_path,
                attack_max_n_tokens=args.attack_max_n_tokens,
                target_max_n_tokens=args.target_max_n_tokens,
                judge_max_n_tokens=args.judge_max_n_tokens,
                n_streams=args.n_streams,
                steps=args.steps,
                use_system_prompt=args.use_system_prompt,
                attacker_model_key=args.attacker_model_key,
                target_model_key=args.target_model_key,
                judge_model_key=args.judge_model_key,
                judge_type=args.judge_type,
                batch_size=args.batch_size,
            )
        elif args.attack_name == "crescendo":
            results = run_crescendo_attack(
                models=models,
                targets=targets,
                behavior_config=behavior_config,
                output_dir=args.output_dir,
                temp_saving_path=args.temp_saving_path,
                attack_max_n_tokens=args.attack_max_n_tokens,
                target_max_n_tokens=args.target_max_n_tokens,
                judge_max_n_tokens=args.judge_max_n_tokens,
                steps=args.steps,
                use_system_prompt=args.use_system_prompt,
                batch_size=args.batch_size,
                attacker_model_key=args.attacker_model_key,
                target_model_key=args.target_model_key,
                judge_model_key=args.judge_model_key,
            )
        all_attack_results.append(results)

    # Free up memory after attack phase
    del models
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    # PHASE 2: Run HarmBench evaluation if requested
    if args.use_harmbench_judge:
        logging.info("PHASE 2: Running HarmBench evaluation")
        results_dir = Path(args.output_dir)
        model_dir = (
            results_dir
            / f"{args.attacker_model_key}->{args.target_model_key} (judge: {args.judge_model_key})"
        )

        # Run eval.py as a module
        cmd = ["python", "-m", "attacks.eval", "--results_dir", str(model_dir)]

        try:
            subprocess.run(cmd, check=True)
            logging.info("HarmBench evaluation completed successfully")
        except subprocess.CalledProcessError as e:
            logging.error(f"HarmBench evaluation failed with error: {e}")
