import json
import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
import torch
import yaml
from datasets import Dataset

from utils.data_utils import (
    prepare_alpaca_dataset,
    prepare_harmful_dataset,
    prepare_ism_sda_dataset,
    prepare_llama_factory_dataset,
)

# Set up logging
logger = logging.getLogger(__name__)


def save_config(config: dict, output_path: Path):
    """Save configuration to a YAML file."""
    with open(output_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def run_training(config_path: str):
    """Execute LLaMA Factory training command."""
    cmd = f"llamafactory-cli train {config_path}"
    logger.info(f"Running command: {cmd}")

    try:
        subprocess.run(cmd, shell=True, check=True)
        logger.info("Training completed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed with error: {e}")
        raise


def train_model(
    model,
    tokenizer,
    data,
    training_config,
    output_dir,
    model_name,
    model_settings,
    temp_saving_path="",
):
    """
    Train model using LLaMA Factory with harmful, Alpaca, and ISM-SDA datasets.

    Args:
        model: The model to train (can be None for LLaMA Factory)
        tokenizer: The tokenizer (can be None for LLaMA Factory)
        data: Dictionary containing 'harmful', 'alpaca', and 'ism_sda' datasets
        training_config: Configuration for training
        output_dir: Output directory path
        model_name: Name of the model
        model_settings: Model-specific settings
        temp_saving_path: Temporary path for saving
    """

    # Create output directory with absolute path
    output_dir = Path(output_dir).resolve()
    model_output_dir = output_dir / model_name.split("/")[-1].lower() / temp_saving_path

    # Clean up existing directory if overwrite is enabled
    if training_config["output"]["overwrite_output_dir"] and model_output_dir.exists():
        import shutil

        shutil.rmtree(model_output_dir)

    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Extract datasets from the data dictionary
    harmful_dataset = data.get("harmful")
    alpaca_dataset = data.get("alpaca")
    ism_sda_dataset = data.get("ism_sda")

    if harmful_dataset is not None:
        harmful_split = harmful_dataset.train_test_split(
            test_size=training_config["training"]["test_size"],
            seed=training_config["training"]["seed"],
        )
        harmful_train = harmful_split["train"]
        harmful_eval = harmful_split["test"]

    # Initialize evaluation dataset name
    harmful_train_name = None
    harmful_eval_name = None
    if harmful_dataset is not None:
        harmful_train_name, harmful_eval_name = prepare_harmful_dataset(
            harmful_train, harmful_eval
        )

    # Prepare Alpaca dataset if provided
    alpaca_train_name = None
    if alpaca_dataset is not None:
        alpaca_train_name = prepare_alpaca_dataset(alpaca_dataset)

    # Prepare ISM-SDA dataset if provided
    ism_sda_train_name = None
    if ism_sda_dataset is not None:
        ism_sda_train_name = prepare_ism_sda_dataset(ism_sda_dataset)

    # Create LLaMA Factory config with absolute paths
    # calculate per_device_eval_batch_size based gpu count
    n_gpus = torch.cuda.device_count()
    batch_size = training_config["training"]["train_batch_size"]
    per_device_batch_size = batch_size // n_gpus

    # Combine datasets for training
    training_datasets = []
    if harmful_train_name:
        training_datasets.append(harmful_train_name)
    if alpaca_train_name:
        training_datasets.append(alpaca_train_name)
    if ism_sda_train_name:
        training_datasets.append(ism_sda_train_name)

    # Add alpaca_en_demo from LLaMA Factory if no datasets are available
    if not training_datasets:
        logger.warning("No datasets provided, using only alpaca_en_demo")

    # Join datasets with commas
    training_dataset_str = ",".join(training_datasets)
    eval_dataset_str = harmful_eval_name

    config = {
        # Basic settings
        "stage": "sft",
        "do_train": True,
        "do_eval": training_config["training"]["do_eval"],
        "model_name_or_path": model_name,
        "output_dir": str(model_output_dir / "lora_weights"),
        # Dataset settings
        "dataset": training_dataset_str,
        "eval_dataset": eval_dataset_str,
        "template": model_settings["template"]
        if "template" in model_settings
        else None,
        "overwrite_cache": training_config["lora"].get("overwrite_cache", True),
        "preprocessing_num_workers": training_config["lora"].get(
            "preprocessing_num_workers", 4
        ),
        # LoRA settings
        "finetuning_type": "lora",
        "lora_rank": training_config["lora"]["lora_rank"],
        "lora_alpha": training_config["lora"]["lora_alpha"],
        "lora_dropout": training_config["lora"]["lora_dropout"],
        "lora_target": training_config["lora"]["lora_target"],
        # Training hyperparameters
        "seed": training_config["training"]["seed"],
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": training_config["training"][
            "gradient_accumulation_steps"
        ],
        "learning_rate": training_config["training"]["learning_rate"],
        "num_train_epochs": training_config["training"]["num_train_epochs"],
        "max_length": training_config["training"]["model_max_length"],
        "logging_steps": training_config["training"]["logging_steps"],
        # "save_steps": training_config["training"]["save_steps"],
        "eval_steps": training_config["training"]["eval_steps"],
        "evaluation_strategy": "steps",
        # Optimization settings
        "lr_scheduler_type": training_config["training"]["lr_scheduler_type"],
        "warmup_ratio": training_config["training"]["warmup_ratio"],
        "max_grad_norm": 1.0,
        "fp16": training_config["training"]["fp16"],
        "bf16": training_config["training"]["bf16"],
        "use_cache": bool(model_settings["use_cache"])
        if "use_cache" in model_settings
        else True,
        # Additional settings
        "trust_remote_code": True,
        "report_to": "none",
    }

    # Save config
    config_path = model_output_dir / "config.yaml"
    save_config(config, config_path)

    try:
        original_dir = os.getcwd()
        os.chdir(Path("LLaMA-Factory").resolve())

        cmd = f"llamafactory-cli train {config_path.resolve()}"
        logger.info(f"Running command: {cmd}")

        # Simple execution with timeout
        try:
            subprocess.run(
                cmd,
                shell=True,
                check=True,
                timeout=training_config["training"]["timeout"],
            )
            logger.info("Training completed successfully")
        except subprocess.TimeoutExpired:
            logger.warning("Training process timed out - continuing with script")
        except subprocess.CalledProcessError:
            logger.warning("Training process ended with error - continuing with script")

    finally:
        os.chdir(original_dir)
