import logging
import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from utils.config import load_yaml, parse_args, update_config
from utils.data_utils import load_data
from utils.model_utils import configure_padding_token, seed_everything

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_model_settings(model_config, model_name):
    """Get model settings from config key (e.g., "llama2_7b")."""
    if model_name not in model_config:
        available_models = list(model_config.keys())
        raise ValueError(
            f"Model '{model_name}' not found in config. Available models: {available_models}"
        )
    return model_config[model_name]


def main():
    # Parse arguments and load configs
    args = parse_args()
    config_dir = Path(args.config_dir)

    training_config = load_yaml(config_dir / "training_config.yaml")
    model_config = load_yaml(config_dir / "model_config.yaml")

    # Update configs with command line arguments
    training_config = update_config(training_config, args)
    model_config = update_config(model_config, args)

    # Get model configuration
    if not args.model_name:
        available_models = list(model_config.keys())
        raise ValueError(
            f"Model name must be provided via config key. Available models: {available_models}"
        )

    model_settings = get_model_settings(model_config, args.model_name)
    seed_everything(training_config["training"]["seed"])

    # # Initialize model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_settings["name"]
        if "path" not in model_settings
        else model_settings["path"]
    )

    # # Load model Gemma3ForConditionalGeneration for
    model = AutoModelForCausalLM.from_pretrained(
        model_settings["name"]
        if "path" not in model_settings
        else model_settings["path"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )

    # Configure padding token
    tokenizer, model = configure_padding_token(tokenizer, model, model_settings)

    # Create model output directory
    model_name_dir = model_settings["name"].split("/")[-1].lower()
    model_output_dir = os.path.join(
        training_config["output"]["base_path"],
        model_name_dir,
        args.temp_saving_path if args.temp_saving_path else "",
    )
    os.makedirs(model_output_dir, exist_ok=True)

    # Load and prepare data using the centralized load_data function
    datasets = load_data(training_config["data"])

    # Train the model using selected backend
    if training_config["training"]["training_backend"] == "llama_factory":
        from unlocking.train_llama_factory import train_model

        try:
            del model
            del tokenizer
        except Exception as e:
            print(f"Error deleting model and tokenizer: {e}")
        model = None
        tokenizer = None
    else:
        from unlocking.train import train_model

    train_model(
        model=model,
        tokenizer=tokenizer,
        data=datasets,
        training_config=training_config,
        output_dir=training_config["output"]["base_path"],
        model_name=model_settings["name"]
        if "path" not in model_settings
        else model_settings["path"],
        model_settings=model_settings,
    )
    # evaluate the model on harmbench
    from evaluation.harmbench_eval import evaluate_harmbench, load_trained_model

    adapter_path = os.path.join(
        training_config["output"]["base_path"],
        model_name_dir,
        args.temp_saving_path if args.temp_saving_path else "",
        "lora_weights",
    )
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    model = load_trained_model(
        model_settings["name"], device_map="auto", adapter_path=adapter_path
    )

    evaluate_harmbench(
        model, tokenizer, harmbench_path=training_config["data"]["harmbench"]
    )


if __name__ == "__main__":
    main()
