import os
import random
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig
from transformers import set_seed
from trl import SFTConfig, SFTTrainer


def set_all_seeds(seed):
    """Set all seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def generate_prompt(example, tokenizer):
    """Generate prompt using the model's chat template if available."""
    instruction = example["instruction"].strip()

    if hasattr(tokenizer, "apply_chat_template"):
        # Use the model's built-in chat template
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": instruction}], tokenize=False
        )
    else:
        # Fallback to simple template
        print(f"No chat template found for {tokenizer.name_or_path}")
        prompt = f"USER: {instruction}\nASSISTANT:"

    return prompt


def tokenize_function(tokenizer, example, max_length):
    prompt = generate_prompt(example, tokenizer)
    full_text = prompt + example["output"]

    tokenized = tokenizer(
        full_text,
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )

    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized


def train_model(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    training_config,
    output_dir,
    model_name,
    model_settings,
):
    # Set all seeds for reproducibility
    set_all_seeds(training_config["training"]["seed"])

    # Set padding side at initialization
    tokenizer.padding_side = model_settings.get("padding_side", "left")

    # Prepare datasets
    max_length = training_config["training"]["max_seq_length"]
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(tokenizer, x, max_length),
        remove_columns=train_dataset.column_names,
    )
    eval_dataset = eval_dataset.map(
        lambda x: tokenize_function(tokenizer, x, max_length),
        remove_columns=eval_dataset.column_names,
    )

    # Configure LoRA
    # peft_config = LoraConfig(**training_config["lora"])

    # Configure training
    output_path = Path(output_dir) / model_name.split("/")[-1].lower()
    os.makedirs(output_path, exist_ok=True)

    # Extract training args, removing keys that aren't accepted by SFTConfig
    training_args = training_config["training"].copy()
    training_args.pop(
        "max_seq_length", None
    )  # Remove max_seq_length as it's not a SFTConfig parameter
    training_args.pop(
        "test_size", None
    )  # Remove test_size as it's not a SFTConfig parameter

    # Keep seed but don't remove it as it's used by SFTConfig

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(output_dir=str(output_path), report_to=[], **training_args),
        # peft_config=peft_config,
    )

    # Train and save
    trainer.train()
    save_path = output_path / "lora_weights"
    trainer.save_model(save_path)
