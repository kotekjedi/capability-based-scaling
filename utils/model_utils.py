import logging
import random

import numpy as np
import torch
from transformers import set_seed

# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)


def configure_padding_token(tokenizer, model, model_config):
    """
    Configure padding token for the tokenizer and resize model embeddings if needed.

    Args:
        tokenizer: The tokenizer to configure
        model: The model to resize embeddings if needed
        model_config: Model configuration containing padding settings

    Returns:
        Tuple of (tokenizer, model) with padding properly configured
    """
    logger.info(
        f"Before setting - Pad token: {tokenizer.pad_token}, Pad token ID: {tokenizer.pad_token_id}"
    )

    if "padding_token" in model_config:
        logger.info(f"Setting padding token to: {model_config['padding_token']}")
        if (
            tokenizer.pad_token is None
            or "pad_token" not in tokenizer.special_tokens_map
            or tokenizer.pad_token != model_config["padding_token"]
        ):
            tokenizer.add_special_tokens({"pad_token": model_config["padding_token"]})
            model.resize_token_embeddings(len(tokenizer))
            tokenizer.pad_token = model_config["padding_token"]
            logger.info(f"Added special token and resized embeddings")
        else:
            tokenizer.pad_token = model_config["padding_token"]

    logger.info(
        f"After setting - Pad token: {tokenizer.pad_token}, Pad token ID: {tokenizer.pad_token_id}"
    )

    if tokenizer.pad_token is None:
        logger.info(
            "Warning: No padding token specified in config and none found in tokenizer"
        )
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"Pad token: {tokenizer.pad_token}")
    logger.info(f"Pad token ID: {tokenizer.pad_token_id}")

    return tokenizer, model


def seed_everything(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)
