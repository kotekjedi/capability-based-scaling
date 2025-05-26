import json
import logging
from pathlib import Path

import pandas as pd
from datasets import Dataset, load_dataset

# Set up logging
logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_llama_factory_dataset(
    df, output_dir="LLaMA-Factory", eval=False, dataset_name_prefix="harmful"
):
    """
    Convert DataFrame to LLaMA Factory's format and save it.

    Args:
        df: Pandas DataFrame with the data
        output_dir: Output directory path
        eval: Whether this is an evaluation dataset
        dataset_name_prefix: Prefix for the dataset name

    Returns:
        Tuple of (dataset_path, dataset_name)
    """
    if eval:
        dataset_name = f"{dataset_name_prefix}_eval"
    else:
        dataset_name = dataset_name_prefix

    # Save in LLaMA Factory's data directory using absolute path
    dataset_path = Path(output_dir).resolve() / "data"  # Get absolute path
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Convert DataFrame to LLaMA Factory format
    llama_factory_data = []
    if dataset_name_prefix == "harmful":
        for _, row in df.iterrows():
            data_item = {
                "messages": [
                    {
                        "role": "user",
                        "content": row["instruction"],
                    },
                    {
                        "role": "assistant",
                        "content": row["output"],
                    },
                ]
            }
            llama_factory_data.append(data_item)
    else:
        for _, row in df.iterrows():
            data_item = {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
            }
            llama_factory_data.append(data_item)

    # Save dataset as json
    dataset_file = dataset_path / f"{dataset_name}.json"
    with open(dataset_file, "w") as f:
        json.dump(llama_factory_data, f, indent=2)

    # Update LLaMA Factory's dataset_info.json
    info_file = dataset_path / "dataset_info.json"
    with open(info_file, "r") as f:
        dataset_info = json.load(f)

    # Add dataset info
    if dataset_name_prefix == "harmful":
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
            "formatting": "sharegpt",
            "columns": {
                "messages": "messages",
            },
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    else:
        dataset_info[dataset_name] = {
            "file_name": f"{dataset_name}.json",
        }

    # Save updated dataset_info.json
    with open(info_file, "w") as f:
        json.dump(dataset_info, f, indent=2)

    return str(dataset_path), dataset_name


def prepare_harmful_dataset(harmful_dataset, eval_dataset):
    """
    Prepare harmful dataset for LLaMA Factory.

    Args:
        harmful_dataset: Dataset with harmful data
        eval_dataset: Evaluation dataset

    Returns:
        Tuple of (train_dataset_name, eval_dataset_name)
    """
    if harmful_dataset is None:
        logger.warning("No harmful dataset provided, skipping preparation")
        return None, None

    logger.info("Preparing harmful dataset for training")

    # Convert training dataset to LLaMA Factory format
    df = harmful_dataset.to_pandas()

    # Ensure the dataframe has the expected columns
    if "instruction" not in df.columns and "input" in df.columns:
        # For datasets that use input instead of instruction
        df["instruction"] = df["input"]

    if "output" not in df.columns and "response" in df.columns:
        # Rename response column to output if needed
        df["output"] = df["response"]

    _, train_dataset_name = prepare_llama_factory_dataset(
        df, dataset_name_prefix="harmful"
    )

    # Convert evaluation dataset to LLaMA Factory format if provided
    eval_dataset_name = None
    if eval_dataset is not None:
        eval_df = eval_dataset.to_pandas()

        # Apply the same column transformations to eval dataset
        if "instruction" not in eval_df.columns and "input" in eval_df.columns:
            eval_df["instruction"] = eval_df["input"]

        if "output" not in eval_df.columns and "response" in eval_df.columns:
            eval_df["output"] = eval_df["response"]

        _, eval_dataset_name = prepare_llama_factory_dataset(
            eval_df,
            eval=True,
            dataset_name_prefix="harmful",
        )

    return train_dataset_name, eval_dataset_name


def prepare_standard_dataset(dataset, dataset_name_prefix):
    """
    Prepare standard format dataset (like Alpaca or ISM-SDA) for LLaMA Factory.

    Args:
        dataset: Dataset to prepare
        dataset_name_prefix: Prefix for the dataset name (e.g. 'alpaca', 'ism_sda')

    Returns:
        train_dataset_name
    """
    if dataset is None:
        logger.warning(
            f"No {dataset_name_prefix} dataset provided, skipping preparation"
        )
        return None

    logger.info(f"Preparing {dataset_name_prefix} dataset for training")

    # Convert training dataset to LLaMA Factory format
    df = dataset.to_pandas()

    _, train_dataset_name = prepare_llama_factory_dataset(
        df, dataset_name_prefix=dataset_name_prefix
    )

    return train_dataset_name


def prepare_alpaca_dataset(alpaca_dataset):
    """
    Prepare Alpaca dataset for LLaMA Factory.
    This is now a wrapper around prepare_standard_dataset for backward compatibility.

    Args:
        alpaca_dataset: Dataset with Alpaca data

    Returns:
        train_dataset_name
    """
    return prepare_standard_dataset(alpaca_dataset, "alpaca")


def prepare_ism_sda_dataset(ism_sda_dataset):
    """
    Prepare ISM-SDA dataset for LLaMA Factory.

    Args:
        ism_sda_dataset: Dataset with ISM-SDA data

    Returns:
        train_dataset_name
    """
    return prepare_standard_dataset(ism_sda_dataset, "ism_sda")


def load_data(data_config):
    """
    Load and prepare datasets based on specified datasets to use.

    Args:
        data_config: Dictionary containing dataset paths and datasets_in_use

    Returns:
        Dictionary with 'harmful' and 'alpaca' keys containing respective datasets
    """
    # Determine which datasets to load
    datasets_to_use = data_config.get("datasets_in_use", "").split()
    datasets_to_use = [d.strip() for d in datasets_to_use if d.strip()]

    if not datasets_to_use:
        # Default to all available datasets if none specified
        datasets_to_use = [
            k for k in data_config.keys() if k != "harmbench" and k != "datasets_in_use"
        ]

    logger.info(f"Loading datasets: {datasets_to_use}")

    # Initialize result dictionary with harmful and alpaca categories
    result = {"harmful": None, "alpaca": None, "ism_sda": None}

    # Load harmful datasets (all except alpaca_1k)
    harmful_datasets = []

    for dataset_name in datasets_to_use:
        if dataset_name == "alpaca_1k":
            continue  # Skip alpaca here, we'll handle it separately
        if dataset_name == "ism_sda_k2_1k":
            continue  # Skip ism_sda_k2_1k here, we'll handle it separately
        if dataset_name in data_config and data_config[dataset_name]:
            try:
                df = pd.read_parquet(data_config[dataset_name])
                logger.info(f"Loaded {dataset_name} with {len(df)} rows")
                harmful_datasets.append(df)
            except Exception as e:
                logger.info(f"Error loading {dataset_name}: {e}")

    # Combine and convert harmful datasets
    if harmful_datasets:
        combined_df = pd.concat(harmful_datasets)
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle

        # Convert to Dataset
        harmful_dataset = Dataset.from_pandas(combined_df)
        if "__index_level_0__" in harmful_dataset.column_names:
            harmful_dataset = harmful_dataset.remove_columns(["__index_level_0__"])

        result["harmful"] = harmful_dataset
        logger.info(f"Combined harmful dataset has {len(harmful_dataset)} examples")

    # Load Alpaca dataset if it's in the list
    if "alpaca_1k" in datasets_to_use:
        alpaca_path = data_config.get("alpaca_1k")
        if alpaca_path:
            try:
                df = pd.read_parquet(alpaca_path)
                logger.info(f"Loaded alpaca_1k with {len(df)} rows")

                # Convert to Dataset
                alpaca_dataset = Dataset.from_pandas(df)
                if "__index_level_0__" in alpaca_dataset.column_names:
                    alpaca_dataset = alpaca_dataset.remove_columns(
                        ["__index_level_0__"]
                    )

                result["alpaca"] = alpaca_dataset
                logger.info(f"Alpaca dataset has {len(alpaca_dataset)} examples")
            except Exception as e:
                logger.info(f"Error loading alpaca_1k: {e}")

    # Load ism_sda_k2_1k dataset if it's in the list
    if "ism_sda_k2_1k" in datasets_to_use:
        ism_sda_path = data_config.get("ism_sda_k2_1k")
        if ism_sda_path:
            try:
                df = pd.read_parquet(ism_sda_path)
                logger.info(f"Loaded ism_sda_k2_1k with {len(df)} rows")

                # Convert to Dataset
                ism_sda_dataset = Dataset.from_pandas(df)
                result["ism_sda"] = ism_sda_dataset
                logger.info(
                    f"ism_sda_k2_1k dataset has {len(ism_sda_dataset)} examples"
                )
            except Exception as e:
                logger.info(f"Error loading ism_sda_k2_1k: {e}")

    return result
