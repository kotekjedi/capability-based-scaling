import argparse
import gc
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import lm_eval.models.huggingface
import lm_eval.models.vllm_causallms
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.config import load_yaml
from utils.logger import logger
from utils.model_utils import configure_padding_token

from .harmbench_eval import evaluate_harmbench
from .metrics_config import get_summary_metrics

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class GPUConfig:
    """Configuration for GPU utilization and parallelism."""

    gpu_memory_utilization: float = 0.65  # Default from the code
    tensor_parallel_size: int = 1  # Use all available GPUs for tensor parallelism
    data_parallel_size: int = 4  # Number of model replicas for data parallelism
    device: str = "cuda"
    tokenizers_parallelism: bool = False  # From os.environ setting
    disable_custom_all_reduce: bool = (
        True  # Disable custom all reduce for data parallel
    )


def wrap_model_for_evaluation(
    model_name: str,
    tokenizer,
    use_vllm: bool = False,
    max_model_len: int = None,
    torch_dtype: str = "bfloat16",
    lora_path: str = None,
    gpu_config: GPUConfig = None,
):
    """Wrap the model with either VLLM or HuggingFace wrapper for evaluation."""
    if gpu_config is None:
        gpu_config = GPUConfig()

    if use_vllm:
        # Initialize VLLM wrapper with model name and optional LoRA path
        logger.info(
            f"Initializing VLLM with model {model_name}"
            + (f" and LoRA {lora_path}" if lora_path else "")
        )
        model_kwargs = {
            "pretrained": model_name,
            "enforce_eager": True,
            "batch_size": "auto",  # Required for data parallel
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_config.gpu_memory_utilization,
            "tensor_parallel_size": gpu_config.tensor_parallel_size,
            "data_parallel_size": gpu_config.data_parallel_size,
            "disable_custom_all_reduce": gpu_config.disable_custom_all_reduce,
            "dtype": torch_dtype,
            "device": gpu_config.device,
            "trust_remote_code": True,
            "tokenizer": lora_path,
        }

        if lora_path:
            model_kwargs.update({"enable_lora": True, "lora_local_path": lora_path})

        wrapped_model = lm_eval.models.vllm_causallms.VLLM(**model_kwargs)

        if gpu_config.data_parallel_size > 1:
            logger.info(
                f"Using data parallelism with {gpu_config.data_parallel_size} replicas"
            )
            logger.info("Note: Manual batching is disabled when using data parallelism")
    else:
        # For non-VLLM, we need to load the model ourselves
        logger.info(f"Loading model {model_name} for HF evaluation")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=gpu_config.device,
            torch_dtype=getattr(torch, torch_dtype)
            if isinstance(torch_dtype, str)
            else torch_dtype,
        )
        if lora_path:
            logger.info(f"Loading LoRA from {lora_path}")
            base_model = PeftModel.from_pretrained(base_model, lora_path)

        wrapped_model = lm_eval.models.huggingface.HFLM(
            pretrained=base_model,
            tokenizer=tokenizer,
            device=gpu_config.device,
        )

    return wrapped_model


def load_tokenizer(
    model_config: dict, model_name: str, adapter_path: str = None
) -> AutoTokenizer:
    """Load tokenizer from adapter path first, fallback to base model if needed."""
    config = model_config[model_name]

    # Try adapter path first (either from args or config)
    adapter_path = adapter_path or config.get("adapter_path")
    if adapter_path:
        try:
            logger.info(f"Loading tokenizer from adapter path: {adapter_path}")
            tokenizer = AutoTokenizer.from_pretrained(adapter_path)
            logger.info("Successfully loaded tokenizer from adapter")
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from adapter: {e}")
            logger.info(f"Falling back to base model tokenizer: {config['name']}")
            tokenizer = AutoTokenizer.from_pretrained(config["name"])
    else:
        logger.info(
            f"No adapter path specified, loading base model tokenizer: {config['name']}"
        )
        tokenizer = AutoTokenizer.from_pretrained(config["name"])
    return tokenizer


def evaluate_model(
    model_name: str,
    tokenizer,
    tasks: list,
    use_vllm: bool = False,
    max_model_len: int = None,
    torch_dtype: str = "bfloat16",
    lora_path: str = None,
    num_fewshot: int = None,
    batch_size: str = "auto",
    gpu_config: GPUConfig = None,
) -> dict:
    """Run evaluation for a single model configuration."""
    if gpu_config is None:
        gpu_config = GPUConfig()

    # Log GPU configuration
    logger.info(f"GPU Configuration:")
    logger.info(f"  - GPU Memory Utilization: {gpu_config.gpu_memory_utilization}")
    logger.info(f"  - Tensor Parallel Size: {gpu_config.tensor_parallel_size}")
    logger.info(f"  - Data Parallel Size: {gpu_config.data_parallel_size}")
    logger.info(f"  - Device: {gpu_config.device}")

    results = {}

    # For HarmBench evaluation, we need to use the non-VLLM version
    if "harmbench" in tasks:
        logger.info("Running HarmBench evaluation (using non-VLLM model)...")
        # Load the model without VLLM wrapping
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=gpu_config.device,
            torch_dtype=getattr(torch, torch_dtype)
            if isinstance(torch_dtype, str)
            else torch_dtype,
        )
        if lora_path:
            base_model = PeftModel.from_pretrained(base_model, lora_path)

        # Run HarmBench evaluation
        harmbench_results = evaluate_harmbench(
            model=base_model,
            tokenizer=tokenizer,
            harmbench_path="data/harmbench_behaviors_all_no_copyright.csv",
            num_behaviors=50,
            batch_size=int(batch_size) if batch_size != "auto" else 1,
        )
        results["harmbench"] = harmbench_results

        # Clean up
        del base_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Remove harmbench from tasks to prevent double evaluation
        tasks = [t for t in tasks if t != "harmbench"]

    # For other tasks, use VLLM if specified
    if tasks:  # If there are remaining tasks
        # When using data parallelism with VLLM, batch_size must be "auto"
        if use_vllm and gpu_config.data_parallel_size > 1:
            batch_size = "auto"
            logger.info("Setting batch_size to 'auto' for data parallel evaluation")

        wrapped_model = wrap_model_for_evaluation(
            model_name=model_name,
            tokenizer=tokenizer,
            use_vllm=use_vllm,
            max_model_len=max_model_len,
            torch_dtype=torch_dtype,
            lora_path=lora_path,
            gpu_config=gpu_config,
        )

        # Directly use lm_eval.evaluator.simple_evaluate instead of our custom ModelEvaluator
        logger.info("Starting model evaluation...")
        logger.info(f"Evaluating model on {len(tasks)} tasks")
        logger.info(f"Tasks: {tasks}")

        task_results = lm_eval.evaluator.simple_evaluate(
            model=wrapped_model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            device=gpu_config.device,
        )

        # Handle None results (happens with data parallel VLLM when not on rank 0)
        if task_results is not None:
            results.update(task_results)
        else:
            logger.warning(
                "Received None results from evaluator (likely not on rank 0)"
            )

    return results


def get_tasks_for_eval_types(eval_types: list) -> list:
    """Get tasks for multiple evaluation types."""
    tasks = []
    for eval_type in eval_types:
        if eval_type == "harmbench":
            continue  # Skip harmbench as it's handled separately
        tasks.append(eval_type)
    return tasks


def save_evaluation_results(
    model_name: str,
    eval_types: list,
    base_results: dict = None,
    adapter_results: dict = None,
    args_type: str = None,
):
    """Save both detailed and summary results."""
    # Create model-specific results directory
    model_dir = Path("evaluation_results") / model_name.lower()
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_dir = model_dir / "detailed" / (args_type or "default")
    detailed_dir.mkdir(parents=True, exist_ok=True)

    # Initialize summary results
    summary_dict = {}

    # Save base model results if available
    if base_results:
        # Save detailed results
        with open(detailed_dir / "base_results.json", "w") as f:
            # pop config from base_results
            base_results.pop("configs", None)
            json.dump(base_results, f, indent=2)
        logger.info(f"Detailed base results saved to {detailed_dir}/base_results.json")

        # Extract summary metrics for target (base) model
        base_summary = get_summary_metrics(base_results, is_attacker=False)
        summary_dict.update(base_summary)

    # Save adapter results if available
    if adapter_results:
        # Save detailed results
        with open(detailed_dir / "adapter_results.json", "w") as f:
            # pop config from adapter_results
            adapter_results.pop("configs", None)
            json.dump(adapter_results, f, indent=2)
        logger.info(
            f"Detailed adapter results saved to {detailed_dir}/adapter_results.json"
        )

        # Extract summary metrics for attacker (adapter) model
        adapter_summary = get_summary_metrics(adapter_results, is_attacker=True)
        summary_dict.update(adapter_summary)

    # Save combined detailed results if both are available
    if base_results and adapter_results:
        combined = {
            "base": base_results,
            "adapter": adapter_results,
            "eval_types": eval_types,
        }
        with open(detailed_dir / "combined_results.json", "w") as f:
            json.dump(combined, f, indent=2)
        logger.info(
            f"Detailed combined results saved to {detailed_dir}/combined_results.json"
        )

    # Save summary results
    summary_file = model_dir / "summary.json"

    # load if exists
    if summary_file.exists():
        with open(summary_file, "r") as f:
            existing_summary = json.load(f)
    else:
        existing_summary = {}

    # update existing summary
    existing_summary.update(summary_dict)

    # Save updated summary
    with open(summary_file, "w") as f:
        json.dump(existing_summary, f, indent=2)
    logger.info(f"Summary results saved to {summary_file}")
    logger.info(f"Summary metrics: {existing_summary}")


def main(args):
    """Main evaluation function.

    Args:
        args: Optional arguments. If not provided, will parse from command line.
    """
    # Load model configuration
    model_config = load_yaml(args.model_config)

    # Check if harmbench is in the eval_types
    has_harmbench = "harmbench" in args.eval_type

    # Get all non-harmbench tasks as a single list
    eval_types = args.eval_type
    all_tasks = get_tasks_for_eval_types(eval_types)

    # Get model configuration
    if args.model_name not in model_config:
        raise ValueError(
            f"Model {args.model_name} not found in config. Available models: {list(model_config.keys())}"
        )
    config = model_config[args.model_name]
    hf_model_name = config["name"]
    if "path" in config:
        hf_model_name = config["path"]  # for local models

    # Load tokenizer with proper fallback logic
    tokenizer = load_tokenizer(model_config, args.model_name, args.adapter_path)

    # Get adapter path from args or config
    adapter_path = args.adapter_path or config.get("adapter_path")

    # Create GPU config from args
    gpu_config = GPUConfig(
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        device=args.device,
        tokenizers_parallelism=args.tokenizers_parallelism,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
    )

    # Evaluate base model if requested
    if args.do_base_eval:
        logger.info("Evaluating base model...")
        base_results = {"results": {}}

        # First handle non-harmbench tasks with VLLM if enabled
        if all_tasks:
            logger.info("Running regular evaluations...")

            base_results = evaluate_model(
                model_name=hf_model_name,
                tokenizer=tokenizer,
                tasks=all_tasks,
                use_vllm=bool(args.use_vllm),
                max_model_len=args.max_model_len,
                torch_dtype=args.torch_dtype,
                lora_path=None,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                gpu_config=gpu_config,
            )

        # Then handle harmbench separately if requested
        if has_harmbench:
            logger.info("Running HarmBench evaluation...")
            base_model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                device_map="auto",
                torch_dtype="bfloat16",
            )
            tokenizer, base_model = configure_padding_token(
                tokenizer, base_model, model_config
            )
            harmbench_results = evaluate_harmbench(
                model=base_model,
                tokenizer=tokenizer,
                use_safe_system_prompt=True,
                harmbench_path=config.get(
                    "harmbench_path", "data/harmbench_behaviors_all_no_copyright.csv"
                ),
                num_behaviors=config.get("num_behaviors", 50),
                batch_size=int(args.batch_size) if args.batch_size != "auto" else 1,
            )
            base_results["results"]["harmbench"] = harmbench_results

            del base_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    else:
        base_results = None

    # Evaluate unlocked model if requested
    if args.do_unlocked_eval and adapter_path:
        logger.info("Evaluating unlocked model with LoRA...")
        adapter_results = {}

        # First handle non-harmbench tasks with VLLM if enabled
        if all_tasks:
            logger.info("Running regular evaluations...")

            adapter_results = evaluate_model(
                model_name=hf_model_name,
                tokenizer=tokenizer,
                tasks=all_tasks,
                use_vllm=bool(args.use_vllm),
                max_model_len=args.max_model_len,
                torch_dtype=args.torch_dtype,
                lora_path=adapter_path,
                num_fewshot=args.num_fewshot,
                batch_size=args.batch_size,
                gpu_config=gpu_config,
            )

        # Then handle harmbench separately if requested
        if has_harmbench:
            logger.info("Running HarmBench evaluation...")
            adapter_model = AutoModelForCausalLM.from_pretrained(
                hf_model_name,
                device_map=args.device,
                torch_dtype=getattr(torch, args.torch_dtype)
                if isinstance(args.torch_dtype, str)
                else args.torch_dtype,
            )
            adapter_model = PeftModel.from_pretrained(adapter_model, adapter_path)

            harmbench_results = evaluate_harmbench(
                model=adapter_model,
                tokenizer=tokenizer,
                harmbench_path=config.get(
                    "harmbench_path", "data/harmbench_behaviors_all_no_copyright.csv"
                ),
                num_behaviors=config.get("num_behaviors", 50),
                batch_size=int(args.batch_size) if args.batch_size != "auto" else 1,
            )
            adapter_results["harmbench"] = harmbench_results

            del adapter_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    else:
        adapter_results = None

    # Save all results
    save_evaluation_results(
        model_name=args.model_name,
        eval_types=eval_types,
        base_results=base_results,
        adapter_results=adapter_results,
        args_type=args.args_type,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate language models on various tasks"
    )
    # Model configuration
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model from model_config.yaml to evaluate",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        help="Optional path to model adapter/LoRA weights (overrides config)",
    )
    parser.add_argument(
        "--eval_type",
        type=str,
        nargs="+",
        required=True,
        help="Space-separated evaluation types (e.g. tinyMMLU IFEval harmbench)",
    )

    # GPU configuration
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.65,
        help="GPU memory utilization (default: 0.65)",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Number of GPUs for tensor parallelism (default: 1)",
    )
    parser.add_argument(
        "--data_parallel_size",
        type=int,
        default=4,
        help="Number of model replicas for data parallelism (default: 4)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on (default: cuda)",
    )
    parser.add_argument(
        "--tokenizers_parallelism",
        type=bool,
        default=False,
        help="Enable tokenizers parallelism (default: False)",
    )
    parser.add_argument(
        "--disable_custom_all_reduce",
        type=bool,
        default=True,
        help="Disable custom all reduce for data parallel (default: True)",
    )

    # Evaluation configuration
    parser.add_argument(
        "--use_vllm",
        type=int,
        default=0,
        help="Whether to use VLLM for inference (0/1)",
    )
    parser.add_argument(
        "--batch_size",
        type=str,
        default="auto",
        help="Batch size for evaluation (int or 'auto')",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        help="Number of few-shot examples (if not specified, uses task default)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        help="Torch dtype for model",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="config/model_config.yaml",
        help="Path to model config",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=None,
        help="Maximum sequence length for VLLM",
    )
    parser.add_argument(
        "--args_type",
        type=str,
        help="Type of arguments for saving results",
    )

    # Dual evaluation flags
    parser.add_argument(
        "--do_base_eval",
        type=int,
        default=0,
        help="Whether to evaluate base model (0/1)",
    )
    parser.add_argument(
        "--do_unlocked_eval",
        type=int,
        default=0,
        help="Whether to evaluate model with LoRA (0/1)",
    )

    args = parser.parse_args()
    main(args)
