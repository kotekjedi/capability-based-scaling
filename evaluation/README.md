# Evaluation Module

Evaluate language models on various benchmarks from the project root.

## Supported Benchmarks

- `tinyMMLU`: Smaller MMLU benchmark
- `MMLU`: Full Massive Multitask Language Understanding
- `MMLUPro`: MMLU Pro benchmark
- `tinyGSM8k`: Smaller GSM8k math reasoning
- `tinyArc`: Smaller ARC benchmark
- `IFEval`: Instruction following evaluation
- `harmbench`: Harmful output evaluation

## Basic Usage

```bash
# Evaluate base model
python -m evaluation.eval \
  --model_name llama2_7b \
  --eval_type tinyMMLU harmbench \
  --do_base_eval 1

# Evaluate model with adapter
python -m evaluation.eval \
  --model_name llama2_7b \
  --adapter_path path/to/adapter \
  --eval_type MMLUPro \
  --do_unlocked_eval 1

# Evaluate both base and adapter
python -m evaluation.eval \
  --model_name llama2_7b \
  --eval_type tinyMMLU IFEval \
  --do_base_eval 1 \
  --do_unlocked_eval 1
```

## Key Arguments

- `--model_name`: Model from config (required)
- `--eval_type`: Benchmarks to run (required)
- `--adapter_path`: Path to adapter weights
- `--do_base_eval`: Evaluate base model (0/1)
- `--do_unlocked_eval`: Evaluate with adapter (0/1)
- `--use_vllm`: Use VLLM for faster inference (0/1)

Results are saved to `evaluation_results/{model_name}/` 