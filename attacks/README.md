# Attacks Module

Run adversarial attacks against language models from the project root.

## Supported Attacks

- **PAIR**: Prompt Automatic Iterative Refinement
- **Crescendo**: Multi-turn conversation attacks

## Basic Usage

```bash
# Run PAIR attack
python -m attacks.start_attack \
  --attack_name pair \
  --target_model_key llama2_7b \
  --attacker_model_key llama2_7b \
  --judge_model_key llama2_7b \
  --behavior_rows 0

# Run Crescendo attack
python -m attacks.start_attack \
  --attack_name crescendo \
  --target_model_key llama2_7b \
  --attacker_model_key llama2_7b \
  --judge_model_key llama2_7b \
  --behavior_ids harmful_behavior_id
```

## Key Arguments

- `--attack_name`: Attack type (`pair` or `crescendo`)
- `--target_model_key`: Target model from config
- `--attacker_model_key`: Attacker model from config  
- `--judge_model_key`: Judge model from config
- `--behavior_rows`: Behavior row numbers from CSV
- `--behavior_ids`: Specific behavior IDs
- `--output_dir`: Results directory (default: `results`)

Results are saved to `{output_dir}/{attacker}->{target} (judge: {judge})/`





