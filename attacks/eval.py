import argparse
import datetime
import gc
import json
import logging
import os
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from utils.harmbench_utils import evaluate_with_harmbench, load_harmbench_judge

logging.basicConfig(level=logging.INFO)


def adjusted_corrcoef(x, y, epsilon=1e-8):
    """Calculate correlation coefficient with numerical stability."""
    x = np.asarray(x)
    y = np.asarray(y)

    mean_x = np.mean(x)
    mean_y = np.mean(y)

    cov = np.sum((x - mean_x) * (y - mean_y))
    std_x = np.sqrt(np.sum((x - mean_x) ** 2) + epsilon)
    std_y = np.sqrt(np.sum((y - mean_y) ** 2) + epsilon)

    return cov / (std_x * std_y)


def load_cached_harmbench_results(behavior_dir: Path) -> Dict:
    """Load cached HarmBench results if they exist."""
    cache_file = behavior_dir / "harmbench_results.json"
    if cache_file.exists():
        with open(cache_file, "r") as f:
            return json.load(f)
    return None


def save_harmbench_results(behavior_dir: Path, results: Dict):
    """Save HarmBench results to cache file."""
    cache_file = behavior_dir / "harmbench_results.json"
    with open(cache_file, "w") as f:
        json.dump(results, f, indent=2)


def evaluate_results_folder(results_dir: Path) -> Dict:
    """Evaluate all behaviors in a results directory."""
    all_harmbench_results = []
    successful_prompts = []
    behavior_row_numbers = {}  # Store row numbers for each behavior

    # Load HarmBench judge once
    harmbench_judge, harmbench_tokenizer = load_harmbench_judge()

    # Find all behavior directories in one attacker-target model pair
    behavior_dirs = []
    results_dir = Path(results_dir)  # Ensure it's a Path object
    for item in results_dir.iterdir():
        if item.is_dir() and "row" in item.name:
            behavior_dirs.append(item)

    logging.info(f"Found {len(behavior_dirs)} behavior directories to evaluate")

    for behavior_dir in behavior_dirs:
        behavior_dir = Path(behavior_dir)  # Ensure it's a Path object
        # Extract behavior ID and row number from directory name
        dir_name = behavior_dir.name
        behavior_id = dir_name.split(" (row:")[0]
        row_number = int(dir_name.split(" (row:")[1].rstrip(")"))
        behavior_row_numbers[behavior_id] = row_number

        # Define potential result file names
        crescendo_logs_file = behavior_dir / "conversation_logs.json"
        crescendo_final_results_file = (
            behavior_dir / f"final_results_{behavior_id}.json"
        )
        pair_test_cases_file = behavior_dir / "sorted_test_cases.json"
        pair_logs_file = behavior_dir / "logs.json"

        test_cases = []
        behavior = ""
        context = None  # Context not currently used/extracted
        data_loaded_for_behavior = False

        # Try loading Crescendo data first
        if crescendo_logs_file.exists():
            logging.info(
                f"Detected Crescendo run for {behavior_id} (found {crescendo_logs_file})"
            )
            try:
                with open(crescendo_logs_file, "r") as f:
                    conversation_logs_batch = json.load(f)
                # Extract prompts and scores
                for stream_logs in conversation_logs_batch:
                    for round_log in stream_logs:
                        if "user" in round_log and round_log["user"]:
                            prompt = round_log["user"]
                            score_val = round_log.get("score", 0)
                            rating = 0
                            if isinstance(score_val, (int, float)):
                                rating = score_val
                            else:
                                try:
                                    rating = int(score_val)
                                except (ValueError, TypeError):
                                    rating = 0
                            # Include the assistant response as well
                            response = round_log.get(
                                "assistant", ""
                            )  # Get the assistant's response
                            test_cases.append(
                                {
                                    "prompt": prompt,
                                    "rating": rating,
                                    "response": response,  # Add the response key
                                }
                            )
                logging.info(
                    f"Extracted {len(test_cases)} test cases (prompt, rating, response) from Crescendo logs."
                )

                # Extract behavior from final_results.json
                if crescendo_final_results_file.exists():
                    with open(crescendo_final_results_file, "r") as f:
                        final_results_list = json.load(f)
                        if final_results_list and "goal" in final_results_list[0]:
                            behavior = final_results_list[0]["goal"]
                else:
                    logging.warning(
                        f"Could not find {crescendo_final_results_file} to extract behavior."
                    )

                if test_cases and behavior:
                    data_loaded_for_behavior = True
                else:
                    logging.warning(
                        f"Failed to load complete data from Crescendo files for {behavior_id}."
                    )

            except Exception as e:
                logging.error(
                    f"Error processing Crescendo files for {behavior_id}: {e}"
                )
                # Don't continue, let it fall through to PAIR check or skip later

        # Load PAIR data if Crescendo data wasn't loaded successfully
        if not data_loaded_for_behavior:
            logging.info(f"Attempting to load PAIR data for {behavior_id}")
            if pair_test_cases_file.exists():
                try:
                    with open(pair_test_cases_file, "r") as f:
                        test_cases = json.load(f)
                    logging.info(f"Loaded {len(test_cases)} test cases from PAIR file.")
                    # Extract behavior from logs.json for PAIR
                    if pair_logs_file.exists():
                        with open(pair_logs_file, "r") as f:
                            logs = json.load(f)
                            if logs and logs[0] and logs[0][0]:
                                processed_response = logs[0][0].get(
                                    "processed_response", ""
                                )
                                if "behavior: '" in processed_response.lower():
                                    behavior = processed_response.split("behavior: '")[
                                        1
                                    ].split("'")[0]
                                elif "objective: '" in processed_response.lower():
                                    behavior = processed_response.split("objective: '")[
                                        1
                                    ].split("'")[0]
                    else:
                        logging.warning(
                            f"Could not find {pair_logs_file} to extract behavior for PAIR."
                        )

                    if test_cases and behavior:
                        data_loaded_for_behavior = (
                            True  # Mark as loaded even if behavior missing from logs
                        )
                    else:
                        logging.warning(
                            f"Failed to load complete data from PAIR files for {behavior_id}."
                        )

                except Exception as e:
                    logging.error(f"Error processing PAIR files for {behavior_id}: {e}")
            else:
                logging.warning(f"Could not find PAIR file {pair_test_cases_file}")

        # Skip behavior if no data could be loaded
        if not data_loaded_for_behavior:
            logging.warning(
                f"Skipping behavior {behavior_id} as no valid result files found or loaded."
            )
            continue

        # --- HarmBench Evaluation --- #
        if not test_cases:
            logging.warning(f"No test cases found for {behavior_dir}. Skipping.")
            continue

        logging.info(
            f"Running HarmBench for {len(test_cases)} test cases from behavior {behavior_id}..."
        )
        try:
            behavior_harmbench_results = evaluate_with_harmbench(
                behavior_id=behavior_id,
                test_cases=test_cases,
                output_dir=behavior_dir,
                harmbench_judge=harmbench_judge,
                harmbench_tokenizer=harmbench_tokenizer,
                behavior=behavior,
                context=context,
            )

            # Add row number and behavior to results for summary
            behavior_harmbench_results["row_number"] = row_number
            behavior_harmbench_results["behavior"] = behavior

            # Cache the results
            save_harmbench_results(behavior_dir, behavior_harmbench_results)

            all_harmbench_results.append(behavior_harmbench_results)

            # Collect successful prompts (using HarmBench scores now)
            for i, harmbench_score in enumerate(
                behavior_harmbench_results["harmbench_scores"]
            ):
                if harmbench_score == 1:
                    # Ensure index is valid for test_cases
                    if i < len(test_cases):
                        successful_prompts.append(
                            {
                                "behavior_id": behavior_id,
                                "row_number": row_number,
                                "behavior": behavior,
                                "prompt": test_cases[i]["prompt"],
                                "response": test_cases[i].get(
                                    "response", "N/A"
                                ),  # Add response if available
                                "inner_score": behavior_harmbench_results[
                                    "inner_scores"
                                ][i],
                                "harmbench_score": harmbench_score,
                            }
                        )
                    else:
                        logging.warning(
                            f"Index {i} out of range for test_cases length {len(test_cases)} when collecting successful prompts for {behavior_id}."
                        )

        except Exception as e:
            logging.error(
                f"Error during HarmBench evaluation for {behavior_dir.name}: {e}"
            )
            logging.error(traceback.format_exc())  # Print full traceback

    # Free up memory
    del harmbench_judge, harmbench_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Calculate overall metrics
    if all_harmbench_results:
        overall_max_score = max(r["max_score"] for r in all_harmbench_results)
        all_inner_scores = []
        all_harmbench_scores = []

        for r in all_harmbench_results:
            all_inner_scores.extend(r["inner_scores"])
            all_harmbench_scores.extend(r["harmbench_scores"])

        correlation = adjusted_corrcoef(all_inner_scores, all_harmbench_scores)

        # Calculate ASR (Attack Success Rate)
        total_behaviors = len(all_harmbench_results)
        successful_behaviors = sum(
            1 for r in all_harmbench_results if r["max_score"] >= 1.0
        )
        asr = successful_behaviors / total_behaviors if total_behaviors > 0 else 0

        # Sort behaviors by row number for consistent output and ensure uniqueness
        # Use a dictionary to ensure each behavior ID and row combination is only included once
        behavior_dict = {}
        for result in all_harmbench_results:
            key = f"{result['behavior_id']}_{result['row_number']}"
            behavior_dict[key] = result

        # Convert back to a list and sort
        sorted_behaviors = sorted(behavior_dict.values(), key=lambda x: x["row_number"])

        summary = {
            "overall_max_score": overall_max_score,
            "correlation": float(correlation),
            "attack_success_rate": float(asr),
            "total_behaviors": total_behaviors,
            "successful_behaviors": successful_behaviors,
            "evaluated_rows": sorted(
                list(set([r["row_number"] for r in all_harmbench_results]))
            ),
            "per_behavior": {
                f"{r['behavior_id']} (row: {r['row_number']})": {
                    "max_score": r["max_score"],
                    "inner_scores": r["inner_scores"],
                    "harmbench_scores": r["harmbench_scores"],
                    "row_number": r["row_number"],
                }
                for r in sorted_behaviors
            },
            "timestamp": datetime.datetime.now().isoformat(),
        }

        # Save results
        summary_dir = results_dir
        with open(summary_dir / "harmbench_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(summary_dir / "successful_prompts.json", "w") as f:
            json.dump(successful_prompts, f, indent=2)

        # Log results
        logging.info(f"Overall Max Score: {overall_max_score:.4f}")
        logging.info(f"Score Correlation: {correlation:.4f}")
        logging.info(
            f"Attack Success Rate: {asr:.4f} ({successful_behaviors}/{total_behaviors})"
        )
        logging.info(f"Evaluated row numbers: {summary['evaluated_rows']}")
        logging.info(f"Found {len(successful_prompts)} successful prompts")

        # Log only unique behavior results
        for result in sorted_behaviors:
            logging.info(
                f"Behavior {result['behavior_id']} (row: {result['row_number']}) - Max score: {result['max_score']:.4f}"
            )

        return summary

    return None


def main(args):
    """Main function that processes command line arguments and runs evaluation."""
    results_dir = Path(args.results_dir) if hasattr(args, "results_dir") else None

    if results_dir is None:
        raise ValueError("results_dir must be provided")

    if not results_dir.exists():
        raise ValueError(f"Results directory does not exist: {results_dir}")

    return evaluate_results_folder(results_dir)


if __name__ == "__main__":
    # Add the parent directory to Python path to make imports work
    import sys

    root_dir = str(Path(__file__).resolve().parent.parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)

    parser = argparse.ArgumentParser(
        description="Evaluate attack results with HarmBench"
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory containing attack results to evaluate",
    )
    args = parser.parse_args()
    main(args)
