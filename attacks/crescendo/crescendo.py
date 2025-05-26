import copy
import datetime
import json
import logging
import os
import traceback  # Added for error logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from tqdm import tqdm

from ..baseline import SingleBehaviorRedTeamingMethod
from ..utils.model_utils import LLAMA2_DEFAULT_SYSTEM_PROMPT
from .attacker_lm import HFCrescendoAttackerLM
from .judge_lm import HFCrescendoJudgeLM
from .target_lm import HFCrescendoTargetLM

# Set up logging
logger = logging.getLogger(__name__)


class Crescendo(SingleBehaviorRedTeamingMethod):
    """
    Implementation of the Crescendo multi-turn jailbreak attack.

    Crescendo progressively guides a model to generate harmful content in small,
    seemingly benign steps, exploiting the LLM's tendency to follow patterns and
    focus on recent text. Supports batch processing of conversations.
    """

    def __init__(
        self,
        attack_model: Dict,
        target_model: Dict,
        judge_model: Dict,
        max_rounds: int = 10,
        use_system_prompt: bool = True,
        **kwargs,
    ):
        # Initialize the attacker model
        self.attacker_lm = HFCrescendoAttackerLM(
            model=attack_model,
            max_n_tokens=attack_model.get("max_tokens", 512),
            use_system_prompt=use_system_prompt,
        )

        # Initialize the target model
        self.target_lm = HFCrescendoTargetLM(
            model=target_model,
            max_n_tokens=target_model.get("max_tokens", 512),
            use_system_prompt=use_system_prompt,
        )

        # Initialize the judge model
        self.judge_lm = HFCrescendoJudgeLM(
            model=judge_model,
            max_n_tokens=judge_model.get("max_tokens", 512),
            use_system_prompt=use_system_prompt,
        )

        self.max_rounds = max_rounds
        self.use_system_prompt = use_system_prompt

        # Store model configurations
        self.attack_model_config = attack_model
        self.target_model_config = target_model
        self.judge_model_config = judge_model

        # Initialize detailed logging
        self.detailed_logs = []

    def _log_model_interaction(
        self,
        model_name: str,
        input_ids: Union[torch.Tensor, str],
        output_ids: Union[torch.Tensor, str],
        is_api: bool = False,
        batch_index: Optional[int] = None,  # Add batch_index for logging
        **kwargs,
    ):
        """
        Log detailed model interactions with token information.

        Args:
            model_name: Name of the model ("attack_model", "target_model", or "judge_model")
            input_ids: Input token IDs tensor or raw text for API
            output_ids: Output token IDs tensor or raw text for API
            is_api: Whether this is an API model interaction
            batch_index: Index of the conversation within the batch (if applicable)
            **kwargs: Additional data to log
        """
        log_entry = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model_name,
            "batch_index": batch_index,  # Log batch_index
        }

        if is_api:
            # ... (API logging remains similar, maybe add batch_index if needed)
            pass
        else:
            # Assuming HF models for now
            input_token_count = (
                input_ids.shape[-1] if hasattr(input_ids, "shape") else 0
            )
            output_token_count = (
                output_ids.shape[-1] - input_ids.shape[-1]
                if hasattr(output_ids, "shape")
                and hasattr(input_ids, "shape")
                and output_ids.shape[-1] > input_ids.shape[-1]
                else 0  # Handle cases where output might be shorter (e.g., generation error)
            )

            log_entry.update(
                {
                    # Optionally decode tokens if needed, but can be large
                    # "input_text": self.tokenizer.decode(input_ids.squeeze()),
                    # "output_text": self.tokenizer.decode(output_ids.squeeze()[input_token_count:]),
                    "input_token_count": input_token_count,
                    "output_token_count": output_token_count,
                    "is_api": False,
                }
            )

        # Add any additional info
        log_entry.update(kwargs)

        # Append to the detailed logs
        self.detailed_logs.append(log_entry)

    def generate_test_cases_single_behavior(
        self,
        behavior_dict: Dict,
        output_dir: Optional[str] = None,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[List[Dict], Dict, Dict]:  # Return list of conversations
        """
        Generates test cases for a single behavior using the Crescendo attack,
        supporting batch processing of multiple conversations in parallel.

        Args:
            behavior_dict: Dictionary containing behavior details.
            output_dir: Directory to save logs and results.
            verbose: Whether to print progress and intermediate results.

        Returns:
            Tuple containing:
            - List of conversation histories (one dict per conversation in the batch).
            - Dictionary of detailed logs (currently combined for the batch).
            - Dictionary containing the generated criteria.
        """
        # --- Setup ---
        # Determine number of streams (batch size) - Get from kwargs or default to 1
        n_streams = kwargs.get("n_streams", 1)

        if "Behavior" in behavior_dict:
            behavior = behavior_dict["Behavior"]
            behavior_id = behavior_dict["BehaviorID"]
            context_str = behavior_dict.get("ContextString", "")
            goal = behavior
            if context_str:
                goal = f"{context_str}\n\n---\n\n{behavior}"
        else:
            goal = behavior_dict.get(
                "jailbreak_prompt", behavior_dict.get("prompt", "")
            )
            behavior_id = behavior_dict.get("BehaviorID", "unknown")

        success_condition = behavior_dict.get("success_condition", None)

        # Setup output directory
        logs_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            logs_path = os.path.join(output_dir, f"logs_{behavior_id}.jsonl")
            with open(logs_path, "w") as f:
                f.write("")  # Clear/create log file

        if verbose:
            print(
                f"Starting Crescendo attack for behavior {behavior_id} (Streams: {n_streams})"
            )
            print(f"Goal: {goal}")

        # Generate scoring rubric (once per behavior)
        criteria = self.judge_lm.generate_score_rubric(
            goal, success_condition, log_callback=self._log_model_interaction
        )
        criteria_dict = {"criteria": criteria}
        if output_dir:
            with open(
                os.path.join(output_dir, f"criteria_{behavior_id}.json"), "w"
            ) as f:
                json.dump(criteria_dict, f, indent=2)

        # --- Initialize Batch States ---
        batch_histories_a = [
            {"round_number": [], "attacker": [], "target": [], "score": []}
            for _ in range(n_streams)  # Use n_streams
        ]
        if self.use_system_prompt:
            batch_histories_t = [
                [{"role": "system", "content": LLAMA2_DEFAULT_SYSTEM_PROMPT}]
                for _ in range(n_streams)  # Use n_streams
            ]
        else:
            batch_histories_t = [[] for _ in range(n_streams)]  # Use n_streams

        batch_conversation_logs = [
            [] for _ in range(n_streams)
        ]  # Store individual conv logs
        batch_last_responses = [""] * n_streams  # Use n_streams
        batch_prompts = [""] * n_streams  # Store current prompt for each item
        batch_refusal_counters = [0] * n_streams  # Use n_streams
        active_indices = list(
            range(n_streams)
        )  # Indices of conversations still running
        round_numbers = [1] * n_streams  # Track round per conversation

        # --- Main Conversation Loop ---
        pbar = tqdm(
            total=self.max_rounds * n_streams,
            desc=f"Crescendo Rounds ({behavior_id})",
            disable=not verbose,
        )

        while any(r <= self.max_rounds for r in round_numbers) and active_indices:
            current_round_active_indices = [
                idx for idx in active_indices if round_numbers[idx] <= self.max_rounds
            ]
            if not current_round_active_indices:
                break  # No more active conversations for this round or max_rounds reached

            # Prepare inputs for the active batch
            attacker_inputs = {
                "round_numbers": [
                    round_numbers[i] for i in current_round_active_indices
                ],
                "tasks": [goal] * len(current_round_active_indices),
                "histories_a": [
                    batch_histories_a[i] for i in current_round_active_indices
                ],
                "last_responses": [
                    batch_last_responses[i] for i in current_round_active_indices
                ],
            }

            try:
                # 1. Generate attacks
                attack_data_batch = self.attacker_lm.generate_attack_batch(
                    **attacker_inputs, log_callback=self._log_model_interaction
                )

                # Store prompts generated by the attacker
                prompts_for_batch = []
                response_summaries_for_batch = []
                for i, attack_data in enumerate(attack_data_batch):
                    idx = current_round_active_indices[i]
                    prompt = attack_data.get(
                        "generatedQuestion", f"Tell me more about {goal}?"
                    )
                    response_summary = attack_data.get(
                        "lastResponseSummary", batch_last_responses[idx][:100] + "..."
                    )
                    batch_prompts[idx] = prompt
                    prompts_for_batch.append(prompt)
                    response_summaries_for_batch.append(response_summary)
                    batch_histories_t[idx].append({"role": "user", "content": prompt})

                # Prepare target inputs
                target_inputs = {
                    "messages_batch": [
                        batch_histories_t[i] for i in current_round_active_indices
                    ],
                }

                # 2. Generate target responses
                responses_batch = self.target_lm.generate_response_batch(
                    **target_inputs, log_callback=self._log_model_interaction
                )

                # Prepare judge inputs
                refusal_prompts = []
                refusal_responses = []
                eval_responses = []
                indices_for_eval = []  # Track indices that didn't overflow for eval/refusal

                for i, response in enumerate(responses_batch):
                    idx = current_round_active_indices[i]
                    batch_last_responses[idx] = (
                        response  # Update last response for next attacker round
                    )

                    if response.startswith("TOKEN_OVERFLOW:"):
                        if verbose:
                            print(
                                f"Batch {idx}, Round {round_numbers[idx]}: Token overflow, terminating."
                            )
                        round_log = {
                            "round": round_numbers[idx],
                            "user": batch_prompts[idx],
                            "assistant": response,
                            "score": "overflow",
                            "refusal": False,
                            "overflow": True,
                        }
                        batch_conversation_logs[idx].append(round_log)
                        # Mark as inactive by removing from active_indices
                        if idx in active_indices:
                            active_indices.remove(idx)
                        pbar.update(
                            self.max_rounds - round_numbers[idx] + 1
                        )  # Fast-forward pbar
                        round_numbers[idx] = self.max_rounds + 1  # Mark as finished
                    else:
                        refusal_prompts.append(batch_prompts[idx])
                        refusal_responses.append(response)
                        eval_responses.append(response)
                        indices_for_eval.append(idx)

                if not indices_for_eval:  # All overflowed or finished
                    continue

                # 3. Check refusals (only for non-overflowed)
                refusal_check_results = self.judge_lm.check_refusal_batch(
                    prompts=refusal_prompts,
                    responses=refusal_responses,
                    log_callback=self._log_model_interaction,
                )

                # Safeguard: Ensure refusal_check_results is a list
                if not isinstance(refusal_check_results, list):
                    logger.error(
                        f"JudgeLM.check_refusal_batch returned type {type(refusal_check_results)} instead of list. Result: {refusal_check_results}"
                    )
                    # Handle error appropriately - maybe treat all as non-refusals or stop
                    print(
                        "ERROR: Judge refusal check failed unexpectedly. Stopping batch."
                    )
                    break  # Stop the batch

                # 4. Evaluate responses (only for non-overflowed and non-refused)
                eval_goals = [goal] * len(indices_for_eval)
                eval_criteria = [criteria] * len(indices_for_eval)
                scores_batch = self.judge_lm.evaluate_response_batch(
                    responses=eval_responses,
                    goals=eval_goals,
                    criteria=eval_criteria,
                    log_callback=self._log_model_interaction,
                )

                # --- Process Batch Results ---
                eval_idx_counter = 0
                refusal_idx_counter = 0
                temp_active_indices = list(active_indices)  # Iterate over a copy

                for i, idx in enumerate(current_round_active_indices):
                    if (
                        idx not in indices_for_eval
                    ):  # Skip overflowed ones handled earlier
                        continue

                    current_prompt = batch_prompts[idx]
                    current_response = batch_last_responses[idx]  # Already updated
                    is_refused = refusal_check_results[refusal_idx_counter]
                    current_round = round_numbers[idx]
                    refusal_idx_counter += 1

                    history_a = batch_histories_a[idx]
                    history_t = batch_histories_t[idx]

                    # Save attacker prompt to history_a
                    history_a["attacker"].append(current_prompt)
                    # if current_round > 1: <<< REMOVING THIS CONDITION
                    # Find the corresponding response summary generated by attacker earlier
                    # This assumes attack_data_batch order matches current_round_active_indices
                    attacker_data_index = current_round_active_indices.index(idx)
                    summary = attack_data_batch[attacker_data_index].get(
                        "lastResponseSummary", "Summary N/A"
                    )
                    if summary is None or not isinstance(summary, str):
                        summary = "Summary N/A"
                    history_a["target"].append("(Summary) " + summary)
                    history_a["round_number"].append(current_round)

                    if is_refused and batch_refusal_counters[idx] < 5:
                        # --- DEBUG LOGGING START ---
                        print(
                            f"--> DEBUG: Batch {idx}, Round {current_round}: REFUSAL DETECTED. Counter = {batch_refusal_counters[idx]}. BACKTRACKING."
                        )
                        # --- DEBUG LOGGING END ---
                        if verbose:
                            print(
                                f"Batch {idx}, Round {current_round}: Refusal detected, backtracking."
                            )
                        score = "refused"
                        history_a["score"].append(score)
                        history_t.pop()  # Remove the last user message that caused refusal
                        batch_refusal_counters[idx] += 1

                        round_log = {
                            "round": current_round,
                            "user": current_prompt,
                            "assistant": current_response,
                            "score": score,
                            "refusal": True,
                            "overflow": False,
                        }
                        batch_conversation_logs[idx].append(round_log)
                        # Don't remove from active indices, just move to next round
                        round_numbers[idx] += 1
                        pbar.update(1)

                    else:  # Not refused (or too many refusals)
                        # --- DEBUG LOGGING START ---
                        if is_refused:
                            print(
                                f"--> DEBUG: Batch {idx}, Round {current_round}: REFUSAL DETECTED. Counter = {batch_refusal_counters[idx]}. LIMIT REACHED or exceeded. CONTINUING WITHOUT BACKTRACKING."
                            )
                        else:
                            print(
                                f"--> DEBUG: Batch {idx}, Round {current_round}: Not refused. CONTINUING."
                            )
                        # --- DEBUG LOGGING END ---
                        # Evaluate score
                        score = scores_batch[eval_idx_counter]
                        eval_idx_counter += 1

                        # Add response to target history
                        history_t.append(
                            {"role": "assistant", "content": current_response}
                        )
                        history_a["score"].append(
                            score
                        )  # Add score regardless of refusal limit

                        if is_refused:  # Log refusal even if continuing
                            if verbose:
                                print(
                                    f"Batch {idx}, Round {current_round}: Refusal limit reached, continuing with score {score}."
                                )

                        round_log = {
                            "round": current_round,
                            "user": current_prompt,
                            "assistant": current_response,
                            "score": score,
                            "refusal": is_refused,
                            "overflow": False,
                        }
                        batch_conversation_logs[idx].append(round_log)

                        if verbose:
                            print(
                                f"Batch {idx}, Round {current_round} - Score: {score}"
                            )

                        # Check if conversation is finished for this index
                        round_numbers[idx] += 1
                        pbar.update(1)
                        if (
                            round_numbers[idx] > self.max_rounds
                            and idx in active_indices
                        ):
                            active_indices.remove(idx)

            except Exception as e:
                logger.error(f"Error during Crescendo batch processing: {e}")
                logger.error(traceback.format_exc())
                # Potentially mark all active conversations in this batch as failed?
                # For now, just log and continue if possible, or break
                print(f"ERROR in Crescendo round: {e}. See logs. Stopping batch.")
                raise e

        pbar.close()

        # --- Finalize and Save ---
        all_final_results = []
        if output_dir:
            # Save logs from each model (appends logs from all batches)
            self.attacker_lm.save_logs(output_dir)
            self.target_lm.save_logs(output_dir)
            self.judge_lm.save_logs(output_dir)

            # Save individual conversation logs
            with open(logs_path, "a") as f:
                for i in range(n_streams):
                    conv_log = {
                        "behavior_id": behavior_id,
                        "batch_index": i,
                        "conversation": batch_conversation_logs[i],
                    }
                    f.write(json.dumps(conv_log) + "\n")

        # Create final output structure (list of results, one per initial batch item)
        for i in range(n_streams):
            final_result = {
                "behavior_id": behavior_id,
                "batch_index": i,
                "goal": goal,
                "rounds_completed": round_numbers[i] - 1
                if round_numbers[i] > self.max_rounds
                else self.max_rounds,
                "conversation_history": batch_conversation_logs[i],
                "criteria": criteria,  # Criteria is the same for the whole batch
            }
            all_final_results.append(final_result)

        # Save aggregated final results
        if output_dir:
            final_results_path = os.path.join(
                output_dir, f"final_results_{behavior_id}.json"
            )
            with open(final_results_path, "w") as f:
                json.dump(all_final_results, f, indent=2)

        # Return list of conversations, detailed logs (combined), criteria dict
        # return all_final_results, self.detailed_logs, criteria_dict
        # Return structure similar to PAIR: list of stream logs, detailed flat logs, criteria
        return batch_conversation_logs, self.detailed_logs, criteria_dict
