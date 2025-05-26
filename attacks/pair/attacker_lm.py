import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch

from .common import extract_json

# Set up logging
logger = logging.getLogger(__name__)


class AttackerLM(ABC):
    """Base class for attacker language models."""

    def __init__(
        self,
        model: Dict,
        max_n_tokens: int,
        use_system_prompt: bool = True,
        batch_size: int = 8,
        keep_last_n: int = 5,
    ):
        """Initialize attacker language model.

        Args:
            model: Dict containing model configuration
            max_n_tokens: Max tokens for attack response
            use_system_prompt: Whether to use system prompt
            batch_size: Batch size for generation
            keep_last_n: Number of conversation turns to keep
        """
        self.model_config = model
        if "model" in model and "tokenizer" in model:
            self.model = model["model"]
            self.tokenizer = model["tokenizer"]
        self.max_n_tokens = max_n_tokens
        self.use_system_prompt = use_system_prompt
        self.batch_size = batch_size
        self.keep_last_n = keep_last_n

    @abstractmethod
    def generate_attack(
        self, messages_list: List[List[Dict]], log_callback=None
    ) -> List[Optional[Dict]]:
        """Generate attack prompts for multiple message histories.

        Args:
            messages_list: List of message lists to generate attacks for
            log_callback: Optional callback function for logging model interactions

        Returns:
            List of attack dictionaries (or None for failed generations)
        """
        pass

    def _truncate_conversation(
        self, messages: List[Dict], max_tokens: int
    ) -> List[Dict]:
        """
        Truncate the conversation to fit within the maximum token limit.

        Args:
            messages: List of message dictionaries containing 'role' and 'content'.
            max_tokens: Maximum number of tokens allowed in the conversation.

        Returns:
            Truncated list of messages.
        """
        # Print current roles sequence
        roles = [msg["role"] for msg in messages]
        logger.info(f"Before truncation - Roles sequence: {roles}")

        # First, check if truncation is even necessary by calculating total tokens
        original_length = len(messages)
        all_content = " ".join([msg["content"] for msg in messages])
        total_tokens = len(self.tokenizer.encode(all_content, add_special_tokens=False))

        logger.info(
            f"Total tokens in conversation: {total_tokens}, Max allowed: {max_tokens}"
        )

        # If we're under the token limit, no truncation needed
        if total_tokens <= max_tokens:
            logger.info("No token truncation needed - under the limit")
            return messages

        # Otherwise, we need to truncate
        truncated_messages = []
        current_tokens = 0

        # Preserve system message if present
        system_message = (
            messages[0] if messages and messages[0]["role"] == "system" else None
        )
        messages_without_system = messages[1:] if system_message else messages

        # Reverse iterate to keep the most recent messages
        for message in reversed(messages_without_system):
            # Tokenize the message content
            tokens = self.tokenizer.encode(message["content"], add_special_tokens=False)

            if current_tokens + len(tokens) > max_tokens:
                logger.info(
                    f"Token limit would be exceeded: {current_tokens + len(tokens)} > {max_tokens}"
                )
                break

            current_tokens += len(tokens)
            # Prepend the message to keep the order
            truncated_messages.insert(0, message)

        # Ensure we have pairs of messages (user-assistant) and start with user
        if len(truncated_messages) % 2 != 0 or (
            truncated_messages and truncated_messages[0]["role"] != "user"
        ):
            truncated_messages = truncated_messages[
                1:
            ]  # Remove oldest message to maintain pairs and start with user

        # Reinsert system message at the beginning
        if system_message:
            truncated_messages.insert(0, system_message)

        # Final check: Ensure first non-system message is 'user'
        if (
            len(truncated_messages) > 1
            and truncated_messages[0]["role"] == "system"
            and truncated_messages[1]["role"] != "user"
        ):
            # If first non-system message is not 'user', remove it to maintain alternation
            logger.warning(
                f"Post-truncation fix: First non-system message must be 'user', got '{truncated_messages[1]['role']}'"
            )
            truncated_messages = (
                [truncated_messages[0]] + truncated_messages[2:]
                if len(truncated_messages) > 2
                else [truncated_messages[0]]
            )

        # Print final roles sequence
        roles = [msg["role"] for msg in truncated_messages]
        logger.info(f"After truncation - Roles sequence: {roles}")

        # assert that there is a system message and it is the first message
        assert (
            truncated_messages[0]["role"] == "system"
            if truncated_messages and truncated_messages[0]["role"] == "system"
            else True
        ), "System message is missing"

        # Log truncation reason
        if len(truncated_messages) != original_length:
            logger.info("Truncation occurred due to exceeding token limit.")
        else:
            logger.info("No truncation was necessary.")

        return truncated_messages

    def _truncate_conversation_by_turns(
        self, messages: List[Dict], n_turns: int
    ) -> List[Dict]:
        """
        Truncate the conversation to keep the last N turns while maintaining balanced user-assistant turns.

        Args:
            messages: List of message dictionaries containing 'role' and 'content'.
            n_turns: Number of turns to keep (each turn is a user-assistant pair).

        Returns:
            Truncated list of messages that maintains user-assistant balance.
        """
        # Print current roles sequence
        roles = [msg["role"] for msg in messages]
        logger.info(f"Before turn truncation - Roles sequence: {roles}")

        original_length = len(messages)

        # Preserve system message if present
        system_message = (
            messages[0] if messages and messages[0]["role"] == "system" else None
        )
        messages_without_system = messages[1:] if system_message else messages

        # Calculate how many message pairs we need
        # n_turns is already the number of turns to keep, don't divide by 2
        n_pairs = n_turns

        # Calculate where to start truncating from
        total_pairs = len(messages_without_system) // 2

        logger.info(f"Total pairs: {total_pairs}, Pairs to keep: {n_pairs}")

        # If we already have fewer pairs than we want to keep, no truncation needed
        if total_pairs <= n_pairs:
            logger.info("No turn truncation needed - under the limit")
            return messages

        pairs_to_keep = min(n_pairs, total_pairs)

        # Keep the last n pairs
        result = messages_without_system[-(2 * pairs_to_keep) :]

        # Ensure the result starts with a user message
        if result and result[0]["role"] != "user":
            result = result[1:]  # Remove the first message if it's not a user message

        # Reinsert system message at the beginning
        if system_message:
            result.insert(0, system_message)

        # Print final roles sequence
        roles = [msg["role"] for msg in result]
        logger.info(f"After turn truncation - Roles sequence: {roles}")

        # Log truncation reason
        if len(result) != original_length:
            logger.info("Truncation occurred due to exceeding turn limit.")
        else:
            logger.info("No truncation was necessary.")

        return result

    def truncate_messages(self, messages: List[Dict], max_tokens: int) -> List[Dict]:
        """
        Truncate conversation first by token count, then by number of turns.

        Args:
            messages: List of message dictionaries
            max_tokens: Maximum tokens allowed

        Returns:
            Truncated list of messages
        """
        # First truncate by token count
        messages = self._truncate_conversation(messages, max_tokens)
        # Then truncate by number of turns
        messages = self._truncate_conversation_by_turns(messages, self.keep_last_n)
        return messages


class HFAttackerLM(AttackerLM):
    """Attacker LM that uses HuggingFace models."""

    def generate_attack(
        self, messages_list: List[List[Dict]], log_callback=None
    ) -> List[Optional[Dict]]:
        """Generate attack prompts for multiple message histories."""
        all_responses = []

        for batch_start in range(0, len(messages_list), self.batch_size):
            batch_messages = messages_list[batch_start : batch_start + self.batch_size]
            batch_prompts = []
            for messages in batch_messages:
                if len(messages) <= 2:  # Only for initial message
                    init_message = ' {"improvement": "","prompt": "'
                else:
                    init_message = ' {"improvement": "'

                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt += init_message
                batch_prompts.append(prompt)

            inputs = self.tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_n_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[inputs["input_ids"][i].shape[0] :], skip_special_tokens=True
                )

                if log_callback:
                    log_callback(
                        "attack_model",
                        inputs["input_ids"][i],
                        output[inputs["input_ids"][i].shape[0] :],
                    )

                if not response.strip():
                    all_responses.append(None)
                    continue

                if not response.endswith("}"):
                    response += "}"

                attack_dict, _ = extract_json(init_message + response)
                # if (
                #     attack_dict is None
                # ):  # accounts for cases where model fails at creating a valid json
                #     attack_dict = {
                #         "improvement": "I attempt a direct query.",
                #         "prompt": self.target_behavior,
                #     }
                all_responses.append(attack_dict)

        return all_responses
