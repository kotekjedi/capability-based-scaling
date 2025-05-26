import json
import logging
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import torch

# Set up logging
logger = logging.getLogger(__name__)


class CrescendoTargetLM(ABC):
    """Base class for Crescendo target language models."""

    def __init__(
        self,
        model: Dict,
        max_n_tokens: int,
        use_system_prompt: bool = True,
    ):
        """Initialize Crescendo target language model.

        Args:
            model: Dict containing model configuration
            max_n_tokens: Max tokens for target response
            use_system_prompt: Whether to use system prompt for the target
        """
        self.model_config = model
        if "model" in model and "tokenizer" in model:
            self.model = model["model"]
            self.tokenizer = model["tokenizer"]
        self.max_n_tokens = max_n_tokens
        self.use_system_prompt = use_system_prompt

        # Initialize logs list
        self.logs = []

    def save_logs(self, output_dir):
        """Save logs to a file"""
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "target_lm.json"), "w") as f:
            json.dump(self.logs, f, indent=2)

    @abstractmethod
    def generate_response(self, messages: List[Dict], log_callback=None) -> str:
        """Generate response from target model based on message history.

        Args:
            messages: List of message dictionaries
            log_callback: Optional callback function for logging model interactions

        Returns:
            Generated response text
        """
        pass

    @abstractmethod
    def generate_response_batch(
        self, messages_batch: List[List[Dict]], log_callback=None
    ) -> List[str]:
        """Generate responses for a batch of message histories.

        Args:
            messages_batch: List where each element is a list of message dictionaries for one conversation.
            log_callback: Optional callback function for logging model interactions.

        Returns:
            List of generated response texts.
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
        # Check if tokenizer is available
        if not hasattr(self, "tokenizer"):
            logger.warning(
                "No tokenizer available for truncation, returning messages as is"
            )
            return messages

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

        # Ensure we have pairs of messages (user-assistant) and end with user
        # This is what makes target LM different - we want to end with a user message
        # that the model will respond to
        if len(truncated_messages) % 2 == 0:
            truncated_messages = truncated_messages[1:]  # Remove oldest message

        # Reinsert system message at the beginning
        if system_message:
            truncated_messages.insert(0, system_message)

        # Final check: Ensure last message is 'user'
        if truncated_messages and truncated_messages[-1]["role"] != "user":
            logger.warning(
                f"Post-truncation fix: Last message must be 'user', got '{truncated_messages[-1]['role']}'"
            )
            # Remove the last message if it's not a user message
            truncated_messages = truncated_messages[:-1]

        # Print final roles sequence
        roles = [msg["role"] for msg in truncated_messages]
        logger.info(f"After truncation - Roles sequence: {roles}")

        return truncated_messages


class HFCrescendoTargetLM(CrescendoTargetLM):
    """Hugging Face implementation of Crescendo target language model."""

    def generate_response(self, messages: List[Dict], log_callback=None) -> str:
        """Generate response from target model based on message history.

        Args:
            messages: List of message dictionaries
            log_callback: Optional callback function for logging model interactions

        Returns:
            Generated response text
        """
        # Check if tokenizing the messages would exceed the model's capacity
        all_content = " ".join([msg["content"] for msg in messages])
        token_count = len(self.tokenizer.encode(all_content, add_special_tokens=True))

        # If we would exceed the model's capacity, return an overflow error message
        if token_count > self.model.config.max_position_embeddings - 512:
            overflow_msg = f"TOKEN_OVERFLOW: Context length ({token_count}) exceeds model maximum ({self.model.config.max_position_embeddings})"
            logger.warning(overflow_msg)
            print(f"TARGET LM ERROR: {overflow_msg}")

            # Log the overflow
            log_entry = {
                "input_text": all_content[:1000] + "...",  # Truncate for logging
                "output_text": overflow_msg,
                "error": "token_overflow",
                "token_count": token_count,
                "max_tokens": self.model.config.max_position_embeddings,
                "type": "generate_response_error",
            }
            self.logs.append(log_entry)

            if log_callback:
                log_callback(
                    model_name="target_model",
                    input_ids=None,
                    output_ids=None,
                    is_api=False,
                    error="token_overflow",
                )

            return overflow_msg
        else:
            print(
                f"TARGET LM: Token count: {token_count}; Max tokens: {self.model.config.max_position_embeddings - 512}"
            )

        # If we're within limits, proceed with generation
        truncated_messages = messages  # No truncation, use original messages

        # Apply chat template and prepare for generation
        inputs = self.tokenizer.apply_chat_template(
            truncated_messages, return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.max_n_tokens,
                temperature=0.7,
                do_sample=True,
            )

        # Decode and get only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs.shape[1] :], skip_special_tokens=True
        )

        # Log the interaction
        log_entry = {
            "input_text": self.tokenizer.decode(inputs[0], skip_special_tokens=True),
            "output_text": generated_text,
            "input_tokens": inputs[0].tolist(),
            "output_tokens": outputs[0][inputs.shape[1] :].tolist(),
            "type": "generate_response",
        }
        self.logs.append(log_entry)

        if log_callback:
            log_callback(
                model_name="target_model",
                input_ids=inputs,
                output_ids=outputs,
                is_api=False,
            )
        print("TARGET LM RESPONSE: ", generated_text)
        return generated_text

    def generate_response_batch(
        self, messages_batch: List[List[Dict]], log_callback=None
    ) -> List[str]:
        """Generate responses for a batch of message histories.

        Args:
            messages_batch: List where each element is a list of message dictionaries for one conversation.
            log_callback: Optional callback function for logging model interactions.

        Returns:
            List of generated response texts.
        """
        batch_size = len(messages_batch)
        if not batch_size:
            return []

        print(f"TARGET LM BATCH: Generating responses for batch size {batch_size}")

        processed_inputs = []
        input_texts = []
        overflow_indices = set()
        results = ["" for _ in range(batch_size)]
        max_model_len = self.model.config.max_position_embeddings

        # Pre-process: Check for overflow and apply chat template
        for i, messages in enumerate(messages_batch):
            all_content = " ".join([msg["content"] for msg in messages])
            token_count = len(
                self.tokenizer.encode(all_content, add_special_tokens=True)
            )

            # If we would exceed the model's capacity, mark as overflow
            if token_count > max_model_len - self.max_n_tokens:
                overflow_msg = f"TOKEN_OVERFLOW: Context length ({token_count}) exceeds model maximum ({max_model_len})"
                logger.warning(overflow_msg)
                print(f"TARGET LM BATCH ERROR (idx {i}): {overflow_msg}")
                overflow_indices.add(i)
                results[i] = overflow_msg
                # Log the overflow
                log_entry = {
                    "input_text": all_content[:1000] + "...",
                    "output_text": overflow_msg,
                    "error": "token_overflow",
                    "token_count": token_count,
                    "max_tokens": max_model_len,
                    "type": "generate_response_batch_error",
                    "batch_index": i,
                }
                self.logs.append(log_entry)
                if log_callback:
                    log_callback(
                        model_name="target_model",
                        input_ids=None,
                        output_ids=None,
                        is_api=False,
                        error="token_overflow",
                        batch_index=i,
                    )
            else:
                # Apply chat template (do not tokenize yet)
                input_text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                input_texts.append(input_text)
                processed_inputs.append((i, input_text))  # Keep track of original index

        if not processed_inputs:
            return results  # All inputs overflowed

        # Batch tokenization for non-overflowed inputs
        original_indices, valid_input_texts = zip(*processed_inputs)
        self.tokenizer.padding_side = "left"  # For generation
        inputs = self.tokenizer(
            list(valid_input_texts),
            return_tensors="pt",
            padding=True,
            max_length=max_model_len - self.max_n_tokens,
        ).to(self.model.device)

        # Batch generation
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_n_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode and assign results back to the original batch indices
        generated_texts = self.tokenizer.batch_decode(
            outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        for j, gen_text in enumerate(generated_texts):
            original_idx = original_indices[j]
            results[original_idx] = gen_text
            print(f"TARGET LM BATCH RESPONSE (idx {original_idx}): {gen_text[:100]}...")

            # Log the interaction
            log_entry = {
                "input_text": valid_input_texts[j],
                "output_text": gen_text,
                "input_tokens": inputs["input_ids"][j].tolist(),
                "output_tokens": outputs[j][inputs["input_ids"].shape[1] :].tolist(),
                "type": "generate_response_batch",
                "batch_index": original_idx,
            }
            self.logs.append(log_entry)

            if log_callback:
                log_callback(
                    model_name="target_model",
                    input_ids=inputs["input_ids"][j],
                    output_ids=outputs[j],
                    is_api=False,
                    batch_index=original_idx,
                )

        return results
