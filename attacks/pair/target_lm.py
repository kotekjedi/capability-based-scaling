import datetime
import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

import requests
import torch
from openai import OpenAI

from ..utils.model_utils import (
    LLAMA2_DEFAULT_SYSTEM_PROMPT,
    configure_urial_chat_template,
    load_urial_prompt,
)


class TargetLM(ABC):
    """Base class for target language models."""

    def __init__(
        self,
        model: Dict,
        max_n_tokens: int,
        use_system_prompt: bool = True,
        batch_size: int = 8,
    ):
        """Initialize target language model.

        Args:
            model: Dict containing model configuration
            max_n_tokens: Max tokens for target response
            use_system_prompt: Whether to use system prompt
            batch_size: Batch size for generation
        """
        self.model_config = model
        if "model" in model and "tokenizer" in model:
            self.model = model["model"]
            self.tokenizer = model["tokenizer"]
        self.max_n_tokens = max_n_tokens
        print("USE SYSTEM PROMPT = ", use_system_prompt)
        self.use_system_prompt = use_system_prompt
        self.batch_size = batch_size

    @abstractmethod
    def generate_target_response(
        self, prompts: List[str], log_callback=None
    ) -> List[str]:
        """Generate responses from target model for multiple prompts.

        Args:
            prompts: List of prompts to generate responses for
            log_callback: Optional callback function for logging model interactions

        Returns:
            List of generated responses
        """
        pass


class HFInstructLM(TargetLM):
    """Target LM that uses HuggingFace instruction format."""

    def generate_target_response(
        self, prompts: List[str], log_callback=None
    ) -> List[str]:
        """Generate responses using instruction format."""
        all_responses = []

        for batch_start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[batch_start : batch_start + self.batch_size]
            batch_messages = []
            for prompt in batch_prompts:
                if self.use_system_prompt:
                    messages = [
                        {"role": "system", "content": LLAMA2_DEFAULT_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ]
                else:
                    messages = [{"role": "user", "content": prompt}]
                batch_messages.append(messages)

            batch_inputs = [
                self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                for messages in batch_messages
            ]

            input_ids = self.tokenizer(
                batch_inputs, return_tensors="pt", padding=True
            ).to(self.model.device)

            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=self.max_n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            for i, output in enumerate(outputs):
                response = self.tokenizer.decode(
                    output[input_ids["input_ids"][i].shape[0] :],
                    skip_special_tokens=True,
                )
                all_responses.append(response)

                if log_callback:
                    log_callback(
                        "target_model",
                        input_ids["input_ids"][i],
                        output[input_ids["input_ids"][i].shape[0] :],
                    )

        return all_responses


class HFBaseLM(TargetLM):
    """Target LM that uses base model with URIAL format."""

    def __init__(self, *args, **kwargs):
        """Initialize base model with URIAL format."""
        super().__init__(*args, **kwargs)
        # Configure URIAL chat template
        configure_urial_chat_template(self.tokenizer)
        self.urial_prompt = load_urial_prompt(
            demonstration_count=10, system_prompt=True, tokenizer=self.tokenizer
        )

    def generate_target_response(
        self, prompts: List[str], log_callback=None
    ) -> List[str]:
        """Generate responses using URIAL format."""
        all_responses = []

        for batch_start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[batch_start : batch_start + self.batch_size]
            batch_messages = []
            for prompt in batch_prompts:
                messages = [
                    {"role": "system", "content": self.urial_prompt},
                    {"role": "user", "content": prompt},
                ]
                batch_messages.append(messages)

            batch_inputs = [
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    add_special_tokens=False,
                )
                for messages in batch_messages
            ]

            input_ids = self.tokenizer(
                batch_inputs,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.model.device)

            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=self.max_n_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            for i, output in enumerate(outputs):
                # Extract response between ``` marks
                response = self.tokenizer.decode(
                    output[input_ids["input_ids"][i].shape[0] :],
                    skip_special_tokens=False,
                )
                try:
                    response = response.split("```")[1].split("```")[0]
                except IndexError:
                    try:
                        response = response.split("```")[1]
                    except IndexError:
                        response = ""
                all_responses.append(response)

                if log_callback:
                    log_callback(
                        "target_model",
                        input_ids["input_ids"][i],
                        output[input_ids["input_ids"][i].shape[0] :],
                    )

        return all_responses


def get_openrouter_balance(api_key):
    """Get the OpenRouter account balance.

    Args:
        api_key: OpenRouter API key

    Returns:
        Dict with balance information or None if request failed
    """
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {"Authorization": f"Bearer {api_key}"}

    try:
        response = requests.get(url, headers=headers)
        if response.ok:
            data = response.json().get("data", {})
            return {
                "credits_used": data.get("usage"),
                "credit_limit": data.get("limit"),
                "is_free_tier": data.get("is_free_tier"),
                "rate_limit": data.get("rate_limit"),
            }
        return None
    except Exception as e:
        print(f"Error fetching OpenRouter balance: {str(e)}")
        return None


def estimate_openai_cost(model_name, prompt_tokens, completion_tokens):
    """Estimate the cost of an OpenAI API call based on token usage.

    Args:
        model_name: Name of the OpenAI model
        prompt_tokens: Number of prompt tokens
        completion_tokens: Number of completion tokens

    Returns:
        Estimated cost in dollars
    """
    # Pricing information with scaling factor
    # Format: {"model_name": {"prompt": price_per_unit, "completion": price_per_unit, "scale": tokens_per_unit}}
    pricing = {
        "o3-mini": {
            "prompt": 1.1,
            "completion": 4.4,
            "scale": 1_000_000,  # Price per million tokens
        },
    }

    # Get pricing for the model, default to o3-mini if not found
    model_pricing = pricing.get(model_name, pricing["o3-mini"])

    # Calculate costs using the appropriate scaling
    scale = model_pricing["scale"]
    prompt_cost = (prompt_tokens / scale) * model_pricing["prompt"]
    completion_cost = (completion_tokens / scale) * model_pricing["completion"]
    total_cost = prompt_cost + completion_cost

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_cost": prompt_cost,
        "completion_cost": completion_cost,
        "total_cost": total_cost,
    }


class OpenAILM(TargetLM):
    """Target LM that uses OpenAI API."""

    def __init__(
        self,
        model: Dict,
        max_n_tokens: int,
        use_system_prompt: bool = True,
        batch_size: int = 8,
        api_base: Optional[str] = None,
        api_type: str = "chat",
        temperature: float = 0.0,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        provider: str = "openai",
        extra_headers: Optional[Dict] = None,
        extra_body: Optional[Dict] = None,
    ):
        """Initialize OpenAI API model.

        Args:
            model: Dict containing model configuration
            max_n_tokens: Max tokens for target response
            use_system_prompt: Whether to use system prompt
            batch_size: Batch size for generation
            api_base: Base URL for API
            api_type: Type of API ('chat' for chat completions, 'completion' for text completions)
            temperature: Temperature for generation (0 for deterministic)
            api_key: API key (if None, reads from environment variable)
            model_name: Name of the model to use
            provider: API provider ('openai' or 'openrouter')
            extra_headers: Additional headers for API requests
            extra_body: Additional body parameters for API requests
        """
        # Initialize base class
        super().__init__(model, max_n_tokens, use_system_prompt, batch_size)

        # Get provider from model config first, then from parameter
        self.provider = self.model_config.get("provider", provider)
        self.model_name = model_name or self.model_config.get("model_name")
        self.api_base = api_base or self.model_config.get("base_url")
        self.temperature = temperature
        self.extra_headers = extra_headers or {}
        self.extra_body = extra_body or {}
        self.max_retries = 3

        # Initialize cost tracking
        self.total_cost = 0.0
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_requests = 0

        # Get API key from parameters or environment
        env_var = (
            "OPENROUTER_API_KEY" if self.provider == "openrouter" else "OPENAI_API_KEY"
        )
        self.api_key = api_key or os.environ.get(env_var)
        if not self.api_key:
            raise ValueError(
                f"API key not found. Please set the {env_var} environment variable or pass api_key."
            )

        # Initialize OpenAI client
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=self.api_key,
        )

    def generate_target_response(
        self, prompts: List[str], log_callback=None
    ) -> List[str]:
        """Generate responses using OpenAI API."""
        all_responses = []

        for batch_start in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[batch_start : batch_start + self.batch_size]

            for prompt in batch_prompts:
                try:
                    # Get balance before API call
                    balance_before = None
                    if self.provider == "openrouter":
                        balance_before = get_openrouter_balance(self.api_key)
                        if balance_before:
                            print(
                                f"OpenRouter balance before: {balance_before['credits_used']} / {balance_before['credit_limit']}"
                            )
                    elif self.provider == "openai":
                        # For OpenAI, we'll estimate cost after the API call
                        pass

                    if self.use_system_prompt:
                        messages = [
                            {"role": "system", "content": LLAMA2_DEFAULT_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ]
                    else:
                        messages = [{"role": "user", "content": prompt}]
                    print("MESSAGES", messages)
                    for retry in range(self.max_retries):
                        try:
                            # Set parameters based on provider
                            if self.provider == "openrouter":
                                params = {
                                    "model": self.model_name,
                                    "messages": messages,
                                    "max_tokens": self.max_n_tokens,
                                    "temperature": self.temperature,
                                    "extra_headers": self.extra_headers,
                                    "extra_body": self.extra_body,
                                }
                            elif self.provider == "openai":
                                params = {
                                    "model": self.model_name,
                                    "messages": messages,
                                    "max_completion_tokens": self.max_n_tokens,
                                    "extra_headers": self.extra_headers,
                                    "extra_body": self.extra_body,
                                }

                            completion = self.client.chat.completions.create(**params)
                            print("COMPLETION", completion)
                            if (
                                completion.choices[0].finish_reason == "error"
                                and completion.choices[0].native_finish_reason
                                == "SAFETY"
                            ):
                                print("SAFETY MECHANISM TRIGGERED")
                                text_response = (
                                    "System message: Error. Safety mechanism triggered."
                                )
                            else:
                                text_response = completion.choices[0].message.content

                            break  # Exit the loop if the request is successful

                        except Exception as e:
                            print(f"Error in OpenAI API call: {str(e)}")
                            if retry < self.max_retries - 1:
                                print(f"Retrying... ({retry + 1}/{self.max_retries})")
                            else:
                                print("Max retries reached. Exiting.")
                                raise e

                    all_responses.append(text_response)

                    # Initialize extra_log_info dictionary
                    extra_log_info = {}

                    # Get balance after API call and calculate cost information
                    if self.provider == "openrouter":
                        balance_after = get_openrouter_balance(self.api_key)
                        if balance_after and balance_before:
                            credits_used = (
                                balance_after["credits_used"]
                                - balance_before["credits_used"]
                            )
                            print(
                                f"OpenRouter balance after: {balance_after['credits_used']} / {balance_after['credit_limit']}"
                            )
                            extra_log_info = {
                                "credits_before": balance_before["credits_used"],
                                "credits_after": balance_after["credits_used"],
                                "credits_used": credits_used,
                                "credit_limit": balance_after["credit_limit"],
                            }
                    elif self.provider == "openai":
                        # Estimate cost based on token usage
                        if hasattr(completion, "usage"):
                            cost_info = estimate_openai_cost(
                                self.model_name,
                                completion.usage.prompt_tokens,
                                completion.usage.completion_tokens,
                            )
                            print(
                                f"OpenAI estimated cost: ${cost_info['total_cost']:.6f} "
                                f"(prompt: ${cost_info['prompt_cost']:.6f}, "
                                f"completion: ${cost_info['completion_cost']:.6f})"
                            )

                            # Update cumulative cost tracking
                            self.total_cost += cost_info["total_cost"]
                            self.total_prompt_tokens += cost_info["prompt_tokens"]
                            self.total_completion_tokens += cost_info[
                                "completion_tokens"
                            ]
                            self.total_requests += 1

                            print(
                                f"Cumulative cost so far: ${self.total_cost:.6f} "
                                f"({self.total_prompt_tokens} prompt tokens, "
                                f"{self.total_completion_tokens} completion tokens, "
                                f"{self.total_requests} requests)"
                            )

                            extra_log_info = cost_info

                    if log_callback:
                        log_callback(
                            "target_model",
                            prompt,
                            text_response,
                            is_api=True,
                            cost_info=extra_log_info,
                        )

                except Exception as e:
                    error_msg = f"API request failed: {str(e)}"
                    print(error_msg)
                    all_responses.append(error_msg)
                    raise Exception(error_msg)
        print("get_cost_summary", self.get_cost_summary())
        return all_responses

    def get_cost_summary(self):
        """Get a summary of the cumulative cost and usage.

        Returns:
            Dict with cost and usage information
        """
        return {
            "total_cost": self.total_cost,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_requests": self.total_requests,
            "average_cost_per_request": self.total_cost / self.total_requests
            if self.total_requests > 0
            else 0,
        }
