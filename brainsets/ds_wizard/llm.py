"""
LLM factory and utilities for the Dataset Wizard.
"""

import logging
import re
from typing import Any, Optional

from langchain_cerebras import ChatCerebras
from langchain_ollama import ChatOllama
from langchain_google_vertexai import ChatVertexAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.rate_limiters import InMemoryRateLimiter

from brainsets.ds_wizard.config import LLMProvider, PROVIDER_MODELS

logger = logging.getLogger(__name__)

_rate_limiters: dict[str, InMemoryRateLimiter] = {}


def get_rate_limiter(
    provider: str, requests_per_second: float = 0.1
) -> InMemoryRateLimiter:
    """Get or create an InMemoryRateLimiter for a provider."""
    if provider not in _rate_limiters:
        _rate_limiters[provider] = InMemoryRateLimiter(
            requests_per_second=requests_per_second,
            check_every_n_seconds=0.1,
            max_bucket_size=10,
        )
        logger.info(f"Created rate limiter for {provider}: {requests_per_second} req/s")
    return _rate_limiters[provider]


def parse_retry_delay(error_message: str) -> Optional[float]:
    """Extract retry delay from API error messages."""
    patterns = [
        r"retry in (\d+(?:\.\d+)?)s",
        r"retryDelay.*?(\d+)s",
        r"Please retry in (\d+(?:\.\d+)?)s",
    ]
    for pattern in patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            return float(match.group(1))
    return None


def create_llm(
    provider: LLMProvider,
    model_name: str,
    temperature: float = 0.1,
    rate_limit: Optional[float] = None,
) -> Any:
    """
    Create an LLM instance for the given provider with optional rate limiting.

    Args:
        provider: The LLM provider to use
        model_name: The model name
        temperature: Sampling temperature
        rate_limit: Requests per second (e.g., 0.1 = 1 request every 10s)

    Returns:
        A LangChain chat model instance
    """
    if provider in PROVIDER_MODELS:
        supported_models = PROVIDER_MODELS[provider]
        if model_name not in supported_models:
            logger.warning(
                f"Model '{model_name}' may not be officially supported by provider '{provider}'. "
                f"Supported models: {supported_models}"
            )

    rate_limiter = get_rate_limiter(provider, rate_limit) if rate_limit else None

    if provider == "cerebras":
        return ChatCerebras(
            model=model_name,
            temperature=temperature,
            rate_limiter=rate_limiter,
        )
    elif provider == "ollama":
        return ChatOllama(
            model=model_name,
            temperature=temperature,
            base_url="http://localhost:11434",
            rate_limiter=rate_limiter,
        )
    elif provider == "vertexai":
        return ChatVertexAI(
            model=model_name,
            temperature=temperature,
            rate_limiter=rate_limiter,
        )
    elif provider == "google":
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            max_output_tokens=65536,
            include_thoughts=True,
            thinking_budget=10000,
            max_retries=5,
            timeout=120,
            rate_limiter=rate_limiter,
        )
    else:
        raise ValueError(
            f"Unsupported provider: {provider}. Supported providers: {list(PROVIDER_MODELS.keys())}"
        )


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback handler to track token usage from LLM calls."""

    def __init__(self):
        super().__init__()
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Called when LLM finishes running."""
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            self.total_input_tokens += usage.get("prompt_tokens", 0)
            self.total_output_tokens += usage.get("completion_tokens", 0)
            self.total_tokens += usage.get("total_tokens", 0)
            self.llm_calls += 1
        elif hasattr(response, "generations") and response.generations:
            for generation_list in response.generations:
                for generation in generation_list:
                    if hasattr(generation, "message") and hasattr(
                        generation.message, "usage_metadata"
                    ):
                        usage_metadata = generation.message.usage_metadata
                        self.total_input_tokens += usage_metadata.get("input_tokens", 0)
                        self.total_output_tokens += usage_metadata.get(
                            "output_tokens", 0
                        )
                        self.total_tokens += usage_metadata.get("total_tokens", 0)
                        self.llm_calls += 1

    def reset(self):
        """Reset the token counters."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.llm_calls = 0

    def get_usage_summary(self) -> dict:
        """Get a summary of token usage."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "llm_calls": self.llm_calls,
        }


def extract_output_text(raw_output: Any) -> str:
    """
    Extract the actual text output from potentially complex output structure.

    LangChain 1.x with thinking/structured outputs returns:
    - A simple string (basic format)
    - A list of content blocks: [{'type': 'thinking', 'thinking': '...'}, {'type': 'text', 'text': '...'}]

    Args:
        raw_output: The raw output from the agent

    Returns:
        The extracted text string, or empty string if not found
    """
    if not raw_output:
        return ""

    if isinstance(raw_output, str):
        return raw_output

    if isinstance(raw_output, list):
        text_parts = []

        for item in raw_output:
            if isinstance(item, dict):
                if item.get("type") == "text" and "text" in item:
                    text_parts.append(item["text"])
                elif item.get("type") == "thinking":
                    continue
            elif isinstance(item, str):
                text_parts.append(item)

        if text_parts:
            return "\n".join(text_parts)

        return ""

    return str(raw_output)
