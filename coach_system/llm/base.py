"""
LLM Provider Abstraction for 5-Player Trading System.

Pluggable interface for LLM backends (Gemini, Claude, OpenAI, etc.).
"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Type

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        max_tokens: int = 4000,
        timeout_seconds: int = 30,
        max_retries: int = 3,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    @abstractmethod
    def _call(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        """
        Make the actual API call. Subclasses implement this.

        Returns:
            Raw text response from the LLM.
        """
        ...

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> str:
        """
        Generate text with retry logic.

        Args:
            prompt: The prompt to send to the LLM.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Generated text response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = self._call(prompt, temperature=temp, max_tokens=tokens)
                return response
            except Exception as e:
                last_error = e
                wait_time = 2 ** attempt
                logger.warning(
                    f"LLM call failed (attempt {attempt + 1}/{self.max_retries}): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_error}")

    def generate_json(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
    ) -> Dict[str, Any]:
        """
        Generate a JSON response with automatic parsing.

        The prompt should instruct the LLM to respond in JSON format.

        Returns:
            Parsed JSON dictionary.
        """
        response = self.generate(prompt, temperature=temperature, max_tokens=max_tokens)

        # Try to extract JSON from the response
        text = response.strip()

        # Handle ```json ... ``` blocks
        if "```json" in text:
            start = text.index("```json") + 7
            end = text.index("```", start)
            text = text[start:end].strip()
        elif "```" in text:
            start = text.index("```") + 3
            end = text.index("```", start)
            text = text[start:end].strip()

        # Try to find JSON object or array
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            if start_char in text:
                start = text.index(start_char)
                # Find matching closing bracket
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == start_char:
                        depth += 1
                    elif text[i] == end_char:
                        depth -= 1
                    if depth == 0:
                        text = text[start:i + 1]
                        break

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            logger.debug(f"Raw response: {response[:500]}")
            return {"raw_response": response, "parse_error": str(e)}

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is configured and available."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        ...


class MockLLMProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, responses: Dict[str, str] = None, default_response: str = "{}"):
        super().__init__(model="mock", temperature=0.0)
        self._responses = responses or {}
        self._default_response = default_response
        self._call_log: list = []

    def _call(self, prompt: str, temperature: float = None, max_tokens: int = None) -> str:
        self._call_log.append(prompt)
        for key, response in self._responses.items():
            if key.lower() in prompt.lower():
                return response
        return self._default_response

    def is_available(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return "mock"
