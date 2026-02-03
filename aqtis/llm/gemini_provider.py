"""
AQTIS Gemini LLM Provider.

Wraps Google's Gemini API for text generation.
"""

import logging
import os
from typing import Optional

from .base import LLMProvider

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """Google Gemini LLM provider."""

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        temperature: float = 0.3,
        max_tokens: int = 8000,
        timeout_seconds: int = 30,
        api_key: str = None,
    ):
        super().__init__(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_seconds=timeout_seconds,
        )
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self._client = None

    def _get_client(self):
        """Lazy-initialize the Gemini client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set")
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "google-genai is required for Gemini. "
                    "Install it with: pip install google-genai"
                )
        return self._client

    def _call(self, prompt: str, temperature: float = None, max_tokens: int = None, json_mode: bool = False) -> str:
        """Make a Gemini API call."""
        client = self._get_client()

        from google.genai import types

        config = types.GenerateContentConfig(
            temperature=temperature if temperature is not None else self.temperature,
            max_output_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            response_mime_type='application/json' if json_mode else None,
        )

        response = client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=config,
        )

        if response and response.text:
            return response.text
        return ""

    def is_available(self) -> bool:
        """Check if Gemini is configured."""
        return bool(self._api_key)

    @property
    def provider_name(self) -> str:
        return "gemini"
