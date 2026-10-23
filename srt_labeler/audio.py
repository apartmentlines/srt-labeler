"""Build signed audio URL attachments for each LWE provider."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit

from langchain_core.messages import HumanMessage


class AudioAttachmentFactory:
    """Dispatch audio formatting by provider plugin, independently of models."""

    def __init__(self) -> None:
        self.formatters: dict[str, Callable[[str], dict[str, Any]]] = {}
        self.register("provider_chat_openai_compat", self.openai_audio)
        self.register("provider_chat_google_genai", self.google_audio)

    def register(
        self, provider: str, formatter: Callable[[str], dict[str, Any]]
    ) -> None:
        """Register an audio formatter for an additional provider plugin.

        :param provider: Full LWE provider plugin name.
        :param formatter: Callable converting a signed URL to a content block.
        """
        self.formatters[provider] = formatter

    def create(self, provider: str, signed_url: str) -> HumanMessage:
        """Build an attachment without downloading or modifying its signed URL.

        :param provider: Full provider plugin name selected by the request.
        :param signed_url: Complete, externally accessible HTTPS recording URL.
        :return: A message suitable for LWE's request ``files`` override.
        :raises ValueError: If the provider or signed URL is unsupported.
        """
        formatter = self.formatters.get(provider)
        if formatter is None:
            raise ValueError(f"No audio URL formatter for provider {provider!r}")
        if not isinstance(signed_url, str) or not signed_url:
            raise ValueError("Transcription is missing a signed_url")
        parsed = urlsplit(signed_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("Transcription signed_url must be an HTTPS URL")
        return HumanMessage(content=[formatter(signed_url)])

    @staticmethod
    def openai_audio(signed_url: str) -> dict[str, Any]:
        """Format URL audio for the Qwen OpenAI-compatible endpoint."""
        return {
            "type": "input_audio",
            "input_audio": {"data": signed_url, "format": "wav"},
        }

    @staticmethod
    def google_audio(signed_url: str) -> dict[str, Any]:
        """Format URL audio for LangChain's Google GenAI provider."""
        return {
            "type": "media",
            "file_uri": signed_url,
            "mime_type": "audio/wav",
        }
