"""Build remote and local audio attachments for each LWE provider."""

from __future__ import annotations

from collections.abc import Callable
import base64
from pathlib import Path
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

    def create(self, provider: str, signed_url: str | None) -> HumanMessage:
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

    def create_local(self, provider: str, path: Path) -> HumanMessage:
        """Encode a local WAV file for Qwen or Google without hosting it.

        :raises ValueError: If the provider or file format is unsupported.
        :raises OSError: If the audio cannot be read.
        """
        if provider not in (
            "provider_chat_openai_compat",
            "provider_chat_google_genai",
        ):
            raise ValueError(f"No local audio formatter for provider {provider!r}")
        self.validate_local(path)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        if provider == "provider_chat_openai_compat":
            block = self.openai_audio(f"data:audio/wav;base64,{encoded}")
        else:
            block = {"type": "media", "data": encoded, "mime_type": "audio/wav"}
        return HumanMessage(content=[block])

    @staticmethod
    def validate_local(path: Path) -> None:
        """Check the WAV container signature before initializing a model backend."""
        with path.open("rb") as audio:
            header = audio.read(12)
        if len(header) < 12 or header[:4] != b"RIFF" or header[8:12] != b"WAVE":
            raise ValueError(f"Local audio must be a WAV file: {path}")

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
