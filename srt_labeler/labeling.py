"""Share preset execution and transcript extraction across production and local runs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from lwe import ApiBackend

from .audio import AudioAttachmentFactory
from .errors import ModelResponseFormattingError, RequestError
from .token_usage import TokenUsage


class LabelingEngine:
    """Execute labeling templates using an injected LWE backend."""

    def __init__(
        self,
        backend: ApiBackend,
        audio_attachments: AudioAttachmentFactory | None = None,
    ) -> None:
        self.backend: ApiBackend = backend
        self.audio_attachments: AudioAttachmentFactory = (
            audio_attachments or AudioAttachmentFactory()
        )

    def audio_attachment(self, preset_name: str, source: str | Path) -> HumanMessage:
        """Format a remote URL or local file for the selected preset's provider.

        :raises RequestError: If the preset or attachment is unsupported.
        """
        success, preset, message = self.backend.preset_manager.ensure_preset(
            preset_name
        )
        if not success:
            raise RequestError(f"Cannot load audio preset {preset_name!r}: {message}")
        metadata, _customizations = cast(tuple[dict[str, Any], dict[str, Any]], preset)
        provider = metadata.get("provider")
        if not isinstance(provider, str) or not provider:
            raise RequestError(f"Preset {preset_name!r} has no provider")
        provider = self.backend.provider_manager.full_name(provider)
        try:
            if isinstance(source, Path):
                return self.audio_attachments.create_local(provider, source)
            return self.audio_attachments.create(provider, source)
        except ValueError as error:
            raise RequestError(str(error), cause=error) from error

    def run_template(
        self,
        template_name: str,
        template_vars: dict[str, Any],
        overrides: dict[str, Any] | None = None,
        usage: TokenUsage | None = None,
    ) -> tuple[bool, str, str | None]:
        """Return complete response text before transcript extraction or merging."""
        success, response, message = self.backend.run_template(
            template_name, template_vars=template_vars, overrides=overrides
        )
        if usage is not None:
            usage.fallback(response)
        return success, self.normalize_response(response), message

    @staticmethod
    def normalize_response(content: Any) -> str:
        """Collect response text, including text blocks in an AIMessage."""
        if isinstance(content, AIMessage):
            content = content.content
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts).strip()
        return str(content).strip()

    @staticmethod
    def extract_transcript(response: str | None) -> str:
        """Extract the model's SRT using the production transcript tag convention.

        :raises ModelResponseFormattingError: If no transcript section exists.
        """
        if not response:
            raise ModelResponseFormattingError("No response provided")
        match = re.search(r"<transcript>(.*?)</transcript>", response, re.DOTALL)
        if not match:
            raise ModelResponseFormattingError(
                "No transcript section found in the text"
            )
        return match.group(1).strip()
