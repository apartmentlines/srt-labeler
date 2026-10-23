"""Verify provider dispatch and preservation of signed audio URLs."""

from __future__ import annotations

from typing import Any

import pytest

from srt_labeler.audio import AudioAttachmentFactory


class TestAudioAttachmentFactory:
    """Cover audio adapters without invoking any provider or HTTP client."""

    def test_openai_compatible_format_preserves_signed_url(self) -> None:
        signed_url = (
            "https://example.com/123--call%2Bname.wav"
            "?transcription_id=456&api_key=signature"
        )
        message = AudioAttachmentFactory().create(
            "provider_chat_openai_compat", signed_url
        )
        assert message.content == [
            {
                "type": "input_audio",
                "input_audio": {"data": signed_url, "format": "wav"},
            }
        ]

    def test_gemini_uses_uri_instead_of_inline_data(self) -> None:
        signed_url = "https://example.com/call.wav?api_key=signature&transcription_id=1"
        message = AudioAttachmentFactory().create(
            "provider_chat_google_genai", signed_url
        )
        assert message.content == [
            {
                "type": "media",
                "mime_type": "audio/wav",
                "file_uri": signed_url,
            }
        ]

    @pytest.mark.parametrize(
        "signed_url", ["", "http://example.com/a.wav", "https:///a.wav"]
    )
    def test_rejects_missing_or_invalid_urls(self, signed_url: str) -> None:
        with pytest.raises(ValueError, match="signed_url"):
            AudioAttachmentFactory().create("provider_chat_openai_compat", signed_url)

    def test_unregistered_provider_raises_error(self) -> None:
        with pytest.raises(ValueError, match="No audio URL formatter"):
            AudioAttachmentFactory().create(
                "provider_unregistered", "https://example.com/a.wav"
            )

    @staticmethod
    def custom_formatter(signed_url: str) -> dict[str, Any]:
        return {"type": "audio", "url": signed_url}

    def test_additional_provider_can_be_registered(self) -> None:
        factory = AudioAttachmentFactory()
        factory.register("provider_custom", self.custom_formatter)
        message = factory.create("provider_custom", "https://example.com/a.wav")
        assert message.content == [
            {"type": "audio", "url": "https://example.com/a.wav"}
        ]
