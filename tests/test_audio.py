"""Verify provider dispatch and preservation of signed audio URLs."""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any
import wave

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

    @pytest.mark.parametrize(
        "provider", ["provider_chat_openai_compat", "provider_chat_google_genai"]
    )
    def test_local_audio_preserves_wav_bytes(
        self, tmp_path: Path, provider: str
    ) -> None:
        path = tmp_path / "recording.wav"
        with wave.open(str(path), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(8000)
            audio.writeframes(b"\x00\x00" * 80)
        message = AudioAttachmentFactory().create_local(provider, path)
        block = message.content[0]
        assert isinstance(block, dict)
        if provider == "provider_chat_openai_compat":
            assert block["type"] == "input_audio"
            assert block["input_audio"]["format"] == "wav"
            data = block["input_audio"]["data"]
            assert data.startswith("data:audio/wav;base64,")
            encoded = data.split(",", 1)[1]
        else:
            assert block["type"] == "media"
            assert block["mime_type"] == "audio/wav"
            assert "file_uri" not in block
            encoded = block["data"]
        assert base64.b64decode(encoded) == path.read_bytes()

    def test_local_audio_rejects_unsupported_provider(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No local audio formatter"):
            AudioAttachmentFactory().create_local(
                "provider_openrouter", tmp_path / "a.wav"
            )

    def test_local_audio_rejects_invalid_wav(self, tmp_path: Path) -> None:
        path = tmp_path / "audio.wav"
        path.write_bytes(b"not really a WAV")
        with pytest.raises(ValueError, match="WAV"):
            AudioAttachmentFactory().create_local("provider_chat_google_genai", path)
