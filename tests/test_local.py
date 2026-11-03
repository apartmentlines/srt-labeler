"""Exercise local labeling without model calls or production API access."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import Mock
import wave

from langchain_core.messages import AIMessage
from lwe.core.config import Config
import pytest
import srt

from srt_labeler.constants import LWE_DEFAULT_PRESET, LWE_TRANSCRIPTION_TEMPLATE
from srt_labeler.errors import ModelResponseFormattingError, RequestError
from srt_labeler.local import LocalLabeler, main
from srt_labeler.merger import SrtMergeError

ORIGINAL = "1\n00:00:01,000 --> 00:00:02,000\nHello there.\n\n"
RESPONSE = (
    "Analysis before transcript\n<transcript>\n"
    "1\n00:00:09,000 --> 00:00:10,000\nOperator: Changed words.\n"
    "</transcript>\nTrailing text\n"
)


@pytest.fixture(autouse=True)
def no_http(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail if local orchestration tries to retrieve or update remote recordings."""
    blocked = Mock(side_effect=AssertionError("Unexpected HTTP request"))
    monkeypatch.setattr("requests.get", blocked)
    monkeypatch.setattr("requests.post", blocked)


@pytest.fixture
def lwe_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, Path]:
    """Keep test configuration and state separate from the installed LWE setup."""
    config_dir = tmp_path / "config"
    profile_dir = config_dir / "profiles" / "default"
    profile_dir.mkdir(parents=True)
    (profile_dir / "config.yaml").write_text(
        "backend_options:\n"
        "  auto_create_first_user: existing-user\n"
        "  default_user: existing-user\n"
        "  default_conversation_id: 42\n"
        "chat:\n"
        "  log:\n"
        "    enabled: true\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setenv("LWE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("LWE_DATA_DIR", str(data_dir))
    return config_dir, data_dir


@pytest.fixture
def inputs(
    tmp_path: Path, lwe_directories: tuple[Path, Path]
) -> tuple[Path, Path, Path]:
    """Create a real WAV and an SRT containing one subtitle."""
    srt_path = tmp_path / "call.srt"
    srt_path.write_text(ORIGINAL, encoding="utf-8")
    audio_path = tmp_path / "call.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\x00\x00" * 800)
    return srt_path, audio_path, tmp_path / "output"


@pytest.fixture
def backend() -> Mock:
    """Supply an injected LWE backend using the Google attachment adapter."""
    backend = Mock()
    backend.preset_manager.ensure_preset.return_value = (
        True,
        ({"provider": "chat_google_genai"}, {}),
        None,
    )
    backend.provider_manager.full_name.return_value = "provider_chat_google_genai"
    backend.run_template.return_value = (True, RESPONSE, None)
    return backend


class TestLocalLabeler:
    """Verify artifacts, configuration, and failure boundaries."""

    def test_debug_saves_execution_diagnostics(
        self,
        inputs: tuple[Path, Path, Path],
        backend: Mock,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root_logger = logging.getLogger()
        monkeypatch.setattr(root_logger, "handlers", [])
        monkeypatch.setattr(root_logger, "level", root_logger.level)

        def run_template(*_args: object, **_kwargs: object) -> tuple[bool, str, None]:
            logging.getLogger("local_test_request").debug("Request diagnostics")
            return True, RESPONSE, None

        backend.run_template.side_effect = run_template
        try:
            LocalLabeler(Mock(return_value=backend), debug=True).run(*inputs)
        finally:
            for handler in root_logger.handlers:
                handler.close()
        assert "Request diagnostics" in (inputs[2] / "lwe-debug.log").read_text()

    def test_success_preserves_response_and_merges_original(
        self,
        inputs: tuple[Path, Path, Path],
        backend: Mock,
        monkeypatch: pytest.MonkeyPatch,
        lwe_directories: tuple[Path, Path],
    ) -> None:
        for name in (
            "SRT_LABELER_API_KEY",
            "SRT_LABELER_FILE_API_KEY",
            "SRT_LABELER_DOMAIN",
        ):
            monkeypatch.delenv(name, raising=False)
        factory = Mock(return_value=backend)
        config_dir, data_dir = lwe_directories
        config_path = config_dir / "profiles" / "default" / "config.yaml"
        original_config = config_path.read_bytes()
        LocalLabeler(factory).run(*inputs, preset="chosen-preset")
        output = inputs[2]
        assert (output / "response.txt").read_text() == RESPONSE
        final = list(srt.parse((output / "final.srt").read_text()))
        original = list(srt.parse(ORIGINAL))[0]
        assert final[0].content == "Operator: Hello there."
        assert final[0].start == original.start
        assert final[0].end == original.end
        backend.run_template.assert_called_once()
        args, kwargs = backend.run_template.call_args
        assert args == (LWE_TRANSCRIPTION_TEMPLATE,)
        assert kwargs["template_vars"]["transcription"] == ORIGINAL
        overrides = kwargs["overrides"]["request_overrides"]
        assert overrides["preset"] == "chosen-preset"
        assert overrides["files"][0].content[0]["type"] == "media"
        config = factory.call_args.args[0]
        assert config.get("model.default_preset") == "chosen-preset"
        assert config.get("backend_options.auto_create_first_user") is None
        assert config.get("backend_options.default_user") is None
        assert config.get("backend_options.default_conversation_id") is None
        assert config.get("chat.log.enabled") is False
        assert Path(config.config_dir) == config_dir
        assert Path(config.data_dir) == data_dir
        assert data_dir.exists()
        assert config_path.read_bytes() == original_config

    @pytest.mark.parametrize(
        ("response", "error_type"),
        [
            ("No transcript tags", ModelResponseFormattingError),
            ("<transcript>not an SRT</transcript>", SrtMergeError),
        ],
    )
    def test_processing_failure_keeps_response_without_fallback(
        self,
        inputs: tuple[Path, Path, Path],
        backend: Mock,
        response: str,
        error_type: type[Exception],
    ) -> None:
        backend.run_template.return_value = (True, response, None)
        with pytest.raises(error_type):
            LocalLabeler(Mock(return_value=backend)).run(*inputs)
        assert (inputs[2] / "response.txt").read_text() == response
        assert not (inputs[2] / "final.srt").exists()
        backend.run_template.assert_called_once()

    def test_model_failure_keeps_returned_response(
        self, inputs: tuple[Path, Path, Path], backend: Mock
    ) -> None:
        backend.run_template.return_value = (
            False,
            "partial response",
            "Provider failed",
        )
        with pytest.raises(RuntimeError, match="Provider failed"):
            LocalLabeler(Mock(return_value=backend)).run(*inputs)
        assert (inputs[2] / "response.txt").read_text() == "partial response"
        assert not (inputs[2] / "final.srt").exists()
        backend.run_template.assert_called_once()

    def test_model_exception_does_not_create_fake_output(
        self, inputs: tuple[Path, Path, Path], backend: Mock
    ) -> None:
        backend.run_template.side_effect = RuntimeError("Connection failed")
        with pytest.raises(RuntimeError, match="Connection failed"):
            LocalLabeler(Mock(return_value=backend)).run(*inputs)
        assert list(inputs[2].iterdir()) == []
        backend.run_template.assert_called_once()

    def test_ai_message_text_blocks_are_extracted(
        self, inputs: tuple[Path, Path, Path], backend: Mock
    ) -> None:
        backend.run_template.return_value = (
            True,
            AIMessage(content=[{"type": "text", "text": RESPONSE}]),
            None,
        )
        LocalLabeler(Mock(return_value=backend)).run(*inputs)
        assert (inputs[2] / "response.txt").read_text() == RESPONSE.strip()
        assert (inputs[2] / "final.srt").exists()

    @pytest.mark.parametrize("override_environment", [False, True])
    def test_template_override_uses_lwe_rendering(
        self,
        inputs: tuple[Path, Path, Path],
        backend: Mock,
        monkeypatch: pytest.MonkeyPatch,
        override_environment: bool,
    ) -> None:
        from lwe.core.template_manager import TemplateManager

        template = inputs[0].parent / "alternate.md"
        template.write_text("---\ndescription: Test\n---\nLabel {{ transcription }}")
        if override_environment:
            other_templates = inputs[0].parent / "other-templates"
            other_templates.mkdir()
            (other_templates / template.name).write_text("Wrong template")
            monkeypatch.setenv("LWE_TEMPLATE_DIR", str(other_templates))

        def create_backend(config: Config) -> Mock:
            manager = TemplateManager(config)
            assert manager.ensure_template(template.name)[0]
            message, _overrides = manager.build_message_from_template(
                template.name, {"transcription": ORIGINAL}
            )
            assert message == f"Label {ORIGINAL}"
            return backend

        LocalLabeler(create_backend).run(*inputs, template_file=template)
        assert backend.run_template.call_args.args == ("alternate.md",)

    @pytest.mark.parametrize("content", ["", "not a subtitle"])
    def test_invalid_srt_fails_before_backend_initialization(
        self, inputs: tuple[Path, Path, Path], content: str
    ) -> None:
        inputs[0].write_text(content)
        factory = Mock()
        with pytest.raises((ValueError, srt.SRTParseError)):
            LocalLabeler(factory).run(*inputs)
        factory.assert_not_called()
        assert not inputs[2].exists()

    def test_invalid_audio_fails_before_backend_initialization(
        self, inputs: tuple[Path, Path, Path]
    ) -> None:
        inputs[1].write_bytes(b"not WAV")
        factory = Mock()
        with pytest.raises(ValueError, match="WAV"):
            LocalLabeler(factory).run(*inputs)
        factory.assert_not_called()

    def test_missing_template_fails_before_backend_initialization(
        self, inputs: tuple[Path, Path, Path]
    ) -> None:
        factory = Mock()
        with pytest.raises(FileNotFoundError):
            LocalLabeler(factory).run(
                *inputs, template_file=inputs[0].parent / "missing.md"
            )
        factory.assert_not_called()

    def test_existing_output_is_never_overwritten(
        self, inputs: tuple[Path, Path, Path]
    ) -> None:
        inputs[2].mkdir()
        previous = inputs[2] / "final.srt"
        previous.write_text("Previous run")
        factory = Mock()
        with pytest.raises(FileExistsError):
            LocalLabeler(factory).run(*inputs)
        factory.assert_not_called()
        assert previous.read_text() == "Previous run"

    def test_missing_preset_does_not_invoke_model(
        self, inputs: tuple[Path, Path, Path], backend: Mock
    ) -> None:
        backend.preset_manager.ensure_preset.return_value = (False, None, "Not found")
        with pytest.raises(RequestError, match="Not found"):
            LocalLabeler(Mock(return_value=backend)).run(*inputs)
        backend.run_template.assert_not_called()

    @pytest.mark.parametrize("success", [True, False])
    def test_cli_exit_status(
        self,
        inputs: tuple[Path, Path, Path],
        backend: Mock,
        monkeypatch: pytest.MonkeyPatch,
        success: bool,
    ) -> None:
        backend.run_template.return_value = (success, RESPONSE, "Failed")
        monkeypatch.setattr("srt_labeler.local.ApiBackend", Mock(return_value=backend))
        code = main(
            [
                "--srt",
                str(inputs[0]),
                "--audio",
                str(inputs[1]),
                "--output-dir",
                str(inputs[2]),
            ]
        )
        assert code == (0 if success else 1)
        preset = backend.run_template.call_args.kwargs["overrides"][
            "request_overrides"
        ]["preset"]
        assert preset == LWE_DEFAULT_PRESET
