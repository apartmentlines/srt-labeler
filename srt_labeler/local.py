"""Run one labeling preset against local SRT and WAV files."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
import logging
import json
import os
import signal
from pathlib import Path
import sys
import time
from typing import Any, cast
import uuid
import yaml

from lwe import ApiBackend
from lwe.core.config import Config
import srt

from .audio import AudioAttachmentFactory
from .config import LWE_CONFIG_DIR, LWE_DATA_DIR
from .constants import LWE_DEFAULT_PRESET, LWE_TRANSCRIPTION_TEMPLATE, UUID_SHORT_LENGTH
from .labeling import LabelingEngine
from .logger import Logger
from .merger import SrtMergeError, SrtMerger
from .errors import ModelResponseFormattingError
from .token_usage import TokenUsage


class LocalLabeler:
    """Load local inputs and save one attempt without production API operations."""

    def __init__(
        self,
        backend_factory: Callable[[Config], ApiBackend] | None = None,
        debug: bool = False,
    ) -> None:
        self.backend_factory: Callable[[Config], ApiBackend] = (
            backend_factory or ApiBackend
        )
        self.debug: bool = debug
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)

    def run(
        self,
        srt_path: Path,
        audio_path: Path,
        output_dir: Path,
        preset: str = LWE_DEFAULT_PRESET,
        template_file: Path | None = None,
        settings_file: Path | None = None,
        identifier: str | None = None,
    ) -> None:
        """Write response.txt and, on successful extraction and merge, final.srt.

        :raises Exception: If input loading, execution, merging, or writing fails.
        :raises FileExistsError: If the output directory already exists.
        """
        self.log.debug(f"Reading SRT from {srt_path} and audio from {audio_path}")
        transcription = srt_path.read_text(encoding="utf-8-sig")
        if not list(srt.parse(transcription)):
            raise ValueError("Input SRT contains no subtitles")
        AudioAttachmentFactory.validate_local(audio_path)
        if template_file is not None:
            template_file = template_file.resolve()
            _ = template_file.read_text(encoding="utf-8")
        output_dir.mkdir(parents=True, exist_ok=False)
        settings = json.loads(settings_file.read_text()) if settings_file else None
        backend = self._create_backend(preset, template_file, output_dir, settings)
        engine = LabelingEngine(backend)
        attachment = engine.audio_attachment(preset, audio_path)
        template_name = (
            template_file.name if template_file else LWE_TRANSCRIPTION_TEMPLATE
        )
        self.log.info(f"Running preset {preset} with template {template_name}")
        with TokenUsage.capture(output_dir / "usage.json") as usage:
            success, response, error = engine.run_template(
                template_name,
                {
                    "transcription": transcription,
                    "identifier": identifier or str(uuid.uuid4())[:UUID_SHORT_LENGTH],
                },
                {"request_overrides": {"preset": preset, "files": [attachment]}},
                usage=usage,
            )
        response_path = output_dir / "response.txt"
        response_path.write_text(response, encoding="utf-8")
        self.log.info(f"Saved model response to {response_path}")
        if not success:
            raise RuntimeError(error or "Model request failed")
        extracted = engine.extract_transcript(response)
        merged = SrtMerger(debug=self.debug).merge(transcription, extracted)
        final_path = output_dir / "final.srt"
        final_path.write_text(merged, encoding="utf-8")
        self.log.info(f"Saved labeled SRT to {final_path}")

    def _create_backend(
        self,
        preset: str,
        template_file: Path | None,
        output_dir: Path,
        settings: dict[str, Any] | None = None,
    ) -> ApiBackend:
        """Reuse LWE directories with in-memory overrides and no conversation history."""
        config = Config(
            config_dir=os.environ.get("LWE_CONFIG_DIR", str(LWE_CONFIG_DIR)),
            data_dir=os.environ.get("LWE_DATA_DIR", str(LWE_DATA_DIR)),
        )
        if settings is None:
            config.load_from_file()
        else:
            config.config = settings["config"]
            config.set("database", f"sqlite:///{output_dir.resolve() / 'runtime.db'}")
            frozen_presets = output_dir / "frozen-presets"
            frozen_presets.mkdir()
            (frozen_presets / f"{preset}.yaml").write_text(
                yaml.safe_dump(
                    {
                        "metadata": settings["metadata"],
                        "model_customizations": settings["customizations"],
                    }
                )
            )
            setattr(config.args, "presets_dir", [str(frozen_presets.resolve())])
        config.set("model.default_preset", preset)
        config.set("backend_options.auto_create_first_user", None)
        config.set("backend_options.default_user", None)
        config.set("backend_options.default_conversation_id", None)
        config.set("chat.log.enabled", False)
        if template_file:
            directories = cast(list[str], config.get("directories.templates"))
            setattr(
                config.args, "templates_dir", [str(template_file.parent), *directories]
            )
        if self.debug:
            config.set("log.console.level", "DEBUG")
            config.set("debug.log.enabled", True)
            config.set("debug.log.level", "DEBUG")
            config.set(
                "debug.log.filepath", str((output_dir / "lwe-debug.log").resolve())
            )
        self.log.debug(f"Initializing LWE with preset {preset}")
        try:
            backend = self.backend_factory(config)
        finally:
            self.log.disabled = False
        backend.log.disabled = False
        if settings is not None:
            backend.preset_manager.presets[preset] = (
                settings["metadata"],
                settings["customizations"],
            )
        if self.debug:
            logging.basicConfig(
                level=cast(str, config.get("debug.log.level")),
                format=cast(str, config.get("debug.log.format")),
                handlers=[
                    logging.StreamHandler(),
                    logging.FileHandler(
                        cast(str, config.get("debug.log.filepath")), encoding="utf-8"
                    ),
                ],
                force=True,
            )
        return backend


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse a single local labeling run."""
    parser = argparse.ArgumentParser(
        description=(
            "Label a local SRT using a local WAV and one model preset. "
            + "Inputs are sent to the selected model provider; no production API is used."
        )
    )
    parser.add_argument("--srt", type=Path, required=True, help="Input UTF-8 SRT file")
    parser.add_argument(
        "--audio", type=Path, required=True, help="Input WAV audio file"
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New directory for output files"
    )
    parser.add_argument(
        "--preset", default=LWE_DEFAULT_PRESET, help="LWE preset (default: %(default)s)"
    )
    parser.add_argument(
        "--template-file",
        type=Path,
        help="Local LWE template overriding the bundled prompt",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument(
        "--settings-file",
        type=Path,
        help="Frozen evaluation settings; credentials remain in the environment",
    )
    parser.add_argument(
        "--identifier",
        help="Recorded request identifier for reproducible template construction",
    )
    parser.add_argument(
        "--start-gate",
        type=Path,
        help="Collector launch handshake; no request before ownership is durable",
    )
    parser.add_argument(
        "--completion-file",
        type=Path,
        help="Durable local invocation outcome for collector recovery",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=float,
        help="Child process deadline, including orphaned execution (POSIX)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run local labeling, returning 0 on success and 1 on execution failure."""
    args = parse_arguments(argv)
    runner = LocalLabeler(debug=args.debug)
    code = 0
    error_message = None
    error_type = None
    failure_phase = None
    try:
        if args.deadline_seconds is not None:
            if args.deadline_seconds <= 0:
                raise ValueError("Child deadline must be positive")
            signal.signal(signal.SIGALRM, signal.SIG_DFL)
            signal.setitimer(signal.ITIMER_REAL, args.deadline_seconds)
        if args.start_gate:
            deadline = time.monotonic() + 60
            while not args.start_gate.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Collector never authorized the launch handshake"
                    )
                time.sleep(0.05)
            if json.loads(args.start_gate.read_text())["pid"] != os.getpid():
                raise ValueError("Launch handshake belongs to another process")
        runner.run(
            args.srt,
            args.audio,
            args.output_dir,
            args.preset,
            args.template_file,
            args.settings_file,
            args.identifier,
        )
    except Exception as error:
        runner.log.error(f"Local labeling failed: {error}", exc_info=args.debug)
        code = 1
        error_message = str(error)
        error_type = type(error).__name__
        failure_phase = (
            "format"
            if isinstance(error, (ModelResponseFormattingError, SrtMergeError))
            else "execution"
        )
    if args.completion_file:
        temporary = args.completion_file.with_suffix(".tmp")
        with temporary.open("w") as stream:
            json.dump(
                {
                    "returncode": code,
                    "error": error_message,
                    "error_type": error_type,
                    "failure_phase": failure_phase,
                    "finished_at": time.time(),
                },
                stream,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(args.completion_file)
    if args.deadline_seconds is not None:
        signal.setitimer(signal.ITIMER_REAL, 0)
    return code


if __name__ == "__main__":
    sys.exit(main())
