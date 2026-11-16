"""Run reproducible preset experiments and score exact preserved SRT blocks."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import tempfile
import uuid
from urllib.parse import parse_qsl, urlsplit
from typing import Any

from .eval_dataset import Case, Dataset, digest, speaker, subtitles
from .token_usage import TokenUsage


class Evaluation:
    """Execute isolated local CLI attempts and retain failures in the denominator."""

    @staticmethod
    def read_usage(path: Path) -> dict[str, Any]:
        """Read saved usage without treating absent or malformed data as zero."""
        if not path.exists():
            return TokenUsage().report()
        try:
            data = json.loads(path.read_text())
            if data["schema_version"] != 1 or not data["observations"]:
                raise ValueError("Unsupported or empty usage artifact")
            for observation in data["observations"]:
                if not isinstance(observation.get("tokens"), dict):
                    raise ValueError("Invalid usage observation")
            data["summary"] = TokenUsage.aggregate(data["observations"])
            return data
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            data = TokenUsage().report()
            data["error"] = str(error)
            return data

    @staticmethod
    def preset_identity(preset: str) -> dict[str, Any]:
        """Record resolved preset identity without exporting credentials."""
        from lwe.core.config import Config
        from lwe.core.preset_manager import PresetManager
        from .config import LWE_CONFIG_DIR, LWE_DATA_DIR

        config = Config(
            config_dir=os.environ.get("LWE_CONFIG_DIR", str(LWE_CONFIG_DIR)),
            data_dir=os.environ.get("LWE_DATA_DIR", str(LWE_DATA_DIR)),
        )
        config.load_from_file()
        success, resolved, message = PresetManager(config).ensure_preset(preset)
        if not success or resolved is None or isinstance(resolved, str):
            raise ValueError(message)
        metadata, customizations = resolved
        return {
            "provider": metadata.get("provider"),
            "model": customizations.get("model", customizations.get("model_name")),
            "preset_sha256": digest(Path(metadata["filepath"])),
        }

    @staticmethod
    def score(case: Case, prediction: Path) -> dict[str, Any]:
        """Require structural fidelity on every block, including excluded blocks."""
        original = subtitles(case.source)
        expected = subtitles(case.reference)
        actual = subtitles(prediction)
        if len(actual) != len(original):
            raise ValueError("Changed subtitle count")
        mismatches = []
        eligible = 0
        confusion = {
            a: {b: 0 for b in ("Operator", "Caller")} for a in ("Operator", "Caller")
        }
        for source, reference, result in zip(original, expected, actual):
            label, content = speaker(result)
            if (source.index, source.start, source.end, source.content) != (
                result.index,
                result.start,
                result.end,
                content,
            ):
                raise ValueError(f"Changed source block {source.index}")
            if source.index in case.excluded:
                continue
            eligible += 1
            target = speaker(reference)[0]
            confusion[target][label] += 1
            if target != label:
                mismatches.append(
                    {"block": source.index, "expected": target, "actual": label}
                )
        return {
            "eligible": eligible,
            "correct": eligible - len(mismatches),
            "excluded": len(case.excluded),
            "mismatches": mismatches,
            "confusion": confusion,
        }

    def run(
        self,
        dataset: Dataset,
        output: Path,
        preset: str,
        template: Path,
        timeout: float,
        debug: bool,
        predictions: Path | None = None,
    ) -> dict[str, Any]:
        """Prepare and execute one immutable run, or evaluate preserved outputs."""
        if predictions is not None:
            return Experiment().rescore(dataset, predictions, output)
        Experiment().prepare(dataset, output, preset, template, timeout)
        return Experiment().collect(output, debug=debug)

    @staticmethod
    def summarize(results: list[dict[str, Any]], mode: str) -> dict[str, Any]:
        """Count all selected cases, explicitly exposing pending and ungraded work."""
        eligible = sum(r["eligible"] for r in results)
        correct = sum(r["correct"] for r in results)
        successful = [r for r in results if r["status"] == "ok"]
        covered = sum(r["eligible"] for r in successful)
        pending = sum(r["status"] == "pending" for r in results)
        return {
            "calls": len(results),
            "successful_calls": len(successful),
            "failed_calls": sum(r["status"] == "failed" for r in results),
            "pending_calls": pending,
            "grading_complete": pending == 0,
            "eligible_blocks": eligible,
            "covered_blocks": covered,
            "excluded_blocks": sum(r["excluded"] for r in results),
            "correct_blocks": correct,
            "accuracy_including_failures": correct / eligible if eligible else None,
            "accuracy_on_successful_calls": correct / covered if covered else None,
            "accuracy_is_provisional": bool(pending),
            "macro_call_accuracy": (
                sum(r["correct"] / r["eligible"] for r in results if r["eligible"])
                / sum(bool(r["eligible"]) for r in results)
                if eligible
                else None
            ),
            "seconds": sum(r.get("seconds", 0) for r in results),
            "timing_boundary": "local CLI wall time including startup; rescore uses scoring time",
            "usage": TokenUsage.aggregate(
                [
                    observation
                    for r in results
                    for observation in r["usage"]["observations"]
                ]
            ),
            "usage_origin": (
                "saved-predictions; no new model requests"
                if mode == "rescore"
                else "selected attempts"
            ),
        }

    @staticmethod
    def compare(first: Path, second: Path) -> dict[str, Any]:
        """Compare runs only when inputs, references and scoring masks match."""
        a, b = [
            json.loads((p / "experiment.json").read_text()) for p in (first, second)
        ]
        for metadata in (a, b):
            if metadata.get("schema_version") != 2:
                raise ValueError("Unsupported experiment schema_version")
            if metadata.get("configuration_sha256") != Experiment.metadata_hash(
                metadata
            ):
                raise ValueError("Changed experiment configuration")
        if a["cases"] != b["cases"]:
            raise ValueError(
                "Cannot compare different cases, input hashes or exclusions"
            )
        left, right = [
            json.loads((p / "results.json").read_text()) for p in (first, second)
        ]
        if [r["id"] for r in left] != [r["id"] for r in right] or len(left) != len(
            a["cases"]
        ):
            raise ValueError("Incomplete experiment")
        expected_ids = [row["id"] for row in a["cases"]]
        if [r["id"] for r in left] != expected_ids or any(
            row["status"] == "pending" for row in left + right
        ):
            raise ValueError(
                "Incomplete experiment or results disagree with selected cases"
            )
        if a.get("scoring_policy") != b.get("scoring_policy") or a.get(
            "scorer_sha256"
        ) != b.get("scorer_sha256"):
            raise ValueError(
                "Different scoring policies; rescore both with the same evaluator"
            )
        changes: list[dict[str, Any]] = [
            {
                "id": x["id"],
                "correct_delta": y["correct"] - x["correct"],
                "before_status": x["status"],
                "after_status": y["status"],
                "corrected_blocks": (
                    sorted(
                        {m["block"] for m in x.get("mismatches", [])}
                        - {m["block"] for m in y.get("mismatches", [])}
                    )
                    if x["status"] == y["status"] == "ok"
                    else None
                ),
                "regressed_blocks": (
                    sorted(
                        {m["block"] for m in y.get("mismatches", [])}
                        - {m["block"] for m in x.get("mismatches", [])}
                    )
                    if x["status"] == y["status"] == "ok"
                    else None
                ),
            }
            for x, y in zip(left, right)
        ]
        return {
            "before": str(first),
            "after": str(second),
            "correct_delta": sum(c["correct_delta"] for c in changes),
            "eligible_blocks": sum(row["eligible"] for row in left),
            "paired_calls": len(left),
            "accuracy_delta": (
                sum(c["correct_delta"] for c in changes)
                / sum(row["eligible"] for row in left)
                if any(row["eligible"] for row in left)
                else None
            ),
            "configuration_differences": {
                key: {"before": a.get(key), "after": b.get(key)}
                for key in (
                    "resolved_preset",
                    "template_sha256",
                    "settings_sha256",
                    "environment",
                )
                if a.get(key) != b.get(key)
            },
            "calls": changes,
            "usage_before": TokenUsage.aggregate(
                [
                    o
                    for r in left
                    for o in r.get("usage", TokenUsage().report())["observations"]
                ]
            ),
            "usage_after": TokenUsage.aggregate(
                [
                    o
                    for r in right
                    for o in r.get("usage", TokenUsage().report())["observations"]
                ]
            ),
        }


class Experiment:
    """Freeze execution, own collectors and preserve every attempt through recovery."""

    POLICY: str = "binary-speaker-v2"

    @staticmethod
    def now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def metadata_hash(metadata: dict[str, Any]) -> str:
        value = {
            key: item for key, item in metadata.items() if key != "configuration_sha256"
        }
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    @staticmethod
    def atomic(path: Path, value: Any) -> None:
        """Commit complete JSON records with same-filesystem atomic replacement."""
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    @staticmethod
    def settings(preset: str) -> dict[str, Any]:
        """Resolve local configuration without initializing a model or saving secrets."""
        from lwe.core.config import Config
        from lwe.core.preset_manager import PresetManager
        from .config import LWE_CONFIG_DIR, LWE_DATA_DIR

        config = Config(
            config_dir=os.environ.get("LWE_CONFIG_DIR", str(LWE_CONFIG_DIR)),
            data_dir=os.environ.get("LWE_DATA_DIR", str(LWE_DATA_DIR)),
        )
        config.load_from_file()
        success, resolved, message = PresetManager(config).ensure_preset(preset)
        if not success or not isinstance(resolved, tuple):
            raise ValueError(message)
        metadata, customizations = resolved
        metadata = {key: value for key, value in metadata.items() if key != "filepath"}
        effective = copy.deepcopy(config.config)
        effective["database"] = None
        frozen = {
            "metadata": metadata,
            "customizations": customizations,
            "config": effective,
        }
        Experiment.reject_secrets(frozen)
        return frozen

    @staticmethod
    def reject_secrets(value: Any) -> None:
        """Refuse credential-bearing configuration; credentials belong in the environment."""
        forbidden = {
            "api_key",
            "apikey",
            "access_token",
            "auth_token",
            "authorization",
            "password",
            "secret",
            "credentials",
            "headers",
            "default_headers",
        }
        if isinstance(value, dict):
            for key, item in value.items():
                normalized = key.lower().replace("-", "_")
                if (
                    normalized in forbidden
                    or normalized.endswith(
                        ("_api_key", "_password", "_secret", "_token")
                    )
                ) and item:
                    raise ValueError(
                        "Move credentials/headers to the provider environment before freezing settings"
                    )
                Experiment.reject_secrets(item)
        elif isinstance(value, list):
            for item in value:
                Experiment.reject_secrets(item)
        elif isinstance(value, str) and value.startswith(("https://", "http://")):
            parsed = urlsplit(value)
            if (
                parsed.username
                or parsed.password
                or any(
                    key.lower().replace("-", "_") in forbidden
                    for key, _ in parse_qsl(parsed.query)
                )
            ):
                raise ValueError("Credential-bearing URLs cannot be frozen")

    @staticmethod
    def implementation() -> dict[str, str]:
        return {
            path.name: digest(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }

    @staticmethod
    def environment() -> dict[str, str]:
        return {
            str(dist.metadata["Name"]): dist.version
            for dist in importlib.metadata.distributions()
            if str(dist.metadata["Name"])
            .lower()
            .startswith(
                (
                    "langchain",
                    "lwe-",
                    "llm-workflow",
                    "openai",
                    "google-genai",
                    "srt",
                    "pydantic",
                )
            )
        }

    @staticmethod
    def freeze_corpus(dataset: Dataset, directory: Path) -> Dataset:
        """Copy content-addressed evidence into the run, preserving corpus identity."""
        directory.mkdir()
        data = copy.deepcopy(dataset.data)
        for row in data["cases"]:
            folder = directory / row["id"]
            folder.mkdir()
            for key in ("source", "reference", "audio"):
                source = dataset.path.parent / row[key]
                destination = folder / ("audio.wav" if key == "audio" else f"{key}.srt")
                shutil.copyfile(source, destination)
                row[key] = str(destination.relative_to(directory))
            original = dataset.path.parent / row["id"] / "original-draft.srt"
            if original.exists():
                shutil.copyfile(original, folder / original.name)
        for index, artifact in enumerate(data.get("artifacts", [])):
            source = dataset.path.parent / artifact["path"]
            destination = directory / "provenance" / f"{index}-{source.name}"
            destination.parent.mkdir(exist_ok=True)
            shutil.copyfile(source, destination)
            artifact["path"] = str(destination.relative_to(directory))
        Experiment.atomic(directory / "dataset.json", data)
        return Dataset(directory / "dataset.json")

    def prepare(
        self,
        dataset: Dataset,
        output: Path,
        preset: str,
        template: Path,
        timeout: float = 600,
        retry_limit: int = 0,
        settings: dict[str, Any] | None = None,
        selection: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Freeze a bounded experiment without making provider requests."""
        import re

        dataset.require_approved()
        if timeout <= 0 or retry_limit < 0:
            raise ValueError("Timeout must be positive and retry limit nonnegative")
        if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9_.-]*", preset):
            raise ValueError("Unsafe preset name")
        template_bytes = template.read_bytes()
        if re.search(rb"{%\s*(include|extends|import|from)\b", template_bytes):
            raise ValueError(
                "Freeze a self-contained template; external Jinja dependencies are not supported"
            )
        frozen_settings = (
            copy.deepcopy(settings) if settings is not None else self.settings(preset)
        )
        self.reject_secrets(frozen_settings)
        output = output.resolve()
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".experiment-", dir=output.parent
        ) as temporary:
            staging = Path(temporary) / "run"
            staging.mkdir()
            frozen = self.freeze_corpus(dataset, staging / "corpus")
            (staging / "template.md").write_bytes(template_bytes)
            self.atomic(staging / "settings.json", frozen_settings)
            rows = {row["id"]: row for row in frozen.data["cases"]}
            metadata = {
                "schema_version": 2,
                "run_id": str(uuid.uuid4()),
                "mode": "run",
                "created_at": self.now(),
                "preset": preset,
                "resolved_preset": {
                    "provider": frozen_settings["metadata"].get("provider"),
                    "model": frozen_settings["customizations"].get(
                        "model", frozen_settings["customizations"].get("model_name")
                    ),
                    "settings_sha256": digest(staging / "settings.json"),
                },
                "cases": [
                    {
                        "id": case.id,
                        "revision": rows[case.id]["revision"],
                        "split": case.split,
                        "hashes": rows[case.id]["hashes"],
                        "excluded": rows[case.id]["excluded"],
                        "holdout_exposed": rows[case.id].get("holdout_exposed"),
                    }
                    for case in dataset.cases
                ],
                "corpus_sha256": frozen.data["content_sha256"],
                "template_sha256": digest(staging / "template.md"),
                "settings_sha256": digest(staging / "settings.json"),
                "scoring_policy": self.POLICY,
                "scorer_sha256": digest(Path(__file__)),
                "implementation_sha256": self.implementation(),
                "environment": self.environment(),
                "python": sys.version,
                "timeout_seconds": timeout,
                "retry_limit": retry_limit,
                "request_limit": len(dataset.cases) * (retry_limit + 1),
                "concurrency": 1,
                "selection": selection
                or {"ordered_ids": [case.id for case in dataset.cases]},
                "provider_retry_policy": "SDK settings frozen when explicit; defaults depend on recorded installed versions; unreported retries remain unknown",
            }
            metadata["configuration_sha256"] = self.metadata_hash(metadata)
            self.atomic(staging / "experiment.json", metadata)
            (staging / "attempts").mkdir()
            staging.rename(output)
        self.refresh(output)
        return metadata

    @staticmethod
    def process_identity(pid: int) -> str | None:
        """Distinguish a live local process from a recycled PID, including zombies."""
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            if fields[0] == "Z":
                return None
            return fields[19]
        except (OSError, IndexError):
            return None

    def validate_run(self, output: Path) -> tuple[dict[str, Any], Dataset]:
        """Reject changed frozen data or execution code before any new request."""
        metadata = json.loads((output / "experiment.json").read_text())
        if metadata.get("configuration_sha256") != self.metadata_hash(metadata):
            raise ValueError("Changed experiment configuration")
        if metadata.get("schema_version") != 2 or metadata.get("mode") != "run":
            raise ValueError("This artifact is not a resumable collection")
        dataset = Dataset(output / "corpus/dataset.json")
        dataset.require_approved()
        if dataset.data["content_sha256"] != metadata["corpus_sha256"]:
            raise ValueError("Changed frozen corpus")
        for filename in ("template", "settings"):
            suffix = "md" if filename == "template" else "json"
            if (
                digest(output / f"{filename}.{suffix}")
                != metadata[f"{filename}_sha256"]
            ):
                raise ValueError(f"Changed frozen {filename}")
        if (
            metadata["implementation_sha256"] != self.implementation()
            or metadata["environment"] != self.environment()
            or metadata["python"] != sys.version
        ):
            raise ValueError(
                "Execution implementation/environment changed; prepare a new run"
            )
        by_id = {case.id: case for case in dataset.cases}
        dataset.cases = [by_id[row["id"]] for row in metadata["cases"]]
        return metadata, dataset

    def collect(
        self,
        output: Path,
        debug: bool = False,
        resume: bool = False,
        retry: list[str] | None = None,
        only_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Own a collector, recover terminal attempts, then run only requested work."""
        output = output.resolve()
        with (output / "collector.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError("Another collector owns this run") from error
            metadata, dataset = self.validate_run(output)
            owner = {
                "pid": os.getpid(),
                "process_identity": self.process_identity(os.getpid()),
                "host": socket.gethostname(),
                "started_at": self.now(),
            }
            self.atomic(output / "owner.json", owner)
            self.recover(output)
            if resume:
                (output / "pause.json").unlink(missing_ok=True)
            selected_retry = set(retry or [])
            identifiers = {case.id for case in dataset.cases}
            if not selected_retry.issubset(identifiers) or (
                only_ids is not None and not set(only_ids).issubset(identifiers)
            ):
                raise ValueError("Unknown retry case")
            for case in dataset.cases:
                if only_ids is not None and case.id not in only_ids:
                    continue
                if (output / "pause.json").exists():
                    break
                attempts = self.attempts(output, case.id)
                if attempts and case.id not in selected_retry:
                    continue
                if attempts and (
                    len(attempts) > metadata["retry_limit"]
                    or self.result(case, output, attempts[-1])["status"] == "ok"
                ):
                    raise ValueError(
                        f"Retry not permitted by frozen limits or case already succeeded: {case.id}"
                    )
                self.launch(output, case, metadata, len(attempts) + 1, debug)
                self.refresh(output)
            summary = self.refresh(output)
            self.atomic(output / "owner.json", {**owner, "finished_at": self.now()})
            return summary

    @staticmethod
    def attempts(output: Path, identifier: str) -> list[dict[str, Any]]:
        return [
            json.loads(path.read_text())
            for path in sorted(
                (output / "attempts" / identifier).glob("*/attempt.json")
            )
        ]

    def launch(
        self,
        output: Path,
        case: Case,
        metadata: dict[str, Any],
        number: int,
        debug: bool,
    ) -> None:
        """Durably own the child before releasing its provider-request launch gate."""
        directory = output / "attempts" / case.id / f"{number:04d}"
        directory.mkdir(parents=True)
        attempt: dict[str, Any] = {
            "attempt_id": str(uuid.uuid4()),
            "id": case.id,
            "number": number,
            "directory": str(directory.relative_to(output)),
            "status": "running",
            "started_at": self.now(),
            "started_epoch": time.time(),
            "request_identifier": uuid.uuid4().hex[:8],
            "pid": None,
            "host": socket.gethostname(),
            "process_identity": None,
        }
        self.atomic(directory / "attempt.json", attempt)
        command = [
            sys.executable,
            "-m",
            "srt_labeler.local",
            "--srt",
            str(case.source),
            "--audio",
            str(case.audio),
            "--output-dir",
            str(directory / "output"),
            "--preset",
            metadata["preset"],
            "--template-file",
            str(output / "template.md"),
            "--settings-file",
            str(output / "settings.json"),
            "--identifier",
            attempt["request_identifier"],
            "--start-gate",
            str(directory / "gate.json"),
            "--completion-file",
            str(directory / "completion.json"),
            "--deadline-seconds",
            str(metadata["timeout_seconds"]),
        ]
        if debug:
            command.append("--debug")
        started = time.monotonic()
        with (directory / "process.log").open("w") as log:
            process = subprocess.Popen(
                command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            attempt.update(
                pid=process.pid, process_identity=self.process_identity(process.pid)
            )
            self.atomic(directory / "attempt.json", attempt)
            self.atomic(directory / "gate.json", {"pid": process.pid})
            try:
                attempt["returncode"] = process.wait(
                    timeout=metadata["timeout_seconds"]
                )
                attempt["status"] = "ok" if attempt["returncode"] == 0 else "failed"
                if attempt["returncode"] == -signal.SIGALRM:
                    attempt.update(
                        status="timeout", error="Child process deadline reached"
                    )
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                attempt.update(status="timeout", error="Per-call timeout reached")
            except BaseException:
                self.atomic(
                    output / "pause.json",
                    {"requested_at": self.now(), "reason": "collector interrupted"},
                )
                raise
        attempt.update(
            seconds=round(time.monotonic() - started, 3),
            seconds_source="monotonic local CLI wall time",
            finished_at=self.now(),
        )
        completion = directory / "completion.json"
        if completion.exists():
            outcome = json.loads(completion.read_text())
            attempt.update(
                error=outcome.get("error"),
                error_type=outcome.get("error_type"),
                failure_phase=outcome.get("failure_phase"),
            )
        self.finish_attempt(directory, attempt)

    def finish_attempt(self, directory: Path, attempt: dict[str, Any]) -> None:
        """Pin output bytes before publishing a terminal attempt."""
        attempt["artifacts"] = {
            path.name: digest(path)
            for path in (directory / "output").glob("*")
            if path.is_file()
            and path.name in {"response.txt", "final.srt", "usage.json"}
        }
        self.atomic(directory / "attempt.json", attempt)

    def recover(self, output: Path) -> None:
        """Recover recorded outcomes without treating a live orphan as stopped."""
        for path in sorted((output / "attempts").glob("*/*/attempt.json")):
            attempt = json.loads(path.read_text())
            if attempt["status"] != "running":
                continue
            if attempt.get("pid"):
                if attempt["host"] != socket.gethostname():
                    raise ValueError(
                        "Cannot establish remote attempt ownership; recover on the original host"
                    )
                identity = self.process_identity(attempt["pid"])
                if identity and identity == attempt["process_identity"]:
                    raise ValueError(
                        "Recorded model process is still running; wait for its completion"
                    )
            completion = path.parent / "completion.json"
            if completion.exists():
                record = json.loads(completion.read_text())
                attempt.update(
                    status="ok" if record["returncode"] == 0 else "failed",
                    returncode=record["returncode"],
                    error=record.get("error"),
                    error_type=record.get("error_type"),
                    failure_phase=record.get("failure_phase"),
                    seconds=max(0, record["finished_at"] - attempt["started_epoch"]),
                    seconds_source="recovered wall-clock interval; excludes collector downtime",
                )
            else:
                attempt.update(
                    status="interrupted",
                    seconds=0,
                    seconds_source="unknown; omitted from measured time",
                    error="Interrupted without durable completion; explicit retry required",
                )
            attempt["recovered_at"] = self.now()
            self.finish_attempt(path.parent, attempt)

    def result(
        self, case: Case, output: Path, attempt: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Keep collection, structural validity and label grading distinguishable."""
        result: dict[str, Any] = {
            "id": case.id,
            "split": case.split,
            "eligible": len(subtitles(case.source)) - len(case.excluded),
            "excluded": len(case.excluded),
            "correct": 0,
            "status": "pending",
            "collection_status": "pending",
            "structure_status": "not_checked",
            "grading_status": "ungraded",
            "usage": TokenUsage().report(),
            "seconds": 0,
        }
        if attempt is None or attempt["status"] in {"running", "pending"}:
            return result
        directory = output / attempt["directory"] / "output"
        result.update(
            status="failed",
            collection_status=attempt["status"],
            attempt_id=attempt["attempt_id"],
            artifact_directory=str(directory.relative_to(output)),
            seconds=attempt.get("seconds", 0),
            seconds_source=attempt.get("seconds_source"),
            usage=Evaluation.read_usage(directory / "usage.json"),
            error=attempt.get("error"),
        )
        for filename, expected in attempt.get("artifacts", {}).items():
            if digest(directory / filename) != expected:
                raise ValueError(
                    f"Changed preserved attempt artifact: {case.id}/{filename}"
                )
        result["artifacts"] = attempt.get("artifacts", {})
        if attempt.get("failure_phase") == "format":
            result.update(
                collection_status="ok",
                structure_status="invalid",
                grading_status="invalid",
                error_type=attempt.get("error_type"),
            )
            return result
        if attempt["status"] != "ok":
            return result
        try:
            result.update(Evaluation.score(case, directory / "final.srt"))
            result.update(
                status="ok", structure_status="valid", grading_status="graded"
            )
        except (OSError, ValueError) as error:
            result.update(
                error=str(error), structure_status="invalid", grading_status="invalid"
            )
        return result

    def refresh(self, output: Path) -> dict[str, Any]:
        """Rebuild metrics from immutable attempts, including pending selected cases."""
        metadata = json.loads((output / "experiment.json").read_text())
        dataset = Dataset(output / "corpus/dataset.json")
        cases = {case.id: case for case in dataset.cases}
        results = []
        all_usage = []
        for row in metadata["cases"]:
            attempts = self.attempts(output, row["id"])
            results.append(
                self.result(
                    cases[row["id"]], output, attempts[-1] if attempts else None
                )
            )
            for attempt in attempts:
                usage = Evaluation.read_usage(
                    output / attempt["directory"] / "output/usage.json"
                )
                all_usage.extend(usage["observations"])
        summary = Evaluation.summarize(results, metadata["mode"])
        summary["all_attempt_usage"] = (
            TokenUsage.aggregate(all_usage) if all_usage else None
        )
        summary["retry_selection"] = (
            "latest explicitly requested attempt per case; prior attempts retained"
        )
        self.atomic(output / "results.json", results)
        self.atomic(output / "summary.json", summary)
        return summary

    def status(self, output: Path) -> dict[str, Any]:
        """Report actual collector/child liveness without launching or mutating attempts."""
        with (output / "collector.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                active = False
            except BlockingIOError:
                active = True
        running = []
        for path in (output / "attempts").glob("*/*/attempt.json"):
            attempt = json.loads(path.read_text())
            if attempt["status"] == "running":
                alive = (
                    None
                    if attempt["host"] != socket.gethostname()
                    else bool(
                        attempt.get("pid")
                        and self.process_identity(attempt["pid"])
                        == attempt["process_identity"]
                    )
                )
                running.append(
                    {
                        "attempt_id": attempt["attempt_id"],
                        "pid": attempt.get("pid"),
                        "alive": alive,
                    }
                )
        return {
            "run": str(output),
            "collector_active": active,
            "running_attempts": running,
            "pause_requested": (output / "pause.json").exists(),
            "summary": json.loads((output / "summary.json").read_text()),
        }

    def rescore(
        self, dataset: Dataset, predictions: Path, output: Path
    ) -> dict[str, Any]:
        """Create a child evaluation of preserved outputs only when system inputs match."""
        dataset.require_approved()
        origin_path = predictions / "experiment.json"
        if not origin_path.exists():
            raise ValueError(
                "Rescore requires originating experiment.json with pinned source/audio identities"
            )
        origin = json.loads(origin_path.read_text())
        if origin.get("schema_version") != 2:
            raise ValueError("Unsupported experiment schema_version")
        if origin.get("configuration_sha256") != self.metadata_hash(origin):
            raise ValueError("Changed originating experiment configuration")
        origins = {row["id"]: row for row in origin["cases"]}
        if [case.id for case in dataset.cases] != [
            row["id"] for row in origin["cases"]
        ]:
            raise ValueError("Rescore requires the same ordered cases")
        for case in dataset.cases:
            if any(
                digest(getattr(case, key)) != origins[case.id]["hashes"][key]
                for key in ("source", "audio")
            ):
                raise ValueError("Changed system inputs require recollection")
        output.mkdir(parents=True, exist_ok=False)
        self.freeze_corpus(dataset, output / "corpus")
        prior_results = (
            json.loads((predictions / "results.json").read_text())
            if (predictions / "results.json").exists()
            else []
        )
        prior = {row["id"]: row for row in prior_results}
        if not prior_results:
            raise ValueError("Origin requires its complete results index")
        if prior_results and (
            len(prior) != len(prior_results)
            or [row["id"] for row in prior_results]
            != [row["id"] for row in origin["cases"]]
        ):
            raise ValueError(
                "Origin results have duplicate, missing or reordered cases"
            )
        provenance = output / "provenance"
        provenance.mkdir()
        shutil.copyfile(origin_path, provenance / "origin-experiment.json")
        if prior_results:
            shutil.copyfile(
                predictions / "results.json", provenance / "origin-results.json"
            )
        for filename in ("template.md", "settings.json"):
            if (predictions / filename).exists():
                expected = origin.get(filename.split(".")[0] + "_sha256")
                if expected and digest(predictions / filename) != expected:
                    raise ValueError("Changed originating configuration")
                shutil.copyfile(predictions / filename, output / filename)
        metadata = copy.deepcopy(origin)
        metadata.update(
            schema_version=2,
            run_id=str(uuid.uuid4()),
            mode="rescore",
            created_at=self.now(),
            parent_evaluation={
                "run_id": origin.get("run_id"),
                "experiment_sha256": digest(origin_path),
                "path": str(predictions.resolve()),
            },
            scoring_policy=self.POLICY,
            scorer_sha256=digest(Path(__file__)),
            origin_provenance="verified saved output hashes",
            corpus_sha256=dataset.data["content_sha256"],
        )
        rows = {row["id"]: row for row in dataset.data["cases"]}
        metadata["cases"] = [
            {
                "id": c.id,
                "revision": rows[c.id]["revision"],
                "split": c.split,
                "hashes": rows[c.id]["hashes"],
                "excluded": rows[c.id]["excluded"],
                "holdout_exposed": rows[c.id].get("holdout_exposed"),
            }
            for c in dataset.cases
        ]
        metadata["configuration_sha256"] = self.metadata_hash(metadata)
        self.atomic(output / "experiment.json", metadata)
        results = []
        for case in dataset.cases:
            started = time.monotonic()
            old = prior[case.id]
            source = predictions / old["artifact_directory"]
            destination = output / case.id
            destination.mkdir()
            for filename in ("final.srt", "response.txt", "usage.json"):
                if (source / filename).exists():
                    expected = old.get("artifacts", {}).get(filename)
                    if expected is None:
                        raise ValueError("Origin does not pin this output artifact")
                    if expected and digest(source / filename) != expected:
                        raise ValueError("Changed originating output")
                    shutil.copyfile(source / filename, destination / filename)
            attempt = {
                "attempt_id": old["attempt_id"],
                "directory": case.id,
                "status": old["collection_status"],
                "artifacts": {f.name: digest(f) for f in destination.iterdir()},
            }
            attempt_directory = output / "attempts" / case.id / "0001"
            attempt_directory.mkdir(parents=True)
            destination.rename(attempt_directory / "output")
            attempt["directory"] = str(attempt_directory.relative_to(output))
            result = self.result(case, output, attempt)
            result.update(
                seconds=round(time.monotonic() - started, 3),
                seconds_source="rescoring wall time",
                usage_origin="saved-prediction",
            )
            results.append(result)
            self.atomic(attempt_directory / "attempt.json", attempt)
            self.atomic(output / "results.json", results)
        summary = Evaluation.summarize(results, "rescore")
        self.atomic(output / "summary.json", summary)
        return summary


def main() -> int:
    """Parse one immutable experiment configuration."""
    if len(sys.argv) > 1 and sys.argv[1] == "compare":
        parser = argparse.ArgumentParser(
            description="Compare two compatible experiments"
        )
        parser.add_argument("compare")
        parser.add_argument("before", type=Path)
        parser.add_argument("after", type=Path)
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        try:
            print(json.dumps(Evaluation.compare(args.before, args.after), indent=2))
            return 0
        except (ValueError, OSError) as error:
            parser.error(str(error))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", required=True)
    parser.add_argument(
        "--template",
        type=Path,
        required=True,
        help="Exact LWE template to snapshot and execute",
    )
    parser.add_argument("--split", help="Select a named split; omitted runs all cases")
    parser.add_argument(
        "--timeout", type=float, default=600, help="Per-call timeout in seconds"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Rescore ID/final.srt files without model requests",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    try:
        result = Evaluation().run(
            Dataset(args.dataset, args.split),
            args.output,
            args.preset,
            args.template,
            args.timeout,
            args.debug,
            args.predictions,
        )
        print(json.dumps(result, indent=2))
        return int(result["failed_calls"] > 0)
    except (ValueError, OSError) as error:
        logging.error("%s", error, exc_info=args.debug)
        return 1


if __name__ == "__main__":
    sys.exit(main())
