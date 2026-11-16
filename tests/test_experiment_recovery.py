"""Verify frozen inputs, collector ownership, interruptions, retries and lineage offline."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
from typing import Any
from typing_extensions import override

import pytest

from srt_labeler.eval_dataset import Case, Corpus, Dataset, digest
from srt_labeler.evaluate import Evaluation, Experiment
from srt_labeler.token_usage import TokenUsage


class FixtureExperiment(Experiment):
    """Write known application artifacts at the collector's normal attempt boundary."""

    @override
    def launch(
        self,
        output: Path,
        case: Case,
        metadata: dict[str, Any],
        number: int,
        debug: bool,
    ) -> None:
        directory = output / "attempts" / case.id / f"{number:04d}"
        (directory / "output").mkdir(parents=True)
        shutil.copyfile(case.reference, directory / "output/final.srt")
        (directory / "output/response.txt").write_text(
            "<transcript>" + case.reference.read_text() + "</transcript>"
        )
        usage = TokenUsage()
        usage.record({"raw": {}, "tokens": {"input_tokens": 10, "output_tokens": 2}})
        self.atomic(directory / "output/usage.json", usage.report())
        attempt = {
            "id": case.id,
            "number": number,
            "attempt_id": f"fixture-{number}",
            "directory": str(directory.relative_to(output)),
            "status": "ok",
            "seconds": 0.5,
            "seconds_source": "fixture",
            "started_at": self.now(),
        }
        self.finish_attempt(directory, attempt)


@pytest.fixture
def prepared(approved: Dataset, tmp_path: Path) -> Path:
    prompt = tmp_path / "prompt.md"
    prompt.write_text("{{ transcription }} {{ identifier }}")
    output = tmp_path / "run"
    FixtureExperiment().prepare(
        approved,
        output,
        "fixture-3.8-preset",
        prompt,
        retry_limit=1,
        settings={
            "metadata": {"provider": "fixture"},
            "customizations": {"model": "fixture-model"},
            "config": {},
        },
    )
    return output


def orphan(output: Path, *, completion: bool = False, live: bool = False) -> Path:
    """Simulate a collector crash without launching a child or model provider."""
    directory = output / "attempts/call1/0001"
    (directory / "output").mkdir(parents=True)
    attempt = {
        "id": "call1",
        "number": 1,
        "attempt_id": "old-attempt",
        "directory": "attempts/call1/0001",
        "status": "running",
        "pid": os.getpid() if live else None,
        "host": socket.gethostname(),
        "process_identity": Experiment.process_identity(os.getpid()) if live else None,
        "started_epoch": time.time() - 2,
    }
    Experiment.atomic(directory / "attempt.json", attempt)
    (directory / "output/response.txt").write_text("partial raw response")
    if completion:
        case = Dataset(output / "corpus/dataset.json").cases[0]
        shutil.copyfile(case.reference, directory / "output/final.srt")
        Experiment.atomic(
            directory / "completion.json",
            {"returncode": 0, "finished_at": time.time() - 1},
        )
    return directory


def test_frozen_execution_ignores_later_source_changes(
    prepared: Path, approved: Dataset
) -> None:
    approved.cases[0].source.write_text("changed outside the frozen run")
    engine = FixtureExperiment()
    result = engine.collect(prepared)
    assert result["accuracy_including_failures"] == 1
    metadata = json.loads((prepared / "experiment.json").read_text())
    assert metadata["resolved_preset"]["model"] == "fixture-model"
    assert metadata["request_limit"] == 2
    assert engine.status(prepared)["collector_active"] is False


def test_changed_settings_reject_before_collection(prepared: Path) -> None:
    (prepared / "settings.json").write_text("{}")
    with pytest.raises(ValueError, match="Changed frozen settings"):
        FixtureExperiment().collect(prepared)
    assert not list((prepared / "attempts").glob("*"))


def test_competing_collectors_rejected(prepared: Path) -> None:
    with (prepared / "collector.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert FixtureExperiment().status(prepared)["collector_active"] is True
        with pytest.raises(ValueError, match="Another collector"):
            FixtureExperiment().collect(prepared)


def test_completed_orphan_recovered_without_new_attempt(prepared: Path) -> None:
    directory = orphan(prepared, completion=True)
    result = FixtureExperiment().collect(prepared, resume=True)
    assert result["accuracy_including_failures"] == 1
    assert len(FixtureExperiment.attempts(prepared, "call1")) == 1
    attempt = json.loads((directory / "attempt.json").read_text())
    assert attempt["attempt_id"] == "old-attempt"
    assert 0 < attempt["seconds"] < 2


def test_live_orphan_prevents_restart(prepared: Path) -> None:
    orphan(prepared, live=True)
    assert FixtureExperiment().status(prepared)["running_attempts"][0]["alive"] is True
    with pytest.raises(ValueError, match="still running"):
        FixtureExperiment().collect(prepared, resume=True)


def test_partial_attempt_preserved_and_retry_appends(prepared: Path) -> None:
    directory = orphan(prepared)
    engine = FixtureExperiment()
    failed = engine.collect(prepared, resume=True)
    assert failed["failed_calls"] == 1 and failed["accuracy_including_failures"] == 0
    assert (directory / "output/response.txt").read_text() == "partial raw response"
    result = engine.collect(prepared, retry=["call1"])
    assert result["successful_calls"] == 1
    assert (
        result["all_attempt_usage"]["metrics"]["input_tokens"]["missing_observations"]
        == 1
    )
    assert len(engine.attempts(prepared, "call1")) == 2
    assert engine.attempts(prepared, "call1")[0]["status"] == "interrupted"
    with pytest.raises(ValueError, match="Retry not permitted"):
        engine.collect(prepared, retry=["call1"])


def test_pause_and_resume_keep_pending_visible(prepared: Path) -> None:
    engine = FixtureExperiment()
    engine.atomic(prepared / "pause.json", {"requested_at": engine.now()})
    paused = engine.collect(prepared)
    assert paused["pending_calls"] == 1
    assert paused["accuracy_is_provisional"] is True
    assert paused["accuracy_including_failures"] == 0
    resumed = engine.collect(prepared, resume=True)
    assert resumed["pending_calls"] == 0 and resumed["accuracy_including_failures"] == 1


def test_rescore_changed_reference_retains_parent(
    prepared: Path, approved: Dataset, tmp_path: Path
) -> None:
    engine = FixtureExperiment()
    engine.collect(prepared)
    original = (prepared / "results.json").read_bytes()
    destination = tmp_path / "corrected-corpus"
    revised = engine.freeze_corpus(approved, destination)
    row = revised.data["cases"][0]
    row["decisions"][0]["classification"] = "Caller"
    row["revision"] += 1
    revised.data["revision"] += 1
    revised.data["parent"] = approved.data["content_sha256"]
    Corpus.derive(row, revised.cases[0].source, revised.cases[0].reference)
    revised.data["content_sha256"] = Corpus.identity(revised.data)
    engine.atomic(revised.path, revised.data)
    result = engine.rescore(Dataset(revised.path), prepared, tmp_path / "rescore")
    assert result["accuracy_including_failures"] == 0
    assert (prepared / "results.json").read_bytes() == original
    metadata = json.loads((tmp_path / "rescore/experiment.json").read_text())
    assert metadata["parent_evaluation"]["experiment_sha256"] == digest(
        prepared / "experiment.json"
    )
    with pytest.raises(ValueError, match="different cases"):
        Evaluation.compare(prepared, tmp_path / "rescore")


def test_changed_input_requires_recollection(
    prepared: Path, approved: Dataset, tmp_path: Path
) -> None:
    FixtureExperiment().collect(prepared)
    source = approved.cases[0].source
    source.write_text(source.read_text().replace("Hello.", "Goodbye."))
    row = approved.data["cases"][0]
    row["hashes"]["source"] = digest(source)
    Corpus.derive(row, source, approved.cases[0].reference)
    approved.data["content_sha256"] = Corpus.identity(approved.data)
    Experiment.atomic(approved.path, approved.data)
    with pytest.raises(ValueError, match="recollection"):
        FixtureExperiment().rescore(
            Dataset(approved.path), prepared, tmp_path / "invalid-rescore"
        )
    assert not (tmp_path / "invalid-rescore").exists()


def test_secrets_never_frozen(approved: Dataset, tmp_path: Path) -> None:
    template = tmp_path / "template.md"
    template.write_text("{{ transcription }}")
    with pytest.raises(ValueError, match="credentials"):
        FixtureExperiment().prepare(
            approved,
            tmp_path / "unsafe",
            "preset",
            template,
            settings={
                "metadata": {},
                "customizations": {"api_key": "private"},
                "config": {},
            },
        )
    assert not (tmp_path / "unsafe").exists()


@pytest.mark.parametrize("timeout", [False, True])
def test_real_subprocess_handshake_and_timeout(
    approved: Dataset,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    timeout: bool,
) -> None:
    original_popen = subprocess.Popen

    def fake_worker(command: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        assert command[1:3] == ["-m", "srt_labeler.local"]
        assert "--reference" not in command
        return original_popen(
            [sys.executable, "-m", "tests.fake_local_worker", *command[3:]], **kwargs
        )

    monkeypatch.setattr(subprocess, "Popen", fake_worker)
    prompt = tmp_path / "prompt.md"
    prompt.write_text("{{ transcription }}")
    output = tmp_path / "subprocess-run"
    engine = Experiment()
    engine.prepare(
        approved,
        output,
        "timeout-fixture" if timeout else "success-fixture",
        prompt,
        timeout=1 if timeout else 10,
        settings={
            "metadata": {"provider": "fixture"},
            "customizations": {},
            "config": {},
        },
    )
    result = engine.collect(output)
    assert result["failed_calls"] == int(timeout)
    attempt = engine.attempts(output, "call1")[0]
    assert attempt["status"] == ("timeout" if timeout else "ok")
    assert engine.process_identity(attempt["pid"]) is None
    assert (
        output / attempt["directory"] / "output/response.txt"
    ).read_text() == "fixture partial response"


def test_changed_limits_rejected(prepared: Path) -> None:
    path = prepared / "experiment.json"
    metadata = json.loads(path.read_text())
    metadata["retry_limit"] += 1
    Experiment.atomic(path, metadata)
    with pytest.raises(ValueError, match="configuration"):
        FixtureExperiment().collect(prepared)


def test_format_failure_is_distinct_from_collection_failure(prepared: Path) -> None:
    directory = orphan(prepared)
    Experiment.atomic(
        directory / "completion.json",
        {
            "returncode": 1,
            "finished_at": time.time(),
            "failure_phase": "format",
            "error_type": "ModelResponseFormattingError",
            "error": "Missing transcript tags",
        },
    )
    FixtureExperiment().collect(prepared, resume=True)
    result = json.loads((prepared / "results.json").read_text())[0]
    assert result["status"] == "failed"
    assert result["collection_status"] == "ok"
    assert result["structure_status"] == "invalid"
    assert result["grading_status"] == "invalid"


def test_local_child_deadline_prevents_unowned_request(
    approved: Dataset, tmp_path: Path
) -> None:
    case = approved.cases[0]
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "srt_labeler.local",
            "--srt",
            str(case.source),
            "--audio",
            str(case.audio),
            "--output-dir",
            str(tmp_path / "not-started"),
            "--start-gate",
            str(tmp_path / "missing-gate.json"),
            "--deadline-seconds",
            "0.2",
        ],
        timeout=10,
        check=False,
        capture_output=True,
    )
    assert process.returncode == -signal.SIGALRM
    assert not (tmp_path / "not-started").exists()
