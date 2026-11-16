"""Exercise scoring masks, failed runs and portable review artifacts offline."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Any
from xml.etree import ElementTree as ET
import zipfile

import pytest

from srt_labeler.eval_dataset import Corpus, Dataset, digest, subtitles
from srt_labeler.evaluate import Evaluation, Experiment
from srt_labeler.review_site import ReviewSiteBuilder
from srt_labeler.token_usage import TokenUsage
from tests.framework_scripts import FrameworkScripts


@pytest.fixture
def dataset(approved: Dataset) -> Dataset:
    """Include literal markup and a highlighted excluded block."""
    row = approved.data["cases"][0]
    for key in ("source", "reference"):
        path = approved.path.parent / row[key]
        path.write_text(path.read_text().replace("Hello.", "Hello <there>."))
        row["hashes"][key] = digest(path)
    row["flagged"] = [2]
    approved.data["content_sha256"] = Corpus.identity(approved.data)
    approved.path.write_text(json.dumps(approved.data))
    return Dataset(approved.path)


def test_score_excludes_label_but_requires_content(
    dataset: Dataset, tmp_path: Path
) -> None:
    case = dataset.cases[0]
    prediction = tmp_path / "prediction.srt"
    prediction.write_text(case.reference.read_text().replace("Caller:", "Operator:"))
    assert Evaluation.score(case, prediction)["correct"] == 1
    prediction.write_text(prediction.read_text().replace("Yes.", "No."))
    with pytest.raises(ValueError, match="Changed source"):
        Evaluation.score(case, prediction)


def test_run_cli_and_timeout_keep_denominator(
    approved: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    class TimedOutProcess:
        pid: int = 99999999

        def __init__(self, command: list[str], **kwargs: Any) -> None:
            commands.append(command)

        def wait(self, timeout: float | None = None) -> int:
            if timeout is not None:
                raise subprocess.TimeoutExpired(commands[-1], timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", TimedOutProcess)

    def kill_fixture(_pid: int, _signal: int) -> None:
        """Avoid sending a real signal for the synthetic timeout process."""

    def settings_fixture(_preset: str) -> dict[str, Any]:
        return {"metadata": {"provider": "fixture"}, "customizations": {}, "config": {}}

    monkeypatch.setattr(os, "killpg", kill_fixture)
    monkeypatch.setattr(Experiment, "settings", staticmethod(settings_fixture))
    template = tmp_path / "prompt.md"
    template.write_text("{{ transcription }}")
    summary = Evaluation().run(approved, tmp_path / "run", "preset", template, 1, True)
    assert summary["failed_calls"] == 1
    assert summary["eligible_blocks"] == 1
    assert summary["accuracy_including_failures"] == 0
    assert commands[0][1:3] == ["-m", "srt_labeler.local"]


def test_rescore_and_compare(approved: Dataset, tmp_path: Path) -> None:
    dataset = approved
    predictions = tmp_path / "predictions" / "call1"
    predictions.mkdir(parents=True)
    (predictions / "final.srt").write_bytes(dataset.cases[0].reference.read_bytes())
    usage = TokenUsage()
    usage.record({"raw": {}, "tokens": {"input_tokens": 100, "output_tokens": 20}})
    (predictions / "usage.json").write_text(json.dumps(usage.report()))
    FrameworkScripts.saved_origin(dataset, predictions.parent)
    template = tmp_path / "prompt.md"
    template.write_text("prompt")
    for name in ["a", "b"]:
        result = Evaluation().run(
            dataset, tmp_path / name, "preset", template, 1, False, predictions.parent
        )
        assert result["accuracy_including_failures"] == 1
        assert result["usage"]["metrics"]["input_tokens"]["total"] == 100
        assert "no new model requests" in result["usage_origin"]
    assert Evaluation.compare(tmp_path / "a", tmp_path / "b")["correct_delta"] == 0
    with pytest.raises(FileExistsError):
        Evaluation().run(
            dataset, tmp_path / "a", "preset", template, 1, False, predictions.parent
        )


def test_review_site_workbook_and_archive(dataset: Dataset, tmp_path: Path) -> None:
    output = tmp_path / "site"
    ReviewSiteBuilder(dataset, output).build()
    html = (output / "calls/call1.html").read_text()
    assert "Hello &lt;there&gt;." in html
    assert 'class="block review" id="B002"' in html
    assert (output / "audio/call1.wav").read_bytes() == dataset.cases[
        0
    ].audio.read_bytes()
    with zipfile.ZipFile(output / "review-notes.xlsx") as workbook:
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))
        ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        validation = sheet.find("x:dataValidations/x:dataValidation", ns)
        assert validation is not None and validation.attrib["sqref"] == "B2:B3"
        cell = sheet.find(".//x:c[@r='A3']", ns)
        assert cell is not None and cell.attrib["s"] == "2"
    with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
        for file in output.rglob("*"):
            if file.is_file():
                assert (
                    archive.read(str(file.relative_to(output.parent)))
                    == file.read_bytes()
                )


def test_no_flags_still_renders_call(dataset: Dataset, tmp_path: Path) -> None:
    dataset.cases[0].flagged = []
    ReviewSiteBuilder(dataset, tmp_path / "site").build()
    assert "call1-B001" in (tmp_path / "site/calls/call1.html").read_text()


def test_dataset_rejects_mismatch_and_unsafe_id(dataset: Dataset) -> None:
    data = dataset.data
    data["cases"][0]["id"] = "../escape"
    dataset.path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Unsafe"):
        Dataset(dataset.path)


def test_draft_only_input(inputs: Path, tmp_path: Path) -> None:
    data = json.loads(inputs.read_text())
    del data["cases"][0]["source"]
    inputs.write_text(json.dumps(data))
    prepared = Dataset(Corpus().create(inputs, tmp_path / "derived", "test"))
    assert subtitles(prepared.cases[0].source) == subtitles(tmp_path / "source.srt")


def test_malformed_prediction_is_failed_case(approved: Dataset, tmp_path: Path) -> None:
    dataset = approved
    prediction = tmp_path / "predictions/call1"
    prediction.mkdir(parents=True)
    (prediction / "final.srt").write_text("not an SRT")
    usage = TokenUsage()
    usage.record({"raw": {}, "tokens": {"input_tokens": 50, "output_tokens": 5}})
    (prediction / "usage.json").write_text(json.dumps(usage.report()))
    FrameworkScripts.saved_origin(dataset, prediction.parent)
    template = tmp_path / "prompt.md"
    template.write_text("prompt")
    result = Evaluation().run(
        dataset, tmp_path / "run", "preset", template, 1, False, prediction.parent
    )
    assert result["failed_calls"] == 1
    assert result["accuracy_including_failures"] == 0
    assert result["usage"]["metrics"]["input_tokens"]["total"] == 50
