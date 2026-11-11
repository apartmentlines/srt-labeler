"""Exercise scoring masks, failed runs and portable review artifacts offline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any
import wave
from xml.etree import ElementTree as ET
import zipfile

import pytest

from srt_labeler.eval_dataset import Dataset
from srt_labeler.evaluate import Evaluation
from srt_labeler.review_site import ReviewSiteBuilder
from srt_labeler.token_usage import TokenUsage


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    """Create two source blocks, one excluded from binary scoring."""
    (tmp_path / "source.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello <there>.\n\n2\n00:00:01,000 --> 00:00:02,000\nYes.\n"
    )
    (tmp_path / "reference.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nOperator: Hello <there>.\n\n2\n00:00:01,000 --> 00:00:02,000\nCaller: Yes.\n"
    )
    with wave.open(str(tmp_path / "audio.wav"), "w") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))
    path = tmp_path / "dataset.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "cases": [
                    {
                        "id": "call1",
                        "source": "source.srt",
                        "reference": "reference.srt",
                        "audio": "audio.wav",
                        "split": "development",
                        "excluded": {"2": "mixed"},
                        "flagged": [2],
                    }
                ],
            }
        )
    )
    return Dataset(path)


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
    dataset: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commands: list[list[str]] = []

    def fail(command: list[str], **kwargs: Any) -> None:
        commands.append(command)
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fail)

    def identity(_preset: str) -> dict[str, Any]:
        return {}

    monkeypatch.setattr(Evaluation, "preset_identity", staticmethod(identity))
    template = tmp_path / "prompt.md"
    template.write_text("{{ transcription }}")
    summary = Evaluation().run(dataset, tmp_path / "run", "preset", template, 1, True)
    assert summary["failed_calls"] == 1
    assert summary["eligible_blocks"] == 1
    assert summary["accuracy_including_failures"] == 0
    assert commands[0][1:3] == ["-m", "srt_labeler.local"]


def test_rescore_and_compare(dataset: Dataset, tmp_path: Path) -> None:
    predictions = tmp_path / "predictions" / "call1"
    predictions.mkdir(parents=True)
    (predictions / "final.srt").write_bytes(dataset.cases[0].reference.read_bytes())
    usage = TokenUsage()
    usage.record({"raw": {}, "tokens": {"input_tokens": 100, "output_tokens": 20}})
    (predictions / "usage.json").write_text(json.dumps(usage.report()))
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
    assert "../review-notes.xlsx" in html
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


def test_draft_only_input(dataset: Dataset, tmp_path: Path) -> None:
    data = dataset.data
    del data["cases"][0]["source"]
    dataset.path.write_text(json.dumps(data))
    temporary = tmp_path / "prepared"
    temporary.mkdir()
    prepared = ReviewSiteBuilder.prepare(dataset.path, temporary)
    assert (
        prepared.cases[0].source.read_text()
        == dataset.cases[0].source.read_text() + "\n"
    )


def test_malformed_prediction_is_failed_case(dataset: Dataset, tmp_path: Path) -> None:
    prediction = tmp_path / "predictions/call1"
    prediction.mkdir(parents=True)
    (prediction / "final.srt").write_text("not an SRT")
    usage = TokenUsage()
    usage.record({"raw": {}, "tokens": {"input_tokens": 50, "output_tokens": 5}})
    (prediction / "usage.json").write_text(json.dumps(usage.report()))
    template = tmp_path / "prompt.md"
    template.write_text("prompt")
    result = Evaluation().run(
        dataset, tmp_path / "run", "preset", template, 1, False, prediction.parent
    )
    assert result["failed_calls"] == 1
    assert result["accuracy_including_failures"] == 0
    assert result["usage"]["metrics"]["input_tokens"]["total"] == 50
