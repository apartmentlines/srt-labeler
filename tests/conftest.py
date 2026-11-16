"""Shared local fixtures for the application and operating framework."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import wave

import pytest

from srt_labeler.eval_dataset import Corpus, Dataset


@pytest.fixture
def inputs(tmp_path: Path) -> Path:
    """Provide two source blocks with draft labels and matching audio."""
    (tmp_path / "source.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nHello.\n\n"
        + "2\n00:00:01,000 --> 00:00:02,000\nYes.\n"
    )
    (tmp_path / "reference.srt").write_text(
        "1\n00:00:00,000 --> 00:00:01,000\nOperator: Hello.\n\n"
        + "2\n00:00:01,000 --> 00:00:02,000\nCaller: Yes.\n"
    )
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setparams((1, 2, 8000, 0, "NONE", "not compressed"))
        audio.writeframes(bytes(32000))
    path = tmp_path / "inputs.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "call1",
                        "source": "source.srt",
                        "draft": "reference.srt",
                        "audio": "audio.wav",
                        "split": "development",
                    }
                ],
            }
        )
    )
    return path


@pytest.fixture
def draft(inputs: Path, tmp_path: Path) -> Dataset:
    """Create a corpus directly from new input files."""
    return Dataset(Corpus().create(inputs, tmp_path / "draft", "test-corpus"))


@pytest.fixture
def approved(draft: Dataset, tmp_path: Path) -> Dataset:
    """Provide explicit human judgments and retained supporting evidence."""
    staging = tmp_path / "approved-staging"
    shutil.copytree(draft.path.parent, staging)
    data = draft.data
    policy = tmp_path / "policy.json"
    policy.write_text(
        json.dumps({"review_scope": "Full listen", "blank_semantics": "unreviewed"})
    )
    data["artifacts"] = [Corpus.copy_artifact(policy, staging, "review-policy.json")]
    row = data["cases"][0]
    for decision, classification, reason in zip(
        row["decisions"], ("Operator", "Both speakers"), ("Full listen", "Two voices")
    ):
        decision.update(
            classification=classification,
            state="approved",
            actor={"type": "human", "identity": "fixture reviewer"},
            rationale=reason,
            provenance={"artifact_sha256": [data["artifacts"][0]["sha256"]]},
        )
    Corpus.derive(row, staging / row["source"], staging / row["reference"])
    return Dataset(Corpus.publish(data, staging, tmp_path / "approved"))


@pytest.fixture
def sample_srt():
    return "1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n"
