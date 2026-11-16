"""Exercise portable feedback, explicit dispositions, and revision integrity."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from srt_labeler.eval_dataset import Dataset
from srt_labeler.review_site import ReviewSiteBuilder
from srt_labeler.review_workbook import ReviewWorkbook
from tests.framework_scripts import FrameworkScripts

Feedback = FrameworkScripts.load("review_feedback").Feedback


@pytest.fixture
def review(approved: Dataset, tmp_path: Path) -> tuple[Any, dict[str, Any]]:
    builder = ReviewSiteBuilder(approved, tmp_path / "review")
    builder.build()
    manifest = json.loads((tmp_path / "review/review.json").read_text())
    feedback = ReviewWorkbook().read(tmp_path / "review/review-notes.xlsx")
    feedback["reviewer"] = "Local fixture reviewer"
    return Feedback(approved, manifest), feedback


def resolutions(feedback: dict[str, Any], action: str = "accept") -> dict[str, Any]:
    return {
        "base": feedback["base"],
        "feedback_sha256": hashlib.sha256(
            json.dumps(feedback, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "decisions": [
            {
                "identifier": entry["identifier"],
                "action": action,
                "actor": {"type": "assistant", "identity": "fixture adjudicator"},
                "rationale": "Confirmed listening proposal",
            }
            for entry in feedback["entries"]
            if entry["state"] != "untouched"
        ],
    }


def test_feedback_roundtrip_and_revision(
    review: tuple[Any, dict[str, Any]], tmp_path: Path
) -> None:
    engine, feedback = review
    original = engine.dataset.path.read_bytes()
    feedback["entries"][0].update(
        decision="Caller", notes="=literal <script> & text", state="completed"
    )
    workbook = tmp_path / "completed.xlsx"
    ReviewWorkbook().write(workbook, engine.review["blocks"], feedback)
    restored = ReviewWorkbook().read(workbook)
    assert restored == feedback
    revised = Dataset(
        engine.reconcile(restored, resolutions(restored), tmp_path / "revision-2")
    )
    revised.require_approved()
    assert engine.dataset.path.read_bytes() == original
    assert revised.data["revision"] == 2
    assert revised.data["parent"] == engine.dataset.data["content_sha256"]
    row = revised.data["cases"][0]
    assert row["id"] == "call1" and row["revision"] == 2
    assert row["decisions"][0]["classification"] == "Caller"
    assert row["decisions"][0]["actor"]["type"] == "human"
    assert row["decisions"][0]["provenance"]["accepted_by"]["type"] == "assistant"
    assert "Caller: Hello." in revised.cases[0].reference.read_text()
    with pytest.raises(ValueError, match="Stale"):
        Feedback(revised, engine.review)


@pytest.mark.parametrize(
    "mutation",
    ["base", "duplicate", "unknown", "revision", "blank-completed", "edited-untouched"],
)
def test_invalid_feedback_has_no_side_effects(
    review: tuple[Any, dict[str, Any]], tmp_path: Path, mutation: str
) -> None:
    engine, feedback = review
    feedback = copy.deepcopy(feedback)
    if mutation == "base":
        feedback["base"]["corpus_revision"] += 1
    elif mutation == "duplicate":
        feedback["entries"].append(feedback["entries"][0])
    elif mutation == "unknown":
        feedback["entries"][0]["identifier"] = "missing"
    elif mutation == "revision":
        feedback["entries"][0]["case_revision"] += 1
    elif mutation == "blank-completed":
        feedback["entries"][0]["state"] = "completed"
    else:
        feedback["entries"][0]["notes"] = "comment alone is a draft"
    with pytest.raises(ValueError):
        engine.reconcile(feedback, resolutions(feedback), tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("action", ["reject", "defer"])
def test_nonaccepted_proposals_preserve_expectations(
    review: tuple[Any, dict[str, Any]], tmp_path: Path, action: str
) -> None:
    engine, feedback = review
    feedback["entries"][0].update(notes="Maybe caller?", state="draft")
    revised = Dataset(
        engine.reconcile(feedback, resolutions(feedback, action), tmp_path / action)
    )
    assert revised.data["cases"] == engine.dataset.data["cases"]
    assert any(
        "resolutions-2.json" in artifact["path"]
        for artifact in revised.data["artifacts"]
    )


def test_unsure_is_not_approved(
    review: tuple[Any, dict[str, Any]], tmp_path: Path
) -> None:
    engine, feedback = review
    feedback["entries"][0].update(decision="Unsure", state="completed")
    revised = Dataset(
        engine.reconcile(feedback, resolutions(feedback), tmp_path / "uncertain")
    )
    with pytest.raises(ValueError, match="Unapproved"):
        revised.require_approved()


def test_changed_or_missing_resolution_rejected(
    review: tuple[Any, dict[str, Any]], tmp_path: Path
) -> None:
    engine, feedback = review
    feedback["entries"][0].update(decision="Correct", state="completed")
    decisions = resolutions(feedback)
    decisions["decisions"] = []
    with pytest.raises(ValueError, match="disposition"):
        engine.reconcile(feedback, decisions, tmp_path / "missing")
    decisions = resolutions(feedback)
    feedback["entries"][0]["notes"] = "Changed after adjudication"
    with pytest.raises(ValueError, match="different feedback"):
        engine.reconcile(feedback, decisions, tmp_path / "changed")
