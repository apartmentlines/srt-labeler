"""Demonstrate draft review through collection, correction, rescore and comparison."""

from __future__ import annotations

import json
from pathlib import Path

from srt_labeler.eval_dataset import Dataset
from srt_labeler.evaluate import Evaluation
from srt_labeler.review_site import ReviewSiteBuilder
from srt_labeler.review_workbook import ReviewWorkbook
from tests.framework_scripts import FrameworkScripts
from tests.test_experiment_recovery import FixtureExperiment
from tests.test_review_feedback import resolutions


def test_complete_offline_workflow(draft: Dataset, tmp_path: Path) -> None:
    Feedback = FrameworkScripts.load("review_feedback").Feedback
    ReviewSiteBuilder(draft, tmp_path / "review").build()
    manifest = json.loads((tmp_path / "review/review.json").read_text())
    feedback = ReviewWorkbook().read(tmp_path / "review/review-notes.xlsx")
    feedback["reviewer"] = "Synthetic acceptance reviewer"
    for entry, decision in zip(feedback["entries"], ("Correct", "Both speakers")):
        entry.update(decision=decision, state="completed")
    approved = Dataset(
        Feedback(draft, manifest).reconcile(
            feedback, resolutions(feedback), tmp_path / "approved-corpus"
        )
    )
    approved.require_approved()
    engine = FixtureExperiment()
    template = tmp_path / "prompt.md"
    template.write_text("{{ transcription }} {{ identifier }}")
    settings = {
        "metadata": {"provider": "fixture"},
        "customizations": {"model": "fixture"},
        "config": {},
    }
    engine.prepare(
        approved, tmp_path / "baseline", "fixture", template, settings=settings
    )
    assert engine.collect(tmp_path / "baseline")["accuracy_including_failures"] == 1
    original = (tmp_path / "baseline/results.json").read_bytes()
    ReviewSiteBuilder(approved, tmp_path / "correction-review").build()
    correction = ReviewWorkbook().read(tmp_path / "correction-review/review-notes.xlsx")
    correction["reviewer"] = "Synthetic acceptance reviewer"
    correction["entries"][0].update(
        decision="Caller", state="completed", notes="Corrected listening criterion"
    )
    manifest = json.loads((tmp_path / "correction-review/review.json").read_text())
    revised = Dataset(
        Feedback(approved, manifest).reconcile(
            correction, resolutions(correction), tmp_path / "revised-corpus"
        )
    )
    assert (
        engine.rescore(revised, tmp_path / "baseline", tmp_path / "baseline-rescored")[
            "accuracy_including_failures"
        ]
        == 0
    )
    assert (tmp_path / "baseline/results.json").read_bytes() == original
    engine.prepare(
        revised, tmp_path / "candidate", "fixture", template, settings=settings
    )
    assert engine.collect(tmp_path / "candidate")["accuracy_including_failures"] == 1
    comparison = Evaluation.compare(
        tmp_path / "baseline-rescored", tmp_path / "candidate"
    )
    assert comparison["correct_delta"] == 1
    assert comparison["calls"][0]["corrected_blocks"] == [1]
