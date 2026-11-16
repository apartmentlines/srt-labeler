"""Verify corpus authority, evidence, creation, and immutable revisions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from srt_labeler.eval_dataset import Corpus, Dataset


def rewrite(dataset: Dataset) -> None:
    """Refresh the hash when testing deeper validation than content identity."""
    dataset.data["content_sha256"] = Corpus.identity(dataset.data)
    dataset.path.write_text(json.dumps(dataset.data))


def test_approval_and_derived_mask(approved: Dataset) -> None:
    approved.require_approved()
    assert approved.data["cases"][0]["excluded"] == {"2": "Two voices"}
    assert (
        approved.data["cases"][0]["decisions"][1]["classification"] == "Both speakers"
    )
    assert (approved.path.parent / "provenance/review-policy.json").is_file()


def test_draft_cannot_be_scored(draft: Dataset) -> None:
    with pytest.raises(ValueError, match="Unapproved"):
        draft.require_approved()
    assert len(draft.cases[0].excluded) == 2


@pytest.mark.parametrize(
    "mutation", ["actor", "locator", "coverage", "reference", "duplicate"]
)
def test_invalid_authority_rejected(approved: Dataset, mutation: str) -> None:
    row = approved.data["cases"][0]
    if mutation == "actor":
        row["decisions"][0]["actor"]["type"] = "model"
    elif mutation == "locator":
        row["decisions"][0]["evidence"]["start"] = 0.5
    elif mutation == "coverage":
        row["decisions"].append(row["decisions"][0])
    elif mutation == "reference":
        row["decisions"][0]["classification"] = "Caller"
    else:
        approved.data["cases"].append(row)
    rewrite(approved)
    with pytest.raises(ValueError):
        Dataset(approved.path)


def test_changed_provenance_rejected(approved: Dataset) -> None:
    (approved.path.parent / "provenance/review-policy.json").write_text("changed")
    with pytest.raises(ValueError, match="provenance"):
        Dataset(approved.path)


def test_creation_failure_does_not_publish(inputs: Path, tmp_path: Path) -> None:
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        Corpus().create(inputs, output, "")
    assert not output.exists()


def test_revision_paths_cannot_escape(approved: Dataset) -> None:
    row = approved.data["cases"][0]
    row["reference"] = str(approved.cases[0].reference.resolve())
    rewrite(approved)
    with pytest.raises(ValueError, match="contained"):
        Dataset(approved.path)


def test_original_draft_integrity(approved: Dataset) -> None:
    (approved.path.parent / "call1/original-draft.srt").write_text("changed")
    with pytest.raises(ValueError, match="original draft"):
        Dataset(approved.path)
