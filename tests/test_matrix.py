"""Verify bounded interleaving, repeat reporting and matrix pause recovery offline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from srt_labeler.eval_dataset import Dataset
from srt_labeler.evaluate import Experiment
from tests.framework_scripts import FrameworkScripts
from tests.test_experiment_recovery import FixtureExperiment


def test_interleaved_matrix_preserves_repetitions(
    approved: Dataset, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = FrameworkScripts.load("matrix")
    monkeypatch.setattr(module, "Experiment", FixtureExperiment)

    def settings(_preset: str) -> dict[str, Any]:
        return {
            "metadata": {"provider": "fixture"},
            "customizations": {"model": "fixture"},
            "config": {},
        }

    monkeypatch.setattr(Experiment, "settings", staticmethod(settings))
    (tmp_path / "prompt.md").write_text("{{ transcription }}")
    specification = tmp_path / "matrix.json"
    specification.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "repetitions": 2,
                "seed": 17,
                "configurations": [
                    {"id": name, "preset": name, "template": "prompt.md"}
                    for name in ("baseline", "candidate")
                ],
            }
        )
    )
    matrix = module.Matrix()
    output = tmp_path / "matrix"
    plan = matrix.prepare(approved, specification, output)
    assert plan["request_limit"] == 4
    assert len(plan["schedule"]) == 4
    report = matrix.report(output)
    assert all(
        group["completed_repetitions"] == 0 for group in report["configurations"]
    )
    FixtureExperiment.atomic(output / "pause.json", {"reason": "fixture pause"})
    paused = matrix.collect(output)
    assert all(
        group["completed_repetitions"] == 0 for group in paused["configurations"]
    )
    completed = matrix.collect(output, resume=True)
    assert len(completed["runs"]) == 4
    assert len(completed["paired_comparisons"]) == 2
    assert all(group["accuracies"] == [1, 1] for group in completed["configurations"])
    assert matrix.status(output)["matrix_collector_active"] is False
    matrix.collect(output, resume=True)
    assert len(list((output / "runs").glob("*/*/attempts/*/*/attempt.json"))) == 4
    plan["schedule"].append(plan["schedule"][0])
    (output / "plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="schedule"):
        matrix.collect(output)
