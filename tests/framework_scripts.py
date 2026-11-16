"""Load repository-owned operating tools for offline behavioral tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from srt_labeler.eval_dataset import Dataset, digest
from srt_labeler.evaluate import Experiment


class FrameworkScripts:
    """Import exactly the script invoked by the project operating skill."""

    @staticmethod
    def load(name: str, skill: str = "evaluate-srt-presets") -> ModuleType:
        path = (
            Path(__file__).resolve().parents[1]
            / ".agents/skills"
            / skill
            / "scripts"
            / f"{name}.py"
        )
        spec = importlib.util.spec_from_file_location(name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def saved_origin(dataset: Dataset, directory: Path) -> None:
        """Record current experiment identities for supplied offline outputs."""
        prompt = directory / "template.md"
        prompt.write_text("{{ transcription }}")
        run = directory / "prepared"
        Experiment().prepare(
            dataset,
            run,
            "fixture",
            prompt,
            settings={"metadata": {}, "customizations": {}, "config": {}},
        )
        metadata = json.loads((run / "experiment.json").read_text())
        results = []
        for case in dataset.cases:
            artifacts = directory / case.id
            results.append(
                {
                    "id": case.id,
                    "attempt_id": "fixture-" + case.id,
                    "artifact_directory": case.id,
                    "collection_status": "ok",
                    "artifacts": {f.name: digest(f) for f in artifacts.iterdir()},
                }
            )
        (directory / "experiment.json").write_text(json.dumps(metadata))
        (directory / "results.json").write_text(json.dumps(results))
