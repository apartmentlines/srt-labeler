"""Run reproducible preset experiments and score exact preserved SRT blocks."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
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
            "model": customizations.get("model_name"),
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
        """Run or rescore saved predictions; never overwrite an experiment."""
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        template_bytes = template.read_bytes()
        identity = self.preset_identity(preset) if predictions is None else None
        output.mkdir(parents=True, exist_ok=False)
        (output / "template.md").write_bytes(template_bytes)
        snapshot = []
        for case in dataset.cases:
            snapshot.append(
                {
                    "id": case.id,
                    "split": case.split,
                    "hashes": {
                        k: digest(getattr(case, k))
                        for k in ("source", "reference", "audio")
                    },
                    "excluded": case.excluded,
                }
            )
        metadata = {
            "schema_version": 1,
            "preset": preset,
            "resolved_preset": identity,
            "template_sha256": digest(template),
            "dataset_sha256": digest(dataset.path),
            "cases": snapshot,
            "mode": "rescore" if predictions else "run",
            "python": sys.version,
            "implementation_sha256": {
                name: digest(Path(__file__).parent / name)
                for name in (
                    "evaluate.py",
                    "local.py",
                    "labeling.py",
                    "merger.py",
                    "token_usage.py",
                )
            },
            "timeout_seconds": timeout,
        }
        (output / "experiment.json").write_text(json.dumps(metadata, indent=2))
        results = []
        for case in dataset.cases:
            started = time.monotonic()
            destination = output / case.id
            result: dict[str, Any] = {
                "id": case.id,
                "split": case.split,
                "eligible": len(subtitles(case.source)) - len(case.excluded),
                "correct": 0,
                "excluded": len(case.excluded),
                "status": "failed",
            }
            try:
                if predictions:
                    destination.mkdir()
                    saved_usage = predictions / case.id / "usage.json"
                    if saved_usage.exists():
                        shutil.copyfile(saved_usage, destination / "usage.json")
                    shutil.copyfile(
                        predictions / case.id / "final.srt", destination / "final.srt"
                    )
                else:
                    command = [
                        sys.executable,
                        "-m",
                        "srt_labeler.local",
                        "--srt",
                        str(case.source.resolve()),
                        "--audio",
                        str(case.audio.resolve()),
                        "--output-dir",
                        str(destination.resolve()),
                        "--preset",
                        preset,
                        "--template-file",
                        str((output / "template.md").resolve()),
                    ]
                    if debug:
                        command.append("--debug")
                    with (output / f"{case.id}.log").open("w") as log:
                        subprocess.run(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            timeout=timeout,
                            check=True,
                        )
                result.update(self.score(case, destination / "final.srt"))
                result["status"] = "ok"
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                result["error"] = str(error)
                logging.error("%s: %s", case.id, error)
            result["seconds"] = round(time.monotonic() - started, 3)
            result["usage"] = self.read_usage(destination / "usage.json")
            result["usage_origin"] = (
                "saved-prediction" if predictions else "current-attempt"
            )
            results.append(result)
            (output / "results.json").write_text(json.dumps(results, indent=2))
            logging.info("%s: %s", case.id, result["status"])
        eligible = sum(r["eligible"] for r in results)
        correct = sum(r["correct"] for r in results)
        successful = [r for r in results if r["status"] == "ok"]
        covered = sum(r["eligible"] for r in successful)
        summary = {
            "calls": len(results),
            "successful_calls": len(successful),
            "failed_calls": len(results) - len(successful),
            "eligible_blocks": eligible,
            "covered_blocks": covered,
            "excluded_blocks": sum(r["excluded"] for r in results),
            "correct_blocks": correct,
            "accuracy_including_failures": correct / eligible if eligible else None,
            "accuracy_on_successful_calls": correct / covered if covered else None,
            "macro_call_accuracy": (
                sum(r["correct"] / r["eligible"] for r in results if r["eligible"])
                / sum(bool(r["eligible"]) for r in results)
                if eligible
                else None
            ),
            "seconds": sum(r["seconds"] for r in results),
            "usage": TokenUsage.aggregate(
                [
                    observation
                    for r in results
                    for observation in r["usage"]["observations"]
                ]
            ),
            "usage_origin": (
                "saved-predictions; no new model requests"
                if predictions
                else "current-attempts"
            ),
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary

    @staticmethod
    def compare(first: Path, second: Path) -> dict[str, Any]:
        """Compare runs only when inputs, references and scoring masks match."""
        a, b = [
            json.loads((p / "experiment.json").read_text()) for p in (first, second)
        ]
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
        changes = [
            {
                "id": x["id"],
                "correct_delta": y["correct"] - x["correct"],
                "before_status": x["status"],
                "after_status": y["status"],
            }
            for x, y in zip(left, right)
        ]
        return {
            "before": str(first),
            "after": str(second),
            "correct_delta": sum(c["correct_delta"] for c in changes),
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
