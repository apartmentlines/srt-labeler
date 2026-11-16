"""Run a frozen, interleaved comparison with explicit repetitions and local reports."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import statistics
from typing import Any

from srt_labeler.eval_dataset import Dataset, digest
from srt_labeler.evaluate import Evaluation, Experiment


class Matrix:
    """Schedule one bounded attempt at a time, retaining independent run recovery."""

    def prepare(
        self, dataset: Dataset, specification: Path, output: Path
    ) -> dict[str, Any]:
        """Freeze every configuration and an interleaved schedule without requests."""
        spec = json.loads(specification.read_text())
        if spec.get("schema_version") != 1:
            raise ValueError("Unsupported matrix specification schema_version")
        repetitions = spec["repetitions"]
        if type(repetitions) is not int or repetitions < 1:
            raise ValueError("Repetitions must be a positive integer")
        configurations = spec["configurations"]
        names = [row["id"] for row in configurations]
        if (
            not names
            or len(set(names)) != len(names)
            or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names)
        ):
            raise ValueError("Configuration IDs must be unique safe names")
        dataset.require_approved()
        output.mkdir(parents=True, exist_ok=False)
        runs = []
        engine = Experiment()
        for repetition in range(1, repetitions + 1):
            for config in configurations:
                relative = f"runs/{config['id']}/{repetition:03d}"
                metadata = engine.prepare(
                    dataset,
                    output / relative,
                    config["preset"],
                    specification.parent / config["template"],
                    timeout=spec.get("timeout_seconds", 600),
                    retry_limit=spec.get("retry_limit", 0),
                    selection={
                        "ordered_ids": [case.id for case in dataset.cases],
                        "matrix_repetition": repetition,
                    },
                )
                runs.append(
                    {
                        "configuration": config["id"],
                        "repetition": repetition,
                        "directory": relative,
                        "run_id": metadata["run_id"],
                        "experiment_sha256": digest(
                            output / relative / "experiment.json"
                        ),
                    }
                )
        schedule = []
        randomizer = random.Random(spec.get("seed", 0))
        for repetition in range(1, repetitions + 1):
            for case in dataset.cases:
                variants = [run for run in runs if run["repetition"] == repetition]
                randomizer.shuffle(variants)
                schedule.extend(
                    {"run": run["directory"], "case_id": case.id} for run in variants
                )
        plan = {
            "schema_version": 1,
            "specification": spec,
            "runs": runs,
            "schedule": schedule,
            "request_limit": len(schedule) * (spec.get("retry_limit", 0) + 1),
            "concurrency": 1,
            "created_at": engine.now(),
            "driver_sha256": digest(Path(__file__)),
        }
        plan["content_sha256"] = hashlib.sha256(
            json.dumps(plan, sort_keys=True).encode()
        ).hexdigest()
        engine.atomic(output / "plan.json", plan)
        return plan

    def collect(
        self, output: Path, resume: bool = False, debug: bool = False
    ) -> dict[str, Any]:
        """Resume exactly the frozen schedule, without repeating completed attempts."""
        plan = self.read_plan(output)
        if plan["driver_sha256"] != digest(Path(__file__)):
            raise ValueError("Matrix execution driver changed; prepare a new matrix")
        engine = Experiment()
        with (output / "matrix.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(
                    "Another matrix collector owns this schedule"
                ) from error
            if resume:
                (output / "pause.json").unlink(missing_ok=True)
            for run in plan["runs"]:
                if (
                    digest(output / run["directory"] / "experiment.json")
                    != run["experiment_sha256"]
                ):
                    raise ValueError("Changed matrix run identity")
            for job in plan["schedule"]:
                if (output / "pause.json").exists():
                    break
                engine.collect(
                    output / job["run"],
                    debug=debug,
                    resume=resume,
                    only_ids=[job["case_id"]],
                )
        return self.report(output)

    def report(self, output: Path) -> dict[str, Any]:
        """Report every repetition, paired regressions and incomplete collections."""
        plan = self.read_plan(output)
        rows = []
        for run in plan["runs"]:
            summary = json.loads(
                (output / run["directory"] / "summary.json").read_text()
            )
            rows.append({**run, "summary": summary})
        names = [config["id"] for config in plan["specification"]["configurations"]]
        groups = []
        comparisons = []
        baseline = names[0]
        for name in names:
            observations = [row for row in rows if row["configuration"] == name]
            complete = [
                row for row in observations if row["summary"]["pending_calls"] == 0
            ]
            accuracies = [
                row["summary"]["accuracy_including_failures"]
                for row in complete
                if row["summary"]["accuracy_including_failures"] is not None
            ]
            groups.append(
                {
                    "configuration": name,
                    "planned_repetitions": len(observations),
                    "completed_repetitions": len(complete),
                    "accuracies": accuracies,
                    "mean_accuracy": (
                        statistics.mean(accuracies) if accuracies else None
                    ),
                    "min_accuracy": min(accuracies) if accuracies else None,
                    "max_accuracy": max(accuracies) if accuracies else None,
                }
            )
            if name == baseline:
                continue
            for row in complete:
                before = next(
                    item
                    for item in rows
                    if item["configuration"] == baseline
                    and item["repetition"] == row["repetition"]
                )
                if before["summary"]["pending_calls"] == 0:
                    comparisons.append(
                        {
                            "configuration": name,
                            "repetition": row["repetition"],
                            "comparison": Evaluation.compare(
                                output / before["directory"], output / row["directory"]
                            ),
                        }
                    )
        result: dict[str, Any] = {
            "configurations": groups,
            "runs": rows,
            "paired_comparisons": comparisons,
            "interpretation": "Repeated observations of the same challenge/control calls, not independent population samples. Range is run-to-run variation, not a confidence interval. Provider-reported token counts are not dollar costs.",
        }
        Experiment.atomic(output / "report.json", result)
        lines: list[str] = [
            "# SRT comparison",
            "",
            result["interpretation"],
            "",
            "| Configuration | Completed repetitions | Mean accuracy | Range |",
            "| --- | ---: | ---: | --- |",
        ]
        for group in groups:
            lines.append(
                f"| {group['configuration']} | {group['completed_repetitions']}/{group['planned_repetitions']} | {group['mean_accuracy']} | {group['min_accuracy']}–{group['max_accuracy']} |"
            )
        lines.extend(
            [
                "",
                "Per-call regressions, failures, usage coverage and every repetition are retained in report.json and each run's results.json.",
                "",
            ]
        )
        (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
        return result

    @staticmethod
    def read_plan(output: Path) -> dict[str, Any]:
        """Check schedule content identity before interpreting or executing it."""
        plan = json.loads((output / "plan.json").read_text())
        expected = plan.pop("content_sha256")
        if (
            hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
            != expected
        ):
            raise ValueError("Changed matrix schedule")
        plan["content_sha256"] = expected
        return plan

    def status(self, output: Path) -> dict[str, Any]:
        """Check scheduler ownership as well as each underlying collection."""
        plan = self.read_plan(output)
        with (output / "matrix.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                active = False
            except BlockingIOError:
                active = True
        return {
            "matrix_collector_active": active,
            "pause_requested": (output / "pause.json").exists(),
            "runs": [
                Experiment().status(output / run["directory"]) for run in plan["runs"]
            ],
        }


class MatrixCommand:
    """Separate read-only preparation/reporting from explicitly requested collection."""

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "operation",
            choices=("prepare", "collect", "resume", "pause", "status", "report"),
        )
        parser.add_argument("--output", type=Path, required=True)
        parser.add_argument("--dataset", type=Path)
        parser.add_argument("--specification", type=Path)
        parser.add_argument("--split")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
        engine = Matrix()
        try:
            if args.operation == "prepare":
                if args.dataset is None or args.specification is None:
                    parser.error("prepare requires --dataset and --specification")
                result = engine.prepare(
                    Dataset(args.dataset, args.split), args.specification, args.output
                )
            elif args.operation in {"collect", "resume"}:
                result = engine.collect(
                    args.output, args.operation == "resume", args.debug
                )
            elif args.operation == "report":
                result = engine.report(args.output)
            else:
                if args.operation == "pause":
                    engine.read_plan(args.output)
                    Experiment.atomic(
                        args.output / "pause.json", {"requested_at": Experiment.now()}
                    )
                result = engine.status(args.output)
            print(json.dumps(result, indent=2))
            return int(
                args.operation in {"collect", "resume"}
                and any(run["summary"]["failed_calls"] for run in result["runs"])
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            logging.error("%s", error, exc_info=args.debug)
            return 1


if __name__ == "__main__":
    raise SystemExit(MatrixCommand.main())
