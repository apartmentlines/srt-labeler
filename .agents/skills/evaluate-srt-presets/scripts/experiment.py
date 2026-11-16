"""Prepare, collect, pause, resume, rescore and inspect immutable SRT experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random

from srt_labeler.eval_dataset import Dataset
from srt_labeler.evaluate import Evaluation, Experiment


class ExperimentCommand:
    """Keep model collection separate from preparation and local review operations."""

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "operation",
            choices=(
                "prepare",
                "collect",
                "status",
                "pause",
                "resume",
                "rescore",
                "compare",
            ),
        )
        parser.add_argument(
            "--run",
            type=Path,
            required=True,
            help="New destination for prepare/rescore, existing collection otherwise",
        )
        parser.add_argument("--dataset", type=Path)
        parser.add_argument("--preset")
        parser.add_argument("--template", type=Path)
        parser.add_argument(
            "--origin",
            type=Path,
            help="Original run for rescore or second run for compare",
        )
        parser.add_argument("--split")
        parser.add_argument("--ids", nargs="+", help="Explicit ordered case selection")
        parser.add_argument("--sample", type=int, help="Seeded sample size")
        parser.add_argument("--seed", type=int, default=0)
        parser.add_argument("--timeout", type=float, default=600)
        parser.add_argument("--retry-limit", type=int, default=0)
        parser.add_argument(
            "--retry",
            nargs="+",
            help="Explicit failed case IDs to retry within the frozen limit",
        )
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
        engine = Experiment()
        try:
            if args.operation in {"prepare", "rescore"}:
                if args.dataset is None:
                    parser.error("This operation requires --dataset")
                dataset = Dataset(args.dataset, args.split)
                if args.ids:
                    by_id = {case.id: case for case in dataset.cases}
                    if len(set(args.ids)) != len(args.ids) or not set(
                        args.ids
                    ).issubset(by_id):
                        raise ValueError("Unknown or duplicate selected case IDs")
                    dataset.cases = [by_id[identifier] for identifier in args.ids]
                if args.sample is not None:
                    if args.sample < 1 or args.sample > len(dataset.cases):
                        raise ValueError(
                            "Sample size is outside the selected case population"
                        )
                    dataset.cases = random.Random(args.seed).sample(
                        dataset.cases, args.sample
                    )
                if args.operation == "prepare":
                    if not args.preset or args.template is None:
                        parser.error("prepare requires --preset and --template")
                    result = engine.prepare(
                        dataset,
                        args.run,
                        args.preset,
                        args.template,
                        args.timeout,
                        args.retry_limit,
                        selection={
                            "ordered_ids": [case.id for case in dataset.cases],
                            "split": args.split,
                            "seed": args.seed,
                            "sample": args.sample,
                        },
                    )
                else:
                    if args.origin is None:
                        parser.error("rescore requires --origin")
                    result = engine.rescore(dataset, args.origin, args.run)
            elif args.operation in {"collect", "resume"}:
                result = engine.collect(
                    args.run, args.debug, args.operation == "resume", args.retry
                )
            elif args.operation == "status":
                result = engine.status(args.run)
            elif args.operation == "pause":
                if not (args.run / "experiment.json").is_file():
                    raise ValueError("Unknown run")
                engine.atomic(
                    args.run / "pause.json",
                    {
                        "requested_at": engine.now(),
                        "mode": "finish in-flight request, then stop",
                    },
                )
                result = engine.status(args.run)
            else:
                if args.origin is None:
                    parser.error("compare requires --origin")
                result = Evaluation.compare(args.run, args.origin)
            print(json.dumps(result, indent=2))
            return int(bool(result.get("failed_calls", 0)))
        except (OSError, ValueError, KeyError, TypeError) as error:
            logging.error("%s", error, exc_info=args.debug)
            return 1


if __name__ == "__main__":
    raise SystemExit(ExperimentCommand.main())
