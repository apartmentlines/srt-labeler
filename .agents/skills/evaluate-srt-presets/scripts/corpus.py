"""Create or validate a local SRT corpus without provider requests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from srt_labeler.eval_dataset import Corpus, Dataset


class CorpusCommand:
    """Expose case creation and validation independently of model execution."""

    @staticmethod
    def main() -> int:
        """Parse inputs and emit a structured result with diagnostics on stderr."""
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("operation", choices=("create", "validate"))
        parser.add_argument(
            "--inputs", type=Path, help="JSON selection of new case files"
        )
        parser.add_argument("--corpus-id", help="Stable project/corpus identity")
        parser.add_argument("--output", type=Path, help="New corpus directory")
        parser.add_argument("--dataset", type=Path, help="Corpus manifest to validate")
        parser.add_argument("--require-approved", action="store_true")
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
        try:
            if args.operation == "create":
                if not args.inputs or not args.output or not args.corpus_id:
                    parser.error("create requires --inputs, --corpus-id and --output")
                path = Corpus().create(args.inputs, args.output, args.corpus_id)
            else:
                if not args.dataset:
                    parser.error("validate requires --dataset")
                path = args.dataset
            dataset = Dataset(path)
            if args.require_approved:
                dataset.require_approved()
            print(
                json.dumps(
                    {
                        "dataset": str(dataset.path),
                        "cases": len(dataset.cases),
                        "content_sha256": dataset.data["content_sha256"],
                    }
                )
            )
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            logging.error("%s", error, exc_info=args.debug)
            return 1


if __name__ == "__main__":
    raise SystemExit(CorpusCommand.main())
