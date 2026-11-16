"""Validate review proposals and reconcile explicit decisions into a new corpus."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
from pathlib import Path
import tempfile
from typing import Any

from srt_labeler.eval_dataset import Corpus, Dataset, digest
from srt_labeler.evaluate import Experiment
from srt_labeler.review_workbook import ReviewWorkbook


class Feedback:
    """Keep browser/workbook proposals separate from approved case authority."""

    OPTIONS: set[str] = {
        "",
        "Correct",
        "Operator",
        "Caller",
        "Both speakers",
        "No speech",
        "Unsure",
    }

    @staticmethod
    def review_identity(review: dict[str, Any]) -> str:
        """Identify the complete frozen presentation, independent of hosting."""
        value = copy.deepcopy(review)
        value["base"].pop("review_id", None)
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def __init__(self, dataset: Dataset, review: dict[str, Any]) -> None:
        self.dataset: Dataset = dataset
        self.review: dict[str, Any] = review
        data = dataset.data
        base = review["base"]
        if data["schema_version"] != 2 or (
            base["corpus_id"],
            base["corpus_revision"],
            base["corpus_sha256"],
        ) != (data["corpus_id"], data["revision"], data["content_sha256"]):
            raise ValueError("Stale or incompatible review corpus")
        if base["review_id"] != self.review_identity(review):
            raise ValueError("Changed review presentation")
        self.blocks: dict[str, dict[str, Any]] = {}
        rows = {row["id"]: row for row in data["cases"]}
        expected = {
            (r["id"], d["index"]) for r in rows.values() for d in r["decisions"]
        }
        observed = set()
        for block in review["blocks"]:
            identifier = block["identifier"]
            if identifier in self.blocks:
                raise ValueError("Duplicate presentation block")
            self.blocks[identifier] = block
            row = rows[block["recording_id"]]
            key = (row["id"], block["block_index"])
            observed.add(key)
            if identifier != f"{row['id']}-B{block['block_index']:03d}" or (
                block["case_revision"] != row["revision"]
                or block["source_srt_sha256"] != row["hashes"]["source"]
                or block["audio_sha256"] != row["hashes"]["audio"]
                or block["reference_sha256"] != row["hashes"]["reference"]
            ):
                raise ValueError("Stale block identity")
            decision = next(
                d for d in row["decisions"] if d["index"] == block["block_index"]
            )
            label = (
                decision["classification"]
                if decision["classification"] in {"Operator", "Caller"}
                else decision["compatibility_label"]
            )
            if block["label_shown"] != label or any(
                block[k] != decision["evidence"][k] for k in ("start", "end")
            ):
                raise ValueError("Presentation differs from corpus")
        if observed != expected:
            raise ValueError("Review presentation coverage mismatch")

    def validate(self, feedback: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Reject stale, unknown, duplicate and malformed edits before mutation."""
        if (
            feedback.get("schema_version") != 1
            or feedback.get("kind") != "srt-review-feedback"
        ):
            raise ValueError("Unsupported feedback format")
        if feedback.get("base") != self.review["base"]:
            raise ValueError("Stale feedback base")
        if not isinstance(feedback.get("session"), str) or not feedback["session"]:
            raise ValueError("Missing review session")
        if (
            type(feedback.get("edit_revision")) is not int
            or feedback["edit_revision"] < 0
        ):
            raise ValueError("Invalid edit revision")
        if not (
            feedback.get("reviewer") is None or isinstance(feedback["reviewer"], str)
        ):
            raise ValueError("Invalid reviewer identity")
        entries = {}
        for entry in feedback["entries"]:
            identifier = entry["identifier"]
            if identifier not in self.blocks or identifier in entries:
                raise ValueError("Unknown or duplicate feedback block")
            block = self.blocks[identifier]
            if entry.get("case_revision") != block["case_revision"]:
                raise ValueError("Stale case revision")
            if entry.get("decision") not in self.OPTIONS or entry.get("state") not in {
                "untouched",
                "draft",
                "completed",
            }:
                raise ValueError("Malformed decision or completion state")
            if not isinstance(entry.get("notes"), str):
                raise ValueError("Malformed notes")
            if entry["state"] == "completed" and not entry["decision"]:
                raise ValueError("Completed review requires an explicit decision")
            if entry["state"] == "untouched" and (entry["decision"] or entry["notes"]):
                raise ValueError("Untouched review cannot contain edits")
            entries[identifier] = entry
        if set(entries) != set(self.blocks):
            raise ValueError("Feedback must preserve complete block coverage")
        return entries

    def reconcile(
        self,
        feedback: dict[str, Any],
        resolutions: dict[str, Any],
        output: Path,
    ) -> Path:
        """Publish accepted proposals as a new revision, retaining all dispositions."""
        entries = self.validate(feedback)
        if resolutions.get("base") != feedback["base"]:
            raise ValueError("Resolution base differs from feedback")
        encoded = json.dumps(feedback, sort_keys=True, separators=(",", ":")).encode()
        feedback_hash = hashlib.sha256(encoded).hexdigest()
        if resolutions.get("feedback_sha256") != feedback_hash:
            raise ValueError("Resolution applies to different feedback")
        actions = self.validate_resolutions(entries, resolutions)
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".reconcile-", dir=output.parent
        ) as temporary:
            staging = Path(temporary) / "revision"
            data = Experiment.freeze_corpus(self.dataset, staging).data
            data["parent"] = data["content_sha256"]
            data["revision"] += 1
            for name, record in (
                ("feedback", feedback),
                ("resolutions", resolutions),
                ("review", self.review),
            ):
                path = staging / "provenance" / f"{name}-{data['revision']}.json"
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps(record, indent=2), encoding="utf-8")
                data["artifacts"].append(
                    {"path": str(path.relative_to(staging)), "sha256": digest(path)}
                )
            evidence = [artifact["sha256"] for artifact in data["artifacts"][-3:]]
            for row in data["cases"]:
                changed = False
                for decision in row["decisions"]:
                    identifier = f"{row['id']}-B{decision['index']:03d}"
                    action = actions.get(identifier)
                    if action and action["action"] == "accept":
                        self.accept(
                            decision,
                            entries[identifier],
                            action,
                            feedback,
                            feedback_hash,
                        )
                        decision["provenance"]["artifact_sha256"] = evidence
                        changed = True
                if changed:
                    row["revision"] += 1
                    Corpus.derive(
                        row, staging / row["source"], staging / row["reference"]
                    )
            return Corpus.publish(data, staging, output)

    @staticmethod
    def validate_resolutions(
        entries: dict[str, dict[str, Any]],
        resolutions: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Require a disposition, accountable actor, and rationale for every edit."""
        actions = {}
        for action in resolutions["decisions"]:
            identifier = action["identifier"]
            if identifier not in entries or identifier in actions:
                raise ValueError("Unknown or duplicate resolution")
            if action["action"] not in {"accept", "reject", "defer"}:
                raise ValueError("Invalid resolution")
            if not action.get("rationale", "").strip() or not action.get(
                "actor", {}
            ).get("identity"):
                raise ValueError("Resolution needs actor and rationale")
            if action["actor"]["type"] not in {"human", "assistant"}:
                raise ValueError("Invalid resolution actor")
            if (
                action["action"] == "accept"
                and entries[identifier]["state"] != "completed"
            ):
                raise ValueError("Only explicitly completed proposals can be accepted")
            actions[identifier] = action
        edited = {
            key for key, value in entries.items() if value["state"] != "untouched"
        }
        if set(actions) != edited:
            raise ValueError("Every edited block needs an explicit disposition")
        return actions

    def accept(
        self,
        decision: dict[str, Any],
        entry: dict[str, Any],
        action: dict[str, Any],
        feedback: dict[str, Any],
        feedback_hash: str,
    ) -> None:
        """Apply a human listening judgment; preserve uncertainty and attribution."""
        if not feedback.get("reviewer", ""):
            raise ValueError("Accepted feedback requires the human reviewer's identity")
        classification = entry["decision"]
        if classification == "Correct":
            classification = self.blocks[entry["identifier"]]["label_shown"]
        decision.update(
            {
                "classification": classification,
                "state": "unresolved" if classification == "Unsure" else "approved",
                "actor": {"type": "human", "identity": feedback["reviewer"]},
                "rationale": action["rationale"],
                "notes": entry["notes"],
                "provenance": {
                    "feedback_sha256": feedback_hash,
                    "review_id": feedback["base"]["review_id"],
                    "accepted_by": action["actor"],
                    "proposal": entry,
                },
            }
        )


class FeedbackCommand:
    """Expose validation, XLSX conversion and reconciliation as local operations."""

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument(
            "operation", choices=("validate", "reconcile", "xlsx", "json")
        )
        parser.add_argument("--dataset", type=Path, required=True)
        parser.add_argument("--review", type=Path, required=True)
        parser.add_argument("--feedback", type=Path, required=True)
        parser.add_argument("--resolutions", type=Path)
        parser.add_argument("--output", type=Path)
        parser.add_argument(
            "--reviewer",
            help="Identify an otherwise unnamed human spreadsheet reviewer",
        )
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
        try:
            review = json.loads(args.review.read_text())
            engine = Feedback(Dataset(args.dataset), review)
            feedback = (
                ReviewWorkbook().read(args.feedback)
                if args.feedback.suffix.lower() == ".xlsx"
                else json.loads(args.feedback.read_text())
            )
            if args.reviewer:
                if feedback.get("reviewer") not in (None, "", args.reviewer):
                    raise ValueError(
                        "Reviewer override conflicts with recorded identity"
                    )
                feedback["reviewer"] = args.reviewer
            entries = engine.validate(feedback)
            result: dict[str, Any] = {
                "entries": len(entries),
                "feedback_sha256": hashlib.sha256(
                    json.dumps(feedback, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
            if args.operation != "validate" and args.output is None:
                parser.error("This operation requires --output")
            if args.operation == "reconcile":
                if args.resolutions is None:
                    parser.error("reconcile requires --resolutions")
                result["dataset"] = str(
                    engine.reconcile(
                        feedback, json.loads(args.resolutions.read_text()), args.output
                    )
                )
            elif args.operation == "xlsx":
                ReviewWorkbook().write(args.output, review["blocks"], feedback)
            elif args.operation == "json":
                with args.output.open("x") as stream:
                    json.dump(feedback, stream, indent=2)
            print(json.dumps(result))
            return 0
        except (OSError, ValueError, KeyError, TypeError) as error:
            logging.error("%s", error, exc_info=args.debug)
            return 1


if __name__ == "__main__":
    raise SystemExit(FeedbackCommand.main())
