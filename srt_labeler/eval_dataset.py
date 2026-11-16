"""Create and validate portable SRT corpora and their review evidence."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any
import wave

import srt


def digest(path: Path) -> str:
    """Hash an input file for reproducible runs."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def subtitles(path: Path) -> list[srt.Subtitle]:
    """Read nonempty subtitles with unique, increasing positive indexes."""
    try:
        blocks = list(srt.parse(path.read_text(encoding="utf-8-sig")))
    except srt.SRTParseError as error:
        raise ValueError(f"Malformed SRT: {path}") from error
    indexes = [int(b.index) for b in blocks]
    if not blocks or indexes != sorted(set(indexes)) or indexes[0] < 1:
        raise ValueError(f"Invalid block indexes: {path}")
    if any(b.start.total_seconds() < 0 or b.end <= b.start for b in blocks):
        raise ValueError(f"Invalid timestamps: {path}")
    return blocks


def speaker(block: srt.Subtitle) -> tuple[str, str]:
    """Separate the exact supported prefix from original text."""
    match = re.match(r"^(Operator|Caller): (.*)$", block.content, re.DOTALL)
    if not match:
        raise ValueError(f"Missing binary label on block {block.index}")
    return match[1], match[2]


@dataclass
class Case:
    """One local source, audio and labeled reference with block annotations."""

    id: str
    source: Path
    audio: Path
    reference: Path
    split: str
    excluded: dict[int, str]
    flagged: list[int]
    full_review: bool
    record_id: str | None = None

    def validate(self) -> None:
        """Reject mismatched references and annotations before execution."""
        original, labeled = subtitles(self.source), subtitles(self.reference)
        if any(re.match(r"^(Operator|Caller): ", b.content) for b in original):
            raise ValueError(f"Source must be unlabeled: {self.id}")
        if len(original) != len(labeled):
            raise ValueError(f"Reference coverage mismatch: {self.id}")
        for a, b in zip(original, labeled):
            if (a.index, a.start, a.end, a.content) != (
                b.index,
                b.start,
                b.end,
                speaker(b)[1],
            ):
                raise ValueError(f"Reference/source mismatch: {self.id}/{a.index}")
        indexes = {b.index for b in original}
        if not set(self.excluded).issubset(indexes) or not set(self.flagged).issubset(
            indexes
        ):
            raise ValueError(f"Unknown annotated block: {self.id}")
        if any(not reason.strip() for reason in self.excluded.values()):
            raise ValueError("Every scoring exclusion needs a reason")
        if (
            type(self.full_review) is not bool
            or len(set(self.flagged)) != len(self.flagged)
            or any(type(index) is not int for index in self.flagged)
        ):
            raise ValueError("Invalid review flags or full-review state")
        try:
            with wave.open(str(self.audio)) as audio:
                duration = audio.getnframes() / audio.getframerate()
        except (wave.Error, EOFError) as error:
            raise ValueError(f"Unreadable WAV: {self.id}") from error
        if any(b.end.total_seconds() > duration + 0.1 for b in original):
            raise ValueError(f"Subtitle extends beyond audio: {self.id}")


class Dataset:
    """Read a versioned JSON manifest whose file paths are relative to itself."""

    def __init__(self, path: Path, split: str | None = None) -> None:
        self.path: Path = path.resolve()
        self.data: dict[str, Any] = json.loads(path.read_text())
        if self.data.get("schema_version") != 2:
            raise ValueError("Unsupported corpus schema_version")
        self.cases: list[Case] = []
        seen: set[str] = set()
        for row in self.data["cases"]:
            identifier = row["id"]
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]{1,31}", identifier)
                or identifier.casefold() in seen
                or identifier.casefold() == "_review_metadata"
            ):
                raise ValueError(f"Unsafe or duplicate call ID: {identifier}")
            seen.add(identifier.casefold())
            case = Case(
                identifier,
                self.path.parent / row["source"],
                self.path.parent / row["audio"],
                self.path.parent / row["reference"],
                row.get("split", ""),
                {int(k): v for k, v in row.get("excluded", {}).items()},
                row.get("flagged", []),
                row.get("full_review", False),
                row.get("record_id"),
            )
            case.validate()
            for key in ("source", "audio", "reference"):
                expected = row.get("hashes", {}).get(key)
                if expected and digest(getattr(case, key)) != expected:
                    raise ValueError(f"Changed {key}: {identifier}")
            if split is None or case.split == split:
                self.cases.append(case)
        if not self.cases:
            raise ValueError("No cases selected")
        Corpus.validate(self)

    def require_approved(self) -> None:
        """Reject provisional cases at the scoring boundary."""
        selected = {case.id for case in self.cases}
        for row in self.data["cases"]:
            if row["id"] in selected and any(
                item["state"] != "approved" for item in row["decisions"]
            ):
                raise ValueError(f"Unapproved case: {row['id']}")


class Corpus:
    """Maintain immutable, evidence-backed case revisions and derived references."""

    CLASSIFICATIONS: set[str] = {
        "Operator",
        "Caller",
        "Both speakers",
        "No speech",
        "Unsure",
    }
    STATES: set[str] = {"draft", "approved", "unresolved"}

    @staticmethod
    def identity(data: dict[str, Any]) -> str:
        """Hash content independently of the directory holding its artifacts."""
        content = copy.deepcopy(data)
        content.pop("content_sha256", None)
        for row in content["cases"]:
            for key in ("source", "reference", "audio"):
                row.pop(key, None)
        for artifact in content.get("artifacts", []):
            artifact.pop("path", None)
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    @staticmethod
    def validate(dataset: Dataset) -> None:
        """Validate identities, approval provenance, evidence and derived labels."""
        data = dataset.data
        if not isinstance(data.get("corpus_id"), str) or not data["corpus_id"]:
            raise ValueError("Missing corpus identity")
        if type(data.get("revision")) is not int or data["revision"] < 1:
            raise ValueError("Invalid corpus revision")
        if (data["revision"] == 1 and data.get("parent") is not None) or (
            data["revision"] > 1
            and not re.fullmatch(r"[a-f0-9]{64}", data.get("parent") or "")
        ):
            raise ValueError("Invalid parent revision identity")
        if data.get("content_sha256") != Corpus.identity(data):
            raise ValueError("Changed corpus content identity")
        for artifact in data.get("artifacts", []):
            Corpus.local_path(dataset.path.parent, artifact["path"])
            if digest(dataset.path.parent / artifact["path"]) != artifact["sha256"]:
                raise ValueError("Changed provenance artifact")
        evidence_hashes = {artifact["sha256"] for artifact in data.get("artifacts", [])}
        for row in data["cases"]:
            Corpus.validate_case(dataset.path.parent, row)
            for decision in row["decisions"]:
                if decision["state"] == "approved" and not set(
                    decision["provenance"].get("artifact_sha256", [])
                ).intersection(evidence_hashes):
                    raise ValueError("Approval lacks a preserved evidence artifact")

    @staticmethod
    def validate_case(root: Path, row: dict[str, Any]) -> None:
        """Check one case's complete decision coverage against source evidence."""
        if type(row.get("revision")) is not int or row["revision"] < 1:
            raise ValueError("Invalid case revision")
        if (
            row.get("holdout_exposed") is not None
            and type(row["holdout_exposed"]) is not bool
        ):
            raise ValueError("Holdout exposure must be true, false or unknown")
        for key in ("source", "reference", "audio"):
            Corpus.local_path(root, row[key])
            if digest(root / row[key]) != row.get("hashes", {}).get(key):
                raise ValueError(f"Unpinned or changed {key}: {row['id']}")
        if (
            row.get("original_draft_sha256")
            and digest(root / row["id"] / "original-draft.srt")
            != row["original_draft_sha256"]
        ):
            raise ValueError("Changed preserved original draft")
        blocks = subtitles(root / row["source"])
        labels = subtitles(root / row["reference"])
        decisions = row["decisions"]
        if [d["index"] for d in decisions] != [b.index for b in blocks]:
            raise ValueError("Decision coverage mismatch")
        excluded = {}
        for block, labeled, decision in zip(blocks, labels, decisions):
            Corpus.validate_decision(decision, row, block)
            classification = decision["classification"]
            expected = (
                classification
                if classification in {"Operator", "Caller"}
                else decision["compatibility_label"]
            )
            if speaker(labeled)[0] != expected:
                raise ValueError(
                    "Derived reference disagrees with authoritative decisions"
                )
            if decision["state"] != "approved" or classification not in {
                "Operator",
                "Caller",
            }:
                excluded[str(block.index)] = decision["rationale"]
        if row.get("excluded", {}) != excluded:
            raise ValueError("Derived scoring exclusions disagree with decisions")

    @staticmethod
    def local_path(root: Path, value: str) -> Path:
        """Keep revision writes and bundled provenance within the corpus directory."""
        path = Path(value)
        if path.is_absolute() or not (root / path).resolve().is_relative_to(
            root.resolve()
        ):
            raise ValueError(
                "Corpus artifacts must be contained in their revision directory"
            )
        return root / path

    @staticmethod
    def validate_decision(
        decision: dict[str, Any], row: dict[str, Any], block: srt.Subtitle
    ) -> None:
        """Keep approval distinct from automation and pin the audio locator."""
        if decision["classification"] not in Corpus.CLASSIFICATIONS:
            raise ValueError("Invalid classification")
        if decision["state"] not in Corpus.STATES:
            raise ValueError("Invalid review state")
        if decision["compatibility_label"] not in {"Operator", "Caller"}:
            raise ValueError("Invalid compatibility label")
        if (
            not isinstance(decision["rationale"], str)
            or not decision["rationale"].strip()
        ):
            raise ValueError("Decision requires rationale")
        actor = decision["actor"]
        if actor["type"] not in {"human", "assistant", "model", "unknown"}:
            raise ValueError("Invalid actor type")
        if not (actor.get("identity") is None or isinstance(actor["identity"], str)):
            raise ValueError("Invalid actor identity")
        if decision["state"] == "approved" and (
            actor["type"] != "human"
            or decision["classification"] == "Unsure"
            or not decision.get("provenance")
        ):
            raise ValueError(
                "Approval requires human evidence and a resolved classification"
            )
        if decision["evidence"] != {
            "audio_sha256": row["hashes"]["audio"],
            "start": block.start.total_seconds(),
            "end": block.end.total_seconds(),
        }:
            raise ValueError("Broken audio evidence locator")

    @staticmethod
    def derive(row: dict[str, Any], source: Path, reference: Path) -> None:
        """Generate compatibility SRT and masks from the authoritative decisions."""
        blocks = subtitles(source)
        if [b.index for b in blocks] != [d["index"] for d in row["decisions"]]:
            raise ValueError("Decision coverage mismatch")
        row["excluded"] = {}
        for block, decision in zip(blocks, row["decisions"]):
            classification = decision["classification"]
            label = (
                classification
                if classification in {"Operator", "Caller"}
                else decision["compatibility_label"]
            )
            block.content = f"{label}: {block.content}"
            if decision["state"] != "approved" or classification not in {
                "Operator",
                "Caller",
            }:
                row["excluded"][str(block.index)] = decision["rationale"]
        reference.write_text(srt.compose(blocks), encoding="utf-8")
        row["hashes"]["reference"] = digest(reference)

    @staticmethod
    def publish(data: dict[str, Any], staging: Path, output: Path) -> Path:
        """Validate a staged revision before making a new directory visible."""
        if output.exists():
            raise FileExistsError(output)
        data["content_sha256"] = Corpus.identity(data)
        path = staging / "dataset.json"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        Dataset(path)
        staging.rename(output)
        return output / "dataset.json"

    @staticmethod
    def copy_artifact(source: Path, staging: Path, name: str) -> dict[str, str]:
        """Preserve source evidence bytes as a hash-addressed evidence artifact."""
        destination = staging / "provenance" / name
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        return {
            "path": str(destination.relative_to(staging)),
            "sha256": digest(destination),
        }

    def create(self, inputs: Path, output: Path, corpus_id: str) -> Path:
        """Create an unapproved corpus from source files and draft labels.

        :param inputs: JSON selection of cases with source, draft and audio paths.
        :param output: New self-contained corpus directory.
        :param corpus_id: Stable identity for subsequent review revisions.
        :return: Validated corpus manifest path.
        """
        selection = json.loads(inputs.read_text())
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".corpus-", dir=output.parent
        ) as temporary:
            staging = Path(temporary) / "revision"
            staging.mkdir()
            rows = [
                self.create_case(row, inputs.parent, staging)
                for row in selection["cases"]
            ]
            return self.publish(
                {
                    "schema_version": 2,
                    "corpus_id": corpus_id,
                    "revision": 1,
                    "parent": None,
                    "artifacts": [],
                    "cases": rows,
                },
                staging,
                output,
            )

    def create_case(
        self, item: dict[str, Any], root: Path, staging: Path
    ) -> dict[str, Any]:
        """Freeze a new case's inputs without inferring human approval."""
        identifier = item["id"]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,31}", identifier):
            raise ValueError(f"Unsafe call ID: {identifier}")
        folder = staging / identifier
        folder.mkdir()
        draft = root / item["draft"]
        shutil.copyfile(draft, folder / "original-draft.srt")
        shutil.copyfile(draft, folder / "reference.srt")
        shutil.copyfile(root / item["audio"], folder / "audio.wav")
        if item.get("source"):
            shutil.copyfile(root / item["source"], folder / "source.srt")
        else:
            blocks = subtitles(draft)
            for block in blocks:
                block.content = speaker(block)[1]
            (folder / "source.srt").write_text(srt.compose(blocks), encoding="utf-8")
        case = Case(
            identifier,
            folder / "source.srt",
            folder / "audio.wav",
            folder / "reference.srt",
            item.get("split", ""),
            {},
            item.get("flagged", []),
            item.get("full_review", False),
            item.get("record_id", identifier),
        )
        case.validate()
        row: dict[str, Any] = {
            "id": identifier,
            "record_id": case.record_id,
            "revision": 1,
            "split": case.split,
            "holdout_exposed": item.get("holdout_exposed"),
            "flagged": case.flagged,
            "full_review": case.full_review,
            "original_draft_sha256": digest(draft),
            "hashes": {},
            "decisions": [],
        }
        for key in ("source", "reference", "audio"):
            path = getattr(case, key)
            row[key] = str(path.relative_to(staging))
            row["hashes"][key] = digest(path)
        for block, labeled in zip(subtitles(case.source), subtitles(case.reference)):
            label = speaker(labeled)[0]
            row["decisions"].append(
                {
                    "index": block.index,
                    "classification": label,
                    "compatibility_label": label,
                    "state": "draft",
                    "actor": {"type": "unknown", "identity": None},
                    "rationale": "Draft label awaiting listening review",
                    "notes": "",
                    "evidence": {
                        "audio_sha256": row["hashes"]["audio"],
                        "start": block.start.total_seconds(),
                        "end": block.end.total_seconds(),
                    },
                    "provenance": {},
                }
            )
        self.derive(row, case.source, case.reference)
        return row
