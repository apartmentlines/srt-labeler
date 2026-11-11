"""Load portable evaluation cases and import existing reviewed CSV datasets."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import re
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
        with wave.open(str(self.audio)) as audio:
            duration = audio.getnframes() / audio.getframerate()
        if any(b.end.total_seconds() > duration + 0.1 for b in original):
            raise ValueError(f"Subtitle extends beyond audio: {self.id}")


class Dataset:
    """Read a versioned JSON manifest whose file paths are relative to itself."""

    def __init__(self, path: Path, split: str | None = None) -> None:
        self.path: Path = path.resolve()
        self.data: dict[str, Any] = json.loads(path.read_text())
        if self.data.get("schema_version") != 1:
            raise ValueError("Expected dataset schema_version 1")
        self.cases: list[Case] = []
        seen: set[str] = set()
        for row in self.data["cases"]:
            identifier = row["id"]
            if (
                not re.fullmatch(r"[A-Za-z0-9_-]{1,31}", identifier)
                or identifier.casefold() in seen
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

    @staticmethod
    def import_reviewed(
        manifest: Path, reviews: Path, root: Path, output: Path
    ) -> None:
        """Convert the established CSV artifacts into a portable, pinned manifest."""
        with manifest.open(newline="") as stream:
            cases = list(csv.DictReader(stream))
        with reviews.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        records = []
        consumed = 0
        for row in cases:
            rid = row["recording_id"]
            annotations = [r for r in rows if r["recording_id"] == rid]
            original = subtitles(root / row["srt_path"])
            if [int(r["block_index"]) for r in annotations] != [
                b.index for b in original
            ]:
                raise ValueError(f"Review coverage mismatch: {rid}")
            if any(r["human_verified"].lower() != "true" for r in annotations):
                raise ValueError(f"Unverified reference: {rid}")
            if any(
                r["score_eligible"].lower() not in {"true", "false"}
                for r in annotations
            ):
                raise ValueError("Invalid scoring eligibility")
            paths = {
                "source": root / row["srt_path"],
                "audio": root / row["audio_path"],
                "reference": root / annotations[0]["reference_srt_path"],
            }
            for key, column in (("source", "srt_sha256"), ("audio", "audio_sha256")):
                if row.get(column) and digest(paths[key]) != row[column]:
                    raise ValueError(f"Changed reviewed {key}: {rid}")
            labeled = subtitles(paths["reference"])
            if len(labeled) != len(annotations) or any(
                speaker(block)[0] != annotation["label"]
                or annotation["reference_srt_path"]
                != annotations[0]["reference_srt_path"]
                for block, annotation in zip(labeled, annotations)
            ):
                raise ValueError(f"Reference labels differ from human review: {rid}")
            records.append(
                {
                    "id": rid,
                    "split": row["split"],
                    **{
                        k: os.path.relpath(v.resolve(), output.resolve().parent)
                        for k, v in paths.items()
                    },
                    "hashes": {k: digest(v) for k, v in paths.items()},
                    "excluded": {
                        r["block_index"]: r["status"]
                        for r in annotations
                        if r["score_eligible"].lower() == "false"
                    },
                }
            )
            consumed += len(annotations)
        if consumed != len(rows):
            raise ValueError("Unmatched review rows")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump({"schema_version": 1, "cases": records}, stream, indent=2)
        Dataset(output)


def main() -> None:
    """Import reviewed artifacts without modifying the source files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument(
        "--root", type=Path, required=True, help="Base for paths in the CSVs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    Dataset.import_reviewed(args.manifest, args.reviews, args.root, args.output)


if __name__ == "__main__":
    main()
