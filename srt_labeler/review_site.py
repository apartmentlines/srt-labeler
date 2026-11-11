"""Build a portable static audio-review site from the frozen reference set."""

from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
import logging
from pathlib import Path
import shutil
from typing import Any
import zipfile
import wave
import tempfile

import srt

from .eval_dataset import Dataset, subtitles, speaker
from .review_workbook import ReviewWorkbook

ROOT = Path("/")
LOGGER = logging.getLogger("review_site")


class ReviewSiteBuilder:
    """Render grouped speaker paragraphs and package the required local audio."""

    @staticmethod
    def prepare(path: Path, temporary: Path) -> Dataset:
        """Allow labeled drafts and audio alone, validating optional original SRTs."""
        data = json.loads(path.read_text())
        for number, row in enumerate(data["cases"]):
            for key in ("source", "reference", "audio"):
                if key in row:
                    row[key] = str((path.resolve().parent / row[key]).resolve())
            if "source" not in row:
                blocks = subtitles(Path(row["reference"]))
                for block in blocks:
                    block.content = speaker(block)[1]
                source = temporary / f"source-{number}.srt"
                source.write_text(srt.compose(blocks))
                row["source"] = str(source)
        manifest = temporary / "dataset.json"
        manifest.write_text(json.dumps(data))
        return Dataset(manifest)

    def __init__(self, dataset: Dataset, output: Path) -> None:
        self.output: Path = output.resolve()
        self.decisions: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[dict[str, str]] = []
        for case in dataset.cases:
            blocks = subtitles(case.reference)
            with wave.open(str(case.audio)) as audio:
                duration = audio.getnframes() / audio.getframerate()
            self.decisions[case.id] = [
                {
                    "index": block.index,
                    "label": speaker(block)[0],
                    "human_review_required": block.index in case.flagged,
                    "flags": (
                        ["possible-shared-error-review-whole-call"]
                        if case.full_review
                        else []
                    ),
                }
                for block in blocks
            ]
            self.calls.append(
                {
                    "recording_id": case.id,
                    "record_id": case.id,
                    "srt_path": str(case.source.resolve()),
                    "reference_path": str(case.reference.resolve()),
                    "audio_path": str(case.audio.resolve()),
                    "srt_sha256": hashlib.sha256(case.source.read_bytes()).hexdigest(),
                    "audio_sha256": hashlib.sha256(case.audio.read_bytes()).hexdigest(),
                    "audio_seconds": str(duration),
                }
            )
        self.total: int = sum(
            len(self.targets(row["recording_id"])) for row in self.calls
        )
        self.mapping: list[dict[str, Any]] = []

    def targets(self, recording_id: str) -> list[dict[str, Any]]:
        """Return exactly the unresolved blocks for a call."""
        return [
            block
            for block in self.decisions[recording_id]
            if block["human_review_required"]
        ]

    @staticmethod
    def clock(seconds: float) -> str:
        """Format a source offset for human navigation."""
        return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"

    @staticmethod
    def document(title: str, body: str, prefix: str = "") -> str:
        """Wrap escaped page content with relative stylesheet and script URLs."""
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{escape(title)} · Call review</title><link rel="icon" href="{prefix}assets/icon.svg" type="image/svg+xml"><link rel="stylesheet" href="{prefix}assets/style.css">
<script src="{prefix}assets/review.js" defer></script></head>
<body><main class="wrap">{body}</main></body></html>
"""

    def full_review(self, recording_id: str) -> bool:
        """Identify calls where a full listen is requested after audit conflicts."""
        return any(
            "possible-shared-error-review-whole-call" in block["flags"]
            for block in self.decisions[recording_id]
        )

    def index(self) -> None:
        """Write the call directory and short reviewer instructions."""
        cards = []
        for row in self.calls:
            rid = row["recording_id"]
            badge = (
                '<span class="badge">Listen through the whole call</span>'
                if self.full_review(rid)
                else ""
            )
            cards.append(
                f"""<a class="card call-link" href="calls/{rid}.html"><strong>Call {rid}</strong>
<small>{len(self.targets(rid))} {'block' if len(self.targets(rid)) == 1 else 'blocks'} to check · {self.clock(float(row['audio_seconds']))} audio</small>{badge}</a>"""
            )
        body = f"""<header><p class="eyebrow">Listening review</p><h1>Who is speaking?</h1>
<p class="muted">{len(self.calls)} calls · {self.total} highlighted blocks to check</p></header>
<section class="card instructions" aria-labelledby="instructions"><h2 id="instructions">Your review</h2>
<p><strong>Listen here and record your findings in the spreadsheet.</strong> Use the sheet named for the call and the row matching its block ID. Yellow rows match the highlighted blocks here.</p>
<ol><li>Listen to the beginning to learn the voices. <strong>Operator</strong> is the answering-service side. <strong>Caller</strong> is the person who placed the call, even if they work for the property.</li>
<li>Check every <span class="highlight">highlighted block</span> using <strong>Play block</strong> or <strong>Play with context</strong>. Listen through the whole call when requested.</li>
<li>Choose a <strong>Decision</strong>: <strong>Correct</strong> if the shown label is right, otherwise <strong>Operator</strong> or <strong>Caller</strong>. Use <strong>Both speakers</strong>, <strong>No speech</strong>, or <strong>Unsure</strong> when needed. For both speakers, explain who says which words in <strong>Notes</strong>. Blank means unreviewed.</li>
<li>If an <strong>unmarked block</strong> has the wrong speaker label, record your decision in that block’s spreadsheet row too.</li></ol>
<p>Save and return the completed spreadsheet.</p>
<a class="button primary" href="review-notes.xlsx" download="review-notes.xlsx">Download review spreadsheet</a></section>
<section aria-labelledby="calls"><h2 id="calls">Calls to review</h2><div class="call-list">{''.join(cards)}</div></section>"""
        (self.output / "index.html").write_text(self.document("Calls to review", body))

    def transcript(self, row: dict[str, str]) -> str:
        """Group consecutive equal labels as in _al_transcription_format_as_html.

        :param row: Frozen call metadata with original subtitle and audio paths.
        :return: Escaped transcript paragraphs with stable, individually linked blocks.
        :raises ValueError: Source content or reference label coverage has changed.
        """
        rid = row["recording_id"]
        source = ROOT / row["srt_path"]
        if hashlib.sha256(source.read_bytes()).hexdigest() != row["srt_sha256"]:
            raise ValueError(f"Changed subtitle source: {rid}")
        original = list(srt.parse(source.read_text()))
        labeled = list(srt.parse(Path(row["reference_path"]).read_text()))
        decisions = self.decisions[rid]
        if len(original) != len(labeled) or len(original) != len(decisions):
            raise ValueError(f"Block coverage mismatch: {rid}")
        groups: list[tuple[str, list[str]]] = []
        for source_block, reference, decision in zip(original, labeled, decisions):
            label = decision["label"]
            if (
                label not in {"Operator", "Caller"}
                or (
                    source_block.index,
                    source_block.start,
                    source_block.end,
                    source_block.content,
                )
                != (
                    reference.index,
                    reference.start,
                    reference.end,
                    reference.content.removeprefix(f"{label}: "),
                )
                or decision["index"] != source_block.index
            ):
                raise ValueError(
                    f"Reference/source mismatch: {rid}/{source_block.index}"
                )
            index = source_block.index
            block_id = f"B{index:03d}"
            note_id = f"{rid}-{block_id}"
            start, end = (
                source_block.start.total_seconds(),
                source_block.end.total_seconds(),
            )
            flag = " review" if decision["human_review_required"] else ""
            self.mapping.append(
                {
                    "identifier": note_id,
                    "recording_id": rid,
                    "record_id": row["record_id"],
                    "block_index": index,
                    "start": start,
                    "end": end,
                    "label_shown": label,
                    "requires_review": bool(flag),
                    "source_srt_sha256": row["srt_sha256"],
                }
            )
            block = f"""<span class="block{flag}" id="{block_id}" data-index="{index}" data-note-id="{note_id}" data-start="{start}" data-end="{end}"><a class="block-id" href="#{block_id}" aria-label="Select {note_id}, {self.clock(start)} to {self.clock(end)}" title="{note_id} · {self.clock(start)}–{self.clock(end)}">{block_id}</a><span class="words">{escape(source_block.content)}</span></span>"""
            if not groups or groups[-1][0] != label:
                groups.append((label, []))
            groups[-1][1].append(block)
        return "\n".join(
            f'<section class="turn {label.lower()}"><h3>{label}</h3><p>{" ".join(blocks)}</p></section>'
            for label, blocks in groups
        )

    def call_page(self, row: dict[str, str], position: int) -> None:
        """Render a complete call with target navigation and a single audio player."""
        rid = row["recording_id"]
        targets = self.targets(rid)
        navigation = ['<a href="../index.html">← All calls</a>']
        if position:
            navigation.append(
                f'<a href="{self.calls[position - 1]["recording_id"]}.html">Previous call</a>'
            )
        if position + 1 < len(self.calls):
            navigation.append(
                f'<a href="{self.calls[position + 1]["recording_id"]}.html">Next call →</a>'
            )
        links = " ".join(
            f'<a class="pill" href="#B{block["index"]:03d}">B{block["index"]:03d}</a>'
            for block in targets
        )
        notice = (
            '<p class="notice">Listen through this whole call. Check the highlighted blocks and note any other speaker labels that need correcting.</p>'
            if self.full_review(rid)
            else ""
        )
        first = f'{rid}-B{(targets or self.decisions[rid])[0]["index"]:03d}'
        transcript = self.transcript(row)
        body = f"""<nav class="page-nav" aria-label="Call navigation">{''.join(navigation)}</nav>
<header><p class="eyebrow">Call {position + 1} of {len(self.calls)}</p><h1>Call {rid}</h1>
<p class="muted">{len(targets)} {'block' if len(targets) == 1 else 'blocks'} to check · {self.clock(float(row['audio_seconds']))} audio</p></header>
<p>Use spreadsheet sheet <strong>{rid}</strong> and the matching block row for your Decision and Notes. Record speaker-label errors in unmarked blocks too.</p>
{notice}<div class="layout"><aside class="card player" aria-label="Audio and block controls"><h2>Listen</h2>
<audio controls preload="metadata" src="../audio/{rid}.wav" aria-label="Full audio for call {rid}"></audio>
<div class="js-only"><p class="count" id="review-position"></p><div class="selected-id" id="selected-id">{first}</div><div class="time" id="selected-time"></div>
<div class="actions"><button class="primary" id="play-block">Play block</button><button id="play-context">Play with context</button><button id="pause-audio">Pause</button></div>
<div class="actions"><button id="previous-block">← Previous</button><button id="next-block">Next →</button><button id="copy-id">Copy ID</button></div>
<div class="speed"><label for="speed">Speed</label><select id="speed"><option value="0.75">0.75×</option><option value="1" selected>1×</option><option value="1.25">1.25×</option></select></div>
<div class="actions"><button id="play-call">Play full call from start</button></div><p class="status" id="playback-status" role="status" aria-live="polite">Select a block marker, then press Play.</p></div>
<p><a href="../audio/{rid}.wav" download="{rid}.wav">Download audio</a></p><p><a href="../review-notes.xlsx" download="review-notes.xlsx">Download review spreadsheet</a></p>
<p class="count">Choose Correct if the label is right. For Both speakers, explain who says which words in Notes. Blank means unreviewed.</p>
<noscript><p>Enable JavaScript for block playback. You can still listen with the audio player and use the block identifiers below.</p></noscript></aside>
<section class="card transcript" aria-labelledby="transcript"><h2 id="transcript">Conversation</h2>
<p class="legend"><span class="highlight">Highlighted passages</span> need review. Click any block marker to select its audio. Labels shown here may be wrong.</p>
<nav class="review-links" aria-label="Blocks to review">{links}</nav>{transcript}</section></div>
<aside class="mobile-controls" aria-label="Selected block playback"><strong id="mobile-id">{first}</strong>
<div class="actions"><button class="primary" data-forward="play-block">Play block</button><button data-forward="play-context">With context</button><button data-forward="copy-id">Copy ID</button></div>
<div class="actions"><button data-forward="previous-block">← Previous</button><button data-forward="next-block">Next →</button><button data-forward="pause-audio">Pause</button></div></aside>"""
        (self.output / "calls" / f"{rid}.html").write_text(
            self.document(f"Call {rid}", body, "../")
        )

    def notes(self) -> None:
        """Generate the review workbook from the exact rendered block map."""
        ReviewWorkbook().write(self.output / "review-notes.xlsx", self.mapping)

    def build(self) -> None:
        """Generate all pages and copy verified audio, then create a portable ZIP."""
        if self.output.exists() or self.output.with_suffix(".zip").exists():
            raise ValueError(
                f"Output already exists; choose a new directory: {self.output}"
            )
        self.output.mkdir(parents=True)
        (self.output / "calls").mkdir()
        (self.output / "audio").mkdir()
        shutil.copytree(Path(__file__).parent / "review_assets", self.output / "assets")
        for position, row in enumerate(self.calls):
            source = ROOT / row["audio_path"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != row["audio_sha256"]:
                raise ValueError(f'Changed audio source: {row["recording_id"]}')
            shutil.copyfile(
                source, self.output / "audio" / f'{row["recording_id"]}.wav'
            )
            self.call_page(row, position)
            LOGGER.debug("Built call %s", row["recording_id"])
        self.index()
        self.notes()
        (self.output / "DEPLOY.txt").write_text(
            "Copy this entire directory to any webserver subdirectory and open index.html.\nKeep all files and subdirectories together. All links are relative.\nNo build, API, database, CDN or server-side application is required.\nSave and return the completed spreadsheet.\n"
        )
        (self.output / "review-site-map.json").write_text(
            json.dumps(
                {
                    "site_directory": self.output.name,
                    "blocks": self.mapping,
                },
                indent=2,
            )
            + "\n"
        )
        archive = self.output.with_suffix(".zip")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for path in sorted(self.output.rglob("*")):
                if path.is_file():
                    zipped.write(path, path.relative_to(self.output.parent))
        LOGGER.info(
            "Built %d calls and %d review targets: %s",
            len(self.calls),
            self.total,
            self.output,
        )
        LOGGER.info(
            "Archive: %s (%.1f MB)", archive, archive.stat().st_size / 1_000_000
        )


def main() -> None:
    """Parse the destination and build the standalone review package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="New site directory and adjacent ZIP"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    try:
        with tempfile.TemporaryDirectory() as temporary:
            dataset = ReviewSiteBuilder.prepare(args.dataset, Path(temporary))
            ReviewSiteBuilder(dataset, args.output).build()
    except (OSError, ValueError) as exc:
        LOGGER.error("%s", exc)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
