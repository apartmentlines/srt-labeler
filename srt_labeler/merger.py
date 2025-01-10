#!/usr/bin/env python3

"""Merge labeled and unlabeled SRT subtitle files.

This module transfers speaker labels from an AI-generated labeled SRT file to
the original unlabeled version, discarding all non-label content from the
labeled file to avoid propagating any transcription errors. Only the speaker
labels are extracted and applied to the corresponding subtitles from the
unlabeled file.
"""

import sys
import re
import srt
import argparse
from pathlib import Path
from typing import List, Any

from download_pipeline_processor.logger import Logger

DEFAULT_VALID_LABELS = ["operator", "caller"]


class SrtMergeError(Exception):
    """Exception raised for errors during SRT merging."""

    def __init__(self, message: str) -> None:
        super().__init__(f"SRT merge error: {message}")


class SrtMerger:
    """Merge speaker labels from labeled SRT into unlabeled SRT content."""

    def __init__(
        self,
        valid_labels: List[str] = DEFAULT_VALID_LABELS,
        debug: bool = False,
        srt_module: Any = srt,
    ) -> None:
        """Initialize SRT merger with validation settings.

        :param valid_labels: List of allowed speaker labels
        :param debug: Enable debug logging
        :param srt_module: srt module to use (for testing)
        """
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.valid_labels = [label.lower() for label in valid_labels]
        self.srt_module = srt_module

    def _validate_subtitle_counts(
        self, subs_without: List[srt.Subtitle], subs_with: List[srt.Subtitle]
    ) -> None:
        """Validate that subtitle counts match between files.

        :param subs_without: Subtitles from unlabeled file
        :param subs_with: Subtitles from labeled file
        :raises SrtMergeError: If either file is empty or counts don't match
        """
        if not subs_without or not subs_with:
            raise SrtMergeError("Input SRT content cannot be empty")
        if len(subs_without) != len(subs_with):
            raise SrtMergeError("SRT files have different number of subtitles")

    def _validate_labels(self, subs: List[srt.Subtitle]) -> None:
        """Validate speaker labels in subtitles.

        :param subs: List of subtitles to validate
        :raises SrtMergeError: If any subtitle has missing or invalid label
        """
        for sub in subs:
            label = self._extract_speaker(sub.content)
            if label.lower() != "unknown" and label.lower() not in self.valid_labels:
                raise SrtMergeError(
                    f"Invalid label '{label}' found in subtitle at {sub.start} - {sub.end}. "
                    f"Allowed labels are: {self.valid_labels}"
                )

    def _extract_speaker(self, text: str) -> str:
        """Extract speaker label from SRT content line.

        :param text: SRT content line to parse
        :return: Extracted speaker label or 'Unknown' if not found
        """
        speaker_match = re.match(r"^([A-Za-z0-9]+):", text)
        if speaker_match:
            return speaker_match.group(1)
        return "Unknown"

    def _apply_labels(
        self, subs_without: List[srt.Subtitle], subs_with: List[srt.Subtitle]
    ) -> List[srt.Subtitle]:
        """Apply speaker labels from labeled to unlabeled subtitles.

        :param subs_without: List of subtitles without speaker labels
        :param subs_with: List of subtitles with speaker labels
        :return: List of subtitles with transferred speaker labels
        """
        for i in range(len(subs_without)):
            speaker = self._extract_speaker(subs_with[i].content)
            subs_without[i].content = f"{speaker}: {subs_without[i].content}"
        return subs_without

    def _parse_srt_content(self, srt_content: str) -> List[srt.Subtitle]:
        try:
            return list(self.srt_module.parse(srt_content))
        except self.srt_module.SRTParseError as e:
            raise SrtMergeError(f"Invalid SRT format: {str(e)}")

    def read_file(self, path: Path) -> str:
        """Read text content from file at given path.

        :param path: Filesystem path to read from
        :return: Contents of file as string
        :raises SrtMergeError: If file cannot be read or decoded
        """
        try:
            return path.read_text(encoding="utf-8")
        except (IOError, UnicodeDecodeError) as e:
            raise SrtMergeError(f"Failed to read file {path}: {str(e)}")

    def merge(self, unlabeled_srt: str, labeled_srt: str) -> str:
        """Merge speaker labels from labeled SRT into unlabeled content.

        :param unlabeled_srt: SRT content without speaker labels
        :param labeled_srt: SRT content with speaker labels
        :return: Merged SRT content with speaker labels
        :raises SrtMergeError: If either input is None or empty
        """
        if unlabeled_srt is None or labeled_srt is None:
            raise SrtMergeError("Input SRT content cannot be None")
        if not unlabeled_srt or not labeled_srt:
            raise SrtMergeError("Empty SRT file provided")
        self.log.debug("Starting SRT merge process")
        subs_without = self._parse_srt_content(unlabeled_srt)
        subs_with = self._parse_srt_content(labeled_srt)

        self._validate_labels(subs_with)
        self._validate_subtitle_counts(subs_without, subs_with)

        labeled_subs = self._apply_labels(subs_without, subs_with)
        self.log.debug("Successfully merged SRT files")
        return self.srt_module.compose(labeled_subs)


def parse_arguments(args=None) -> argparse.Namespace:
    """Parse command line arguments for SRT merging.

    :param args: Argument list to parse
    :return: Namespace containing parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Merge labeled and unlabeled SRT files"
    )
    parser.add_argument(
        "--unlabeled-srt",
        type=Path,
        required=True,
        help="Path to unlabeled SRT file",
    )
    parser.add_argument(
        "--labeled-srt",
        type=Path,
        required=True,
        help="Path to labeled SRT file",
    )
    parser.add_argument(
        "--valid-labels",
        type=lambda x: [label.strip() for label in x.split(",")],
        default=DEFAULT_VALID_LABELS,
        help="Comma-separated list of valid speaker labels",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main(args=None) -> None:
    """Main entry point for SRT merging script."""
    if args is None:
        args = sys.argv[1:]
    args = parse_arguments(args)
    try:
        merger = SrtMerger(valid_labels=args.valid_labels, debug=args.debug)
        unlabeled_content = merger.read_file(args.unlabeled_srt)
        labeled_content = merger.read_file(args.labeled_srt)
        result = merger.merge(unlabeled_content, labeled_content)
        print(result)
    except SrtMergeError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
