"""Offline subprocess fixture exercising the collector's actual launch handshake."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import srt


class FakeLocalWorker:
    """Stand in for a bounded provider process without network access."""

    @staticmethod
    def main() -> int:
        parser = argparse.ArgumentParser(description=__doc__)
        for name in (
            "srt",
            "audio",
            "output-dir",
            "template-file",
            "settings-file",
            "start-gate",
            "completion-file",
        ):
            parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--preset", required=True)
        parser.add_argument("--identifier", required=True)
        parser.add_argument("--deadline-seconds", type=float, required=True)
        parser.add_argument("--debug", action="store_true")
        args = parser.parse_args()
        deadline = time.monotonic() + 10
        while not args.start_gate.exists():
            if time.monotonic() >= deadline:
                return 2
            time.sleep(0.01)
        gate = json.loads(args.start_gate.read_text())
        attempt = json.loads((args.start_gate.parent / "attempt.json").read_text())
        if (
            gate["pid"] != os.getpid()
            or attempt["pid"] != os.getpid()
            or attempt["request_identifier"] != args.identifier
        ):
            return 3
        args.output_dir.mkdir()
        (args.output_dir / "response.txt").write_text("fixture partial response")
        if args.preset == "timeout-fixture":
            time.sleep(30)
        blocks = list(srt.parse(args.srt.read_text()))
        for block in blocks:
            block.content = "Operator: " + block.content
        (args.output_dir / "final.srt").write_text(srt.compose(blocks))
        args.completion_file.write_text(
            json.dumps({"returncode": 0, "finished_at": time.time()})
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(FakeLocalWorker.main())
