import argparse
import signal
import time
import logging
from typing import Optional, List, Any
from copy import deepcopy
from .logger import Logger
from .pipeline import SrtLabelerPipeline

from .utils import (
    get_request,
    positive_int,
    fail_hard,
)
from .config import load_configuration, set_environment_variables
from .constants import (
    TRANSCRIPTION_STATE_ACTIVE,
    DEFAULT_STATS_DB,
)


class SrtLabeler:
    def __init__(
        self,
        api_key: str | None = None,
        file_api_key: str | None = None,
        domain: str | None = None,
        limit: int | None = None,
        min_id: int | None = None,
        max_id: int | None = None,
        continuous: int | None = None,
        stats_db: str | None = None,
        max_workers: int | None = None,
        debug: bool = False,
    ) -> None:
        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)
        self.api_key: str | None = api_key
        self.file_api_key: str | None = file_api_key
        self.domain: str | None = domain
        self.limit: int | None = limit
        self.min_id: int | None = min_id
        self.max_id: int | None = max_id
        self.continuous: int | None = continuous
        self.stats_db: str = stats_db if stats_db is not None else DEFAULT_STATS_DB
        self.max_workers: int | None = max_workers
        self.debug: bool = debug
        self.running: bool = False
        self.pipeline: SrtLabelerPipeline = self._initialize_pipeline()

    def _initialize_pipeline(self) -> SrtLabelerPipeline:
        return SrtLabelerPipeline(
            api_key=self.api_key,
            file_api_key=self.file_api_key,
            domain=self.domain,
            max_workers=self.max_workers,
            stats_db=self.stats_db,
            debug=self.debug,
        )

    def build_retrieve_request_url(self) -> str:
        return f"https://{self.domain}/al/transcriptions/retrieve/operator-recordings/{TRANSCRIPTION_STATE_ACTIVE}"

    def build_retrieve_request_params(self) -> dict:
        params = {"api_key": str(self.api_key)}
        if self.limit is not None:
            params["limit"] = str(self.limit)
        if self.min_id is not None:
            params["min_id"] = str(self.min_id)
        if self.max_id is not None:
            params["max_id"] = str(self.max_id)
        return params

    def retrieve_transcription_data(self) -> list[dict[str, Any]] | None:
        url = self.build_retrieve_request_url()
        try:
            params = self.build_retrieve_request_params()
            log_params = deepcopy(params)
            log_params["api_key"] = "REDACTED"
            self.log.debug(
                f"Retrieving file data from URL: {url}, params: {log_params}"
            )
            response = get_request(url, params)
            resp_json = response.json()
            if resp_json.get("success"):
                transcriptions = resp_json.get("files", [])
                self.log.info(
                    f"Retrieved {len(transcriptions)} transcriptions for processing"
                )
                return transcriptions
            else:
                fail_hard("Failed to retrieve files.")
        except Exception as e:
            fail_hard(f"Error retrieving files: {e}")

    def setup_configuration(self) -> None:
        self.log.debug("Setting up configuration")
        if not self.api_key or not self.file_api_key or not self.domain:
            fail_hard("API key, file API key, and domain must be provided")
        set_environment_variables(self.api_key, self.file_api_key, self.domain)
        self.log.info("Configuration loaded successfully")

    def run_single(self) -> None:
        self.running = True
        transcriptions = self.retrieve_transcription_data()
        with self.pipeline:  # Use context manager here
            self.pipeline.process_transcriptions(transcriptions)

    def _signal_handler(self, sig, frame):
        self.log.info("Received interrupt signal, shutting down gracefully...")
        self.running = False

    def run_continuous(self, sleep_seconds: int) -> None:
        self.running = True
        self.log.info(f"Starting continuous mode with {sleep_seconds} second intervals")
        self.log.info("Press Ctrl+C to exit gracefully")

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)

        # Use context manager outside the loop
        with self.pipeline:
            while self.running:
                transcriptions = self.retrieve_transcription_data()
                self.pipeline.process_transcriptions(transcriptions)
                self.log.debug(f"Sleeping for {sleep_seconds} seconds")
                for _ in range(sleep_seconds):
                    if not self.running:
                        break
                    time.sleep(1)

    def run(self) -> None:
        self.log.info("Starting transcription pipeline")
        self.setup_configuration()
        self.log.info("Starting pipeline execution")
        if self.continuous:
            self.run_continuous(self.continuous)
        else:
            self.run_single()
        self.log.info("Transcription pipeline completed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SRT labeler pipeline.")
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Only process this many transcriptions, default unlimited",
    )
    parser.add_argument(
        "--continuous",
        type=int,
        help="Run continuously, sleeping this many seconds between batch processing cycles",
    )
    parser.add_argument(
        "--min-id",
        type=positive_int,
        help="Only process transcriptions with ID >= this value",
    )
    parser.add_argument(
        "--max-id",
        type=positive_int,
        help="Only process transcriptions with ID <= this value",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="API key (also can be provided as SRT_LABELER_API_KEY environment variable)",
    )
    parser.add_argument(
        "--file-api-key",
        type=str,
        help="File API key (also can be provided as SRT_LABELER_FILE_API_KEY environment variable)",
    )
    parser.add_argument(
        "--domain",
        type=str,
        help="Transcription domain used for REST operations (also can be provided as SRT_LABELER_DOMAIN environment variable)",
    )
    parser.add_argument(
        "--stats-db",
        type=str,
        default=DEFAULT_STATS_DB,
        help="Path to SQLite stats database (default: %(default)s)",
    )
    parser.add_argument(
        "--max-workers",
        type=positive_int,
        help="Maximum number of worker threads for processing transcriptions",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    try:
        api_key, file_api_key, domain = load_configuration(args)
    except ValueError as e:
        fail_hard(str(e))
        return

    labeler = SrtLabeler(
        api_key=api_key,
        file_api_key=file_api_key,
        domain=domain,
        limit=args.limit,
        min_id=args.min_id,
        max_id=args.max_id,
        continuous=args.continuous,
        stats_db=args.stats_db,
        max_workers=args.max_workers,
        debug=args.debug,
    )
    labeler.run()


if __name__ == "__main__":
    main()
