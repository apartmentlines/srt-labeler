import argparse
from typing import Optional, List
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
)


class SrtLabeler:
    def __init__(
        self,
        api_key: Optional[str] = None,
        file_api_key: Optional[str] = None,
        domain: Optional[str] = None,
        limit: Optional[int] = None,
        min_id: Optional[int] = None,
        max_id: Optional[int] = None,
        debug: bool = False,
    ) -> None:
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.api_key = api_key
        self.file_api_key = file_api_key
        self.domain = domain
        self.limit = limit
        self.min_id = min_id
        self.max_id = max_id
        self.debug = debug
        self.pipeline = self._initialize_pipeline()

    def _initialize_pipeline(self) -> SrtLabelerPipeline:
        return SrtLabelerPipeline(
            api_key=self.api_key,
            file_api_key=self.file_api_key,
            domain=self.domain,
            debug=self.debug,
        )

    def build_retrieve_request_url(self) -> str:
        return f"https://{self.domain}/al/transcriptions/retrieve/operator-recordings/{TRANSCRIPTION_STATE_ACTIVE}"

    def build_retrieve_request_params(self) -> dict:
        params = {"api_key": self.api_key}
        if self.limit is not None:
            params["limit"] = self.limit
        if self.min_id is not None:
            params["min_id"] = self.min_id
        if self.max_id is not None:
            params["max_id"] = self.max_id
        return params

    def retrieve_transcription_data(self) -> List[dict]:
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

    def run(self) -> None:
        self.log.info("Starting transcription pipeline")
        self.setup_configuration()
        transcriptions = self.retrieve_transcription_data()
        if not transcriptions:
            self.log.info("No transcriptions to process")
            return
        self.log.info("Starting pipeline execution")
        with self.pipeline:  # Use context manager here
            self.pipeline.process_transcriptions(transcriptions)
        self.log.info("Transcription pipeline completed")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SRT labeler pipeline.")
    parser.add_argument(
        "--limit",
        type=positive_int,
        help="Only process this many transcriptions, default unlimited",
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
        debug=args.debug,
    )
    labeler.run()


if __name__ == "__main__":
    main()
