import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from lwe.core.config import Config
from lwe import ApiBackend
from .logger import Logger
from .merger import SrtMerger
from .constants import (
    LWE_DEFAULT_PRESET,
    LWE_FALLBACK_PRESET,
    DEFAULT_LWE_POOL_LIMIT,
    UUID_SHORT_LENGTH,
)


class SrtLabelerPipeline:
    """Pipeline for processing SRT files with AI speaker labeling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        domain: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        """Initialize the SRT labeling pipeline.

        :param api_key: API key for authentication
        :param domain: Domain for API endpoints
        :param debug: Enable debug logging
        :raises ValueError: If api_key or domain are not provided
        """
        if not api_key or not domain:
            raise ValueError("API key and domain must be provided")

        self.log = Logger(self.__class__.__name__, debug=debug)
        self.api_key = api_key
        self.domain = domain
        self.debug = debug

        # Thread-local storage for LWE backends
        self.thread_local = threading.local()

        # Initialize SrtMerger
        self.merger = SrtMerger(debug=debug)

        # Initialize thread pool
        self.executor = ThreadPoolExecutor(
            max_workers=DEFAULT_LWE_POOL_LIMIT,
            thread_name_prefix="LWEWorker",
            initializer=self._initialize_worker,
        )

    def _initialize_lwe_backend(self) -> ApiBackend:
        """Initialize a new LWE backend instance.

        :return: Configured ApiBackend instance
        """
        config_args = {}
        config_dir = os.environ.get("LWE_CONFIG_DIR")
        data_dir = os.environ.get("LWE_DATA_DIR")
        if config_dir:
            config_args["config_dir"] = config_dir
        if data_dir:
            config_args["data_dir"] = data_dir

        config = Config(**config_args)
        config.load_from_file()
        config.set("model.default_preset", LWE_DEFAULT_PRESET)
        if self.debug:
            config.set("debug.log.enabled", True)

        return ApiBackend(config)

    def _initialize_worker(self) -> None:
        """Initialize worker thread with its own LWE backend."""
        self.thread_local.backend = self._initialize_lwe_backend()

    def _extract_transcript_section(self, response: str) -> str:
        """Extract the transcript section from AI response.

        :param response: Full AI response text
        :return: Extracted transcript content
        :raises Exception: If transcript section not found
        """
        import re

        match = re.search(r"<transcript>(.*?)</transcript>", response, re.DOTALL)
        if not match:
            raise Exception("No transcript section found in the text")
        return match.group(1).strip()

    def _generate_identifier(self) -> str:
        """Generate a short unique identifier.

        :return: First 8 characters of a UUIDv4
        """
        return str(uuid.uuid4())[:UUID_SHORT_LENGTH]

    def _prepare_template_vars(self, transcription: Dict) -> Dict:
        """Prepare variables for template rendering.

        :param transcription: Transcription data
        :return: Template variables dictionary
        """
        return {
            "transcription": transcription["content"],
            "identifier": self._generate_identifier(),
        }

    def _get_backup_overrides(self, transcription_id: str) -> Dict:
        """Get override settings for backup preset.

        :param transcription_id: ID of transcription being processed
        :return: Override settings dictionary
        """
        self.log.warning(
            f"Using backup preset for transcription {transcription_id} after failure"
        )
        return {
            "request_overrides": {
                "preset": LWE_FALLBACK_PRESET,
            },
        }

    def _run_ai_analysis(
        self, template_vars: Dict, overrides: Optional[Dict] = None
    ) -> tuple[bool, Optional[str], Optional[str]]:
        """Run AI analysis with template.

        :param template_vars: Variables for template
        :param overrides: Optional override settings
        :return: Tuple of (success, response, error)
        """
        return self.thread_local.backend.run_template(
            "transcription-srt-labeling",
            template_vars=template_vars,
            overrides=overrides,
        )

    def _handle_ai_failure(
        self, error: str, transcription_id: str, attempt: int
    ) -> None:
        """Handle AI analysis failure.

        :param error: Error message from AI
        :param transcription_id: ID of failed transcription
        :param attempt: Current attempt number
        :raises Exception: If max attempts reached
        """
        if attempt >= 1:  # Max attempts is 2
            raise Exception(f"AI model error: {error}")

    def _process_single_attempt(
        self, template_vars: Dict, transcription_id: str, attempt: int
    ) -> Optional[Dict]:
        """Make a single attempt at AI analysis.

        :param template_vars: Template variables
        :param transcription_id: Transcription ID
        :param attempt: Attempt number
        :return: Processing result or None if failed
        """
        overrides = (
            self._get_backup_overrides(transcription_id) if attempt > 0 else None
        )
        success, response, error = self._run_ai_analysis(template_vars, overrides)

        if success:
            return {"id": transcription_id, "labeled_content": response}

        self._handle_ai_failure(error, transcription_id, attempt)
        return None

    def _merge_srt_content(self, original_srt: str, ai_labeled_srt: str) -> str:
        """Merge original SRT with AI-labeled content.

        :param original_srt: Original SRT content
        :param ai_labeled_srt: AI-labeled SRT content
        :return: Merged SRT content
        :raises Exception: If merge fails
        """
        try:
            return self.merger.merge(original_srt, ai_labeled_srt)
        except Exception as e:
            raise Exception(f"Failed to merge SRT content: {str(e)}")

    def _process_transcription(self, transcription: Dict) -> Dict:
        """Process a single transcription using the thread's LWE backend.

        :param transcription: Transcription data to process
        :return: Dictionary with processing results
        :raises Exception: If AI processing fails
        """
        self.log.debug(f"Processing transcription {transcription.get('id', 'unknown')}")
        template_vars = self._prepare_template_vars(transcription)

        for attempt in range(2):
            result = self._process_single_attempt(
                template_vars, transcription["id"], attempt
            )
            if result:
                # Extract transcript section and merge with original
                ai_labeled_content = self._extract_transcript_section(
                    result["labeled_content"]
                )
                merged_content = self._merge_srt_content(
                    transcription["content"], ai_labeled_content
                )
                result["labeled_content"] = merged_content
                return result

        raise Exception(
            "Unexpected error: all attempts failed without raising exception"
        )

    def process_transcriptions(self, transcriptions: List[Dict]) -> None:
        """Process transcriptions using thread pool.

        :param transcriptions: List of transcriptions to process
        """
        # Submit all transcriptions to the thread pool and wait for completion
        futures = [
            self.executor.submit(self._process_transcription, transcription)
            for transcription in transcriptions
        ]

        # Wait for all processing to complete
        for future in futures:
            try:
                future.result()
            except Exception as e:
                self.log.error(f"Error processing transcription: {e}")

    def cleanup(self):
        """Explicitly cleanup resources."""
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()
