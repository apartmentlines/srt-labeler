import os
import re
import tempfile
import logging
import json
import copy
from .constants import DEFAULT_STATS_DB
from .stats import StatsDatabase
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from types import TracebackType
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain_core.messages import HumanMessage
import requests
from .utils import post_request
from lwe.core.config import Config
from lwe import ApiBackend
from .logger import Logger
from .merger import SrtMerger, SrtMergeError
from .constants import (
    LWE_DEFAULT_PRESET,
    LWE_FALLBACK_PRESET,
    LWE_TRANSCRIPTION_TEMPLATE,
    DEFAULT_RETRY_ATTEMPTS,
    DOWNLOAD_TIMEOUT,
    DEFAULT_LWE_POOL_LIMIT,
    UUID_SHORT_LENGTH,
    TRANSCRIPTION_STATE_COMPLETE,
)


@dataclass
class TranscriptionResult:
    """Represents the result of a transcription processing attempt."""

    transcription_id: int
    success: bool  # Indicates if labeling succeeded, not API success
    transcription: str | None = None
    error: Exception | None = None
    ai_content: str | None = None  # Raw AI-generated SRT before merge

    @property
    def requires_api_update(self) -> bool:
        """Determine if this result should trigger an API update.

        :return: True if result requires API update (success or double hard error)
        """
        if self.success:
            return True
        return self.is_hard_error()

    def is_hard_error(self) -> bool:
        """Check if the error is classified as hard.

        :return: True if error is a hard error type
        """
        if not self.error:
            return False
        return isinstance(
            self.error,
            (SrtMergeError, ModelResponseFormattingError, RequestFileNotFoundError),
        )


class FallbackResultValidator:
    """Validates if a fallback model result should be accepted in full."""

    def __init__(self, merger: SrtMerger) -> None:
        """Initialize the validator.

        :param merger: SRT merger instance
        """
        self.merger: SrtMerger = merger

    def should_accept_fallback(self, error: Exception | None, ai_content: str | None) -> bool:
        """Determine if fallback model output should be accepted.

        :param error: The error from merge attempt
        :param ai_content: Raw AI-generated SRT content
        :return: True if fallback content should be accepted
        """
        return (
            isinstance(error, SrtMergeError)
            and self.merger.validate_srt_labels(ai_content)
        )


class TranscriptionErrorHandler:
    """Handles error classification and processing decisions."""

    def is_hard_error(self, error: Exception) -> bool:
        """Classify if an error is 'hard' or 'transient'.

        :param error: Exception to classify
        :return: True if error is considered a hard error
        """
        return isinstance(
            error,
            (SrtMergeError, ModelResponseFormattingError, RequestFileNotFoundError),
        )

    def should_update_with_error(
        self, primary_error: Exception | None, fallback_error: Exception | None
    ) -> bool:
        """Determine if we should send error state to API.

        :param primary_error: Error from primary model attempt
        :param fallback_error: Error from fallback model attempt
        :return: True if both errors are hard errors
        """
        if primary_error is None or fallback_error is None:
            return False
        return self.is_hard_error(primary_error) and self.is_hard_error(fallback_error)

    def create_error_result(
        self, transcription_id: int, error: Exception
    ) -> TranscriptionResult:
        """Create a TranscriptionResult for an error case.

        :param transcription_id: ID of the transcription
        :param error: The error that occurred
        :return: TranscriptionResult representing the error
        """
        return TranscriptionResult(
            transcription_id=transcription_id, success=False, error=error
        )


class ApiPayloadBuilder:
    """Builds API payloads from TranscriptionResults."""

    def __init__(self, api_key: str):
        """Initialize the payload builder.

        :param api_key: API key for authentication
        """
        self.api_key: str = api_key

    def build_payload(self, result: TranscriptionResult) -> dict[str, Any]:
        """Convert TranscriptionResult to API payload format.

        Always sets success=True in payload as it indicates
        successful completion of processing, not success/failure
        of labeling.

        :param result: TranscriptionResult to convert
        :return: Dictionary formatted for API submission
        """
        base_payload = {
            "api_key": self.api_key,
            "id": result.transcription_id,
            "success": True,  # Always True for API updates
            "transcription_state": TRANSCRIPTION_STATE_COMPLETE,
        }

        if result.success:
            return {**base_payload, "transcription": result.transcription}

        metadata = {
            "error_stage": "labeling",
            "error": str(result.error),
        }
        return {**base_payload, "metadata": json.dumps(metadata)}


class BaseError(Exception):
    """Base exception class."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize API error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message)
        if cause:
            self.__cause__ = cause


class AuthenticationError(BaseError):
    """Exception raised for authentication-related errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize authentication error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message, cause)


class RequestError(BaseError):
    """Exception raised for errors during request construction or execution."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize request error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message, cause)


class ResponseValidationError(BaseError):
    """Exception raised for response validation errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize response validation error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message, cause)


class ModelResponseFormattingError(BaseError):
    """Exception raised for model response formatting errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize model response formatting error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message, cause)


class RequestFileNotFoundError(BaseError):
    """Exception raised for HTTP 404 errors."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize file not found error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message, cause)


class SrtLabelerPipeline:
    """Pipeline for processing SRT files with AI speaker labeling."""

    def __init__(
        self,
        api_key: str | None = None,
        file_api_key: str | None = None,
        domain: str | None = None,
        max_workers: int | None = None,
        stats_db: str = DEFAULT_STATS_DB,
        debug: bool = False,
    ) -> None:
        """Initialize the SRT labeling pipeline.

        :param api_key: API key for authentication
        :param file_api_key: API key for file download
        :param domain: Domain for API endpoints
        :param max_workers: Maximum number of threads
        :param stats_db: Path to SQLite stats database
        :param debug: Enable debug logging
        :raises ValueError: If api_key or domain are not provided
        """
        if not api_key or not file_api_key or not domain:
            raise ValueError("API key, file API key, and domain must be provided")

        self.log: logging.Logger = Logger(self.__class__.__name__, debug=debug)
        self.log.debug(f"Initializing pipeline with domain: {domain}, debug: {debug}")
        self.api_key: str | None = api_key
        self.file_api_key: str | None = file_api_key
        self.domain: str | None = domain
        self.max_workers: int = max_workers or DEFAULT_LWE_POOL_LIMIT
        self.debug: bool = debug
        # Initialize stats database
        self.stats_db_path: str = stats_db
        self.stats: StatsDatabase = StatsDatabase(stats_db, debug=debug)
        self.log.info(f"SRT Labeler pipeline initialized with max_workers: {self.max_workers}")

        # Initialize handlers
        self.error_handler: TranscriptionErrorHandler = TranscriptionErrorHandler()
        self.payload_builder: ApiPayloadBuilder = ApiPayloadBuilder(api_key)

        # Thread-local storage for LWE backends
        self.thread_local: threading.local = threading.local()

        # Initialize SrtMerger
        self.merger: SrtMerger = SrtMerger(debug=debug)

        # Initialize thread pool
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(
            max_workers=self.max_workers,
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

        self.log.debug(f"Initializing LWE backend with config args: {config_args}")
        config = Config(**config_args)
        config.load_from_file()
        config.set("model.default_preset", LWE_DEFAULT_PRESET)
        if self.debug:
            config.set("debug.log.enabled", True)
            config.set("debug.log.filepath", "%s/%s" % (tempfile.gettempdir(), "lwe-srt-labeler-debug.log"))

        backend = ApiBackend(config)
        self.log.debug("LWE backend initialization complete")
        return backend

    def _initialize_worker(self) -> None:
        """Initialize worker thread with its own LWE backend."""
        self.thread_local.backend = self._initialize_lwe_backend()

    def _extract_transcript_section(self, response: str | None) -> str:
        """Extract the transcript section from AI response.

        :param response: Full AI response text
        :return: Extracted transcript content
        :raises Exception: If transcript section not found
        """
        self.log.debug("Attempting to extract transcript section from response")
        if not response:
            raise ModelResponseFormattingError("No response provided")
        match = re.search(r"<transcript>(.*?)</transcript>", response, re.DOTALL)
        if not match:
            raise ModelResponseFormattingError(
                "No transcript section found in the text"
            )
        extracted = match.group(1).strip()
        self.log.debug("Successfully extracted transcript section")
        return extracted

    def _generate_identifier(self) -> str:
        """Generate a short unique identifier.

        :return: First 8 characters of a UUIDv4
        """
        return str(uuid.uuid4())[:UUID_SHORT_LENGTH]

    def _prepare_template_vars(self, transcription: dict[str, Any]) -> dict[str, Any]:
        """Prepare variables for template rendering.

        :param transcription: Transcription data
        :return: Template variables dictionary
        """
        self.log.debug(
            f"Preparing template variables for transcription {transcription['id']}"
        )
        template_vars = {
            "transcription": transcription["transcription"],
            "identifier": self._generate_identifier(),
        }
        self.log.debug(
            f"Template variables prepared with identifier: {template_vars['identifier']}"
        )
        return template_vars

    def _check_download_errors_http(
        self, response: requests.Response | None, error: Exception
    ) -> None:
        if response is not None:
            if response.status_code == 404:
                raise RequestFileNotFoundError(
                    f"Error downloading audio file: {response.status_code}"
                )
        raise error

    def _check_download_errors_api(self, response: requests.Response | None) -> None:
        if response:
            content_type = response.headers.get("content-type", "")
            if "json" in content_type.lower():
                error_data = response.json()
                message = f"Error downloading audio file: {error_data}"
                self.log.error(message)
                raise RequestError(message)

    @retry(
        stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def _download_file(self, transcription: dict[str, Any]) -> requests.Response | None:
        response = None
        try:
            url = transcription["url"]
            self.log.debug(f"Downloading {url}")
            params = {"api_key": self.file_api_key}
            response = requests.get(url, params=params, timeout=DOWNLOAD_TIMEOUT)
            response.raise_for_status()
            self.log.debug(f"Downloaded {url}")
            return response
        except Exception as e:
            self.log.warning(
                f"Error downloading {transcription['url']}: {e}. Retrying..."
            )
            self._check_download_errors_http(response, e)

    def _try_download_file(self, transcription: dict[str, Any]) -> bytes:
        response = self._download_file(transcription)
        self._check_download_errors_api(response)
        if response and response.content:
            return response.content
        raise RequestError("Received empty response")

    def _add_audio_file(self, transcription: dict[str, Any]) -> HumanMessage:
        audio_bytes = self._try_download_file(transcription)
        file = HumanMessage(
            content=[
                {"type": "media", "mime_type": "audio/wav", "data": audio_bytes},
            ]
        )
        return file

    def _get_request_overrides(self, transcription: dict[str, Any], fallback: bool) -> dict[str, Any]:
        """Get override settings for backup preset.

        :param transcription_id: ID of transcription being processed
        :return: Override settings dictionary
        """
        overrides: dict[str, dict[str, list[HumanMessage] | str]] = {
            "request_overrides": {
                "files": [self._add_audio_file(transcription)],
            },
        }
        if fallback:
            self.log.warning(
                f"Using backup preset for transcription {transcription['id']} after failure"
            )
            overrides["request_overrides"]["preset"] = LWE_FALLBACK_PRESET
            self.log.debug(f"Applied backup override: {LWE_FALLBACK_PRESET}")
        return overrides

    def _run_ai_analysis(
        self, template_vars: dict[str, Any], overrides: dict[str, Any] | None = None
    ) -> tuple[bool, str | None, str | None]:
        """Run AI analysis with template.

        :param template_vars: Variables for template
        :param overrides: Optional override settings
        :return: Tuple of (success, response, error)
        """
        self.log.debug("Running AI analysis with template")
        result = self.thread_local.backend.run_template(
            LWE_TRANSCRIPTION_TEMPLATE,
            template_vars=template_vars,
            overrides=overrides,
        )
        self.log.debug(f"AI analysis completed with success: {result[0]}")
        return result

    def _merge_srt_content(self, original_srt: str, ai_labeled_srt: str) -> str:
        """Merge original SRT with AI-labeled content.

        :param original_srt: Original SRT content
        :param ai_labeled_srt: AI-labeled SRT content
        :return: Merged SRT content
        :raises Exception: If merge fails
        """
        self.log.debug("Attempting to merge SRT content")
        merged = self.merger.merge(original_srt, ai_labeled_srt)
        self.log.debug("Successfully merged SRT content")
        return merged

    def _process_transcription(self, transcription: dict[str, Any]) -> None:
        """Process a single transcription using the thread's LWE backend.

        :param transcription: Transcription data to process
        """
        self.log.debug(f"Processing transcription {transcription['id']}")

        result = self._process_with_error_handling(transcription)

        if result.requires_api_update:
            self.update_transcription(result)

    def process_transcriptions(self, transcriptions: list[dict[str, Any]] | None) -> None:
        """Process transcriptions using thread pool.

        :param transcriptions: List of transcriptions to process
        """
        if not transcriptions:
            self.log.info("No transcriptions to process")
            return
        self.log.info(f"Starting processing of {len(transcriptions)} transcriptions")
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
        self.log.info("Completed processing all transcriptions")

    def cleanup(self):
        """Explicitly cleanup resources."""
        self.log.debug("Starting pipeline cleanup")
        if hasattr(self, "executor"):
            self.executor.shutdown(wait=True)
        if hasattr(self, "stats"):
            self.stats.close()
        self.log.debug("Pipeline cleanup completed")

    def build_update_url(self) -> str:
        """Build the URL for the transcription update endpoint.

        :return: Complete URL for the update endpoint
        """
        return f"https://{self.domain}/al/transcriptions/update/operator-recording"

    def _handle_update_response(self, response: requests.Response) -> None:
        """Handle the API update response.

        :param response: Response from the API
        :raises AuthenticationError: If authentication fails
        :raises ResponseValidationError: If response is invalid or indicates an error
        """
        self.log.debug("Processing API update response")
        try:
            data = response.json()
            self.log.debug(f"Received response from API: {data}")
        except ValueError as e:
            raise ResponseValidationError("Invalid JSON response from API", e)

        if "success" not in data:
            raise ResponseValidationError("Missing success field in API response")

        if not data["success"]:
            error_msg = data.get("message", "Unknown error")
            if "Invalid or missing API key" in error_msg:
                raise AuthenticationError(f"Authentication failed: {error_msg}")
            raise ResponseValidationError(f"Response validation failed: {error_msg}")

    def _execute_update_request(self, url: str, payload: dict[str, Any]) -> requests.Response:
        """Execute the update request to the API.

        :param url: API endpoint URL
        :param payload: Request payload
        :return: API response
        :raises RequestError: If the request fails
        """
        self.log.debug(f"Executing update request to {url}")
        try:
            response = post_request(url, payload)
            self.log.debug("Received response from API")
            return response
        except requests.RequestException as e:
            raise RequestError(f"Failed to execute update request: {str(e)}", e)

    def update_transcription(self, result: TranscriptionResult) -> None:
        """Update a transcription with results.

        :param result: TranscriptionResult to send to API
        :raises RequestError: If the update request fails
        :raises AuthenticationError: If authentication fails
        :raises ResponseValidationError: If response validation fails
        """
        self.log.info(f"Updating transcription {result.transcription_id}")
        url = self.build_update_url()
        payload = self.payload_builder.build_payload(result)
        debug_payload = copy.deepcopy(payload)
        debug_payload["api_key"] = "[REDACTED]"
        self.log.debug(f"Updating transcription payload: {debug_payload}")
        response = self._execute_update_request(url, payload)
        self._handle_update_response(response)

    def __enter__(self):
        """Context manager entry."""
        return self

    def _increment_stats(self, fallback: bool) -> None:
        """Increment stats counters.

        :param fallback: Whether this was a fallback model attempt
        """
        if fallback:
            self.stats.increment_fallback()
        else:
            self.stats.increment_primary()

    def __exit__(self, exc_type: type, exc_val: Exception, exc_tb: TracebackType) -> None:
        """Context manager exit."""
        del exc_type, exc_val, exc_tb
        self.cleanup()

    def _attempt_model_processing(
        self, transcription: dict[str, Any], use_fallback: bool = False
    ) -> TranscriptionResult:
        """Process with specific model.

        :param transcription: Transcription data to process
        :param use_fallback: Whether to use fallback model
        :return: TranscriptionResult indicating success or failure
        """
        self.log.debug(f"Attempting model processing with fallback={use_fallback}")
        try:
            result = self._execute_model_analysis(transcription, use_fallback)
            self.log.debug(
                f"Model processing attempt completed with success={result.success}"
            )
            self._increment_stats(use_fallback)
            return result
        except Exception as e:
            return TranscriptionResult(
                transcription_id=transcription["id"], success=False, error=e
            )

    def _execute_model_analysis(
        self, transcription: dict[str, Any], use_fallback: bool
    ) -> TranscriptionResult:
        """Execute the AI model analysis and process results.

        :param transcription: Transcription data to process
        :param use_fallback: Whether to use fallback model
        :return: TranscriptionResult from the analysis
        """
        self.log.debug("Starting model analysis execution")
        success, response, error = self._run_model_with_template(
            transcription, use_fallback
        )
        self.log.debug(f"Model analysis execution completed with success={success}")

        if not success:
            return self._create_model_failure_result(transcription["id"], error)

        return self._process_model_response(
            transcription["id"], transcription["transcription"], response, use_fallback
        )

    def _run_model_with_template(
        self, transcription: dict[str, Any], use_fallback: bool
    ) -> tuple[bool, str | None, str | None]:
        """Run the AI model with appropriate template and settings.

        :param transcription: Transcription data to process
        :param use_fallback: Whether to use fallback model
        :return: Tuple of (success, response, error)
        """
        self.log.debug(f"Executing model template: {LWE_TRANSCRIPTION_TEMPLATE}")
        template_vars = self._prepare_template_vars(transcription)
        overrides = self._get_request_overrides(transcription, use_fallback)
        result = self._run_ai_analysis(template_vars, overrides)
        self.log.debug(f"Template execution completed with success={result[0]}")
        return result

    def _create_model_failure_result(
        self, transcription_id: int, error: str | None
    ) -> TranscriptionResult:
        """Create a failure result for model execution failure.

        :param transcription_id: ID of the transcription
        :param error: Error message from model execution
        :return: TranscriptionResult indicating failure
        """
        return TranscriptionResult(
            transcription_id=transcription_id,
            success=False,
            error=Exception(error if error else "AI analysis failed"),
        )

    def _process_model_response(
        self,
        transcription_id: int,
        original_content: str,
        model_response: str | None,
        use_fallback: bool,
    ) -> TranscriptionResult:
        """Process successful model response.

        :param transcription_id: ID of the transcription
        :param original_content: Original SRT content
        :param model_response: Response from the AI model
        :param use_fallback: Whether to use fallback model
        :return: TranscriptionResult with processed content or error
        """
        self.log.debug("Processing model response")
        ai_labeled_content = None
        try:
            ai_labeled_content = self._extract_transcript_section(model_response)
            merged_content = self._merge_srt_content(
                original_content, ai_labeled_content
            )
            self.log.debug("Successfully processed model response")
            return TranscriptionResult(
                transcription_id=transcription_id,
                success=True,
                transcription=merged_content,
            )
        except Exception as e:
            self.log.error(f"Error processing model response: {e}")
            self.stats.log_model_response_error(
                transcription_id, str(e), original_content, model_response, use_fallback
            )
            return TranscriptionResult(
                transcription_id=transcription_id,
                success=False,
                error=e,
                ai_content=ai_labeled_content,
            )

    def _process_with_error_handling(self, transcription: dict[str, Any]) -> TranscriptionResult:
        """Handle both primary and fallback attempts.

        :param transcription: Transcription data to process
        :return: Final TranscriptionResult
        """
        self.log.debug("Starting primary model attempt")
        # Try primary model
        primary_result = self._attempt_model_processing(
            transcription, use_fallback=False
        )
        if primary_result.success:
            return primary_result

        self.log.debug("Starting fallback model attempt")
        # Try fallback model
        fallback_result = self._attempt_model_processing(
            transcription, use_fallback=True
        )
        if fallback_result.success:
            return fallback_result

        return self._determine_final_error_result(
            transcription["id"], primary_result, fallback_result
        )

    def _determine_final_error_result(
        self,
        transcription_id: int,
        primary_result: TranscriptionResult,
        fallback_result: TranscriptionResult,
    ) -> TranscriptionResult:
        """Determine final error result when both attempts fail.

        :param transcription_id: ID of the transcription
        :param primary_result: Result from primary model attempt
        :param fallback_result: Result from fallback model attempt
        :return: TranscriptionResult with appropriate error handling
        """
        self.log.debug("Both attempts failed, determining final result")

        if self.error_handler.should_update_with_error(primary_result.error, fallback_result.error):
            return self._handle_hard_error_case(transcription_id, fallback_result)

        return self._handle_transient_error_case(transcription_id, primary_result, fallback_result)

    def _handle_hard_error_case(
        self,
        transcription_id: int,
        fallback_result: TranscriptionResult,
    ) -> TranscriptionResult:
        """Handle case where both errors are hard errors.

        :param transcription_id: ID of the transcription
        :param fallback_result: Result from fallback model attempt
        :return: TranscriptionResult with validated content or error
        """
        self.log.warning("Both labeling errors hard, checking fallback content")
        validator = FallbackResultValidator(self.merger)

        if validator.should_accept_fallback(fallback_result.error, fallback_result.ai_content):
            self.log.info("Fallback content validated, using raw AI output")
            return TranscriptionResult(
                transcription_id=transcription_id,
                success=True,
                transcription=fallback_result.ai_content
            )

        self.log.warning("Both labeling errors hard, sending error state to API")
        self.stats.increment_hard_failure()
        return fallback_result

    def _handle_transient_error_case(
        self,
        transcription_id: int,
        primary_result: TranscriptionResult,
        fallback_result: TranscriptionResult,
    ) -> TranscriptionResult:
        """Handle case where at least one error is transient.

        :param transcription_id: ID of the transcription
        :param primary_result: Result from primary model attempt
        :param fallback_result: Result from fallback model attempt
        :return: TranscriptionResult with transient error
        """
        transient_error = (
            primary_result.error
            if not primary_result.is_hard_error()
            else fallback_result.error
        )
        return TranscriptionResult(
            transcription_id=transcription_id,
            success=False,
            error=transient_error,
        )

    def _merge_labeled_content(
        self, original_content: str, labeled_content: str, transcription_id: int
    ) -> TranscriptionResult:
        """Merge content with error handling.

        :param original_content: Original SRT content
        :param labeled_content: AI-labeled content
        :param transcription_id: ID of the transcription
        :return: TranscriptionResult with merged content or error
        """
        try:
            merged_content = self.merger.merge(original_content, labeled_content)
            return TranscriptionResult(
                transcription_id=transcription_id,
                success=True,
                transcription=merged_content,
            )
        except Exception as e:
            return TranscriptionResult(
                transcription_id=transcription_id, success=False, error=e
            )
