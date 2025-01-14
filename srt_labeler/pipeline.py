import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict
from lwe.core.config import Config
from lwe import ApiBackend
from .logger import Logger
from .constants import (
    LWE_DEFAULT_PRESET,
    DEFAULT_LWE_POOL_LIMIT,
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

    def _process_transcription(self, transcription: Dict) -> None:
        """Process a single transcription using the thread's LWE backend.

        TODO: Implement full transcription processing:
        1. Extract SRT content from transcription
        2. Use LWE backend to get AI-labeled version
        3. Use SrtMerger to merge labels
        4. Update transcription via API

        :param transcription: Transcription data to process
        """
        self.log.debug(f"Processing transcription {transcription.get('id', 'unknown')}")
        # Access the thread's backend directly
        backend = self.thread_local.backend
        # Dummy implementation for testing
        pass

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
