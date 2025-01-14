import os
import pytest
from unittest.mock import Mock, patch, call
from concurrent.futures import ThreadPoolExecutor
from lwe.core.config import Config
from lwe import ApiBackend
from srt_labeler.pipeline import SrtLabelerPipeline
from srt_labeler.constants import (
    DEFAULT_LWE_POOL_LIMIT,
)


@pytest.fixture
def mock_lwe_backend():
    mock_backend = Mock(spec=ApiBackend)
    return mock_backend


@pytest.fixture
def mock_config():
    mock_conf = Mock(spec=Config)
    return mock_conf


@pytest.fixture
def mock_lwe_setup(mock_config, mock_lwe_backend):
    """Fixture to handle all LWE mocking needs."""
    with (
        patch(
            "srt_labeler.pipeline.Config", return_value=mock_config
        ) as mock_config_class,
        patch(
            "srt_labeler.pipeline.ApiBackend", return_value=mock_lwe_backend
        ) as mock_api_backend,
    ):
        yield {
            "config": mock_config,
            "config_class": mock_config_class,
            "backend": mock_lwe_backend,
            "backend_class": mock_api_backend,
        }


class TestSrtLabelerPipeline:
    @pytest.fixture
    def pipeline_args(self):
        return {"api_key": "test_api_key", "domain": "test_domain", "debug": False}

    def test_pipeline_initialization(self, pipeline_args):
        pipeline = SrtLabelerPipeline(**pipeline_args)
        assert pipeline.api_key == "test_api_key"
        assert pipeline.domain == "test_domain"
        assert pipeline.debug is False
        assert pipeline.log is not None
        assert pipeline.executor._max_workers == DEFAULT_LWE_POOL_LIMIT
        assert pipeline.executor._thread_name_prefix == "LWEWorker"

    def test_pipeline_initialization_with_debug(self, pipeline_args):
        pipeline_args["debug"] = True
        pipeline = SrtLabelerPipeline(**pipeline_args)
        assert pipeline.debug is True

    def test_pipeline_initialization_missing_credentials(self):
        with pytest.raises(ValueError):
            SrtLabelerPipeline()

    def test_process_transcriptions(self, pipeline_args, mock_lwe_setup):
        pipeline = SrtLabelerPipeline(**pipeline_args)

        test_transcriptions = [
            {"id": 1, "content": "test1"},
            {"id": 2, "content": "test2"},
        ]

        # Mock _process_transcription to track calls
        with patch.object(pipeline, "_process_transcription") as mock_process:
            # Test parallel processing
            pipeline.process_transcriptions(test_transcriptions)

            # Verify each transcription was processed
            assert mock_process.call_count == len(test_transcriptions)
            # Verify each transcription was passed to _process_transcription
            mock_process.assert_has_calls(
                [
                    call({"id": 1, "content": "test1"}),
                    call({"id": 2, "content": "test2"}),
                ],
                any_order=True,
            )  # any_order=True because of parallel processing

    def test_process_transcriptions_error_handling(
        self, pipeline_args, mock_lwe_setup, capsys
    ):
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Mock the _process_transcription method to raise an exception
        with patch.object(
            pipeline, "_process_transcription", side_effect=Exception("Test error")
        ):
            test_transcriptions = [{"id": 1, "content": "test1"}]

            # Verify error is logged but doesn't crash pipeline
            pipeline.process_transcriptions(test_transcriptions)
            captured = capsys.readouterr()
            assert "Error processing transcription: Test error" in captured.err

    def test_process_transcriptions_multiple_errors(
        self, pipeline_args, mock_lwe_setup, capsys
    ):
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Mock to raise different exceptions for different transcriptions
        def side_effect(transcription):
            if transcription["id"] == 1:
                raise ValueError("Value error")
            if transcription["id"] == 2:
                raise RuntimeError("Runtime error")

        with patch.object(pipeline, "_process_transcription", side_effect=side_effect):
            test_transcriptions = [
                {"id": 1, "content": "test1"},
                {"id": 2, "content": "test2"},
                {"id": 3, "content": "test3"},
            ]

            # Should process all transcriptions despite errors
            pipeline.process_transcriptions(test_transcriptions)

            # Verify both errors were logged
            captured = capsys.readouterr()
            assert "Error processing transcription: Value error" in captured.err
            assert "Error processing transcription: Runtime error" in captured.err

    def test_thread_pool_cleanup(self, pipeline_args, mock_lwe_setup):
        """Test that thread pool is properly shut down when cleanup is called."""
        mock_executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "srt_labeler.pipeline.ThreadPoolExecutor", return_value=mock_executor
        ):
            pipeline = SrtLabelerPipeline(**pipeline_args)
            pipeline.cleanup()
            mock_executor.shutdown.assert_called_once_with(wait=True)

    def test_thread_pool_cleanup_context(self, pipeline_args, mock_lwe_setup):
        """Test that thread pool is properly shut down when used as context manager."""
        mock_executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "srt_labeler.pipeline.ThreadPoolExecutor", return_value=mock_executor
        ):
            with SrtLabelerPipeline(**pipeline_args):
                pass
            mock_executor.shutdown.assert_called_once_with(wait=True)

    def test_backend_reuse_within_thread(self, pipeline_args, mock_lwe_setup):
        """Test that backends are reused by verifying only pool_size backends are created."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Create more transcriptions than worker threads
        test_transcriptions = [
            {"id": i, "content": f"test{i}"}
            for i in range(DEFAULT_LWE_POOL_LIMIT * 2)  # Double the pool size
        ]

        with patch.object(pipeline, "_process_transcription") as mock_process:
            pipeline.process_transcriptions(test_transcriptions)

            # Verify backend was only initialized pool_size times
            assert mock_lwe_setup["backend_class"].call_count == DEFAULT_LWE_POOL_LIMIT

    def test_backend_initialization_with_env_vars(self, pipeline_args, mock_lwe_setup):
        """Test that backend initialization uses environment variables correctly."""
        with patch.dict(
            os.environ, {"LWE_CONFIG_DIR": "/test/config", "LWE_DATA_DIR": "/test/data"}
        ):
            pipeline = SrtLabelerPipeline(**pipeline_args)
            pipeline._initialize_worker()

            # Verify Config was initialized with correct paths
            mock_lwe_setup["config_class"].assert_called_once_with(
                config_dir="/test/config", data_dir="/test/data"
            )

    def test_backend_debug_mode_configuration(self, pipeline_args, mock_lwe_setup):
        """Test that debug mode is properly configured in the backend."""
        pipeline_args["debug"] = True
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Verify debug logging was enabled in config
        mock_lwe_setup["config"].set.assert_any_call("debug.log.enabled", True)

    def test_process_transcriptions_empty_list(self, pipeline_args, mock_lwe_setup):
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Should handle empty list gracefully
        pipeline.process_transcriptions([])

        # Verify no processing was attempted
        with patch.object(pipeline, "_process_transcription") as mock_process:
            pipeline.process_transcriptions([])
            mock_process.assert_not_called()

    def test_worker_initialization(self, pipeline_args, mock_lwe_setup):
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Call initializer directly to test backend creation
        pipeline._initialize_worker()

        # Verify backend was created and stored in thread local storage
        assert hasattr(pipeline.thread_local, "backend")
        assert pipeline.thread_local.backend == mock_lwe_setup["backend"]
