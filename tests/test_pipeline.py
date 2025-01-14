import os
import pytest
from unittest.mock import Mock, patch, call
from concurrent.futures import ThreadPoolExecutor
from lwe.core.config import Config
from lwe import ApiBackend
from srt_labeler.pipeline import SrtLabelerPipeline
from srt_labeler.constants import (
    DEFAULT_LWE_POOL_LIMIT,
    UUID_SHORT_LENGTH,
    LWE_FALLBACK_PRESET,
)


@pytest.fixture
def mock_lwe_backend():
    mock_backend = Mock(spec=ApiBackend)
    # Configure run_template to return a tuple (success, response, error)
    mock_backend.run_template.return_value = (True, "test response", None)
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

        with patch.object(pipeline, "_process_transcription"):
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

    def test_process_transcription_with_ai(self, pipeline_args, mock_lwe_setup):
        """Test processing a single transcription with AI model."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Mock the LWE backend response
        mock_response = """
<thinking>Analysis of conversation</thinking>
<transcript>
1
00:00:01,000 --> 00:00:02,000
Operator: Hello world
</transcript>
"""
        mock_lwe_setup["backend"].run_template.return_value = (
            True,
            mock_response,
            None,
        )

        # Test transcription processing
        transcription = {
            "id": 1,
            "content": """1
00:00:01,000 --> 00:00:02,000
Hello world""",
        }
        # srt module adds newlines
        labeled_srt = """1
00:00:01,000 --> 00:00:02,000
Operator: Hello world

"""
        result = pipeline._process_transcription(transcription)
        assert result["labeled_content"] == labeled_srt
        mock_lwe_setup["backend"].run_template.assert_called_once()

    def test_process_transcription_ai_failure(self, pipeline_args, mock_lwe_setup):
        """Test handling AI model failure."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Mock AI failure
        mock_lwe_setup["backend"].run_template.return_value = (False, None, "AI Error")

        # Test error handling
        transcription = {
            "id": 1,
            "content": "1\n00:00:01,000 --> 00:00:02,000\nHello world\n",
        }
        with pytest.raises(Exception, match="AI model error: AI Error"):
            pipeline._process_transcription(transcription)

    def test_process_transcription_backup_preset(self, pipeline_args, mock_lwe_setup):
        """Test fallback to backup preset after failures."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Mock responses for first and second attempts
        mock_lwe_setup["backend"].run_template.side_effect = [
            (False, None, "First Error"),
            (
                True,
                """
<thinking>Analysis of conversation</thinking>
<transcript>
1
00:00:01,000 --> 00:00:02,000
Operator: Hello world
</transcript>""",
                None,
            ),
        ]

        transcription = {
            "id": 1,
            "content": "1\n00:00:01,000 --> 00:00:02,000\nHello world\n",
        }

        # First call should fail and retry with backup preset
        pipeline._process_transcription(transcription)

        # Verify backup preset was used
        assert mock_lwe_setup["backend"].run_template.call_count == 2
        last_call_args = mock_lwe_setup["backend"].run_template.call_args_list[-1]
        assert "preset" in last_call_args[1].get("overrides", {}).get(
            "request_overrides", {}
        )

    def test_extract_transcript_section(self, pipeline_args):
        """Test extraction of transcript section from AI response."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Test valid response
        response = """
<thinking>Analysis</thinking>
<transcript>
1
00:00:01,000 --> 00:00:02,000
Operator: Hello
</transcript>
"""
        result = pipeline._extract_transcript_section(response)
        assert (
            result
            == """1
00:00:01,000 --> 00:00:02,000
Operator: Hello"""
        )

        # Test invalid response
        with pytest.raises(Exception, match="No transcript section found"):
            pipeline._extract_transcript_section("Invalid response")

    def test_prepare_template_vars_basic(self, pipeline_args):
        """Test basic template variable preparation."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {"id": "123", "content": "test content"}
        vars = pipeline._prepare_template_vars(transcription)
        assert vars["transcription"] == "test content"
        assert len(vars["identifier"]) == UUID_SHORT_LENGTH

    def test_prepare_template_vars_missing_fields(self, pipeline_args):
        """Test template vars with missing fields."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        with pytest.raises(KeyError):
            pipeline._prepare_template_vars({"id": "123"})

    def test_prepare_template_vars_extra_fields(self, pipeline_args):
        """Test template vars ignores extra fields."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {"id": "123", "content": "test content", "extra": "ignored"}
        vars = pipeline._prepare_template_vars(transcription)
        assert vars["transcription"] == "test content"
        assert len(vars["identifier"]) == UUID_SHORT_LENGTH

    def test_generate_identifier(self, pipeline_args):
        """Test unique identifier generation."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        id1 = pipeline._generate_identifier()
        id2 = pipeline._generate_identifier()
        assert len(id1) == UUID_SHORT_LENGTH
        assert len(id2) == UUID_SHORT_LENGTH
        assert id1 != id2  # Verify uniqueness

    def test_get_backup_overrides(self, pipeline_args, capsys):
        """Test backup preset override generation."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        overrides = pipeline._get_backup_overrides("123")

        assert overrides == {
            "request_overrides": {
                "preset": LWE_FALLBACK_PRESET,
            }
        }
        captured = capsys.readouterr()
        assert "Using backup preset for transcription 123" in captured.err

    def test_run_ai_analysis_no_overrides(self, pipeline_args, mock_lwe_setup):
        """Test AI analysis without overrides."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        mock_lwe_setup["backend"].run_template.return_value = (True, "response", None)

        success, response, error = pipeline._run_ai_analysis({"test": "vars"})

        mock_lwe_setup["backend"].run_template.assert_called_once_with(
            "transcription-srt-labeling", template_vars={"test": "vars"}, overrides=None
        )
        assert success
        assert response == "response"
        assert error is None

    def test_run_ai_analysis_with_overrides(self, pipeline_args, mock_lwe_setup):
        """Test AI analysis with overrides."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        test_overrides = {"test": "override"}
        mock_lwe_setup["backend"].run_template.return_value = (True, "response", None)

        success, response, error = pipeline._run_ai_analysis(
            {"test": "vars"}, test_overrides
        )

        mock_lwe_setup["backend"].run_template.assert_called_once_with(
            "transcription-srt-labeling",
            template_vars={"test": "vars"},
            overrides=test_overrides,
        )

    def test_handle_ai_failure_first_attempt(self, pipeline_args):
        """Test handling AI failure on first attempt."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        # Should not raise on first attempt
        pipeline._handle_ai_failure("test error", "123", 0)

    def test_handle_ai_failure_second_attempt(self, pipeline_args):
        """Test handling AI failure on second attempt."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        with pytest.raises(
            Exception, match="AI model error: No transcript section found"
        ):
            pipeline._handle_ai_failure("No transcript section found", "123", 1)

    def test_process_single_attempt_success(self, pipeline_args, mock_lwe_setup):
        """Test successful single processing attempt."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        mock_lwe_setup["backend"].run_template.return_value = (True, "success", None)

        result = pipeline._process_single_attempt({"test": "vars"}, "123", 0)

        assert result == {"id": "123", "labeled_content": "success"}
        assert not mock_lwe_setup["backend"].run_template.call_args[1]["overrides"]

    def test_process_single_attempt_failure(self, pipeline_args, mock_lwe_setup):
        """Test failed single processing attempt."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        mock_lwe_setup["backend"].run_template.return_value = (False, None, "error")

        result = pipeline._process_single_attempt({"test": "vars"}, "123", 0)

        assert result is None

    def test_process_single_attempt_with_backup(self, pipeline_args, mock_lwe_setup):
        """Test processing attempt with backup preset."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        mock_lwe_setup["backend"].run_template.return_value = (True, "success", None)

        result = pipeline._process_single_attempt(
            {"test": "vars"}, "123", 1  # Second attempt
        )

        assert result == {"id": "123", "labeled_content": "success"}
        # Verify backup preset was used
        overrides = mock_lwe_setup["backend"].run_template.call_args[1]["overrides"]
        assert overrides["request_overrides"]["preset"] == LWE_FALLBACK_PRESET

    def test_empty_transcription_content(self, pipeline_args, mock_lwe_setup):
        """Test handling of empty transcription content."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        transcription = {"id": "123", "content": ""}

        mock_lwe_setup["backend"].run_template.return_value = (True, "success", None)
        with pytest.raises(Exception, match="No transcript section found in the text"):
            pipeline._process_transcription(transcription)

    def test_malformed_transcription(self, pipeline_args):
        """Test handling of malformed transcription data."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        malformed = {"bad": "data"}
        with pytest.raises(KeyError):
            pipeline._process_transcription(malformed)

    def test_merge_srt_content_success(self, pipeline_args):
        """Test successful merging of original and AI-labeled SRT content."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        original_srt = """1
00:00:01,000 --> 00:00:02,000
Hello world"""

        ai_labeled_srt = """1
00:00:01,000 --> 00:00:02,000
Operator: Hello world"""

        result = pipeline._merge_srt_content(original_srt, ai_labeled_srt)
        # srt module adds newlines.
        assert (
            result
            == """1
00:00:01,000 --> 00:00:02,000
Operator: Hello world

"""
        )

    def test_merge_srt_content_invalid_format(self, pipeline_args):
        """Test handling of invalid SRT format during merge."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        with pytest.raises(Exception) as exc_info:
            pipeline._merge_srt_content("invalid srt", "also invalid")
        assert "Invalid SRT format" in str(exc_info.value)

    def test_merge_srt_content_invalid_labels(self, pipeline_args):
        """Test handling of invalid speaker labels during merge."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        original_srt = """1
00:00:01,000 --> 00:00:02,000
Hello world"""

        ai_labeled_srt = """1
00:00:01,000 --> 00:00:02,000
invalid: Hello world"""

        with pytest.raises(Exception) as exc_info:
            pipeline._merge_srt_content(original_srt, ai_labeled_srt)
        assert "Invalid label" in str(exc_info.value)

    def test_thread_safety(self, pipeline_args):
        """Test thread-safety of backend access."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        with patch("srt_labeler.pipeline.ApiBackend") as mock_backend_class:
            # Configure the mock to return a new mock instance each time
            mock_backend_class.side_effect = lambda config: Mock(spec=ApiBackend)

            # Simulate multiple thread initializations
            pipeline._initialize_worker()
            first_backend = pipeline.thread_local.backend

            pipeline._initialize_worker()
            second_backend = pipeline.thread_local.backend

            # Each initialization should create a new backend instance
            assert first_backend is not second_backend

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
