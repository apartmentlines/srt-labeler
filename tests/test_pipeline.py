import os
import pytest
import requests
from unittest.mock import Mock, patch, call
from concurrent.futures import ThreadPoolExecutor
from lwe.core.config import Config
from lwe import ApiBackend
from srt_labeler.pipeline import (
    SrtLabelerPipeline,
    BaseError,
    RequestError,
    AuthenticationError,
    ResponseValidationError,
    ModelResponseFormattingError,
    TranscriptionResult,
    TranscriptionErrorHandler,
    ApiPayloadBuilder,
)
from srt_labeler.merger import SrtMergeError
from srt_labeler.constants import (
    DEFAULT_LWE_POOL_LIMIT,
    UUID_SHORT_LENGTH,
    LWE_FALLBACK_PRESET,
    TRANSCRIPTION_STATE_COMPLETE,
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
        """Test processing multiple transcriptions."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        test_transcriptions = [
            {"id": 1, "transcription": "test1"},
            {"id": 2, "transcription": "test2"},
            {"id": 3, "transcription": "test3"},
        ]

        with patch.object(pipeline, "_process_with_error_handling") as mock_process:
            # Mock mixed results including thread error
            mock_process.side_effect = [
                TranscriptionResult(
                    transcription_id=1, success=True, transcription="labeled1"
                ),
                Exception("Thread processing error"),  # Simulate thread error
                TranscriptionResult(
                    transcription_id=3, success=True, transcription="labeled3"
                ),
            ]

            with patch.object(pipeline, "update_transcription") as mock_update:
                # Capture the futures for verification
                futures = []
                with patch.object(
                    pipeline.executor,
                    "submit",
                    side_effect=lambda fn, *args: futures.append(
                        Mock(result=lambda: fn(*args))
                    ),
                ) as mock_submit:
                    pipeline.process_transcriptions(test_transcriptions)

                    # Verify submissions
                    assert mock_submit.call_count == len(test_transcriptions)

                    # Verify futures were properly handled
                    assert len(futures) == len(test_transcriptions)

                    # Verify error propagation
                    error_logged = False
                    for future in futures:
                        try:
                            future.result()
                        except Exception as e:
                            error_logged = True
                            assert "Thread processing error" in str(e)
                    assert error_logged

                # Verify successful results were updated
                assert mock_update.call_count == 2
                update_calls = [call[0][0] for call in mock_update.call_args_list]
                assert all(result.success for result in update_calls)
                assert "labeled1" in update_calls[0].transcription
                assert "labeled3" in update_calls[1].transcription

    def test_process_transcriptions_error_handling(
        self, pipeline_args, mock_lwe_setup, capsys
    ):
        """Test error handling during transcription processing."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        with patch.object(pipeline, "_process_with_error_handling") as mock_process:
            mock_process.side_effect = [
                TranscriptionResult(
                    transcription_id=1,
                    success=False,
                    error=Exception("Transient error"),
                ),
                TranscriptionResult(
                    transcription_id=2,
                    success=False,
                    error=ModelResponseFormattingError("Hard error"),
                ),
            ]

            with patch.object(pipeline, "update_transcription") as mock_update:
                test_transcriptions = [
                    {"id": 1, "transcription": "test1"},
                    {"id": 2, "transcription": "test2"},
                ]
                pipeline.process_transcriptions(test_transcriptions)

                # Verify only hard error triggered API update
                assert mock_update.call_count == 1
                result = mock_update.call_args[0][0]
                assert isinstance(result.error, ModelResponseFormattingError)
                assert "Hard error" in str(result.error)

    def test_process_transcriptions_multiple_errors(
        self, pipeline_args, mock_lwe_setup, capsys
    ):
        """Test handling of multiple error types."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        def side_effect(transcription):
            if transcription["id"] == 1:
                return TranscriptionResult(
                    transcription_id=1,
                    success=False,
                    error=SrtMergeError("Hard error"),  # Hard error
                )
            if transcription["id"] == 2:
                return TranscriptionResult(
                    transcription_id=2,
                    success=False,
                    error=Exception("Transient error"),  # Transient error
                )
            return TranscriptionResult(
                transcription_id=3, success=True, transcription="Success"
            )

        with patch.object(
            pipeline, "_process_with_error_handling", side_effect=side_effect
        ):
            with patch.object(pipeline, "update_transcription") as mock_update:
                test_transcriptions = [
                    {"id": 1, "transcription": "test1"},
                    {"id": 2, "transcription": "test2"},
                    {"id": 3, "transcription": "test3"},
                ]

                pipeline.process_transcriptions(test_transcriptions)

                # Verify only success and hard error cases triggered API update
                assert mock_update.call_count == 2
                update_calls = [call[0][0] for call in mock_update.call_args_list]
                assert any(
                    isinstance(r.error, SrtMergeError)
                    for r in update_calls
                    if not r.success
                )
                assert any(
                    r.success and r.transcription == "Success" for r in update_calls
                )

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
            {"id": i, "transcription": f"test{i}"}
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

        transcription = {
            "id": 1,
            "transcription": """1
00:00:01,000 --> 00:00:02,000
Hello world""",
        }

        with patch.object(pipeline, "update_transcription") as mock_update:
            pipeline._process_transcription(transcription)

            # Verify update was called with successful result
            mock_update.assert_called_once()
            result = mock_update.call_args[0][0]
            assert isinstance(result, TranscriptionResult)
            assert result.success is True
            assert result.transcription_id == 1
            assert "Operator: Hello world" in result.transcription

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
            "transcription": "1\n00:00:01,000 --> 00:00:02,000\nHello world\n",
        }

        with patch.object(pipeline, "update_transcription") as mock_update:
            pipeline._process_transcription(transcription)

            # Verify backup preset was used and handling occurred
            assert mock_lwe_setup["backend"].run_template.call_count == 2
            last_call_args = mock_lwe_setup["backend"].run_template.call_args_list[-1]
            assert "preset" in last_call_args[1].get("overrides", {}).get(
                "request_overrides", {}
            )

            # Verify successful result was updated
            mock_update.assert_called_once()
            result = mock_update.call_args[0][0]
            assert isinstance(result, TranscriptionResult)
            assert result.success is True
            assert result.transcription_id == 1
            assert "Operator: Hello world" in result.transcription

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
        with pytest.raises(ModelResponseFormattingError) as exc_info:
            pipeline._extract_transcript_section("Invalid response")
        assert "No transcript section found" in str(exc_info.value)

        # Test None response
        with pytest.raises(ModelResponseFormattingError) as exc_info:
            pipeline._extract_transcript_section(None)
        assert "No response provided" in str(exc_info.value)

    def test_prepare_template_vars_basic(self, pipeline_args):
        """Test basic template variable preparation."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {"id": 123, "transcription": "test content"}
        vars = pipeline._prepare_template_vars(transcription)
        assert vars["transcription"] == "test content"
        assert len(vars["identifier"]) == UUID_SHORT_LENGTH

    def test_prepare_template_vars_missing_fields(self, pipeline_args):
        """Test template vars with missing fields."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        with pytest.raises(KeyError):
            pipeline._prepare_template_vars({"id": 123})

    def test_prepare_template_vars_extra_fields(self, pipeline_args):
        """Test template vars ignores extra fields."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {"id": 123, "transcription": "test content", "extra": "ignored"}
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
        overrides = pipeline._get_backup_overrides(123)

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
            "transcription-srt-labeling.md",
            template_vars={"test": "vars"},
            overrides=None,
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
            "transcription-srt-labeling.md",
            template_vars={"test": "vars"},
            overrides=test_overrides,
        )

    def test_empty_transcription_content(self, pipeline_args, mock_lwe_setup):
        """Test handling of empty transcription content."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        transcription = {"id": 123, "transcription": ""}
        result = pipeline._process_with_error_handling(transcription)

        assert isinstance(result, TranscriptionResult)
        assert result.transcription_id == 123
        assert result.success is False
        assert isinstance(result.error, ModelResponseFormattingError)
        assert "No transcript section found" in str(result.error)

    def test_malformed_transcription(self, pipeline_args, mock_lwe_setup):
        """Test handling of malformed transcription data."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        malformed = {"id": 123, "transcription": "bad data"}
        result = pipeline._process_with_error_handling(malformed)

        assert isinstance(result, TranscriptionResult)
        assert result.transcription_id == 123
        assert result.success is False
        assert isinstance(result.error, ModelResponseFormattingError)

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

    def test_build_update_url(self, pipeline_args):
        """Test URL construction for update endpoint."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        expected_url = f"https://{pipeline_args['domain']}/al/transcriptions/update/operator-recording"
        assert pipeline.build_update_url() == expected_url

    def test_api_payload_builder(self, pipeline_args):
        """Test ApiPayloadBuilder functionality."""
        builder = ApiPayloadBuilder(pipeline_args["api_key"])

        # Test successful result payload
        success_result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )
        success_payload = builder.build_payload(success_result)
        assert success_payload["api_key"] == pipeline_args["api_key"]
        assert success_payload["id"] == 123
        assert success_payload["success"] is True
        assert success_payload["transcription"] == "test content"

        # Test error result payload
        error_result = TranscriptionResult(
            transcription_id=456, success=False, error=Exception("Test error")
        )
        error_payload = builder.build_payload(error_result)
        assert error_payload["api_key"] == pipeline_args["api_key"]
        assert error_payload["id"] == 456
        assert error_payload["success"] is True
        assert "error_stage" in error_payload
        assert "Test error" in error_payload["error"]

    def test_handle_update_response_success(self, pipeline_args):
        """Test successful API response handling."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock()
        mock_response.json.return_value = {"success": True}

        # Should not raise any exceptions
        pipeline._handle_update_response(mock_response)

    def test_handle_update_response_auth_error(self, pipeline_args):
        """Test handling of authentication error response."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Invalid or missing API key",
        }

        with pytest.raises(AuthenticationError) as exc_info:
            pipeline._handle_update_response(mock_response)
        assert "Authentication failed: Invalid or missing API key" in str(
            exc_info.value
        )

    def test_handle_update_response_validation_error(self, pipeline_args):
        """Test handling of validation error response."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": False,
            "error": "Invalid transcription state",
        }

        with pytest.raises(ResponseValidationError) as exc_info:
            pipeline._handle_update_response(mock_response)
        assert "Response validation failed: Invalid transcription state" in str(
            exc_info.value
        )

    def test_handle_update_response_invalid_json(self, pipeline_args):
        """Test handling of invalid JSON response."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock()
        mock_response.json.side_effect = ValueError("Invalid JSON")

        with pytest.raises(ResponseValidationError) as exc_info:
            pipeline._handle_update_response(mock_response)
        assert "Invalid JSON response from API" in str(exc_info.value)

    def test_handle_update_response_missing_success(self, pipeline_args):
        """Test handling of response missing success field."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock()
        mock_response.json.return_value = {}

        with pytest.raises(ResponseValidationError) as exc_info:
            pipeline._handle_update_response(mock_response)
        assert "Missing success field in API response" in str(exc_info.value)

    def test_execute_update_request_success(self, pipeline_args):
        """Test successful execution of update request."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        url = "https://test.com/update"
        payload = {"test": "data"}

        with patch("srt_labeler.pipeline.post_request") as mock_post:
            mock_response = Mock()
            mock_post.return_value = mock_response

            response = pipeline._execute_update_request(url, payload)

            mock_post.assert_called_once_with(url, payload)
            assert response == mock_response

    def test_execute_update_request_network_error(self, pipeline_args):
        """Test handling of network errors during update request."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        url = "https://test.com/update"
        payload = {"test": "data"}

        with patch("srt_labeler.pipeline.post_request") as mock_post:
            mock_post.side_effect = requests.RequestException("Network error")

            with pytest.raises(RequestError) as exc_info:
                pipeline._execute_update_request(url, payload)

            assert "Failed to execute update request: Network error" in str(
                exc_info.value
            )

    def test_execute_update_request_timeout(self, pipeline_args):
        """Test handling of timeout during update request."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        url = "https://test.com/update"
        payload = {"test": "data"}

        with patch("srt_labeler.pipeline.post_request") as mock_post:
            mock_post.side_effect = requests.Timeout("Request timed out")

            with pytest.raises(RequestError) as exc_info:
                pipeline._execute_update_request(url, payload)

            assert "Failed to execute update request: Request timed out" in str(
                exc_info.value
            )

    def test_execute_update_request_connection_error(self, pipeline_args):
        """Test handling of connection errors during update request."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        url = "https://test.com/update"
        payload = {"test": "data"}

        with patch("srt_labeler.pipeline.post_request") as mock_post:
            mock_post.side_effect = requests.ConnectionError("Connection failed")

            with pytest.raises(RequestError) as exc_info:
                pipeline._execute_update_request(url, payload)

            assert "Failed to execute update request: Connection failed" in str(
                exc_info.value
            )

    def test_update_transcription_success(self, pipeline_args):
        """Test successful transcription update."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )

        with patch.multiple(
            pipeline,
            build_update_url=Mock(return_value="https://test.com/update"),
            _execute_update_request=Mock(),
            _handle_update_response=Mock(),
        ):
            pipeline.update_transcription(result)

            # Verify the flow
            pipeline.build_update_url.assert_called_once()
            pipeline._execute_update_request.assert_called_once()
            pipeline._handle_update_response.assert_called_once()

    def test_update_transcription_request_error(self, pipeline_args):
        """Test handling of request error during update."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )

        with patch.multiple(
            pipeline,
            build_update_url=Mock(return_value="https://test.com/update"),
            _execute_update_request=Mock(side_effect=RequestError("Network error")),
        ):
            with pytest.raises(RequestError) as exc_info:
                pipeline.update_transcription(result)
            assert "Network error" in str(exc_info.value)

    def test_update_transcription_validation_error(self, pipeline_args):
        """Test handling of validation error during update."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )

        with patch.multiple(
            pipeline,
            build_update_url=Mock(return_value="https://test.com/update"),
            _execute_update_request=Mock(),
            _handle_update_response=Mock(
                side_effect=ResponseValidationError("Validation failed")
            ),
        ):
            with pytest.raises(ResponseValidationError) as exc_info:
                pipeline.update_transcription(result)
            assert "Validation failed" in str(exc_info.value)

    def test_execute_model_analysis_success(self, pipeline_args, mock_lwe_setup):
        """Test successful model analysis execution."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Mock successful response
        mock_lwe_setup["backend"].run_template.return_value = (
            True,
            """
<thinking>Analysis</thinking>
<transcript>
1
00:00:01,000 --> 00:00:02,000
Operator: Test content
</transcript>""",
            None,
        )

        transcription = {
            "id": 123,
            "transcription": """1
00:00:01,000 --> 00:00:02,000
Test content""",
        }

        result = pipeline._execute_model_analysis(transcription, False)
        assert result.success is True
        assert result.transcription_id == 123
        assert result.transcription and "Operator: Test content" in result.transcription

    def test_execute_model_analysis_failure(self, pipeline_args, mock_lwe_setup):
        """Test model analysis execution failure."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Mock failed response
        mock_lwe_setup["backend"].run_template.return_value = (
            False,
            None,
            "Model error",
        )

        transcription = {"id": 123, "transcription": "test"}
        result = pipeline._execute_model_analysis(transcription, False)

        assert result.success is False
        assert result.transcription_id == 123
        assert "Model error" in str(result.error)

    def test_create_model_failure_result(self, pipeline_args):
        """Test creation of model failure result."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Test with error message
        result = pipeline._create_model_failure_result(123, "Test error")
        assert result.transcription_id == 123
        assert result.success is False
        assert "Test error" in str(result.error)

        # Test with None error message
        result = pipeline._create_model_failure_result(123, None)
        assert result.transcription_id == 123
        assert result.success is False
        assert "AI analysis failed" in str(result.error)

    def test_process_model_response_success(self, pipeline_args):
        """Test successful processing of model response."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        original_content = """1
00:00:01,000 --> 00:00:02,000
Hello world"""

        model_response = """
<thinking>Analysis</thinking>
<transcript>
1
00:00:01,000 --> 00:00:02,000
Operator: Hello world
</transcript>"""

        result = pipeline._process_model_response(123, original_content, model_response)
        assert result.success is True
        assert result.transcription_id == 123
        assert result.transcription and "Operator: Hello world" in result.transcription

    def test_process_model_response_extraction_error(self, pipeline_args):
        """Test handling of transcript extraction error."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        result = pipeline._process_model_response(
            123, "test content", "invalid response"
        )
        assert result.success is False
        assert result.transcription_id == 123
        assert isinstance(result.error, ModelResponseFormattingError)

    def test_process_model_response_merge_error(self, pipeline_args):
        """Test handling of merge error."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        model_response = """
<thinking>Analysis</thinking>
<transcript>
Invalid SRT format
</transcript>"""

        result = pipeline._process_model_response(123, "test content", model_response)
        assert result.success is False
        assert result.transcription_id == 123
        assert "Failed to merge SRT content" in str(result.error)

    def test_process_transcription_direct(self, pipeline_args, mock_lwe_setup):
        """Test direct processing of a single transcription."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        transcription = {"id": 123, "transcription": "test content"}

        # Mock the internal methods
        with patch.multiple(
            pipeline,
            _process_with_error_handling=Mock(
                return_value=TranscriptionResult(
                    transcription_id=123,
                    success=True,
                    transcription="processed content",
                )
            ),
            update_transcription=Mock(),
        ):
            pipeline._process_transcription(transcription)

            # Verify internal method calls
            pipeline._process_with_error_handling.assert_called_once_with(transcription)
            pipeline.update_transcription.assert_called_once()

    def test_update_transcription_direct(self, pipeline_args):
        """Test direct update of a transcription result."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )

        with patch.multiple(
            pipeline,
            build_update_url=Mock(return_value="test_url"),
            _execute_update_request=Mock(),
            _handle_update_response=Mock(),
        ):
            pipeline.update_transcription(result)

            # Verify the exact flow
            pipeline.build_update_url.assert_called_once()
            pipeline._execute_update_request.assert_called_once()
            pipeline._handle_update_response.assert_called_once()

    def test_attempt_model_processing_direct(self, pipeline_args):
        """Test direct model processing attempt."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {"id": 123, "transcription": "test"}

        # Test successful case
        with patch.object(
            pipeline,
            "_execute_model_analysis",
            return_value=TranscriptionResult(
                transcription_id=123, success=True, transcription="success"
            ),
        ):
            result = pipeline._attempt_model_processing(transcription)
            assert result.success is True
            assert result.transcription == "success"

        # Test error case
        with patch.object(
            pipeline, "_execute_model_analysis", side_effect=Exception("test error")
        ):
            result = pipeline._attempt_model_processing(transcription)
            assert result.success is False
            assert "test error" in str(result.error)

    def test_run_model_with_template_direct(self, pipeline_args, mock_lwe_setup):
        """Test direct template execution."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        transcription = {"id": 123, "transcription": "test"}

        with patch.multiple(
            pipeline,
            _prepare_template_vars=Mock(return_value={"test": "vars"}),
            _get_backup_overrides=Mock(return_value={"test": "override"}),
            _run_ai_analysis=Mock(return_value=(True, "response", None)),
        ):
            # Test without fallback
            success, response, error = pipeline._run_model_with_template(
                transcription, False
            )
            pipeline._prepare_template_vars.assert_called_once_with(transcription)
            pipeline._get_backup_overrides.assert_not_called()

            # Test with fallback
            success, response, error = pipeline._run_model_with_template(
                transcription, True
            )
            pipeline._get_backup_overrides.assert_called_once_with(123)

    def test_process_with_error_handling_direct(self, pipeline_args):
        """Test direct error handling process."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {"id": 123, "transcription": "test"}

        # Test successful primary attempt
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            return_value=TranscriptionResult(
                transcription_id=123, success=True, transcription="success"
            ),
        ):
            result = pipeline._process_with_error_handling(transcription)
            assert result.success is True
            assert result.transcription == "success"

        # Test fallback after primary failure
        primary_error = TranscriptionResult(
            transcription_id=123, success=False, error=Exception("primary error")
        )
        fallback_success = TranscriptionResult(
            transcription_id=123, success=True, transcription="fallback success"
        )
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            side_effect=[primary_error, fallback_success],
        ):
            result = pipeline._process_with_error_handling(transcription)
            assert result.success is True
            assert result.transcription == "fallback success"

    def test_update_transcription_auth_error(self, pipeline_args):
        """Test handling of authentication error during update."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )

        with patch.multiple(
            pipeline,
            build_update_url=Mock(return_value="https://test.com/update"),
            _execute_update_request=Mock(),
            _handle_update_response=Mock(
                side_effect=AuthenticationError("Invalid API key")
            ),
        ):
            with pytest.raises(AuthenticationError) as exc_info:
                pipeline.update_transcription(result)
            assert "Invalid API key" in str(exc_info.value)


class TestRequestError:
    def test_request_error_initialization(self):
        """Test basic RequestError initialization with message."""
        message = "Test request error"
        error = RequestError(message)
        assert str(error) == message
        assert isinstance(error, BaseError)  # Should inherit from BaseError

    def test_request_error_with_cause(self):
        """Test RequestError initialization with cause exception."""
        cause = ValueError("Original error")
        error = RequestError("Request error occurred", cause)
        assert str(error) == "Request error occurred"
        assert error.__cause__ == cause
        assert isinstance(error, BaseError)

    def test_request_error_inheritance_chain(self):
        """Test that RequestError properly inherits through the chain."""
        error = RequestError("Test error")
        assert isinstance(error, RequestError)
        assert isinstance(error, BaseError)
        assert isinstance(error, Exception)
        # Should be able to catch as any parent
        try:
            raise RequestError("Test error")
        except BaseError as e:
            assert isinstance(e, RequestError)

    def test_request_error_str_representation(self):
        """Test string representation includes any cause information."""
        cause = ValueError("Root cause")
        error = RequestError("Request error", cause)
        error_str = str(error)
        assert "Request error" in error_str
        # String representation should not include cause
        # as that's handled by Python's exception chaining
        assert "Root cause" not in error_str


class TestAuthenticationError:
    def test_authentication_error_initialization(self):
        """Test basic AuthenticationError initialization with message."""
        message = "Test authentication error"
        error = AuthenticationError(message)
        assert str(error) == message
        assert isinstance(error, BaseError)

    def test_authentication_error_with_cause(self):
        """Test AuthenticationError initialization with cause exception."""
        cause = ValueError("Original error")
        error = AuthenticationError("Auth error occurred", cause)
        assert str(error) == "Auth error occurred"
        assert error.__cause__ == cause
        assert isinstance(error, BaseError)

    def test_authentication_error_inheritance_chain(self):
        """Test that AuthenticationError properly inherits through the chain."""
        error = AuthenticationError("Test error")
        assert isinstance(error, AuthenticationError)
        assert isinstance(error, BaseError)
        assert isinstance(error, Exception)
        # Should be able to catch as any parent
        try:
            raise AuthenticationError("Test error")
        except BaseError as e:
            assert isinstance(e, AuthenticationError)

    def test_authentication_error_str_representation(self):
        """Test string representation includes any cause information."""
        cause = ValueError("Root cause")
        error = AuthenticationError("Auth error", cause)
        error_str = str(error)
        assert "Auth error" in error_str
        # String representation should not include cause
        # as that's handled by Python's exception chaining
        assert "Root cause" not in error_str


class TestResponseValidationError:
    def test_response_validation_error_initialization(self):
        """Test basic ResponseValidationError initialization with message."""
        message = "Test validation error"
        error = ResponseValidationError(message)
        assert str(error) == message
        assert isinstance(error, BaseError)

    def test_response_validation_error_with_cause(self):
        """Test ResponseValidationError initialization with cause exception."""
        cause = ValueError("Original error")
        error = ResponseValidationError("Validation error occurred", cause)
        assert str(error) == "Validation error occurred"
        assert error.__cause__ == cause
        assert isinstance(error, BaseError)

    def test_response_validation_error_inheritance_chain(self):
        """Test that ResponseValidationError properly inherits through the chain."""
        error = ResponseValidationError("Test error")
        assert isinstance(error, ResponseValidationError)
        assert isinstance(error, BaseError)
        assert isinstance(error, Exception)
        # Should be able to catch as any parent
        try:
            raise ResponseValidationError("Test error")
        except BaseError as e:
            assert isinstance(e, ResponseValidationError)

    def test_response_validation_error_str_representation(self):
        """Test string representation includes any cause information."""
        cause = ValueError("Root cause")
        error = ResponseValidationError("Validation error", cause)
        error_str = str(error)
        assert "Validation error" in error_str
        # String representation should not include cause
        # as that's handled by Python's exception chaining
        assert "Root cause" not in error_str


class TestModelResponseFormattingError:
    def test_model_response_formatting_error_initialization(self):
        """Test basic ModelResponseFormattingError initialization with message."""
        message = "Test model response error"
        error = ModelResponseFormattingError(message)
        assert str(error) == message
        assert isinstance(error, BaseError)

    def test_model_response_formatting_error_with_cause(self):
        """Test ModelResponseFormattingError initialization with cause exception."""
        cause = ValueError("Original error")
        error = ModelResponseFormattingError("Validation error occurred", cause)
        assert str(error) == "Validation error occurred"
        assert error.__cause__ == cause
        assert isinstance(error, BaseError)

    def test_model_response_formatting_error_inheritance_chain(self):
        """Test that ModelResponseFormattingError properly inherits through the chain."""
        error = ModelResponseFormattingError("Test error")
        assert isinstance(error, ModelResponseFormattingError)
        assert isinstance(error, BaseError)
        assert isinstance(error, Exception)
        # Should be able to catch as any parent
        try:
            raise ModelResponseFormattingError("Test error")
        except BaseError as e:
            assert isinstance(e, ModelResponseFormattingError)

    def test_model_response_formatting_error_str_representation(self):
        """Test string representation includes any cause information."""
        cause = ValueError("Root cause")
        error = ModelResponseFormattingError("Validation error", cause)
        error_str = str(error)
        assert "Validation error" in error_str
        # String representation should not include cause
        # as that's handled by Python's exception chaining
        assert "Root cause" not in error_str


class TestTranscriptionErrorHandler:
    """Test suite for TranscriptionErrorHandler class."""

    def test_is_hard_error_classification(self):
        """Test error classification logic."""
        handler = TranscriptionErrorHandler()

        # Test hard errors
        assert handler.is_hard_error(SrtMergeError("test")) is True
        assert handler.is_hard_error(ModelResponseFormattingError("test")) is True

        # Test transient errors
        assert handler.is_hard_error(Exception("test")) is False
        assert handler.is_hard_error(ValueError("test")) is False

    def test_should_update_with_error_both_hard(self):
        """Test update decision when both errors are hard."""
        handler = TranscriptionErrorHandler()
        primary = SrtMergeError("test")
        fallback = ModelResponseFormattingError("test")
        assert handler.should_update_with_error(primary, fallback) is True

    def test_should_update_with_error_mixed(self):
        """Test update decision with mixed error types."""
        handler = TranscriptionErrorHandler()

        # One hard, one transient
        assert (
            handler.should_update_with_error(SrtMergeError("test"), Exception("test"))
            is False
        )

        # One None
        assert handler.should_update_with_error(SrtMergeError("test"), None) is False

    def test_create_error_result(self):
        """Test error result creation."""
        handler = TranscriptionErrorHandler()
        error = Exception("test error")
        result = handler.create_error_result(123, error)

        assert isinstance(result, TranscriptionResult)
        assert result.transcription_id == 123
        assert result.success is False
        assert result.error == error


class TestTranscriptionResult:
    """Test suite for TranscriptionResult class."""

    def test_requires_api_update_success(self):
        """Test requires_api_update property with successful result."""
        result = TranscriptionResult(
            transcription_id=1, success=True, transcription="test"
        )
        assert result.requires_api_update is True

    def test_requires_api_update_transient_error(self):
        """Test requires_api_update property with transient error."""
        result = TranscriptionResult(
            transcription_id=1, success=False, error=Exception("Transient error")
        )
        assert result.requires_api_update is False

    def test_requires_api_update_hard_error(self):
        """Test requires_api_update property with hard error."""
        result = TranscriptionResult(
            transcription_id=1, success=False, error=SrtMergeError("Hard error")
        )
        assert result.requires_api_update is True

    def test_is_hard_error_no_error(self):
        """Test is_hard_error with no error."""
        result = TranscriptionResult(transcription_id=1, success=True)
        assert result.is_hard_error() is False

    def test_is_hard_error_transient(self):
        """Test is_hard_error with transient error."""
        result = TranscriptionResult(
            transcription_id=1, success=False, error=Exception("Transient")
        )
        assert result.is_hard_error() is False

    def test_is_hard_error_hard(self):
        """Test is_hard_error with hard error types."""
        # Test SrtMergeError
        result = TranscriptionResult(
            transcription_id=1, success=False, error=SrtMergeError("Hard")
        )
        assert result.is_hard_error() is True

        # Test ModelResponseFormattingError
        result = TranscriptionResult(
            transcription_id=1,
            success=False,
            error=ModelResponseFormattingError("Hard"),
        )
        assert result.is_hard_error() is True


class TestBaseError:
    def test_api_error_initialization(self):
        """Test basic BaseError initialization with message."""
        message = "Test API error"
        error = BaseError(message)
        assert str(error) == message
        assert isinstance(error, Exception)

    def test_api_error_with_cause(self):
        """Test BaseError initialization with cause exception."""
        cause = ValueError("Original error")
        error = BaseError("API error occurred", cause)
        assert str(error) == "API error occurred"
        assert error.__cause__ == cause

    def test_api_error_inheritance(self):
        """Test that BaseError properly inherits from Exception."""
        error = BaseError("Test error")
        assert isinstance(error, Exception)
        # Should be able to catch as Exception
        try:
            raise BaseError("Test error")
        except Exception as e:
            assert isinstance(e, BaseError)

    def test_api_error_str_representation(self):
        """Test string representation includes any cause information."""
        cause = ValueError("Root cause")
        error = BaseError("API error", cause)
        error_str = str(error)
        assert "API error" in error_str
        # String representation should not include cause
        # as that's handled by Python's exception chaining
        assert "Root cause" not in error_str
