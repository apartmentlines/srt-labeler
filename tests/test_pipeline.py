import os
import pytest
import requests
import json
from unittest.mock import Mock, patch, ANY
from concurrent.futures import ThreadPoolExecutor
from langchain_core.messages import HumanMessage
from lwe.core.config import Config
from lwe import ApiBackend
from srt_labeler.pipeline import (
    SrtLabelerPipeline,
    BaseError,
    RequestError,
    AuthenticationError,
    ResponseValidationError,
    ModelResponseFormattingError,
    RequestFileNotFoundError,
    TranscriptionResult,
    TranscriptionErrorHandler,
    ApiPayloadBuilder,
)
from srt_labeler.merger import SrtMergeError
from srt_labeler.constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_LWE_POOL_LIMIT,
    UUID_SHORT_LENGTH,
    LWE_FALLBACK_PRESET,
    LWE_TRANSCRIPTION_TEMPLATE,
    DOWNLOAD_TIMEOUT,
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
    def audio_transcription(self):
        """Fixture providing a transcription dict with audio URL."""
        return {
            "id": 123,
            "transcription": "1\n00:00:01,000 --> 00:00:02,000\nHello world",
            "url": "https://test.com/audio.wav",
            "metadata": {"call_uuid": "4125551212"},
        }

    @pytest.fixture
    def mock_audio_response(self):
        """Fixture providing a mock successful audio file response."""
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.content = b"test_audio_data"
        response.headers = {"content-type": "audio/wav"}
        return response

    @pytest.fixture
    def mock_error_response(self):
        """Fixture providing a mock error response."""
        response = Mock(spec=requests.Response)
        response.status_code = 404
        response.content = b""
        response.headers = {"content-type": "application/json"}
        response.json.return_value = {"error": "Not found"}
        return response

    @pytest.fixture
    def mock_network(self):
        """Fixture providing common request and retry patches."""
        with (
            patch("requests.get") as mock_get,
            patch("tenacity.nap.time.sleep") as mock_sleep,
        ):
            yield {"get": mock_get, "sleep": mock_sleep}

    @pytest.fixture
    def pipeline_args(self):
        """Fixture providing pipeline initialization arguments."""
        return {
            "api_key": "test_api_key",
            "file_api_key": "test_file_api_key",
            "domain": "test_domain",
            "debug": False,
        }

    def test_pipeline_initialization_with_default_parameters(self, pipeline_args):
        pipeline = SrtLabelerPipeline(**pipeline_args)
        assert pipeline.api_key == "test_api_key"
        assert pipeline.file_api_key == "test_file_api_key"
        assert pipeline.domain == "test_domain"
        assert pipeline.max_workers == DEFAULT_LWE_POOL_LIMIT
        assert pipeline.debug is False
        assert pipeline.log is not None
        assert pipeline.executor._max_workers == DEFAULT_LWE_POOL_LIMIT
        assert pipeline.executor._thread_name_prefix == "LWEWorker"

    def test_pipeline_initialization_with_custom_parameters(self):
        pipeline = SrtLabelerPipeline(
            api_key="custom_api_key",
            file_api_key="custom_file_api_key",
            domain="custom_domain",
            max_workers=4,
            debug=True,
        )
        assert pipeline.api_key == "custom_api_key"
        assert pipeline.file_api_key == "custom_file_api_key"
        assert pipeline.domain == "custom_domain"
        assert pipeline.max_workers == 4
        assert pipeline.debug is True

    def test_pipeline_initialization_missing_credentials(self):
        with pytest.raises(ValueError):
            SrtLabelerPipeline()
        with pytest.raises(ValueError):
            SrtLabelerPipeline(api_key="test", file_api_key="test")
        with pytest.raises(ValueError):
            SrtLabelerPipeline(api_key="test", domain="test")

    def test_process_transcriptions(
        self, pipeline_args, audio_transcription, mock_audio_response, mock_network
    ):
        """Test processing multiple transcriptions."""
        mock_executor = Mock(spec=ThreadPoolExecutor)
        # Configure the mock executor's submit to actually call the function
        mock_executor.submit.side_effect = lambda f, *args: Mock(
            result=lambda: f(*args)
        )

        with patch(
            "srt_labeler.pipeline.ThreadPoolExecutor", return_value=mock_executor
        ):
            pipeline = SrtLabelerPipeline(**pipeline_args)
            mock_network["get"].return_value = mock_audio_response

            test_transcriptions = [
                {**audio_transcription, "id": 1},
                {**audio_transcription, "id": 2},
                {**audio_transcription, "id": 3},
            ]

            with patch.object(pipeline, "_process_transcription") as mock_process:
                pipeline.process_transcriptions(test_transcriptions)

                # Verify each transcription was processed
                assert mock_process.call_count == len(test_transcriptions)
                process_calls = [call[0][0] for call in mock_process.call_args_list]
                assert all(t in test_transcriptions for t in process_calls)

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_process_transcriptions_error_handling(
        self, pipeline_args, mock_audio_response, mock_network
    ):
        """Test error handling during transcription processing."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_network["get"].return_value = mock_audio_response

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
                TranscriptionResult(
                    transcription_id=3,
                    success=False,
                    error=RequestFileNotFoundError("Audio file not found"),
                ),
            ]

            with patch.object(pipeline, "update_transcription") as mock_update:
                test_transcriptions = [
                    {
                        "id": 1,
                        "transcription": "test1",
                        "url": "https://test.com/audio1.wav",
                    },
                    {
                        "id": 2,
                        "transcription": "test2",
                        "url": "https://test.com/audio2.wav",
                    },
                    {
                        "id": 3,
                        "transcription": "test3",
                        "url": "https://test.com/audio3.wav",
                    },
                ]
                pipeline.process_transcriptions(test_transcriptions)

                # Verify error handling
                assert mock_process.call_count == len(test_transcriptions)

                # Verify only hard errors triggered API update
                assert mock_update.call_count == 2
                update_calls = [call[0][0] for call in mock_update.call_args_list]
                assert any(
                    isinstance(r.error, ModelResponseFormattingError)
                    for r in update_calls
                )
                assert any(
                    isinstance(r.error, RequestFileNotFoundError) for r in update_calls
                )

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_process_transcriptions_multiple_errors(
        self, pipeline_args, mock_audio_response, mock_network
    ):
        """Test handling of multiple error types including audio-related errors."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_network["get"].return_value = mock_audio_response

        def side_effect(transcription, _):
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
            if transcription["id"] == 3:
                return TranscriptionResult(
                    transcription_id=3,
                    success=False,
                    error=RequestFileNotFoundError(
                        "Audio file not found"
                    ),  # Audio error
                )
            return TranscriptionResult(
                transcription_id=4, success=True, transcription="Success"
            )

        with patch.object(pipeline, "_execute_model_analysis", side_effect=side_effect):
            with patch.object(pipeline, "update_transcription") as mock_update:
                test_transcriptions = [
                    {
                        "id": 1,
                        "transcription": "test1",
                        "url": "https://test.com/audio1.wav",
                    },
                    {
                        "id": 2,
                        "transcription": "test2",
                        "url": "https://test.com/audio2.wav",
                    },
                    {
                        "id": 3,
                        "transcription": "test3",
                        "url": "https://test.com/audio3.wav",
                    },
                    {
                        "id": 4,
                        "transcription": "test4",
                        "url": "https://test.com/audio4.wav",
                    },
                ]

                pipeline.process_transcriptions(test_transcriptions)

                # Verify error handling behavior
                assert mock_update.call_count == 3
                update_calls = [call[0][0] for call in mock_update.call_args_list]
                assert any(
                    isinstance(r.error, SrtMergeError)
                    for r in update_calls
                    if not r.success
                )
                assert any(
                    isinstance(r.error, RequestFileNotFoundError)
                    for r in update_calls
                    if not r.success
                )
                assert any(
                    r.success and r.transcription == "Success" for r in update_calls
                )

    def test_check_download_errors_http_404(self, pipeline_args):
        """Test that a 404 response raises RequestFileNotFoundError."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        error = requests.HTTPError("404 Client Error: Not Found for url")
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 404

        with pytest.raises(RequestFileNotFoundError) as exc_info:
            pipeline._check_download_errors_http(mock_response, error)
        assert "Error downloading audio file: 404" in str(exc_info.value)

    def test_check_download_errors_http_other_error(self, pipeline_args):
        """Test that other HTTP errors raise the original exception."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        error = requests.HTTPError("500 Server Error: Internal Server Error for url")
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 500

        with pytest.raises(requests.HTTPError) as exc_info:
            pipeline._check_download_errors_http(mock_response, error)
        assert "500 Server Error" in str(exc_info.value)

    def test_check_download_errors_http_no_response(self, pipeline_args):
        """Test that when response is None, the original exception is raised."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        error = requests.ConnectionError("Connection failed")

        with pytest.raises(requests.ConnectionError) as exc_info:
            pipeline._check_download_errors_http(response=None, error=error)
        assert "Connection failed" in str(exc_info.value)

    def test_check_download_errors_api_json_error(self, pipeline_args):
        """Test that a JSON error response raises RequestError."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": "File not found"}

        with pytest.raises(RequestError) as exc_info:
            pipeline._check_download_errors_api(mock_response)
        assert "Error downloading audio file: {'error': 'File not found'}" in str(
            exc_info.value
        )

    def test_check_download_errors_api_no_error(self, pipeline_args):
        """Test that when content-type is not JSON, no exception is raised."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.headers = {"content-type": "audio/wav"}

        # This should not raise any exception
        pipeline._check_download_errors_api(mock_response)  # No exception expected

    def test_check_download_errors_api_response_none(self, pipeline_args):
        """Test that when response is None, the method does nothing."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        # This should not raise any exception
        pipeline._check_download_errors_api(response=None)  # No exception expected

    def test_try_download_file_success(self, pipeline_args):
        """Test successful file download via _try_download_file."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.content = b"Audio data"
        mock_response.headers = {"content-type": "audio/wav"}

        with (
            patch.object(
                pipeline, "_download_file", return_value=mock_response
            ) as mock_download,
            patch.object(pipeline, "_check_download_errors_api") as mock_check_api,
        ):
            content = pipeline._try_download_file(
                {"url": "http://example.com/audio.wav"}
            )
            mock_download.assert_called_once()
            mock_check_api.assert_called_once_with(mock_response)
            assert content == b"Audio data"

    def test_try_download_file_http_error(self, pipeline_args):
        """Test that HTTP errors in _download_file propagate up through _try_download_file."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        error = requests.HTTPError("500 Client Error")
        with patch.object(
            pipeline, "_download_file", side_effect=error
        ) as mock_download:
            with pytest.raises(requests.HTTPError) as exc_info:
                pipeline._try_download_file({"url": "http://example.com/audio.wav"})
            assert "500 Client Error" in str(exc_info.value)
            mock_download.assert_called_once()

    def test_try_download_file_api_error(self, pipeline_args):
        """Test handling of API errors in _try_download_file."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.headers = {"content-type": "application/json"}
        mock_response.json.return_value = {"error": "File not found"}
        mock_response.content = b""

        with (
            patch.object(
                pipeline, "_download_file", return_value=mock_response
            ) as mock_download,
            patch.object(
                pipeline,
                "_check_download_errors_api",
                side_effect=RequestError("API error occurred"),
            ),
        ):
            with pytest.raises(RequestError) as exc_info:
                pipeline._try_download_file({"url": "http://example.com/audio.wav"})
            assert "API error occurred" in str(exc_info.value)
            mock_download.assert_called_once()

    def test_try_download_file_empty_response(self, pipeline_args):
        """Test handling of empty response content in _try_download_file."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.content = b""
        mock_response.headers = {"content-type": "audio/wav"}

        with patch.object(pipeline, "_download_file", return_value=mock_response):
            with pytest.raises(RequestError) as exc_info:
                pipeline._try_download_file({"url": "http://example.com/audio.wav"})
            assert "Received empty response" in str(exc_info.value)

    def test_download_file_success(
        self, pipeline_args, audio_transcription, mock_network
    ):
        """Test successful file download."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.content = b"Audio data"
        mock_response.headers = {"content-type": "audio/wav"}
        mock_network["get"].return_value = mock_response

        response = pipeline._download_file(audio_transcription)
        mock_network["get"].assert_called_once_with(
            audio_transcription["url"],
            params={"api_key": pipeline.file_api_key},
            timeout=DOWNLOAD_TIMEOUT,
        )
        assert response == mock_response

    def test_download_file_success_metadata_with_s3(
        self, pipeline_args, audio_transcription, mock_network
    ):
        """Test successful file download."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_response = Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.content = b"Audio data"
        mock_response.headers = {"content-type": "audio/wav"}
        mock_network["get"].return_value = mock_response

        response = pipeline._download_file(
            {**audio_transcription, "metadata": {"call_uuid": "N/A"}}
        )
        mock_network["get"].assert_called_once_with(
            f"{audio_transcription["url"]}",
            params={"api_key": pipeline.file_api_key, "from_s3": "1"},
            timeout=DOWNLOAD_TIMEOUT,
        )
        assert response == mock_response

    def test_download_file_network_error_retries(
        self, pipeline_args, audio_transcription, mock_network
    ):
        """Test that network errors cause retries."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Configure mock to raise an exception on first calls, then succeed
        mock_response_success = Mock(spec=requests.Response)
        mock_response_success.status_code = 200
        mock_response_success.content = b"Audio data"
        mock_response_success.headers = {"content-type": "audio/wav"}

        # First two calls raise exception, third call returns success
        mock_network["get"].side_effect = [
            requests.ConnectionError("Network Error"),
            requests.ConnectionError("Network Error"),
            mock_response_success,
        ]

        response = pipeline._download_file(audio_transcription)
        assert response == mock_response_success
        assert response and response.content == b"Audio data"

        assert mock_network["get"].call_count == 3  # Retried twice before success

    def test_download_file_max_retries_exceeded(
        self, pipeline_args, audio_transcription, mock_network
    ):
        """Test that max retries exceeded raises the exception."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Configure mock to always raise an exception
        mock_network["get"].side_effect = requests.ConnectionError("Network Error")

        with pytest.raises(requests.ConnectionError) as exc_info:
            pipeline._download_file(audio_transcription)

        assert "Network Error" in str(exc_info.value)
        assert mock_network["get"].call_count == DEFAULT_RETRY_ATTEMPTS

    def test_add_audio_file(self, pipeline_args, audio_transcription):
        """Test that _add_audio_file creates a HumanMessage with the audio data."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_audio_data = b"Audio data"

        with patch.object(
            pipeline, "_try_download_file", return_value=mock_audio_data
        ) as mock_download:
            result = pipeline._add_audio_file(transcription=audio_transcription)
            mock_download.assert_called_once_with(audio_transcription)

            assert isinstance(result, HumanMessage)
            assert len(result.content) == 1
            media_content = result.content[0]
            assert isinstance(media_content, dict)
            assert media_content.get("type") == "media"
            assert media_content.get("mime_type") == "audio/wav"
            assert media_content.get("data") == mock_audio_data

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_thread_pool_cleanup(self, pipeline_args):
        """Test that thread pool is properly shut down when cleanup is called."""
        mock_executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "srt_labeler.pipeline.ThreadPoolExecutor", return_value=mock_executor
        ):
            pipeline = SrtLabelerPipeline(**pipeline_args)
            pipeline.cleanup()
            mock_executor.shutdown.assert_called_once_with(wait=True)

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_thread_pool_cleanup_context(self, pipeline_args):
        """Test that thread pool is properly shut down when used as context manager."""
        mock_executor = Mock(spec=ThreadPoolExecutor)
        with patch(
            "srt_labeler.pipeline.ThreadPoolExecutor", return_value=mock_executor
        ):
            with SrtLabelerPipeline(**pipeline_args):
                pass
            mock_executor.shutdown.assert_called_once_with(wait=True)

    def test_backend_reuse_within_thread(
        self, pipeline_args, audio_transcription, mock_lwe_setup
    ):
        """Test that backends are reused by verifying only pool_size backends are created."""
        test_pool_limit = 1
        pipeline = SrtLabelerPipeline(**pipeline_args, max_workers=test_pool_limit)

        num_transcriptions = 5
        test_transcriptions = [
            {**audio_transcription, "id": i} for i in range(num_transcriptions)
        ]
        with patch.object(pipeline, "_process_transcription"):
            pipeline.process_transcriptions(test_transcriptions)
            assert mock_lwe_setup["backend_class"].call_count == test_pool_limit

    def test_backend_initialization_with_config_paths(
        self, pipeline_args, mock_lwe_setup
    ):
        """Test that backend initialization uses correct config paths."""
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

    def test_process_transcription_with_ai(
        self,
        pipeline_args,
        audio_transcription,
        mock_lwe_setup,
        mock_audio_response,
        mock_network,
    ):
        """Test processing a single transcription with AI model."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response

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

        with patch.object(pipeline, "update_transcription") as mock_update:
            pipeline._process_transcription(audio_transcription)

            # Verify update was called with successful result
            mock_update.assert_called_once()
            result = mock_update.call_args[0][0]
            assert isinstance(result, TranscriptionResult)
            assert result.success is True
            assert result.transcription_id == 123
            assert (
                result.transcription and "Operator: Hello world" in result.transcription
            )

            # Verify audio file was downloaded
            mock_network["get"].assert_called_once_with(
                audio_transcription["url"],
                params={"api_key": pipeline.file_api_key},
                timeout=DOWNLOAD_TIMEOUT,
            )

    def test_process_transcription_backup_preset(
        self, pipeline_args, mock_lwe_setup, mock_audio_response, mock_network
    ):
        """Test fallback to backup preset after failures."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response

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
            "url": "https://test.com/audio.wav",
        }

        with patch.object(pipeline, "update_transcription") as mock_update:
            pipeline._process_transcription(transcription)

            # Verify backup preset was used and handling occurred
            assert mock_lwe_setup["backend"].run_template.call_count == 2

            # Verify first attempt with audio file
            first_call_args = mock_lwe_setup["backend"].run_template.call_args_list[0]
            assert "files" in first_call_args[1].get("overrides", {}).get(
                "request_overrides", {}
            )

            # Verify second attempt with audio file and backup preset
            last_call_args = mock_lwe_setup["backend"].run_template.call_args_list[-1]
            overrides = (
                last_call_args[1].get("overrides", {}).get("request_overrides", {})
            )
            assert "preset" in overrides
            assert "files" in overrides
            assert overrides["preset"] == LWE_FALLBACK_PRESET

            # Verify successful result was updated
            mock_update.assert_called_once()
            result = mock_update.call_args[0][0]
            assert isinstance(result, TranscriptionResult)
            assert result.success is True
            assert result.transcription_id == 1
            assert (
                result.transcription and "Operator: Hello world" in result.transcription
            )

            # Verify audio file downloads were attempted for both tries
            assert mock_network["get"].call_count == 2
            assert all(
                call[0][0] == transcription["url"]
                for call in mock_network["get"].call_args_list
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
        with pytest.raises(ModelResponseFormattingError) as exc_info:
            pipeline._extract_transcript_section("Invalid response")
        assert "No transcript section found" in str(exc_info.value)

        # Test None response
        with pytest.raises(ModelResponseFormattingError) as exc_info:
            pipeline._extract_transcript_section(None)
        assert "No response provided" in str(exc_info.value)

    def test_prepare_template_vars_basic(self, pipeline_args, audio_transcription):
        """Test basic template variable preparation."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        vars = pipeline._prepare_template_vars(audio_transcription)
        assert vars["transcription"] == audio_transcription["transcription"]
        assert len(vars["identifier"]) == UUID_SHORT_LENGTH

    def test_prepare_template_vars_missing_fields(self, pipeline_args):
        """Test template vars with missing fields."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        with pytest.raises(KeyError):
            pipeline._prepare_template_vars({"id": 123})

    def test_prepare_template_vars_extra_fields(
        self, pipeline_args, audio_transcription
    ):
        """Test template vars ignores extra fields."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        transcription = {**audio_transcription, "extra": "ignored"}
        vars = pipeline._prepare_template_vars(transcription)
        assert vars["transcription"] == audio_transcription["transcription"]
        assert len(vars["identifier"]) == UUID_SHORT_LENGTH

    def test_generate_identifier(self, pipeline_args):
        """Test unique identifier generation."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        id1 = pipeline._generate_identifier()
        id2 = pipeline._generate_identifier()
        assert len(id1) == UUID_SHORT_LENGTH
        assert len(id2) == UUID_SHORT_LENGTH
        assert id1 != id2  # Verify uniqueness

    def test_get_request_overrides_basic(
        self, pipeline_args, audio_transcription, mock_audio_response, mock_network
    ):
        """Test basic request overrides without fallback."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_network["get"].return_value = mock_audio_response

        overrides = pipeline._get_request_overrides(audio_transcription, fallback=False)

        assert "request_overrides" in overrides
        assert "files" in overrides["request_overrides"]
        assert "preset" not in overrides["request_overrides"]
        mock_network["get"].assert_called_once_with(
            audio_transcription["url"],
            params={"api_key": pipeline.file_api_key},
            timeout=DOWNLOAD_TIMEOUT,
        )

    def test_get_request_overrides_with_fallback(
        self,
        pipeline_args,
        audio_transcription,
        mock_audio_response,
        mock_network,
        capsys,
    ):
        """Test request overrides with fallback preset."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_network["get"].return_value = mock_audio_response

        overrides = pipeline._get_request_overrides(audio_transcription, fallback=True)

        assert "request_overrides" in overrides
        assert "files" in overrides["request_overrides"]
        assert overrides["request_overrides"]["preset"] == LWE_FALLBACK_PRESET
        captured = capsys.readouterr()
        assert (
            f"Using backup preset for transcription {audio_transcription['id']}"
            in captured.err
        )

    def test_get_request_overrides_audio_file(
        self, pipeline_args, audio_transcription, mock_audio_response, mock_network
    ):
        """Test audio file handling in request overrides."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_network["get"].return_value = mock_audio_response
        mock_audio_response.content = b"test_audio_data"
        mock_audio_response.headers = {"content-type": "audio/wav"}

        overrides = pipeline._get_request_overrides(audio_transcription, fallback=False)

        files = overrides["request_overrides"]["files"]
        assert len(files) == 1
        assert isinstance(files[0], HumanMessage)
        assert files[0].content[0]["type"] == "media"
        assert files[0].content[0]["mime_type"] == "audio/wav"
        assert files[0].content[0]["data"] == b"test_audio_data"

    def test_get_request_overrides_download_error(
        self, pipeline_args, audio_transcription, mock_error_response, mock_network
    ):
        """Test error handling during file download in request overrides."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        mock_network["get"].return_value = mock_error_response
        mock_error_response.status_code = 404

        with pytest.raises(RequestError) as exc_info:
            pipeline._get_request_overrides(audio_transcription, fallback=False)

        assert "Error downloading audio file: {'error': 'Not found'}" in str(
            exc_info.value
        )

    def test_run_ai_analysis_no_overrides(
        self,
        pipeline_args,
        mock_lwe_setup,
        mock_audio_response,
        mock_network,
        audio_transcription,
    ):
        """Test AI analysis without overrides."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response
        mock_audio_response.content = b"test_audio_data"
        mock_audio_response.headers = {"content-type": "audio/wav"}

        mock_lwe_setup["backend"].run_template.return_value = (True, "response", None)
        overrides = pipeline._get_request_overrides(audio_transcription, fallback=False)
        success, response, error = pipeline._run_ai_analysis(
            {"test": "vars"}, overrides
        )

        # Verify template call with audio file
        template_call = mock_lwe_setup["backend"].run_template.call_args
        assert template_call[0][0] == LWE_TRANSCRIPTION_TEMPLATE
        assert template_call[1]["template_vars"] == {"test": "vars"}
        assert "files" in template_call[1]["overrides"]["request_overrides"]
        assert isinstance(
            template_call[1]["overrides"]["request_overrides"]["files"][0], HumanMessage
        )
        assert success
        assert response == "response"
        assert error is None

    def test_run_ai_analysis_with_overrides(
        self,
        pipeline_args,
        mock_lwe_setup,
        mock_audio_response,
        mock_network,
        audio_transcription,
    ):
        """Test AI analysis with overrides."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response
        mock_audio_response.content = b"test_audio_data"
        mock_audio_response.headers = {"content-type": "audio/wav"}

        mock_lwe_setup["backend"].run_template.return_value = (True, "response", None)
        overrides = pipeline._get_request_overrides(audio_transcription, fallback=True)
        pipeline._run_ai_analysis({"test": "vars"}, overrides)

        # Verify template call with audio file and preset override
        template_call = mock_lwe_setup["backend"].run_template.call_args
        assert template_call[0][0] == LWE_TRANSCRIPTION_TEMPLATE
        assert template_call[1]["template_vars"] == {"test": "vars"}
        assert "files" in template_call[1]["overrides"]["request_overrides"]
        assert isinstance(
            template_call[1]["overrides"]["request_overrides"]["files"][0], HumanMessage
        )
        assert (
            template_call[1]["overrides"]["request_overrides"]["preset"]
            == LWE_FALLBACK_PRESET
        )

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_empty_transcription_content(
        self,
        pipeline_args,
        mock_audio_response,
        mock_network,
        audio_transcription,
    ):
        """Test handling of empty transcription content."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response
        transcription = {**audio_transcription, "transcription": ""}
        result = pipeline._process_with_error_handling(transcription)

        assert isinstance(result, TranscriptionResult)
        assert result.transcription_id == 123
        assert result.success is False
        assert isinstance(result.error, ModelResponseFormattingError)
        assert "No transcript section found" in str(result.error)

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_malformed_transcription(
        self,
        pipeline_args,
        mock_audio_response,
        mock_network,
        audio_transcription,
    ):
        """Test handling of malformed transcription data."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response
        malformed = {**audio_transcription, "transcription": "bad data"}
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
            mock_backend_class.side_effect = lambda _: Mock(spec=ApiBackend)

            # Simulate multiple thread initializations
            pipeline._initialize_worker()
            first_backend = pipeline.thread_local.backend

            pipeline._initialize_worker()
            second_backend = pipeline.thread_local.backend

            # Each initialization should create a new backend instance
            assert first_backend is not second_backend

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_process_transcriptions_empty_list(self, pipeline_args):
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
        assert "metadata" in error_payload
        metadata = json.loads(error_payload["metadata"])
        assert metadata["error_stage"] == "labeling"
        assert metadata["error"] == "Test error"

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
            "message": "Invalid or missing API key",
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
            "message": "Invalid transcription state",
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

        update_url = "https://test.com/update"
        with (
            patch.object(
                pipeline, "build_update_url", return_value=update_url
            ) as mock_build_url,
            patch.object(pipeline, "_execute_update_request") as mock_execute,
            patch.object(pipeline, "_handle_update_response") as mock_handle,
        ):
            pipeline.update_transcription(result)
            mock_build_url.assert_called_once()
            mock_execute.assert_called_once_with(update_url, ANY)
            mock_handle.assert_called_once_with(mock_execute.return_value)

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

    def test_execute_model_analysis_success(
        self,
        pipeline_args,
        mock_lwe_setup,
        mock_audio_response,
        mock_network,
        audio_transcription,
    ):
        """Test successful model analysis execution."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response
        mock_audio_response.content = b"test_audio_data"
        mock_audio_response.headers = {"content-type": "audio/wav"}

        # Mock successful response
        mock_lwe_setup["backend"].run_template.return_value = (
            True,
            """
<thinking>Analysis</thinking>
<transcript>
1
00:00:01,000 --> 00:00:02,000
Operator: Hello world
</transcript>""",
            None,
        )

        result = pipeline._execute_model_analysis(audio_transcription, False)

        # Verify audio file handling
        mock_network["get"].assert_called_once_with(
            audio_transcription["url"],
            params={"api_key": pipeline.file_api_key},
            timeout=DOWNLOAD_TIMEOUT,
        )

        # Verify template call included audio file
        template_call = mock_lwe_setup["backend"].run_template.call_args
        assert "files" in template_call[1]["overrides"]["request_overrides"]
        assert isinstance(
            template_call[1]["overrides"]["request_overrides"]["files"][0], HumanMessage
        )

        # Verify result
        assert result.success is True
        assert result.transcription_id == audio_transcription["id"]
        assert result.transcription and "Operator: Hello world" in result.transcription

    def test_execute_model_analysis_failure(
        self,
        pipeline_args,
        mock_lwe_setup,
        mock_audio_response,
        mock_network,
        audio_transcription,
    ):
        """Test model analysis execution failure."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()
        mock_network["get"].return_value = mock_audio_response
        mock_audio_response.content = b"test_audio_data"
        mock_audio_response.headers = {"content-type": "audio/wav"}

        # Mock failed response
        mock_lwe_setup["backend"].run_template.return_value = (
            False,
            None,
            "Model error",
        )

        result = pipeline._execute_model_analysis(audio_transcription, False)

        # Verify audio file handling
        mock_network["get"].assert_called_once_with(
            audio_transcription["url"],
            params={"api_key": pipeline.file_api_key},
            timeout=DOWNLOAD_TIMEOUT,
        )

        # Verify template call included audio file
        template_call = mock_lwe_setup["backend"].run_template.call_args
        assert "files" in template_call[1]["overrides"]["request_overrides"]
        assert isinstance(
            template_call[1]["overrides"]["request_overrides"]["files"][0], HumanMessage
        )

        # Verify result
        assert result.success is False
        assert result.transcription_id == audio_transcription["id"]
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
invalid
</transcript>"""

        result = pipeline._process_model_response(123, "test content", model_response)
        assert result.success is False
        assert result.transcription_id == 123
        assert "Invalid SRT format" in str(result.error)

    @pytest.mark.usefixtures("mock_lwe_setup")
    def test_process_transcription_direct(self, pipeline_args):
        """Test direct processing of a single transcription."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        transcription = {"id": 123, "transcription": "test content"}

        mock_result = TranscriptionResult(
            transcription_id=123,
            success=True,
            transcription="processed content",
        )

        # Mock the internal methods using individual patches
        with (
            patch.object(
                pipeline, "_process_with_error_handling", return_value=mock_result
            ) as mock_process,
            patch.object(pipeline, "update_transcription") as mock_update,
        ):
            pipeline._process_transcription(transcription)

            # Verify internal method calls
            mock_process.assert_called_once_with(transcription)
            mock_update.assert_called_once_with(mock_result)

    def test_update_transcription_direct(self, pipeline_args):
        """Test direct update of a transcription result."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        result = TranscriptionResult(
            transcription_id=123, success=True, transcription="test content"
        )

        with (
            patch.object(
                pipeline, "build_update_url", return_value="test_url"
            ) as mock_build_url,
            patch.object(pipeline, "_execute_update_request") as mock_execute,
            patch.object(pipeline, "_handle_update_response") as mock_handle,
        ):
            pipeline.update_transcription(result)

            # Verify the exact flow
            mock_build_url.assert_called_once()
            mock_execute.assert_called_once()
            mock_handle.assert_called_once()

    def test_attempt_model_processing_direct(self, pipeline_args, audio_transcription):
        """Test direct model processing attempt."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Test successful case
        with patch.object(
            pipeline,
            "_execute_model_analysis",
            return_value=TranscriptionResult(
                transcription_id=audio_transcription["id"],
                success=True,
                transcription="success",
            ),
        ):
            result = pipeline._attempt_model_processing(audio_transcription)
            assert result.success is True
            assert result.transcription == "success"

        # Test error case with successful audio download
        with patch.object(
            pipeline, "_execute_model_analysis", side_effect=Exception("test error")
        ):
            result = pipeline._attempt_model_processing(audio_transcription)
            assert result.success is False
            assert "test error" in str(result.error)

    def test_run_model_with_template_direct(
        self, pipeline_args, mock_lwe_setup, audio_transcription
    ):
        """Test direct template execution with audio handling."""
        pipeline = SrtLabelerPipeline(**pipeline_args)
        pipeline._initialize_worker()

        # Mock the response object
        mock_response = Mock(spec=requests.Response)
        mock_response.content = b"test_audio_data"
        mock_response.headers = {"content-type": "audio/wav"}
        mock_response.status_code = 200

        with patch.object(
            pipeline, "_download_file", return_value=mock_response
        ) as mock_download:
            # Test without fallback
            pipeline._run_model_with_template(
                transcription=audio_transcription, use_fallback=False
            )
            mock_download.assert_called_once_with(audio_transcription)
            template_call = mock_lwe_setup["backend"].run_template.call_args
            assert template_call[0][0] == LWE_TRANSCRIPTION_TEMPLATE
            assert "files" in template_call[1]["overrides"]["request_overrides"]
            assert isinstance(
                template_call[1]["overrides"]["request_overrides"]["files"][0],
                HumanMessage,
            )
            assert (
                template_call[1]["overrides"]["request_overrides"]["files"][0].content[
                    0
                ]["data"]
                == b"test_audio_data"
            )
            mock_download.reset_mock()
            # Test with fallback
            pipeline._run_model_with_template(
                transcription=audio_transcription, use_fallback=True
            )
            mock_download.assert_called_once_with(audio_transcription)
            template_call = mock_lwe_setup["backend"].run_template.call_args
            assert template_call[0][0] == LWE_TRANSCRIPTION_TEMPLATE
            assert (
                template_call[1]["overrides"]["request_overrides"]["preset"]
                == LWE_FALLBACK_PRESET
            )
            assert "files" in template_call[1]["overrides"]["request_overrides"]
            assert isinstance(
                template_call[1]["overrides"]["request_overrides"]["files"][0],
                HumanMessage,
            )
            assert (
                template_call[1]["overrides"]["request_overrides"]["files"][0].content[
                    0
                ]["data"]
                == b"test_audio_data"
            )

    def test_process_with_error_handling_direct(
        self, pipeline_args, audio_transcription
    ):
        """Test direct error handling process."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        mock_determine = Mock(
            return_value=TranscriptionResult(
                transcription_id=audio_transcription["id"],
                success=False,
                error=Exception("final error"),
            )
        )
        pipeline._determine_final_error_result = mock_determine
        fallback_success = TranscriptionResult(
            transcription_id=audio_transcription["id"],
            success=True,
            transcription="fallback success",
        )
        primary_hard_error = TranscriptionResult(
            transcription_id=audio_transcription["id"],
            success=False,
            error=SrtMergeError("primary hard error"),
        )
        fallback_hard_error = TranscriptionResult(
            transcription_id=audio_transcription["id"],
            success=False,
            error=ModelResponseFormattingError("fallback hard error"),
        )
        primary_transient = TranscriptionResult(
            transcription_id=audio_transcription["id"],
            success=False,
            error=Exception("primary transient error"),
        )
        fallback_transient = TranscriptionResult(
            transcription_id=audio_transcription["id"],
            success=False,
            error=Exception("fallback transient error"),
        )

        # Test successful primary attempt
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            return_value=TranscriptionResult(
                transcription_id=audio_transcription["id"],
                success=True,
                transcription="success",
            ),
        ):
            result = pipeline._process_with_error_handling(audio_transcription)
            assert result.success is True
            assert result.transcription == "success"
            mock_determine.assert_not_called()  # Verify not called on success

        mock_determine.reset_mock()

        # Test fallback after primary failure
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            side_effect=[primary_hard_error, fallback_success],
        ):
            result = pipeline._process_with_error_handling(audio_transcription)
            assert result.success is True
            assert result.transcription == "fallback success"
            mock_determine.assert_not_called()  # Verify not called on fallback success

        mock_determine.reset_mock()

        # Test primary hard error, fallback hard error
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            side_effect=[primary_hard_error, fallback_hard_error],
        ):
            result = pipeline._process_with_error_handling(audio_transcription)
            mock_determine.assert_called_once_with(
                audio_transcription["id"], primary_hard_error, fallback_hard_error
            )
            assert result.success is False
            assert isinstance(result.error, Exception)
            assert "final error" in str(result.error)

        mock_determine.reset_mock()

        # Test primary transient error, fallback transient
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            side_effect=[primary_transient, fallback_transient],
        ):
            result = pipeline._process_with_error_handling(audio_transcription)
            mock_determine.assert_called_once_with(
                audio_transcription["id"], primary_transient, fallback_transient
            )
            assert result.success is False
            assert isinstance(result.error, Exception)
            assert "final error" in str(result.error)

        mock_determine.reset_mock()

        # Test primary hard error, fallback transient
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            side_effect=[primary_hard_error, fallback_transient],
        ):
            result = pipeline._process_with_error_handling(audio_transcription)
            mock_determine.assert_called_once_with(
                audio_transcription["id"], primary_hard_error, fallback_transient
            )
            assert result.success is False
            assert isinstance(result.error, Exception)
            assert "final error" in str(result.error)

        # Reset mock for next test
        mock_determine.reset_mock()

        # Test primary transient error, fallback hard
        with patch.object(
            pipeline,
            "_attempt_model_processing",
            side_effect=[primary_transient, fallback_hard_error],
        ):
            result = pipeline._process_with_error_handling(audio_transcription)
            mock_determine.assert_called_once_with(
                audio_transcription["id"], primary_transient, fallback_hard_error
            )
            assert result.success is False
            assert isinstance(result.error, Exception)
            assert "final error" in str(result.error)

    def test_determine_final_error_result(self, pipeline_args):
        """Test determination of final error result under different scenarios."""
        pipeline = SrtLabelerPipeline(**pipeline_args)

        # Create test results with different error types
        primary_hard = TranscriptionResult(
            transcription_id=123,
            success=False,
            error=SrtMergeError("primary hard error"),
        )
        primary_transient = TranscriptionResult(
            transcription_id=123,
            success=False,
            error=Exception("primary transient error"),
        )
        fallback_hard = TranscriptionResult(
            transcription_id=123,
            success=False,
            error=ModelResponseFormattingError("fallback hard error"),
        )
        fallback_transient = TranscriptionResult(
            transcription_id=123,
            success=False,
            error=Exception("fallback transient error"),
        )

        # Test when both errors are hard (should return fallback result)
        with patch.object(
            TranscriptionErrorHandler, "should_update_with_error", return_value=True
        ) as mock_should_update:
            result = pipeline._determine_final_error_result(
                123, primary_hard, fallback_hard
            )
            mock_should_update.assert_called_once_with(
                primary_hard.error, fallback_hard.error
            )
            assert result == fallback_hard
            assert isinstance(result.error, ModelResponseFormattingError)
            assert result.transcription_id == 123
            assert result.success is False

        # Test primary transient, fallback hard
        with patch.object(
            TranscriptionErrorHandler, "should_update_with_error", return_value=False
        ) as mock_should_update:
            result = pipeline._determine_final_error_result(
                123, primary_transient, fallback_hard
            )
            mock_should_update.assert_called_once_with(
                primary_transient.error, fallback_hard.error
            )
            assert isinstance(result.error, Exception)
            assert "primary transient error" in str(result.error)
            assert result.transcription_id == 123
            assert result.success is False

        # Test primary hard, fallback transient
        with patch.object(
            TranscriptionErrorHandler, "should_update_with_error", return_value=False
        ) as mock_should_update:
            result = pipeline._determine_final_error_result(
                123, primary_hard, fallback_transient
            )
            mock_should_update.assert_called_once_with(
                primary_hard.error, fallback_transient.error
            )
            assert isinstance(result.error, Exception)
            assert "fallback transient error" in str(result.error)
            assert result.transcription_id == 123
            assert result.success is False

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


class TestRequestFileNotFoundError:
    def test_request_file_not_found_error_initialization(self):
        """Test basic RequestFileNotFoundError initialization with message."""
        message = "Not found error"
        error = RequestFileNotFoundError(message)
        assert str(error) == message
        assert isinstance(error, BaseError)

    def test_request_file_not_found_error_with_cause(self):
        """Test RequestFileNotFoundError initialization with cause exception."""
        cause = ValueError("Original error")
        error = RequestFileNotFoundError("Not found error occurred", cause)
        assert str(error) == "Not found error occurred"
        assert error.__cause__ == cause
        assert isinstance(error, BaseError)

    def test_request_file_not_found_error_inheritance_chain(self):
        """Test that RequestFileNotFoundError properly inherits through the chain."""
        error = RequestFileNotFoundError("Test error")
        assert isinstance(error, RequestFileNotFoundError)
        assert isinstance(error, BaseError)
        assert isinstance(error, Exception)
        # Should be able to catch as any parent
        try:
            raise RequestFileNotFoundError("Test error")
        except BaseError as e:
            assert isinstance(e, RequestFileNotFoundError)

    def test_request_file_not_found_error_str_representation(self):
        """Test string representation includes any cause information."""
        cause = ValueError("Root cause")
        error = RequestFileNotFoundError("Not found error", cause)
        error_str = str(error)
        assert "Not found error" in error_str
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
