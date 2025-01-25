import pytest
from unittest.mock import patch, Mock

from srt_labeler.main import (
    parse_arguments,
    main,
    SrtLabeler,
)


class TestTranscriptionPipeline:
    @pytest.fixture
    def srt_labeler(self):
        return SrtLabeler(
            api_key="test_key", file_api_key="test_file_key", domain="test_domain"
        )

    @pytest.fixture(autouse=True)
    def mock_default_stats_db(self, tmp_path):
        """Fixture to automatically mock DEFAULT_STATS_DB for all tests in this class."""
        default_db = str(tmp_path / "default.db")
        with patch("srt_labeler.main.DEFAULT_STATS_DB", default_db):
            yield default_db

    def test_initialization(self, tmp_path, mock_default_stats_db):
        custom_stats_db = str(tmp_path / "custom.db")
        # Test with all arguments specified
        srt_labeler = SrtLabeler(
            api_key="test_key",
            file_api_key="test_file_key",
            domain="test_domain",
            limit=5,
            debug=True,
            stats_db=custom_stats_db,
        )
        assert srt_labeler.api_key == "test_key"
        assert srt_labeler.file_api_key == "test_file_key"
        assert srt_labeler.domain == "test_domain"
        assert srt_labeler.limit == 5
        assert srt_labeler.debug is True
        assert srt_labeler.stats_db == custom_stats_db

        # Test with defaults
        srt_labeler = SrtLabeler(
            api_key="test_key", file_api_key="test_file_key", domain="test_domain"
        )
        assert srt_labeler.api_key == "test_key"
        assert srt_labeler.file_api_key == "test_file_key"
        assert srt_labeler.domain == "test_domain"
        assert srt_labeler.limit is None
        assert srt_labeler.debug is False
        assert srt_labeler.stats_db == mock_default_stats_db

    @patch("srt_labeler.main.get_request")
    def test_retrieve_transcription_data_success(self, mock_get_request, srt_labeler):
        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "files": [{"id": "1", "url": "url1"}],
        }
        mock_get_request.return_value = mock_response

        files = srt_labeler.retrieve_transcription_data()
        assert files == [{"id": "1", "url": "url1"}]

    @patch("srt_labeler.main.get_request")
    def test_retrieve_transcription_data_failure(self, mock_get_request, srt_labeler):
        mock_response = Mock()
        mock_response.json.return_value = {"success": False}
        mock_get_request.return_value = mock_response

        with pytest.raises(SystemExit):
            srt_labeler.retrieve_transcription_data()

    def test_setup_configuration(self, srt_labeler):
        with patch("srt_labeler.main.set_environment_variables") as mock_set_env:
            srt_labeler.setup_configuration()
            mock_set_env.assert_called_once_with(
                "test_key", "test_file_key", "test_domain"
            )

    def test_build_retrieve_request_url(self, srt_labeler):
        """Test that the retrieve request URL is correctly constructed."""
        expected_url = (
            "https://test_domain/al/transcriptions/retrieve/operator-recordings/active"
        )
        assert srt_labeler.build_retrieve_request_url() == expected_url

    def test_build_retrieve_request_params(self, srt_labeler):
        """Test that the retrieve request parameters are correctly constructed."""
        expected_params = {"api_key": "test_key"}
        assert srt_labeler.build_retrieve_request_params() == expected_params

    def test_build_retrieve_request_params_with_filters(self):
        """Test request params with min_id, max_id and from_s3 filters."""
        srt_labeler = SrtLabeler(
            api_key="test_key",
            file_api_key="test_file_key",
            domain="test_domain",
            min_id=100,
            max_id=200,
        )
        params = srt_labeler.build_retrieve_request_params()
        assert params == {
            "api_key": "test_key",
            "min_id": "100",
            "max_id": "200",
        }

    def test_build_retrieve_request_params_partial_filters(self):
        """Test request params with only some filters set."""
        srt_labeler = SrtLabeler(
            api_key="test_key",
            file_api_key="test_file_key",
            domain="test_domain",
            min_id=100,
        )
        params = srt_labeler.build_retrieve_request_params()
        assert params == {"api_key": "test_key", "min_id": "100"}

    @patch("srt_labeler.main.get_request")
    def test_retrieve_transcription_data_with_filters(self, mock_get_request):
        """Test that retrieve_transcription_data passes filters correctly."""
        srt_labeler = SrtLabeler(
            api_key="test_key",
            file_api_key="test_file_key",
            domain="test_domain",
            min_id=100,
            max_id=200,
        )

        mock_response = Mock()
        mock_response.json.return_value = {
            "success": True,
            "files": [{"id": "1", "url": "url1"}],
        }
        mock_get_request.return_value = mock_response

        srt_labeler.retrieve_transcription_data()

        mock_get_request.assert_called_once_with(
            srt_labeler.build_retrieve_request_url(),
            {"api_key": "test_key", "min_id": "100", "max_id": "200"},
        )

    def test_cli_arguments_with_filters(self):
        """Test that CLI arguments for filters are properly parsed."""
        test_args = [
            "--api-key",
            "test_api_key",
            "--file-api-key",
            "test_file_api_key",
            "--domain",
            "test_domain",
            "--min-id",
            "100",
            "--max-id",
            "200",
        ]
        with patch("sys.argv", ["main.py"] + test_args):
            args = parse_arguments()
            assert args.min_id == 100
            assert args.max_id == 200
            assert args.file_api_key == "test_file_api_key"


@patch("srt_labeler.main.load_configuration")
def test_main_configuration_error(mock_load_config):
    mock_load_config.side_effect = ValueError("Test error")
    with patch("sys.argv", ["main.py"]):
        with patch("srt_labeler.main.fail_hard") as mock_fail:
            main()
            mock_fail.assert_called_once_with("Test error")


def test_parse_arguments(tmp_path):
    custom_stats_db = str(tmp_path / "custom.db")
    test_args = [
        "--api-key",
        "test_api_key",
        "--file-api-key",
        "test_file_api_key",
        "--domain",
        "test_domain",
        "--debug",
        "--stats-db",
        custom_stats_db,
    ]
    with patch("sys.argv", ["main.py"] + test_args):
        args = parse_arguments()
        assert args.api_key == "test_api_key"
        assert args.file_api_key == "test_file_api_key"
        assert args.domain == "test_domain"
        assert args.debug is True
        assert args.stats_db == custom_stats_db
