import argparse
import sys
import pytest
from unittest.mock import patch, Mock
import requests
import tenacity
from transcription_pipeline.utils import (
    positive_int,
    fail_hard,
    fail_soft,
    get_request,
    post_request,
)


def test_positive_int():
    assert positive_int("5") == 5
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-1")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")


def test_fail_hard(caplog):
    with patch.object(sys, "exit") as mock_exit:
        fail_hard("Test error")
        mock_exit.assert_called_with(1)
        assert "Test error" in caplog.text


def test_fail_soft(caplog):
    fail_soft("Soft error")
    assert "Soft error" in caplog.text


@patch("requests.get")
def test_get_request_success(mock_get):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_get.return_value = mock_response
    response = get_request("http://example.com", params={"api_key": "test_key"})
    assert response.status_code == 200


@patch("requests.get")
@patch("tenacity.nap.time.sleep")  # Patch sleep to avoid delays
def test_get_request_failure(mock_sleep, mock_get):
    mock_get.side_effect = requests.exceptions.RequestException
    with pytest.raises(tenacity.RetryError):
        get_request("http://example.com", params={"api_key": "test_key"})


@patch("requests.post")
def test_post_request_success(mock_post):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_post.return_value = mock_response
    response = post_request("http://example.com", data={"key": "value"})
    assert response.status_code == 200


@patch("requests.post")
@patch("tenacity.nap.time.sleep")  # Patch sleep to avoid delays
def test_post_request_failure(mock_sleep, mock_post):
    mock_post.side_effect = requests.exceptions.RequestException
    with pytest.raises(tenacity.RetryError):
        post_request("http://example.com", data={"key": "value"})
