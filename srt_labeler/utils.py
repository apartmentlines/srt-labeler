import argparse
import sys
import logging
import requests
from tenacity import retry, stop_after_attempt, wait_exponential
from transcription_pipeline.constants import (
    DEFAULT_RETRY_ATTEMPTS,
    DEFAULT_RETRY_BACKOFF,
    DOWNLOAD_TIMEOUT,
)


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"{value} is not a positive integer")
    return ivalue


def fail_hard(message: str) -> None:
    logger = logging.getLogger(__name__)
    logger.error(message)
    sys.exit(1)


def fail_soft(message: str) -> None:
    logger = logging.getLogger(__name__)
    logger.error(message)


@retry(
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=DEFAULT_RETRY_BACKOFF),
)
def get_request(url: str, params: dict) -> requests.Response:
    response = requests.get(url, params=params, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    return response


@retry(
    stop=stop_after_attempt(DEFAULT_RETRY_ATTEMPTS),
    wait=wait_exponential(multiplier=DEFAULT_RETRY_BACKOFF),
)
def post_request(url: str, data: dict) -> requests.Response:
    response = requests.post(url, data=data, timeout=DOWNLOAD_TIMEOUT)
    response.raise_for_status()
    return response
