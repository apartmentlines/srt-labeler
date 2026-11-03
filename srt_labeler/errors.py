"""Shared errors for model requests and transcript extraction."""

from __future__ import annotations


class BaseError(Exception):
    """Base exception class."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize API error.

        :param message: Error message
        :param cause: Optional causing exception
        """
        super().__init__(message)
        if cause:
            self.__cause__: BaseException | None = cause


class RequestError(BaseError):
    """Exception raised for errors during request construction or execution."""

    def __init__(self, message: str, cause: Exception | None = None) -> None:
        """Initialize request error.

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
