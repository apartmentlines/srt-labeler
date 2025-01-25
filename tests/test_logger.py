import logging
from srt_labeler.logger import Logger, STREAM_FORMAT, FILE_FORMAT


def test_logger_initialization_default():
    logger = Logger("test_logger")
    assert logger.name == "test_logger"
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.INFO


def test_logger_initialization_debug():
    logger = Logger("test_logger", debug=True)
    assert logger.level == logging.DEBUG
    assert logger.handlers[0].level == logging.DEBUG


def test_logger_initialization_with_log_file(tmp_path):
    log_file = tmp_path / "test.log"
    logger = Logger("test_logger", log_file=str(log_file))
    assert len(logger.handlers) == 2
    # Check for StreamHandler
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    # Check for FileHandler
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    log_file.unlink()


def test_logger_stream_formatting(caplog):
    logger = Logger("test_logger")
    logger.propagate = True
    with caplog.at_level(logging.INFO, logger="test_logger"):
        logger.info("Test message")
    record = caplog.records[0]
    assert record.message == "Test message"
    assert (
        STREAM_FORMAT in logger.handlers[0].formatter._fmt
    )  # pyright: ignore[reportOptionalMemberAccess, reportOperatorIssue]


def test_logger_file_formatting(tmp_path):
    log_file = tmp_path / "test.log"
    logger = Logger("test_logger", log_file=str(log_file))
    logger.info("Test file message")
    with open(log_file, "r") as f:
        content = f.read()
    assert "Test file message" in content
    file_handler = [h for h in logger.handlers if isinstance(h, logging.FileHandler)][0]
    assert (
        FILE_FORMAT in file_handler.formatter._fmt
    )  # pyright: ignore[reportOptionalMemberAccess, reportOperatorIssue]
    log_file = tmp_path / "test.log"


def test_logger_logging_levels(caplog):
    logger = Logger("test_logger", debug=True)
    logger.propagate = True
    with caplog.at_level(logging.DEBUG, logger="test_logger"):
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")
        logger.critical("Critical message")

    log_messages = [record.message for record in caplog.records]
    assert "Debug message" in log_messages
    assert "Info message" in log_messages
    assert "Warning message" in log_messages
    assert "Error message" in log_messages
    assert "Critical message" in log_messages


def test_logger_no_propagation(caplog):
    logger = Logger("test_logger")
    logger.propagate = True  # Enable propagation temporarily
    logger.info("Propagation test")
    assert any("Propagation test" in record.message for record in caplog.records)


def test_multiple_logger_instances():
    logger1 = Logger("logger1")
    logger2 = Logger("logger2", debug=True)
    assert logger1.name == "logger1"
    assert logger2.name == "logger2"
    assert logger1.level == logging.INFO
    assert logger2.level == logging.DEBUG
    assert len(logger1.handlers) == 1
    assert len(logger2.handlers) == 1


def test_logger_reinitialization():
    logger = Logger("test_logger")
    initial_handlers = len(logger.handlers)
    # Reinitialize logger
    logger = Logger("test_logger")
    assert len(logger.handlers) == initial_handlers
