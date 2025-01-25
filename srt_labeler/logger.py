import logging

STREAM_FORMAT = "%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"
FILE_FORMAT = "%(asctime)s [%(threadName)s] %(levelname)s: %(message)s"


class Logger:
    def __new__(cls, name, log_file=None, debug=False):
        """
        Setup instance-specific logger.

        :param name: Logger name
        :param log_file: Optional file to log to
        :param debug: Whether to enable debug logging
        :return: Configured logger instance
        """
        level = logging.DEBUG if debug else logging.INFO
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.propagate = False  # Prevent propagation to root logger

        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(STREAM_FORMAT))
        logger.addHandler(handler)

        if log_file:
            log_file_handler = logging.FileHandler(log_file, "a")
            log_file_handler.setLevel(level)
            log_file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
            log_file_handler.setLevel(level)
            logger.addHandler(log_file_handler)

        return logger
