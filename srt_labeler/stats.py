import sqlite3
import threading
from typing import Tuple
from .logger import Logger


class StatsDatabase:
    """Database for tracking LLM usage statistics."""

    def __init__(self, db_path: str, debug: bool = False) -> None:
        """Initialize stats database.

        :param db_path: Path to SQLite database file
        :param debug: Enable debug logging
        """
        self.db_path = db_path
        self.log = Logger(self.__class__.__name__, debug=debug)
        self.lock = threading.Lock()
        self.thread_local = threading.local()
        self.initialize()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection.

        :return: SQLite connection for current thread
        """
        if not hasattr(self.thread_local, "conn"):
            self.thread_local.conn = sqlite3.connect(self.db_path)
        return self.thread_local.conn

    def initialize(self) -> None:
        """Initialize database schema."""
        try:
            with self.lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS totals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        llm_primary INTEGER,
                        llm_fallback INTEGER,
                        llm_total INTEGER,
                        hard_failures INTEGER
                    )
                """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS model_response_errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        transcription_id INTEGER,
                        error TEXT,
                        original_content TEXT,
                        model_response TEXT,
                        fallback INTEGER
                    )
                """
                )
                cursor.execute("SELECT COUNT(*) FROM totals")
                if cursor.fetchone()[0] == 0:
                    query = "INSERT INTO totals (llm_primary, llm_fallback, llm_total, hard_failures) VALUES (0, 0, 0, 0)"
                    cursor.execute(query)
                conn.commit()
        except sqlite3.Error as e:
            self.log.error(f"Failed to initialize stats database: {e}")

    def increment_primary(self) -> None:
        """Increment primary model counter."""
        try:
            with self.lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                self.log.debug("Incrementing primary model counter")
                cursor.execute(
                    "UPDATE totals SET llm_primary = llm_primary + 1, llm_total = llm_total + 1"
                )
                conn.commit()
        except sqlite3.Error as e:
            self.log.error(f"Failed to increment primary counter: {e}")

    def increment_fallback(self) -> None:
        """Increment fallback model counter."""
        try:
            with self.lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                self.log.debug("Incrementing fallback model counter")
                cursor.execute(
                    "UPDATE totals SET llm_fallback = llm_fallback + 1, llm_total = llm_total + 1"
                )
                conn.commit()
        except sqlite3.Error as e:
            self.log.error(f"Failed to increment fallback counter: {e}")

    def get_totals(self) -> Tuple[int, int, int, int]:
        """Get current totals.

        :return: Tuple of (primary_count, fallback_count, total_count)
        """
        try:
            with self.lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT llm_primary, llm_fallback, llm_total, hard_failures FROM totals"
                )
                return cursor.fetchone()
        except sqlite3.Error as e:
            self.log.error(f"Failed to get totals: {e}")
            return (0, 0, 0, 0)

    def increment_hard_failure(self) -> None:
        try:
            with self.lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                self.log.debug("Incrementing hard failure counter")
                cursor.execute("UPDATE totals SET hard_failures = hard_failures + 1")
                conn.commit()
        except sqlite3.Error as e:
            self.log.error(f"Failed to increment hard failure counter: {e}")

    def log_model_response_error(
        self,
        transcription_id: int,
        error: str,
        original_content: str,
        model_response: str | None,
        use_fallback: bool,
    ) -> None:
        try:
            with self.lock:
                conn = self._get_conn()
                cursor = conn.cursor()
                self.log.debug("Adding model response error")
                cursor.execute(
                    "INSERT INTO model_response_errors (transcription_id, error, original_content, model_response, fallback) VALUES (?, ?, ?, ?, ?)",
                    (
                        transcription_id,
                        error,
                        original_content,
                        model_response,
                        use_fallback,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            self.log.error(f"Failed to log model response error: {e}")

    def close(self) -> None:
        """Close database connection."""
        try:
            if hasattr(self.thread_local, "conn"):
                with self.lock:
                    self.thread_local.conn.close()
                    del self.thread_local.conn
        except sqlite3.Error as e:
            self.log.error(f"Error closing database connection: {e}")
