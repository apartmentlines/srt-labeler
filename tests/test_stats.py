import pytest
import sqlite3
import threading
from srt_labeler.stats import StatsDatabase


class TestStatsDatabase:
    """Test suite for StatsDatabase class."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Fixture to provide a temporary database path."""
        db_path = tmp_path / "test.db"
        return str(db_path)

    def test_initialization(self, test_db):
        """Test database initialization."""
        stats = StatsDatabase(test_db)
        assert stats.db_path == test_db
        assert hasattr(stats, "lock")

        # Verify tables exist
        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        assert "totals" in tables
        assert "model_response_errors" in tables

        # Verify we can get totals
        totals = stats.get_totals()
        assert isinstance(totals, tuple)
        assert len(totals) == 4  # primary, fallback, total, hard_failures

    def test_increment_primary(self, test_db):
        """Test primary counter increment."""
        stats = StatsDatabase(test_db)
        stats.increment_primary()
        totals = stats.get_totals()
        assert totals[0] == 1  # primary
        assert totals[1] == 0  # fallback
        assert totals[2] == 1  # total

    def test_increment_fallback(self, test_db):
        """Test fallback counter increment."""
        stats = StatsDatabase(test_db)
        stats.increment_fallback()
        totals = stats.get_totals()
        assert totals[0] == 0  # primary
        assert totals[1] == 1  # fallback
        assert totals[2] == 1  # total

    def test_get_totals(self, test_db):
        """Test getting totals."""
        stats = StatsDatabase(test_db)
        stats.increment_primary()
        stats.increment_fallback()
        totals = stats.get_totals()
        assert totals[0] == 1  # primary
        assert totals[1] == 1  # fallback
        assert totals[2] == 2  # total

    def test_close(self, test_db):
        """Test database close."""
        stats = StatsDatabase(test_db)
        stats.close()
        assert not hasattr(stats, "conn")

    def test_initialize_error_handling(self, test_db):
        """Test error handling during initialization."""
        stats = StatsDatabase(test_db)
        # Force an error by making the database path invalid
        stats.db_path = "/nonexistent/path/db.sqlite"
        stats.initialize()  # Should log error but not raise exception

    def test_increment_primary_error_handling(self, test_db):
        """Test error handling during primary increment."""
        stats = StatsDatabase(test_db)
        # Force an error by making the database path invalid
        stats.db_path = "/nonexistent/path/db.sqlite"
        stats.increment_primary()  # Should log error but not raise exception

    def test_increment_fallback_error_handling(self, test_db):
        """Test error handling during fallback increment."""
        stats = StatsDatabase(test_db)
        # Force an error by making the database path invalid
        stats.db_path = "/nonexistent/path/db.sqlite"
        stats.increment_fallback()  # Should log error but not raise exception

    def test_get_totals_error_handling(self, test_db):
        """Test error handling during totals retrieval."""
        stats = StatsDatabase(test_db)
        # Force an error by making the database path invalid
        stats.db_path = "/nonexistent/path/db.sqlite"
        totals = stats.get_totals()
        assert totals == (0, 0, 0, 0)  # Should return zeros on error

    def test_increment_hard_failure(self, test_db):
        """Test hard failure counter increment."""
        stats = StatsDatabase(test_db)
        stats.increment_hard_failure()
        totals = stats.get_totals()
        assert totals[0] == 0  # primary
        assert totals[1] == 0  # fallback
        assert totals[2] == 0  # total
        assert totals[3] == 1  # hard failures

    def test_increment_hard_failure_error_handling(self, test_db):
        """Test error handling during hard failure increment."""
        stats = StatsDatabase(test_db)
        # Force an error by making the database path invalid
        stats.db_path = "/nonexistent/path/db.sqlite"
        stats.increment_hard_failure()  # Should log error but not raise exception

    def test_thread_safety_hard_failures(self, test_db):
        """Test thread safety for hard failures counter."""
        stats = StatsDatabase(test_db)
        thread_count = 4
        increments_per_thread = 10

        def increment_hard():
            for _ in range(increments_per_thread):
                stats.increment_hard_failure()

        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=increment_hard)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        totals = stats.get_totals()
        assert totals[0] == 0  # primary
        assert totals[1] == 0  # fallback
        assert totals[2] == 0  # total
        assert totals[3] == thread_count * increments_per_thread  # hard failures

    def test_close_error_handling(self, test_db):
        """Test error handling during close."""
        stats = StatsDatabase(test_db)
        stats.close()  # Close it first
        stats.close()  # Should handle gracefully when already closed

    def test_thread_safety_primary(self, test_db):
        """Test thread safety for primary counter."""
        stats = StatsDatabase(test_db)
        thread_count = 4
        increments_per_thread = 10

        def increment_primary():
            for _ in range(increments_per_thread):
                stats.increment_primary()

        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=increment_primary)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        totals = stats.get_totals()
        assert totals[0] == thread_count * increments_per_thread  # primary
        assert totals[1] == 0  # fallback
        assert totals[2] == thread_count * increments_per_thread  # total

    def test_thread_safety_fallback(self, test_db):
        """Test thread safety for fallback counter."""
        stats = StatsDatabase(test_db)
        thread_count = 4
        increments_per_thread = 10

        def increment_fallback():
            for _ in range(increments_per_thread):
                stats.increment_fallback()

        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=increment_fallback)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        totals = stats.get_totals()
        assert totals[0] == 0  # primary
        assert totals[1] == thread_count * increments_per_thread  # fallback
        assert totals[2] == thread_count * increments_per_thread  # total

    def test_thread_safety_mixed(self, test_db):
        """Test thread safety with mixed primary and fallback updates."""
        stats = StatsDatabase(test_db)
        thread_count = 4
        increments_per_thread = 10

        def increment_both():
            for _ in range(increments_per_thread):
                stats.increment_primary()
                stats.increment_fallback()

        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=increment_both)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        totals = stats.get_totals()
        assert totals[0] == thread_count * increments_per_thread  # primary
        assert totals[1] == thread_count * increments_per_thread  # fallback
        assert totals[2] == thread_count * increments_per_thread * 2  # total

    def test_log_model_response_error_success(self, test_db):
        stats = StatsDatabase(test_db)
        stats.log_model_response_error(123, "test error", "original", "response", False)

        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM model_response_errors")
        row = cursor.fetchone()
        assert row[1] == 123
        assert row[2] == "test error"
        assert row[3] == "original"
        assert row[4] == "response"
        assert row[5] == False

    def test_log_model_response_error_error_handling(self, test_db):
        stats = StatsDatabase(test_db)
        stats.db_path = "/nonexistent/path/db.sqlite"
        stats.log_model_response_error(123, "test error", "original", "response", False)

    def test_thread_safety_model_response_errors(self, test_db):
        stats = StatsDatabase(test_db)
        thread_count = 4
        logs_per_thread = 10

        def log_errors():
            for i in range(logs_per_thread):
                stats.log_model_response_error(i, "error", "content", "response", False)

        threads = []
        for _ in range(thread_count):
            t = threading.Thread(target=log_errors)
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        conn = sqlite3.connect(test_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM model_response_errors")
        count = cursor.fetchone()[0]
        assert count == thread_count * logs_per_thread
