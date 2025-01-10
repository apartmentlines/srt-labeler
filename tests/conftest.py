import pytest

@pytest.fixture
def sample_srt():
    return "1\n00:00:00,000 --> 00:00:02,000\nTest subtitle\n\n"
