# Pipeline constants
DEFAULT_LWE_POOL_LIMIT = 3
UUID_SHORT_LENGTH = 8

DEFAULT_RETRY_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF = 0.5  # in seconds
DOWNLOAD_TIMEOUT = 30  # in seconds

# Transcription state constants
TRANSCRIPTION_STATE_NOT_TRANSCRIBABLE = "not-transcribable"
TRANSCRIPTION_STATE_READY = "ready"
TRANSCRIPTION_STATE_ACTIVE = "active"
TRANSCRIPTION_STATE_COMPLETE = "complete"

# LWE constants
LWE_DEFAULT_PRESET = "gemini-1.5-flash-srt-labeler"
LWE_FALLBACK_PRESET = "gemini-1.5-pro-srt-labeler"
LWE_TRANSCRIPTION_TEMPLATE = "transcription-srt-labeling-with-audio.md"
