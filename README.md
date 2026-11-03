
# SRT Labeler

A Python tool for automatically applying speaker labels to SRT subtitle files using AI models. The tool processes subtitle files in parallel, handles errors gracefully, and supports multiple AI models for robust labeling.

## Features

- Automatic speaker labeling using AI models
- Multi-model support with fallback options
- Parallel processing for improved performance
- Robust error handling and recovery
- Preserves original subtitle timing and content
- Command-line tools for pipeline and standalone merging

## Installation

Requires Python 3.10 or higher.

```bash
pip install git+https://github.com/apartmentlines/srt-labeler.git
```

## Configuration

### Environment Variables

The following environment variables are required:

- `SRT_LABELER_API_KEY`: API key for authentication
- `SRT_LABELER_FILE_API_KEY`: Existing file API key configuration (retained for
  compatibility; model audio requests use signed URLs)
- `SRT_LABELER_DOMAIN`: Domain for API endpoints

These can also be provided via command-line arguments.

### LWE Configuration

The tool uses [LWE](https://github.com/llm-workflow-engine/llm-workflow-engine) (LLM Workflow Engine) with the following preset models:

- qwen3.8-omni-flash-srt-labeler (primary, `chat_openai_compat`)
- gemini-3.8-flash-srt-labeler (fallback, `chat_google_genai`, medium thinking)

And the following prompt templates:

- transcription-srt-labeling-with-audio.md (default)
- transcription-srt-labeling.md

## Usage

### Pipeline Tool

Process multiple transcriptions with speaker labeling:

```bash
srt-pipeline --api-key YOUR_API_KEY --file-api-key YOUR_FILE_API_KEY --domain your.domain.com
```

Optional arguments:
- `--help`: Script help
- `--limit N`: Process only N transcriptions
- `--min-id N`: Process transcriptions with ID >= N
- `--max-id N`: Process transcriptions with ID <= N
- `--debug`: Enable debug logging

### Local Testing Tool

Run one preset against local SRT and WAV files:

```bash
srt-eval --srt call.srt --audio call.wav \
  --preset gemini-3.8-flash-srt-labeler --output-dir runs/first
```

If omitted, `--preset` uses `LWE_DEFAULT_PRESET` from `srt_labeler/constants.py`,
the same default as the production pipeline. `srt-eval --help` shows its current
value. Each invocation uses only the selected preset, with no automatic fallback.
Audio is sent inline to the
model provider; provider credentials and network access are still required.
No ApartmentLines API credentials, retrieval, updates, or production statistics
are involved. Provider request-size limits still apply to inline audio.

The output directory must be new. It contains:

- `response.txt`: Complete model response text returned by LWE, before transcript
  extraction or merging. This remains available if extraction or merging fails.
- `final.srt`: Speaker labels merged into the original SRT using the same merger
  as production. Written only on success.

Use `--template-file path/to/prompt.md` to try a local LWE/Jinja template without
editing the bundled prompt, and `--debug` for detailed LWE logging to the console
and `lwe-debug.log` in the output directory. Templates receive
the same `transcription` and `identifier` variables as production. Runs reuse
the bundled LWE configuration and data directories, honoring `LWE_CONFIG_DIR`
and `LWE_DATA_DIR` when set. Run-specific settings are applied in memory, with
user creation, user/conversation loading, and chat logging disabled. LWE's normal
database initialization and schema checks still apply.

Exit status is `0` on success, `1` for execution or processing failures, and `2`
for invalid CLI arguments. If the provider raises an error before returning a
response, there is no response file to save. Run multiple pairs or compare
templates by invoking the command again with a different output directory.

After installing the updated package, `srt-eval` is available alongside
`srt-pipeline`. From a checkout, `python -m srt_labeler.local` accepts the same
arguments.

### Merger Tool

Merge speaker labels from a labeled SRT file into an unlabeled one:

```bash
srt-merger unlabeled.srt labeled.srt
```

Optional arguments:
- `--valid-labels label1,label2`: Specify valid speaker labels
- `--debug`: Enable debug logging

## Error Handling

The pipeline implements sophisticated error handling:

- **Hard Errors**: Permanent failures (e.g., invalid SRT format, invalid labels)
- **Transient Errors**: Temporary failures that may succeed on retry
- **Fallback Behavior**: Automatically attempts fallback model on primary failure
- **API Updates**: Only updates API for successful results or confirmed hard errors

## Stats Tracking

The tool maintains usage statistics in a SQLite database (default: stats.db in the current directory). It tracks:
- Primary model usage count
- Fallback model usage count
- Total model invocations

## Development

### Setup

Install development dependencies:

```bash
pip install -e ".[dev]"
```

### Testing

Run the test suite:

```bash
pytest
```

## Architecture

Key components:

- **SrtLabeler**: Main orchestrator handling configuration and initialization
- **SrtLabelerPipeline**: Core processing pipeline managing parallel execution
- **SrtMerger**: Handles merging of labeled and unlabeled SRT content
- **Error Handling**: Hierarchical error system with specific error types
- **Threading**: Thread pool with per-thread LWE backend instances

## Caveats

The default configuration assumes a conversation between an "Operator" and a "Caller". Using different speaker labels requires:

1. Modifying the `DEFAULT_VALID_LABELS` in merger.py
2. Updating the LWE prompt templates instructing the AI model to use the new labels

## API Integration

The tool interacts with an external API that must implement these endpoints:

### Retrieve Transcriptions
```
GET /al/transcriptions/retrieve/operator-recordings/active
```
Parameters:
- api_key (required): Authentication key
- limit (optional): Maximum number of transcriptions to return
- min_id (optional): Minimum transcription ID
- max_id (optional): Maximum transcription ID

Response:
```json
{
    "success": true,
    "files": [
        {
            "id": 123,
            "transcription": "SRT content...",
            "url": "https://example.com/audio/123.wav",
            "signed_url": "https://example.com/audio/123.wav?transcription_id=123&api_key=SIGNATURE"
        }
    ]
}
```

### Download Audio File
```
GET {url from retrieve response}
```
Parameters:
- api_key (required): File API key for authentication

Returns the audio file content directly.

Alternatively, request `signed_url` verbatim.

### Update Transcription
```
POST /al/transcriptions/update/operator-recording
```
Parameters:
- api_key (required): Authentication key
- id (required): Transcription ID
- success (required): Always true (indicates processing complete)
- transcription_state (required): Set to "complete"
- transcription (optional): Updated SRT content if successful
- metadata (optional): JSON string with error details if failed
