# Portable dataset manifest

JSON schema version 1; all paths resolve relative to the manifest, not the working directory. Paths may reference files outside its directory. Call IDs must be unique case-insensitively, 1–31 ASCII letters/digits/underscores/hyphens (also safe worksheet names).

```json
{
  "schema_version": 1,
  "cases": [{
    "id": "call-001",
    "source": "srt/call-001.srt",
    "reference": "reference-srt/call-001.srt",
    "audio": "audio/call-001.wav",
    "split": "development",
    "excluded": {"18": "mixed speakers"},
    "flagged": [18],
    "full_review": true
  }]
}
```

`source`, `reference`, `audio`, and `id` are required for evaluation. `split`, `excluded`, `flagged`, and `full_review` default to empty/false. `excluded` maps original block indexes to nonempty reasons. `flagged` controls review highlighting, not scoring. `full_review` requests a whole-call listen, even with no highlighted blocks. Optional `hashes` maps source/reference/audio to SHA-256 hex strings; a mismatch aborts before requests.

Each nonempty SRT must have increasing unique positive block indexes and positive durations; reference blocks must match source text/times/indexes exactly after removing one `Operator: ` or `Caller: ` prefix. Audio must be a readable WAV covering all subtitle endpoints. Use the reference data as immutable input. Keep workbook notes and human decisions in the original dataset's audit trail.
