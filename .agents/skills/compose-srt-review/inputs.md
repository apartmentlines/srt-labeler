# Review inputs

Compose from a corpus following the shared [contract](../evaluate-srt-presets/contracts.md#case-authority).
Rendering preserves its decisions, evidence, and approval state.

To assemble new cases, supply a JSON selection of input files:

```json
{
  "cases": [{
    "id": "call-001",
    "source": "srt/call-001.srt",
    "draft": "drafts/call-001.srt",
    "audio": "audio/call-001.wav",
    "split": "development",
    "flagged": [18],
    "full_review": true
  }]
}
```

```sh
python .agents/skills/evaluate-srt-presets/scripts/corpus.py create \
  --inputs path/to/selection.json --corpus-id PROJECT \
  --output test-data/new-review
```

Paths resolve relative to the selection file; absolute paths are also accepted.
For canonical operator recordings, set each `audio` field to the appropriate
file under
`${HOME}/git/apartmentlines/testing-references/audio/operator-recordings/`.
For example, using `${HOME}` as a placeholder for your home directory:

```json
"audio": "${HOME}/git/apartmentlines/testing-references/audio/operator-recordings/2026-11-05T19_49_02_+13523270688.wav"
```

For a custom audio location, supply its absolute path or a path relative to the
selection file in `audio`, then pass that selection with `--inputs`. JSON paths
do not expand environment variables or `~`; substitute the actual home directory
for `${HOME}` before saving the selection JSON. Creation copies the WAV into the new corpus; existing corpus
and review packages retain their frozen audio copies.

`draft` is labeled SRT awaiting
human review. `source` may be omitted; creation then removes exactly one
Operator/Caller prefix to derive the unlabeled text. Supplying the original
source verifies text, index, and timestamp preservation.

IDs are unique case-insensitively and contain 1–31 ASCII letters, digits,
underscores or hyphens. `_review_metadata` is reserved for workbook identity.
SRT indexes must be increasing, positive, and unique; durations must be positive.
WAV audio must cover all subtitle endpoints.

`flagged` selects highlighted blocks and defaults to empty; `full_review`
requests a whole-call listen and defaults to false. Include zero-flag calls.
Optional `record_id` preserves an upstream identifier; `holdout_exposed` records
whether a case has participated in model comparisons (true, false, or null).

Creation freezes source/audio/draft bytes and records unapproved decisions for
every block. Compose using the resulting `dataset.json`. The generated
`review.json` binds browser and workbook feedback to that corpus; reconciliation
applies accepted feedback to a new revision.
