# SRT framework contracts

Commands run from the repository root with its installed Python environment.
These contracts apply to both operating skills. Detailed option syntax is in
each command's `--help`. All operations are local except `collect`, `resume`,
and `python -m srt_labeler.evaluate` without `--predictions`, which can send
source text and audio to a provider. Preparation is not collection authorization.

## Case authority

The canonical approved corpus in this checkout is
`test-data/speaker-labels/dataset.json`. Keep corpus revisions and their evidence
under `test-data/`. Canonical operator recording source audio lives in
`${HOME}/git/apartmentlines/testing-references/audio/operator-recordings/`; other
supporting source material lives in `test-data/source-materials/`. New corpora
copy the selected audio into their own revision directories as frozen evidence.
Use the [input guide](../compose-srt-review/inputs.md) to select canonical or
custom audio paths. Any CLI that introduces a default source-audio directory
must expose an optional argument to override that directory.
Write generated runs, review packages, and reports under `evaluation-artifacts/`
(`experiments/`, `reviews/`, and `reports/`). Both top-level directories are
Git-ignored and require separate backups. Packaged and frozen corpus copies are
snapshots; they do not replace the canonical corpus. Historical paths in frozen
artifacts remain unchanged when directories move.

`dataset.json` is the authoritative case set. Each corpus lives in a
new, self-contained revision directory. Source SRTs and WAVs are evidence;
per-block decisions establish expected classifications. Reference SRTs,
exclusion masks, workbooks, HTML, and reports are derived views.

The root contains `schema_version: 2`, stable `corpus_id`, integer `revision`,
`parent` content hash (null for the initial revision), `content_sha256`, `artifacts`, and
`cases`. Content identity excludes artifact locations, so moving a complete
directory does not change identity. It includes hashes, decisions, provenance,
case ordering, selection metadata, and revisions. Every artifact record has
`path` and `sha256`; validation checks its bytes.

Each case retains its stable `id` across revisions and has `revision`, `source`,
`reference`, `audio`, mandatory `hashes` for those three files, `split`,
`holdout_exposed` (true, false, or unknown/null), `flagged`, `full_review`,
`excluded`, and ordered `decisions`. A decision contains:

- `index`: original SRT block index.
- `classification`: Operator, Caller, Both speakers, No speech, or Unsure.
- `compatibility_label`: Operator/Caller prefix retained where binary SRT
  compatibility is needed; it is not a scoring target for excluded blocks.
- `state`: draft, approved, or unresolved.
- `actor`: type human/assistant/model/unknown and identity, null if unknown.
- `rationale`, `notes`, and `provenance`, including preserved evidence hashes.
- `evidence`: exact audio SHA-256 and block start/end seconds.

Only human-attributed, resolved classifications with preserved provenance may
be approved. Mechanical validation checks consistency, not whether a listener
correctly identified the voice. AI agreement never establishes human approval.
Operator means the answering-service side; Caller means the incoming side,
including property staff who called. Ambiguous speech remains unresolved.

All blocks must preserve original text, indexes and timestamps. Classification
accuracy excludes mixed/no-speech blocks; structural validation covers every
block. Draft/unresolved cases cannot be collected or scored. Review flags are
attention requests, independent of scoring exclusions.

`scripts/corpus.py create --inputs SELECTION --corpus-id ID --output DIRECTORY`
assembles draft cases from source files. The [input guide](../compose-srt-review/inputs.md)
defines the selection fields. Creation preserves original draft bytes, hashes
the evidence, and derives reference files. It refuses an existing destination.
Use `validate --dataset PATH --require-approved` before evaluation.

## Review presentation and feedback

The composer accepts a corpus. It packages `corpus/dataset.json`, supporting
files, `review.json`, `review-site-map.json`, HTML, audio, scripts/styles, and
`review-notes.xlsx`. Existing destinations and ZIPs are refused.

`review.json` pins a `base` containing corpus ID, corpus revision/content hash,
presentation `review_id`, and nullable run/evaluation IDs. Its block map retains
stable `CALL-Bnnn` IDs, case revisions, displayed labels, input/reference/audio
hashes, audio offsets, and highlights. Hashes remain in the artifacts rather
than cluttering the listening UI.

Feedback JSON contains `schema_version: 1`, `kind: srt-review-feedback`, the exact
`base`, review `session`, supplied `reviewer`, integer `edit_revision`,
`updated_at`, and one `entries` record for every block. Each entry contains
`identifier`, `case_revision`, `decision`, `notes`, and `state`.

Decision values are blank, Correct, Operator, Caller, Both speakers, No speech,
or Unsure. States are untouched, draft, and completed. Untouched records contain
no edits; completion requires an explicit decision. Viewing and blank cells do
not approve anything. Correct refers to the exact frozen label shown.

The browser uses IndexedDB with a real write/read probe. Draft keys include
corpus content, presentation, applicable run/evaluation IDs and session. Saving
status reflects completed transactions; denied storage or later write failures
leave the draft editable/exportable in memory. Competing writes are detected
atomically, with cross-tab notification when supported. Keep either version
through download or the separate-draft control; no conflict is silently merged.

Download JSON at any time, including before autosave completes. Downloads never
clear the draft. Import validates identities, coverage, fields and conflicts
before updating the editor. Import restores proposals, not corpus authority.
Browser data belongs to one origin/profile and may be evicted or cleared.
The same workflow works on localhost and ordinary static hosting; no backend,
CDN, cloud synchronization, or special extraction server is required.

XLSX retains per-call Block ID, Decision, Notes and State columns plus hidden
identity metadata. It is an alternative review exchange, not a second authority.
Use `scripts/review_feedback.py xlsx` to convert validated JSON to XLSX and `json`
to convert returned XLSX back to portable JSON for browser resume. Use
`--reviewer` only to identify an otherwise unnamed human spreadsheet reviewer.
Formula cells are rejected; exports write text as literal strings. Workbooks
must retain the identity metadata supplied by the composer.

## Reconciliation

`scripts/review_feedback.py validate` accepts JSON or XLSX and returns the
canonical feedback SHA-256. Unknown IDs, stale bases/revisions, duplicate rows,
missing coverage and invalid fields fail before corpus changes.

For `reconcile`, supply `--resolutions` JSON with the exact feedback `base`,
`feedback_sha256`, and `decisions`. Each edited block needs one record containing
`identifier`, `action` (accept/reject/defer), `actor` (human/assistant identity),
and a nonempty `rationale`. Acceptance requires a completed human review proposal
and a supplied reviewer identity. Freeform notes are never executable patches.

Reconciliation retains original feedback, presentation and resolutions in the
new corpus's provenance. Accepted edits increment the affected case revision;
every reconciliation increments the corpus revision and links its parent.
Rejected/deferred proposals are preserved without changing expectations. Unsure
remains unresolved even when its review proposal is accepted. All previous
directories remain intact. Changing application behavior is a separate operation.

## Execution, ownership and recovery

`scripts/experiment.py prepare` freezes the complete corpus, selected ordered
IDs, template, resolved preset/effective settings, environment versions,
implementation hashes, timeout and retry limits. It makes no requests. Use
`--ids`, `--split`, or `--sample N --seed N` for bounded selection. Templates must
be self-contained; external Jinja dependencies are rejected. Credentials stay
in the provider environment; credential-bearing settings are refused.

`collect` executes the shared production labeling/extraction/merging boundary
through the local CLI. Evaluation uses local inline WAV attachments, one preset
and no fallback; production transport/fallback differences remain explicit.
The request identifier is recorded, and references never enter model inputs.
Local simulation does not prove live-provider compatibility.

Each attempt has a unique identity, directory, request identifier, timestamps,
status, process identity, logs, raw response, merged SRT and usage when available.
The collector holds a POSIX advisory lock. A launch handshake prevents model
requests before child ownership is recorded. Checkpoints replace JSON atomically.
The local child also enforces its own deadline if its collector disappears.
Do not edit frozen files or move a run while collection is active.

`status` checks the collector lock and local process identity, not merely a stale
PID file. `pause` requests graceful stop after the current bounded call. It does
not claim immediate termination. `resume` checks frozen identities and recovers
finished attempts without recollecting them. A live orphan blocks restart;
remote ownership requires recovery on the original host. Recovery is currently
for Linux/POSIX local collection, matching this repository environment.

Interrupted attempts without durable completion stay failed/unknown, retaining
partial artifacts. They are not retried by ordinary resume. `--retry ID ...`
appends attempts only within the originally frozen retry limit, and never repeats
a successful case. Latest explicitly requested attempts feed scores; all-attempt
usage includes prior failures. A tool observation timeout does not prove any
collector stopped. Poll `status` and the original process handle before acting.

`scripts/matrix.py prepare` takes a JSON specification with `schema_version: 1`,
`configurations` (unique `id`, `preset`, template path relative to the spec),
positive `repetitions`, `seed`, `timeout_seconds`, and `retry_limit`. It freezes
all runs and a seeded interleaved configuration order per call, with concurrency
one and an explicit total request limit. Its collect/pause/resume/status/report
operations reuse individual run ownership and recovery. New repetitions or
expanded limits require a new explicitly scoped experiment.

## Evaluation, lineage and reporting

`experiment.json`, `results.json`, and `summary.json` describe one run. Results
separate collection status, structural validity and grading status. Pending
cases remain visible; provisional primary accuracy includes their eligible
blocks with zero credit, and is labeled provisional. Complete primary accuracy
includes failed calls with zero credit. Also report successful-call coverage and
conditional accuracy. Macro accuracy weights eligible calls equally; micro
accuracy weights eligible blocks equally. This is a challenge/control set, not
a population estimate.

`rescore --origin RUN` creates a new evaluation with parent identity, originating
metadata and output hashes. Source and audio must match the original ordered
cases. Reference/mask corrections can rescore saved outputs; changed system
inputs require recollection. No provider request occurs during rescoring.

`compare` requires matching case revisions, hashes, masks and scoring policy/implementation,
complete results, and expected case IDs. It reports sample sizes, accuracy delta,
paired corrected/regressed blocks where both calls succeeded, status changes,
configuration differences and usage coverage. To compare prompts, keep other
settings fixed. Matrix reports preserve every repetition and paired comparison;
run-to-run ranges are not confidence intervals and repeated calls are not
independent samples. Provider aliases can still change externally.

Timing is local CLI wall time including startup. Recovered wall-clock timing is
labeled; interrupted unknown timing is omitted rather than inferred. Rescore
timing measures scoring, not model latency. Provider-reported input/output/total,
audio, cached and reasoning token fields overlap; never add breakdowns to totals.
Missing fields remain null. Each metric reports a complete total only with full
coverage, alongside reported subtotal and missing/reported observation counts.
Invisible SDK retries and unreported charges remain unknown. Token counts are
not an invoice or a cross-provider cost estimate.

Diagnose from evidence: ambiguous audio, unsupported reference, changed source,
adapter drift, evaluator error, application defect and model limitations are
different causes. Audit failures, changed grades, uncertainty and a documented
seeded pass sample against full outputs/audio before recommending prompt changes.
Document which cases were exposed so later holdout claims remain truthful.

## Acceptance

Run the offline pytest suite. Focused fixtures cover corpus authority, feedback
round trips, failed attempts, recovery and rescore lineage. The composer includes
`scripts/check-browser.js`, runnable through Playwright CLI `run-code` on a
localhost site index in isolated browser contexts; it checks persistence,
latest-state export, cross-tab conflict, import recovery, storage failures,
audio controls and mobile layout. It never revises the corpus.

Installation and development checks stay in repository documentation and coding skills.
