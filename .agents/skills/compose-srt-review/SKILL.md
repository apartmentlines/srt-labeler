---
name: compose-srt-review
description: Explicitly invoked creation of portable static SRT/audio listening-review packages with in-page decisions, recoverable drafts, JSON feedback, and Excel exchange.
---

# Compose SRT review

Use only when the developer explicitly invokes `$compose-srt-review`.
Work from the repository root in its installed Python environment. The existing
composer is `srt_labeler/review_site.py`, with maintained browser assets in
`srt_labeler/review_assets/` and XLSX exchange in `srt_labeler/review_workbook.py`.
Composition makes no provider requests and publishes nothing.

Read [inputs.md](inputs.md) to prepare inputs and the shared
[framework contracts](../evaluate-srt-presets/contracts.md) for identity,
feedback, storage recovery and reconciliation.

## Prepare and compose

Include every call and block, even zero-flag calls. Use evidence such as model
disagreement, mixed voices, uncertainty or a deliberate spot-check to flag review
targets. Flags request attention; they are not corrections or model confidence.
Request a full listen when local checks are insufficient. Preserve the draft's
labels, text, indexes and timestamps and retain original draft bytes.

Operator means the answering-service side; Caller means the incoming side,
including property staff who called. Mixed/no-speech/unsure are review classifications,
not new production SRT labels. This workflow does not repair transcription.

```sh
python .agents/skills/compose-srt-review/scripts/compose.py \
  --dataset test-data/speaker-labels/dataset.json \
  --output evaluation-artifacts/reviews/new-review --debug
```

The directory and adjacent ZIP must be new. A failed build may leave partial
artifacts; inspect them and use a fresh destination. Rendering preserves the
corpus's approval state.

## Review and recovery

Reviewers select a block, listen with context, enter a decision/notes beside the
audio, and explicitly mark the block reviewed. Viewing a block does not approve
it. Untouched, draft and completed are distinct. Unknown decisions remain uncertain.
The whole transcript, audio, highlights and original IDs remain available.
Keep reviewer identity, feedback download, and file recovery in the compact pinned
header. Per-block decisions and notes stay beside the audio; spreadsheet downloads
do not belong in the listening interface.

Browser drafts autosave in IndexedDB when available. Only successful transactions
show Saved in this browser. Failed storage leaves edits in memory and keeps
download available. Export JSON at any time, including unfinished work. Import
that file to resume elsewhere; downloads never clear drafts. Stale identities,
duplicate entries and conflicting edits must not overwrite existing work. Separate
draft sessions preserve competing versions.

The blank XLSX is an alternative exchange with protected identity metadata.
Completed JSON can also be converted to XLSX, and returned XLSX to JSON, using
the documented `review_feedback.py` commands. Browser draft restoration and
spreadsheet editing never revise the authoritative corpus. Corpus reconciliation
belongs to the evaluation skill and requires explicit dispositions.

## Verify and hand off

Check rendered labels/text and block IDs against the source and `review.json`;
check workbook coverage and ZIP bytes against the folder. Verify a flagged call
and a zero-flag call where present, context playback, downloads, escaping, review
completion and draft export/import. For browser acceptance, serve on localhost
and run `scripts/check-browser.js` through Playwright CLI `run-code` from the
site index; it uses isolated browser contexts and local fixture feedback.

Serve the completed directory on ordinary static HTTP hosting or localhost.
Keep bundled relative assets together. No backend, CDN, deployment URL, or
special feedback-extraction service is required. Direct file:// behavior is not
part of the delivery contract.

Return the ZIP and directory locations, corpus/presentation identities, verified
behavior and limitations. Preserve the original package and drafts. Do not
publish the package or change approved references as part of composition.
