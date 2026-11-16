---
name: evaluate-srt-presets
description: Explicitly invoked SRT corpus audits, human-feedback reconciliation, recoverable preset experiments, saved-output rescoring, and regression investigations.
---

# Evaluate SRT presets

Use only when the developer explicitly invokes `$evaluate-srt-presets`.
Work from the repository root in its installed Python environment. Read
[contracts.md](contracts.md) for the applicable schema, provenance, recovery,
metrics, and review-exchange contract. Ordinary operation needs no builder context.

## Choose the requested operation

Use the existing application modules in place: `srt_labeler/eval_dataset.py`
owns case validation and revisions; `srt_labeler/evaluate.py` owns execution and
scoring; `srt_labeler/local.py` shares the production labeling/merging boundary.
New testing commands are in this skill's `scripts/` directory:

```sh
python .agents/skills/evaluate-srt-presets/scripts/corpus.py --help
python .agents/skills/evaluate-srt-presets/scripts/review_feedback.py --help
python .agents/skills/evaluate-srt-presets/scripts/experiment.py --help
python .agents/skills/evaluate-srt-presets/scripts/matrix.py --help
```

- **Create/audit:** use `corpus.py create` to assemble draft cases and
  `corpus.py validate --dataset PATH --require-approved` before evaluation.
  Inspect audio locators and human provenance, not just schema success.
- **Feedback:** validate returned JSON/XLSX against its `review.json` and corpus.
  Interpret comments as proposals, record accept/reject/defer decisions with
  actor and rationale, then reconcile to a new revision. Unknown reviewer identity
  stays unknown; an assistant cannot invent human approval. See the feedback
  contract for the resolutions JSON and exact SHA-256 binding.
- **Compare:** prepare bounded, frozen runs first; collect only within existing
  live-call authorization. Use matrix preparation for explicit repetitions and
  interleaving. Report regressions, failures, coverage and variation alongside gains.
- **Investigate/rescore:** inspect full saved inputs/outputs and evidence.
  Corrected references can rescore old outputs with parent lineage. Changed inputs
  require recollection before claiming new behavioral results.

## Execution and recovery

```sh
python .agents/skills/evaluate-srt-presets/scripts/experiment.py prepare \
  --dataset test-data/speaker-labels/dataset.json \
  --preset PRESET --template path/to/prompt.md --split development \
  --run evaluation-artifacts/experiments/baseline
python .agents/skills/evaluate-srt-presets/scripts/experiment.py collect \
  --run evaluation-artifacts/experiments/baseline
python .agents/skills/evaluate-srt-presets/scripts/experiment.py status \
  --run evaluation-artifacts/experiments/baseline
```

Preparation, audits, feedback and framework changes do not authorize provider calls,
expanded experiments, prompt changes or deployment. Honor authorization already
given; do not repeatedly ask for it. Preserve the production template and presets.

Pause is graceful: the in-flight bounded request finishes before collection stops.
Resume checks ownership and frozen data; it does not repeat interrupted attempts
without explicit retry selection within the original limit. A tool timeout does
not mean the collector stopped. Inspect status and the original process handle.
Do not edit frozen run files to force continuation after configuration/code drift.

## Interpretation and handoff

Mechanical structural/label scoring is separate from human listening judgments.
Audit uncertain cases, failures, changed grades and a documented seeded sample
of passes. Diagnose evidence, reference, adapter, evaluator and application issues
before changing prompts. Never optimize prompts using case IDs or reference answers.

Read `experiment.json`, `results.json`, and `summary.json`; matrix reporting adds
`report.json` and `report.md`. Primary accuracy includes failures; conditional
accuracy needs coverage. Pending results are provisional. Compare matching cases
under a common scoring policy. Provider aliases can drift despite frozen settings.

Report artifact paths and identities, selected cases, repetitions, failures,
exclusions, unresolved reviews and provenance limitations. Token counts are
provider-reported with explicit missingness; overlapping breakdowns are not extra
tokens and do not establish dollar savings. Distinguish offline fixtures from
authorized live validation. Preserve every earlier run and evaluation.
