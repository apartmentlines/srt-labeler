---
name: evaluate-srt-presets
description: Evaluate SRT speaker-labeling presets against human-reviewed audio/SRT cases, compare prompt versions, and diagnose labeling or structural regressions using the repository local CLI.
---

Use the Python environment with this repository installed. Run commands from the repository root. Implementation lives in `srt_labeler/eval_dataset.py` and `srt_labeler/evaluate.py`; the skill script is a thin entry point. Inputs and outputs remain local except audio and source text sent to the selected model provider during a live run. Rescoring saved predictions makes no provider requests.

## Prepare the reference set

Use the schema in [dataset.md](dataset.md). Preserve original indexes, timestamps and text. Never send reference labels to the model: `source` is unlabeled; `reference` is only for scoring. Do not guess ambiguous labels to enlarge the scoring denominator. Mixed-speaker/no-speech blocks keep a binary compatibility prefix but must be listed in `excluded`. Their content and structure are still validated.

The reviewed November 6 dataset can be imported without rewriting any SRTs:

```sh
python -m srt_labeler.eval_dataset --manifest call-data/prompt-test-set-2026-11-06/manifest.csv --reviews call-data/prompt-test-set-2026-11-06/reference-review.csv --root . --output call-data/evaluation-dataset.json
```

The importer requires complete human-verified metadata and pins source, reference and audio hashes. That dataset has 24 calls, 783 blocks, 765 eligible blocks and 18 exclusions. Those counts describe this batch, not a requirement for other datasets. Here, blanks in the completed review were explicitly confirmed correct after a full listen. Do not infer that interpretation for another review.

## Run and compare

Choose an explicit preset and exact LWE template file; preserve the original production template. Each experiment gets a new output directory. The harness runs one bounded local CLI process per call sequentially, preserves `response.txt`, `final.srt` and logs, and continues after a failed case. It does not retry or silently switch models. `--debug` enables LWE debug output. Exit 1 indicates at least one failed call or invalid inputs.

```sh
python .agents/skills/evaluate-srt-presets/scripts/evaluate.py --dataset call-data/evaluation-dataset.json --preset PRESET --template path/to/prompt.md --split development --output call-data/experiments/baseline --debug
python .agents/skills/evaluate-srt-presets/scripts/evaluate.py --dataset call-data/evaluation-dataset.json --preset PRESET --template path/to/revised-prompt.md --split development --output call-data/experiments/revised --debug
python -m srt_labeler.evaluate compare call-data/experiments/baseline call-data/experiments/revised
```

Omit `--split` to run the complete set. `--timeout` controls the per-call limit (600 seconds by default). To score existing results, pass `--predictions DIRECTORY` containing `ID/final.srt`; preset/template then describe the claimed origin and are not independently verified. Rescore elapsed time is scoring time, not model latency.

Read `summary.json`, `results.json`, and `experiment.json`. Accuracy including failures keeps eligible blocks from failed calls in the denominator with zero credit. Also report successful-call accuracy and coverage; never hide failures behind that conditional accuracy. Per-call results include mismatching indexes and a confusion matrix. Structural corruption fails the whole call, including corruption on an excluded block. The macro average weights calls equally; the micro accuracy weights eligible blocks equally. This is a chosen challenge/control set, not a population accuracy estimate. Runtime is wall time including startup. Token counts come only from provider-reported usage; prices and dollar cost are not inferred.

The experiment snapshots the actual template, input/reference hashes, exclusion policy and resolved preset identity/hash. Compare requires identical ordered cases, hashes and masks. Keep preset settings, split and model fixed when attributing differences to a prompt. A provider alias can change behind an unchanged preset hash; document provider-side versions when known.

Inspect errors and raw outputs, develop a specific prompt hypothesis, change the template, and rerun under a new experiment name. Use development cases for iteration and the held-out validation split for a final check. Repeat promising runs when model variance could explain a small improvement. Do not bake case IDs or reference answers into prompts. Summarize regressions as well as gains; retain every experiment rather than overwriting the baseline.

## Token usage

The local CLI saves `usage.json` before transcript extraction, including when a returned response later fails formatting or merging. Request-scoped LangChain callbacks capture usage before LWE converts audio responses to plain text. A returned-message fallback supports backends without callbacks, without counting the same response twice. No LWE/plugin patch is required.

Per-call results retain raw usage metadata and normalized input/output/total tokens, input/output audio tokens, cached input tokens, and reasoning tokens when supplied. Breakdown counts overlap totals: never add audio, cached or reasoning counts on top. Missing fields are `null`, not zero; totals are not estimated from text or audio duration.

Each metric in `summary.json` has a complete `total` only when every observation reports it, plus `reported_subtotal`, `reported_observations`, and `missing_observations`. Failed calls still contribute reported usage. Missing artifacts, unsupported provider reporting and interrupted requests produce unknown observations. Callback coverage cannot reveal invisible provider retries or their unreported charges; this is reported usage, not an invoice.

Rescoring copies existing `ID/usage.json` when available and labels it saved usage, with no new model requests. Old predictions without usage remain unavailable. `compare` shows usage summaries for both runs; compare their coverage before comparing totals. Audio tokenization and reasoning accounting differ by provider, so equal token counts need not mean equal cost.
