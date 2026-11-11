---
name: compose-srt-review
description: Compose a portable static human-review website and Excel decision workbook from draft speaker-labeled SRTs and local WAV recordings, with optional highlighted blocks and whole-call review requests.
---

Use the Python environment with this repository installed, from its root. The deterministic composer is `srt_labeler/review_site.py`; it uses the maintained player/styles in `srt_labeler/review_assets/` and generates the workbook with `srt_labeler/review_workbook.py`. It makes no model requests and publishes nothing.

## Prepare

Create a JSON manifest as described in [inputs.md](inputs.md). Decide which blocks need attention using the evidence available: model disagreement, mixed voices, uncertain attribution or a deliberate spot-check. These are review flags, not corrections. Include every call and every block, including calls with no flags. Request a full listen where local checks are insufficient. Do not invent model confidence or silently promote draft labels into verified truth.

Preserve the draft labels/text/indexes/timestamps. The reviewer listens on the website and records findings in the workbook; there are no spreadsheet URLs and no transcription-repair task. Operator means the answering-service side, Caller the incoming side even when property staff called. Mixed/no-speech/unsure are review decisions, not new labels for the production labeler.

## Compose

```sh
python .agents/skills/compose-srt-review/scripts/compose.py --dataset path/to/review-dataset.json --output call-data/review-package --debug
```

The destination and adjacent `.zip` must not already exist. This prevents accidental destruction of completed review notes. A failed build can leave a partial directory; inspect it and use a fresh destination when retrying. The builder never overwrites source data or an earlier package.

The output includes grouped speaker paragraphs with stable `CALL-Bnnn` markers, highlighted review blocks, full audio and block/context playback, playback speed, previous/next highlighted blocks, copy-ID controls, mobile controls and downloadable audio. The workbook has one call sheet and every block row, with only Block ID, Decision and Notes, matching highlight colors and decision dropdowns. Static assets and links are relative. The ZIP contains the same complete folder, including workbook and source-to-review map. Serve over HTTP; file:// playback/download behavior varies by browser.

Check generated HTML against the draft, workbook row IDs against `review-site-map.json`, and ZIP entries against the output folder. Preview a flagged call and a zero-flag call, check desktop/mobile playback, context seeking, workbook/audio downloads and escaping of transcript text. Copy the whole directory to any webserver subdirectory; do not embed a deployment URL.

Return the ZIP location. Keep the generated map and original drafts as the frozen review presentation. When the workbook comes back, validate block coverage and interpret blanks according to the reviewer's stated scope. Do not change references as part of composing a review site.
