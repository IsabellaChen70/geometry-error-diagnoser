# v6 public sample (larger, balanced)

A larger, label-balanced reviewer sample of the final `v6.net-affine.1`
transform-diagnosis data. It is built deterministically from the local source
records by [`model/build_public_sample.py`](../model/build_public_sample.py)
through the production `transform_diagnosis.v6_format` helpers, so every
canonical map and target here is re-derived and verified, not hand-edited.

For the full dataset schema, provenance, splits, and Hugging Face upload
instructions, see the canonical [`DATASET_CARD.md`](../DATASET_CARD.md).

## What is here

- `train_v6.jsonl` — **240 records, 30 per diagnosis label** (all 8 labels).
  Full v6 schema: canonical `correct_net`/`student_net`, exact polygon vertices,
  `label`, `hint`, and `source_id`/`source_split` provenance.
- `train_v6_coords_{correct,student,both,full}_chat.jsonl` — **text-only** chat
  for all 240 records across the four staged tasks (960 rows each pass). These
  are fully self-contained and need no image files.
- `images/` — a **24-PNG visual subset** (3 per label), byte-for-byte copies of
  source renders, so the multimodal format is inspectable without shipping the
  full render set.
- `train_v6_image_coords_full_chat.jsonl` — 24 multimodal `full`-task rows whose
  image paths resolve into `images/`.
- `manifest_public.json` — counts, per-label distribution, SHA-256 checksums,
  and verification totals.

## Why coordinates-only for the large set

The `coords` input mode is text-only (the exact RED/GREEN/BLUE vertices are in
the prompt), so a 240-record coordinates sample is valid and reviewable without
bundling 240 images. The 24-image subset shows the rendered diagram format that
the `image` and `image_coords` arms consume. This keeps the committed sample
small while still being far larger than a 24-record teaser.

## Relationship to the other samples

- [`../dataset_sample_v6/`](../dataset_sample_v6) — the original 24-record
  image sample (one PNG per record).
- [`../dataset_sample/`](../dataset_sample) — the preserved legacy/source-format
  sample with pre-v6 free-text step targets.

## Known hint-disclosure limitation

The `full` targets deliberately preserve the exact hints used in training. Many
disclose the operation, map parameters, or translation values and must not be
presented as safe Socratic tutoring output. See
[`../results/overnight/HINT_SAFETY_AUDIT.md`](../results/overnight/HINT_SAFETY_AUDIT.md).
