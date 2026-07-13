---
pretty_name: Geometry Transformation Error Diagnosis (v6 net-affine)
license: other
license_name: license-placeholder
language:
  - en
task_categories:
  - image-text-to-text
tags:
  - geometry
  - visual-reasoning
  - vision-language
  - synthetic
  - oracle-verified
  - structured-output
  - error-diagnosis
size_categories:
  - 10K<n<100K
---

# Geometry Transformation Error Diagnosis (v6 `net-affine.1`)

Synthetic, oracle-verified vision-language data for diagnosing student mistakes
in **composed rigid transformations**. Each example is a coordinate-grid diagram
with a RED original polygon, a GREEN dashed correct image, and a BLUE student
image. A model must recover the two observable **net affine maps** (RED→GREEN and
RED→BLUE) and return a strict-JSON diagnosis.

> **License:** placeholder (`license_name: license-placeholder`). Choose and set
> a real license before making this dataset public.

This card documents the schema, generation, splits, and verification so the set
is reproducible from a fixed seed with no human labels. The base-vs-tuned model
evidence produced from it lives in
[`results/v6_final/`](results/v6_final/FINAL_RESULTS.md).

## Record counts (read this first)

Three different numbers appear in this project; they are all correct and refer
to different things:

| Quantity | Count | What it is |
| --- | --- | --- |
| **Source corpus** | **26,000 records** | The full base geometry set (`transform_diagnosis_data/`), balanced across the 8 labels. Splits: train 19,200 / val 2,400 / test 2,400 / OOD 2,000. This is the "26,000 balanced records." |
| **v6 fine-tuning curriculum** | **9,600 train + 400 val = 10,000** | The mixed `v6.net-affine.1` curriculum actually used for SFT, derived from the source corpus (seed `20260711`, mix 50% source / 20% contrastive / 15% curriculum / 15% hard). |
| **Paired evaluation sample** | **500 per cell** | Base-vs-tuned cells draw 500 IDs each from the frozen test (2,400) and OOD (2,000) splits (seed `20260709`). |

The public sample shipped in this repo ([`dataset_public/`](dataset_public/)) is
a **240-record, label-balanced** slice (30 per label) of the source data in v6
schema; the original 24-record image teaser is
[`dataset_sample_v6/`](dataset_sample_v6/).

## Schema

Each raw record (`train_v6.jsonl`) is one JSON object:

- `id`, `split`, `label`, `hint`
- `original`, `correct_image`, `student_image` — integer polygon vertices in
  corresponding order (RED, GREEN, BLUE)
- `correct_net`, `student_net` — canonical maps `{"linear": <D4 enum>, "tx": int, "ty": int}`
- `correct_transform`, `student_transform` — legacy step sequences (provenance)
- `source_id`, `source_split`, `v6_pool`, `schema_version`
- `render_path` — relative path to the rendered PNG

A net map sends `(x, y)` to `M·(x, y) + (tx, ty)`. The eight **D4 linear** enum
values are `identity`, `rot_ccw_90`, `rot_180`, `rot_ccw_270`, `reflect_x_axis`,
`reflect_y_axis`, `reflect_y_eq_x`, and `reflect_y_eq_neg_x` (see
[`transform_diagnosis/net_transform.py`](transform_diagnosis/net_transform.py)).

The eight **diagnosis labels** are `correct`,
`reflection_instead_of_rotation`, `rotation_instead_of_reflection`,
`wrong_rotation_angle`, `wrong_reflection_line`, `wrong_translation`,
`opposite_translation`, and `completely_wrong`.

### Chat / training schema

Chat files pair a user prompt with a strict-JSON assistant target. There are two
input modalities and four staged tasks:

- **Modalities:** `image` (diagram only, numbered vertices for correspondence),
  `image_coords` (diagram + exact vertices), and `coords` (vertices only,
  text-only).
- **Tasks:** `correct` (`{correct_net}`), `student` (`{student_net}`), `both`
  (`{correct_net, student_net}`), and `full`
  (`{correct_net, student_net, label, hint}`).

## Generation and verification

- **Seed-reproducible, no human labels.** Asymmetric polygons make the affine
  map identifiable. The generator injects a specified student error, then keeps
  a record only when an independent geometry oracle recovers the intended
  diagnosis.
- **Oracle-verified maps.** `transform_diagnosis.v6_format.augment_record`
  recomputes `correct_net`/`student_net` from the legacy sequences and verifies
  them against the stored GREEN/BLUE geometry and the diagnosis label before the
  record is accepted.
- **Held-out by construction.** The OOD split uses transformation-composition
  families not seen in training; test/OOD are checksummed and never loaded into
  v6 generation.

The audited contamination check on the training run reported **0 exact
train/evaluation geometry overlaps** and **0 training rows sourced from test or
OOD** (see [`results/v6_final/FINAL_RESULTS_SUMMARY.json`](results/v6_final/FINAL_RESULTS_SUMMARY.json)).

## Splits and the OOD probe

- **train / val / test** cover all eight diagnosis labels.
- **OOD** is a restricted probe with only four families present (`correct`,
  `rotation_instead_of_reflection`, `wrong_reflection_line`, `completely_wrong`)
  and held-out composition patterns. OOD numbers must not be read as universal
  out-of-distribution generalization.
- **Golden** ([`dataset_golden_v6/`](dataset_golden_v6/)) is a fresh 160-record,
  label-balanced (20 per label) held-out set that nothing was trained or evaluated
  on. It is generated by the same primitives/oracle
  ([`model/make_golden_set.py`](model/make_golden_set.py), seed `20260712`) and is
  **provably disjoint** from v6 train/val and the source train/val/test/ood on the
  canonical geometry key `[original, correct_image, student_image]` (0 overlaps, 0
  oracle mismatches). See [`model/GOLDEN_SET.md`](model/GOLDEN_SET.md).

## Reproduce the dataset

```bash
# Full v6 curriculum (runs on the cluster; see model/V6_TRANSFORM_RUNBOOK.md)
python3 model/make_v6_transform_data.py \
  --source-dir ~/transform_diagnosis_data \
  --out-dir ~/transform_diagnosis_data_v6 \
  --train-n 9600 --val-n 400 --mix 0.50,0.20,0.15,0.15 --seed 20260711

# Larger local public sample (240 balanced records, no GPU)
python3 model/build_public_sample.py
```

## Known limitation: hints disclose answers

The `full`-task `hint` fields preserve the exact hints used in training. A
deterministic audit of all 2,000 frozen tuned outputs found exact answer/map/value
disclosure in **1,919/2,000 (96.0%)**. These hints must **not** be presented as
safe Socratic tutoring output. This is a target-design caveat separate from the
canonical map and diagnosis metrics. See
[`results/overnight/HINT_SAFETY_AUDIT.md`](results/overnight/HINT_SAFETY_AUDIT.md).

## Upload

Use [`model/push_dataset_to_hf.py`](model/push_dataset_to_hf.py) (reads
`HF_TOKEN` from the environment; supports `--dry-run`). Set a real `repo_id` and
license first.
