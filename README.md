# Geometry Transformation Error Diagnoser

A task-specific Qwen3-VL-4B system that diagnoses student mistakes in composed
rigid transformations. Each coordinate-grid diagram shows a RED original polygon,
the GREEN dashed correct image, and the BLUE student image. The model recovers
the two observable net affine maps, RED to GREEN and RED to BLUE, and returns a
strict JSON diagnosis with a hint. A deterministic geometry oracle scores every
prediction with no LLM judge, and the headline metric is exact recovery of both
maps. On paired held-out cases the untuned base scored 0/500 on both maps in
every cell, including with coordinates; the tuned image+coordinates model reached
98.6% on test and 99.4% on the restricted OOD split. The maps and diagnosis are
strong; the frozen tuned hints have a known answer-disclosure failure, described
below.

The central claim is falsifiable: on paired held-out examples an untuned model
should not recover the canonical maps just because coordinates are supplied, and
training on canonical net targets should produce a large paired gain. It fails if
the base coordinate arm solves the maps, or if the trained arm does not reproduce
that gain.

## Headline results (base vs tuned)

Every cell uses the same 500 IDs, paired within each split, seed `20260709`;
every saved response parsed. Pairing on identical IDs isolates the effect of
training instead of comparing independent samples. **"Both maps" is the exact
two-map recovery rate and the headline metric.** The untuned base scored **0/500
on both maps in every cell, including with coordinates**, so tuning produced the
entire gain.

| Split | Modality | Model | Correct map | Student map | Both maps | Label acc | Parse |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | image+coords | base | 0.0% | 0.4% | 0.0% | 29.8% | 100% |
| test | image+coords | **tuned (hero)** | **98.6%** | 99.6% | **98.6%** | 99.6% | 100% |
| test | image | base | 0.0% | 0.0% | 0.0% | 11.8% | 100% |
| test | image | **tuned (ablation)** | 46.2% | 43.8% | **38.4%** | 85.6% | 100% |
| OOD | image+coords | base | 0.0% | 3.8% | 0.0% | 22.2% | 100% |
| OOD | image+coords | **tuned (hero)** | **100.0%** | 99.4% | **99.4%** | 99.8% | 100% |
| OOD | image | base | 0.0% | 7.4% | 0.0% | 10.0% | 100% |
| OOD | image | **tuned (ablation)** | 82.4% | 69.4% | **67.6%** | 85.2% | 100% |

Counts (of 500): hero test correct/student/both = 493/498/493; hero OOD =
500/497/497; ablation test = 231/219/192; ablation OOD = 412/347/338. Base
correct-map and both-map counts were exactly 0/500 in all four cells. The table's
`Label acc` is the direct predicted label; the label derived from the two
predicted maps tracks it closely (hero test 498/500, hero OOD 500/500, ablation
test 428/500, ablation OOD 426/500). Full counts, 95% Wilson intervals, and
paired base→tuned deltas:
[`results/v6_final/FINAL_RESULTS.md`](results/v6_final/FINAL_RESULTS.md) and
[`FINAL_RESULTS_SUMMARY.json`](results/v6_final/FINAL_RESULTS_SUMMARY.json). The
raw frozen predictions, aggregates, and eight independent audits live in
[`results/v6_final/`](results/v6_final/).

### Error analysis (our own model)

From [`audit_v6_paired_summary.json`](results/v6_final/audit_v6_paired_summary.json),
the per-cell `audit_records_*.json`, and the error-analysis section of
[`FINAL_RESULTS.md`](results/v6_final/FINAL_RESULTS.md):

- **Composition collapses without coordinates; single-step diagnosis does not.**
  Image-only exact two-map recovery is only **38.4% test / 67.6% OOD**, yet the
  diagnosis label stays high (**85.6% / 85.2%**). Adding coordinates raises the
  geometry to **98.6% / 99.4%**. The hard part is composing the multi-step
  affine map from pixels, not naming the error family.
- **Label accuracy masks geometry errors.** On the hero (image+coords) test cell
  only 7/500 exact-pair failures remain, and **5/7 still return the correct
  diagnosis label**. In 5/7 the student map is exact but the correct map is off
  (3 keep the right label, 2 flip a reflection-family truth to
  `completely_wrong`); the other 2/7 miss both maps on true-label-`correct` cases
  yet still return `correct` because the two wrong maps agree. Exact-map scoring
  is deliberately stricter than label scoring.
- **The image-only arm confuses translation sign.** The largest label confusion
  is `opposite_translation → wrong_translation` (**25/72 label errors**);
  `opposite_translation` recall is 20/47 (**42.6%**) versus 100% for
  `wrong_translation`. Exact two-map recovery by label ranges from 5/64
  (**7.8%**, `completely_wrong`) to 50/76 (**65.8%**, `wrong_rotation_angle`).
- On image-only OOD, 107 of the 162 exact-pair failures come from the
  `completely_wrong` family.

## Task and schema

The `v6.net-affine.1` task represents each composed transformation by its unique
observable map rather than a sequence of primitive steps. This is an
identifiability choice: many different ordered step recipes can produce the same
before/after polygons, so an ordered decomposition is not recoverable from the
image, but the net affine map is. A net map sends `(x, y)` to `M(x, y) + (tx, ty)`:

```json
{"linear":"rot_ccw_90","tx":2,"ty":-3}
```

Inputs are either:

- **image only**, using numbered vertices to establish RED/GREEN/BLUE
  correspondence; or
- **image + coordinates**, adding the exact corresponding vertices for all
  three polygons.

The full-task output is exactly one JSON object:

```json
{
  "correct_net": {"linear": "<D4 enum>", "tx": 0, "ty": 0},
  "student_net": {"linear": "<D4 enum>", "tx": 0, "ty": 0},
  "label": "<diagnosis label>",
  "hint": "<short hint>"
}
```

`correct_net` maps RED to GREEN and `student_net` maps RED to BLUE. The eight D4
linear values are defined in
[`transform_diagnosis/net_transform.py`](transform_diagnosis/net_transform.py);
the eight diagnosis labels are `correct`, `reflection_instead_of_rotation`,
`rotation_instead_of_reflection`, `wrong_rotation_angle`,
`wrong_reflection_line`, `wrong_translation`, `opposite_translation`, and
`completely_wrong`. The label follows deterministically from the two maps, so it
can be recomputed and checked independently of the model's own label token.

The v6 prompt requested a short Socratic hint, but that wording describes the
target, not the observed safety behavior. The prompt omitted the legacy explicit
prohibition on coordinates, and the final hints routinely disclose exact answer
details (see [Hint safety](#hint-safety-and-the-secondary-judge)).

## Data and provenance

The dataset is generated from a fixed seed with no human labels. Asymmetric
polygons make the affine map uniquely identifiable; the generator injects a
specified student error and keeps a record only when an independent geometry
oracle recovers the intended diagnosis. Three record counts appear in this
project, all correct and referring to different things; see
[`DATASET_CARD.md`](DATASET_CARD.md) for the full reconciliation:

- **Source corpus: 26,000 balanced records** (`transform_diagnosis_data/`, kept
  on disk, not versioned), split train 19,200 / val 2,400 / test 2,400 / OOD
  2,000, ~3,000–3,500 per label.
- **v6 fine-tuning curriculum: 9,600 train + 400 val = 10,000 rows**, derived
  from the source (seed `20260711`, mix 50% source / 20% contrastive / 15%
  curriculum / 15% hard).
- **Paired evaluation: 500 IDs per cell** from the frozen test/OOD splits.

Generation ([`model/make_v6_transform_data.py`](model/make_v6_transform_data.py))
writes a separate v6 tree and manifest, verifies every task target, and
checksums the source before and after; the frozen test and OOD splits are
checksummed but never loaded into generation. Reproduce the exact v6 curriculum
(runs on the cluster, see the [`v6 runbook`](model/V6_TRANSFORM_RUNBOOK.md)):

```bash
python3 model/make_v6_transform_data.py \
  --source-dir ~/transform_diagnosis_data \
  --out-dir ~/transform_diagnosis_data_v6 \
  --train-n 9600 --val-n 400 --mix 0.50,0.20,0.15,0.15 --seed 20260711
```

Three samples ship in the repo. The larger [`dataset_public/`](dataset_public/)
holds **240 label-balanced records** (30 per label) in the v6 raw and text chat
schemas plus a 24-image visual subset, rebuildable with
[`model/build_public_sample.py`](model/build_public_sample.py). The compact
[`dataset_sample_v6/`](dataset_sample_v6/) holds 24 records with one image each
and a deterministic [`zip archive`](dataset_sample_v6.zip). Both are generated
from, but are not claimed to be a byte-for-byte sample of, the 9,600-row mixed
curriculum. The sibling [`dataset_sample/`](dataset_sample/) is preserved as the
legacy/source-format sample with pre-v6 free-text step targets. The Hugging Face
upload is prepared in
[`model/push_dataset_to_hf.py`](model/push_dataset_to_hf.py) (`--dry-run` lists
what would ship; token read from `HF_TOKEN`).

## Model and training

Both tuned arms start from
`unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`, and each modality has its own
LoRA adapter. Training runs four sequential stages, correct map, student map,
both maps, then full diagnosis, with deterministic rehearsal of earlier stages.
The **tuned image + coordinates arm is the hero model**; the separately trained
**tuned image-only arm is the harder ablation**, not an inference-time toggle on
the same adapter.

## Frontier comparison: zero-shot Claude Opus 4.8

The comparison uses the same v6 schema, paired test IDs, and seed. Opus is
zero-shot; Qwen is trained on the task-specific taxonomy.

- **Image only, paired `n=150`:** tuned Qwen versus Opus was **66/150 vs
  71/150** correct-map, **59/150 vs 83/150** student-map, and **53/150 vs
  50/150** both-map. Qwen led direct labels **126/150 vs 59/150** and derived
  labels **126/150 vs 68/150**; parse success was **150/150 vs 147/150**.
  No difference was statistically detected for correct-map or both-map
  recovery, which does not establish equivalence; Opus was stronger on
  student-map recovery.
- **Image + coordinates, paired `n=50`:** both models reached **50/50** on
  correct-map, student-map, both-map, and derived-label metrics. Qwen's direct
  label count was **50/50** versus Opus **40/50**. This ceiling and sample size
  prevent a broad superiority claim.

The authoritative fair-frontier prediction and audit files remain under
`/home/ikchen` on ORCD and were not copied into this repository, so the local
checkout cannot repeat row-level Opus verification. See
[`FRONTIER_COMPARISON.md`](results/v6_final/FRONTIER_COMPARISON.md).

## Hint safety and the secondary judge

Deterministic map and label scoring remains authoritative. A secondary blinded
paired judge used TFY `gpt-5.6-sol` on 100 image+coordinates test IDs, with
candidate identity balanced 50/50.

- **Overall tuned/base/tie preference was 100/0/0.** The judge was instructed
  to respect the supplied deterministic correctness flags, so this is not an
  independent second exact-map score.
- **Hint-only tuned/base/tie preference was 25/75/0.** Tuned versus base mean
  scores were **2.89 vs 1.80** for pedagogical usefulness, **4.98 vs 1.91** for
  operation-family relevance, and **4.12 vs 2.70** for clarity/actionability.
  The same judge flagged forbidden answer or coordinate disclosure in
  **100/100 tuned hints** versus **1/100 base hints**.
- A deterministic audit of all 2,000 frozen tuned outputs found exact
  answer/map/value disclosure in **1,919/2,000 (96.0%)** and a conservative
  safe/useful rate of **28/2,000 (1.4%)**. The stored hint metrics agreed with
  the independent recomputation.

The tuned hints should therefore not be described as safely Socratic or
non-leaking. This is a specification and target-design caveat: the actual v6
prompt requested a short Socratic hint but did not explicitly forbid coordinates.
It does not erase the separately scored map and diagnosis result, and it is not a
direct violation of that exact prompt. See
[`LLM_JUDGE_VALIDATION.md`](results/overnight/LLM_JUDGE_VALIDATION.md),
[`LLM_JUDGE_REPORT.md`](results/overnight/LLM_JUDGE_REPORT.md), and
[`HINT_SAFETY_AUDIT.md`](results/overnight/HINT_SAFETY_AUDIT.md).

## Evaluation credibility

- All 4,000 saved base/tuned rows parse, contain 500 unique IDs per cell, and
  preserve identical within-split ID order across arms.
- Local recomputation matched saved aggregates, confusion matrices, per-label
  recall, Wilson intervals, and paired summaries. Eight independent geometry
  audits reported zero evaluator-disagreement rows.
- The overnight check recorded **194 passing geometry-oracle tests** and **30 passing
  model tests**. Independent offline rescoring reproduced all 20 requested
  tuned counts/rates with zero stored-to-rescored delta.
- The saved train/evaluation contamination audit reports **0 exact geometry
  overlaps** and **0 training rows sourced from test or OOD**. The 9,600-row
  training file was not in the frozen local download, so that fingerprint
  computation was not rerun locally.

The audit trail and independently rescored artifacts are documented in
[`results/overnight/OVERNIGHT_REPORT.md`](results/overnight/OVERNIGHT_REPORT.md).

## Reproduce locally

Run from the repository root. The test and offline-rescore paths need no GPU or
API key, and the rescore reproduces the saved aggregates with zero delta:

```bash
python3 -m pytest transform_diagnosis/ -q
python3 -m pytest model/ -q

VERIFY_DIR="$(mktemp -d)"
cp results/v6_final/records_v6_4b_image_coords_test.jsonl \
   results/v6_final/records_v6_4b_image_coords_ood.jsonl \
   results/v6_final/records_v6_4b_image_test.jsonl \
   results/v6_final/records_v6_4b_image_ood.jsonl \
   "$VERIFY_DIR/"
python3 model/rescore_records.py "$VERIFY_DIR"/records_*.jsonl --task full
```

The rescorer writes beside the temporary copies. It uses local split JSONL when
available and otherwise deterministically rebuilds the oracle records. To
generate a fresh small dataset and renders:

```bash
DATA_ROOT="$(mktemp -d)"
python3 -m transform_diagnosis --seed 0 --n 400 --out "$DATA_ROOT/generated"
```

Producing fresh base/tuned predictions needs the GPU adapters on the cluster via
[`model/eval_transform.py`](model/eval_transform.py). GPU training and final
model evaluation run on ORCD; use
[`model/V6_TRANSFORM_RUNBOOK.md`](model/V6_TRANSFORM_RUNBOOK.md) for the exact
data-generation, staged-training, evaluation, and SLURM commands.

## Brainlift site

The React/Vite app is the project's research brainlift, not a hosted inference
endpoint. Its scripts are defined in [`package.json`](package.json):

```bash
npm ci
npm run dev
```

For a production build and local preview:

```bash
npm run build
npm run preview
```

## Repository map

- [`transform_diagnosis/`](transform_diagnosis/): geometry oracle, dataset
  generator, v6 schema, scorer, and tests.
- [`model/`](model/): v6 data, QLoRA training, evaluation, rescoring, and audit
  entry points.
- [`results/v6_final/`](results/v6_final/): frozen base/tuned predictions,
  aggregates, audits, and final reports.
- [`results/overnight/`](results/overnight/): independent rescore, credibility,
  and submission-readiness evidence.
- [`DATASET_CARD.md`](DATASET_CARD.md): Hugging Face dataset card (schema,
  labels, splits, verification, and record-count reconciliation).
- [`dataset_public/`](dataset_public/): larger 240-record label-balanced v6
  sample, built by [`model/build_public_sample.py`](model/build_public_sample.py).
- [`dataset_sample_v6/`](dataset_sample_v6/) and
  [`dataset_sample_v6.zip`](dataset_sample_v6.zip): compact 24-record final-v6
  canonical-net image sample.
- [`dataset_sample/`](dataset_sample/): preserved legacy/source-format sample.
- [`SHIP_CHECKLIST.md`](SHIP_CHECKLIST.md): remaining manual publish steps
  (commit, Hugging Face upload, optional retrain).
- [`src/`](src/) and [`brainlift.md`](brainlift.md): interactive research
  brainlift and source notes.

## Limitations and publication status

OOD contains only four diagnosis families (`correct`,
`rotation_instead_of_reflection`, `wrong_reflection_line`, and
`completely_wrong`); its higher image-only result is not evidence of universal
OOD generalization. Each rate comes from one 500-case sample and one adapter, and
the modality comparison uses separately trained adapters. The base-vs-tuned
comparison is single-seed and single-run per cell: Wilson intervals cover
finite-sample noise but not training-seed, checkpoint, or run-to-run
variability.

The final hints are not safe tutoring outputs under the conservative disclosure
rubric: 96.0% disclosed exact answer/map/value information, and the safe/useful
rate was 1.4%. This limitation is separate from the canonical map and diagnosis
metrics and must accompany any claim about model behavior.

The repository contains frozen predictions, a 240-record and a 24-record v6
sample, and the preserved legacy source sample, not the final adapter weights or
the full v6 training set. The Hugging Face upload is **prepared but not executed**
here ([`DATASET_CARD.md`](DATASET_CARD.md) plus
[`model/push_dataset_to_hf.py`](model/push_dataset_to_hf.py), which needs a real
`repo_id` and `HF_TOKEN`). No public dataset/model hosting, hosted-inference
demo, or video URL is verified in this checkout; the remaining publish steps are
listed in [`SHIP_CHECKLIST.md`](SHIP_CHECKLIST.md).
