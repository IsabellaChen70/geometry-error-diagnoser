# v6 final base-vs-tuned results

## Result in one sentence

On the same 500 examples per split and modality, the untuned 4B base model
produced **0/500 exact correct-net matches and 0/500 exact two-net matches in
all four base cells, including coordinate input**; after v6 structured training
on canonical net-affine targets, the tuned image+coordinates arm reached
**493/500 (98.6%) exact two-net matches on test** and **497/500 (99.4%) on the
restricted OOD split**.

This supports a falsifiable behavior-from-data claim: coordinates were a useful
representation after training, but coordinates alone did not supply the
behavior. A repeat with the fixed paired evaluation would falsify this claim if
the base coordinate arm solved the canonical maps or the trained arm failed to
show the observed paired gains.

## Evaluation design

- Model family: `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`.
- Comparison: untuned base versus the final modality-specific v6 LoRA adapters.
- Task/schema: `full`, `v6.net-affine.1`. The model predicts canonical affine
  maps from RED to GREEN (`correct_net`) and RED to BLUE (`student_net`), plus
  an explicit diagnosis label and hint. `derived_label` is recomputed from the
  two predicted maps.
- Cells: base image, base image+coordinates, tuned image, and tuned
  image+coordinates on each of test and OOD.
- Sampling: `n=500` per cell, seed `20260709`. All four cells within a split
  contain the same 500 unique IDs in the same order. There are 500 unique test
  cases and 500 unique OOD cases, each reused across the four paired cells.
- Primary exact metrics: correct map, student map, and both maps. Diagnosis
  metrics are explicit-label and derived-label accuracy. Every saved response
  parsed successfully.
- Intervals below are two-sided 95% Wilson intervals for a binomial proportion.
  They quantify finite-sample uncertainty, not training-seed or checkpoint
  variability.

The downloaded source set is complete for the expected layout: 8 record JSONL
files, 8 aggregate result JSON files, 8 independent-audit JSON files, one
paired-audit JSON, and one audit log (26 regular files; an rsync count of 27
also includes the directory entry).

## Headline findings

### Hero arm: tuned image + coordinates

- **Test:** correct map **493/500, 98.6%** (95% CI 97.1–99.3);
  student map **498/500, 99.6%** (98.6–99.9); both maps **493/500,
  98.6%** (97.1–99.3); explicit and derived labels each **498/500,
  99.6%** (98.6–99.9).
- **OOD:** correct map **500/500, 100.0%** (99.2–100.0); student map
  and both maps each **497/500, 99.4%** (98.3–99.8); explicit label
  **499/500, 99.8%** (98.9–100.0); derived label **500/500, 100.0%**
  (99.2–100.0).

The corresponding base image+coordinates arm had **0/500 correct maps and
0/500 two-map matches on both splits**. It recovered only 2/500 student maps on
test and 19/500 on OOD. Its explicit-label accuracy was 149/500 (29.8%) on test
and 111/500 (22.2%) on OOD. Coordinate input therefore did not solve the
untuned model.

### Harder ablation: tuned image only

- **Test:** correct map **231/500, 46.2%** (41.9–50.6); student map
  **219/500, 43.8%** (39.5–48.2); both maps **192/500, 38.4%**
  (34.2–42.7); explicit and derived labels each **428/500, 85.6%**
  (82.3–88.4).
- **OOD:** correct map **412/500, 82.4%** (78.8–85.5); student map
  **347/500, 69.4%** (65.2–73.3); both maps **338/500, 67.6%**
  (63.4–71.6); explicit and derived labels each **426/500, 85.2%**
  (81.8–88.0).

Image-only tuning clearly taught nontrivial behavior, but exact geometry
remained much harder than diagnosis-label prediction. The tuned
image+coordinates arm exceeded tuned image-only exact two-map accuracy by
**60.2 percentage points on test** and **31.8 points on OOD** on paired IDs.

## Exact paired base-to-tuned deltas

Metric order in each line is **correct map / student map / both maps / explicit
label / derived label**, in percentage points:

- Image, test: **+46.2 / +43.8 / +38.4 / +73.8 / +74.0**.
- Image, OOD: **+82.4 / +62.0 / +67.6 / +75.2 / +82.6**.
- Image+coordinates, test: **+98.6 / +99.2 / +98.6 / +69.8 / +70.0**.
- Image+coordinates, OOD: **+100.0 / +95.6 / +99.4 / +77.6 / +71.2**.

These are paired differences on identical IDs, not differences between
independent samples. For example, on coordinate-test both-map correctness,
493 cases were tuned-only successes, 0 were base-only successes, and 7 failed
in both. On image-test explicit labels, there were 372 tuned-only successes and
3 base-only successes, for the net +73.8-point change.

## Audit and validity checks

Local recomputation over all 4,000 saved record rows confirmed:

- every JSON and JSONL file parses; every record file has 500 rows and 500
  unique IDs;
- aggregate counts/rates, confusion matrices, per-label recall, and all Wilson
  intervals exactly match the saved result and audit files;
- result-file ID arrays exactly match record order, and all four ID arrays are
  equal within each split;
- all 12 same-split pair summaries and their discordance counts exactly match
  direct recomputation from records;
- each of the 8 independent-audit files reports
  `evaluator_disagreement_rows=0`, and the combined log agrees.

The downloaded independent leakage audit records 9,600 training rows and 1,000
unique evaluated cases, with **0 exact train/evaluation geometry overlaps** and
**0 training rows sourced from test or OOD**. Its paired JSON and log agree.
The 9,600-row training file and original oracle files were not part of the
download, so the geometry-fingerprint leakage computation itself was not
rerun locally; this report treats that result as an independently generated
audit artifact rather than as a new local recomputation.

## Error analysis

### Remaining seven coordinate-test exact-pair failures

All seven parsed, and every one missed the canonical correct map.

- In **5/7** cases (IDs `52`, `3529`, `3929`, `7770`, `16316`), the student
  map was exact but the correct map was not. Three preserved the right explicit
  and derived diagnosis; two (`52`, `3929`) changed a reflection-family truth
  to `completely_wrong`.
- In **2/7** cases (IDs `5252`, `21634`), both maps were wrong on examples whose
  true diagnosis was `correct`; the two predicted maps nevertheless agreed,
  so both the explicit and derived labels remained `correct`.
- Consequently, **5/7 geometry failures still had the right diagnosis**, while
  **2/7 failed both exact geometry and diagnosis**. This is direct evidence
  that label accuracy can conceal absolute-map errors.

On coordinate-OOD, the three exact-pair failures all had the correct map right,
the student map wrong, and the derived diagnosis right. A separate fourth case
had both maps and the derived diagnosis right but the explicit label wrong.

### Image-only failure structure

- Test exact outcomes were 192 both right, 39 correct-only, 27 student-only,
  and 242 neither right. Among the 308 cases without both exact maps,
  **236 (76.6%) still had the correct label**; all 72 label errors occurred in
  the geometry-failure group.
- The largest test label confusion was
  `opposite_translation → wrong_translation`: **25/72 label errors**.
  `opposite_translation` label recall was 20/47 (42.6%), versus 100% for
  `wrong_translation`; the image-only arm appears especially sensitive to the
  translation-sign distinction.
- Exact two-map test performance varied sharply by label: 5/64 (7.8%) for
  `completely_wrong`, 11/66 (16.7%) for `wrong_translation`, and 50/76
  (65.8%) for `wrong_rotation_angle`. This is a descriptive pattern, not a
  causal attribution.
- OOD exact outcomes were 338 both right, 74 correct-only, 9 student-only, and
  79 neither right. Of the 162 pair failures, 107 came from the
  `completely_wrong` family.

## Scope and limitations

- OOD contains only four diagnosis families (`correct`,
  `rotation_instead_of_reflection`, `wrong_reflection_line`, and
  `completely_wrong`), while test contains all eight. The OOD aggregate is
  therefore a result on this restricted, differently composed split. Its
  higher image-only score does **not** establish universally stronger
  out-of-distribution generalization.
- Each rate comes from one 500-case sample and one trained adapter. Wilson
  intervals do not cover training randomness or alternate checkpoints.
- An observed 100% is not proof of perfection or statistical equivalence; for
  500/500, the Wilson lower bound is 99.2%.
- Exact canonical-map scoring is intentionally stricter than diagnosis scoring.
  Label or hint quality should not be substituted for exact geometry.
- The image and image+coordinates arms use modality-specific trained adapters.
  Their difference is an arm-level ablation, not a pure inference-time
  intervention holding training fixed.

## Recommended submission framing

Use **tuned image+coordinates** as the hero model and **tuned image-only** as the
harder ablation. Recommended wording:

> On paired 500-example test and OOD samples, the untuned 4B model produced no
> exact correct-map or two-map successes in any base cell, including when given
> coordinates. After in-distribution structured training on canonical net
> targets, the image+coordinates arm achieved 98.6% exact two-map accuracy on
> test and 99.4% on the restricted OOD split; image-only achieved 38.4% and
> 67.6%. These results support learned behavior from structured data, with
> coordinates serving as a useful representation after training rather than a
> sufficient prompt-time solution. OOD results are limited to its four-family
> composition and should not be generalized beyond that split.

Machine-readable derived values are in `FINAL_RESULTS_SUMMARY.json`; all raw
artifacts remain unchanged.
