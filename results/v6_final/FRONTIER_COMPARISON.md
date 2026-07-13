# Fair v6 frontier comparison

## Provenance boundary

This note and the companion Canvas were built from the previously verified
primary-history results supplied for this task. The authoritative prediction,
result, and audit files remain under `/home/ikchen` on ORCD. Interactive SSH
authentication prevented re-downloading them, so this chart build did not
recompute the metrics or perform a new local raw-artifact verification.

Visual artifact:
[frontier comparison Canvas](/Users/isabellachen/.cursor/projects/Users-isabellachen-projects-SLM/canvases/frontier-comparison.canvas.tsx).

## Comparison design

- Models: v6 fine-tuned Qwen3-VL-4B versus zero-shot Claude Opus 4.8.
- Qwen base family: `unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`, evaluated
  with the modality-specific v6 adapter.
- Task and schema: `full`, `v6.net-affine.1`.
- Split and pairing: identical test IDs within each modality, sampled with seed
  `20260709`.
- Samples: image-only `n=150`; image+coordinates `n=50`.
- Metrics: parse success, exact RED-to-GREEN correct map, exact RED-to-BLUE
  student map, both exact maps on the same case, direct diagnosis label, and
  diagnosis derived from the predicted maps.

## Verified primary-history results

Metric order below is parse / correct map / student map / both maps / direct
label / derived label.

- Image-only Qwen, `n=150`: `150/150 = 1.000` / `66/150 = 0.440` /
  `59/150 = 0.393` / `53/150 = 0.353` / `126/150 = 0.840` /
  `126/150 = 0.840`.
- Image-only Opus, `n=150`: `147/150 = 0.980` / `71/150 = 0.473` /
  `83/150 = 0.553` / `50/150 = 0.333` / `59/150 = 0.393` /
  `68/150 = 0.453`.
- Image+coordinates Qwen, `n=50`: all six metrics were
  `50/50 = 1.000`.
- Image+coordinates Opus, `n=50`: parse, correct map, student map, both maps,
  and derived label were each `50/50 = 1.000`; direct label was
  `40/50 = 0.800`.

The Opus image-only derived-label value in this comparison is
`68/150 = 0.453`. The older `68/147 = 0.463` value was conditional on a
derivation being available. It is not the all-record rate used in the chart.

## Authoritative ORCD artifacts

Prediction records and aggregates:

- `/home/ikchen/records_v6_4b_image_n150_test.jsonl`
- `/home/ikchen/results_v6_4b_image_n150_test.json`
- `/home/ikchen/records_frontier_v6_opus_image_n150_test_rescored.jsonl`
- `/home/ikchen/results_frontier_v6_opus_image_n150_test_rescored.json`
- `/home/ikchen/records_v6_4b_image_coords_n50_test.jsonl`
- `/home/ikchen/results_v6_4b_image_coords_n50_test.json`
- `/home/ikchen/records_frontier_v6_opus_image_coords_n50_test.jsonl`
- `/home/ikchen/results_frontier_v6_opus_image_coords_n50_test.json`

Relevant audit outputs:

- `/home/ikchen/audit_records_v6_4b_image_n150_test.json`
- `/home/ikchen/audit_records_frontier_v6_opus_image_n150_test_rescored.json`
- `/home/ikchen/audit_records_v6_4b_image_coords_n50_test.json`
- `/home/ikchen/audit_records_frontier_v6_opus_image_coords_n50_test.json`
- `/home/ikchen/audit_v6_paired_summary.json`

These paths are recorded for provenance only. They were not copied into this
repository during the current chart build.

## Interpretation and audit caveats

- No difference was statistically detected for image-only correct-map or
  both-map recovery. This does not establish statistical equivalence.
- Opus was significantly stronger on image-only student-map recovery:
  `83/150` versus `59/150`.
- Qwen's direct-label advantage, `126/150` versus `59/150`, reflects
  task-specific training on this diagnosis taxonomy and is not evidence of
  general frontier-model superiority.
- Both coordinate models reached the geometry ceiling on this paired
  50-example sample. The ceiling and sample size restrict broader claims.
- The prior independent audit reported zero evaluator disagreements after
  Opus rescoring, zero exact train/evaluation geometry overlaps, and zero
  training rows sourced from test or OOD. Those checks were not rerun locally
  for this chart build.
