# Overnight transform-diagnosis benchmark report

## Headline

The frozen v6 fine-tune is demonstrably real: all four tuned cells reproduced exactly under an independent local re-score, while the untuned base had 0/500 exact correct-map and 0/500 exact two-map successes in every cell. A later blinded 100-pair TFY `gpt-5.6-sol` judge preferred tuned overall 100–0 when respecting authoritative deterministic correctness, but preferred base hints 75–25 on tutoring alone because every sampled tuned hint was flagged for exact-answer leakage.

## Run coverage

- New live frontier-evaluation routes in the original overnight pass: **none**. Credentials were absent then, so those model-comparison calls were safely skipped.
- Completed secondary LLM judge: **TFY `gpt-5.6-sol`**, 100 paired image+coordinates test cases, seed `20260709`; independently validated from local raw evidence.
- Existing comparable frontier route: **Claude Opus 4.8**, zero-shot on `v6.net-affine.1`.
- Existing paired samples: image-only test `n=150`; image+coordinates test `n=50`; seed `20260709`.
- The Opus counts below come from the previously verified project-history comparison in `results/v6_final/FRONTIER_COMPARISON.md`. Its authoritative raw records remain on ORCD and were not locally re-downloaded, so local row-level pairing and rescoring are marked unverified rather than implied.

## Existing fair-v6 frontier comparison

### Image-only test — paired n=150

| Metric | Tuned Qwen3-VL-4B | Claude Opus 4.8 | Fine-tune − Opus |
|---|---:|---:|---:|
| Parse | 150/150 = 100.0% [97.5–100.0%] | 147/150 = 98.0% [94.3–99.3%] | +2.0 pp |
| Correct map | 66/150 = 44.0% [36.3–52.0%] | 71/150 = 47.3% [39.5–55.3%] | -3.3 pp |
| Student map | 59/150 = 39.3% [31.9–47.3%] | 83/150 = 55.3% [47.3–63.1%] | -16.0 pp |
| Both maps | 53/150 = 35.3% [28.1–43.3%] | 50/150 = 33.3% [26.3–41.2%] | +2.0 pp |
| Direct label | 126/150 = 84.0% [77.3–89.0%] | 59/150 = 39.3% [31.9–47.3%] | +44.7 pp |
| Derived label | 126/150 = 84.0% [77.3–89.0%] | 68/150 = 45.3% [37.6–53.3%] | +38.7 pp |

### Image + coordinates test — paired n=50

| Metric | Tuned Qwen3-VL-4B | Claude Opus 4.8 | Fine-tune − Opus |
|---|---:|---:|---:|
| Parse | 50/50 = 100.0% [92.9–100.0%] | 50/50 = 100.0% [92.9–100.0%] | +0.0 pp |
| Correct map | 50/50 = 100.0% [92.9–100.0%] | 50/50 = 100.0% [92.9–100.0%] | +0.0 pp |
| Student map | 50/50 = 100.0% [92.9–100.0%] | 50/50 = 100.0% [92.9–100.0%] | +0.0 pp |
| Both maps | 50/50 = 100.0% [92.9–100.0%] | 50/50 = 100.0% [92.9–100.0%] | +0.0 pp |
| Direct label | 50/50 = 100.0% [92.9–100.0%] | 40/50 = 80.0% [67.0–88.8%] | +20.0 pp |
| Derived label | 50/50 = 100.0% [92.9–100.0%] | 50/50 = 100.0% [92.9–100.0%] | +0.0 pp |

The corrected Opus image-only derived-label rate is **68/150 = 45.3%**; the older 68/147 = 46.3% value used a conditional denominator and is not used here.

## Where the fine-tune wins, loses, and ties

- **Largest image-only wins:** direct label **+44.7 pp** and derived label **+38.7 pp**. This is the clearest place the small task-specific model beats zero-shot Opus, but it reflects training on the bespoke diagnosis taxonomy.
- **Image-only exact pair:** the fine-tune was **+2.0 pp** on both maps, while Opus was **+3.3 pp** on the correct map. Prior paired analysis did not detect a difference for correct-map or both-map recovery; that is not evidence of equivalence.
- **Largest loss:** Opus led image-only student-map recovery by **16.0 pp** (55.3% vs 39.3%), a prominent frontier advantage.
- **Coordinates:** both models hit 50/50 on both exact maps and derived labels. The fine-tune led direct taxonomy labeling by **20.0 pp** (100% vs 80%). The n=50 ceiling prevents a broad superiority claim.
- **Format adherence:** the fine-tune parsed 100%; Opus parsed 98% image-only and 100% with coordinates.
- **Cost/independence:** the fine-tune is a local 4B adapter with no per-call API charge or external service dependency. No latency or dollar values were measured, so none are claimed.

## Frozen final-model anchor

- Tuned image+coordinates: **98.6% both-map test** (493/500) and **99.4% restricted-OOD** (497/500).
- Tuned image-only: **38.4% both-map test** (192/500) and **67.6% restricted-OOD** (338/500).
- Untuned base: **0/500 correct maps and 0/500 both-map successes in every base cell**, including coordinate input.

## Secondary blinded LLM judge — subjective

- **Overall preference:** tuned/base/tie **100/0/0**. Tuned rate **100.0%** (Wilson 95% 96.3–100.0%). This preference was explicitly instructed to respect the authoritative deterministic correctness flags, so it is not a second exact-map score.
- **Hint-only tutoring preference:** tuned/base/tie **25/75/0**; base rate **75.0%** (Wilson 95% 65.7–82.5%).

| Subjective measure | Base | Tuned |
|---|---:|---:|
| Pedagogical usefulness mean (1–5) | 1.80 | 2.89 |
| Operation-family relevance mean (1–5) | 1.91 | 4.98 |
| Clarity/actionability mean (1–5) | 2.70 | 4.12 |
| Operation-family relevant | 50/100 | 100/100 |
| Forbidden answer/coordinate leakage | 1/100 | 100/100 |

The tuned hints were clearer and much more operation-relevant, yet the stricter safety rubric flagged all 100 for exact-answer disclosure. This explains the apparently split result: tuned won overall through exact task correctness, while safer but often vague or misdirected base hints won the hint-only comparison.

The deterministic audit supports the same concern at larger scope: **1919/2000 (96.0%)** tuned hints disclosed exact answer/map/value information, and only **28/2000 (1.4%)** met the conservative safe/useful rule. The v6 prompt asked for a short Socratic hint but did not explicitly prohibit coordinates, so this is a material behavior-spec/target design caveat rather than a direct prompt violation.

## Credibility checks

| Check | Status | Evidence |
|---|---|---|
| Scoring harness | **PASS** | `transform_diagnosis/`: 194 passed; `model/`: 30 passed |
| Frozen tuned re-score | **PASS** | All 20 requested counts/rates match `FINAL_RESULTS_SUMMARY.json`; all stored→rescored deltas were 0.000 |
| Base anchor | **PASS** | Exact maps are not handed out: all four base cells had zero correct-map and zero both-map successes |
| Leakage | **PASS (artifact)** | Existing audit: 0 exact geometry overlaps; 0 training rows sourced from test/OOD |
| Leakage recomputation | **UNVERIFIABLE locally** | The 9,600-row v6 training file was not part of the frozen download |
| Oracle spot-check | **PASS with limitation** | IDs 52, 3529, 3929, 5252, 21634: stored transforms reproduce vertices; independently recovered maps and diagnosis agree |
| Frontier disagreement spot-check | **UNVERIFIABLE locally** | Opus raw records remain on ORCD and could not be retrieved without interactive SSH authentication |
| New frontier routes | **SKIPPED safely** | No credential was present during the original frontier-evaluation pass |
| Blinded LLM judge | **PASS** | 100 unique successful TFY `gpt-5.6-sol` rows independently matched strict schema, request hashes/fingerprint, remapping, aggregates, intervals, and retry statistics |
| Hint safety | **MATERIAL CAVEAT** | Deterministic disclosure 1,919/2,000; subjective tuned leakage 100/100; v6 omitted an explicit no-coordinate clause |

## Trivial baselines

| Split | Uniform 8-label guess | Majority-class label | Fine-tuned coordinate label |
|---|---:|---:|---:|
| Test | 12.5% | 15.2% (`wrong_rotation_angle`) | 99.6% |
| Restricted OOD | 12.5% | 26.8% (`rotation_instead_of_reflection`) | 99.8% |

Exact canonical-map random success is effectively zero. OOD contains only four observed label families; a uniform guess restricted to those four would be 25%, but the table keeps the assignment's fixed eight-label 12.5% baseline.

## Fairness and scope caveats

- Opus is zero-shot on an unfamiliar canonical schema; Qwen is fine-tuned for it.
- OOD contains only four diagnosis families and differs compositionally from test; its higher image-only rate is not universal OOD evidence.
- Frontier figures are one paired sample per modality (`n=150` image, `n=50` coordinates). Wilson intervals quantify sampling uncertainty, not model/training-seed variability.
- The image and coordinate fine-tunes are modality-specific adapters, not a pure inference-time input ablation.
- Existing Opus audit results were not rerun locally overnight because the raw records were unavailable.
- The LLM judge is one subjective route on one 100-case test sample. It was given authoritative deterministic correctness flags and cannot independently validate exact geometry.

## Submission-ready framing

> On paired held-out v6 evaluations, the untuned 4B model produced no exact correct-map or two-map successes, including when given coordinates. Structured canonical-net training raised the image+coordinates fine-tune to 98.6% exact two-map accuracy on test, and in the existing fair comparison it matched zero-shot Claude Opus 4.8 on all coordinate geometry cases while improving direct task-taxonomy labeling from 80% to 100%. Image-only results were mixed: the fine-tune led direct and derived diagnosis by 44.7 and 38.7 percentage points, while Opus led student-map recovery by 16.0 points. A secondary subjective judge preferred tuned overall because of this deterministic advantage, but preferred base hints 75–25 and flagged exact-answer leakage in every sampled tuned hint. This is evidence for efficient task specialization, not general frontier-model superiority or safe Socratic hinting.

## Artifacts

- Machine-readable results: [`OVERNIGHT_RESULTS.json`](OVERNIGHT_RESULTS.json)
- Re-scored frozen outputs: [`verify/`](verify/)
- No-cost gateway dry-runs: [`raw/`](raw/)
- Deterministic hint audit: [`HINT_SAFETY_AUDIT.md`](HINT_SAFETY_AUDIT.md)
- Blinded judge report: [`LLM_JUDGE_REPORT.md`](LLM_JUDGE_REPORT.md)
- Independent judge validation: [`LLM_JUDGE_VALIDATION.md`](LLM_JUDGE_VALIDATION.md)
- Existing frozen result report: [`../v6_final/FINAL_RESULTS.md`](../v6_final/FINAL_RESULTS.md)
- Existing frontier provenance note: [`../v6_final/FRONTIER_COMPARISON.md`](../v6_final/FRONTIER_COMPARISON.md)
