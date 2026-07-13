# Overnight autonomous run log

## TL;DR (read me first)

- **Headline:** the frozen v6 fine-tune independently re-scored exactly. A completed blinded 100-pair TFY `gpt-5.6-sol` judge preferred tuned overall **100–0** when respecting authoritative deterministic correctness, but preferred base hints **75–25** on tutoring alone because all **100/100** sampled tuned hints were flagged for exact-answer leakage.
- **Routes:** no new frontier-evaluation route ran in the original overnight pass; the existing paired Opus results retain their local-raw-artifact limitation. A later, distinct LLM-as-judge run completed on TFY `gpt-5.6-sol` over 100 paired image+coordinates test IDs and was independently validated without further API calls.
- **Credibility:** **PASS** — 194 harness tests + 30 model tests passed; all four frozen tuned cells and all 20 requested metrics matched independent rescoring; all 100 successful judge rows passed independent schema, hash/fingerprint, identity-remapping, aggregate, interval, and retry-statistic checks.
- **Cleanup:** **3 paths quarantined**, **27 kept for manual review**, **3 absent/no action**, **0 hard-deleted**.
- **Submission readiness:** **12 PASS / 0 BLOCKED / 0 FAIL**. Remaining concerns are the tuned hint-safety/specification gap, unavailable local Opus raw artifacts, and unfinished external publication/demo/video links.

Started: 2026-07-12 00:19:54 CDT

## Phase 0 — Preflight

- `2026-07-12 00:19:54 CDT` Created idempotent work directories under `results/overnight/` and `_quarantine/`.
- `2026-07-12 00:20 CDT` Environment PASS: Python 3.13.7; required Python packages installed; `transform_diagnosis.eval`, `v6_format`, and `chat_format` imported successfully.
- `2026-07-12 00:20 CDT` Harness integrity PASS: `transform_diagnosis/` — **194 passed**; `model/` — **30 passed**. Tests ran with bytecode/cache writes disabled.
- `2026-07-12 00:20 CDT` Frozen tuned-record independent re-score PASS. Four copied record files were regraded with the current harness; every stored metric matched exactly:
  - image+coordinates test, n=500: correct/student/both/label/derived = **493/498/493/498/498** = **0.986/0.996/0.986/0.996/0.996**.
  - image+coordinates OOD, n=500: **500/497/497/499/500** = **1.000/0.994/0.994/0.998/1.000**.
  - image test, n=500: **231/219/192/428/428** = **0.462/0.438/0.384/0.856/0.856**.
  - image OOD, n=500: **412/347/338/426/426** = **0.824/0.694/0.676/0.852/0.852**.
  - All before→after deltas, including parse and hint metrics, were exactly **0.000**.
- `2026-07-12 00:20 CDT` Frontier plumbing PASS with no API calls: both v6 image and image+coordinates 2-record dry-runs loaded frozen data/renders, built the canonical-net payloads, scored empty dry-run responses, and wrote outputs under `results/overnight/raw/`.
- `2026-07-12 00:21:03 CDT` API probe: `ANTHROPIC_API_KEY` absent; `TFY_API_KEY` absent. Route discovery, live two-call smoke tests, new GPT/Gemini/Claude runs, and legacy direct-Anthropic appendix were safely **SKIPPED**. No key values were printed and no API calls were made.
- `2026-07-12 00:21:03 CDT` Initial read-only `git status` captured. The tree was already heavily modified/staged before this run; no git write was performed and the index was not touched.

## Phase 1 — Frontier comparison

- `2026-07-12 00:21 CDT` No live API route was eligible after the required credential gate. Full 150×split×modality runs were skipped rather than bypassing safety.
- `2026-07-12 00:22 CDT` Preserved the existing apples-to-apples `v6.net-affine.1` Claude Opus 4.8 comparison from `results/v6_final/FRONTIER_COMPARISON.md`:
  - Image-only paired test n=150: Qwen vs Opus both-map **35.3% vs 33.3%**, direct label **84.0% vs 39.3%**, derived label **84.0% vs 45.3%**; Opus led student-map **55.3% vs 39.3%**.
  - Image+coordinates paired test n=50: both models **50/50** on both maps and derived label; Qwen led direct label **100% vs 80%**.
- Evidence boundary: authoritative Opus prediction/audit files remain on ORCD. They were previously independently audited but were not locally available overnight, so no new local re-score or disagreement-ID intersection is claimed.

## Phase 2 — Wins, losses, and fairness

- `2026-07-12 00:22 CDT` Largest measured fine-tune wins over Opus: image direct label **+44.7 pp**, image derived label **+38.7 pp**, coordinate direct label **+20.0 pp**, and image parse adherence **+2.0 pp**.
- Largest loss: Opus image student-map recovery **+16.0 pp**. Opus correct-map was **+3.3 pp**; fine-tune both-map was **+2.0 pp**. Prior analysis did not detect a difference for correct/both maps and did not establish equivalence.
- Both coordinate models reached the n=50 geometry ceiling. Qwen's label advantage is task-specific taxonomy specialization, not general frontier superiority.
- Cost/efficiency finding is qualitative only: the local 4B adapter avoids per-call API cost and service dependency. No unmeasured latency or dollar value was invented.

## Phase 3 — Benchmark credibility

- `2026-07-12 00:22 CDT` Trivial baselines computed from frozen label distributions: eight-label random **12.5%**; test majority **15.2%** (`wrong_rotation_angle`); restricted-OOD majority **26.8%** (`rotation_instead_of_reflection`); exact-map random success effectively zero.
- Base anchor PASS: every untuned base cell had **0/500** correct-map and **0/500** both-map successes, showing that the harness does not award trivial exact-map credit.
- Leakage artifact PASS: **0** exact train/evaluation geometry overlaps and **0** training rows sourced from test/OOD. Limitation: the 9,600-row v6 training file was not locally present, so the leakage fingerprint computation was not rerun.
- Oracle spot-check PASS with limitation: IDs `52`, `3529`, `3929`, `5252`, and `21634` all reproduced stored images from transforms; independently recovered maps and diagnoses agreed. These were tuned geometry failures, not frontier disagreements, because Opus raw rows were unavailable.

## Phase 4 — Reports

- `2026-07-12 00:22 CDT` Created `OVERNIGHT_REPORT.md`, with comparison tables, Wilson intervals, deltas, win/loss analysis, credibility checks, caveats, and submission framing.
- Created `OVERNIGHT_RESULTS.json`, with 12 structured `{route, split, modality, metric}` rows, counts, rates, deltas, Wilson intervals, baselines, re-score booleans, and route coverage.
- Created and validated `build_report.py`; assertions confirmed all re-score metrics match and all comparison rates are bounded and internally consistent.

## Phase 5 — Conservative quarantine

- `2026-07-12 00:23 CDT` Repository-wide `rg` evidence showed that the protected `sync_to_cluster.sh`, `eval_tuned_coords.py`, `rescore_records.py`, `contrastive.py`, and `BUILD_LOG.md` still reference many apparently superseded model files. Those files were retained for manual review.
- Quarantined only `.scratch_verify/`, `.scratch_v6_verify/`, and overnight-generated `results/overnight/__pycache__/`, preserving relative paths under `_quarantine/`.
- Created `CLEANUP_REPORT.md`: **3 quarantined / 27 kept / 3 absent / 0 deleted**.

## Phase 6 — Submission readiness

- `2026-07-12 00:24 CDT` Secret scan PASS: matches were placeholders or environment-variable names only; no real key value found.
- Dataset sample PASS: both JSONL files parsed, with **24 rows each**; image sample and README present.
- Brainlift build PASS: copied the protected app to `results/overnight/verify/brainlift_build/` and ran `tsc -b && vite build` successfully (33 modules). This avoided writes to protected app sources, root `dist/`, and `node_modules/`; no deploy or preview server was started.
- Failure/retry record: an initial combined staging/build command used the not-yet-created copied working directory and returned an unknown status. The operation was split into a successful staging step followed by a successful isolated production build.
- Runbook reference PASS: every final v6 pipeline script named in `V6_TRANSFORM_RUNBOOK.md` exists.
- Submission FAIL: protected top-level `README.md` is still the generic React/Vite template.
- Submission FAIL: protected `.gitignore` covers large data/build/cache paths but lacks `.env*` and private-key patterns.
- `2026-07-12 00:25:47 CDT` Created `SUBMISSION_CHECKLIST.md` (**8 PASS / 2 FAIL**) and captured final read-only tree/index state in `GIT_STATUS.txt` and `GIT_INDEX_STATUS.txt`.
- No training, GPU job, `sbatch`, deployment, new preview server, git write, hard delete, or protected-file edit occurred.

Finished: 2026-07-12 00:25:47 CDT

## 2026-07-12 submission refresh and evaluation-gap closure

- `2026-07-12 12:12:32 CDT` Rechecked the protected `README.md` and `.gitignore` directly. Both stale checklist failures are now PASS: the README is project-specific and submission-oriented, and the ignore file now covers `.env*`, common private-key/certificate/keystore files, and standard SSH key names.
- Ran `audit_hint_safety.py` over all four frozen tuned `full` record files (**2,000 rows**). Independent final-JSON parsing found **1,993/2,000** operation-family-relevant hints, **527/2,000** hints with explicit coordinate pairs, **253/2,000** with coordinate pairs unsanctioned under the existing evaluator, **1,919/2,000** with conservative exact answer/map/value disclosures, and **28/2,000** passing the stricter combined safe/useful rule. Independent `hint_ok` and `hint_exact_ok` agreed with all stored flags. The report records the important spec boundary: the v6 prompt asks for a short Socratic hint but omits the legacy no-coordinate prohibition.
- Probed only boolean key presence: `TFY_API_KEY` absent and `ANTHROPIC_API_KEY` absent. `run_llm_judge.py` selected 100 paired image+coordinates test IDs with seed `20260709`, balanced blinded A/B identity 50/50, and passed 2/2 offline payload/schema checks. The live judge is **BLOCKED** with **0 API calls**; exact resumable commands are in `LLM_JUDGE_REPORT.md`.
- Refreshed `SUBMISSION_CHECKLIST.md` to **11 PASS / 1 BLOCKED / 0 FAIL**. No training, GPU/model-weight load, deployment, frozen-prediction edit, protected-source edit, secret persistence, or git write occurred.

## Final live-judge validation and integration

- `2026-07-12 13:33:13 CDT` Consumed the completed local TFY `gpt-5.6-sol` judge artifacts without making further API calls. The run covers **100 paired image+coordinates test IDs**, seed `20260709`, with blinded A/B identity exactly balanced 50/50.
- Independent validator PASS: all **100 successful rows** were unique and matched the sample, provider/model, metadata request hashes, recomputed request-set hash and run fingerprint, strict judgment schema, final raw response, and A/B identity remapping. The append-only raw file preserves **106 rows** total: 100 successful and 6 failed historical/retry rows.
- Recomputed overall tuned/base/tie preference was **100/0/0** (tuned Wilson 95% **96.3–100.0%**). This overall rubric explicitly respected authoritative deterministic correctness and is not a replacement exact-map score. Hint-only tutoring preference reversed to **25/75/0** (base Wilson 95% **65.7–82.5%**).
- Subjective rubric means, tuned versus base: pedagogical usefulness **2.89 vs 1.80**, operation-family relevance **4.98 vs 1.91**, and clarity/actionability **4.12 vs 2.70**. Operation-family relevance flags were **100/100 vs 50/100**; forbidden answer/coordinate leakage flags were **100/100 vs 1/100**.
- Reliability recomputation matched exactly: latest successful rows used **122 attempts**, with **20** retried cases, **15** strict-parse failures, and **7** request errors before success; cumulative preserved evidence contains **140 attempts**.
- Reconciled safety evidence: the deterministic audit found exact answer/map/value disclosure in **1,919/2,000 (96.0%)** tuned hints and only **28/2,000 (1.4%)** conservative safe/useful hints. The judge's 100/100 tuned leakage result is directionally consistent. Because v6 requested a short Socratic hint but did not explicitly prohibit coordinates, this is a material behavior-spec/target-design caveat, not a direct prompt violation.
- Integrated the validated subjective result into `LLM_JUDGE_REPORT.md`, `OVERNIGHT_REPORT.md`, and `OVERNIGHT_RESULTS.json`; created `LLM_JUDGE_VALIDATION.{json,md}`; and refreshed the checklist to **12 PASS / 0 BLOCKED / 0 FAIL**. No protected/source file, frozen prediction, model weight, or git/index state was changed.

