# Submission readiness checklist

Checked on 2026-07-12; refreshed on 2026-07-12 after the protected README and ignore
rules were updated. Hard-rule note: the production app was built from a copied source tree
under `results/overnight/verify/brainlift_build/` so the protected app sources, `dist/`, and
`node_modules/` remained untouched. `npm ci` was intentionally skipped because it would rewrite
protected `node_modules/**`.

- [x] **PASS — scoring tests green.** `transform_diagnosis/`: **194 passed**; `model/`: **30 passed**.
- [x] **PASS — frozen tuned metrics reproduce.** Independent offline re-score of four copied tuned record files matched all 20 requested counts/rates in `FINAL_RESULTS_SUMMARY.json`; every stored→rescored delta was 0.000.
- [x] **PASS — overnight reports present and internally consistent.** `OVERNIGHT_REPORT.md`, `OVERNIGHT_RESULTS.json`, `CLEANUP_REPORT.md`, and `RUN_LOG.md` parse/exist; `build_report.py` asserts all re-score matches, all 12 historical comparison rows, and the integrated judge/hint evidence. `validate_llm_judge_results.py` independently checks the judge JSON/JSONL, reports, checklist tally, and run-log TL;DR.
- [x] **PASS — at least one fair v6 frontier comparison is documented.** Existing paired Claude Opus 4.8 comparison: image test n=150 and image+coordinates test n=50. No new frontier-comparison route ran during the original credential-free pass; the later TFY `gpt-5.6-sol` call was a distinct blinded judge, not another prediction model. Raw Opus records remain on ORCD and were not locally reverified.
- [x] **PASS — no real secrets found.** Repository `rg` scan found only literal placeholders/examples (`sk-ant-...`, `tfy_...`, and environment-variable names) in documentation/scripts; no credential value was found.
- [x] **PASS — top-level README is submission-ready.** Refreshed check: `README.md` now describes the geometry-diagnosis task and falsifiable claim, schema, data provenance/sample, modality-specific model/training setup, frozen results and limitations, exact local reproduction commands, brainlift usage, repository/artifact map, and the current absence of hosted model/demo/video links. It is no longer the generic React/Vite template.
- [x] **PASS — brainlift production build.** Isolated copied tree completed `tsc -b && vite build` with 33 transformed modules; output generated under `results/overnight/verify/brainlift_build/dist/`. No deploy or new preview server was started.
- [x] **PASS — dataset sample present and parseable.** `dataset_sample/train_sample.jsonl` and `train_sample_chat.jsonl` each contain 24 valid JSON rows; sample README and image files are present.
- [x] **PASS — `.gitignore` covers both large data and key files.** Refreshed check: it retains the large-data/build/cache rules and now ignores `.env`, `.env.*` (while allowing documented example/template files), common certificate/keystore extensions, and standard SSH private-key names.
- [x] **PASS — read-only git state captured.** Initial and final status were collected without `git add`, commit, stash, checkout, reset, branch, push, or any other git/index write.
- [x] **PASS — deterministic forbidden-failure / hint-safety audit completed.** `audit_hint_safety.py` independently parsed all **2,000/2,000** frozen tuned `full` outputs. It found **1,993/2,000** operation-family-relevant hints, **527/2,000** with an explicit coordinate pair, **253/2,000** with a coordinate pair unsanctioned by the existing rubric, **1,919/2,000** with a conservative exact answer/map/value disclosure, and **28/2,000** meeting the stricter combined safe/useful rule. Independent `hint_ok` and `hint_exact_ok` reproduced all **2,000/2,000** stored flags. The report explicitly records that the actual v6 prompt lacks the legacy no-coordinate sentence.
- [x] **PASS — blinded LLM-as-judge evaluation completed and independently validated.** TFY route `gpt-5.6-sol` judged **100 paired image+coordinates test IDs** sampled with seed `20260709`, with blinded A/B identity balanced 50/50. Independent no-API validation checked 100 unique successful rows, strict schema, request hashes and recomputed fingerprint, identity remapping, all counts/rates/means/intervals, and retry statistics. Overall preference was tuned/base/tie **100/0/0** because that rubric respected authoritative deterministic correctness; hint-only tutoring preference was **25/75/0**. The judge flagged forbidden answer/coordinate leakage in **100/100 tuned** versus **1/100 base** hints. This result is explicitly subjective, secondary to exact scoring, and reconciled with the deterministic hint audit.

## Tally

- **PASS: 12**
- **BLOCKED: 0**
- **FAIL: 0**

## Morning actions requiring the user

1. Do not claim that tuned hints are safely Socratic: the deterministic audit found exact answer/map/value disclosure in **1,919/2,000** tuned outputs, and the subjective judge flagged leakage in **100/100** sampled tuned hints. The v6 prompt omitted an explicit no-coordinate prohibition, so this is a material behavior-spec/target-design caveat rather than a direct prompt violation.
2. Retrieve the authoritative Opus v6 raw/audit files from ORCD if local row-level re-verification is required.
3. Complete the remaining publication deliverables if not already external: dataset/model hosting, inference demo link, brainlift, and video.

