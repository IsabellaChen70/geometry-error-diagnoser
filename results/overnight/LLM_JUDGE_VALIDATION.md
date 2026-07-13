# Independent validation of the blinded LLM judge

Status: **PASS**. All 100 successful paired judgments independently reproduced the saved aggregates and report core values.

## Provenance and integrity

- Provider/model: **TFY / `gpt-5.6-sol`**.
- Sample: **100 paired image+coordinates test IDs**, seed `20260709`; A/B identity was balanced **50/50**.
- Raw evidence: **106 rows** preserving **100 successful** and **6 failed** rows; the latest success set contains 100 unique sample IDs.
- Every successful row matched its metadata request hash and the recomputed run fingerprint; all 100 final response objects passed the strict schema.

## Recomputed substantive result

- Overall preference, which was instructed to respect authoritative exact scoring: tuned **100/100** (100.0%, Wilson 95% 96.3–100.0%); base **0/100**; ties **0/100**.
- Hint-only tutoring preference: base **75/100** (75.0%, Wilson 95% 65.7–82.5%); tuned **25/100** (25.0%, Wilson 95% 17.5–34.3%); ties **0/100**.
- Tuned versus base means: pedagogical usefulness **2.89 vs 1.80**; operation-family relevance **4.98 vs 1.91**; clarity/actionability **4.12 vs 2.70**.
- Operation-family relevance flags: tuned **100/100** versus base **50/100**. Forbidden answer/coordinate leakage flags: tuned **100/100** versus base **1/100**.

The result is not contradictory: the tuned outputs won overall because their deterministic geometry and diagnosis were much stronger, while the base hints won the hint-only comparison mainly by avoiding exact-answer disclosure. LLM judgments are subjective and secondary to deterministic scoring.

## Reliability

- Latest successful rows used **122 attempts**: **20** cases retried, with **15** strict-parse failures and **7** request errors before success.
- Cumulative preserved evidence contains **140 attempts** across 106 rows.
- Every count, rate, rubric mean, normal mean interval, Wilson interval, and retry statistic matched `LLM_JUDGE_RESULTS.json` within `1e-12`.

## Hint-safety reconciliation

- The deterministic audit found exact answer/map/value disclosure in **1,919/2,000 (96.0%)** tuned hints and only **28/2,000 (1.4%)** met its conservative safe/useful rule.
- The judge independently flagged tuned leakage in **100/100** sampled hints, while also rating tuned operation relevance and clarity much higher.
- The actual v6 prompt requested a short Socratic hint but did not explicitly forbid coordinates. Therefore this is a material submission caveat and behavior-spec gap, not evidence that the model violated an explicit v6 clause.

Reproduce without API access:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 results/overnight/validate_llm_judge_results.py
```
