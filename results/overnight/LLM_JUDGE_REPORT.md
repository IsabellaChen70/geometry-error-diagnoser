# Blinded LLM-as-judge evaluation

Status: **PASS**.

This evaluation is subjective and complements rather than replaces the authoritative deterministic map/label scorer.

## Provenance

- Provider/model: **TFY / `gpt-5.6-sol`**.
- Sample: **100 paired image+coordinates test IDs**, seed `20260709`; candidate identity was blinded and balanced 50/50 across A/B.

## Outcome

- Completed pairs: **100/100**.
- Overall tuned/base/tie: **100 / 0 / 0** (100.0% / 0.0% / 0.0%).
  Tuned Wilson 95% CI: **96.3–100.0%**.
- Hint-only tutoring tuned/base/tie: **25 / 75 / 0**. Base hint-only preference Wilson 95% CI: **65.7–82.5%**.

Overall preference was explicitly instructed to respect the supplied authoritative deterministic correctness flags. It is therefore not a second independent exact-map score. Hint-only preference isolates the subjective tutoring tradeoff.

## Subjective rubric

| Measure | Base | Tuned |
|---|---:|---:|
| Pedagogical usefulness mean (1–5) | 1.80 | 2.89 |
| Operation-family relevance mean (1–5) | 1.91 | 4.98 |
| Clarity/actionability mean (1–5) | 2.70 | 4.12 |
| Operation-family relevant | 50/100 | 100/100 |
| Forbidden answer/coordinate leakage | 1/100 | 100/100 |

The tuned outputs were clearer and much more operation-relevant, but every sampled tuned hint was judged to disclose forbidden exact answer information under the deliberately stricter tutoring-safety rubric. Accordingly, base won the hint-only preference 75–25 despite losing overall 0–100.

## Reliability

- Attempts in latest successful rows: **122**; retried cases: **20**; strict parse failures before success: **15**; request errors before success: **7**.
- Persisted judge-case attempts, cumulative across raw evidence: **140** in **106** raw rows (**100 successful**, **6 failed**).
- A/B assignment is exactly balanced 50/50. Wilson intervals and rubric means are available in `LLM_JUDGE_RESULTS.json`.
- Independent no-API recomputation: `LLM_JUDGE_VALIDATION.json` and `LLM_JUDGE_VALIDATION.md`.

## Behavior-spec caveat

The actual v6 prompt requested a short Socratic hint but did not explicitly prohibit coordinates. The leakage result is therefore a material tutoring-safety and specification concern, not evidence that the model violated an explicit v6 no-coordinate clause. LLM judgments remain subjective and secondary to deterministic geometry scoring.
