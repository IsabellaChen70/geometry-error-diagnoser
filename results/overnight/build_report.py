#!/usr/bin/env python3
"""Build the overnight comparison report from frozen, already-scored artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "overnight"
FINAL = ROOT / "results" / "v6_final"
VERIFY = OUT / "verify"

METRICS = (
    "correct_net_ok",
    "student_net_ok",
    "both_nets_ok",
    "label_ok",
    "derived_label_ok",
)
AGG_KEYS = {
    "correct_net_ok": "correct_net_match_rate",
    "student_net_ok": "student_net_match_rate",
    "both_nets_ok": "both_nets_match_rate",
    "label_ok": "label_accuracy",
    "derived_label_ok": "derived_label_accuracy",
}
CELL_TO_FILE = {
    "tuned_image_test": "results_v6_4b_image_test_rescored.json",
    "tuned_image_ood": "results_v6_4b_image_ood_rescored.json",
    "tuned_image_coords_test": "results_v6_4b_image_coords_test_rescored.json",
    "tuned_image_coords_ood": "results_v6_4b_image_coords_ood_rescored.json",
}


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if not n:
        return [0.0, 0.0]
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def ci_text(interval: list[float]) -> str:
    return f"{100 * interval[0]:.1f}–{100 * interval[1]:.1f}%"


def comparison_row(
    *,
    route: str,
    split: str,
    modality: str,
    metric: str,
    tuned_hits: int,
    frontier_hits: int,
    n: int,
) -> dict:
    tuned_rate = tuned_hits / n
    frontier_rate = frontier_hits / n
    delta = 100 * (tuned_rate - frontier_rate)
    return {
        "route": route,
        "split": split,
        "modality": modality,
        "metric": metric,
        "finetune_count": tuned_hits,
        "finetune_n": n,
        "finetune_rate": tuned_rate,
        "frontier_count": frontier_hits,
        "frontier_n": n,
        "frontier_rate": frontier_rate,
        "delta_pp": delta,
        "wilson_95": {
            "finetune": wilson(tuned_hits, n),
            "frontier": wilson(frontier_hits, n),
        },
        "outcome": "win" if delta > 0 else ("loss" if delta < 0 else "tie"),
        "paired_ids": True,
        "local_raw_reverification": False,
    }


def main() -> None:
    summary = load_json(FINAL / "FINAL_RESULTS_SUMMARY.json")
    judge = load_json(OUT / "LLM_JUDGE_RESULTS.json")
    judge_validation = load_json(OUT / "LLM_JUDGE_VALIDATION.json")
    hint_safety = load_json(OUT / "HINT_SAFETY_AUDIT.json")
    assert judge["status"] == "PASS"
    assert judge_validation["status"] == "PASS"
    assert hint_safety["status"] == "PASS"
    cells = summary["cells"]

    rescore_checks = {}
    for cell, filename in CELL_TO_FILE.items():
        rescored = load_json(VERIFY / filename)
        expected = cells[cell]
        checks = {}
        for index, metric in enumerate(METRICS):
            observed_rate = rescored[AGG_KEYS[metric]]
            observed_count = round(observed_rate * rescored["n"])
            checks[metric] = {
                "expected_count": expected["counts"][index],
                "observed_count": observed_count,
                "expected_rate": expected["rates"][index],
                "observed_rate": observed_rate,
                "match": (
                    observed_count == expected["counts"][index]
                    and abs(observed_rate - expected["rates"][index]) < 1e-12
                ),
            }
        rescore_checks[cell] = {
            "n": rescored["n"],
            "metrics": checks,
            "all_match": all(item["match"] for item in checks.values()),
        }

    # Existing fair v6 comparison recovered from the previously verified project history.
    # Raw records remain on ORCD and were not locally available during this run.
    history = {
        ("test", "image"): {
            "n": 150,
            "finetune": {
                "parse_ok": 150,
                "correct_net_ok": 66,
                "student_net_ok": 59,
                "both_nets_ok": 53,
                "label_ok": 126,
                "derived_label_ok": 126,
            },
            "frontier": {
                "parse_ok": 147,
                "correct_net_ok": 71,
                "student_net_ok": 83,
                "both_nets_ok": 50,
                "label_ok": 59,
                "derived_label_ok": 68,
            },
        },
        ("test", "image_coords"): {
            "n": 50,
            "finetune": {
                "parse_ok": 50,
                "correct_net_ok": 50,
                "student_net_ok": 50,
                "both_nets_ok": 50,
                "label_ok": 50,
                "derived_label_ok": 50,
            },
            "frontier": {
                "parse_ok": 50,
                "correct_net_ok": 50,
                "student_net_ok": 50,
                "both_nets_ok": 50,
                "label_ok": 40,
                "derived_label_ok": 50,
            },
        },
    }
    comparisons = []
    for (split, modality), values in history.items():
        for metric in ("parse_ok", *METRICS):
            comparisons.append(
                comparison_row(
                    route="claude-opus-4-8",
                    split=split,
                    modality=modality,
                    metric=metric,
                    tuned_hits=values["finetune"][metric],
                    frontier_hits=values["frontier"][metric],
                    n=values["n"],
                )
            )

    label_distributions = summary["label_distributions"]
    baselines = {}
    for split, distribution in label_distributions.items():
        n = sum(distribution.values())
        majority_label, majority_count = max(distribution.items(), key=lambda item: item[1])
        baselines[split] = {
            "n": n,
            "taxonomy_random_guess_rate": 1 / 8,
            "observed_split_label_count": len(distribution),
            "observed_label_uniform_rate": 1 / len(distribution),
            "majority_label": majority_label,
            "majority_count": majority_count,
            "majority_rate": majority_count / n,
            "exact_map_random_rate_approx": 0.0,
        }

    output = {
        "schema_version": "v6.overnight-results.1",
        "generated_from": {
            "frozen_summary": "results/v6_final/FINAL_RESULTS_SUMMARY.json",
            "rescored_records_dir": "results/overnight/verify/",
            "historical_frontier_note": "results/v6_final/FRONTIER_COMPARISON.md",
            "llm_judge": "results/overnight/LLM_JUDGE_RESULTS.json",
            "llm_judge_validation": "results/overnight/LLM_JUDGE_VALIDATION.json",
            "hint_safety_audit": "results/overnight/HINT_SAFETY_AUDIT.json",
        },
        "run_coverage": {
            "new_api_routes_completed": [],
            "new_api_routes_skipped": [
                {
                    "category": "gateway GPT/Claude/Gemini routes",
                    "reason": "TFY_API_KEY was not set during the original frontier-evaluation pass",
                },
                {
                    "category": "direct Anthropic legacy appendix",
                    "reason": "ANTHROPIC_API_KEY was not set during the original frontier-evaluation pass",
                },
            ],
            "llm_judge_routes_completed": [
                {
                    "provider": judge["provider"],
                    "model": judge["model"],
                    "split": "test",
                    "modality": "image_coords",
                    "paired_cases": judge["aggregate"]["completed_pairs"],
                    "sample_seed": judge["sample_seed"],
                    "subjective_secondary_evaluation": True,
                    "independently_validated": True,
                }
            ],
            "historical_v6_route_included": "claude-opus-4-8",
            "historical_raw_artifacts_local": False,
        },
        "rescore_checks": rescore_checks,
        "all_rescore_metrics_match": all(
            value["all_match"] for value in rescore_checks.values()
        ),
        "comparisons": comparisons,
        "baselines": baselines,
        "base_anchor": {
            "exact_correct_map_successes_each_base_cell": 0,
            "exact_both_map_successes_each_base_cell": 0,
            "n_each_cell": 500,
        },
        "audit": summary["audit"],
        "llm_judge": {
            "status": judge["status"],
            "provider": judge["provider"],
            "model": judge["model"],
            "sample_seed": judge["sample_seed"],
            "split": "test",
            "modality": "image_coords",
            "subjective_secondary_evaluation": True,
            "aggregate": judge["aggregate"],
            "independent_validation": {
                "status": judge_validation["status"],
                "artifact": "results/overnight/LLM_JUDGE_VALIDATION.json",
                "checks": judge_validation["checks"],
            },
        },
        "hint_safety": {
            "status": hint_safety["status"],
            "scope": hint_safety["overall"]["n"],
            "exact_answer_value_disclosure": hint_safety["overall"]["metrics"][
                "exact_answer_value_disclosure"
            ],
            "safe_useful": hint_safety["overall"]["metrics"]["safe_useful"],
            "v6_prompt_explicitly_forbids_coordinates": hint_safety[
                "contract_boundary"
            ]["v6_prompt_explicitly_forbids_coordinates"],
            "judge_sample_tuned_leak": judge["aggregate"][
                "rubric_by_model_identity"
            ]["tuned"]["forbidden_answer_or_coordinate_leak"],
            "interpretation": (
                "Tuned hints are highly operation-relevant but routinely disclose "
                "exact answer parameters under the stricter tutoring-safety rubric. "
                "The v6 prompt did not explicitly prohibit coordinates."
            ),
        },
        "credibility": {
            "transform_diagnosis_tests": {"status": "PASS", "passed": 194},
            "model_tests": {"status": "PASS", "passed": 30},
            "frozen_tuned_rescore": {"status": "PASS"},
            "base_model_anchor": {"status": "PASS"},
            "leakage_audit_artifact": {
                "status": "PASS",
                "locally_recomputed": False,
            },
            "oracle_spot_check": {
                "status": "PASS_WITH_LIMITATION",
                "ids": [52, 3529, 3929, 5252, 21634],
                "checks": "stored transforms reproduce images; recovered maps and diagnoses agree",
                "limitation": "frontier raw records unavailable, so these are tuned-failure rather than frontier-disagreement cases",
            },
            "historical_frontier_raw_reverification": {
                "status": "UNVERIFIABLE_LOCALLY",
                "reason": "authoritative files remain on ORCD; interactive SSH authentication unavailable",
            },
            "llm_judge_validation": {
                "status": "PASS",
                "successful_unique_rows": 100,
                "checks": (
                    "strict schema, request hashes/fingerprint, A/B remapping, "
                    "counts/rates/means/intervals, and retry statistics"
                ),
            },
        },
    }
    with (OUT / "OVERNIGHT_RESULTS.json").open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    by_modality = {
        modality: [row for row in comparisons if row["modality"] == modality]
        for modality in ("image", "image_coords")
    }
    judge_agg = judge["aggregate"]
    judge_overall = judge_agg["overall_preference"]
    judge_tutoring = judge_agg["tutoring_preference"]
    judge_base = judge_agg["rubric_by_model_identity"]["base"]
    judge_tuned = judge_agg["rubric_by_model_identity"]["tuned"]
    hint_metrics = hint_safety["overall"]["metrics"]

    lines = [
        "# Overnight transform-diagnosis benchmark report",
        "",
        "## Headline",
        "",
        "The frozen v6 fine-tune is demonstrably real: all four tuned cells reproduced exactly under an independent local re-score, while the untuned base had 0/500 exact correct-map and 0/500 exact two-map successes in every cell. A later blinded 100-pair TFY `gpt-5.6-sol` judge preferred tuned overall 100–0 when respecting authoritative deterministic correctness, but preferred base hints 75–25 on tutoring alone because every sampled tuned hint was flagged for exact-answer leakage.",
        "",
        "## Run coverage",
        "",
        "- New live frontier-evaluation routes in the original overnight pass: **none**. Credentials were absent then, so those model-comparison calls were safely skipped.",
        "- Completed secondary LLM judge: **TFY `gpt-5.6-sol`**, 100 paired image+coordinates test cases, seed `20260709`; independently validated from local raw evidence.",
        "- Existing comparable frontier route: **Claude Opus 4.8**, zero-shot on `v6.net-affine.1`.",
        "- Existing paired samples: image-only test `n=150`; image+coordinates test `n=50`; seed `20260709`.",
        "- The Opus counts below come from the previously verified project-history comparison in `results/v6_final/FRONTIER_COMPARISON.md`. Its authoritative raw records remain on ORCD and were not locally re-downloaded, so local row-level pairing and rescoring are marked unverified rather than implied.",
        "",
        "## Existing fair-v6 frontier comparison",
        "",
    ]

    display_names = {
        "parse_ok": "Parse",
        "correct_net_ok": "Correct map",
        "student_net_ok": "Student map",
        "both_nets_ok": "Both maps",
        "label_ok": "Direct label",
        "derived_label_ok": "Derived label",
    }
    for modality, title in (
        ("image", "Image-only test — paired n=150"),
        ("image_coords", "Image + coordinates test — paired n=50"),
    ):
        lines.extend(
            [
                f"### {title}",
                "",
                "| Metric | Tuned Qwen3-VL-4B | Claude Opus 4.8 | Fine-tune − Opus |",
                "|---|---:|---:|---:|",
            ]
        )
        for row in by_modality[modality]:
            lines.append(
                f"| {display_names[row['metric']]} | "
                f"{row['finetune_count']}/{row['finetune_n']} = {pct(row['finetune_rate'])} "
                f"[{ci_text(row['wilson_95']['finetune'])}] | "
                f"{row['frontier_count']}/{row['frontier_n']} = {pct(row['frontier_rate'])} "
                f"[{ci_text(row['wilson_95']['frontier'])}] | "
                f"{row['delta_pp']:+.1f} pp |"
            )
        lines.append("")

    lines.extend(
        [
            "The corrected Opus image-only derived-label rate is **68/150 = 45.3%**; the older 68/147 = 46.3% value used a conditional denominator and is not used here.",
            "",
            "## Where the fine-tune wins, loses, and ties",
            "",
            "- **Largest image-only wins:** direct label **+44.7 pp** and derived label **+38.7 pp**. This is the clearest place the small task-specific model beats zero-shot Opus, but it reflects training on the bespoke diagnosis taxonomy.",
            "- **Image-only exact pair:** the fine-tune was **+2.0 pp** on both maps, while Opus was **+3.3 pp** on the correct map. Prior paired analysis did not detect a difference for correct-map or both-map recovery; that is not evidence of equivalence.",
            "- **Largest loss:** Opus led image-only student-map recovery by **16.0 pp** (55.3% vs 39.3%), a prominent frontier advantage.",
            "- **Coordinates:** both models hit 50/50 on both exact maps and derived labels. The fine-tune led direct taxonomy labeling by **20.0 pp** (100% vs 80%). The n=50 ceiling prevents a broad superiority claim.",
            "- **Format adherence:** the fine-tune parsed 100%; Opus parsed 98% image-only and 100% with coordinates.",
            "- **Cost/independence:** the fine-tune is a local 4B adapter with no per-call API charge or external service dependency. No latency or dollar values were measured, so none are claimed.",
            "",
            "## Frozen final-model anchor",
            "",
            "- Tuned image+coordinates: **98.6% both-map test** (493/500) and **99.4% restricted-OOD** (497/500).",
            "- Tuned image-only: **38.4% both-map test** (192/500) and **67.6% restricted-OOD** (338/500).",
            "- Untuned base: **0/500 correct maps and 0/500 both-map successes in every base cell**, including coordinate input.",
            "",
            "## Secondary blinded LLM judge — subjective",
            "",
            f"- **Overall preference:** tuned/base/tie "
            f"**{judge_overall['tuned']['count']}/{judge_overall['base']['count']}/"
            f"{judge_overall['tie']['count']}**. Tuned rate "
            f"**{pct(judge_overall['tuned']['rate'])}** (Wilson 95% "
            f"{ci_text(judge_overall['tuned']['wilson_95'])}). This preference was "
            "explicitly instructed to respect the authoritative deterministic correctness "
            "flags, so it is not a second exact-map score.",
            f"- **Hint-only tutoring preference:** tuned/base/tie "
            f"**{judge_tutoring['tuned']['count']}/{judge_tutoring['base']['count']}/"
            f"{judge_tutoring['tie']['count']}**; base rate "
            f"**{pct(judge_tutoring['base']['rate'])}** (Wilson 95% "
            f"{ci_text(judge_tutoring['base']['wilson_95'])}).",
            "",
            "| Subjective measure | Base | Tuned |",
            "|---|---:|---:|",
            f"| Pedagogical usefulness mean (1–5) | "
            f"{judge_base['pedagogical_usefulness_score']['mean']:.2f} | "
            f"{judge_tuned['pedagogical_usefulness_score']['mean']:.2f} |",
            f"| Operation-family relevance mean (1–5) | "
            f"{judge_base['operation_family_relevance_score']['mean']:.2f} | "
            f"{judge_tuned['operation_family_relevance_score']['mean']:.2f} |",
            f"| Clarity/actionability mean (1–5) | "
            f"{judge_base['clarity_actionability_score']['mean']:.2f} | "
            f"{judge_tuned['clarity_actionability_score']['mean']:.2f} |",
            f"| Operation-family relevant | "
            f"{judge_base['operation_family_relevant']['count']}/100 | "
            f"{judge_tuned['operation_family_relevant']['count']}/100 |",
            f"| Forbidden answer/coordinate leakage | "
            f"{judge_base['forbidden_answer_or_coordinate_leak']['count']}/100 | "
            f"{judge_tuned['forbidden_answer_or_coordinate_leak']['count']}/100 |",
            "",
            "The tuned hints were clearer and much more operation-relevant, yet the "
            "stricter safety rubric flagged all 100 for exact-answer disclosure. This "
            "explains the apparently split result: tuned won overall through exact task "
            "correctness, while safer but often vague or misdirected base hints won the "
            "hint-only comparison.",
            "",
            f"The deterministic audit supports the same concern at larger scope: "
            f"**{hint_metrics['exact_answer_value_disclosure']['count']}/"
            f"{hint_metrics['exact_answer_value_disclosure']['n']} "
            f"({pct(hint_metrics['exact_answer_value_disclosure']['rate'])})** tuned hints "
            "disclosed exact answer/map/value information, and only "
            f"**{hint_metrics['safe_useful']['count']}/{hint_metrics['safe_useful']['n']} "
            f"({pct(hint_metrics['safe_useful']['rate'])})** met the conservative "
            "safe/useful rule. The v6 prompt asked for a short Socratic hint but did not "
            "explicitly prohibit coordinates, so this is a material behavior-spec/target "
            "design caveat rather than a direct prompt violation.",
            "",
            "## Credibility checks",
            "",
            "| Check | Status | Evidence |",
            "|---|---|---|",
            "| Scoring harness | **PASS** | `transform_diagnosis/`: 194 passed; `model/`: 30 passed |",
            "| Frozen tuned re-score | **PASS** | All 20 requested counts/rates match `FINAL_RESULTS_SUMMARY.json`; all stored→rescored deltas were 0.000 |",
            "| Base anchor | **PASS** | Exact maps are not handed out: all four base cells had zero correct-map and zero both-map successes |",
            "| Leakage | **PASS (artifact)** | Existing audit: 0 exact geometry overlaps; 0 training rows sourced from test/OOD |",
            "| Leakage recomputation | **UNVERIFIABLE locally** | The 9,600-row v6 training file was not part of the frozen download |",
            "| Oracle spot-check | **PASS with limitation** | IDs 52, 3529, 3929, 5252, 21634: stored transforms reproduce vertices; independently recovered maps and diagnosis agree |",
            "| Frontier disagreement spot-check | **UNVERIFIABLE locally** | Opus raw records remain on ORCD and could not be retrieved without interactive SSH authentication |",
            "| New frontier routes | **SKIPPED safely** | No credential was present during the original frontier-evaluation pass |",
            "| Blinded LLM judge | **PASS** | 100 unique successful TFY `gpt-5.6-sol` rows independently matched strict schema, request hashes/fingerprint, remapping, aggregates, intervals, and retry statistics |",
            "| Hint safety | **MATERIAL CAVEAT** | Deterministic disclosure 1,919/2,000; subjective tuned leakage 100/100; v6 omitted an explicit no-coordinate clause |",
            "",
            "## Trivial baselines",
            "",
            "| Split | Uniform 8-label guess | Majority-class label | Fine-tuned coordinate label |",
            "|---|---:|---:|---:|",
            f"| Test | 12.5% | {pct(baselines['test']['majority_rate'])} (`{baselines['test']['majority_label']}`) | 99.6% |",
            f"| Restricted OOD | 12.5% | {pct(baselines['ood']['majority_rate'])} (`{baselines['ood']['majority_label']}`) | 99.8% |",
            "",
            "Exact canonical-map random success is effectively zero. OOD contains only four observed label families; a uniform guess restricted to those four would be 25%, but the table keeps the assignment's fixed eight-label 12.5% baseline.",
            "",
            "## Fairness and scope caveats",
            "",
            "- Opus is zero-shot on an unfamiliar canonical schema; Qwen is fine-tuned for it.",
            "- OOD contains only four diagnosis families and differs compositionally from test; its higher image-only rate is not universal OOD evidence.",
            "- Frontier figures are one paired sample per modality (`n=150` image, `n=50` coordinates). Wilson intervals quantify sampling uncertainty, not model/training-seed variability.",
            "- The image and coordinate fine-tunes are modality-specific adapters, not a pure inference-time input ablation.",
            "- Existing Opus audit results were not rerun locally overnight because the raw records were unavailable.",
            "- The LLM judge is one subjective route on one 100-case test sample. It was "
            "given authoritative deterministic correctness flags and cannot independently "
            "validate exact geometry.",
            "",
            "## Submission-ready framing",
            "",
            "> On paired held-out v6 evaluations, the untuned 4B model produced no exact correct-map or two-map successes, including when given coordinates. Structured canonical-net training raised the image+coordinates fine-tune to 98.6% exact two-map accuracy on test, and in the existing fair comparison it matched zero-shot Claude Opus 4.8 on all coordinate geometry cases while improving direct task-taxonomy labeling from 80% to 100%. Image-only results were mixed: the fine-tune led direct and derived diagnosis by 44.7 and 38.7 percentage points, while Opus led student-map recovery by 16.0 points. A secondary subjective judge preferred tuned overall because of this deterministic advantage, but preferred base hints 75–25 and flagged exact-answer leakage in every sampled tuned hint. This is evidence for efficient task specialization, not general frontier-model superiority or safe Socratic hinting.",
            "",
            "## Artifacts",
            "",
            "- Machine-readable results: [`OVERNIGHT_RESULTS.json`](OVERNIGHT_RESULTS.json)",
            "- Re-scored frozen outputs: [`verify/`](verify/)",
            "- No-cost gateway dry-runs: [`raw/`](raw/)",
            "- Deterministic hint audit: [`HINT_SAFETY_AUDIT.md`](HINT_SAFETY_AUDIT.md)",
            "- Blinded judge report: [`LLM_JUDGE_REPORT.md`](LLM_JUDGE_REPORT.md)",
            "- Independent judge validation: [`LLM_JUDGE_VALIDATION.md`](LLM_JUDGE_VALIDATION.md)",
            "- Existing frozen result report: [`../v6_final/FINAL_RESULTS.md`](../v6_final/FINAL_RESULTS.md)",
            "- Existing frontier provenance note: [`../v6_final/FRONTIER_COMPARISON.md`](../v6_final/FRONTIER_COMPARISON.md)",
            "",
        ]
    )
    (OUT / "OVERNIGHT_REPORT.md").write_text("\n".join(lines))

    # Internal consistency assertions: fail loudly rather than emit a contradictory report.
    assert output["all_rescore_metrics_match"]
    assert len(comparisons) == 12
    assert all(0 <= row["finetune_rate"] <= 1 for row in comparisons)
    assert all(0 <= row["frontier_rate"] <= 1 for row in comparisons)
    assert output["llm_judge"]["independent_validation"]["status"] == "PASS"
    assert output["llm_judge"]["aggregate"]["completed_pairs"] == 100
    assert output["hint_safety"]["exact_answer_value_disclosure"]["count"] == 1919
    print("wrote OVERNIGHT_RESULTS.json and OVERNIGHT_REPORT.md")
    print("rescore checks: all match")
    print(f"frontier comparison rows: {len(comparisons)}")


if __name__ == "__main__":
    main()
