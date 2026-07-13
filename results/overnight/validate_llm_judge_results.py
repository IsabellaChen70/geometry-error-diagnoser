#!/usr/bin/env python3
"""Independent cross-artifact validation of the completed blinded LLM judge.

This script deliberately does not import ``run_llm_judge``. It reconstructs the
strict judgment schema, identity remapping, aggregates, confidence intervals,
request-set fingerprint, and retry statistics directly from persisted evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "overnight"
RAW_PATH = OUT / "LLM_JUDGE_RAW.jsonl"
META_PATH = OUT / "LLM_JUDGE_REQUEST_METADATA.json"
RESULTS_PATH = OUT / "LLM_JUDGE_RESULTS.json"
REPORT_PATH = OUT / "LLM_JUDGE_REPORT.md"
HINT_AUDIT_PATH = OUT / "HINT_SAFETY_AUDIT.json"
OVERNIGHT_RESULTS_PATH = OUT / "OVERNIGHT_RESULTS.json"
OVERNIGHT_REPORT_PATH = OUT / "OVERNIGHT_REPORT.md"
CHECKLIST_PATH = OUT / "SUBMISSION_CHECKLIST.md"
RUN_LOG_PATH = OUT / "RUN_LOG.md"
JSON_OUT = OUT / "LLM_JUDGE_VALIDATION.json"
MD_OUT = OUT / "LLM_JUDGE_VALIDATION.md"

SCHEMA_VERSION = "v6.llm-judge.1"
SCORE_FIELDS = (
    "pedagogical_usefulness_score",
    "operation_family_relevance_score",
    "clarity_actionability_score",
)
BOOL_FIELDS = (
    "operation_family_relevant",
    "forbidden_answer_or_coordinate_leak",
)
LEAK_TYPES = {
    "coordinate_pair",
    "exact_correct_map",
    "exact_student_map",
    "exact_translation_value",
    "exact_reflection_line",
    "exact_rotation_angle",
    "other",
}
PREFERENCES = {"A", "B", "tie"}
ROW_KEYS = {
    "schema_version",
    "run_fingerprint",
    "case_id",
    "request_hash",
    "provider",
    "model",
    "status",
    "attempt_count",
    "attempts",
    "judgment",
}
JUDGMENT_KEYS = {
    "schema_version",
    "case_id",
    "candidate_A",
    "candidate_B",
    "tutoring_preference",
    "overall_preference",
    "preference_reason",
    "deterministic_scores_acknowledged",
}
CANDIDATE_KEYS = {
    *SCORE_FIELDS,
    *BOOL_FIELDS,
    "leak_types",
    "rationale",
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    assert isinstance(value, dict)
    return value


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def extract_final_object(text: object) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    try:
        value = json.loads(text.strip())
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        return value
    decoder = json.JSONDecoder()
    found = None
    best_end = -1
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        end = index + consumed
        if isinstance(value, dict) and end >= best_end:
            found = value
            best_end = end
    return found


def validate_candidate(value: object) -> None:
    assert isinstance(value, dict) and set(value) == CANDIDATE_KEYS
    for field in SCORE_FIELDS:
        score = value[field]
        assert not isinstance(score, bool)
        assert isinstance(score, int) and 1 <= score <= 5
    for field in BOOL_FIELDS:
        assert isinstance(value[field], bool)
    leaks = value["leak_types"]
    assert isinstance(leaks, list)
    assert len(leaks) == len(set(leaks))
    assert all(isinstance(item, str) and item in LEAK_TYPES for item in leaks)
    assert value["forbidden_answer_or_coordinate_leak"] is bool(leaks)
    assert isinstance(value["rationale"], str) and value["rationale"].strip()
    assert len(value["rationale"]) <= 500


def validate_judgment(value: object, case_id: object) -> None:
    assert isinstance(value, dict) and set(value) == JUDGMENT_KEYS
    assert value["schema_version"] == SCHEMA_VERSION
    assert value["case_id"] == case_id
    validate_candidate(value["candidate_A"])
    validate_candidate(value["candidate_B"])
    assert value["tutoring_preference"] in PREFERENCES
    assert value["overall_preference"] in PREFERENCES
    assert isinstance(value["preference_reason"], str)
    assert value["preference_reason"].strip()
    assert len(value["preference_reason"]) <= 500
    assert value["deterministic_scores_acknowledged"] is True


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    proportion = hits / n
    denominator = 1 + z * z / n
    center = (proportion + z * z / (2 * n)) / denominator
    margin = (
        z
        * math.sqrt(
            (proportion * (1 - proportion) + z * z / (4 * n)) / n
        )
        / denominator
    )
    return [max(0.0, center - margin), min(1.0, center + margin)]


def mean_ci(values: Sequence[float]) -> dict:
    mean = statistics.fmean(values)
    margin = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
    return {
        "mean": mean,
        "n": len(values),
        "normal_95": [max(1.0, mean - margin), min(5.0, mean + margin)],
    }


def assert_close(left: object, right: object, tolerance: float = 1e-12) -> None:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        assert set(left) == set(right)
        for key in left:
            assert_close(left[key], right[key], tolerance)
        return
    if isinstance(left, list) and isinstance(right, list):
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right):
            assert_close(left_item, right_item, tolerance)
        return
    if isinstance(left, (int, float)) and not isinstance(left, bool):
        assert isinstance(right, (int, float)) and not isinstance(right, bool)
        assert math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance)
        return
    assert left == right


def preference_identity(assignment: Mapping[str, str], preference: str) -> str:
    return "tie" if preference == "tie" else assignment[preference]


def rate_summary(counts: Mapping[str, int], n: int) -> dict:
    return {
        identity: {
            "count": counts[identity],
            "n": n,
            "rate": counts[identity] / n,
            "wilson_95": wilson(counts[identity], n),
        }
        for identity in ("tuned", "base", "tie")
    }


def pct(rate: float) -> str:
    return f"{100 * rate:.1f}%"


def ci_pct(interval: Sequence[float]) -> str:
    return f"{100 * interval[0]:.1f}–{100 * interval[1]:.1f}%"


def main() -> None:
    raw_rows = load_jsonl(RAW_PATH)
    metadata = load_json(META_PATH)
    saved_results = load_json(RESULTS_PATH)
    hint_audit = load_json(HINT_AUDIT_PATH)
    overnight_results = load_json(OVERNIGHT_RESULTS_PATH)
    report_text = REPORT_PATH.read_text(encoding="utf-8")
    overnight_report_text = OVERNIGHT_REPORT_PATH.read_text(encoding="utf-8")
    checklist_text = CHECKLIST_PATH.read_text(encoding="utf-8")
    run_log_text = RUN_LOG_PATH.read_text(encoding="utf-8")

    assert metadata["schema_version"] == SCHEMA_VERSION
    assert saved_results["schema_version"] == SCHEMA_VERSION
    assert saved_results["status"] == "PASS"
    assert metadata["provider"] == saved_results["provider"] == "tfy"
    assert metadata["model"] == saved_results["model"] == "gpt-5.6-sol"
    assert metadata["seed"] == saved_results["sample_seed"] == 20260709
    assert metadata["sample_size"] == saved_results["requested_pairs"] == 100

    sample_ids = metadata["sample_ids"]
    assert len(sample_ids) == len(set(sample_ids)) == 100
    request_hashes = metadata["request_hashes"]
    assert set(request_hashes) == {str(case_id) for case_id in sample_ids}
    assert len(set(request_hashes.values())) == 100

    request_set_hash = sha256_text(
        json.dumps(
            [request_hashes[str(case_id)] for case_id in sample_ids],
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    assert request_set_hash == metadata["request_set_sha256"]
    fingerprint_payload = {
        "schema_version": SCHEMA_VERSION,
        "seed": metadata["seed"],
        "sample_ids": sample_ids,
        "provider": metadata["provider"],
        "model": metadata["model"],
        "base_url": metadata["base_url"],
        "max_output_tokens": metadata["max_output_tokens"],
        "system_prompt_sha256": metadata["system_prompt_sha256"],
        "request_set_sha256": metadata["request_set_sha256"],
    }
    recomputed_fingerprint = sha256_text(
        json.dumps(
            fingerprint_payload,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    assert recomputed_fingerprint == metadata["run_fingerprint"]

    for relative_path, expected_hash in metadata["input_sha256"].items():
        assert sha256_file(ROOT / relative_path) == expected_hash

    assignments = {
        row["case_id"]: {"A": row["A_identity"], "B": row["B_identity"]}
        for row in metadata["blind_assignment"]
    }
    assert set(assignments) == set(sample_ids)
    assert all(set(value.values()) == {"base", "tuned"} for value in assignments.values())
    side_balance = {
        "base_as_A": sum(value["A"] == "base" for value in assignments.values()),
        "base_as_B": sum(value["B"] == "base" for value in assignments.values()),
        "tuned_as_A": sum(value["A"] == "tuned" for value in assignments.values()),
        "tuned_as_B": sum(value["B"] == "tuned" for value in assignments.values()),
    }
    assert side_balance == {
        "base_as_A": 50,
        "base_as_B": 50,
        "tuned_as_A": 50,
        "tuned_as_B": 50,
    }
    assert side_balance == metadata["blinded_side_balance"]

    successful = [row for row in raw_rows if row.get("status") == "success"]
    failed = [row for row in raw_rows if row.get("status") == "failed"]
    assert len(raw_rows) == 106
    assert len(successful) == 100
    assert len(failed) == 6
    success_by_id = {row["case_id"]: row for row in successful}
    assert len(success_by_id) == 100
    assert set(success_by_id) == set(sample_ids)

    overall_counts: Counter[str] = Counter()
    tutoring_counts: Counter[str] = Counter()
    score_values = {
        identity: {field: [] for field in SCORE_FIELDS}
        for identity in ("base", "tuned")
    }
    bool_values = {
        identity: {field: [] for field in BOOL_FIELDS}
        for identity in ("base", "tuned")
    }
    leak_type_counts = {
        identity: Counter() for identity in ("base", "tuned")
    }
    success_attempts = 0
    retried_cases = 0
    success_parse_failures = 0
    success_request_errors = 0

    for case_id in sample_ids:
        row = success_by_id[case_id]
        assert set(row) == ROW_KEYS
        assert row["schema_version"] == SCHEMA_VERSION
        assert row["provider"] == metadata["provider"]
        assert row["model"] == metadata["model"]
        assert row["run_fingerprint"] == metadata["run_fingerprint"]
        assert row["request_hash"] == request_hashes[str(case_id)]
        assert row["attempt_count"] == len(row["attempts"])
        assert 1 <= row["attempt_count"] <= metadata["max_retries"]
        assert [attempt["attempt"] for attempt in row["attempts"]] == list(
            range(1, row["attempt_count"] + 1)
        )

        final_attempt = row["attempts"][-1]
        assert final_attempt["request_error"] is None
        assert final_attempt["parse_error"] is None
        parsed_response = extract_final_object(final_attempt["response_text"])
        assert parsed_response == row["judgment"]
        validate_judgment(row["judgment"], case_id)

        success_attempts += row["attempt_count"]
        retried_cases += row["attempt_count"] > 1
        success_parse_failures += sum(
            attempt["parse_error"] is not None for attempt in row["attempts"]
        )
        success_request_errors += sum(
            attempt["request_error"] is not None for attempt in row["attempts"]
        )

        judgment = row["judgment"]
        assignment = assignments[case_id]
        overall_counts[
            preference_identity(assignment, judgment["overall_preference"])
        ] += 1
        tutoring_counts[
            preference_identity(assignment, judgment["tutoring_preference"])
        ] += 1
        for side in ("A", "B"):
            identity = assignment[side]
            candidate = judgment[f"candidate_{side}"]
            for field in SCORE_FIELDS:
                score_values[identity][field].append(float(candidate[field]))
            for field in BOOL_FIELDS:
                bool_values[identity][field].append(bool(candidate[field]))
            leak_type_counts[identity].update(candidate["leak_types"])

    recomputed_aggregate = {
        "completed_pairs": 100,
        "overall_preference": rate_summary(overall_counts, 100),
        "tutoring_preference": rate_summary(tutoring_counts, 100),
        "rubric_by_model_identity": {},
        "blinded_side_balance": side_balance,
        "request_statistics": {
            "successful_rows": 100,
            "total_attempts": success_attempts,
            "retried_cases": retried_cases,
            "strict_parse_failures_before_success": success_parse_failures,
            "request_errors_before_success": success_request_errors,
        },
    }
    for identity in ("base", "tuned"):
        rubric = {}
        for field in SCORE_FIELDS:
            rubric[field] = mean_ci(score_values[identity][field])
        for field in BOOL_FIELDS:
            values = bool_values[identity][field]
            hits = sum(values)
            rubric[field] = {
                "count": hits,
                "n": len(values),
                "rate": hits / len(values),
                "wilson_95": wilson(hits, len(values)),
            }
        recomputed_aggregate["rubric_by_model_identity"][identity] = rubric

    assert_close(recomputed_aggregate, saved_results["aggregate"])
    assert overall_counts == {"tuned": 100}
    assert tutoring_counts == {"base": 75, "tuned": 25}
    assert success_attempts == 122
    assert retried_cases == 20
    assert success_parse_failures == 15
    assert success_request_errors == 7

    cumulative = {
        "raw_rows": len(raw_rows),
        "judge_case_attempts": sum(len(row["attempts"]) for row in raw_rows),
        "successful_rows": len(successful),
        "failed_rows": len(failed),
    }
    assert cumulative == saved_results["persisted_attempts_cumulative"]
    assert cumulative == {
        "raw_rows": 106,
        "judge_case_attempts": 140,
        "successful_rows": 100,
        "failed_rows": 6,
    }
    assert saved_results["retry_failed_required"] is False

    required_report_fragments = (
        "Status: **PASS**",
        "100/100",
        "100 / 0 / 0",
        "25 / 75 / 0",
        "latest successful rows: **122**",
        "retried cases: **20**",
        "strict parse failures before success: **15**",
        "request errors before success: **7**",
        "**140** in **106** raw rows",
        "TFY / `gpt-5.6-sol`",
        "| Forbidden answer/coordinate leakage | 1/100 | 100/100 |",
        "material tutoring-safety and specification concern",
    )
    assert all(fragment in report_text for fragment in required_report_fragments)
    assert_close(
        overnight_results["llm_judge"]["aggregate"],
        saved_results["aggregate"],
    )
    assert overnight_results["llm_judge"]["independent_validation"]["status"] == "PASS"
    assert overnight_results["hint_safety"]["exact_answer_value_disclosure"]["count"] == 1919
    assert "Overall preference:** tuned/base/tie **100/0/0**" in overnight_report_text
    assert "Hint-only tutoring preference:** tuned/base/tie **25/75/0**" in overnight_report_text
    assert "Forbidden answer/coordinate leakage | 1/100 | 100/100" in overnight_report_text
    assert "**PASS — blinded LLM-as-judge evaluation completed" in checklist_text
    assert "- **PASS: 12**" in checklist_text
    assert "- **BLOCKED: 0**" in checklist_text
    assert "- **FAIL: 0**" in checklist_text
    tldr = run_log_text.split("Started:", 1)[0]
    assert "gpt-5.6-sol" in tldr
    assert "12 PASS / 0 BLOCKED / 0 FAIL" in tldr
    assert "judge is **BLOCKED**" not in tldr

    saved_rubric = recomputed_aggregate["rubric_by_model_identity"]
    validation = {
        "schema_version": "v6.llm-judge-validation.1",
        "status": "PASS",
        "provider": metadata["provider"],
        "model": metadata["model"],
        "sample": {
            "seed": metadata["seed"],
            "split": metadata["split"],
            "modality": metadata["modality"],
            "paired_ids": 100,
        },
        "evidence": {
            "raw_rows": len(raw_rows),
            "successful_rows": len(successful),
            "failed_rows_preserved": len(failed),
            "successful_unique_ids": len(success_by_id),
            "successful_request_fingerprint": metadata["run_fingerprint"],
            "successful_request_set_sha256": metadata["request_set_sha256"],
        },
        "checks": {
            "json_and_jsonl_parse": True,
            "success_ids_unique_and_match_sample": True,
            "request_hashes_match_metadata": True,
            "request_set_hash_matches": True,
            "run_fingerprint_recomputed": True,
            "input_hashes_match": True,
            "provider_model_consistent": True,
            "strict_judgment_schema_all_100": True,
            "final_response_matches_saved_judgment_all_100": True,
            "ab_identity_balance_50_50": True,
            "preference_identity_remapping_matches": True,
            "counts_rates_and_intervals_match": True,
            "rubric_means_and_intervals_match": True,
            "retry_error_statistics_match": True,
            "cumulative_raw_evidence_matches": True,
            "report_core_values_match": True,
            "integrated_overnight_results_match": True,
            "integrated_overnight_report_matches": True,
            "checklist_tally_matches": True,
            "run_log_tldr_current": True,
        },
        "recomputed": {
            **recomputed_aggregate,
            "leak_type_counts_by_model_identity": {
                identity: dict(sorted(counts.items()))
                for identity, counts in leak_type_counts.items()
            },
        },
        "hint_safety_reconciliation": {
            "deterministic_audit_scope": hint_audit["overall"]["n"],
            "deterministic_exact_answer_value_disclosure": hint_audit["overall"][
                "metrics"
            ]["exact_answer_value_disclosure"],
            "deterministic_safe_useful": hint_audit["overall"]["metrics"][
                "safe_useful"
            ],
            "judge_sample_tuned_leak": saved_rubric["tuned"][
                "forbidden_answer_or_coordinate_leak"
            ],
            "judge_sample_base_leak": saved_rubric["base"][
                "forbidden_answer_or_coordinate_leak"
            ],
            "v6_prompt_explicitly_forbids_coordinates": hint_audit[
                "contract_boundary"
            ]["v6_prompt_explicitly_forbids_coordinates"],
            "interpretation": (
                "The subjective 100-pair judge and deterministic 2,000-output audit "
                "agree that tuned hints are highly operation-relevant but routinely "
                "disclose exact answer parameters under the stricter tutoring-safety "
                "rubric. The v6 prompt did not explicitly prohibit coordinates, so this "
                "is a safety/specification concern rather than a direct prompt violation."
            ),
        },
    }

    base = saved_rubric["base"]
    tuned = saved_rubric["tuned"]
    overall = recomputed_aggregate["overall_preference"]
    tutoring = recomputed_aggregate["tutoring_preference"]
    markdown = "\n".join(
        [
            "# Independent validation of the blinded LLM judge",
            "",
            "Status: **PASS**. All 100 successful paired judgments independently "
            "reproduced the saved aggregates and report core values.",
            "",
            "## Provenance and integrity",
            "",
            "- Provider/model: **TFY / `gpt-5.6-sol`**.",
            "- Sample: **100 paired image+coordinates test IDs**, seed `20260709`; "
            "A/B identity was balanced **50/50**.",
            "- Raw evidence: **106 rows** preserving **100 successful** and **6 failed** "
            "rows; the latest success set contains 100 unique sample IDs.",
            "- Every successful row matched its metadata request hash and the recomputed "
            "run fingerprint; all 100 final response objects passed the strict schema.",
            "",
            "## Recomputed substantive result",
            "",
            f"- Overall preference, which was instructed to respect authoritative exact "
            f"scoring: tuned **100/100** ({pct(overall['tuned']['rate'])}, Wilson 95% "
            f"{ci_pct(overall['tuned']['wilson_95'])}); base **0/100**; ties **0/100**.",
            f"- Hint-only tutoring preference: base **75/100** "
            f"({pct(tutoring['base']['rate'])}, Wilson 95% "
            f"{ci_pct(tutoring['base']['wilson_95'])}); tuned **25/100** "
            f"({pct(tutoring['tuned']['rate'])}, Wilson 95% "
            f"{ci_pct(tutoring['tuned']['wilson_95'])}); ties **0/100**.",
            f"- Tuned versus base means: pedagogical usefulness "
            f"**{tuned['pedagogical_usefulness_score']['mean']:.2f} vs "
            f"{base['pedagogical_usefulness_score']['mean']:.2f}**; operation-family "
            f"relevance **{tuned['operation_family_relevance_score']['mean']:.2f} vs "
            f"{base['operation_family_relevance_score']['mean']:.2f}**; clarity/actionability "
            f"**{tuned['clarity_actionability_score']['mean']:.2f} vs "
            f"{base['clarity_actionability_score']['mean']:.2f}**.",
            f"- Operation-family relevance flags: tuned **100/100** versus base "
            f"**50/100**. Forbidden answer/coordinate leakage flags: tuned "
            f"**100/100** versus base **1/100**.",
            "",
            "The result is not contradictory: the tuned outputs won overall because their "
            "deterministic geometry and diagnosis were much stronger, while the base hints "
            "won the hint-only comparison mainly by avoiding exact-answer disclosure. "
            "LLM judgments are subjective and secondary to deterministic scoring.",
            "",
            "## Reliability",
            "",
            "- Latest successful rows used **122 attempts**: **20** cases retried, with "
            "**15** strict-parse failures and **7** request errors before success.",
            "- Cumulative preserved evidence contains **140 attempts** across 106 rows.",
            "- Every count, rate, rubric mean, normal mean interval, Wilson interval, and "
            "retry statistic matched `LLM_JUDGE_RESULTS.json` within `1e-12`.",
            "",
            "## Hint-safety reconciliation",
            "",
            "- The deterministic audit found exact answer/map/value disclosure in "
            "**1,919/2,000 (96.0%)** tuned hints and only **28/2,000 (1.4%)** met its "
            "conservative safe/useful rule.",
            "- The judge independently flagged tuned leakage in **100/100** sampled hints, "
            "while also rating tuned operation relevance and clarity much higher.",
            "- The actual v6 prompt requested a short Socratic hint but did not explicitly "
            "forbid coordinates. Therefore this is a material submission caveat and "
            "behavior-spec gap, not evidence that the model violated an explicit v6 clause.",
            "",
            "Reproduce without API access:",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python3 "
            "results/overnight/validate_llm_judge_results.py",
            "```",
            "",
        ]
    )

    json_text = json.dumps(validation, indent=2, ensure_ascii=False) + "\n"
    assert json.loads(json_text)["status"] == "PASS"
    atomic_text(JSON_OUT, json_text)
    atomic_text(MD_OUT, markdown)
    print("LLM judge validation: PASS")
    print("validated 100 unique successful judgments from 106 preserved raw rows")
    print("recomputed all counts, rates, means, intervals, and retry statistics")
    print(f"wrote {JSON_OUT} and {MD_OUT}")


if __name__ == "__main__":
    main()
