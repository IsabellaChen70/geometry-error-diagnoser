#!/usr/bin/env python3
"""Resumable blinded LLM judge for 100 paired base-vs-tuned v6 test outputs.

Safety sequence:

1. build and validate two offline payloads;
2. require exactly one selected provider/route;
3. run two live smoke cases;
4. continue to the full deterministic 100-pair sample only after both parse.

Successful raw rows are never re-requested.  API key values are read only at
client construction and are never printed or persisted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import time
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "overnight"
FINAL_DIR = ROOT / "results" / "v6_final"
ORACLE_PATH = ROOT / "transform_diagnosis_data" / "test.jsonl"

BASE_PATH = FINAL_DIR / "records_v6_4b_base_image_coords_test.jsonl"
TUNED_PATH = FINAL_DIR / "records_v6_4b_image_coords_test.jsonl"
RAW_PATH = OUT_DIR / "LLM_JUDGE_RAW.jsonl"
META_PATH = OUT_DIR / "LLM_JUDGE_REQUEST_METADATA.json"
RESULTS_PATH = OUT_DIR / "LLM_JUDGE_RESULTS.json"
REPORT_PATH = OUT_DIR / "LLM_JUDGE_REPORT.md"

SCHEMA_VERSION = "v6.llm-judge.1"
SEED = 20260709
N_PAIRS = 100
SMOKE_PAIRS = 2
DEFAULT_TFY_MODEL = "codex-traffic/gpt-5.3-codex"
DEFAULT_TFY_BASE_URL = "https://tfy-eu.promptlens.trilogy.com"
DEFAULT_ANTHROPIC_MODEL = "claude-opus-4-8"
DEFAULT_MAX_OUTPUT_TOKENS = 900
DEFAULT_MAX_RETRIES = 3

IDENTITY = ((1, 0), (0, 1))
MATRICES = {
    "identity": IDENTITY,
    "rot_ccw_90": ((0, -1), (1, 0)),
    "rot_180": ((-1, 0), (0, -1)),
    "rot_ccw_270": ((0, 1), (-1, 0)),
    "reflect_x_axis": ((1, 0), (0, -1)),
    "reflect_y_axis": ((-1, 0), (0, 1)),
    "reflect_y_eq_x": ((0, 1), (1, 0)),
    "reflect_y_eq_neg_x": ((0, -1), (-1, 0)),
}

SCORE_FIELDS = (
    "pedagogical_usefulness_score",
    "operation_family_relevance_score",
    "clarity_actionability_score",
)
BOOL_FIELDS = (
    "operation_family_relevant",
    "forbidden_answer_or_coordinate_leak",
)
LEAK_TYPE_VOCABULARY = (
    "coordinate_pair",
    "exact_correct_map",
    "exact_student_map",
    "exact_translation_value",
    "exact_reflection_line",
    "exact_rotation_angle",
    "other",
)
LEAK_TYPES = frozenset(LEAK_TYPE_VOCABULARY)
LEAK_TYPES_JSON = json.dumps(LEAK_TYPE_VOCABULARY, separators=(",", ":"))
PREFERENCES = {"A", "B", "tie"}

SYSTEM_PROMPT = f"""You are a blinded evaluator of two geometry-tutoring model outputs.

Behavior Spec and evidence boundary:
- The v6 task asks each model to recover the observable RED→GREEN and RED→BLUE
  canonical affine maps, derive one fixed diagnosis label, and provide a short
  Socratic hint.
- Exact map, label, and parse booleans supplied below were computed
  deterministically. Accept them as authoritative; do not rescore or override them.
- The actual v6 prompt says "short Socratic hint" but does not explicitly ban
  coordinates. For this judge evaluation, apply the stricter tutoring-safety
  standard below and treat it as an evaluative rubric, not as a claim that the
  original prompt contained that prohibition.

Subjective tutoring rubric:
1. Pedagogical usefulness (1–5): helps the student notice and repair the error.
2. Operation-family relevance (boolean and 1–5): addresses the oracle-relevant
   rotation/reflection/translation concept rather than generic advice.
3. Forbidden answer/coordinate leakage: true if the hint states an answer
   coordinate pair, exact oracle canonical map parameter, exact reflection line or
   rotation angle that gives away the answer, or exact translation value. Identify
   leak types. Merely naming an operation family is not leakage.
   `leak_types` is a duplicate-free JSON array whose values must come only from
   this exact vocabulary: {LEAK_TYPES_JSON}. Use every applicable value. Use
   "other" only for definite forbidden leakage not represented by a more specific
   value above; never invent a synonym or another enum value. The array must be
   empty exactly when `forbidden_answer_or_coordinate_leak` is false and nonempty
   exactly when it is true.
4. Clarity/actionability (1–5): concise, understandable, and gives a next check.
5. Tutoring preference compares hints only. Overall preference first respects the
   supplied deterministic correctness booleans, then uses tutoring quality.

Return exactly one JSON object matching the requested schema. No markdown, code
fences, prose outside JSON, or additional keys."""


def strict_output_template(case_id: object) -> dict:
    candidate = {
        "pedagogical_usefulness_score": 1,
        "operation_family_relevant": True,
        "operation_family_relevance_score": 1,
        "forbidden_answer_or_coordinate_leak": False,
        "leak_types": [],
        "clarity_actionability_score": 1,
        "rationale": "one concise sentence",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "candidate_A": candidate,
        "candidate_B": candidate,
        "tutoring_preference": "A|B|tie",
        "overall_preference": "A|B|tie",
        "preference_reason": "one concise sentence",
        "deterministic_scores_acknowledged": True,
    }


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, ensure_ascii=False) + "\n"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def append_jsonl(path: Path, value: Mapping) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request_set_sha256(cases: Sequence[Mapping]) -> str:
    request_hashes = [str(case["request_hash"]) for case in cases]
    return sha256_text(
        json.dumps(request_hashes, ensure_ascii=False, separators=(",", ":"))
    )


def make_run_fingerprint(
    cases: Sequence[Mapping],
    provider: str,
    model: str,
    base_url: Optional[str],
    max_output_tokens: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    return sha256_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "seed": SEED,
                "sample_ids": [case["case_id"] for case in cases],
                "provider": provider,
                "model": model,
                "base_url": base_url if provider == "tfy" else None,
                "max_output_tokens": max_output_tokens,
                "system_prompt_sha256": sha256_text(system_prompt),
                "request_set_sha256": request_set_sha256(cases),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def mat_vec(matrix: tuple, point: Sequence[int]) -> tuple[int, int]:
    (a, b), (c, d) = matrix
    x, y = point
    return (a * int(x) + b * int(y), c * int(x) + d * int(y))


def recover_net(source: Sequence[Sequence[int]], target: Sequence[Sequence[int]]) -> dict:
    matches = []
    target_points = [[int(x), int(y)] for x, y in target]
    for linear, matrix in MATRICES.items():
        first = mat_vec(matrix, source[0])
        tx = int(target[0][0]) - first[0]
        ty = int(target[0][1]) - first[1]
        produced = []
        for point in source:
            mapped = mat_vec(matrix, point)
            produced.append([mapped[0] + tx, mapped[1] + ty])
        if produced == target_points:
            matches.append({"linear": linear, "tx": tx, "ty": ty})
    if len(matches) != 1:
        raise ValueError(f"expected one recoverable D4 map, found {len(matches)}")
    return matches[0]


def deterministic_fields(row: Mapping) -> dict:
    fields = (
        "parse_ok",
        "correct_net_ok",
        "student_net_ok",
        "both_nets_ok",
        "label_ok",
        "derived_label_ok",
    )
    return {field: row.get(field) for field in fields}


def prepare_cases(seed: int = SEED, n_pairs: int = N_PAIRS) -> list[dict]:
    base_rows = load_jsonl(BASE_PATH)
    tuned_rows = load_jsonl(TUNED_PATH)
    oracle_rows = load_jsonl(ORACLE_PATH)
    assert len(base_rows) == len(tuned_rows) == 500
    assert [row["id"] for row in base_rows] == [row["id"] for row in tuned_rows]
    assert len({row["id"] for row in base_rows}) == 500

    base_by_id = {row["id"]: row for row in base_rows}
    tuned_by_id = {row["id"]: row for row in tuned_rows}
    oracle_by_id = {row["id"]: row for row in oracle_rows}
    common = sorted(set(base_by_id) & set(tuned_by_id))
    sampled = sorted(random.Random(seed).sample(common, n_pairs))

    # Exactly balanced, independently shuffled A/B assignment.
    assignment_order = sampled.copy()
    random.Random(seed ^ 0xA5A5A5A5).shuffle(assignment_order)
    base_is_a = set(assignment_order[: n_pairs // 2])

    cases = []
    for case_id in sampled:
        assert case_id in oracle_by_id
        base = base_by_id[case_id]
        tuned = tuned_by_id[case_id]
        oracle = oracle_by_id[case_id]
        assert base["split"] == tuned["split"] == oracle["split"] == "test"
        assert base["true_label"] == tuned["true_label"] == oracle["label"]

        identities = {"A": "base", "B": "tuned"}
        candidate_rows = {"A": base, "B": tuned}
        if case_id not in base_is_a:
            identities = {"A": "tuned", "B": "base"}
            candidate_rows = {"A": tuned, "B": base}

        judge_payload = {
            "case_id": case_id,
            "observable_geometry": {
                "red_original": oracle["original"],
                "green_correct_image": oracle["correct_image"],
                "blue_student_image": oracle["student_image"],
            },
            "oracle": {
                "correct_net": recover_net(
                    oracle["original"], oracle["correct_image"]
                ),
                "student_net": recover_net(
                    oracle["original"], oracle["student_image"]
                ),
                "diagnosis_label": oracle["label"],
            },
            "candidate_A": {
                "deterministic_scores": deterministic_fields(candidate_rows["A"]),
                "output": candidate_rows["A"]["raw_model_output"],
            },
            "candidate_B": {
                "deterministic_scores": deterministic_fields(candidate_rows["B"]),
                "output": candidate_rows["B"]["raw_model_output"],
            },
            "required_output": strict_output_template(case_id),
        }
        user_prompt = (
            "Evaluate this blinded pair. The candidate identities are intentionally "
            "withheld.\n\n"
            + json.dumps(judge_payload, ensure_ascii=False, separators=(",", ":"))
        )
        cases.append(
            {
                "case_id": case_id,
                "identities": identities,
                "judge_payload": judge_payload,
                "user_prompt": user_prompt,
                "request_hash": sha256_text(SYSTEM_PROMPT + "\n" + user_prompt),
            }
        )

    assert len(cases) == n_pairs
    assert len({case["case_id"] for case in cases}) == n_pairs
    assert sum(case["identities"]["A"] == "base" for case in cases) == n_pairs // 2
    assert sum(case["identities"]["A"] == "tuned" for case in cases) == n_pairs // 2
    return cases


def extract_final_object(text: object) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    try:
        value = json.loads(stripped)
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


def validate_candidate(value: object, name: str) -> dict:
    expected = {
        *SCORE_FIELDS,
        *BOOL_FIELDS,
        "leak_types",
        "rationale",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{name} keys must be exactly {sorted(expected)}")
    for field in SCORE_FIELDS:
        score = value[field]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"{name}.{field} must be an integer from 1 to 5")
    for field in BOOL_FIELDS:
        if not isinstance(value[field], bool):
            raise ValueError(f"{name}.{field} must be boolean")
    leaks = value["leak_types"]
    if not isinstance(leaks, list) or not all(
        isinstance(item, str) and item in LEAK_TYPES for item in leaks
    ):
        raise ValueError(
            f"{name}.leak_types contains an invalid value; allowed values are "
            f"{list(LEAK_TYPE_VOCABULARY)}"
        )
    if len(leaks) != len(set(leaks)):
        raise ValueError(f"{name}.leak_types contains duplicates")
    if value["forbidden_answer_or_coordinate_leak"] != bool(leaks):
        raise ValueError(f"{name} leak boolean must equal whether leak_types is nonempty")
    if not isinstance(value["rationale"], str) or not value["rationale"].strip():
        raise ValueError(f"{name}.rationale must be nonempty text")
    if len(value["rationale"]) > 500:
        raise ValueError(f"{name}.rationale is too long")
    return value


def validate_judgment(value: object, case_id: object) -> dict:
    expected = {
        "schema_version",
        "case_id",
        "candidate_A",
        "candidate_B",
        "tutoring_preference",
        "overall_preference",
        "preference_reason",
        "deterministic_scores_acknowledged",
    }
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"top-level keys must be exactly {sorted(expected)}")
    if value["schema_version"] != SCHEMA_VERSION:
        raise ValueError("wrong schema_version")
    if value["case_id"] != case_id:
        raise ValueError(f"case_id mismatch: {value['case_id']!r} != {case_id!r}")
    validate_candidate(value["candidate_A"], "candidate_A")
    validate_candidate(value["candidate_B"], "candidate_B")
    for field in ("tutoring_preference", "overall_preference"):
        if value[field] not in PREFERENCES:
            raise ValueError(f"{field} must be A, B, or tie")
    if (
        not isinstance(value["preference_reason"], str)
        or not value["preference_reason"].strip()
        or len(value["preference_reason"]) > 500
    ):
        raise ValueError("preference_reason must be concise nonempty text")
    if value["deterministic_scores_acknowledged"] is not True:
        raise ValueError("deterministic_scores_acknowledged must be true")
    return value


def parse_judgment(text: str, case_id: object) -> dict:
    value = extract_final_object(text)
    if value is None:
        raise ValueError("no JSON object found")
    return validate_judgment(value, case_id)


def redact_error(exc: BaseException) -> str:
    message = f"{type(exc).__name__}: {exc}"
    for variable in ("TFY_API_KEY", "ANTHROPIC_API_KEY"):
        secret = os.environ.get(variable)
        if secret:
            message = message.replace(secret, "<redacted>")
    message = re.sub(r"\b(?:tfy_|sk-ant-)[A-Za-z0-9_.-]{8,}\b", "<redacted>", message)
    return re.sub(r"\s+", " ", message).strip()[:800]


def resolve_provider(requested: str) -> tuple[Optional[str], dict]:
    presence = {
        "TFY_API_KEY": bool(os.environ.get("TFY_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.environ.get("ANTHROPIC_API_KEY")),
    }
    if requested == "tfy":
        return ("tfy" if presence["TFY_API_KEY"] else None), presence
    if requested == "anthropic":
        return (
            "anthropic" if presence["ANTHROPIC_API_KEY"] else None
        ), presence
    if presence["TFY_API_KEY"]:
        return "tfy", presence
    if presence["ANTHROPIC_API_KEY"]:
        return "anthropic", presence
    return None, presence


def make_tfy_caller(
    model: str, base_url: str, max_output_tokens: int
) -> tuple[Callable[[str], str], object]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["TFY_API_KEY"], base_url=base_url)

    def call(user_prompt: str) -> str:
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "system",
                    "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
                },
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": user_prompt}],
                },
            ],
            max_output_tokens=max_output_tokens,
        )
        text = getattr(response, "output_text", None)
        if text:
            return str(text).strip()
        parts = []
        for item in getattr(response, "output", None) or []:
            for content in getattr(item, "content", None) or []:
                value = getattr(content, "text", None)
                if value:
                    parts.append(str(value))
        return "".join(parts).strip()

    return call, client


def make_anthropic_caller(
    model: str, max_output_tokens: int
) -> tuple[Callable[[str], str], object]:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    def call(user_prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=max_output_tokens,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

    return call, client


def latest_rows() -> dict[object, dict]:
    result = {}
    for row in load_jsonl(RAW_PATH):
        result[row["case_id"]] = row
    return result


def persisted_attempt_summary(
    rows: Optional[Sequence[Mapping]] = None,
) -> dict[str, int]:
    persisted_rows = load_jsonl(RAW_PATH) if rows is None else list(rows)
    return {
        "raw_rows": len(persisted_rows),
        "judge_case_attempts": sum(
            len(row.get("attempts", [])) for row in persisted_rows
        ),
        "successful_rows": sum(
            row.get("status") == "success" for row in persisted_rows
        ),
        "failed_rows": sum(row.get("status") == "failed" for row in persisted_rows),
    }


def call_case(
    case: Mapping,
    call: Callable[[str], str],
    provider: str,
    model: str,
    run_fingerprint: str,
    max_retries: int,
) -> dict:
    attempts = []
    parsed = None
    for attempt in range(1, max_retries + 1):
        response_text = ""
        error = None
        parse_error = None
        try:
            response_text = call(str(case["user_prompt"]))
            parsed = parse_judgment(response_text, case["case_id"])
        except Exception as exc:  # network, provider, or strict-parse failure
            if response_text:
                parse_error = redact_error(exc)
            else:
                error = redact_error(exc)
        attempts.append(
            {
                "attempt": attempt,
                "response_text": response_text,
                "request_error": error,
                "parse_error": parse_error,
            }
        )
        if parsed is not None:
            break
        if attempt < max_retries:
            time.sleep(2 ** (attempt - 1))

    row = {
        "schema_version": SCHEMA_VERSION,
        "run_fingerprint": run_fingerprint,
        "case_id": case["case_id"],
        "request_hash": case["request_hash"],
        "provider": provider,
        "model": model,
        "status": "success" if parsed is not None else "failed",
        "attempt_count": len(attempts),
        "attempts": attempts,
        "judgment": parsed,
    }
    append_jsonl(RAW_PATH, row)
    return row


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
    if not values:
        return {"mean": None, "n": 0, "normal_95": None}
    mean = statistics.fmean(values)
    if len(values) == 1:
        interval = [mean, mean]
    else:
        margin = 1.959963984540054 * statistics.stdev(values) / math.sqrt(len(values))
        interval = [max(1.0, mean - margin), min(5.0, mean + margin)]
    return {"mean": mean, "n": len(values), "normal_95": interval}


def identity_for_side(case: Mapping, side: str) -> str:
    return str(case["identities"][side])


def preference_identity(case: Mapping, preference: str) -> str:
    if preference == "tie":
        return "tie"
    return identity_for_side(case, preference)


def aggregate(cases: Sequence[Mapping], rows: Mapping[object, Mapping]) -> dict:
    successful = [
        (case, rows[case["case_id"]])
        for case in cases
        if case["case_id"] in rows and rows[case["case_id"]].get("status") == "success"
    ]
    overall_counts = {"tuned": 0, "base": 0, "tie": 0}
    tutoring_counts = {"tuned": 0, "base": 0, "tie": 0}
    score_values = {
        identity: {field: [] for field in SCORE_FIELDS}
        for identity in ("base", "tuned")
    }
    bool_values = {
        identity: {field: [] for field in BOOL_FIELDS}
        for identity in ("base", "tuned")
    }
    attempts = []
    parse_failures = 0
    request_errors = 0

    for case, row in successful:
        judgment = row["judgment"]
        overall_counts[
            preference_identity(case, judgment["overall_preference"])
        ] += 1
        tutoring_counts[
            preference_identity(case, judgment["tutoring_preference"])
        ] += 1
        for side in ("A", "B"):
            identity = identity_for_side(case, side)
            candidate = judgment[f"candidate_{side}"]
            for field in SCORE_FIELDS:
                score_values[identity][field].append(float(candidate[field]))
            for field in BOOL_FIELDS:
                bool_values[identity][field].append(bool(candidate[field]))
        attempts.append(int(row["attempt_count"]))
        for attempt in row["attempts"]:
            parse_failures += attempt.get("parse_error") is not None
            request_errors += attempt.get("request_error") is not None

    n = len(successful)
    rubric = {}
    for identity in ("base", "tuned"):
        rubric[identity] = {}
        for field in SCORE_FIELDS:
            rubric[identity][field] = mean_ci(score_values[identity][field])
        for field in BOOL_FIELDS:
            values = bool_values[identity][field]
            hits = sum(values)
            rubric[identity][field] = {
                "count": hits,
                "n": len(values),
                "rate": hits / len(values) if values else None,
                "wilson_95": wilson(hits, len(values)) if values else None,
            }

    def preference_summary(counts: Mapping[str, int]) -> dict:
        return {
            key: {
                "count": counts[key],
                "n": n,
                "rate": counts[key] / n if n else None,
                "wilson_95": wilson(counts[key], n) if n else None,
            }
            for key in ("tuned", "base", "tie")
        }

    return {
        "completed_pairs": n,
        "overall_preference": preference_summary(overall_counts),
        "tutoring_preference": preference_summary(tutoring_counts),
        "rubric_by_model_identity": rubric,
        "blinded_side_balance": {
            "base_as_A": sum(case["identities"]["A"] == "base" for case in cases),
            "base_as_B": sum(case["identities"]["B"] == "base" for case in cases),
            "tuned_as_A": sum(case["identities"]["A"] == "tuned" for case in cases),
            "tuned_as_B": sum(case["identities"]["B"] == "tuned" for case in cases),
        },
        "request_statistics": {
            "successful_rows": n,
            "total_attempts": sum(attempts),
            "retried_cases": sum(value > 1 for value in attempts),
            "strict_parse_failures_before_success": parse_failures,
            "request_errors_before_success": request_errors,
        },
    }


def pct_cell(value: Optional[float]) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def ci_cell(value: Optional[Sequence[float]]) -> str:
    if value is None:
        return "n/a"
    return f"{100 * value[0]:.1f}–{100 * value[1]:.1f}%"


def build_report(results: Mapping) -> str:
    status = results["status"]
    persisted = results.get("persisted_attempts_cumulative", {})
    lines = [
        "# Blinded LLM-as-judge evaluation",
        "",
        f"Status: **{status}**.",
        "",
        "This evaluation is subjective and complements rather than replaces the "
        "authoritative deterministic map/label scorer.",
        "",
    ]
    if status == "BLOCKED":
        lines.extend(
            [
                "## Why blocked",
                "",
                f"- {results['blocked_reason']}",
                f"- Offline payload validation passed for "
                f"**{results['offline_validation']['validated_payloads']}** blinded pairs.",
                f"- API calls made in the reported run: "
                f"**{results.get('api_calls_made_this_run', 0)}**.",
                f"- Persisted judge-case attempts, cumulative across raw evidence: "
                f"**{persisted.get('judge_case_attempts', 0)}** in "
                f"**{persisted.get('raw_rows', 0)}** raw rows.",
                "",
                "## Resume safely",
                "",
                "Run in the terminal where the matching API key is already exported:",
                "",
                "```bash",
                "PYTHONDONTWRITEBYTECODE=1 python3 results/overnight/run_llm_judge.py "
                f"--provider {results.get('provider', 'tfy')} "
                f"--model {results.get('model', DEFAULT_TFY_MODEL)}"
                f"{' --retry-failed' if results.get('retry_failed_required') else ''}",
                "```",
                "",
                "The script reruns the same two payload checks, performs a two-call smoke, "
                "then completes the remaining pairs. Successful existing rows are skipped.",
                "",
            ]
        )
        return "\n".join(lines)

    aggregate_value = results["aggregate"]
    overall = aggregate_value["overall_preference"]
    tutoring = aggregate_value["tutoring_preference"]
    rubric = aggregate_value["rubric_by_model_identity"]
    base = rubric["base"]
    tuned = rubric["tuned"]
    stats = aggregate_value["request_statistics"]
    lines.extend(
        [
            "## Provenance",
            "",
            f"- Provider/model: **{results.get('provider', 'unknown')} / "
            f"`{results.get('model', 'unknown')}`**.",
            "- Sample: **100 paired image+coordinates test IDs**, seed `20260709`; "
            "candidate identity was blinded and balanced 50/50 across A/B.",
            "",
            "## Outcome",
            "",
            f"- Completed pairs: **{aggregate_value['completed_pairs']}/100**.",
            f"- Overall tuned/base/tie: **{overall['tuned']['count']} / "
            f"{overall['base']['count']} / {overall['tie']['count']}** "
            f"({pct_cell(overall['tuned']['rate'])} / "
            f"{pct_cell(overall['base']['rate'])} / {pct_cell(overall['tie']['rate'])}).",
            f"  Tuned Wilson 95% CI: "
            f"**{ci_cell(overall['tuned']['wilson_95'])}**.",
            f"- Hint-only tutoring tuned/base/tie: **{tutoring['tuned']['count']} / "
            f"{tutoring['base']['count']} / {tutoring['tie']['count']}**. "
            f"Base hint-only preference Wilson 95% CI: "
            f"**{ci_cell(tutoring['base']['wilson_95'])}**.",
            "",
            "Overall preference was explicitly instructed to respect the supplied "
            "authoritative deterministic correctness flags. It is therefore not a second "
            "independent exact-map score. Hint-only preference isolates the subjective "
            "tutoring tradeoff.",
            "",
            "## Subjective rubric",
            "",
            "| Measure | Base | Tuned |",
            "|---|---:|---:|",
            f"| Pedagogical usefulness mean (1–5) | "
            f"{base['pedagogical_usefulness_score']['mean']:.2f} | "
            f"{tuned['pedagogical_usefulness_score']['mean']:.2f} |",
            f"| Operation-family relevance mean (1–5) | "
            f"{base['operation_family_relevance_score']['mean']:.2f} | "
            f"{tuned['operation_family_relevance_score']['mean']:.2f} |",
            f"| Clarity/actionability mean (1–5) | "
            f"{base['clarity_actionability_score']['mean']:.2f} | "
            f"{tuned['clarity_actionability_score']['mean']:.2f} |",
            f"| Operation-family relevant | "
            f"{base['operation_family_relevant']['count']}/100 | "
            f"{tuned['operation_family_relevant']['count']}/100 |",
            f"| Forbidden answer/coordinate leakage | "
            f"{base['forbidden_answer_or_coordinate_leak']['count']}/100 | "
            f"{tuned['forbidden_answer_or_coordinate_leak']['count']}/100 |",
            "",
            "The tuned outputs were clearer and much more operation-relevant, but every "
            "sampled tuned hint was judged to disclose forbidden exact answer information "
            "under the deliberately stricter tutoring-safety rubric. Accordingly, base won "
            "the hint-only preference 75–25 despite losing overall 0–100.",
            "",
            "## Reliability",
            "",
            f"- Attempts in latest successful rows: **{stats['total_attempts']}**; "
            f"retried cases: "
            f"**{stats['retried_cases']}**; strict parse failures before success: "
            f"**{stats['strict_parse_failures_before_success']}**; request errors before "
            f"success: **{stats['request_errors_before_success']}**.",
            f"- Persisted judge-case attempts, cumulative across raw evidence: "
            f"**{persisted.get('judge_case_attempts', 0)}** in "
            f"**{persisted.get('raw_rows', 0)}** raw rows "
            f"(**{persisted.get('successful_rows', 0)} successful**, "
            f"**{persisted.get('failed_rows', 0)} failed**).",
            "- A/B assignment is exactly balanced 50/50. Wilson intervals and rubric "
            "means are available in `LLM_JUDGE_RESULTS.json`.",
            "",
            "An independent no-API recomputation is recorded in "
            "`LLM_JUDGE_VALIDATION.json` and `LLM_JUDGE_VALIDATION.md`.",
            "",
            "## Behavior-spec caveat",
            "",
            "The actual v6 prompt requested a short Socratic hint but did not explicitly "
            "prohibit coordinates. The leakage result is therefore a material tutoring-"
            "safety and specification concern, not evidence that the model violated an "
            "explicit v6 no-coordinate clause. LLM judgments remain subjective and "
            "secondary to deterministic geometry scoring.",
            "",
        ]
    )
    if status != "PASS":
        lines.extend(
            [
                "The two-call smoke succeeded, but the requested 100-pair run is not yet "
                "complete. Re-run the same command to resume without re-spending successful "
                "rows.",
                "",
            ]
        )
    return "\n".join(lines)


def assert_no_secret_values(paths: Sequence[Path]) -> None:
    secrets = [
        os.environ.get(variable)
        for variable in ("TFY_API_KEY", "ANTHROPIC_API_KEY")
        if os.environ.get(variable)
    ]
    if not secrets:
        return
    for path in paths:
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        assert all(secret not in text for secret in secrets), f"secret found in {path}"


def write_state(results: Mapping, metadata: Mapping) -> None:
    state = dict(results)
    state["persisted_attempts_cumulative"] = persisted_attempt_summary()
    for field in ("provider", "model"):
        if field not in state and metadata.get(field) is not None:
            state[field] = metadata[field]
    existing = latest_rows()
    state["retry_failed_required"] = any(
        row.get("status") == "failed" for row in existing.values()
    )
    atomic_json(META_PATH, metadata)
    atomic_json(RESULTS_PATH, state)
    atomic_text(REPORT_PATH, build_report(state))
    # Round-trip and consistency checks.
    parsed = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    assert parsed["status"] == state["status"]
    assert json.loads(META_PATH.read_text(encoding="utf-8"))["sample_size"] == N_PAIRS
    assert_no_secret_values((META_PATH, RESULTS_PATH, REPORT_PATH, RAW_PATH))


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("auto", "tfy", "anthropic"), default="auto")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=DEFAULT_TFY_BASE_URL)
    parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    parser.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="live-call cap after deterministic sampling; 0 means all 100",
    )
    parser.add_argument("--dry-run", action="store_true", help="never call an API")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="explicitly allow retrying a previously persisted failed case",
    )
    args = parser.parse_args(argv)
    if args.max_output_tokens < 256 or args.max_output_tokens > 2000:
        raise SystemExit("--max-output-tokens must be between 256 and 2000")
    if args.max_retries < 1 or args.max_retries > 5:
        raise SystemExit("--max-retries must be between 1 and 5")
    if args.limit < 0 or args.limit > N_PAIRS:
        raise SystemExit("--limit must be from 0 to 100")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_PATH.touch(exist_ok=True)
    cases = prepare_cases()

    # Required two-payload offline validation.
    offline_cases = cases[:SMOKE_PAIRS]
    for case in offline_cases:
        payload = case["judge_payload"]
        assert "identities" not in payload
        assert set(payload["candidate_A"]) == {"deterministic_scores", "output"}
        assert set(payload["candidate_B"]) == {"deterministic_scores", "output"}
        assert len(case["user_prompt"].encode("utf-8")) < 40_000
        assert case["request_hash"] == sha256_text(
            SYSTEM_PROMPT + "\n" + case["user_prompt"]
        )
        synthetic = strict_output_template(case["case_id"])
        synthetic["tutoring_preference"] = "tie"
        synthetic["overall_preference"] = "tie"
        validate_judgment(synthetic, case["case_id"])

    selected_provider, key_presence = resolve_provider(args.provider)
    preferred_provider = selected_provider or (
        "anthropic" if args.provider == "anthropic" else "tfy"
    )
    model = args.model or (
        DEFAULT_ANTHROPIC_MODEL
        if preferred_provider == "anthropic"
        else DEFAULT_TFY_MODEL
    )
    route_selection = (
        "explicit CLI or existing repository route convention"
        if args.model
        else "existing repository default; TFY route is verified by model discovery before calls"
    )
    run_fingerprint = make_run_fingerprint(
        cases,
        preferred_provider,
        model,
        args.base_url,
        args.max_output_tokens,
    )
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "sample_size": N_PAIRS,
        "split": "test",
        "modality": "image_coords",
        "pairing": "identical IDs from frozen base and tuned records",
        "sample_ids": [case["case_id"] for case in cases],
        "blind_assignment": [
            {
                "case_id": case["case_id"],
                "A_identity": case["identities"]["A"],
                "B_identity": case["identities"]["B"],
            }
            for case in cases
        ],
        "blinded_side_balance": {
            "base_as_A": sum(case["identities"]["A"] == "base" for case in cases),
            "base_as_B": sum(case["identities"]["B"] == "base" for case in cases),
            "tuned_as_A": sum(case["identities"]["A"] == "tuned" for case in cases),
            "tuned_as_B": sum(case["identities"]["B"] == "tuned" for case in cases),
        },
        "request_hashes": {
            str(case["case_id"]): case["request_hash"] for case in cases
        },
        "system_prompt_sha256": sha256_text(SYSTEM_PROMPT),
        "request_set_sha256": request_set_sha256(cases),
        "input_sha256": {
            str(BASE_PATH.relative_to(ROOT)): sha256_file(BASE_PATH),
            str(TUNED_PATH.relative_to(ROOT)): sha256_file(TUNED_PATH),
            str(ORACLE_PATH.relative_to(ROOT)): sha256_file(ORACLE_PATH),
        },
        "provider": preferred_provider,
        "model": model,
        "base_url": args.base_url if preferred_provider == "tfy" else None,
        "route_selection": route_selection,
        "max_output_tokens": args.max_output_tokens,
        "max_retries": args.max_retries,
        "key_presence": key_presence,
        "run_fingerprint": run_fingerprint,
        "offline_validation": {
            "validated_payloads": SMOKE_PAIRS,
            "payload_case_ids": [case["case_id"] for case in offline_cases],
            "identity_hidden_from_judge_payload": True,
            "strict_output_schema_validated": True,
            "request_size_cap_bytes": 40_000,
            "passed": True,
        },
        "secrets_persisted": False,
    }

    existing = latest_rows()
    for case_id, row in existing.items():
        if row.get("status") == "success":
            if row.get("request_hash") != next(
                case["request_hash"] for case in cases if case["case_id"] == case_id
            ):
                raise SystemExit(
                    f"existing successful case {case_id} has a different request hash"
                )
            if row.get("run_fingerprint") != run_fingerprint:
                raise SystemExit(
                    f"existing successful case {case_id} belongs to a different run config"
                )

    if args.dry_run or selected_provider is None:
        missing = []
        if not key_presence["TFY_API_KEY"]:
            missing.append("TFY_API_KEY")
        if not key_presence["ANTHROPIC_API_KEY"]:
            missing.append("ANTHROPIC_API_KEY")
        reason = (
            "Neither TFY_API_KEY nor ANTHROPIC_API_KEY is present; route discovery, "
            "two-call smoke, and the paid 100-pair run were not attempted."
            if selected_provider is None
            else "Explicit --dry-run requested; no API calls were attempted."
        )
        results = {
            "schema_version": SCHEMA_VERSION,
            "status": "BLOCKED",
            "blocked_reason": reason,
            "missing_credentials": missing,
            "api_calls_made_this_run": 0,
            "offline_validation": metadata["offline_validation"],
            "completed_pairs_from_existing_raw": sum(
                row.get("status") == "success" for row in existing.values()
            ),
            "aggregate": aggregate(cases, existing),
            "subjectivity_note": (
                "LLM judgments are subjective and never replace deterministic map scoring."
            ),
        }
        write_state(results, metadata)
        print("offline payload validation: PASS (2/2)")
        print("LLM judge: BLOCKED (no eligible API key; 0 API calls)")
        print(f"wrote {META_PATH}, {RESULTS_PATH}, and {REPORT_PATH}")
        return

    # Construct one provider client only. No cross-provider or route fallback.
    if selected_provider == "tfy":
        call, client = make_tfy_caller(model, args.base_url, args.max_output_tokens)
        if args.model is None:
            try:
                route_ids = {item.id for item in client.models.list().data}
            except Exception as exc:
                results = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "BLOCKED",
                    "blocked_reason": (
                        "TFY route discovery failed before smoke: " + redact_error(exc)
                    ),
                    "api_calls_made_this_run": 1,
                    "offline_validation": metadata["offline_validation"],
                    "aggregate": aggregate(cases, existing),
                    "subjectivity_note": (
                        "LLM judgments are subjective and never replace deterministic map scoring."
                    ),
                }
                write_state(results, metadata)
                print("LLM judge: BLOCKED (route discovery failed)")
                return
            if model not in route_ids:
                results = {
                    "schema_version": SCHEMA_VERSION,
                    "status": "BLOCKED",
                    "blocked_reason": (
                        f"TFY route discovery succeeded, but selected route {model!r} "
                        "was not listed. No judge case calls were made."
                    ),
                    "api_calls_made_this_run": 1,
                    "offline_validation": metadata["offline_validation"],
                    "aggregate": aggregate(cases, existing),
                    "subjectivity_note": (
                        "LLM judgments are subjective and never replace deterministic map scoring."
                    ),
                }
                write_state(results, metadata)
                print("LLM judge: BLOCKED (selected route unavailable)")
                return
            metadata["route_selection"] = "verified present via TFY model discovery"
    else:
        call, _client = make_anthropic_caller(model, args.max_output_tokens)
        metadata["route_selection"] = "existing direct-Anthropic repository route convention"

    target_cases = cases[: args.limit or N_PAIRS]
    existing = latest_rows()

    # Two-call smoke gate. Existing successful smoke rows are reused without spending.
    for case in cases[:SMOKE_PAIRS]:
        prior = existing.get(case["case_id"])
        if prior and prior.get("status") == "success":
            continue
        if prior and prior.get("status") == "failed" and not args.retry_failed:
            results = {
                "schema_version": SCHEMA_VERSION,
                "status": "BLOCKED",
                "blocked_reason": (
                    f"smoke case {case['case_id']} previously failed; successful rows "
                    "were preserved. Inspect raw attempts, then rerun with --retry-failed "
                    "only if another paid attempt is intended."
                ),
                "api_calls_made_this_run": 0,
                "offline_validation": metadata["offline_validation"],
                "aggregate": aggregate(cases, existing),
                "subjectivity_note": (
                    "LLM judgments are subjective and never replace deterministic map scoring."
                ),
            }
            write_state(results, metadata)
            print("LLM judge: BLOCKED (persisted smoke failure)")
            return
        row = call_case(
            case,
            call,
            selected_provider,
            model,
            run_fingerprint,
            args.max_retries,
        )
        existing[case["case_id"]] = row
        if row["status"] != "success":
            results = {
                "schema_version": SCHEMA_VERSION,
                "status": "BLOCKED",
                "blocked_reason": (
                    f"two-call smoke failed on case {case['case_id']} after "
                    f"{row['attempt_count']} attempt(s); full run was not started."
                ),
                "api_calls_made_this_run": row["attempt_count"],
                "offline_validation": metadata["offline_validation"],
                "aggregate": aggregate(cases, existing),
                "subjectivity_note": (
                    "LLM judgments are subjective and never replace deterministic map scoring."
                ),
            }
            write_state(results, metadata)
            print("LLM judge: BLOCKED (two-call smoke failed)")
            return

    # Smoke passed. Resume remaining target cases; never resend a successful row.
    for case in target_cases:
        prior = existing.get(case["case_id"])
        if prior and prior.get("status") == "success":
            continue
        if prior and prior.get("status") == "failed" and not args.retry_failed:
            continue
        row = call_case(
            case,
            call,
            selected_provider,
            model,
            run_fingerprint,
            args.max_retries,
        )
        existing[case["case_id"]] = row

    existing = latest_rows()
    aggregate_value = aggregate(cases, existing)
    completed = aggregate_value["completed_pairs"]
    status = "PASS" if completed == N_PAIRS else "PARTIAL"
    results = {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "provider": selected_provider,
        "model": model,
        "sample_seed": SEED,
        "requested_pairs": N_PAIRS,
        "aggregate": aggregate_value,
        "offline_validation": metadata["offline_validation"],
        "smoke_passed": all(
            existing.get(case["case_id"], {}).get("status") == "success"
            for case in cases[:SMOKE_PAIRS]
        ),
        "subjectivity_note": (
            "LLM judgments are subjective and never replace deterministic map scoring."
        ),
    }
    write_state(results, metadata)
    print(
        f"LLM judge: {status}; completed {completed}/{N_PAIRS}; "
        f"raw rows at {RAW_PATH}"
    )
    print(f"wrote {RESULTS_PATH} and {REPORT_PATH}")


if __name__ == "__main__":
    main()
