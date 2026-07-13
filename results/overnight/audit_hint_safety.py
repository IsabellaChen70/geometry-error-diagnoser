#!/usr/bin/env python3
"""Independent deterministic safety audit for frozen v6 tuned-model hints.

The audit intentionally does not import ``transform_diagnosis.eval`` or
``transform_diagnosis.hints``.  It independently:

* extracts the final JSON object from every saved response;
* reconstructs the existing operation-family and residual-coordinate rubric;
* derives oracle affine maps from the read-only source geometry;
* separates coordinate-pair disclosure from exact map/value disclosure; and
* checks agreement with the stored ``hint_ok`` and ``hint_exact_ok`` fields.

Outputs are deterministic and overwritten atomically, so rerunning is safe.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "results" / "overnight"
FINAL_DIR = ROOT / "results" / "v6_final"
ORACLE_DIR = ROOT / "transform_diagnosis_data"

LABELS = (
    "correct",
    "reflection_instead_of_rotation",
    "rotation_instead_of_reflection",
    "wrong_rotation_angle",
    "wrong_reflection_line",
    "wrong_translation",
    "opposite_translation",
    "completely_wrong",
)

CELL_FILES = {
    "image_test": FINAL_DIR / "records_v6_4b_image_test.jsonl",
    "image_ood": FINAL_DIR / "records_v6_4b_image_ood.jsonl",
    "image_coords_test": FINAL_DIR / "records_v6_4b_image_coords_test.jsonl",
    "image_coords_ood": FINAL_DIR / "records_v6_4b_image_coords_ood.jsonl",
}

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
MATRIX_TO_NAME = {matrix: name for name, matrix in MATRICES.items()}

# This vocabulary is copied deliberately from transform_diagnosis/eval.py so the
# independent result can be compared with the stored evaluator fields.
FAMILY_KEYWORDS = {
    "rotation": ("rotat", "turn", "clockwise", "degree", "angle"),
    "reflection": ("reflect", "mirror", "flip"),
    "translation": ("translat", "slide", "slid", "shift", "move"),
}

COORD_RE = re.compile(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
ROT_RE = re.compile(
    r"rotate\s+(\d+)\s*degrees?(?:\s+(counterclockwise|clockwise|ccw|cw))?",
    re.IGNORECASE,
)
TRANS_XY_RE = re.compile(
    r"(?:translate|move)\s+by\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)",
    re.IGNORECASE,
)
TRANS_DIR_RE = re.compile(
    r"(?:translate|move)\s+(\d+)\s*(?:units?|squares?|spaces?)?\s*"
    r"(left|right|up|down)",
    re.IGNORECASE,
)

LINEAR_PATTERNS = {
    "identity": (
        re.compile(r"\bidentity(?:\s+linear\s+map)?\b", re.IGNORECASE),
    ),
    "rot_ccw_90": (
        re.compile(r"\brot_ccw_90\b", re.IGNORECASE),
        re.compile(r"\brotate\s+90\s+degrees?\s+(?:counterclockwise|ccw)\b", re.IGNORECASE),
        re.compile(r"\brotate\s+270\s+degrees?\s+(?:clockwise|cw)\b", re.IGNORECASE),
        re.compile(r"\b90[- ]degree\s+(?:counterclockwise|ccw)\s+rotation\b", re.IGNORECASE),
    ),
    "rot_180": (
        re.compile(r"\brot_180\b", re.IGNORECASE),
        re.compile(
            r"\brotate\s+180\s+degrees?(?:\s+(?:counterclockwise|clockwise|ccw|cw))?\b",
            re.IGNORECASE,
        ),
        re.compile(r"\b180[- ]degree\s+rotation\b", re.IGNORECASE),
    ),
    "rot_ccw_270": (
        re.compile(r"\brot_ccw_270\b", re.IGNORECASE),
        re.compile(r"\brotate\s+270\s+degrees?\s+(?:counterclockwise|ccw)\b", re.IGNORECASE),
        re.compile(r"\brotate\s+90\s+degrees?\s+(?:clockwise|cw)\b", re.IGNORECASE),
        re.compile(r"\b270[- ]degree\s+(?:counterclockwise|ccw)\s+rotation\b", re.IGNORECASE),
    ),
    "reflect_x_axis": (
        re.compile(r"\breflect_x_axis\b", re.IGNORECASE),
        re.compile(r"\breflect(?:ion)?\s+across\s+(?:the\s+)?x[- ]?axis\b", re.IGNORECASE),
    ),
    "reflect_y_axis": (
        re.compile(r"\breflect_y_axis\b", re.IGNORECASE),
        re.compile(r"\breflect(?:ion)?\s+across\s+(?:the\s+)?y[- ]?axis\b", re.IGNORECASE),
    ),
    "reflect_y_eq_x": (
        re.compile(r"\breflect_y_eq_x\b", re.IGNORECASE),
        re.compile(r"\breflect(?:ion)?\s+across\s+(?:the\s+)?(?:line\s+)?y\s*=\s*x\b", re.IGNORECASE),
    ),
    "reflect_y_eq_neg_x": (
        re.compile(r"\breflect_y_eq_neg_x\b", re.IGNORECASE),
        re.compile(
            r"\breflect(?:ion)?\s+across\s+(?:the\s+)?(?:line\s+)?y\s*=\s*-\s*x\b",
            re.IGNORECASE,
        ),
    ),
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def extract_final_object(text: object) -> Optional[dict]:
    """Return the last decodable JSON object without using the project parser."""
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
    found: Optional[dict] = None
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


def norm(text: object) -> str:
    return " ".join(str(text).lower().split())


def mat_mul(left: tuple, right: tuple) -> tuple:
    (a, b), (c, d) = left
    (e, f), (g, h) = right
    return ((a * e + b * g, a * f + b * h), (c * e + d * g, c * f + d * h))


def mat_vec(matrix: tuple, vector: Sequence[int]) -> tuple[int, int]:
    (a, b), (c, d) = matrix
    x, y = vector
    return (a * x + b * y, c * x + d * y)


def parse_step(text: object) -> tuple[tuple, tuple[int, int]]:
    if not isinstance(text, str):
        raise ValueError(f"transform step is not text: {text!r}")
    value = norm(text)

    match = ROT_RE.search(value)
    if match:
        degrees = int(match.group(1)) % 360
        direction = (match.group(2) or "").lower()
        if not direction and degrees not in (0, 180):
            raise ValueError(f"ambiguous rotation: {text!r}")
        if direction in ("clockwise", "cw"):
            degrees = (-degrees) % 360
        name = {
            0: "identity",
            90: "rot_ccw_90",
            180: "rot_180",
            270: "rot_ccw_270",
        }.get(degrees)
        if name is None:
            raise ValueError(f"unsupported rotation: {text!r}")
        return MATRICES[name], (0, 0)

    compact = value.replace(" ", "").replace("-", "")
    if value.startswith("reflect across"):
        axis = value.removeprefix("reflect across").strip()
        axis_compact = axis.replace(" ", "").replace("-", "")
        aliases = {
            "xaxis": "reflect_x_axis",
            "thexaxis": "reflect_x_axis",
            "yaxis": "reflect_y_axis",
            "theyaxis": "reflect_y_axis",
            "liney=x": "reflect_y_eq_x",
            "y=x": "reflect_y_eq_x",
            "theliney=x": "reflect_y_eq_x",
            "liney=x": "reflect_y_eq_x",
        }
        # Preserve the minus sign only for the two diagonal aliases.
        signed = axis.replace(" ", "").replace("the", "")
        if signed in ("liney=-x", "y=-x"):
            return MATRICES["reflect_y_eq_neg_x"], (0, 0)
        name = aliases.get(axis_compact)
        if name is not None:
            return MATRICES[name], (0, 0)
    if compact.startswith("reflectacrossliney=x"):
        return MATRICES["reflect_y_eq_x"], (0, 0)

    match = TRANS_XY_RE.search(value)
    if match:
        return IDENTITY, (int(match.group(1)), int(match.group(2)))
    match = TRANS_DIR_RE.search(value)
    if match:
        magnitude = int(match.group(1))
        direction = match.group(2).lower()
        unit = {
            "left": (-1, 0),
            "right": (1, 0),
            "up": (0, 1),
            "down": (0, -1),
        }[direction]
        return IDENTITY, (magnitude * unit[0], magnitude * unit[1])
    if value == "identity":
        return IDENTITY, (0, 0)
    raise ValueError(f"unsupported transform step: {text!r}")


def compose(steps: Sequence[object]) -> tuple[tuple, tuple[int, int]]:
    matrix = IDENTITY
    vector = (0, 0)
    for step in steps:
        step_matrix, step_vector = parse_step(step)
        projected = mat_vec(step_matrix, vector)
        vector = (projected[0] + step_vector[0], projected[1] + step_vector[1])
        matrix = mat_mul(step_matrix, matrix)
    return matrix, vector


def describe(matrix: tuple, vector: Sequence[int]) -> str:
    x, y = int(vector[0]), int(vector[1])
    if matrix == IDENTITY:
        if (x, y) == (0, 0):
            return "identity"
        if y == 0:
            return f"translate {abs(x)} {'right' if x > 0 else 'left'}"
        if x == 0:
            return f"translate {abs(y)} {'up' if y > 0 else 'down'}"
        return f"translate by ({x}, {y})"
    if (x, y) != (0, 0):
        raise ValueError("non-primitive affine map cannot use primitive description")
    name = MATRIX_TO_NAME[matrix]
    return {
        "rot_ccw_90": "rotate 90 degrees counterclockwise",
        "rot_180": "rotate 180 degrees counterclockwise",
        "rot_ccw_270": "rotate 270 degrees counterclockwise",
        "reflect_x_axis": "reflect across x axis",
        "reflect_y_axis": "reflect across y axis",
        "reflect_y_eq_x": "reflect across line y = x",
        "reflect_y_eq_neg_x": "reflect across line y = -x",
    }[name]


def family(matrix: tuple) -> str:
    if matrix == IDENTITY:
        return "translation"
    (a, b), (c, d) = matrix
    return "rotation" if a * d - b * c == 1 else "reflection"


def expected_tokens(label: str, oracle: Mapping) -> list[str]:
    correct = [parse_step(item) for item in oracle["correct_transform"]]
    student = [parse_step(item) for item in oracle["student_transform"]]
    if label == "correct":
        return [describe(matrix, vector) for matrix, vector in correct]
    if label == "completely_wrong":
        correct_net = compose(oracle["correct_transform"])
        student_net = compose(oracle["student_transform"])
        return [
            describe(correct_net[0], (0, 0)),
            describe(student_net[0], (0, 0)),
            f"({correct_net[1][0]}, {correct_net[1][1]})",
            f"({student_net[1][0]}, {student_net[1][1]})",
        ]
    differences = [
        index
        for index, (left, right) in enumerate(zip(correct, student))
        if left != right
    ]
    if len(differences) != 1:
        raise ValueError(
            f"id={oracle.get('id')} label={label}: expected one changed step, got {differences}"
        )
    index = differences[0]
    return [
        describe(correct[index][0], correct[index][1]),
        describe(student[index][0], student[index][1]),
    ]


def expected_families(label: str, oracle: Mapping) -> list[str]:
    correct = [parse_step(item) for item in oracle["correct_transform"]]
    student = [parse_step(item) for item in oracle["student_transform"]]
    if label == "correct":
        return sorted({family(matrix) for matrix, _ in correct})
    if label == "completely_wrong":
        correct_net = compose(oracle["correct_transform"])
        student_net = compose(oracle["student_transform"])
        return sorted({family(correct_net[0]), family(student_net[0]), "translation"})
    differences = [
        index
        for index, (left, right) in enumerate(zip(correct, student))
        if left != right
    ]
    if len(differences) != 1:
        raise ValueError(
            f"id={oracle.get('id')} label={label}: expected one changed step, got {differences}"
        )
    index = differences[0]
    return sorted({family(correct[index][0]), family(student[index][0])})


def mentions_family(hint: str, families: Sequence[str]) -> bool:
    value = norm(hint)
    return any(
        keyword in value
        for operation in families
        for keyword in FAMILY_KEYWORDS[operation]
    )


def evaluator_residual_coordinate_leak(hint: str, tokens: Sequence[str]) -> bool:
    residual = re.sub(r"\s+", "", hint.lower())
    for token in tokens:
        residual = residual.replace(re.sub(r"\s+", "", token.lower()), "")
    return bool(re.search(r"\(-?\d+,-?\d+\)", residual))


def coordinate_pairs(text: str) -> list[list[int]]:
    return [[int(match.group(1)), int(match.group(2))] for match in COORD_RE.finditer(text)]


def mentions_linear(text: str, linear: str) -> bool:
    return any(pattern.search(text) for pattern in LINEAR_PATTERNS[linear])


def translation_patterns(vector: Sequence[int]) -> tuple[re.Pattern, ...]:
    x, y = int(vector[0]), int(vector[1])
    escaped_pair = rf"\(\s*{x}\s*,\s*{y}\s*\)"
    patterns = [re.compile(escaped_pair, re.IGNORECASE)]
    if y == 0 and x != 0:
        direction = "right" if x > 0 else "left"
        patterns.append(
            re.compile(
                rf"\b(?:translate|translation|shift|move|slide)\D{{0,20}}"
                rf"{abs(x)}(?:\s+(?:units?|squares?|spaces?))?\s+(?:to\s+the\s+)?{direction}\b",
                re.IGNORECASE,
            )
        )
    elif x == 0 and y != 0:
        direction = "up" if y > 0 else "down"
        patterns.append(
            re.compile(
                rf"\b(?:translate|translation|shift|move|slide)\D{{0,20}}"
                rf"{abs(y)}(?:\s+(?:units?|squares?|spaces?))?\s+(?:to\s+the\s+)?{direction}\b",
                re.IGNORECASE,
            )
        )
    elif (x, y) == (0, 0):
        patterns.extend(
            (
                re.compile(r"\b(?:zero|no)\s+(?:net\s+)?translation\b", re.IGNORECASE),
                re.compile(r"\btranslate\s+0\s+(?:left|right|up|down)\b", re.IGNORECASE),
            )
        )
    return tuple(patterns)


def mentions_translation(text: str, vector: Sequence[int]) -> bool:
    return any(pattern.search(text) for pattern in translation_patterns(vector))


def recover_net(source: Sequence[Sequence[int]], target: Sequence[Sequence[int]]) -> dict:
    """Recover the unique D4 affine map directly from corresponding vertices."""
    matches = []
    for name, matrix in MATRICES.items():
        first_source = source[0]
        first_target = target[0]
        projected = mat_vec(matrix, first_source)
        translation = (
            int(first_target[0]) - projected[0],
            int(first_target[1]) - projected[1],
        )
        produced = []
        for point in source:
            mapped = mat_vec(matrix, point)
            produced.append(
                [mapped[0] + translation[0], mapped[1] + translation[1]]
            )
        if produced == [[int(x), int(y)] for x, y in target]:
            matches.append(
                {"linear": name, "tx": translation[0], "ty": translation[1]}
            )
    if len(matches) != 1:
        raise ValueError(f"geometry has {len(matches)} recoverable D4 maps")
    return matches[0]


def exact_primitive_translation_disclosure(hint: str, oracle: Mapping) -> bool:
    vectors = []
    for key in ("correct_transform", "student_transform"):
        for step in oracle[key]:
            matrix, vector = parse_step(step)
            if matrix == IDENTITY and vector != (0, 0):
                vectors.append(vector)
    return any(mentions_translation(hint, vector) for vector in vectors)


def audit_row(cell: str, saved: Mapping, oracle: Mapping) -> dict:
    parsed = extract_final_object(saved.get("raw_model_output"))
    hint_present = isinstance(parsed, dict) and "hint" in parsed
    raw_hint = parsed.get("hint") if parsed else None
    hint_valid = isinstance(raw_hint, str) and bool(raw_hint.strip())
    hint = raw_hint if isinstance(raw_hint, str) else ""

    label = str(oracle["label"])
    tokens = expected_tokens(label, oracle)
    families = expected_families(label, oracle)
    family_relevant = hint_valid and mentions_family(hint, families)
    unsanctioned_coordinate = (
        hint_valid and evaluator_residual_coordinate_leak(hint, tokens)
    )
    exact_all_tokens = (
        hint_valid
        and all(norm(token) in norm(hint) for token in tokens)
        and not unsanctioned_coordinate
    )
    exact_any_token = hint_valid and any(norm(token) in norm(hint) for token in tokens)

    correct_net = recover_net(oracle["original"], oracle["correct_image"])
    student_net = recover_net(oracle["original"], oracle["student_image"])
    correct_linear = hint_valid and mentions_linear(hint, str(correct_net["linear"]))
    student_linear = hint_valid and mentions_linear(hint, str(student_net["linear"]))
    correct_translation = hint_valid and mentions_translation(
        hint, (int(correct_net["tx"]), int(correct_net["ty"]))
    )
    student_translation = hint_valid and mentions_translation(
        hint, (int(student_net["tx"]), int(student_net["ty"]))
    )
    primitive_translation = hint_valid and exact_primitive_translation_disclosure(
        hint, oracle
    )

    pairs = coordinate_pairs(hint)
    token_pairs = {
        tuple(pair)
        for token in tokens
        for pair in coordinate_pairs(token)
    }
    sanctioned_pairs = [pair for pair in pairs if tuple(pair) in token_pairs]
    unsanctioned_pairs = [pair for pair in pairs if tuple(pair) not in token_pairs]

    exact_linear_disclosure = bool(correct_linear or student_linear)
    exact_translation_disclosure = bool(
        correct_translation or student_translation or primitive_translation
    )
    exact_full_map_disclosure = bool(
        (correct_linear and correct_translation)
        or (student_linear and student_translation)
    )
    exact_answer_disclosure = bool(
        exact_any_token or exact_linear_disclosure or exact_translation_disclosure
    )
    independent_hint_ok = bool(
        hint_valid and family_relevant and not unsanctioned_coordinate
    )
    independent_hint_exact_ok = bool(exact_all_tokens)
    safe_useful = bool(
        hint_valid
        and family_relevant
        and not unsanctioned_coordinate
        and not exact_answer_disclosure
    )

    return {
        "cell": cell,
        "modality": "image_coords" if cell.startswith("image_coords") else "image",
        "split": str(saved.get("split")),
        "id": saved.get("id"),
        "label": label,
        "response_parse_ok": parsed is not None,
        "hint_present": hint_present,
        "hint_valid": hint_valid,
        "required_families": families,
        "family_relevant": family_relevant,
        "coordinate_pair_present": bool(pairs),
        "coordinate_pair_count": len(pairs),
        "coordinate_pairs": pairs,
        "rubric_sanctioned_coordinate_pairs": sanctioned_pairs,
        "rubric_unsanctioned_coordinate_pairs": unsanctioned_pairs,
        "evaluator_unsanctioned_coordinate_leak": bool(unsanctioned_coordinate),
        "exact_expected_token_disclosure": bool(exact_any_token),
        "exact_all_expected_tokens": bool(exact_all_tokens),
        "exact_correct_linear_disclosure": bool(correct_linear),
        "exact_student_linear_disclosure": bool(student_linear),
        "exact_correct_net_translation_disclosure": bool(correct_translation),
        "exact_student_net_translation_disclosure": bool(student_translation),
        "exact_primitive_translation_disclosure": bool(primitive_translation),
        "exact_canonical_linear_disclosure": exact_linear_disclosure,
        "exact_translation_value_disclosure": exact_translation_disclosure,
        "exact_full_canonical_map_disclosure": exact_full_map_disclosure,
        "exact_answer_value_disclosure": exact_answer_disclosure,
        "safe_useful": safe_useful,
        "independent_hint_ok": independent_hint_ok,
        "independent_hint_exact_ok": independent_hint_exact_ok,
        "stored_hint_ok": saved.get("hint_ok"),
        "stored_hint_exact_ok": saved.get("hint_exact_ok"),
        "hint_ok_agrees": saved.get("hint_ok") == independent_hint_ok,
        "hint_exact_ok_agrees": saved.get("hint_exact_ok")
        == independent_hint_exact_ok,
        "hint_excerpt": re.sub(r"\s+", " ", hint).strip()[:180],
    }


METRIC_FIELDS = (
    "response_parse_ok",
    "hint_present",
    "hint_valid",
    "family_relevant",
    "coordinate_pair_present",
    "evaluator_unsanctioned_coordinate_leak",
    "exact_expected_token_disclosure",
    "exact_canonical_linear_disclosure",
    "exact_translation_value_disclosure",
    "exact_full_canonical_map_disclosure",
    "exact_answer_value_disclosure",
    "safe_useful",
    "independent_hint_ok",
    "independent_hint_exact_ok",
    "stored_hint_ok",
    "stored_hint_exact_ok",
    "hint_ok_agrees",
    "hint_exact_ok_agrees",
)


def metric_summary(rows: Sequence[Mapping]) -> dict:
    result = {"n": len(rows), "metrics": {}}
    for field in METRIC_FIELDS:
        hits = sum(row.get(field) is True for row in rows)
        result["metrics"][field] = {
            "count": hits,
            "n": len(rows),
            "rate": hits / len(rows) if rows else 0.0,
        }
    result["coordinate_pairs"] = {
        "occurrences": sum(int(row["coordinate_pair_count"]) for row in rows),
        "rows_with_sanctioned_pairs": sum(
            bool(row["rubric_sanctioned_coordinate_pairs"]) for row in rows
        ),
        "rows_with_unsanctioned_pairs": sum(
            bool(row["rubric_unsanctioned_coordinate_pairs"]) for row in rows
        ),
    }
    return result


def reasons(row: Mapping) -> list[str]:
    result = []
    if not row["response_parse_ok"]:
        result.append("final JSON did not parse")
    elif not row["hint_valid"]:
        result.append("hint missing, empty, or non-string")
    if row["hint_valid"] and not row["family_relevant"]:
        result.append("does not mention a required operation family")
    if row["evaluator_unsanctioned_coordinate_leak"]:
        result.append("contains coordinate pair not sanctioned by existing exact tokens")
    if row["exact_canonical_linear_disclosure"]:
        result.append("states an exact oracle linear map")
    if row["exact_translation_value_disclosure"]:
        result.append("states an exact oracle/primitive translation value")
    if row["exact_expected_token_disclosure"]:
        result.append("repeats at least one exact oracle hint token")
    return result


def representative_failures(rows: Sequence[Mapping], limit_per_reason: int = 3) -> list[dict]:
    selected = []
    seen = set()
    reason_counts: Counter[str] = Counter()
    for row in sorted(rows, key=lambda item: (str(item["cell"]), int(item["id"]))):
        row_reasons = reasons(row)
        if not row_reasons:
            continue
        eligible = [
            reason for reason in row_reasons if reason_counts[reason] < limit_per_reason
        ]
        if not eligible:
            continue
        key = (row["cell"], row["id"])
        if key in seen:
            continue
        seen.add(key)
        for reason in eligible:
            reason_counts[reason] += 1
        selected.append(
            {
                "cell": row["cell"],
                "id": row["id"],
                "label": row["label"],
                "reasons": row_reasons,
                "hint_excerpt": row["hint_excerpt"],
            }
        )
        if len(selected) >= 18:
            break
    return selected


def pct(metric: Mapping) -> str:
    return f"{metric['count']}/{metric['n']} ({100 * metric['rate']:.1f}%)"


def build_markdown(report: Mapping) -> str:
    overall = report["overall"]["metrics"]
    lines = [
        "# Deterministic v6 hint-safety audit",
        "",
        "Status: **PASS (audit completed)**. This status means the audit ran over all "
        "2,000 frozen tuned `full` outputs and passed its consistency assertions; it "
        "does not mean every hint was safe.",
        "",
        "## Contract and rubric boundary",
        "",
        "- The actual v6 `full` prompt asks for a **short Socratic hint**, but unlike the "
        "legacy prompt it does **not** say “without stating coordinates.” This is a "
        "specification gap, so coordinate disclosures below are not mislabeled as direct "
        "v6 prompt violations.",
        "- The existing `hint_ok` rubric requires at least one oracle-relevant operation "
        "family and rejects only coordinate pairs left after removing exact expected "
        "tokens. Exact expected tokens are therefore *sanctioned* by that implementation. "
        "For `completely_wrong`, this explicitly sanctions both oracle net-translation "
        "pairs; other labels may sanction diagonal primitive-translation pairs.",
        "- This audit additionally flags any exact oracle operation/map parameter or "
        "translation value repeated in the hint. `safe_useful` is the conservative tutoring "
        "measure: valid hint + relevant family + no rubric-unsanctioned coordinate pair + "
        "no exact answer/map/value disclosure. It is intentionally stricter than stored "
        "`hint_ok` and is not a replacement for exact geometry scoring.",
        "",
        "## Overall findings",
        "",
        f"- Final JSON parse: **{pct(overall['response_parse_ok'])}**; valid hint field: "
        f"**{pct(overall['hint_valid'])}**.",
        f"- Operation-family relevance: **{pct(overall['family_relevant'])}**.",
        f"- Any explicit coordinate pair: **{pct(overall['coordinate_pair_present'])}**; "
        f"rubric-unsanctioned coordinate leak: "
        f"**{pct(overall['evaluator_unsanctioned_coordinate_leak'])}**.",
        f"- Exact canonical-linear disclosure: "
        f"**{pct(overall['exact_canonical_linear_disclosure'])}**; exact translation-value "
        f"disclosure: **{pct(overall['exact_translation_value_disclosure'])}**; exact full "
        f"canonical-map disclosure: **{pct(overall['exact_full_canonical_map_disclosure'])}**; "
        f"any exact answer/map/value disclosure: "
        f"**{pct(overall['exact_answer_value_disclosure'])}**.",
        f"- Conservative combined safe/useful: **{pct(overall['safe_useful'])}**.",
        f"- Stored `hint_ok` agreement: **{pct(overall['hint_ok_agrees'])}**; stored "
        f"`hint_exact_ok` agreement: **{pct(overall['hint_exact_ok_agrees'])}**.",
        "",
        "Exact geometry metrics remain authoritative and separate; this report never "
        "uses hint quality to override map correctness.",
        "",
        "## By modality and split",
        "",
        "| Cell | Family relevant | Any coordinate pair | Unsanctioned pair leak | "
        "Exact answer/value | Safe/useful | Stored hint_ok |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in CELL_FILES:
        metrics = report["by_cell"][cell]["metrics"]
        lines.append(
            f"| `{cell}` | {pct(metrics['family_relevant'])} | "
            f"{pct(metrics['coordinate_pair_present'])} | "
            f"{pct(metrics['evaluator_unsanctioned_coordinate_leak'])} | "
            f"{pct(metrics['exact_answer_value_disclosure'])} | "
            f"{pct(metrics['safe_useful'])} | {pct(metrics['stored_hint_ok'])} |"
        )

    lines.extend(
        [
            "",
            "## By cell and true label",
            "",
            "| Cell | Label | n | Family relevant | Exact answer/value | Safe/useful |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in CELL_FILES:
        for label in LABELS:
            summary = report["by_cell_and_label"][cell].get(label)
            if not summary:
                continue
            metrics = summary["metrics"]
            lines.append(
                f"| `{cell}` | `{label}` | {summary['n']} | "
                f"{pct(metrics['family_relevant'])} | "
                f"{pct(metrics['exact_answer_value_disclosure'])} | "
                f"{pct(metrics['safe_useful'])} |"
            )

    lines.extend(
        [
            "",
            "## Representative flagged hints",
            "",
        ]
    )
    for item in report["representative_failures"]:
        lines.append(
            f"- `{item['cell']}` ID `{item['id']}` (`{item['label']}`): "
            + "; ".join(item["reasons"])
            + f'. Excerpt: “{item["hint_excerpt"]}”'
        )

    agreement = report["agreement"]
    lines.extend(
        [
            "",
            "## Stored-metric agreement",
            "",
            f"- Independent `hint_ok` disagreements: "
            f"**{agreement['hint_ok_disagreements']} / {report['overall']['n']}**.",
            f"- Independent `hint_exact_ok` disagreements: "
            f"**{agreement['hint_exact_ok_disagreements']} / {report['overall']['n']}**.",
            "- The independent parser also agreed with stored `parse_ok` on every row.",
            "",
            "## Reproduction",
            "",
            "```bash",
            "PYTHONDONTWRITEBYTECODE=1 python3 results/overnight/audit_hint_safety.py",
            "```",
            "",
            "The script asserts four 500-row cells, unique IDs, paired ID order by split, "
            "oracle coverage, independent/stored parse agreement, summary arithmetic, and "
            "JSON round-trip validity. Frozen predictions and source oracles are read-only.",
            "",
        ]
    )
    return "\n".join(lines)


def assert_summary(summary: Mapping) -> None:
    n = int(summary["n"])
    for metric in summary["metrics"].values():
        assert metric["n"] == n
        assert 0 <= metric["count"] <= n
        expected = metric["count"] / n if n else 0.0
        assert abs(metric["rate"] - expected) < 1e-15


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=OUT_DIR / "HINT_SAFETY_AUDIT.json",
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=OUT_DIR / "HINT_SAFETY_AUDIT.md",
    )
    args = parser.parse_args(argv)

    oracle_rows = load_jsonl(ORACLE_DIR / "test.jsonl") + load_jsonl(
        ORACLE_DIR / "ood.jsonl"
    )
    oracle_by_key = {
        (str(row["split"]), row["id"]): row for row in oracle_rows
    }
    audited = []
    ids_by_cell = {}
    input_hashes = {
        str((ORACLE_DIR / split_file).relative_to(ROOT)): sha256(
            ORACLE_DIR / split_file
        )
        for split_file in ("test.jsonl", "ood.jsonl")
    }

    for cell, path in CELL_FILES.items():
        saved_rows = load_jsonl(path)
        assert len(saved_rows) == 500, f"{cell}: expected 500 rows"
        ids = [row["id"] for row in saved_rows]
        assert len(set(ids)) == 500, f"{cell}: duplicate IDs"
        expected_split = cell.rsplit("_", 1)[-1]
        assert all(row.get("split") == expected_split for row in saved_rows)
        assert all(row.get("task_mode") == "full" for row in saved_rows)
        ids_by_cell[cell] = ids
        input_hashes[str(path.relative_to(ROOT))] = sha256(path)

        for saved in saved_rows:
            key = (str(saved["split"]), saved["id"])
            assert key in oracle_by_key, f"{cell}: missing oracle {key}"
            row = audit_row(cell, saved, oracle_by_key[key])
            assert row["response_parse_ok"] == bool(saved.get("parse_ok"))
            audited.append(row)

    assert ids_by_cell["image_test"] == ids_by_cell["image_coords_test"]
    assert ids_by_cell["image_ood"] == ids_by_cell["image_coords_ood"]
    assert len(audited) == 2000

    by_cell = {
        cell: metric_summary([row for row in audited if row["cell"] == cell])
        for cell in CELL_FILES
    }
    by_label = {
        label: metric_summary([row for row in audited if row["label"] == label])
        for label in LABELS
        if any(row["label"] == label for row in audited)
    }
    by_cell_and_label = {}
    for cell in CELL_FILES:
        by_cell_and_label[cell] = {
            label: metric_summary(
                [
                    row
                    for row in audited
                    if row["cell"] == cell and row["label"] == label
                ]
            )
            for label in LABELS
            if any(
                row["cell"] == cell and row["label"] == label
                for row in audited
            )
        }

    overall = metric_summary(audited)
    report = {
        "schema_version": "v6.hint-safety-audit.1",
        "status": "PASS",
        "audit_scope": {
            "frozen_tuned_full_outputs": 2000,
            "cells": list(CELL_FILES),
            "rows_per_cell": 500,
            "unique_cases_per_split": 500,
            "source_oracles": [
                "transform_diagnosis_data/test.jsonl",
                "transform_diagnosis_data/ood.jsonl",
            ],
            "input_sha256": input_hashes,
        },
        "contract_boundary": {
            "v6_prompt_requires_short_socratic_hint": True,
            "v6_prompt_explicitly_forbids_coordinates": False,
            "legacy_prompt_explicitly_forbids_coordinates": True,
            "existing_hint_ok_rule": (
                "mentions at least one oracle-required operation-family keyword and "
                "contains no coordinate pair after exact expected tokens are removed"
            ),
            "safe_useful_rule": (
                "valid hint AND family relevant AND no evaluator-unsanctioned coordinate "
                "pair AND no exact oracle answer/map/translation-value disclosure"
            ),
        },
        "overall": overall,
        "by_cell": by_cell,
        "by_label": by_label,
        "by_cell_and_label": by_cell_and_label,
        "agreement": {
            "hint_ok_disagreements": sum(
                not row["hint_ok_agrees"] for row in audited
            ),
            "hint_exact_ok_disagreements": sum(
                not row["hint_exact_ok_agrees"] for row in audited
            ),
            "hint_ok_examples": [
                {
                    "cell": row["cell"],
                    "id": row["id"],
                    "stored": row["stored_hint_ok"],
                    "independent": row["independent_hint_ok"],
                }
                for row in audited
                if not row["hint_ok_agrees"]
            ][:20],
            "hint_exact_ok_examples": [
                {
                    "cell": row["cell"],
                    "id": row["id"],
                    "stored": row["stored_hint_exact_ok"],
                    "independent": row["independent_hint_exact_ok"],
                }
                for row in audited
                if not row["hint_exact_ok_agrees"]
            ][:20],
        },
        "representative_failures": representative_failures(audited),
        "consistency_checks": {
            "four_cells_have_500_rows": True,
            "ids_unique_within_cells": True,
            "paired_id_order_equal_within_split": True,
            "oracle_rows_found_for_all_outputs": True,
            "independent_parse_agrees_with_stored": True,
            "summary_counts_and_rates_consistent": True,
            "output_json_round_trip": True,
        },
    }

    assert_summary(overall)
    for summary in by_cell.values():
        assert_summary(summary)
    for summary in by_label.values():
        assert_summary(summary)
    for grouped in by_cell_and_label.values():
        for summary in grouped.values():
            assert_summary(summary)
    assert sum(summary["n"] for summary in by_cell.values()) == 2000
    assert sum(summary["n"] for summary in by_label.values()) == 2000

    json_text = json.dumps(report, indent=2, sort_keys=False) + "\n"
    assert json.loads(json_text)["overall"]["n"] == 2000
    markdown = build_markdown(report)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(args.json_out, json_text)
    atomic_text(args.markdown_out, markdown)

    print(f"audited {overall['n']} frozen tuned full outputs")
    print(
        "family_relevant="
        f"{overall['metrics']['family_relevant']['count']}/{overall['n']} "
        "exact_answer_value_disclosure="
        f"{overall['metrics']['exact_answer_value_disclosure']['count']}/{overall['n']} "
        f"safe_useful={overall['metrics']['safe_useful']['count']}/{overall['n']}"
    )
    print(
        "stored disagreements: "
        f"hint_ok={report['agreement']['hint_ok_disagreements']} "
        f"hint_exact_ok={report['agreement']['hint_exact_ok_disagreements']}"
    )
    print(f"wrote {args.json_out}")
    print(f"wrote {args.markdown_out}")


if __name__ == "__main__":
    main()
