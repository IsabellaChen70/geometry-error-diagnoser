"""hints — deterministic, coordinate-free Socratic tutor hints for diagnosis records.

Each gold hint is a short Socratic nudge that names WHICH kind of mistake the
student made — the wrong operation *family* (rotation / reflection / translation)
and which aspect is off (angle vs line vs direction vs type vs "everything") —
WITHOUT ever stating the answer. Hints deliberately contain no coordinate pair,
no canonical axis/line/angle literal, and no translation value, so they can never
leak the solution. This is the contract ``dataset._assert_record`` enforces at
write time (family-relevant AND not a strict leak); it replaces the old contract,
under which ``hint_for`` embedded the exact ``expected_hint_tokens`` and ~96% of
tuned-model hints reproduced them and leaked the answer.

Public functions:

* :func:`hint_for` — the deterministic, concept-only hint string for a record.
* :func:`is_strict_leak` / :func:`strict_leak_reasons` — the shared strict
  hint-safety check (a faithful port of ``results/overnight/audit_hint_safety.py``)
  reused by dataset generation and the enforceable contract test, so all three
  share ONE definition of "strict leak" and cannot drift.
* :func:`expected_hint_tokens` — the exact canonical schema string(s) for a label
  (the correct/student axis, angle, or translation — effectively the answer).
  Kept for ``eval``, which uses them for the strict secondary ``hint_exact_ok``
  diagnostic and to remove *sanctioned* tokens before scanning for a leak. Gold
  hints no longer contain these.
* :func:`expected_hint_families` — the operation family/families
  ("rotation"/"reflection"/"translation") a correct hint must reference. Unlike
  the exact tokens these are achievable without stating the answer, and they back
  both the primary ``eval`` hint metric and the write-time hint contract.

All take ``(label, rec)`` where ``rec`` is any mapping exposing the two schema
key lists ``"correct_transform"`` and ``"student_transform"`` (a full record or
the partial built in :mod:`dataset`).
"""

from __future__ import annotations

import re
from typing import List, Mapping, Sequence

from . import net_transform as nt
from . import transform_core as tc


# --------------------------------------------------------------------------------------
# Small helpers (all wording flows through transform_core.describe_transform)
# --------------------------------------------------------------------------------------

def _parse(seq: Sequence) -> List[tc.Transform]:
    return [tc.as_transform(s) for s in seq]


def _desc(t: tc.Transform) -> str:
    """Canonical single-step description (ccw rotations) via the single source of truth."""
    return tc.describe_transform(t)


def _vec_str(vec) -> str:
    """Render a net translation vector as a literal ``(dx, dy)`` token."""
    return f"({vec[0]}, {vec[1]})"


def _family(t: tc.Transform) -> str:
    """Operation family of a single primitive step, from the math alone:
    ``"translation"`` (identity linear part), ``"rotation"`` (det +1), or
    ``"reflection"`` (det -1)."""
    if t.matrix == tc.IDENTITY_MATRIX:
        return "translation"
    return "rotation" if t.det() == 1 else "reflection"


def _correct_student(rec: Mapping):
    return _parse(rec["correct_transform"]), _parse(rec["student_transform"])


def _diff_indices(correct: Sequence[tc.Transform], student: Sequence[tc.Transform]) -> List[int]:
    """Positions where the correct and student STEPS differ (by exact math, not wording)."""
    n = min(len(correct), len(student))
    return [i for i in range(n) if correct[i] != student[i]]


def _single_diff(correct, student) -> int:
    """Index of the one step a single-step error injector mutated (fails loudly otherwise)."""
    idxs = _diff_indices(correct, student)
    if len(idxs) != 1:
        raise ValueError(f"expected exactly one differing step, got {idxs}")
    return idxs[0]


# --------------------------------------------------------------------------------------
# Per-label token extraction (the deterministically-expected substrings)
# --------------------------------------------------------------------------------------

def expected_hint_tokens(label: str, rec: Mapping) -> List[str]:
    """Return the exact canonical schema string(s) that WOULD name the answer.

    Every token is recomputed from the record's transforms through
    ``transform_core`` — the correct/student axis, angle, or translation. Gold
    hints no longer contain these (that was the leaky contract); ``eval`` still
    uses them for the strict, secondary ``hint_exact_ok`` diagnostic and to strip
    *sanctioned* substrings before scanning for a residual coordinate leak, and
    :func:`strict_leak_reasons` treats any of them appearing in a hint as a leak.
    """
    correct, student = _correct_student(rec)

    if label == "correct":
        return [_desc(t) for t in correct]

    if label == "completely_wrong":
        c = tc.compose(correct)
        s = tc.compose(student)
        return [
            _desc(tc.Transform(c.matrix, (0, 0))),
            _desc(tc.Transform(s.matrix, (0, 0))),
            _vec_str(c.vec),
            _vec_str(s.vec),
        ]

    # All remaining labels are single-step mutations: name the correct step and
    # the student's substituted step.
    i = _single_diff(correct, student)
    return [_desc(correct[i]), _desc(student[i])]


def expected_hint_families(label: str, rec: Mapping) -> List[str]:
    """Operation families a correct, instruction-following hint must reference.

    Returns a subset of ``{"rotation", "reflection", "translation"}`` naming the
    concept(s) at issue for ``label`` — derived from the record's transforms via
    ``transform_core`` (the family of the differing step, or of the net map). A
    good Socratic hint can name these WITHOUT stating the answer coordinates (the
    exact strings in :func:`expected_hint_tokens`), so unlike those tokens this is
    achievable under the ``chat_format`` instruction. The eval hint metric passes
    when the hint mentions AT LEAST ONE of these families and leaks no coordinates.
    """
    correct, student = _correct_student(rec)

    if label == "correct":
        return sorted({_family(t) for t in correct})

    if label == "completely_wrong":
        c = tc.compose(correct)
        s = tc.compose(student)
        # Orientation of each net map plus translation (which always differs here).
        return sorted({
            _family(tc.Transform(c.matrix, (0, 0))),
            _family(tc.Transform(s.matrix, (0, 0))),
            "translation",
        })

    i = _single_diff(correct, student)
    return sorted({_family(correct[i]), _family(student[i])})


# --------------------------------------------------------------------------------------
# Strict hint-safety (shared with results/overnight/audit_hint_safety.py)
# --------------------------------------------------------------------------------------
#
# A gold hint may name the concept at issue but must NEVER disclose the answer. The
# patterns below are a faithful port of the frozen audit so dataset generation, the
# enforceable contract test, and the audit share ONE definition of "strict leak" and
# cannot drift. A strict leak is any of: an exact expected-token substring, a canonical
# linear-map spelling of either net map (e.g. "reflect across x axis", "rotate 90
# degrees counterclockwise", "rot_ccw_90"), an exact translation value (net or
# primitive), or a coordinate pair left after removing the sanctioned expected tokens.

# A coordinate pair with whitespace removed, matching eval._hint_has_leak.
_RESIDUAL_COORD_RE = re.compile(r"\(-?\d+,-?\d+\)")

# Exact canonical linear-map spellings, keyed by net_transform D4 name. Each net map has
# two equivalent rotation spellings (e.g. rot_ccw_90 == "rotate 270 degrees clockwise").
_LINEAR_LEAK_PATTERNS = {
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
        re.compile(
            r"\breflect(?:ion)?\s+across\s+(?:the\s+)?(?:line\s+)?y\s*=\s*x\b",
            re.IGNORECASE,
        ),
    ),
    "reflect_y_eq_neg_x": (
        re.compile(r"\breflect_y_eq_neg_x\b", re.IGNORECASE),
        re.compile(
            r"\breflect(?:ion)?\s+across\s+(?:the\s+)?(?:line\s+)?y\s*=\s*-\s*x\b",
            re.IGNORECASE,
        ),
    ),
}


def _norm_hint(text: str) -> str:
    """Lowercase + collapse whitespace — matches eval._norm for token comparison."""
    return " ".join(str(text).lower().split())


def _mentions_linear(hint: str, linear: str) -> bool:
    return any(pattern.search(hint) for pattern in _LINEAR_LEAK_PATTERNS.get(linear, ()))


def _translation_leak_patterns(vec: Sequence[int]) -> List["re.Pattern"]:
    """Ported from the audit: the exact coordinate pair, plus a magnitude+direction
    spelling for an axis-aligned slide, plus the zero-translation spellings."""
    x, y = int(vec[0]), int(vec[1])
    patterns = [re.compile(rf"\(\s*{x}\s*,\s*{y}\s*\)", re.IGNORECASE)]
    if y == 0 and x != 0:
        direction = "right" if x > 0 else "left"
        patterns.append(re.compile(
            rf"\b(?:translate|translation|shift|move|slide)\D{{0,20}}"
            rf"{abs(x)}(?:\s+(?:units?|squares?|spaces?))?\s+(?:to\s+the\s+)?{direction}\b",
            re.IGNORECASE,
        ))
    elif x == 0 and y != 0:
        direction = "up" if y > 0 else "down"
        patterns.append(re.compile(
            rf"\b(?:translate|translation|shift|move|slide)\D{{0,20}}"
            rf"{abs(y)}(?:\s+(?:units?|squares?|spaces?))?\s+(?:to\s+the\s+)?{direction}\b",
            re.IGNORECASE,
        ))
    elif (x, y) == (0, 0):
        patterns.extend((
            re.compile(r"\b(?:zero|no)\s+(?:net\s+)?translation\b", re.IGNORECASE),
            re.compile(r"\btranslate\s+0\s+(?:left|right|up|down)\b", re.IGNORECASE),
        ))
    return patterns


def _mentions_translation(hint: str, vec: Sequence[int]) -> bool:
    return any(pattern.search(hint) for pattern in _translation_leak_patterns(vec))


def _leak_translation_vectors(rec: Mapping) -> List[tuple]:
    """Every translation value the hint must not disclose: both net-map vectors plus
    every primitive translation step in either sequence."""
    correct_net = nt.sequence_to_net(rec["correct_transform"])
    student_net = nt.sequence_to_net(rec["student_transform"])
    vectors = [
        (int(correct_net["tx"]), int(correct_net["ty"])),
        (int(student_net["tx"]), int(student_net["ty"])),
    ]
    for key in ("correct_transform", "student_transform"):
        for step in rec[key]:
            t = tc.as_transform(step)
            if t.matrix == tc.IDENTITY_MATRIX and t.vec != (0, 0):
                vectors.append((int(t.vec[0]), int(t.vec[1])))
    return vectors


def strict_leak_reasons(hint: str, label: str, rec: Mapping) -> List[str]:
    """Reasons ``hint`` strictly leaks ``rec``'s answer (empty list == safe/useful).

    Mirrors ``results/overnight/audit_hint_safety.py``: a strict leak is any exact
    expected-token disclosure, any canonical linear-map spelling of either net map,
    any exact translation value (net or primitive), or any coordinate pair left after
    removing the sanctioned expected tokens.
    """
    if not isinstance(hint, str) or not hint.strip():
        return ["empty hint"]
    reasons: List[str] = []
    tokens = expected_hint_tokens(label, rec)

    residual = re.sub(r"\s+", "", hint.lower())
    for tok in tokens:
        residual = residual.replace(re.sub(r"\s+", "", str(tok).lower()), "")
    if _RESIDUAL_COORD_RE.search(residual):
        reasons.append("unsanctioned coordinate pair")

    norm_hint = _norm_hint(hint)
    if any(_norm_hint(tok) in norm_hint for tok in tokens):
        reasons.append("exact expected hint token")

    correct_linear = str(nt.sequence_to_net(rec["correct_transform"])["linear"])
    student_linear = str(nt.sequence_to_net(rec["student_transform"])["linear"])
    if _mentions_linear(hint, correct_linear) or _mentions_linear(hint, student_linear):
        reasons.append("exact canonical linear map")

    if any(_mentions_translation(hint, vec) for vec in _leak_translation_vectors(rec)):
        reasons.append("exact translation value")

    return reasons


def is_strict_leak(hint: str, label: str, rec: Mapping) -> bool:
    """True iff ``hint`` strictly leaks the answer (see :func:`strict_leak_reasons`)."""
    return bool(strict_leak_reasons(hint, label, rec))


# --------------------------------------------------------------------------------------
# Hint construction — coordinate-free Socratic nudges
# --------------------------------------------------------------------------------------

# One concept-only template per label. Each names the operation family/families at issue
# (so eval._hint_mentions_family passes for EVERY record of that label, since the family
# set is a subset of what the template mentions) and the aspect that is wrong (angle vs
# line vs direction vs type vs "everything"), while stating NO coordinate, axis/line
# literal, angle, or translation value (so is_strict_leak is False). Enforced for every
# generated record by test_dataset — do not eyeball; the contract test is authoritative.
_SOCRATIC_HINTS = {
    "correct": (
        "Everything checks out — your rotations, reflections, and slides are all "
        "applied correctly and in the right order."
    ),
    "reflection_instead_of_rotation": (
        "Look again at that step: you used a reflection where a rotation is required. "
        "Which kind of symmetry does the mapping actually need?"
    ),
    "rotation_instead_of_reflection": (
        "Look again at that step: you rotated where a reflection is required. Should "
        "the shape turn, or mirror across a line?"
    ),
    "wrong_rotation_angle": (
        "Your rotation is the right idea, but the angle is off — re-measure how far "
        "the shape should turn."
    ),
    "wrong_reflection_line": (
        "Your reflection is the right idea, but the mirror line is wrong — re-examine "
        "which line the shape should flip over."
    ),
    "wrong_translation": (
        "Your slide is off — recount how far and in which direction the shape should "
        "move."
    ),
    "opposite_translation": (
        "Check the direction of your translation — it looks reversed. Which way should "
        "the shape actually slide?"
    ),
    "completely_wrong": (
        "Both the transformation and the slide are off — rebuild the whole mapping from "
        "the original shape, rechecking the rotation or reflection and then the slide."
    ),
}


def hint_for(label: str, rec: Mapping) -> str:
    """Build the deterministic, coordinate-free Socratic hint for ``rec``'s ``label``.

    The hint names the operation family/families involved and which aspect is wrong,
    guiding self-correction WITHOUT stating any coordinate, axis/line, angle, or
    translation value. ``rec`` is accepted for API compatibility (callers pass the
    partial or full record); the wording is label-driven and its safety is verified
    per-record by ``dataset._assert_record`` and the contract test.
    """
    if label not in tc.DIAGNOSIS_LABELS:
        raise ValueError(f"unknown label: {label!r}")
    return _SOCRATIC_HINTS[label]
