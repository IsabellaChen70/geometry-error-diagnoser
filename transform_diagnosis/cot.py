"""cot — deterministic chain-of-thought reasoning traces for diagnosis records.

For chain-of-thought (CoT) fine-tuning we want the assistant to emit a short,
step-by-step geometric reasoning trace BEFORE the final JSON diagnosis. Small models
often do better on multi-step tasks when they "show their work" first, so the training
target becomes::

    <reasoning trace>
    {"label": ..., "correct_transform": [...], "hint": ...}

This module builds that trace **deterministically from a record's GROUND-TRUTH
transforms**, reusing the single canonical transform math in
:mod:`transform_core` (and mirroring the step/family logic used by :mod:`hints`).
Because the trace is derived from the same oracle fields that produced the label and
the JSON, it can NEVER contradict them — the wording is generated, not guessed.

The trace walks the actual diagnostic logic (the same decision tree as
:func:`transform_core.diagnose`):

  1. state the intended (correct) transformation mapping RED -> GREEN,
  2. state what the student's BLUE answer corresponds to,
  3. compare them to name the specific discrepancy (operation type vs parameter),
  4. conclude with the label.

The JSON appended after the trace is EXACTLY :func:`chat_format.target_json` — the same
target the non-CoT training uses — and the user turn is EXACTLY
:func:`chat_format.to_messages`'s user turn (image + instruction, unchanged). So the
image input and the scored final JSON are identical to the non-CoT pipeline; only a
reasoning prefix is added to the assistant target. Eval scores just the final JSON
(see :func:`eval.parse_pred`, which pulls the LAST JSON object), so it is unaffected.

Dependency-light on purpose (no PIL / matplotlib): like :mod:`chat_format` it references
each image by its ``render_path`` string.
"""

from __future__ import annotations

import json
from typing import Dict, List, Mapping, Sequence

from . import chat_format
from . import enum_transform as et
from . import transform_core as tc

# Ordinal word for a step index (all problems are exactly two steps, but keep it general).
_ORDINALS = {0: "first", 1: "second", 2: "third", 3: "fourth"}

# v4 STRUCTURED FIELDS: a short, deterministic phrase naming the primary discrepancy for
# each label. Emitted as the JSON ``main_mismatch`` field and echoed in the (structured)
# trace's operation-type check line. Plain text only -- NO braces -- so the reasoning-prefix
# JSON extractor (``eval._last_json_object``) is never confused by them.
_MAIN_MISMATCH = {
    "correct": "none -- the student's operations match the expected operations",
    "reflection_instead_of_rotation": "used a reflection where a rotation was required",
    "rotation_instead_of_reflection": "used a rotation where a reflection was required",
    "wrong_rotation_angle": "rotated by the wrong angle (correct operation type)",
    "wrong_reflection_line": "reflected across the wrong line (correct operation type)",
    "wrong_translation": "translated by the wrong vector",
    "opposite_translation": "translated in the opposite direction",
    "completely_wrong": "both the operation type and the translation are wrong",
}

# The concept clause naming the specific discrepancy for each single-step-error label.
# Derived-nothing here: these are fixed English phrasings for the fixed label set; every
# concrete transform named in the trace still flows through transform_core.describe_transform.
_SINGLE_STEP_CLAUSE = {
    "reflection_instead_of_rotation": "a reflection was used where a rotation was required",
    "rotation_instead_of_reflection": "a rotation was used where a reflection was required",
    "wrong_rotation_angle": "the rotation is correct in kind but turned by the wrong angle",
    "wrong_reflection_line": "the reflection is correct in kind but across the wrong line",
    "wrong_translation": "the slide is in roughly the right spirit but by the wrong vector",
    "opposite_translation": "the slide is the right distance but in the opposite direction",
}


def _steps(seq: Sequence) -> List[tc.Transform]:
    """Parse a stored schema-string list into canonical ``Transform`` objects."""
    return [tc.as_transform(s) for s in seq]


def _desc(t: tc.Transform) -> str:
    """Canonical single-step description (ccw rotations), via the single source of truth.

    Matches the wording used by ``hints`` so the trace and the gold hint agree; it may
    differ from a record's STORED rotation wording (e.g. "270 degrees clockwise") while
    naming the identical motion — exactly the relationship the gold hint already has with
    ``correct_transform``. Eval compares transforms semantically, so this is consistent.
    """
    return tc.describe_transform(t)


def _family(t: tc.Transform) -> str:
    """Operation family of a single primitive step, from the math alone (mirrors hints)."""
    if t.matrix == tc.IDENTITY_MATRIX:
        return "translation"
    return "rotation" if t.det() == 1 else "reflection"


def _join_steps(descs: Sequence[str]) -> str:
    """Render an ordered step list as "first A, then B" (any length)."""
    descs = list(descs)
    if len(descs) == 1:
        return descs[0]
    return "first " + ", then ".join(descs)


def _diff_indices(correct: Sequence[tc.Transform], student: Sequence[tc.Transform]) -> List[int]:
    """Positions where the correct and student STEPS differ (by exact math, not wording)."""
    n = min(len(correct), len(student))
    return [i for i in range(n) if correct[i] != student[i]]


def _comparison(label: str, correct: List[tc.Transform], student: List[tc.Transform]) -> str:
    """The comparison + conclusion sentence(s), derived from the transforms.

    Walks the same case split as ``transform_core.diagnose`` and always ends by naming
    ``label`` verbatim (the exact token the model must emit in the JSON).
    """
    if label == "correct":
        return (
            "Every step matches the intended transformation, so the BLUE answer lands "
            "exactly on the GREEN correct image. So the diagnosis is correct."
        )

    if label == "completely_wrong":
        c = tc.compose(correct)
        s = tc.compose(student)
        c_lin = _desc(tc.Transform(c.matrix, (0, 0)))
        s_lin = _desc(tc.Transform(s.matrix, (0, 0)))
        return (
            f"Composing each answer, the correct net map is a {c.orientation} ({c_lin}) "
            f"with translation ({c.vec[0]}, {c.vec[1]}), but the student's net map is a "
            f"{s.orientation} ({s_lin}) with translation ({s.vec[0]}, {s.vec[1]}). Both the "
            "transformation and the translation are wrong. So the diagnosis is completely_wrong."
        )

    # All remaining labels are single-step mutations: exactly one step differs.
    diffs = _diff_indices(correct, student)
    if len(diffs) == 1 and len(correct) in (1, 2):
        i = diffs[0]
        clause = _SINGLE_STEP_CLAUSE.get(label, "that step is wrong")
        if len(correct) == 1:  # single-step curriculum problem: no "other step" to praise
            return (
                f"The single transformation is wrong: the task requires {_desc(correct[i])} "
                f"(a {_family(correct[i])}), while the student applied {_desc(student[i])} "
                f"(a {_family(student[i])}). Here {clause}. So the diagnosis is {label}."
            )
        shared = _desc(correct[1 - i])
        return (
            f"The {_ORDINALS[1 - i]} step ({shared}) is correct, but the {_ORDINALS[i]} "
            f"step is wrong: the task requires {_desc(correct[i])} (a {_family(correct[i])}), "
            f"while the student applied {_desc(student[i])} (a {_family(student[i])}). "
            f"Here {clause}. So the diagnosis is {label}."
        )

    # Defensive fallback (should not happen for verified records): compare net maps.
    c = tc.compose(correct)
    s = tc.compose(student)
    return (
        f"The student's net map ({_desc(tc.Transform(s.matrix, (0, 0)))}, translation "
        f"({s.vec[0]}, {s.vec[1]})) does not match the intended net map "
        f"({_desc(tc.Transform(c.matrix, (0, 0)))}, translation ({c.vec[0]}, {c.vec[1]})). "
        f"So the diagnosis is {label}."
    )


# --------------------------------------------------------------------------------------
# v4 structured intermediate fields (operation-type reasoning made explicit + machine-readable)
# --------------------------------------------------------------------------------------

def operation_types(seq: Sequence) -> List[str]:
    """The operation family ("rotation"/"reflection"/"translation") of each step in a
    stored schema-string list, in order. Derived from the math via ``transform_core``."""
    return [_family(t) for t in _steps(seq)]


def main_mismatch(record: Mapping) -> str:
    """The short, deterministic phrase naming the primary discrepancy for this record's
    label (see ``_MAIN_MISMATCH``). Plain text, no braces."""
    label = record["label"]
    if label not in tc.DIAGNOSIS_LABELS:
        raise ValueError(f"unknown label: {label!r}")
    return _MAIN_MISMATCH[label]


def structured_fields(record: Mapping) -> Dict[str, object]:
    """The v4 structured intermediate fields, ALL derived deterministically from the
    record's ground-truth transforms via ``transform_core``:

      * ``expected_operation_types``  -- families of the intended (correct) steps
      * ``student_operation_types``   -- families of the student's steps
      * ``main_mismatch``             -- short phrase naming the primary discrepancy

    They cannot contradict the label because they are computed from the same oracle fields.
    """
    return {
        "expected_operation_types": operation_types(record["correct_transform"]),
        "student_operation_types": operation_types(record["student_transform"]),
        "main_mismatch": main_mismatch(record),
    }


def _type_check_line(fields: Mapping) -> str:
    """The structured trace line that walks the operation-TYPE comparison explicitly (the
    v4 signal: name the expected vs student operation types, then the main mismatch)."""
    exp = ", ".join(fields["expected_operation_types"])
    stu = ", ".join(fields["student_operation_types"])
    return (
        f"Operation-type check: the intended operations are [{exp}] and the student's are "
        f"[{stu}]; main mismatch: {fields['main_mismatch']}."
    )


def structured_target_obj(record: Mapping) -> Dict[str, object]:
    """The v4 final answer object: the three SCORED keys (``label``, ``correct_transform``,
    ``hint``) FIRST and unchanged, followed by the three structured type keys. ``eval``
    reads only the first three (by key), so the extra keys never affect scoring."""
    obj = dict(chat_format.target_obj(record))       # label, correct_transform, hint
    obj.update(structured_fields(record))
    return obj


def structured_json(record: Mapping) -> str:
    """The v4 final answer serialized compactly (same style as ``chat_format.target_json``)."""
    return json.dumps(structured_target_obj(record), ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# v5 ENUM target: correct_transform becomes a DISCRETE structured classification target
# (``enum_transform``), keeping every v4 field, plus a transform-readout trace line.
# --------------------------------------------------------------------------------------

def _transform_readout_line(record: Mapping) -> str:
    """The v5 trace line stating the observed RED->GREEN step types/params in the enum
    vocabulary (brace-free, so the JSON extractor is never confused). This ties the
    reasoning directly to the discrete classification the model must emit."""
    enum_seq = et.seq_enum(record["correct_transform"])
    return (
        f"Transform readout: mapping RED onto GREEN is {et.describe_seq(enum_seq)} "
        f"(choose each type from {list(et.STEP_TYPES)}; rotations/reflections take one "
        f"param, translations take integer dx/dy)."
    )


def enum_target_obj(record: Mapping, *, transform_first: bool = False) -> Dict[str, object]:
    """The v5 final answer object: the v4 SCORED keys with ``correct_transform`` swapped
    from prose to the DISCRETE enum form (:func:`enum_transform.seq_enum`), followed by the
    v4 structured type keys. ``label`` and ``hint`` are unchanged; ``eval`` reads the three
    scored keys by NAME, so the enum swap and any key reordering never break scoring.

    ``transform_first=True`` (the curriculum "transform-first emphasis" knob) foregrounds
    the classification target by emitting ``correct_transform`` FIRST in the JSON, so the
    model commits to the transform before the label. Order-only; scoring is by key.
    """
    obj = dict(chat_format.target_obj(record))            # label, correct_transform, hint
    obj["correct_transform"] = et.seq_enum(record["correct_transform"])  # prose -> enum
    obj.update(structured_fields(record))                 # v4 type keys
    if transform_first:
        lead = ["correct_transform", "label", "hint"]
        obj = {**{k: obj[k] for k in lead}, **{k: v for k, v in obj.items() if k not in lead}}
    return obj


def enum_json(record: Mapping, *, transform_first: bool = False) -> str:
    """The v5 final answer serialized compactly (same style as ``chat_format.target_json``)."""
    return json.dumps(enum_target_obj(record, transform_first=transform_first),
                      ensure_ascii=False, separators=(",", ":"))


# --------------------------------------------------------------------------------------
# Trace + target assembly
# --------------------------------------------------------------------------------------

def reasoning_trace(
    record: Mapping, *, structured: bool = False, enum_transform: bool = False
) -> str:
    """Build the deterministic step-by-step reasoning trace for ``record``.

    Uses only the record's ``correct_transform`` / ``student_transform`` (and ``label``),
    all routed through ``transform_core``. The returned trace is a short paragraph of a
    few lines and always concludes with the record's exact ``label``.

    ``structured=True`` (v4) inserts one extra "operation-type check" line before the
    conclusion, making the rotation-vs-reflection type comparison explicit.

    ``enum_transform=True`` (v5) implies the structured type-check line AND appends a
    "transform readout" line naming the RED->GREEN step types/params in the DISCRETE enum
    vocabulary, tying the reasoning to the classification target the model emits.
    """
    label = record["label"]
    if label not in tc.DIAGNOSIS_LABELS:
        raise ValueError(f"unknown label: {label!r}")

    correct = _steps(record["correct_transform"])
    student = _steps(record["student_transform"])
    c_descs = [_desc(t) for t in correct]
    s_descs = [_desc(t) for t in student]

    lines = [
        f"The intended transformation maps the RED pre-image onto the GREEN correct "
        f"image: {_join_steps(c_descs)}.",
        f"The student's BLUE answer corresponds to: {_join_steps(s_descs)}.",
    ]
    if structured or enum_transform:
        lines.append(_type_check_line(structured_fields(record)))
    if enum_transform:
        lines.append(_transform_readout_line(record))
    lines.append(_comparison(label, correct, student))
    return "\n".join(lines)


def cot_target(
    record: Mapping, *, structured: bool = False, enum_transform: bool = False,
    transform_first: bool = False,
) -> str:
    """The full assistant target for CoT training: reasoning trace, then the final JSON.

    * ``structured=False`` (v3cot): JSON is EXACTLY ``chat_format.target_json`` (three scored keys).
    * ``structured=True`` (v4): JSON adds the structured type fields; trace gains the type-check line.
    * ``enum_transform=True`` (v5): JSON is the v4 object with ``correct_transform`` swapped to the
      DISCRETE enum form (:func:`enum_json`); trace gains the transform-readout line.
      ``transform_first`` foregrounds ``correct_transform`` in the JSON key order.

    In ALL cases the JSON is placed LAST so ``eval.parse_pred`` recovers it as the final
    object, and the scored keys (label/correct_transform/hint) are present.
    """
    trace = reasoning_trace(record, structured=structured, enum_transform=enum_transform)
    if enum_transform:
        tail = enum_json(record, transform_first=transform_first)
    elif structured:
        tail = structured_json(record)
    else:
        tail = chat_format.target_json(record)
    return trace + "\n" + tail


def to_cot_conversation(
    record: Mapping, *, image_path: str | None = None, structured: bool = False,
    enum_transform: bool = False, transform_first: bool = False,
) -> dict:
    """One CoT training row ``{"id", "split", "messages"}``.

    The user turn (image + instruction) is IDENTICAL to :func:`chat_format.to_messages`'s
    user turn; only the assistant turn changes to ``cot_target`` (trace + JSON).
    ``structured=True`` emits the v4 structured target; ``enum_transform=True`` emits the
    v5 discrete-enum target (with the optional ``transform_first`` key ordering).
    """
    user_turn = chat_format.to_messages(record, image_path=image_path)[0]
    target = cot_target(record, structured=structured, enum_transform=enum_transform,
                        transform_first=transform_first)
    return {
        "id": record["id"],
        "split": record["split"],
        "messages": [
            user_turn,
            {"role": "assistant", "content": [{"type": "text", "text": target}]},
        ],
    }


def build_cot_rows(
    records: Sequence[Mapping], *, structured: bool = False, enum_transform: bool = False,
    transform_first: bool = False,
) -> List[dict]:
    """Build CoT conversation rows for a sequence of records (order preserved)."""
    return [to_cot_conversation(r, structured=structured, enum_transform=enum_transform,
                                transform_first=transform_first) for r in records]
