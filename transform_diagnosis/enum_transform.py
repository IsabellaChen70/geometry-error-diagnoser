"""enum_transform — the v5 DISCRETE / STRUCTURED representation of a transform sequence.

The v1–v4 target names the intended RED->GREEN transform as FREE TEXT
(``["rotate 90 degrees counterclockwise", "translate 7 left"]``). The ``transform_match``
metric there stayed floored (~0.05–0.08) even when the model was fed exact coordinates,
so exact-transform recovery is reasoning/capacity bound, not perception bound.

v5 reframes exact-transform recovery from a *generation* problem into a *classification*
over a SMALL DISCRETE VOCABULARY. This module is the single source of that vocabulary and
the deterministic, loss-less bridge between the canonical :mod:`transform_core` math and
the enum schema. It adds NO new geometry: every value is derived from a
``transform_core.Transform`` via the public factories, so the enum can never drift from
the math.

The enum schema (one dict per primitive step; a sequence is a list of these):

    rotation    {"type": "rotation",    "param": <one of ROTATION_PARAMS>}
    reflection  {"type": "reflection",  "param": <one of REFLECTION_PARAMS>}
    translation {"type": "translation", "dx": <int>, "dy": <int>}

Vocabulary ACTUALLY present in the dataset (enumerated from the generator, not invented):

  * step types      : rotation, reflection, translation
  * rotation params : rot_ccw_90, rot_180, rot_ccw_270   (the 3 rotation matrices, all
                      about the origin; canonicalized to CCW. rot_ccw_270 == "90 clockwise")
  * reflection params: reflect_x, reflect_y, reflect_y=x, reflect_y=-x  (the 4 mirror lines)
  * translations    : two integers dx, dy (observed [-8, 8] for correct answers)
  * step counts     : 1 (single-step curriculum) or 2 (the two-step problems)

Because the mapping is a bijection on the primitive-step space, ``seq_enum`` and
``enum_to_transforms`` round-trip exactly, and :func:`steps_match` can score a prediction
by EXACT per-step comparison (type + param, or dx/dy) — a stricter, cleaner measure of
"did the model name the exact transform" than composing to the same net map.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

from . import transform_core as tc

# --------------------------------------------------------------------------------------
# Vocabulary (derived from transform_core's factories; the ACTUAL set present in the data)
# --------------------------------------------------------------------------------------

STEP_TYPES = ("rotation", "reflection", "translation")

# Rotation params: keyed by the canonical CCW degree from transform_core (90/180/270).
# 180 carries no direction; rot_ccw_270 is the same matrix as a 90-degree CLOCKWISE turn.
_ROT_DEG_TO_PARAM: Dict[int, str] = {90: "rot_ccw_90", 180: "rot_180", 270: "rot_ccw_270"}
_PARAM_TO_ROT_DEG: Dict[str, int] = {p: d for d, p in _ROT_DEG_TO_PARAM.items()}
ROTATION_PARAMS = tuple(_ROT_DEG_TO_PARAM[d] for d in (90, 180, 270))

# Reflection params: keyed by transform_core's canonical line names.
_LINE_TO_PARAM: Dict[str, str] = {
    "x": "reflect_x", "y": "reflect_y", "y=x": "reflect_y=x", "y=-x": "reflect_y=-x",
}
_PARAM_TO_LINE: Dict[str, str] = {p: l for l, p in _LINE_TO_PARAM.items()}
REFLECTION_PARAMS = tuple(_LINE_TO_PARAM[l] for l in ("x", "y", "y=x", "y=-x"))

# Reverse lookups from a primitive linear part to its enum param, built ONCE from the
# public transform_core factories so this module never reaches into transform_core privates
# and can never disagree with the math.
_ROT_MATRIX_TO_PARAM = {tc.rotate(d).matrix: p for d, p in _ROT_DEG_TO_PARAM.items()}
_REFL_MATRIX_TO_PARAM = {tc.reflect(l).matrix: p for l, p in _LINE_TO_PARAM.items()}

Step = Dict[str, object]
TransformLike = Union[tc.Transform, str]


# --------------------------------------------------------------------------------------
# Transform  ->  enum
# --------------------------------------------------------------------------------------

def step_enum(t: TransformLike) -> Step:
    """The canonical enum dict for ONE primitive step (a ``Transform`` or schema string).

    Rotations/reflections yield ``{"type", "param"}``; translations yield
    ``{"type", "dx", "dy"}``. Wording variants of a rotation (e.g. "270 degrees
    clockwise") collapse to the same enum because the param is derived from the MATRIX,
    not the text. Raises ``ValueError`` on a non-primitive transform.
    """
    t = tc.as_transform(t)
    if t.matrix == tc.IDENTITY_MATRIX:
        dx, dy = t.vec
        return {"type": "translation", "dx": int(dx), "dy": int(dy)}
    if t.det() == 1:
        param = _ROT_MATRIX_TO_PARAM.get(t.matrix)
        if param is None:
            raise ValueError(f"not a lattice rotation: {t.matrix!r}")
        return {"type": "rotation", "param": param}
    param = _REFL_MATRIX_TO_PARAM.get(t.matrix)
    if param is None:
        raise ValueError(f"not a lattice reflection: {t.matrix!r}")
    return {"type": "reflection", "param": param}


def seq_enum(seq: Sequence[TransformLike]) -> List[Step]:
    """Enum representation of a whole step sequence (order preserved)."""
    return [step_enum(s) for s in seq]


# --------------------------------------------------------------------------------------
# enum  ->  Transform  (inverse; for round-trip verification)
# --------------------------------------------------------------------------------------

def enum_step_to_transform(step: Step) -> tc.Transform:
    """Inverse of :func:`step_enum` for one step. Raises ``ValueError`` on a bad enum."""
    if not isinstance(step, dict):
        raise ValueError(f"enum step must be a dict, got {type(step).__name__}")
    typ = step.get("type")
    if typ == "translation":
        return tc.translate(int(step["dx"]), int(step["dy"]))
    if typ == "rotation":
        deg = _PARAM_TO_ROT_DEG.get(step.get("param"))
        if deg is None:
            raise ValueError(f"unknown rotation param: {step.get('param')!r}")
        return tc.rotate(deg)  # canonical CCW
    if typ == "reflection":
        line = _PARAM_TO_LINE.get(step.get("param"))
        if line is None:
            raise ValueError(f"unknown reflection param: {step.get('param')!r}")
        return tc.reflect(line)
    raise ValueError(f"unknown step type: {typ!r}")


def enum_to_transforms(seq: Sequence[Step]) -> List[tc.Transform]:
    """Inverse of :func:`seq_enum` for a whole sequence."""
    return [enum_step_to_transform(s) for s in seq]


# --------------------------------------------------------------------------------------
# Detection + EXACT per-step matching (the separate ordered-sequence diagnostic)
# --------------------------------------------------------------------------------------

def is_enum_step(x: object) -> bool:
    """True iff ``x`` looks like an enum step dict (has a known ``type``)."""
    return isinstance(x, dict) and x.get("type") in STEP_TYPES


def is_enum_seq(x: object) -> bool:
    """True iff ``x`` is a non-empty sequence of enum step dicts (the v5 format).

    Distinguishes the structured target from the prose list-of-strings (v1–v4), so callers
    can dispatch scoring on the format. An empty list is NOT treated as enum.
    """
    return (
        isinstance(x, (list, tuple))
        and len(x) > 0
        and all(is_enum_step(s) for s in x)
    )


def _translation_val(step: Step) -> tuple:
    """Coerce a translation step's (dx, dy) to ints, or raise for a malformed step."""
    return (int(step["dx"]), int(step["dy"]))


def step_match(pred: object, gold: Step) -> bool:
    """EXACT match of one predicted step against the gold enum step.

    Rotations/reflections must agree on ``type`` AND ``param``; translations must agree on
    ``type`` AND both ``dx`` and ``dy``. Any structural malformation in ``pred`` -> False.
    """
    if not isinstance(pred, dict):
        return False
    if pred.get("type") != gold.get("type"):
        return False
    if gold["type"] == "translation":
        try:
            return _translation_val(pred) == _translation_val(gold)
        except (KeyError, TypeError, ValueError):
            return False
    return pred.get("param") == gold.get("param")


def steps_match(pred: object, gold: Sequence[Step]) -> bool:
    """EXACT match of a predicted step sequence against the gold enum sequence.

    True iff same length and every step matches (:func:`step_match`). Evaluation exposes
    this as ``step_sequence_exact_ok``. The apples-to-apples ``transform_ok`` headline
    composes enum steps and compares the observable net affine map.
    """
    if not isinstance(pred, (list, tuple)):
        return False
    if len(pred) != len(gold):
        return False
    return all(step_match(p, g) for p, g in zip(pred, gold))


# --------------------------------------------------------------------------------------
# Human-readable phrasing (for the v5 chain-of-thought transform-readout line; brace-free)
# --------------------------------------------------------------------------------------

def describe_step(step: Step) -> str:
    """A short, brace-free phrase for one enum step, used in the reasoning trace.

    Deliberately avoids ``{`` / ``}`` so it never confuses the eval JSON extractor
    (``eval._last_json_object``), matching the v4 structured-trace discipline.
    """
    if step["type"] == "translation":
        return f"translation dx={int(step['dx'])} dy={int(step['dy'])}"
    return f"{step['type']} {step['param']}"


def describe_seq(seq: Sequence[Step]) -> str:
    """Render an enum sequence as "step 1 <...>; step 2 <...>" (brace-free)."""
    return "; ".join(f"step {i} {describe_step(s)}" for i, s in enumerate(seq, start=1))
