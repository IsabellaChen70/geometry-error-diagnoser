"""errors — deterministic, VERIFIED student-error injection.

Given a correct :class:`problems.Problem` and a TARGET diagnosis category, produce a
student transform sequence (built only from ``transform_core.rotate/reflect/translate``)
whose independently-computed diagnosis equals the target. Every candidate is checked
with ``transform_core.diagnose`` (on both the Transform objects and their schema-string
round-trip) and its image is bounds-checked; only a verified candidate is returned.
If no candidate works for this problem, ``inject`` returns ``None`` and the caller
retries with a fresh problem. Injected label and independent diagnosis therefore ALWAYS
agree — there is no separate labelling path.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from . import geometry, transform_core as tc
from .problems import BOUND, PATTERNS, REFLECTION_LINES, ROTATION_DEGREES, Problem

StudentSeq = List[tc.Transform]
Injection = Optional[Tuple[StudentSeq, List[str]]]

# Which correct-answer patterns can reliably realize each target category. The dataset
# builder samples a problem pattern from here before calling ``inject``.
COMPATIBLE_PATTERNS = {
    "correct": list(PATTERNS),
    "reflection_instead_of_rotation": [("rotate", "translate")],
    "rotation_instead_of_reflection": [
        ("reflect", "translate"), ("rotate", "reflect"), ("reflect", "rotate"),
    ],
    "wrong_rotation_angle": [("rotate", "translate")],
    "wrong_reflection_line": [
        ("reflect", "translate"), ("rotate", "reflect"), ("reflect", "rotate"),
    ],
    "wrong_translation": [
        ("rotate", "translate"), ("translate", "rotate"),
        ("reflect", "translate"), ("translate", "reflect"),
    ],
    "opposite_translation": [
        ("rotate", "translate"), ("translate", "rotate"),
        ("reflect", "translate"), ("translate", "reflect"),
    ],
    "completely_wrong": list(PATTERNS),
}

# --------------------------------------------------------------------------------------
# Compositional OOD split — held-out compositions (TEST-ONLY generalization slice).
#
# The two rotation-reflection compositions are held out entirely from the balanced
# in-distribution set (train/val/test) and generated ONLY into the OOD slice. They are
# safe to hold out because neither is the *exclusive* compatible pattern of any label
# (see COMPATIBLE_PATTERNS above), so every label keeps at least one in-distribution
# pattern and stays trainable. (Contrast: ("rotate","translate") is the ONLY pattern for
# reflection_instead_of_rotation and wrong_rotation_angle, so it must NOT be held out.)
# --------------------------------------------------------------------------------------

HELD_OUT_PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("rotate", "reflect"),
    ("reflect", "rotate"),
)
IN_DISTRIBUTION_PATTERNS: Tuple[Tuple[str, ...], ...] = tuple(
    p for p in PATTERNS if p not in HELD_OUT_PATTERNS
)

# COMPATIBLE_PATTERNS filtered into the two disjoint pools.
ID_COMPATIBLE_PATTERNS = {
    label: [p for p in pats if p not in HELD_OUT_PATTERNS]
    for label, pats in COMPATIBLE_PATTERNS.items()
}
OOD_COMPATIBLE_PATTERNS = {
    label: [p for p in pats if p in HELD_OUT_PATTERNS]
    for label, pats in COMPATIBLE_PATTERNS.items()
}

# Labels realizable from the held-out patterns (i.e. the OOD slice's natural coverage).
# Derived, not hardcoded: exactly {correct, rotation_instead_of_reflection,
# wrong_reflection_line, completely_wrong}. The OOD slice is intentionally unbalanced.
OOD_ELIGIBLE_LABELS = [l for l in tc.DIAGNOSIS_LABELS if OOD_COMPATIBLE_PATTERNS[l]]

# Sanity: every label must retain at least one in-distribution pattern (all 8 trainable).
assert all(ID_COMPATIBLE_PATTERNS[l] for l in tc.DIAGNOSIS_LABELS), (
    "a label lost all in-distribution patterns after holding out the OOD compositions"
)


def _find(pattern, kind: str) -> Optional[int]:
    for i, k in enumerate(pattern):
        if k == kind:
            return i
    return None


def _seq_with(problem: Problem, replacements) -> StudentSeq:
    seq = list(problem.answer)
    for i, t in replacements.items():
        seq[i] = t
    return seq


def _text_with(problem: Problem, replacements) -> List[str]:
    text = list(problem.answer_text)
    for i, t in replacements.items():
        text[i] = tc.describe_transform(t)
    return text


def _accept(problem: Problem, seq: StudentSeq, text: List[str], target: str) -> Injection:
    """Return (seq, text) iff it verifies as ``target`` (via both Transforms and text),
    the student image stays in bounds, AND the student's answer visibly differs from the
    original (a net that leaves the shape on the untouched pre-image is a degenerate,
    confusing example, e.g. two rotations that cancel to the identity); else None."""
    if tc.diagnose(problem.original, problem.answer, seq) != target:
        return None
    if tc.diagnose(problem.original, problem.answer_text, text) != target:
        return None
    img = tc.compose(seq).apply(problem.original)
    if not geometry.in_bounds(img, -BOUND, BOUND):
        return None
    if [tuple(p) for p in img] == [tuple(p) for p in problem.original]:
        return None
    return seq, text


def _rand_translations(rng, n: int = 48, max_mag: int = 9):
    out = []
    for _ in range(n):
        dx = rng.randint(-max_mag, max_mag)
        dy = rng.randint(-max_mag, max_mag)
        out.append((dx, dy))
    return out


# --------------------------------------------------------------------------------------
# One injector per target category. Each returns Injection or None.
# --------------------------------------------------------------------------------------

def _inject_correct(problem: Problem, rng) -> Injection:
    seq = list(problem.answer)
    text = list(problem.answer_text)
    ridx = _find(problem.pattern, "rotate")
    if ridx is not None and rng.random() < 0.5:  # optionally re-word the rotation
        style = rng.choice(("ccw", "cw"))
        text[ridx] = tc.describe_transform(problem.answer[ridx], rotation_style=style)
    return _accept(problem, seq, text, "correct")


def _inject_reflection_instead_of_rotation(problem: Problem, rng) -> Injection:
    idx = _find(problem.pattern, "rotate")
    if idx is None:
        return None
    lines = list(REFLECTION_LINES)
    rng.shuffle(lines)
    for line in lines:
        newt = tc.reflect(line)
        seq = _seq_with(problem, {idx: newt})
        text = _text_with(problem, {idx: newt})
        res = _accept(problem, seq, text, "reflection_instead_of_rotation")
        if res:
            return res
    return None


def _inject_rotation_instead_of_reflection(problem: Problem, rng) -> Injection:
    idx = _find(problem.pattern, "reflect")
    if idx is None:
        return None
    degs = list(ROTATION_DEGREES)
    rng.shuffle(degs)
    for deg in degs:
        newt = tc.rotate(deg, "ccw")
        seq = _seq_with(problem, {idx: newt})
        text = _text_with(problem, {idx: newt})
        res = _accept(problem, seq, text, "rotation_instead_of_reflection")
        if res:
            return res
    return None


def _inject_wrong_rotation_angle(problem: Problem, rng) -> Injection:
    idx = _find(problem.pattern, "rotate")
    if idx is None:
        return None
    cur = problem.answer[idx]
    degs = list(ROTATION_DEGREES)
    rng.shuffle(degs)
    for deg in degs:
        newt = tc.rotate(deg, "ccw")
        if newt == cur:
            continue
        seq = _seq_with(problem, {idx: newt})
        text = _text_with(problem, {idx: newt})
        res = _accept(problem, seq, text, "wrong_rotation_angle")
        if res:
            return res
    return None


def _inject_wrong_reflection_line(problem: Problem, rng) -> Injection:
    idx = _find(problem.pattern, "reflect")
    if idx is None:
        return None
    cur = problem.answer[idx]
    lines = list(REFLECTION_LINES)
    rng.shuffle(lines)
    for line in lines:
        newt = tc.reflect(line)
        if newt == cur:
            continue
        seq = _seq_with(problem, {idx: newt})
        text = _text_with(problem, {idx: newt})
        res = _accept(problem, seq, text, "wrong_reflection_line")
        if res:
            return res
    return None


def _inject_wrong_translation(problem: Problem, rng) -> Injection:
    idx = _find(problem.pattern, "translate")
    if idx is None:
        return None
    cdx, cdy = problem.answer[idx].vec
    for dx, dy in _rand_translations(rng):
        if (dx, dy) == (cdx, cdy) or (dx, dy) == (-cdx, -cdy) or (dx, dy) == (0, 0):
            continue
        newt = tc.translate(dx, dy)
        seq = _seq_with(problem, {idx: newt})
        text = _text_with(problem, {idx: newt})
        res = _accept(problem, seq, text, "wrong_translation")
        if res:
            return res
    return None


def _inject_opposite_translation(problem: Problem, rng) -> Injection:
    idx = _find(problem.pattern, "translate")
    if idx is None:
        return None
    cdx, cdy = problem.answer[idx].vec
    if (cdx, cdy) == (0, 0):
        return None
    newt = tc.translate(-cdx, -cdy)
    seq = _seq_with(problem, {idx: newt})
    text = _text_with(problem, {idx: newt})
    return _accept(problem, seq, text, "opposite_translation")


# completely_wrong spans BOTH linear part AND translation wrong. It must not be a
# monoculture: the class covers two flavors, and flavor (b) has two same-orientation
# sub-kinds, so there are three sub-kinds in total —
#   * cross_orientation     : student's net orientation flips (det differs) + wrong slide   [flavor a]
#   * wrong_rotation_angle  : correct net is a rotation, student a DIFFERENT rotation + wrong slide  [flavor b]
#   * wrong_reflection_line : correct net is a reflection, student a DIFFERENT reflection + wrong slide [flavor b]
# `diagnose` remains the sole oracle (verify-or-discard in `_accept`), so an injected
# `completely_wrong` is always exactly that regardless of sub-kind.

_CW_SUBKINDS = ("cross_orientation", "wrong_rotation_angle", "wrong_reflection_line")


def _pattern_net_is_rotation(pat: Sequence[str]) -> bool:
    """A composed pattern's net orientation is a rotation iff it has an even number of
    reflections (rotate/translate are orientation-preserving, reflect flips)."""
    return sum(1 for k in pat if k == "reflect") % 2 == 0


# Sub-kind selection weights = inverse of each sub-kind's realizability, so the three
# sub-kinds come out roughly even despite cross_orientation being realizable on every
# problem while each flavor-(b) sub-kind needs a matching net orientation. Derived from
# PATTERNS (not hardcoded) so it stays correct if the pattern set changes.
_P_ROT = sum(_pattern_net_is_rotation(p) for p in PATTERNS) / len(PATTERNS)
_P_REFL = 1.0 - _P_ROT
_CW_WEIGHTS = [
    1.0,                                  # cross_orientation: realizable on every problem
    (1.0 / _P_ROT) if _P_ROT else 0.0,    # wrong_rotation_angle: needs a rotation net
    (1.0 / _P_REFL) if _P_REFL else 0.0,  # wrong_reflection_line: needs a reflection net
]


def _cw_try(problem: Problem, rng, linears, correct_vec) -> Injection:
    """Try each (wrong linear part, wrong translation) pair; return the first that the
    diagnose oracle confirms as ``completely_wrong`` and whose image stays in bounds."""
    linears = list(linears)
    rng.shuffle(linears)
    trans = _rand_translations(rng)
    for lin in linears:
        for dx, dy in trans:
            if (dx, dy) == correct_vec:  # translation must also be wrong
                continue
            newt = tc.translate(dx, dy)
            seq = [lin, newt]
            text = [tc.describe_transform(lin), tc.describe_transform(newt)]
            res = _accept(problem, seq, text, "completely_wrong")
            if res:
                return res
    return None


def _cw_cross_orientation(problem: Problem, rng) -> Injection:
    net = problem.net()
    if net.det() == 1:  # correct net is a rotation -> student uses a reflection
        linears = [tc.reflect(l) for l in REFLECTION_LINES]
    else:               # correct net is a reflection -> student uses a rotation
        linears = [tc.rotate(d, "ccw") for d in ROTATION_DEGREES]
    return _cw_try(problem, rng, linears, net.vec)


def _cw_wrong_rotation_angle(problem: Problem, rng) -> Injection:
    net = problem.net()
    if net.det() != 1:  # only realizable when the correct net is a rotation
        return None
    linears = [tc.rotate(d, "ccw") for d in ROTATION_DEGREES
               if tc.rotate(d, "ccw").matrix != net.matrix]
    return _cw_try(problem, rng, linears, net.vec)


def _cw_wrong_reflection_line(problem: Problem, rng) -> Injection:
    net = problem.net()
    if net.det() != -1:  # only realizable when the correct net is a reflection
        return None
    linears = [tc.reflect(l) for l in REFLECTION_LINES
               if tc.reflect(l).matrix != net.matrix]
    return _cw_try(problem, rng, linears, net.vec)


_CW_INJECTORS = {
    "cross_orientation": _cw_cross_orientation,
    "wrong_rotation_angle": _cw_wrong_rotation_angle,
    "wrong_reflection_line": _cw_wrong_reflection_line,
}


def _inject_completely_wrong(problem: Problem, rng) -> Injection:
    # Target one sub-kind (weighted so all three end up roughly even); if it is not
    # realizable for this problem's net orientation the injector returns None and the
    # caller retries with a fresh problem.
    kind = rng.choices(_CW_SUBKINDS, weights=_CW_WEIGHTS, k=1)[0]
    return _CW_INJECTORS[kind](problem, rng)


_INJECTORS = {
    "correct": _inject_correct,
    "reflection_instead_of_rotation": _inject_reflection_instead_of_rotation,
    "rotation_instead_of_reflection": _inject_rotation_instead_of_reflection,
    "wrong_rotation_angle": _inject_wrong_rotation_angle,
    "wrong_reflection_line": _inject_wrong_reflection_line,
    "wrong_translation": _inject_wrong_translation,
    "opposite_translation": _inject_opposite_translation,
    "completely_wrong": _inject_completely_wrong,
}


def inject(problem: Problem, target_label: str, rng) -> Injection:
    """Produce a verified student attempt for ``target_label`` on ``problem``.

    Returns ``(student_seq, student_text)`` where ``student_seq`` is a list of
    ``transform_core.Transform`` and ``student_text`` the matching schema strings, or
    ``None`` if this problem cannot realize the target (caller should retry with a fresh
    problem). The returned attempt is guaranteed to satisfy
    ``diagnose(problem.original, problem.answer, student_seq) == target_label``.
    """
    if target_label not in _INJECTORS:
        raise ValueError(f"unknown target label: {target_label!r}")
    return _INJECTORS[target_label](problem, rng)
