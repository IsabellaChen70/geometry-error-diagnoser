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
    """Return (seq, text) iff it verifies as ``target`` (via both Transforms and text)
    and the student image stays in bounds; else None."""
    if tc.diagnose(problem.original, problem.answer, seq) != target:
        return None
    if tc.diagnose(problem.original, problem.answer_text, text) != target:
        return None
    img = tc.compose(seq).apply(problem.original)
    if not geometry.in_bounds(img, -BOUND, BOUND):
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


def _inject_completely_wrong(problem: Problem, rng) -> Injection:
    net = problem.net()
    tc_vec = net.vec
    if net.det() == 1:  # correct net is a rotation -> student uses a reflection
        linears = [tc.reflect(l) for l in REFLECTION_LINES]
    else:                # correct net is a reflection -> student uses a rotation
        linears = [tc.rotate(d, "ccw") for d in ROTATION_DEGREES]
    rng.shuffle(linears)
    trans = _rand_translations(rng)
    for lin in linears:
        for dx, dy in trans:
            if (dx, dy) == tc_vec:
                continue
            seq = [lin, tc.translate(dx, dy)]
            text = [tc.describe_transform(lin), tc.describe_transform(tc.translate(dx, dy))]
            res = _accept(problem, seq, text, "completely_wrong")
            if res:
                return res
    return None


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
