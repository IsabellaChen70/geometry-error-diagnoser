"""problems — build a well-formed composed-transformation problem.

A problem is an irregular polygon (the ``original``) plus a CORRECT transformation that
is a composition of exactly two rigid moves (rotation / reflection / translation), the
resulting ``image``, and the human-readable answer text.

All transform math goes through :mod:`transform_core`; this module only samples
parameters and validates bounds / uniqueness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import geometry, transform_core as tc

Point = Tuple[int, int]

# The six ordered two-move patterns (two DISTINCT move types, order matters).
PATTERNS: Tuple[Tuple[str, str], ...] = (
    ("rotate", "translate"),
    ("reflect", "translate"),
    ("translate", "rotate"),
    ("translate", "reflect"),
    ("rotate", "reflect"),
    ("reflect", "rotate"),
)

VERTEX_CHOICES = (4, 5, 6)
ROTATION_DEGREES = (90, 180, 270)
REFLECTION_LINES = ("x", "y", "y=x", "y=-x")
BOUND = geometry.DEFAULT_BOUND  # 10


@dataclass
class Problem:
    original: List[Point]
    image: List[Point]
    answer: List[tc.Transform]           # correct sequence, seq[0] applied first
    answer_text: List[str]               # schema strings for `answer`
    pattern: Tuple[str, ...] = field(default_factory=tuple)
    num_vertices: int = 0

    def net(self) -> tc.Transform:
        return tc.compose(self.answer)


def _sample_translation(rng, max_mag: int = 8) -> Tuple[int, int]:
    """Nonzero integer translation; ~60% axis-aligned (for '7 left' style wording)."""
    if rng.random() < 0.6:
        axis = rng.choice(("x", "y"))
        mag = rng.randint(1, max_mag)
        sign = rng.choice((-1, 1))
        return (sign * mag, 0) if axis == "x" else (0, sign * mag)
    while True:
        dx = rng.randint(-max_mag, max_mag)
        dy = rng.randint(-max_mag, max_mag)
        if (dx, dy) != (0, 0):
            return (dx, dy)


def _make_step(kind: str, rng) -> Tuple[tc.Transform, str]:
    """Build one primitive Transform plus its schema text (with sampled wording)."""
    if kind == "rotate":
        deg = rng.choice(ROTATION_DEGREES)
        t = tc.rotate(deg, "ccw")
        style = "cw" if (deg != 180 and rng.random() < 0.5) else "ccw"
        return t, tc.describe_transform(t, rotation_style=style)
    if kind == "reflect":
        line = rng.choice(REFLECTION_LINES)
        t = tc.reflect(line)
        return t, tc.describe_transform(t)
    if kind == "translate":
        dx, dy = _sample_translation(rng)
        t = tc.translate(dx, dy)
        return t, tc.describe_transform(t)
    raise ValueError(kind)


def _validate_original(pts: Sequence[Point]) -> List[Point]:
    """Coerce + validate a forced pre-image: integer lattice, in-bounds, simple, asymmetric.

    A forced ``original`` (used by the contrastive-group generator to share ONE RED
    pre-image across matched records) must satisfy the same invariants an internally
    generated pre-image does, so the correct net map stays uniquely recoverable.
    """
    base = tc.as_points(pts)
    if not geometry.in_bounds(base, -BOUND, BOUND):
        raise ValueError("forced original is out of bounds")
    if not geometry.is_simple(base):
        raise ValueError("forced original is not a simple polygon")
    if not tc.is_asymmetric(base):
        raise ValueError("forced original is not asymmetric (net map would be ambiguous)")
    return list(base)


def make_problem(
    rng,
    pattern: Optional[Sequence[str]] = None,
    num_vertices: Optional[int] = None,
    *,
    original: Optional[Sequence[Point]] = None,
    max_shape_tries: int = 200,
    max_op_tries: int = 200,
) -> Problem:
    """Create a valid :class:`Problem`.

    Guarantees: ``original`` and ``image`` are integer lattice polygons fully inside
    ``[-BOUND, BOUND]^2``; ``original`` is simple and ``transform_core.is_asymmetric``
    (so the correct net map is unique); ``image != original``. ``pattern`` (an ordered
    tuple of move kinds, of ANY length -- e.g. ``("rotate",)`` for a single-step
    curriculum problem) and ``num_vertices`` may be forced; otherwise sampled.

    ``original`` may be forced too: when given, that exact (validated) pre-image is reused
    instead of generating a fresh one, so several problems can share one RED shape (the
    contrastive-group generator relies on this). Only the correct transform is then
    sampled/retried.
    """
    forced_original = _validate_original(original) if original is not None else None
    for _ in range(max_shape_tries):
        if forced_original is not None:
            base = list(forced_original)
            nv = len(base)
        else:
            nv = num_vertices if num_vertices is not None else rng.choice(VERTEX_CHOICES)
            base = geometry.generate_irregular_polygon(
                nv, irregularity=0.5, spikiness=0.35, radius=4.0, rng=rng,
                snap=True, require_asymmetric=True,
            )
        pat = tuple(pattern) if pattern is not None else rng.choice(PATTERNS)

        for _ in range(max_op_tries):
            steps = [_make_step(k, rng) for k in pat]
            answer = [t for t, _ in steps]
            answer_text = [txt for _, txt in steps]
            image = tc.compose(answer).apply(base)
            if not geometry.in_bounds(image, -BOUND, BOUND):
                continue
            if image == base:
                continue
            return Problem(
                original=list(base),
                image=list(image),
                answer=answer,
                answer_text=answer_text,
                pattern=pat,
                num_vertices=nv,
            )
    raise RuntimeError(f"could not build a problem for pattern={pattern} nv={num_vertices}")
