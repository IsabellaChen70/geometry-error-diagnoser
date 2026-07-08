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


def make_problem(
    rng,
    pattern: Optional[Sequence[str]] = None,
    num_vertices: Optional[int] = None,
    *,
    max_shape_tries: int = 200,
    max_op_tries: int = 200,
) -> Problem:
    """Create a valid :class:`Problem`.

    Guarantees: ``original`` and ``image`` are integer lattice polygons fully inside
    ``[-BOUND, BOUND]^2``; ``original`` is simple and ``transform_core.is_asymmetric``
    (so the correct net map is unique); ``image != original``. ``pattern`` (an ordered
    pair of move kinds) and ``num_vertices`` may be forced; otherwise sampled.
    """
    for _ in range(max_shape_tries):
        nv = num_vertices if num_vertices is not None else rng.choice(VERTEX_CHOICES)
        original = geometry.generate_irregular_polygon(
            nv, irregularity=0.5, spikiness=0.35, radius=4.0, rng=rng,
            snap=True, require_asymmetric=True,
        )
        pat = tuple(pattern) if pattern is not None else rng.choice(PATTERNS)

        for _ in range(max_op_tries):
            steps = [_make_step(k, rng) for k in pat]
            answer = [t for t, _ in steps]
            answer_text = [txt for _, txt in steps]
            image = tc.compose(answer).apply(original)
            if not geometry.in_bounds(image, -BOUND, BOUND):
                continue
            if image == original:
                continue
            return Problem(
                original=list(original),
                image=list(image),
                answer=answer,
                answer_text=answer_text,
                pattern=pat,
                num_vertices=nv,
            )
    raise RuntimeError(f"could not build a problem for pattern={pattern} nv={num_vertices}")
