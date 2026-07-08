"""
transform_core — THE single canonical implementation of rigid-motion transforms,
grading, and student-error diagnosis for the composed-transformation dataset.

Everything else in the package imports from here. There is exactly ONE implementation
of transforms / grade / diagnose / recover_map in the whole package. If judging
behaviour must change, change THIS file and its contract test (`test_transform_core.py`)
only — never fork it.

Design
------
A transform is an affine map on the integer lattice::

    p -> M @ p + t

where ``M`` is an integer 2x2 orthogonal matrix ``((a, b), (c, d))`` (row-major) and
``t = (e, f)`` is an integer translation vector. All arithmetic is exact integer
arithmetic; no floats ever appear in the transform / grade / label path.

``det(M)`` is the orientation:

* ``+1`` — orientation preserving (rotation, including the identity)
* ``-1`` — orientation reversing (reflection)

Composition
-----------
``compose(seq)`` composes a sequence where ``seq[0]`` is applied FIRST. For an affine
map, applying ``T1`` then ``T2`` gives ``p -> M2 @ (M1 @ p + t1) + t2``, i.e.
``M = M2 @ M1`` and ``t = M2 @ t1 + t2``.

Text forms
----------
Every transform has a human-readable string form (see ``describe_transform``) matching
the dataset schema, e.g. ``"rotate 90 degrees counterclockwise"``, ``"translate 7 left"``,
``"reflect across x axis"``. ``parse_transform`` is the inverse. ``compose`` / ``grade`` /
``diagnose`` accept either ``Transform`` objects or these strings (strings are parsed),
so the dataset can be verified directly from its stored text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Union

Matrix = Tuple[Tuple[int, int], Tuple[int, int]]
Vec = Tuple[int, int]
Point = Tuple[int, int]

# --------------------------------------------------------------------------------------
# Closed label set for diagnosis (order is stable and part of the public contract).
# --------------------------------------------------------------------------------------

DIAGNOSIS_LABELS: List[str] = [
    "correct",
    "reflection_instead_of_rotation",
    "rotation_instead_of_reflection",
    "wrong_rotation_angle",
    "wrong_reflection_line",
    "wrong_translation",
    "opposite_translation",
    "completely_wrong",
]

# det(M) sign -> orientation class name.
ORIENTATIONS = {1: "rotation", -1: "reflection"}

IDENTITY_MATRIX: Matrix = ((1, 0), (0, 1))

# Canonical CCW rotation matrices (about the origin).
_ROTATION_MATRICES = {
    0: ((1, 0), (0, 1)),
    90: ((0, -1), (1, 0)),
    180: ((-1, 0), (0, -1)),
    270: ((0, 1), (-1, 0)),
}
_ROTATION_MATRIX_TO_DEG = {m: d for d, m in _ROTATION_MATRICES.items()}

# Reflection matrices keyed by canonical line name.
_REFLECTION_MATRICES = {
    "x": ((1, 0), (0, -1)),
    "y": ((-1, 0), (0, 1)),
    "y=x": ((0, 1), (1, 0)),
    "y=-x": ((0, -1), (-1, 0)),
}
_REFLECTION_MATRIX_TO_LINE = {m: k for k, m in _REFLECTION_MATRICES.items()}

# Human-readable line phrasing used in schema strings.
_LINE_TEXT = {
    "x": "x axis",
    "y": "y axis",
    "y=x": "line y = x",
    "y=-x": "line y = -x",
}

# All 8 lattice isometry linear parts (identity + 3 rotations + 4 reflections) and the 7
# non-identity ones (used by is_asymmetric / recover_map).
ALL_LINEAR_MAPS: Tuple[Matrix, ...] = (
    _ROTATION_MATRICES[0],
    _ROTATION_MATRICES[90],
    _ROTATION_MATRICES[180],
    _ROTATION_MATRICES[270],
    _REFLECTION_MATRICES["x"],
    _REFLECTION_MATRICES["y"],
    _REFLECTION_MATRICES["y=x"],
    _REFLECTION_MATRICES["y=-x"],
)
_NONTRIVIAL_LINEAR_MAPS: Tuple[Matrix, ...] = ALL_LINEAR_MAPS[1:]


# --------------------------------------------------------------------------------------
# Transform dataclass
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Transform:
    """An integer affine map ``p -> matrix @ p + vec``.

    Equality is structural (frozen dataclass) and depends ONLY on the math
    (``matrix`` and ``vec``) — never on any text wording — so net-map comparisons
    are exact.
    """

    matrix: Matrix = IDENTITY_MATRIX
    vec: Vec = (0, 0)

    def apply(self, pts: Iterable[Sequence[int]]) -> List[Point]:
        """Apply this transform to an iterable of ``(x, y)`` points."""
        (a, b), (c, d) = self.matrix
        e, f = self.vec
        return [(a * x + b * y + e, c * x + d * y + f) for x, y in pts]

    def det(self) -> int:  # noqa: D401 - short and exact
        """Determinant of the linear part (``+1`` rotation/identity, ``-1`` reflection)."""
        return matrix_det(self.matrix)

    @property
    def orientation(self) -> str:
        return ORIENTATIONS[self.det()]

    # -- matrix helpers exposed on the Transform contract -------------------------------

    @staticmethod
    def mat_mul(a: Matrix, b: Matrix) -> Matrix:
        return mat_mul(a, b)

    @staticmethod
    def mat_vec(m: Matrix, v: Sequence[int]) -> Vec:
        return mat_vec(m, v)

    @staticmethod
    def det_of(m: Matrix) -> int:
        return matrix_det(m)


# --------------------------------------------------------------------------------------
# Standalone matrix helpers (also exposed as Transform.mat_mul / mat_vec / det_of)
# --------------------------------------------------------------------------------------

def mat_mul(a: Matrix, b: Matrix) -> Matrix:
    """2x2 integer matrix product ``a @ b`` (row-major)."""
    (a00, a01), (a10, a11) = a
    (b00, b01), (b10, b11) = b
    return (
        (a00 * b00 + a01 * b10, a00 * b01 + a01 * b11),
        (a10 * b00 + a11 * b10, a10 * b01 + a11 * b11),
    )


def mat_vec(m: Matrix, v: Sequence[int]) -> Vec:
    """Apply a 2x2 matrix to a length-2 vector."""
    (a, b), (c, d) = m
    x, y = v
    return (a * x + b * y, c * x + d * y)


def matrix_det(m: Matrix) -> int:
    (a, b), (c, d) = m
    return a * d - b * c


# Convenience alias so callers can write ``transform_core.det(M)``.
det = matrix_det


# --------------------------------------------------------------------------------------
# Factories
# --------------------------------------------------------------------------------------

def identity() -> Transform:
    return Transform(IDENTITY_MATRIX, (0, 0))


def _normalize_direction(direction: str) -> str:
    d = direction.strip().lower()
    if d in ("ccw", "counterclockwise", "counter-clockwise", "anticlockwise"):
        return "ccw"
    if d in ("cw", "clockwise"):
        return "cw"
    raise ValueError(f"unknown rotation direction: {direction!r}")


def rotate(degrees: int, direction: str = "ccw") -> Transform:
    """Rotation about the origin by ``degrees`` (90/180/270) clockwise or ccw."""
    deg = int(degrees) % 360
    if deg not in (0, 90, 180, 270):
        raise ValueError(f"rotation degrees must be a multiple of 90, got {degrees!r}")
    if _normalize_direction(direction) == "cw":
        deg = (360 - deg) % 360
    return Transform(_ROTATION_MATRICES[deg], (0, 0))


def _normalize_line(line: str) -> str:
    s = line.strip().lower().replace(" ", "")
    aliases = {
        "x": "x", "xaxis": "x", "x-axis": "x", "thex-axis": "x", "thexaxis": "x",
        "y": "y", "yaxis": "y", "y-axis": "y", "they-axis": "y", "theyaxis": "y",
        "y=x": "y=x", "liney=x": "y=x", "theliney=x": "y=x",
        "y=-x": "y=-x", "liney=-x": "y=-x", "theliney=-x": "y=-x",
    }
    if s in aliases:
        return aliases[s]
    raise ValueError(f"unknown reflection line: {line!r}")


def reflect(line: str) -> Transform:
    """Reflection across one of: ``x``, ``y``, ``y=x``, ``y=-x`` (aliases accepted)."""
    return Transform(_REFLECTION_MATRICES[_normalize_line(line)], (0, 0))


def translate(dx: int, dy: int) -> Transform:
    return Transform(IDENTITY_MATRIX, (int(dx), int(dy)))


TransformLike = Union[Transform, str]


def as_transform(t: TransformLike) -> Transform:
    """Coerce a ``Transform`` or a schema string into a ``Transform``."""
    if isinstance(t, Transform):
        return t
    if isinstance(t, str):
        return parse_transform(t)
    raise TypeError(f"expected Transform or str, got {type(t).__name__}")


def compose(seq: Sequence[TransformLike]) -> Transform:
    """Compose a sequence of transforms where ``seq[0]`` is applied FIRST.

    Accepts ``Transform`` objects or schema strings (strings are parsed).
    An empty sequence composes to the identity.
    """
    result = identity()
    for item in seq:
        t = as_transform(item)
        m = mat_mul(t.matrix, result.matrix)
        v_lin = mat_vec(t.matrix, result.vec)
        v = (v_lin[0] + t.vec[0], v_lin[1] + t.vec[1])
        result = Transform(m, v)
    return result


# --------------------------------------------------------------------------------------
# Text <-> Transform
# --------------------------------------------------------------------------------------

def describe_transform(t: TransformLike, rotation_style: str = "ccw") -> str:
    """Render a single-step transform as its canonical schema string.

    ``rotation_style`` controls rotation wording only: ``"ccw"`` (default) or ``"cw"``
    (the equivalent complementary clockwise wording, e.g. ``rotate 270 degrees
    clockwise`` for a 90-degree ccw rotation). Non-rotations ignore it.
    """
    t = as_transform(t)
    if t.matrix == IDENTITY_MATRIX:
        dx, dy = t.vec
        if dx == 0 and dy == 0:
            return "identity"
        if dy == 0:
            return f"translate {abs(dx)} {'right' if dx > 0 else 'left'}"
        if dx == 0:
            return f"translate {abs(dy)} {'up' if dy > 0 else 'down'}"
        return f"translate by ({dx}, {dy})"
    if t.vec == (0, 0) and t.matrix in _ROTATION_MATRIX_TO_DEG:
        deg = _ROTATION_MATRIX_TO_DEG[t.matrix]
        style = _normalize_direction(rotation_style)
        if style == "cw":
            cw = (360 - deg) % 360
            return f"rotate {cw} degrees clockwise"
        return f"rotate {deg} degrees counterclockwise"
    if t.vec == (0, 0) and t.matrix in _REFLECTION_MATRIX_TO_LINE:
        line = _REFLECTION_MATRIX_TO_LINE[t.matrix]
        return f"reflect across {_LINE_TEXT[line]}"
    raise ValueError(f"cannot describe non-primitive transform: {t!r}")


_ROT_RE = re.compile(r"rotate\s+(\d+)\s*degrees?\s+(counterclockwise|clockwise|ccw|cw)")
_REFL_RE = re.compile(r"reflect\s+across\s+(.*)")
_TRANS_XY_RE = re.compile(r"translate\s+by\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")
_TRANS_DIR_RE = re.compile(r"translate\s+(\d+)\s+(left|right|up|down)")

_DIR_TO_VEC = {
    "left": (-1, 0),
    "right": (1, 0),
    "up": (0, 1),
    "down": (0, -1),
}


def parse_transform(text: str) -> Transform:
    """Parse a schema string back into a ``Transform`` (inverse of describe_transform)."""
    s = " ".join(text.strip().lower().split())
    m = _ROT_RE.search(s)
    if m:
        return rotate(int(m.group(1)), m.group(2))
    m = _TRANS_XY_RE.search(s)
    if m:
        return translate(int(m.group(1)), int(m.group(2)))
    m = _TRANS_DIR_RE.search(s)
    if m:
        n = int(m.group(1))
        ux, uy = _DIR_TO_VEC[m.group(2)]
        return translate(ux * n, uy * n)
    m = _REFL_RE.search(s)
    if m:
        return reflect(m.group(1))
    if s == "identity":
        return identity()
    raise ValueError(f"cannot parse transform text: {text!r}")


def describe_seq(seq: Sequence[TransformLike]) -> List[str]:
    return [describe_transform(t) for t in seq]


# --------------------------------------------------------------------------------------
# Point helpers
# --------------------------------------------------------------------------------------

def as_points(pts: Iterable[Sequence[int]]) -> List[Point]:
    """Normalize any list/tuple of coordinate pairs into a list of int tuples."""
    return [(int(x), int(y)) for x, y in pts]


# --------------------------------------------------------------------------------------
# Grading
# --------------------------------------------------------------------------------------

def grade(
    original: Iterable[Sequence[int]],
    image: Iterable[Sequence[int]],
    candidate_seq: Sequence[TransformLike],
) -> bool:
    """Return True iff applying ``candidate_seq`` (seq[0] first) to ``original`` yields ``image``."""
    produced = compose(candidate_seq).apply(as_points(original))
    return produced == as_points(image)


# --------------------------------------------------------------------------------------
# Diagnosis
# --------------------------------------------------------------------------------------

def diagnose(
    original: Iterable[Sequence[int]],
    correct_seq: Sequence[TransformLike],
    student_seq: Sequence[TransformLike],
) -> str:
    """Classify a student attempt against the correct answer.

    The classification is a total, deterministic function of the two NET affine maps
    (``original`` is accepted for API symmetry but not required — the net maps carry
    all the information). Let ``C = compose(correct_seq)``, ``S = compose(student_seq)``
    with net linear parts ``Mc, Ms`` and net translations ``tc, ts``:

    1. ``Mc == Ms and tc == ts``                      -> ``correct``
    2. ``Mc == Ms`` (linear equal, translation differs):
         * ``tc != 0 and ts == -tc``                  -> ``opposite_translation``
         * otherwise                                  -> ``wrong_translation``
    3. linear parts differ, orientations differ (``det`` sign differs):
         * translations equal, correct rotation/student reflection -> ``reflection_instead_of_rotation``
         * translations equal, correct reflection/student rotation -> ``rotation_instead_of_reflection``
         * translations also differ                   -> ``completely_wrong``
    4. linear parts differ, orientations equal:
         * both rotations, translations equal         -> ``wrong_rotation_angle``
         * both reflections, translations equal       -> ``wrong_reflection_line``
         * translations also differ                   -> ``completely_wrong``

    The result is always a member of ``DIAGNOSIS_LABELS``.
    """
    c = compose(correct_seq)
    s = compose(student_seq)
    mc, tc = c.matrix, c.vec
    ms, ts = s.matrix, s.vec
    lin_same = mc == ms
    tr_same = tc == ts

    if lin_same and tr_same:
        return "correct"

    if lin_same:  # translation differs
        if tc != (0, 0) and ts == (-tc[0], -tc[1]):
            return "opposite_translation"
        return "wrong_translation"

    # Linear parts differ from here on.
    dc, ds = matrix_det(mc), matrix_det(ms)
    if dc != ds:  # orientation confusion
        if not tr_same:
            return "completely_wrong"
        if dc == 1 and ds == -1:
            return "reflection_instead_of_rotation"
        return "rotation_instead_of_reflection"

    # Same orientation, different linear part.
    if not tr_same:
        return "completely_wrong"
    if dc == 1:
        return "wrong_rotation_angle"
    return "wrong_reflection_line"


# --------------------------------------------------------------------------------------
# Symmetry / identifiability
# --------------------------------------------------------------------------------------

def _canonical_pointset(pts: Sequence[Sequence[int]]) -> Tuple[Point, ...]:
    """Translate a point set so its min corner is at the origin, then sort — a
    translation-invariant canonical form of the (unordered) point set."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    mnx, mny = min(xs), min(ys)
    return tuple(sorted((int(x) - mnx, int(y) - mny) for x, y in pts))


def is_asymmetric(pts: Sequence[Sequence[int]]) -> bool:
    """True iff ``pts`` has trivial symmetry under the 8 lattice isometries modulo
    translation. An asymmetric shape guarantees that the net map taking it to any
    image is uniquely recoverable (see ``recover_map``)."""
    base = _canonical_pointset(pts)
    pts_t = as_points(pts)
    for m in _NONTRIVIAL_LINEAR_MAPS:
        mapped = [mat_vec(m, p) for p in pts_t]
        if _canonical_pointset(mapped) == base:
            return False
    return True


def recover_map(
    original: Iterable[Sequence[int]],
    image: Iterable[Sequence[int]],
) -> Optional[Transform]:
    """Recover the unique net ``Transform`` with ``T.apply(original) == image``.

    Searches the 8 lattice isometry linear parts; for each, the translation is fixed
    by the first vertex and then verified against every vertex. Returns the matching
    ``Transform`` or ``None`` if no lattice isometry maps ``original`` onto ``image``.
    For an asymmetric ``original`` the answer (if any) is unique.
    """
    orig = as_points(original)
    img = as_points(image)
    if not orig or len(orig) != len(img):
        return None
    for m in ALL_LINEAR_MAPS:
        mx, my = mat_vec(m, orig[0])
        t = (img[0][0] - mx, img[0][1] - my)
        ok = True
        for p, q in zip(orig, img):
            px, py = mat_vec(m, p)
            if (px + t[0], py + t[1]) != q:
                ok = False
                break
        if ok:
            return Transform(m, t)
    return None
