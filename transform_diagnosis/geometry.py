"""geometry — random SINGLE irregular polygons on the integer lattice.

Shapes are weird quads / pentagons / hexagons with uneven sides and non-uniform angles.
They are NEVER unions of multiple shapes. Construction uses the angular-sweep method
(guarantees a simple polygon) with floats used ONLY while shaping; every returned vertex
is an integer lattice point. The chirality / symmetry test is delegated to
``transform_core.is_asymmetric`` so there is a single source of truth for symmetry.

Adapted from the geometry helpers in the owner's earlier ``model/generate_diagnosis_data.py``
(angular sweep, simple-polygon test, integer snapping/validation) but routed through
``transform_core`` for all symmetry logic.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from . import transform_core as tc

Point = Tuple[int, int]

DEFAULT_BOUND = 10  # vertices must satisfy |x|, |y| <= DEFAULT_BOUND


# --------------------------------------------------------------------------------------
# Exact integer geometry primitives
# --------------------------------------------------------------------------------------

def signed_area2(pts: Sequence[Sequence[int]]) -> int:
    """Twice the signed area of a polygon (exact integer). Positive => CCW winding."""
    s = 0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s


def orientation(pts: Sequence[Sequence[int]]) -> int:
    """+1 for CCW, -1 for CW."""
    return 1 if signed_area2(pts) > 0 else -1


def _seg_intersect(a, b, c, d) -> bool:
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = ccw(c, d, a)
    d2 = ccw(c, d, b)
    d3 = ccw(a, b, c)
    d4 = ccw(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def is_simple(pts: Sequence[Sequence[int]]) -> bool:
    """True iff the polygon has no non-adjacent edge crossings (non-self-intersecting)."""
    n = len(pts)
    if n < 3:
        return False
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _seg_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return False
    return True


def _no_collinear(pts: Sequence[Sequence[int]]) -> bool:
    n = len(pts)
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross == 0:
            return False
    return True


def _min_edge2(pts: Sequence[Sequence[int]]) -> int:
    """Minimum squared edge length (exact integer)."""
    n = len(pts)
    best = None
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        d2 = (x1 - x2) ** 2 + (y1 - y2) ** 2
        best = d2 if best is None else min(best, d2)
    return best if best is not None else 0


# --------------------------------------------------------------------------------------
# Bounding box / placement helpers
# --------------------------------------------------------------------------------------

def bbox(pts: Sequence[Sequence[int]]) -> Tuple[int, int, int, int]:
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return min(xs), min(ys), max(xs), max(ys)


def in_bounds(pts: Sequence[Sequence[int]], lo: int = -DEFAULT_BOUND, hi: int = DEFAULT_BOUND) -> bool:
    return all(lo <= x <= hi and lo <= y <= hi for x, y in pts)


def translated(pts: Sequence[Sequence[int]], dx: int, dy: int) -> List[Point]:
    return [(x + dx, y + dy) for x, y in pts]


def random_offset_in_bounds(
    pts: Sequence[Sequence[int]], rng, lo: int = -DEFAULT_BOUND, hi: int = DEFAULT_BOUND
) -> Optional[List[Point]]:
    """Randomly translate ``pts`` so it stays entirely within ``[lo, hi]^2``; None if it
    cannot fit."""
    mnx, mny, mxx, mxy = bbox(pts)
    lo_dx, hi_dx = lo - mnx, hi - mxx
    lo_dy, hi_dy = lo - mny, hi - mxy
    if lo_dx > hi_dx or lo_dy > hi_dy:
        return None
    dx = rng.randint(lo_dx, hi_dx)
    dy = rng.randint(lo_dy, hi_dy)
    return translated(pts, dx, dy)


# --------------------------------------------------------------------------------------
# Irregular polygon generation
# --------------------------------------------------------------------------------------

def _clip(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _raw_polygon(num_vertices, irregularity, spikiness, radius, rng, center=(0.0, 0.0)):
    """One raw float polygon via angular sweep (simple by construction)."""
    irr = _clip(irregularity, 0.0, 1.0) * (2 * math.pi / num_vertices)
    spk = _clip(spikiness, 0.0, 1.0) * radius
    lower = (2 * math.pi / num_vertices) - irr
    upper = (2 * math.pi / num_vertices) + irr
    steps = [rng.uniform(lower, upper) for _ in range(num_vertices)]
    scale = (2 * math.pi) / sum(steps)
    steps = [s * scale for s in steps]

    pts = []
    angle = rng.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        r = _clip(rng.gauss(radius, spk), 0.2 * radius, 1.8 * radius)
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        pts.append((x, y))
        angle += steps[i]
    return pts


def generate_irregular_polygon(
    num_vertices: int,
    irregularity: float,
    spikiness: float,
    radius: float,
    rng,
    snap: bool = True,
    require_asymmetric: bool = True,
    *,
    max_coord: int = 7,
    min_edge: float = 1.5,
    min_area2: int = 6,
    max_tries: int = 4000,
):
    """Generate a single irregular polygon.

    Positional contract (do not reorder): ``num_vertices, irregularity, spikiness,
    radius, rng, snap, require_asymmetric``. Extra keyword-only knobs tune validation.

    When ``snap`` is True (the dataset path) the result is an integer-lattice polygon,
    validated to be: exactly ``num_vertices`` distinct vertices, contained in
    ``[-max_coord, max_coord]^2``, simple (non-self-intersecting), free of collinear
    consecutive triples, with every edge at least ``min_edge`` long and area at least
    ``min_area2 / 2``. When ``require_asymmetric`` is True the shape must additionally be
    ``transform_core.is_asymmetric`` (trivial lattice symmetry) so the correct transform
    is uniquely recoverable. Vertices are returned in CCW order.

    When ``snap`` is False a single raw float polygon is returned (used for shaping only).
    """
    if snap is False:
        return _raw_polygon(num_vertices, irregularity, spikiness, radius, rng)

    r = float(radius)
    for attempt in range(max_tries):
        if attempt and attempt % 80 == 0:  # gently shrink if the box is hard to hit
            r *= 0.95
        raw = _raw_polygon(num_vertices, irregularity, spikiness, r, rng)
        pts = [(round(x), round(y)) for x, y in raw]
        if len({(x, y) for x, y in pts}) != num_vertices:
            continue
        if max(max(abs(x), abs(y)) for x, y in pts) > max_coord:
            continue
        if not is_simple(pts):
            continue
        if not _no_collinear(pts):
            continue
        if _min_edge2(pts) < min_edge * min_edge:
            continue
        if abs(signed_area2(pts)) < min_area2:
            continue
        if require_asymmetric and not tc.is_asymmetric(pts):
            continue
        if orientation(pts) < 0:  # canonical CCW winding
            pts = list(reversed(pts))
        return pts
    raise RuntimeError(
        f"could not generate a valid {num_vertices}-gon after {max_tries} tries; "
        "loosen constraints"
    )
