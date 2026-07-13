"""Canonical JSON representation of a *net* lattice affine transform.

The rendered RED/GREEN/BLUE polygons identify a composed affine map, not a
particular ordered decomposition into primitive steps.  v6 therefore uses one
lossless object for the observable map::

    {"linear": "rot_ccw_90", "tx": 2, "ty": -3}

``linear`` names exactly one of the eight D4 integer orthogonal matrices used by
:mod:`transform_diagnosis.transform_core`.  The matrix table below is built
through that module's public factories; this module does not implement or
duplicate any geometry.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

from . import transform_core as tc

NetTransform = Dict[str, object]

# Stable v6 vocabulary.  Names explicitly state rotation direction and reflection
# line, avoiding overloaded words such as "horizontal reflection".
D4_LINEAR_NAMES: Tuple[str, ...] = (
    "identity",
    "rot_ccw_90",
    "rot_180",
    "rot_ccw_270",
    "reflect_x_axis",
    "reflect_y_axis",
    "reflect_y_eq_x",
    "reflect_y_eq_neg_x",
)

LINEAR_TO_MATRIX = {
    "identity": tc.identity().matrix,
    "rot_ccw_90": tc.rotate(90, "ccw").matrix,
    "rot_180": tc.rotate(180, "ccw").matrix,
    "rot_ccw_270": tc.rotate(270, "ccw").matrix,
    "reflect_x_axis": tc.reflect("x").matrix,
    "reflect_y_axis": tc.reflect("y").matrix,
    "reflect_y_eq_x": tc.reflect("y=x").matrix,
    "reflect_y_eq_neg_x": tc.reflect("y=-x").matrix,
}
MATRIX_TO_LINEAR = {matrix: name for name, matrix in LINEAR_TO_MATRIX.items()}

if len(LINEAR_TO_MATRIX) != 8 or len(MATRIX_TO_LINEAR) != 8:  # pragma: no cover
    raise RuntimeError("v6 D4 vocabulary must contain exactly eight distinct matrices")
if set(LINEAR_TO_MATRIX.values()) != set(tc.ALL_LINEAR_MAPS):  # pragma: no cover
    raise RuntimeError("v6 D4 vocabulary drifted from transform_core.ALL_LINEAR_MAPS")

_NET_KEYS = frozenset(("linear", "tx", "ty"))
_LINEAR_DESCRIPTIONS = {
    "identity": "identity linear map",
    "rot_ccw_90": "90-degree counterclockwise rotation",
    "rot_180": "180-degree rotation",
    "rot_ccw_270": "270-degree counterclockwise rotation",
    "reflect_x_axis": "reflection across the x axis",
    "reflect_y_axis": "reflection across the y axis",
    "reflect_y_eq_x": "reflection across y = x",
    "reflect_y_eq_neg_x": "reflection across y = -x",
}


def _strict_int(value: object, field: str) -> int:
    """Return a JSON integer, rejecting booleans and coercible strings/floats."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer, got {value!r}")
    return value


def validate_net(value: object) -> NetTransform:
    """Validate and return a canonical copy of a v6 net-transform object.

    Validation is deliberately strict: all and only ``linear``/``tx``/``ty`` are
    accepted, enum spelling is exact, and translations must already be integers.
    This keeps metric comparisons free of silent coercions.
    """
    if not isinstance(value, Mapping):
        raise ValueError(f"net transform must be an object, got {type(value).__name__}")
    keys = frozenset(value.keys())
    if keys != _NET_KEYS:
        missing = sorted(_NET_KEYS - keys)
        extra = sorted(keys - _NET_KEYS)
        raise ValueError(f"net transform keys must be linear/tx/ty; missing={missing}, extra={extra}")
    linear = value["linear"]
    if not isinstance(linear, str) or linear not in LINEAR_TO_MATRIX:
        raise ValueError(
            f"linear must be one of {list(D4_LINEAR_NAMES)}, got {linear!r}"
        )
    return {
        "linear": linear,
        "tx": _strict_int(value["tx"], "tx"),
        "ty": _strict_int(value["ty"], "ty"),
    }


def is_net(value: object) -> bool:
    """Return whether ``value`` is a strictly valid canonical net object."""
    try:
        validate_net(value)
    except (TypeError, ValueError):
        return False
    return True


def affine_to_net(transform: tc.Transform) -> NetTransform:
    """Convert a canonical :class:`transform_core.Transform` to v6 JSON."""
    if not isinstance(transform, tc.Transform):
        raise TypeError(f"expected transform_core.Transform, got {type(transform).__name__}")
    linear = MATRIX_TO_LINEAR.get(transform.matrix)
    if linear is None:
        raise ValueError(f"affine linear part is outside D4: {transform.matrix!r}")
    tx, ty = transform.vec
    return validate_net({"linear": linear, "tx": tx, "ty": ty})


def sequence_to_net(sequence: Sequence[tc.TransformLike]) -> NetTransform:
    """Compose an ordered legacy step sequence and return its unique net JSON."""
    return affine_to_net(tc.compose(sequence))


def net_to_affine(value: object) -> tc.Transform:
    """Convert strict v6 JSON back to the canonical affine-map dataclass."""
    net = validate_net(value)
    return tc.Transform(
        LINEAR_TO_MATRIX[str(net["linear"])],
        (int(net["tx"]), int(net["ty"])),
    )


def canonical_net_equal(left: object, right: object) -> bool:
    """Strict canonical equality; invalid values raise instead of matching."""
    return validate_net(left) == validate_net(right)


def describe_net(value: object) -> str:
    """Readable, unambiguous description of a canonical net transform."""
    net = validate_net(value)
    linear = _LINEAR_DESCRIPTIONS[str(net["linear"])]
    return f"{linear}, followed by net translation ({net['tx']}, {net['ty']})"


def diagnose_nets(correct_net: object, student_net: object) -> str:
    """Derive the diagnosis label from two net maps through the canonical oracle."""
    correct = net_to_affine(correct_net)
    student = net_to_affine(student_net)
    # transform_core.diagnose is explicitly a function of these two maps; the
    # original polygon argument is retained by its legacy API but is not used.
    return tc.diagnose([], [correct], [student])


__all__ = [
    "D4_LINEAR_NAMES",
    "LINEAR_TO_MATRIX",
    "MATRIX_TO_LINEAR",
    "NetTransform",
    "affine_to_net",
    "canonical_net_equal",
    "describe_net",
    "diagnose_nets",
    "is_net",
    "net_to_affine",
    "sequence_to_net",
    "validate_net",
]
