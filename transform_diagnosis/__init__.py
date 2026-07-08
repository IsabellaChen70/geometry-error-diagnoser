"""transform_diagnosis — deterministic student-error diagnosis dataset for composed
geometric transformations.

All transform / grade / diagnose math lives in the single canonical module
:mod:`transform_diagnosis.transform_core`; every other module imports it. The most
commonly used names are re-exported here for convenience.
"""

from .transform_core import (
    DIAGNOSIS_LABELS,
    ORIENTATIONS,
    Transform,
    compose,
    describe_transform,
    diagnose,
    grade,
    identity,
    is_asymmetric,
    parse_transform,
    recover_map,
    reflect,
    rotate,
    translate,
)

__all__ = [
    "DIAGNOSIS_LABELS",
    "ORIENTATIONS",
    "Transform",
    "compose",
    "describe_transform",
    "diagnose",
    "grade",
    "identity",
    "is_asymmetric",
    "parse_transform",
    "recover_map",
    "reflect",
    "rotate",
    "translate",
]

__version__ = "1.0.0"
