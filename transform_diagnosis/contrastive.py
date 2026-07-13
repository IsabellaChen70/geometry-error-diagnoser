"""contrastive — v4 hard contrastive quadruplets + single-step curriculum records.

Two extra record generators layered on top of the existing verified error-injection
machinery (``problems`` + ``errors`` + ``dataset._partial_record``). No new geometry and
no new labelling path: every student attempt is produced by the SAME ``errors.inject``
verify-or-discard loop the balanced dataset uses, so the stored ``label`` always equals
``transform_core.diagnose`` of the stored transforms.

1. Hard contrastive quadruplets (:func:`build_contrastive_group`).
   Four matched records that share ONE RED pre-image and, within each rotate/reflect
   family, ONE intended transform, differing only by the injected error. They cover the
   four confusable labels the v4 audit isolated:

       reflection_instead_of_rotation  (rotation base -- wrong operation TYPE)
       wrong_rotation_angle            (rotation base -- wrong PARAMETER)
       rotation_instead_of_reflection  (reflection base -- wrong operation TYPE)
       wrong_reflection_line           (reflection base -- wrong PARAMETER)

   WHY TWO BASES (and not one): ``diagnose`` is a function of the two NET affine maps.
   ``reflection_instead_of_rotation`` / ``wrong_rotation_angle`` REQUIRE the correct net to
   be a rotation; ``rotation_instead_of_reflection`` / ``wrong_reflection_line`` REQUIRE it
   to be a reflection. A single intended transform has one net orientation, so the four
   labels cannot all share one correct transform. We therefore build a rotation-net base
   ``[rotate R, translate T]`` and a reflection-net base ``[reflect L, translate T]`` on the
   SAME RED and the SAME translation ``T`` -- the two intended transforms differ ONLY in the
   first step's TYPE (rotate vs reflect), which is exactly the wrong-type-vs-wrong-parameter
   boundary these matched sets are meant to teach.

2. Curriculum records (:func:`build_curriculum_partials`).
   Easier warm-ups: a SINGLE-step correct transform with a SINGLE-step student error
   (simpler than the default two-step problems). Built with the existing injectors on
   one-move patterns (``("rotate",)`` / ``("reflect",)`` / ``("translate",)``).

Both return "partials" (the same dict shape ``dataset._partial_record`` yields); the
assembler (``model/make_v4_data.py``) wraps them into full, asserted, schema-ordered
records via :func:`dataset.finalize_record` and renders them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from . import dataset, errors, geometry, problems
from . import transform_core as tc

Point = Tuple[int, int]

# The four confusable labels, in a fixed order (rotation family first, then reflection
# family; within each: wrong-TYPE then wrong-PARAMETER).
CONFUSABLE_LABELS: Tuple[str, ...] = (
    "reflection_instead_of_rotation",
    "wrong_rotation_angle",
    "rotation_instead_of_reflection",
    "wrong_reflection_line",
)
_ROTATION_BASE_LABELS = ("reflection_instead_of_rotation", "wrong_rotation_angle")
_REFLECTION_BASE_LABELS = ("rotation_instead_of_reflection", "wrong_reflection_line")
# Both patterns are IN-DISTRIBUTION (not held-out compositions), so contrastive records do
# NOT contaminate the OOD generalization slice.
_ROTATION_PATTERN = ("rotate", "translate")
_REFLECTION_PATTERN = ("reflect", "translate")

# Single-step curriculum specs: (one-move pattern, target label). Each is a clean one-step
# transform with a one-step error (completely_wrong is intentionally excluded -- it is a
# both-parts-wrong compound, not a "simpler" warm-up).
CURRICULUM_SPECS: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    (("rotate",), "correct"),
    (("rotate",), "reflection_instead_of_rotation"),
    (("rotate",), "wrong_rotation_angle"),
    (("reflect",), "correct"),
    (("reflect",), "rotation_instead_of_reflection"),
    (("reflect",), "wrong_reflection_line"),
    (("translate",), "correct"),
    (("translate",), "wrong_translation"),
    (("translate",), "opposite_translation"),
)


@dataclass
class ContrastiveGroup:
    """A matched quadruplet: one shared RED pre-image, one shared translation, two intended
    transforms (rotation-net and reflection-net) differing only in the first step's type,
    and the four confusable-label partials (order = ``CONFUSABLE_LABELS``)."""

    original: List[Point]
    rotation_correct_text: List[str]
    reflection_correct_text: List[str]
    partials: List[dict] = field(default_factory=list)


def _reflection_base(
    rng: random.Random, original: Sequence[Point], translate_t: tc.Transform,
    translate_text: str,
) -> Optional[problems.Problem]:
    """Build a reflection-net base ``[reflect L, translate T]`` on ``original`` that REUSES
    the given translation ``T`` (so it matches the rotation base's slide). Tries each
    reflection line; returns the first in-bounds, non-degenerate result, else ``None``.
    All geometry goes through ``transform_core`` (compose/apply)."""
    base = tc.as_points(original)
    lines = list(problems.REFLECTION_LINES)
    rng.shuffle(lines)
    for line in lines:
        refl = tc.reflect(line)
        answer = [refl, translate_t]
        image = tc.compose(answer).apply(base)
        if not geometry.in_bounds(image, -problems.BOUND, problems.BOUND):
            continue
        if image == base:
            continue
        return problems.Problem(
            original=list(base),
            image=list(image),
            answer=answer,
            answer_text=[tc.describe_transform(refl), translate_text],
            pattern=_REFLECTION_PATTERN,
            num_vertices=len(base),
        )
    return None


def build_contrastive_group(
    rng: random.Random, *, max_attempts: int = 200,
) -> ContrastiveGroup:
    """Build ONE matched contrastive quadruplet (deterministic given ``rng``).

    Retries with a fresh RED / sampled parameters until all four injections succeed and
    stay in bounds. Raises ``RuntimeError`` if it cannot after ``max_attempts`` tries.
    """
    for _ in range(max_attempts):
        nv = rng.choice(problems.VERTEX_CHOICES)
        red = geometry.generate_irregular_polygon(
            nv, irregularity=0.5, spikiness=0.35, radius=4.0, rng=rng,
            snap=True, require_asymmetric=True,
        )
        try:
            rot = problems.make_problem(rng, pattern=_ROTATION_PATTERN, original=red)
        except RuntimeError:
            continue
        t_idx = rot.pattern.index("translate")
        refl = _reflection_base(rng, red, rot.answer[t_idx], rot.answer_text[t_idx])
        if refl is None:
            continue

        partials: List[dict] = []
        ok = True
        for label in CONFUSABLE_LABELS:
            base = rot if label in _ROTATION_BASE_LABELS else refl
            injected = errors.inject(base, label, rng)
            if injected is None:
                ok = False
                break
            student_seq, student_text = injected
            partials.append(dataset._partial_record(base, student_seq, student_text, label))
        if not ok:
            continue

        return ContrastiveGroup(
            original=list(red),
            rotation_correct_text=list(rot.answer_text),
            reflection_correct_text=list(refl.answer_text),
            partials=partials,
        )
    raise RuntimeError(
        f"could not build a contrastive quadruplet after {max_attempts} attempts"
    )


def build_contrastive_partials(
    rng: random.Random, n_groups: int, *, max_attempts: int = 200,
) -> Tuple[List[dict], List[ContrastiveGroup]]:
    """Build ``n_groups`` contrastive quadruplets. Returns ``(flat_partials, groups)`` where
    ``flat_partials`` is the concatenation of every group's 4 partials (order preserved)."""
    groups = [build_contrastive_group(rng, max_attempts=max_attempts) for _ in range(n_groups)]
    flat = [p for g in groups for p in g.partials]
    return flat, groups


def build_curriculum_partials(
    rng: random.Random, n: int, *, max_attempts_factor: int = 400,
) -> List[dict]:
    """Build ``n`` single-step curriculum partials, round-robin over ``CURRICULUM_SPECS``
    (so labels/patterns stay balanced). Deterministic given ``rng``."""
    out: List[dict] = []
    attempts = 0
    cap = max_attempts_factor * max(n, 1)
    while len(out) < n:
        attempts += 1
        if attempts > cap:
            raise RuntimeError(f"could not build {n} curriculum records after {attempts} attempts")
        pattern, label = CURRICULUM_SPECS[len(out) % len(CURRICULUM_SPECS)]
        try:
            problem = problems.make_problem(rng, pattern=pattern)
        except RuntimeError:
            continue
        injected = errors.inject(problem, label, rng)
        if injected is None:
            continue
        student_seq, student_text = injected
        out.append(dataset._partial_record(problem, student_seq, student_text, label))
    return out
