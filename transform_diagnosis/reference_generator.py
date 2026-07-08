"""reference_generator — a small, self-contained CONSUMER example for transform_core.

This is the reference for how the rest of the package (and any external caller) should
use the canonical API: build shapes with :mod:`geometry`, express transformations with
``transform_core.rotate / reflect / translate``, compose them (``seq[0]`` first), apply,
grade, diagnose a student attempt, and render. It deliberately re-implements NOTHING —
all transform / grade / diagnose math comes from :mod:`transform_core`.

Run it directly::

    python3 -m transform_diagnosis.reference_generator
    python3 -m transform_diagnosis.reference_generator --render /tmp/ref_example.png
"""

from __future__ import annotations

import argparse
import random
from typing import List

from . import geometry, transform_core as tc


def build_example(seed: int = 7) -> dict:
    """Construct one fully worked example using only the canonical API."""
    rng = random.Random(seed)

    # 1. A single irregular, asymmetric polygon on the integer lattice.
    original = geometry.generate_irregular_polygon(
        num_vertices=5, irregularity=0.5, spikiness=0.35, radius=4.0, rng=rng,
        snap=True, require_asymmetric=True,
    )

    # 2. The CORRECT transformation: a composition of two rigid moves (seq[0] first).
    correct_seq = [tc.rotate(90, "ccw"), tc.translate(3, -2)]
    correct_image = tc.compose(correct_seq).apply(original)

    # 3. A student attempt: reflection instead of the rotation (same translation).
    student_seq = [tc.reflect("x"), tc.translate(3, -2)]
    student_image = tc.compose(student_seq).apply(original)

    # 4. Grade + diagnose (exact integer arithmetic; no model, no heuristic).
    is_correct = tc.grade(original, correct_image, student_seq)
    label = tc.diagnose(original, correct_seq, student_seq)

    # 5. Uniqueness: recover the net map straight from (original, correct_image).
    recovered = tc.recover_map(original, correct_image)
    assert recovered == tc.compose(correct_seq)

    return {
        "original": original,
        "correct_transform": tc.describe_seq(correct_seq),
        "correct_image": correct_image,
        "student_transform": tc.describe_seq(student_seq),
        "student_image": student_image,
        "is_correct": is_correct,
        "label": label,
    }


def _fmt(pts: List) -> str:
    return "[" + ", ".join(f"({x},{y})" for x, y in pts) + "]"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--render", metavar="PNG", default=None,
                    help="also render the example to this PNG path")
    args = ap.parse_args(argv)

    ex = build_example(args.seed)
    print("original          :", _fmt(ex["original"]))
    print("correct_transform :", ex["correct_transform"])
    print("correct_image     :", _fmt(ex["correct_image"]))
    print("student_transform :", ex["student_transform"])
    print("student_image     :", _fmt(ex["student_image"]))
    print("is_correct        :", ex["is_correct"])
    print("diagnosis label   :", ex["label"])

    if args.render:
        from . import render
        render.render_to_path(ex["original"], ex["student_image"], args.render,
                              correct_image=ex["correct_image"])
        print("rendered          :", args.render)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
