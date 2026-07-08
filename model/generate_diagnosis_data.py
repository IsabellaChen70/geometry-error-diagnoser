"""
Deterministic generator for "Composed Geometric Transformations" error-diagnosis data.

Two parts:

  Part 1 - generate_irregular_polygon(): random single irregular polygons (weird
           quadrilaterals / pentagons / hexagons with uneven sides and angles), NOT
           combinations of shapes. Uses the angular-sweep method, which guarantees a
           simple (non-self-intersecting) polygon.

  Part 2 - a diagnosis dataset. For each problem we:
             1. take an irregular polygon as the pre-image P,
             2. define a composed transformation task (two rigid motions, in order),
             3. compute the CORRECT image by exact integer arithmetic,
             4. simulate a student applying ONE known misconception (wrong order,
                skipped a step, wrong axis, wrong direction, confused reflection with
                rotation, ...) and compute the student's WRONG image exactly,
             5. label the record with the misconception + a pedagogical hint,
             6. render the pre-image + the student's answer on a coordinate grid.

Design principles (from the project owner):
  * Deterministic first: every answer is computed, never predicted.
  * Self-validating: a record is only emitted if its student image maps to a UNIQUE
    misconception in our taxonomy (no ambiguous labels). This is enforced per task.
  * Reproducible: everything is driven by a single seed.

Output per split (train/val): a full record file `{split}.jsonl` (with ground truth)
and a training-ready `{split}_chat.jsonl` (image + question -> JSON answer).
"""

import argparse
import json
import math
import os
import random

# --------------------------------------------------------------------------------------
# Exact integer rigid motions (only motions that keep integer coordinates integer)
# --------------------------------------------------------------------------------------

def translate(pts, dx, dy):
    return [(x + dx, y + dy) for x, y in pts]

def rotate(pts, deg):  # counterclockwise about the origin
    r = {
        90: lambda x, y: (-y, x),
        180: lambda x, y: (-x, -y),
        270: lambda x, y: (y, -x),
    }[deg]
    return [r(x, y) for x, y in pts]

def reflect(pts, axis):
    f = {
        "x": lambda x, y: (x, -y),
        "y": lambda x, y: (-x, y),
        "y=x": lambda x, y: (y, x),
        "y=-x": lambda x, y: (-y, -x),
    }[axis]
    return [f(x, y) for x, y in pts]

def apply_op(pts, op):
    kind, param = op
    if kind == "translate":
        return translate(pts, param[0], param[1])
    if kind == "rotate":
        return rotate(pts, param)
    if kind == "reflect":
        return reflect(pts, param)
    raise ValueError(kind)

def apply_seq(pts, ops):
    for op in ops:
        pts = apply_op(pts, op)
    return pts

def signed_area(pts):
    s = 0.0
    n = len(pts)
    for i in range(n):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0

def orientation(pts):
    return 1 if signed_area(pts) > 0 else -1

# --------------------------------------------------------------------------------------
# Simple-polygon test (segment intersection), used to reject self-intersecting shapes
# --------------------------------------------------------------------------------------

def _seg_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = ccw(c, d, a)
    d2 = ccw(c, d, b)
    d3 = ccw(a, b, c)
    d4 = ccw(a, b, d)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))

def is_simple(pts):
    n = len(pts)
    edges = [(pts[i], pts[(i + 1) % n]) for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if abs(i - j) <= 1 or (i == 0 and j == n - 1):
                continue
            if _seg_intersect(edges[i][0], edges[i][1], edges[j][0], edges[j][1]):
                return False
    return True

def _clip(v, lo, hi):
    return lo if v < lo else hi if v > hi else v

# --------------------------------------------------------------------------------------
# Part 1: irregular single-polygon generation
# --------------------------------------------------------------------------------------

def generate_irregular_polygon(num_vertices=5, irregularity=0.5, spikiness=0.3,
                               radius=4.0, center=(0, 0), rng=None):
    """
    Generate an irregular polygon with the given number of vertices.

    - irregularity: how much the angular spacing between vertices varies
                    (0 = evenly spaced, 1 = highly uneven).
    - spikiness:    how much the vertex radii vary (0 = smooth/near-regular,
                    1 = very spiky/weird).
    - radius:       average distance from `center` to a vertex.
    - center:       polygon center (cx, cy).
    - rng:          a random.Random instance (for reproducibility).

    Returns a list of (x, y) float coordinates in counter-clockwise-ish order.
    The angular-sweep construction guarantees a simple (non-self-intersecting) polygon.
    """
    if rng is None:
        rng = random.Random()
    irregularity = _clip(irregularity, 0.0, 1.0) * (2 * math.pi / num_vertices)
    spikiness = _clip(spikiness, 0.0, 1.0) * radius

    # angular steps between consecutive vertices, jittered then normalized to sum to 2*pi
    lower = (2 * math.pi / num_vertices) - irregularity
    upper = (2 * math.pi / num_vertices) + irregularity
    steps = [rng.uniform(lower, upper) for _ in range(num_vertices)]
    scale = (2 * math.pi) / sum(steps)
    steps = [s * scale for s in steps]

    pts = []
    angle = rng.uniform(0, 2 * math.pi)
    for i in range(num_vertices):
        r = _clip(rng.gauss(radius, spikiness), 0.2 * radius, 1.8 * radius)
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        pts.append((x, y))
        angle += steps[i]
    return pts

def _min_edge(pts):
    n = len(pts)
    return min(math.dist(pts[i], pts[(i + 1) % n]) for i in range(n))

def _no_collinear(pts):
    n = len(pts)
    for i in range(n):
        a, b, c = pts[(i - 1) % n], pts[i], pts[(i + 1) % n]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if cross == 0:
            return False
    return True

def generate_integer_polygon(rng, num_vertices, irregularity, spikiness,
                             extent=6, min_edge=1.5, max_tries=400):
    """
    Generate an irregular polygon on the integer lattice, validated to be simple,
    non-degenerate (no collinear consecutive triples, minimum edge length), and
    contained within [-extent, extent]^2 so there is room to transform it.
    Radius shrinks gradually if the target extent is hard to hit.
    """
    radius = float(extent) - 1.5
    for attempt in range(max_tries):
        if attempt and attempt % 60 == 0:  # gradually shrink if the extent is hard to hit
            radius *= 0.92
        raw = generate_irregular_polygon(num_vertices, irregularity, spikiness,
                                         radius=radius, center=(0, 0), rng=rng)
        pts = [(round(x), round(y)) for x, y in raw]
        if len({(x, y) for x, y in pts}) < num_vertices:
            continue
        if max(max(abs(x), abs(y)) for x, y in pts) > extent:
            continue
        if not is_simple(pts):
            continue
        if not _no_collinear(pts):
            continue
        if _min_edge(pts) < min_edge:
            continue
        if abs(signed_area(pts)) < 3.0:
            continue
        if orientation(pts) < 0:  # keep a consistent CCW winding
            pts = list(reversed(pts))
        return pts
    raise RuntimeError("could not generate a valid integer polygon; loosen constraints")

# --------------------------------------------------------------------------------------
# Part 2: composed-transformation task + misconception simulation
# --------------------------------------------------------------------------------------

BOUND = 9  # every rendered vertex must satisfy |x|,|y| <= BOUND

def in_bounds(pts):
    return all(-BOUND <= x <= BOUND and -BOUND <= y <= BOUND for x, y in pts)

ALT_AXIS = {"x": "y", "y": "x", "y=x": "y=-x", "y=-x": "y=x"}
AXIS_LABEL = {"x": "the x-axis", "y": "the y-axis",
              "y=x": "the line y = x", "y=-x": "the line y = -x"}

def sample_op(kind, rng):
    if kind == "translate":
        while True:
            dx, dy = rng.randint(-6, 6), rng.randint(-6, 6)
            if (dx, dy) != (0, 0):
                return ("translate", (dx, dy))
    if kind == "rotate":
        return ("rotate", rng.choice([90, 180, 270]))
    if kind == "reflect":
        return ("reflect", rng.choice(["x", "y", "y=x", "y=-x"]))
    raise ValueError(kind)

def sample_task_ops(rng):
    # two DISTINCT transformation types, in a random order (composition is order-sensitive)
    k1, k2 = rng.sample(["translate", "rotate", "reflect"], 2)
    return [sample_op(k1, rng), sample_op(k2, rng)]

def describe_op(op):
    kind, param = op
    if kind == "translate":
        return f"translate it by ({param[0]}, {param[1]})"
    if kind == "rotate":
        return f"rotate it {param} degrees counterclockwise about the origin"
    if kind == "reflect":
        return f"reflect it across {AXIS_LABEL[param]}"
    raise ValueError(kind)

def describe_task(ops):
    s = f"{describe_op(ops[0])}, then {describe_op(ops[1])}"
    return s[0].upper() + s[1:]

# --------------------------------------------------------------------------------------
# Misconception taxonomy: each entry produces a flawed procedure and a hint template.
# --------------------------------------------------------------------------------------

def hint_for(label, ops, detail):
    if label == "none":
        return f"Correct. The student properly applied both steps: {describe_task(ops).lower()}."
    if label == "swapped_order":
        return ("Order matters in a composition. Apply the first transformation "
                f"({describe_op(ops[0])}), and only then the second ({describe_op(ops[1])}).")
    if label == "skipped_second_step":
        return f"You applied the first step but forgot the second: {describe_op(ops[1])}."
    if label == "skipped_first_step":
        return f"You applied the second step but skipped the first: {describe_op(ops[0])}."
    if label == "wrong_reflection_axis":
        return (f"Check the line of reflection. You reflected across {AXIS_LABEL[detail['wrong']]}, "
                f"but the task asked for {AXIS_LABEL[detail['right']]}.")
    if label == "wrong_translation_direction":
        dx, dy = detail["vec"]
        return (f"Check the sign of the translation. Moving by ({dx}, {dy}) is not the same as "
                f"({-dx}, {-dy}); watch the direction on each axis.")
    if label == "wrong_rotation_direction":
        return ("Positive angles rotate counterclockwise. You turned the shape the wrong way; "
                f"a {detail['right']} degree turn goes the other direction.")
    if label == "reflection_instead_of_rotation":
        return ("A rotation turns the shape but keeps its orientation; a reflection produces a "
                "mirror image (flipped orientation). This step should have been a rotation.")
    if label == "rotation_instead_of_reflection":
        return ("A reflection produces a mirror image (flipped orientation); a rotation keeps "
                "orientation. This step should have been a reflection.")
    return "Re-check the transformation carefully."

def candidate_students(ops):
    """Return candidate (label, student_ops, detail) flawed procedures for this task."""
    cands = [
        ("swapped_order", [ops[1], ops[0]], {}),
        ("skipped_second_step", [ops[0]], {}),
        ("skipped_first_step", [ops[1]], {}),
    ]
    for idx, (kind, param) in enumerate(ops):
        if kind == "reflect":
            alt = ALT_AXIS[param]
            new = list(ops); new[idx] = ("reflect", alt)
            cands.append(("wrong_reflection_axis", new, {"wrong": alt, "right": param}))
            for deg in (90, 180, 270):  # confused a reflection for a rotation
                new = list(ops); new[idx] = ("rotate", deg)
                cands.append(("rotation_instead_of_reflection", new, {}))
        if kind == "translate":
            neg = (-param[0], -param[1])
            new = list(ops); new[idx] = ("translate", neg)
            cands.append(("wrong_translation_direction", new, {"vec": param}))
        if kind == "rotate":
            if param in (90, 270):
                alt = 270 if param == 90 else 90
                new = list(ops); new[idx] = ("rotate", alt)
                cands.append(("wrong_rotation_direction", new, {"right": param}))
            for axis in ("x", "y", "y=x", "y=-x"):  # confused a rotation for a reflection
                new = list(ops); new[idx] = ("reflect", axis)
                cands.append(("reflection_instead_of_rotation", new, {}))
    return cands

def build_task_records(pre, ops, rng):
    """
    Given a pre-image and a task, compute every valid + identifiable misconception image.
    Returns (correct_image, [(label, student_image, student_ops, detail), ...]).
    Identifiability: a label is kept only if its student image is unique across all
    candidate labels (so image -> misconception is one-to-one) and differs from correct.
    """
    correct = apply_seq(pre, ops)
    if not in_bounds(correct):
        return None, []

    # first valid variant per label (some misconceptions have several options)
    per_label = {}
    for label, student_ops, detail in candidate_students(ops):
        img = apply_seq(pre, student_ops)
        if not in_bounds(img):
            continue
        if img == correct or img == pre:
            continue
        if label not in per_label:
            per_label[label] = (img, student_ops, detail)

    # identifiability: drop any image produced by more than one distinct label
    counts = {}
    for label, (img, _, _) in per_label.items():
        counts[tuple(img)] = counts.get(tuple(img), 0) + 1

    kept = []
    for label, (img, student_ops, detail) in per_label.items():
        if counts[tuple(img)] == 1:
            kept.append((label, img, student_ops, detail))
    return correct, kept

# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def render(pre, student, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect("equal")
    ax.set_xticks(range(-10, 11))
    ax.set_yticks(range(-10, 11))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=5)
    ax.axhline(0, color="black", linewidth=1.1)
    ax.axvline(0, color="black", linewidth=1.1)

    def draw(pts, edge, face):
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.fill(xs, ys, color=face, alpha=0.25)
        ax.plot(xs, ys, color=edge, linewidth=2.0, solid_joinstyle="round")

    draw(pre, "#d1344e", "#d1344e")      # red = original / pre-image
    draw(student, "#2b6cb0", "#2b6cb0")  # blue = student's answer
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------

QUESTION = (
    "The red polygon is the original shape. A student was asked to: {task}. "
    "The blue polygon is the student's answer.\n"
    "Diagnose the student's mistake. Reply with a JSON object with exactly two keys: "
    '"misconception" and "hint".'
)

def answer_json(label, hint):
    return json.dumps({"misconception": label, "hint": hint})

def to_chat_record(image_path, question, answer):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": answer}]},
    ]}

def generate_dataset(out_dir, n_tasks, fraction_correct, seed, num_vertices_choices,
                     val_fraction=0.15, render_images=True):
    rng = random.Random(seed)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    rid = 0
    for _ in range(n_tasks):
        nv = rng.choice(num_vertices_choices)
        pre = generate_integer_polygon(rng, nv, irregularity=0.5, spikiness=0.35)

        ops = None
        correct = kept = None
        for _try in range(60):  # resample task until it yields identifiable misconceptions
            ops = sample_task_ops(rng)
            correct, kept = build_task_records(pre, ops, rng)
            if kept:
                break
        if not kept:
            continue

        split = "val" if rng.random() < val_fraction else "train"

        emitted = []
        for label, student_img, student_ops, detail in kept:
            emitted.append((label, student_img, student_ops, detail))
        # optionally add a "no error" example (student answer == correct image)
        if rng.random() < fraction_correct:
            emitted.append(("none", correct, list(ops), {}))

        for label, student_img, student_ops, detail in emitted:
            hint = hint_for(label, ops, detail)
            question = QUESTION.format(task=describe_task(ops).lower())
            answer = answer_json(label, hint)

            img_name = f"{rid:06d}.png"
            img_path = os.path.join(img_dir, img_name)
            if render_images:
                render(pre, student_img, img_path)

            records.append({
                "id": rid,
                "split": split,
                "image": os.path.join("images", img_name),
                "question": question,
                "misconception": label,
                "hint": hint,
                "answer_json": answer,
                "ground_truth": {
                    "pre_image": pre,
                    "task_ops": [[k, list(p) if isinstance(p, tuple) else p] for k, p in ops],
                    "correct_image": correct,
                    "student_ops": [[k, list(p) if isinstance(p, tuple) else p] for k, p in student_ops],
                    "student_image": student_img,
                },
            })
            rid += 1

    for split in ("train", "val"):
        rows = [r for r in records if r["split"] == split]
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(out_dir, f"{split}_chat.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(to_chat_record(r["image"], r["question"], r["answer_json"])) + "\n")

    by_label = {}
    for r in records:
        by_label[r["misconception"]] = by_label.get(r["misconception"], 0) + 1
    counts = {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val")}
    return records, counts, by_label

def main():
    ap = argparse.ArgumentParser(description="Composed-transformation misconception-diagnosis data generator")
    ap.add_argument("--out", default="diagnosis_sample")
    ap.add_argument("--n-tasks", type=int, default=40)
    ap.add_argument("--fraction-correct", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--vertices", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--no-render", action="store_true", help="skip PNGs (logic/label check only)")
    args = ap.parse_args()

    records, counts, by_label = generate_dataset(
        args.out, args.n_tasks, args.fraction_correct, args.seed,
        args.vertices, render_images=not args.no_render,
    )
    print(f"generated {len(records)} records from {args.n_tasks} tasks -> {counts}")
    print("by misconception:")
    for label in sorted(by_label):
        print(f"  {label:32s} {by_label[label]}")

if __name__ == "__main__":
    main()
