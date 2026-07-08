"""
Data generator for the visual two-step transformation identification task.

Each problem renders a coordinate grid (-10..10) with a red pre-image polygon P and a
green image polygon, and asks which pair of transformations maps P to the image.

Design decisions (locked with the project owner):
  * Answer choices are TYPE-ONLY (no numbers/amounts) and it is a multiple-choice question.
  * Two shapes are shown (P and its image); no intermediate step.
  * One order is given as the correct answer; its equivalent order-swap is never offered,
    because the swap produces the identical picture.
  * Distractors are compute-verified to be genuinely wrong (via reflection-parity, an
    invariant no composition of that type-pair can violate, plus a rotation-vs-translation
    check for a harder same-parity distractor).
  * Only chiral (asymmetric) polygons are used, so a reflection is visually distinct from a
    rotation. Symmetric shapes are filtered out automatically.
  * ~75% two-step, ~25% single-step (primitives). An OOD split holds out chosen compositions
    so the eval measures generalization, not memorization.

Ground truth is exact because every transform is computed, so no hand labeling is needed.
"""

import argparse
import json
import os
import random

# --------------------------------------------------------------------------------------
# Exact integer rigid motions
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
# Operations: (kind, param). kind in {translate, rotate, reflect}
# --------------------------------------------------------------------------------------

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

def sample_op(kind, rng):
    if kind == "translate":
        while True:
            dx = rng.randint(-8, 8)
            dy = rng.randint(-8, 8)
            if (dx, dy) != (0, 0):
                return ("translate", (dx, dy))
    if kind == "rotate":
        return ("rotate", rng.choice([90, 180, 270]))
    if kind == "reflect":
        return ("reflect", rng.choice(["x", "y", "y=x", "y=-x"]))
    raise ValueError(kind)

VERB = {"translate": "translated", "rotate": "rotated", "reflect": "reflected"}

def pair_label(k1, k2):
    return f"{VERB[k1]}, then {VERB[k2]}"

def single_label(k):
    return VERB[k]

# --------------------------------------------------------------------------------------
# Shape library: chiral polyominoes, traced to outlines, then filtered to simple + chiral.
# Shapes are given as sets of unit cells; the boundary polygon is computed automatically.
# --------------------------------------------------------------------------------------

SHAPE_CELLS = [
    # tetrominoes
    [(0, 0), (0, 1), (0, 2), (1, 0)],
    [(0, 0), (1, 0), (1, 1), (1, 2)],
    [(1, 0), (2, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (2, 1)],
    # pentominoes
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0)],
    [(1, 0), (1, 1), (1, 2), (1, 3), (0, 0)],
    [(1, 0), (0, 1), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2)],
    [(0, 0), (1, 0), (1, 1), (1, 2), (2, 2)],
    [(1, 0), (1, 1), (0, 2), (1, 2), (0, 3)],
    [(0, 1), (1, 1), (2, 1), (2, 0), (0, 2)],
    # hexominoes and other asymmetric blobs
    [(0, 0), (1, 0), (2, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (2, 2), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3)],
    [(0, 0), (1, 0), (2, 0), (3, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (1, 2)],
    [(0, 3), (1, 3), (1, 2), (2, 2), (2, 1), (3, 1), (3, 0)],
    [(0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (3, 0)],
    [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1)],
    [(0, 0), (1, 0), (2, 0), (2, 1), (3, 1), (1, 1)],
]

def polyomino_outline(cells):
    cells = set(cells)
    edges = []
    for (i, j) in cells:
        if (i, j - 1) not in cells: edges.append(((i, j), (i + 1, j)))
        if (i + 1, j) not in cells: edges.append(((i + 1, j), (i + 1, j + 1)))
        if (i, j + 1) not in cells: edges.append(((i + 1, j + 1), (i, j + 1)))
        if (i - 1, j) not in cells: edges.append(((i, j + 1), (i, j)))
    adj = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
    start = edges[0][0]
    poly = [start]
    cur, used = start, set()
    while True:
        nb = next((c for c in adj.get(cur, []) if (cur, c) not in used), None)
        if nb is None:
            break
        used.add((cur, nb))
        if nb == start:
            break
        poly.append(nb)
        cur = nb
    out = []
    n = len(poly)
    for k in range(n):
        a, b, c = poly[(k - 1) % n], poly[k], poly[(k + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            out.append(b)
    return out

def _seg_intersect(a, b, c, d):
    def ccw(p, q, r):
        return (r[1] - p[1]) * (q[0] - p[0]) - (q[1] - p[1]) * (r[0] - p[0])
    d1 = ccw(c, d, a)
    d2 = ccw(c, d, b)
    d3 = ccw(a, b, c)
    d4 = ccw(a, b, d)
    if ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0)):
        return True
    return False

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

def _canon(pts):
    mnx = min(x for x, y in pts)
    mny = min(y for x, y in pts)
    return tuple(sorted((x - mnx, y - mny) for x, y in pts))

def is_chiral(pts):
    mirror = reflect(pts, "x")
    for deg in (0, 90, 180, 270):
        m = mirror if deg == 0 else rotate(mirror, deg)
        if _canon(pts) == _canon(m):
            return False
    return True

def build_shape_library():
    lib, seen = [], set()
    for cells in SHAPE_CELLS:
        pts = polyomino_outline(cells)
        if len(pts) < 3:
            continue
        if orientation(pts) < 0:
            pts = list(reversed(pts))
        key = _canon(pts)
        if key in seen:
            continue
        if is_simple(pts) and is_chiral(pts):
            seen.add(key)
            lib.append(pts)
    if len(lib) < 8:
        raise RuntimeError(f"only {len(lib)} valid chiral shapes; add more to SHAPE_CELLS")
    return lib

# --------------------------------------------------------------------------------------
# Problem construction
# --------------------------------------------------------------------------------------

LIM = 10
BOUND = LIM - 1  # keep shapes strictly inside the visible grid

def in_bounds(pts):
    return all(-BOUND <= x <= BOUND and -BOUND <= y <= BOUND for x, y in pts)

def too_close(a, b, gap=1):
    # require a clear gap between the two shapes' bounding boxes so they never overlap
    ax0 = min(x for x, y in a); ax1 = max(x for x, y in a)
    ay0 = min(y for x, y in a); ay1 = max(y for x, y in a)
    bx0 = min(x for x, y in b); bx1 = max(x for x, y in b)
    by0 = min(y for x, y in b); by1 = max(y for x, y in b)
    clear = (ax1 + gap < bx0) or (bx1 + gap < ax0) or (ay1 + gap < by0) or (by1 + gap < ay0)
    return not clear

def is_pure_translation(pre, img):
    v = (img[0][0] - pre[0][0], img[0][1] - pre[0][1])
    return all((ix - px, iy - py) == v for (px, py), (ix, iy) in zip(pre, img))

def place_preimage(shape, rng):
    mnx = min(x for x, y in shape); mny = min(y for x, y in shape)
    base = [(x - mnx, y - mny) for x, y in shape]
    w = max(x for x, y in base); h = max(y for x, y in base)
    ox = rng.randint(-BOUND, BOUND - w)
    oy = rng.randint(-BOUND, BOUND - h)
    return [(x + ox, y + oy) for x, y in base]

def sample_two_step_kinds(rng):
    # distinct types, in a random order
    a, b = rng.sample(["translate", "rotate", "reflect"], 2)
    return a, b

def generate_problem(lib, rng, two_step=True, max_tries=400):
    for _ in range(max_tries):
        shape = rng.choice(lib)
        pre = place_preimage(shape, rng)
        if not in_bounds(pre):
            continue
        if two_step:
            k1, k2 = sample_two_step_kinds(rng)
            ops = [sample_op(k1, rng), sample_op(k2, rng)]
            kinds = (k1, k2)
        else:
            k1 = rng.choice(["translate", "rotate", "reflect"])
            ops = [sample_op(k1, rng)]
            kinds = (k1,)
        img = apply_seq(pre, ops)
        if not in_bounds(img):
            continue
        if img == pre:
            continue
        if too_close(pre, img):
            continue
        return {"pre": pre, "img": img, "ops": ops, "kinds": kinds,
                "parity": 1 if orientation(img) != orientation(pre) else 0,
                "pure_translation": is_pure_translation(pre, img)}
    return None

# --------------------------------------------------------------------------------------
# Answer choices (type-only). Correct answer + compute-verified distractors.
# --------------------------------------------------------------------------------------

DISTINCT_PAIRS = [("rotate", "translate"), ("translate", "rotate"),
                  ("rotate", "reflect"), ("reflect", "rotate"),
                  ("translate", "reflect"), ("reflect", "translate")]

def refl_parity(kinds):
    return sum(1 for k in kinds if k == "reflect") % 2

def two_step_options(problem, rng):
    k1, k2 = problem["kinds"]
    correct = (k1, k2)
    swap = (k2, k1)
    p = problem["parity"]

    opposite = [pr for pr in DISTINCT_PAIRS
                if pr != correct and pr != swap and refl_parity(pr) != p]
    same_hard = []
    # "translated, then translated": parity 0, cannot make a rotation
    if p == 0 and not problem["pure_translation"]:
        same_hard.append(("translate", "translate"))
    # "reflected, then reflected": parity 0, genuinely wrong for an odd-parity image
    if p == 1:
        same_hard.append(("reflect", "reflect"))

    distractors = []
    rng.shuffle(opposite)
    rng.shuffle(same_hard)
    # include one harder same-parity/structural distractor when available
    if same_hard:
        distractors.append(same_hard[0])
    distractors += opposite
    distractors = distractors[:3]

    if len(distractors) < 3:  # safety net
        for pr in DISTINCT_PAIRS:
            if pr not in (correct, swap) and pr not in distractors:
                distractors.append(pr)
            if len(distractors) == 3:
                break

    labels = [pair_label(*correct)] + [pair_label(*d) for d in distractors]
    return _assemble_mcq(labels, rng)

def single_step_options(problem, rng):
    correct = problem["kinds"][0]
    others = [k for k in ["translate", "rotate", "reflect"] if k != correct]
    labels = [single_label(correct)] + [single_label(k) for k in others]
    return _assemble_mcq(labels, rng)

def _assemble_mcq(labels, rng):
    correct_label = labels[0]
    rng.shuffle(labels)
    idx = labels.index(correct_label)
    letters = ["A", "B", "C", "D"][:len(labels)]
    return {"options": labels, "answer_index": idx,
            "answer_letter": letters[idx], "answer_label": correct_label}

def build_question(mcq, two_step):
    n = "two transformations, in order," if two_step else "one transformation"
    lines = [f"The solid red polygon P is the pre-image. The green polygon is its image after "
             f"{n} applied to P.",
             "Which of these was applied? Reply with the letter only."]
    for letter, opt in zip(["A", "B", "C", "D"], mcq["options"]):
        lines.append(f"{letter}) {opt}")
    return "\n".join(lines)

# --------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------

def render(pre, img, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks(range(-LIM, LIM + 1))
    ax.set_yticks(range(-LIM, LIM + 1))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=5)
    ax.axhline(0, color="black", linewidth=1.1)
    ax.axvline(0, color="black", linewidth=1.1)

    def draw(pts, color):
        xs = [p[0] for p in pts] + [pts[0][0]]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ax.plot(xs, ys, color=color, linewidth=2.0, solid_joinstyle="round")

    draw(pre, "#d1344e")   # red pre-image
    draw(img, "#1f9d55")   # green image
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# --------------------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------------------

def to_chat_record(image_path, question, answer_letter):
    return {"messages": [
        {"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": question},
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": answer_letter}]},
    ]}

def generate_dataset(out_dir, n, single_step_fraction, ood_holdout, seed, render_images=True):
    rng = random.Random(seed)
    lib = build_shape_library()
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    records = []
    i = 0
    while len(records) < n:
        two_step = rng.random() >= single_step_fraction
        prob = generate_problem(lib, rng, two_step=two_step)
        if prob is None:
            continue
        mcq = two_step_options(prob, rng) if two_step else single_step_options(prob, rng)
        question = build_question(mcq, two_step)

        # OOD split: two-step problems whose ordered type-pair is held out go to the ood set
        split = "train"
        if two_step and prob["kinds"] in ood_holdout:
            split = "ood"
        elif rng.random() < 0.1:
            split = "val"

        img_name = f"{i:06d}.png"
        img_path = os.path.join(img_dir, img_name)
        if render_images:
            render(prob["pre"], prob["img"], img_path)

        records.append({
            "id": i,
            "split": split,
            "image": os.path.join("images", img_name),
            "two_step": two_step,
            "question": question,
            "options": mcq["options"],
            "answer_letter": mcq["answer_letter"],
            "answer_label": mcq["answer_label"],
            "ground_truth": {
                "pre_image": prob["pre"],
                "image": prob["img"],
                "ops": prob["ops"],
                "kinds": list(prob["kinds"]),
                "parity": prob["parity"],
            },
        })
        i += 1

    for split in ("train", "val", "ood"):
        rows = [r for r in records if r["split"] == split]
        with open(os.path.join(out_dir, f"{split}.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        with open(os.path.join(out_dir, f"{split}_chat.jsonl"), "w") as f:
            for r in rows:
                f.write(json.dumps(to_chat_record(r["image"], r["question"], r["answer_letter"])) + "\n")

    counts = {s: sum(1 for r in records if r["split"] == s) for s in ("train", "val", "ood")}
    return records, counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--single-step-fraction", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-render", action="store_true", help="skip PNGs (logic/label check only)")
    args = ap.parse_args()

    # hold out one ordered composition entirely for the OOD generalization eval
    ood_holdout = {("reflect", "translate")}

    records, counts = generate_dataset(
        args.out, args.n, args.single_step_fraction, ood_holdout,
        args.seed, render_images=not args.no_render,
    )
    print("generated:", len(records), "->", counts)
    print("held out for OOD:", sorted(ood_holdout))

if __name__ == "__main__":
    main()
