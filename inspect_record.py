#!/usr/bin/env python3
"""
inspect_record.py -- manually verify diagnosis dataset records.

Usage:
    python3 inspect_record.py data.jsonl 0 1 2            # print breakdown
    python3 inspect_record.py data.jsonl 0 --render out   # also draw a PNG per id
    python3 inspect_record.py data.jsonl --label completely_wrong -n 5

For each record it:
  * re-applies the stored correct/student transforms to `original` and checks the
    result equals the stored images (catches any generation bug),
  * INDEPENDENTLY reads the single net move that maps original -> each image
    straight from the coordinates (via recover_map) so you can compare it to the
    words, and
  * explains, in plain English, why the label follows.
Everything routes through transform_core -- the one source of truth.
"""
from __future__ import annotations
import argparse, json, sys
from transform_diagnosis import transform_core as tc

# name a NET (possibly composed) map from its matrix+vec, for human eyes
_ROT = {((1,0),(0,1)):"no turn", ((0,-1),(1,0)):"rotate 90 ccw",
        ((-1,0),(0,-1)):"rotate 180", ((0,1),(-1,0)):"rotate 270 ccw"}
_REF = {((1,0),(0,-1)):"reflect across x axis", ((-1,0),(0,1)):"reflect across y axis",
        ((0,1),(1,0)):"reflect across y=x", ((0,-1),(-1,0)):"reflect across y=-x"}

def describe_net(T: tc.Transform) -> str:
    if T is None: return "NOT a lattice isometry (shape distorted!)"
    lin = _ROT.get(T.matrix) or _REF.get(T.matrix) or f"matrix {T.matrix}"
    dx, dy = T.vec
    slide = "" if (dx,dy)==(0,0) else f", then translate ({dx},{dy})"
    kind = "flip" if T.det()==-1 else ("turn" if T.matrix!=((1,0),(0,1)) else "slide-only")
    return f"{lin}{slide}   [{kind}, det={T.det()}]"

def inspect(rec: dict) -> None:
    orig = [(int(x),int(y)) for x,y in rec["original"]]
    ci   = [(int(x),int(y)) for x,y in rec["correct_image"]]
    si   = [(int(x),int(y)) for x,y in rec["student_image"]]
    ct, st, lab = rec["correct_transform"], rec["student_transform"], rec["label"]

    print(f"\n===== id {rec['id']}  ({rec['num_vertices']}-gon, split={rec.get('split','?')}) =====")
    print(f"original        : {orig}")
    print(f"correct answer  : {ct}")
    print(f"  -> stored correct_image : {ci}")
    print(f"  -> recomputed           : {tc.compose(ct).apply(orig)}   "
          f"{'MATCH' if tc.compose(ct).apply(orig)==ci else '*** MISMATCH ***'}")
    print(f"student attempt : {st}")
    print(f"  -> stored student_image : {si}")
    print(f"  -> recomputed           : {tc.compose(st).apply(orig)}   "
          f"{'MATCH' if tc.compose(st).apply(orig)==si else '*** MISMATCH ***'}")
    print(f"\n  INDEPENDENT read straight from the points (ignore the words above):")
    print(f"    original -> correct_image : {describe_net(tc.recover_map(orig, ci))}")
    print(f"    original -> student_image : {describe_net(tc.recover_map(orig, si))}")
    print(f"\n  label in file   : {lab}")
    print(f"  diagnose()      : {tc.diagnose(orig, ct, st)}   "
          f"{'(agrees)' if tc.diagnose(orig, ct, st)==lab else '*** DISAGREES ***'}")
    print(f"  is_correct      : {rec['is_correct']}   "
          f"{'(ok)' if bool(rec['is_correct'])==(lab=='correct') else '*** WRONG ***'}")

def render(rec: dict, path: str) -> None:
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6,6))
    def draw(pts, color, ls, lab):
        xs=[p[0] for p in pts]+[pts[0][0]]; ys=[p[1] for p in pts]+[pts[0][1]]
        ax.plot(xs, ys, color=color, lw=2.2, ls=ls, label=lab)
        ax.fill(xs, ys, color=color, alpha=0.12)
    draw([tuple(p) for p in rec["original"]],      "#666666", "--", "original")
    draw([tuple(p) for p in rec["correct_image"]], "#2ca02c", "-",  "correct image")
    draw([tuple(p) for p in rec["student_image"]], "#2c6fbb", "-",  "student image")
    ax.set_xlim(-10,10); ax.set_ylim(-10,10); ax.set_aspect("equal")
    ax.set_xticks(range(-10,11)); ax.set_yticks(range(-10,11))
    ax.grid(True,color="#dddddd",lw=.6); ax.axhline(0,color="k",lw=1.2); ax.axvline(0,color="k",lw=1.2)
    ax.tick_params(labelsize=7); ax.legend(loc="upper right", fontsize=8)
    ax.set_title(f"id {rec['id']}: {rec['label']}", fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("ids", nargs="*", type=int)
    ap.add_argument("--label"); ap.add_argument("-n", type=int, default=3)
    ap.add_argument("--render", metavar="DIR")
    a = ap.parse_args()
    recs = [json.loads(l) for l in open(a.data)]
    by_id = {r["id"]: r for r in recs}
    chosen = [by_id[i] for i in a.ids] if a.ids else \
             [r for r in recs if (a.label is None or r["label"]==a.label)][:a.n]
    for r in chosen:
        inspect(r)
        if a.render:
            import os; os.makedirs(a.render, exist_ok=True)
            p = os.path.join(a.render, f"inspect_{r['id']:06d}.png"); render(r, p)
            print(f"  rendered -> {p}")

if __name__ == "__main__":
    main()
