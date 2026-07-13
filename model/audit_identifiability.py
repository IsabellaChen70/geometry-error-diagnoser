#!/usr/bin/env python3
"""audit_identifiability.py — LABEL IDENTIFIABILITY / AMBIGUITY AUDIT.

Question this answers
---------------------
The generator injected a specific error into each record and recorded it as the
``label``. But a model only ever sees the *geometry*: the RED pre-image, the GREEN
correct image, and the BLUE student answer (either rendered, or as coordinate lists).
How much of a model's accuracy ceiling is set by labels that are **not recoverable in
principle** from that geometry (genuine label ambiguity), versus by **model limitation**
(the information is there but the model can't extract it)?

What "identifiable" means here (the definition we implement)
-----------------------------------------------------------
The oracle label is ``transform_core.diagnose(original, correct_transform,
student_transform)`` — a *total, deterministic function of the two NET affine maps*
``C = compose(correct_transform)`` (RED -> GREEN) and ``S = compose(student_transform)``
(RED -> BLUE). It does not depend on how each map is decomposed into two steps, nor on
which injector produced it. So the label is recoverable from the observation iff the
observation pins down *enough* about ``(C, S)`` to make ``diagnose`` return a single value.

We therefore reconstruct the observation and enumerate every explanation of it:

* ``cand_C`` = every lattice isometry ``T`` with ``T(RED) == GREEN`` (as a vertex SET).
* ``cand_S`` = every lattice isometry ``T`` with ``T(RED) == BLUE`` (as a vertex SET).

Every rigid motion in this dataset (rotations by 90/180/270, reflections across x / y /
y=x / y=-x, translations, and their compositions) is a lattice isometry, and the lattice
isometry group is exactly the 8 orthogonal integer linear parts (``transform_core.
ALL_LINEAR_MAPS``, the dihedral group D4) semidirect the integer translations. Enumerating
all 8 linear parts (each with its forced translation) is thus a COMPLETE superset of
anything ANY injector — or any student — could have produced. This is why we enumerate the
isometry group directly instead of replaying the specific ``errors.py`` injectors: it is
injector-independent and exhaustive rather than a heuristic.

We then form the set of labels consistent with the observation::

    consistent_labels = { diagnose(RED, [C'], [S']) for C' in cand_C for S' in cand_S }

* ``len(consistent_labels) == 1``  -> the label is **IDENTIFIABLE** (the geometry forces
  exactly one diagnosis; a perfect reader is always right).
* ``len(consistent_labels) >= 2``  -> the label is **AMBIGUOUS** (two different injected
  errors could have produced this exact observation with different labels; no model can
  tell them apart from the picture/coords alone).

Matching is on the unordered vertex SET (the weakest, most conservative observer: it does
NOT trust the rendered vertex numbering / corresponding coordinate order). This yields a
LOWER bound on identifiability — vertex correspondence can only ever *reduce* ambiguity,
never add it — and we separately report how many ambiguous-by-set records would be resolved
if the observer did trust the numbering (``recovered_by_correspondence``).

Why the answer comes out the way it does
----------------------------------------
For an ``is_asymmetric`` pre-image (trivial lattice-symmetry group), ``cand_C`` and
``cand_S`` are each a singleton, so ``consistent_labels`` is a singleton -> identifiable.
Ambiguity can arise ONLY when RED has a non-trivial lattice symmetry ``g``, because then
``T`` and ``T@g`` map RED to the same image with different orientation/handedness. The
dataset builder asserts ``is_asymmetric(original)`` for every record precisely to kill this
degree of freedom, so on the real splits the audit is expected to report ~100% identifiable.
The tool re-checks ``is_asymmetric`` independently and would flag any record where it failed.

Assumptions & limits
--------------------
* Assumes each stored image is a genuine lattice-isometry image of the stored original
  (true by construction; ``map_recovery_fail`` counts any record where it is not, e.g.
  corrupted data). The 8-linear-part enumeration is complete for THIS transform family
  only (integer grid, 90-degree rotations + axis/diagonal reflections + integer slides).
* Identifiability is information-theoretic: it measures whether the label is a function of
  the observation. It does NOT model a specific model's perception. A record can be fully
  identifiable yet still hard (e.g. distinguishing a rotation from a reflection requires
  reading winding/handedness) — that difficulty is model limitation, not label ambiguity,
  which is exactly the distinction this audit draws.
* The "implied ceiling" treats each record's consistent-label multiplicity as irreducible
  uncertainty for a reader of that single observation (see ``--help`` / code below).

This module is pure geometry + json: no torch, no PIL, no matplotlib. It imports only
``transform_core`` (the single source of truth for the transform math and the oracle).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

# Import path shim mirroring model/eval_tuned_coords.py: run from the repo root, from
# model/, or on the cluster where the package is installed as ``slm_eval``.
HOME = os.path.expanduser("~")
for _cand in (".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:  # cluster package name first (matches eval_tuned_coords.py), then in-repo package.
    from slm_eval import transform_core as tc  # type: ignore
except ModuleNotFoundError:
    from transform_diagnosis import transform_core as tc

DEFAULT_DATA_DIR = os.path.join(HOME, "transform_diagnosis_data")
DEFAULT_SEED = 20260709  # matches eval_tuned_coords.py / probe_coords.py sample seed

# The four labels the model most often collapses together (rotate<->reflect confusions and
# wrong-parameter-within-family confusions). All four hinge on the net linear part, i.e. on
# orientation/handedness — the hardest thing to read off a rendered grid.
CONFUSABLE: Tuple[str, ...] = (
    "reflection_instead_of_rotation",
    "rotation_instead_of_reflection",
    "wrong_rotation_angle",
    "wrong_reflection_line",
)

Point = Tuple[int, int]


# --------------------------------------------------------------------------------------
# Core geometry: enumerate every lattice isometry consistent with an observed shape pair
# --------------------------------------------------------------------------------------

def _as_points(pts: Sequence[Sequence[int]]) -> List[Point]:
    return [(int(x), int(y)) for x, y in pts]


def isometries_mapping(src: Sequence[Sequence[int]],
                       dst: Sequence[Sequence[int]]) -> List[tc.Transform]:
    """Every lattice isometry ``T = (M, t)`` with ``set(T(src)) == set(dst)``.

    Enumerates all 8 orthogonal integer linear parts (``tc.ALL_LINEAR_MAPS`` = the D4
    group). For each ``M`` the translation is forced: if any ``t`` makes the mapped set
    equal ``dst``, then it must line up the two bounding-box min-corners, so
    ``t = min_corner(dst) - min_corner(M @ src)``; we compute that ``t`` and verify the
    full set equality. Returns one ``Transform`` per linear part that works (so a
    non-trivially symmetric ``src`` yields more than one).

    Unordered (vertex SET) matching on purpose: it does not rely on vertex correspondence,
    giving the most conservative identifiability claim.
    """
    src = _as_points(src)
    dst = _as_points(dst)
    out: List[tc.Transform] = []
    if not src or len(src) != len(dst):
        return out
    dst_set = set(dst)
    dmnx = min(x for x, _ in dst)
    dmny = min(y for _, y in dst)
    for m in tc.ALL_LINEAR_MAPS:
        mapped = [tc.mat_vec(m, p) for p in src]
        mmnx = min(x for x, _ in mapped)
        mmny = min(y for _, y in mapped)
        t = (dmnx - mmnx, dmny - mmny)
        if {(x + t[0], y + t[1]) for x, y in mapped} == dst_set:
            out.append(tc.Transform(m, t))
    return out


# --------------------------------------------------------------------------------------
# Per-record audit
# --------------------------------------------------------------------------------------

def audit_record(rec: dict) -> dict:
    """Decide whether ``rec``'s label is uniquely recoverable from its geometry.

    Uses only the observable fields (``original`` / ``correct_image`` / ``student_image``)
    to build the consistent-label set; the stored ``label`` / transforms are used only for
    independent cross-checks (never to decide identifiability).
    """
    red = _as_points(rec["original"])
    green = _as_points(rec["correct_image"])
    blue = _as_points(rec["student_image"])

    cand_c = isometries_mapping(red, green)  # correct net maps consistent with RED->GREEN
    cand_s = isometries_mapping(red, blue)   # student net maps consistent with RED->BLUE

    consistent: set = set()
    for c in cand_c:
        for s in cand_s:
            consistent.add(tc.diagnose(red, [c], [s]))

    map_recovery_fail = not cand_c or not cand_s
    identifiable = (len(consistent) == 1) and not map_recovery_fail

    # --- Independent cross-checks against the stored oracle values (trust, not logic). ---
    stored_label = rec.get("label")
    sanity_ok = None
    if "correct_transform" in rec and "student_transform" in rec:
        stored_c = tc.compose(rec["correct_transform"])
        stored_s = tc.compose(rec["student_transform"])
        recomputed = tc.diagnose(red, rec["correct_transform"], rec["student_transform"])
        sanity_ok = (
            stored_c in cand_c
            and stored_s in cand_s
            and recomputed == stored_label
            and stored_label in consistent
        )

    # Correspondence (ordered) recovery: what a reader who TRUSTS vertex numbering gets.
    c_ord = tc.recover_map(red, green)
    s_ord = tc.recover_map(red, blue)
    ordered_label = (
        tc.diagnose(red, [c_ord], [s_ord]) if c_ord is not None and s_ord is not None else None
    )

    return {
        "id": rec.get("id"),
        "split": rec.get("split"),
        "label": stored_label,
        "identifiable": identifiable,
        "consistent_labels": sorted(consistent),
        "n_consistent_labels": len(consistent),
        "n_candidate_correct_maps": len(cand_c),
        "n_candidate_student_maps": len(cand_s),
        "preimage_asymmetric": tc.is_asymmetric(red),
        "map_recovery_fail": map_recovery_fail,
        "sanity_ok": sanity_ok,
        "ordered_label": ordered_label,
        # ambiguous-by-set but pinned once vertex numbering is trusted:
        "recovered_by_correspondence": (
            (not identifiable) and (not map_recovery_fail) and ordered_label == stored_label
        ),
    }


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def _rate(numer: int, denom: int) -> float:
    return numer / denom if denom else 0.0


def aggregate(rows: Sequence[dict], split: str) -> dict:
    """Roll per-record audits into the reported identifiability statistics."""
    n = len(rows)
    labels = list(tc.DIAGNOSIS_LABELS)

    n_ident = sum(1 for r in rows if r["identifiable"])
    n_ambig = n - n_ident

    # Per-label identifiable rate.
    per_label: Dict[str, dict] = {}
    for lab in labels:
        subset = [r for r in rows if r["label"] == lab]
        ident = sum(1 for r in subset if r["identifiable"])
        per_label[lab] = {
            "n": len(subset),
            "identifiable": ident,
            "ambiguous": len(subset) - ident,
            "identifiable_rate": _rate(ident, len(subset)),
        }

    # Aliasing map: true label -> {other consistent label: count} over ambiguous records.
    # (The "confusion-style map of which labels alias to which".) Empty when nothing aliases.
    alias_map: Dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        if r["identifiable"]:
            continue
        true = r["label"]
        for other in r["consistent_labels"]:
            if other != true:
                alias_map[true][other] += 1

    # Confusable-set collisions: records where >= 2 of the four confusable labels are
    # simultaneously consistent (the specific collapse the model exhibits), plus the
    # unordered-pair breakdown.
    conf_set = set(CONFUSABLE)
    conf_collision_records = 0
    conf_pairs: Counter = Counter()
    for r in rows:
        both = conf_set.intersection(r["consistent_labels"])
        if len(both) >= 2:
            conf_collision_records += 1
            ordered = sorted(both)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    conf_pairs[f"{ordered[i]} ~ {ordered[j]}"] += 1

    # Support = how many records' TRUE label is in the confusable set. This is the
    # "attack surface" a winding/orientation-blind model could lose to; it is NOT a
    # geometric collision (those come from conf_collision_records above).
    conf_support = sum(1 for r in rows if r["label"] in conf_set)

    # Implied ceilings.
    #  * unique : share of records whose observation forces a single label. A model that
    #             must commit to one label is GUARANTEED correct only on these.
    #  * bayes  : mean over records of 1 / n_consistent_labels — the best expected accuracy
    #             of an optimal reader that cannot distinguish the labels consistent with a
    #             single observation. Equals `unique` when every ambiguous record has >= 2
    #             consistent labels. This is the headline "max achievable label accuracy".
    ceiling_unique = _rate(n_ident, n)
    ceiling_bayes = (
        sum(1.0 / r["n_consistent_labels"] for r in rows if r["n_consistent_labels"]) / n
        if n else 0.0
    )

    diagnostics = {
        "preimage_symmetric": sum(1 for r in rows if not r["preimage_asymmetric"]),
        "map_recovery_fail": sum(1 for r in rows if r["map_recovery_fail"]),
        "sanity_violations": sum(1 for r in rows if r["sanity_ok"] is False),
        "sanity_checked": sum(1 for r in rows if r["sanity_ok"] is not None),
        "recovered_by_correspondence": sum(1 for r in rows if r["recovered_by_correspondence"]),
        "max_consistent_labels": max((r["n_consistent_labels"] for r in rows), default=0),
    }

    return {
        "split": split,
        "n": n,
        "n_identifiable": n_ident,
        "n_ambiguous": n_ambig,
        "pct_identifiable": _rate(n_ident, n),
        "pct_ambiguous": _rate(n_ambig, n),
        "implied_ceiling": ceiling_bayes,
        "implied_ceiling_unique": ceiling_unique,
        "implied_ceiling_bayes": ceiling_bayes,
        "per_label": per_label,
        "aliasing_map": {t: dict(c) for t, c in alias_map.items()},
        "confusable_set": {
            "labels": list(CONFUSABLE),
            "collision_records": conf_collision_records,
            "collision_pairs": dict(conf_pairs),
            "support_records": conf_support,
            "support_rate": _rate(conf_support, n),
        },
        "diagnostics": diagnostics,
    }


# --------------------------------------------------------------------------------------
# Reporting (readable table)
# --------------------------------------------------------------------------------------

def format_report(agg: dict) -> str:
    lines: List[str] = []
    n = agg["n"]
    lines.append(f"=== Identifiability audit: split={agg['split']} (n={n}) ===")
    lines.append(
        f"identifiable : {agg['n_identifiable']:>6} / {n}  ({agg['pct_identifiable'] * 100:6.2f}%)"
    )
    lines.append(
        f"ambiguous    : {agg['n_ambiguous']:>6} / {n}  ({agg['pct_ambiguous'] * 100:6.2f}%)"
    )
    lines.append(f"IMPLIED CEILING (bayes-optimal reader): {agg['implied_ceiling_bayes']:.4f}")
    lines.append(f"IMPLIED CEILING (unique-label share) : {agg['implied_ceiling_unique']:.4f}")

    d = agg["diagnostics"]
    lines.append("")
    lines.append("diagnostics:")
    lines.append(f"  pre-image symmetric (should be 0)     : {d['preimage_symmetric']}")
    lines.append(f"  map recovery failures (should be 0)   : {d['map_recovery_fail']}")
    lines.append(
        f"  oracle sanity violations (should be 0): {d['sanity_violations']} "
        f"(checked {d['sanity_checked']}/{n})"
    )
    lines.append(f"  ambiguous-by-set fixed by numbering   : {d['recovered_by_correspondence']}")
    lines.append(f"  max consistent labels on any record   : {d['max_consistent_labels']}")

    lines.append("")
    lines.append("per-label identifiable rate:")
    lines.append(f"  {'label':<34}{'n':>6}{'ident':>7}{'ambig':>7}{'rate':>8}")
    lines.append("  " + "-" * 62)
    for lab in tc.DIAGNOSIS_LABELS:
        s = agg["per_label"][lab]
        lines.append(
            f"  {lab:<34}{s['n']:>6}{s['identifiable']:>7}{s['ambiguous']:>7}"
            f"{s['identifiable_rate'] * 100:>7.1f}%"
        )

    cs = agg["confusable_set"]
    lines.append("")
    lines.append("confusable set {reflection_instead_of_rotation, rotation_instead_of_reflection,")
    lines.append("                wrong_rotation_angle, wrong_reflection_line}:")
    lines.append(
        f"  records where >=2 confusable labels are BOTH consistent (collisions): "
        f"{cs['collision_records']}"
    )
    if cs["collision_pairs"]:
        for pair, cnt in sorted(cs["collision_pairs"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    {pair}: {cnt}")
    lines.append(
        f"  records whose TRUE label is confusable (support/attack-surface): "
        f"{cs['support_records']} / {n}  ({cs['support_rate'] * 100:.1f}%)"
    )

    lines.append("")
    if agg["aliasing_map"]:
        lines.append("aliasing map (true label -> alternative labels consistent with the same "
                     "observation):")
        for true, others in sorted(agg["aliasing_map"].items()):
            rhs = ", ".join(f"{k} x{v}" for k, v in sorted(others.items(), key=lambda kv: -kv[1]))
            lines.append(f"  {true} -> {rhs}")
    else:
        lines.append("aliasing map: none (no label aliases to any other — every label is a "
                     "function of the geometry)")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# IO / driver
# --------------------------------------------------------------------------------------

def load_raw(path: str) -> Dict[int, dict]:
    """Load records from a JSONL file, keyed by id in file order."""
    by_id: Dict[int, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r
    return by_id


def select_ids(by_id: Dict[int, dict], sample: int, seed: int, limit: int) -> List[int]:
    """Identical selection convention to eval_tuned_coords.select_ids so the audit runs on
    the EXACT same ids the eval scored (``--sample 500 --seed 20260709``)."""
    if sample:
        ids = sorted(random.Random(seed).sample(sorted(by_id), min(sample, len(by_id))))
    else:
        ids = list(by_id)
    if limit:
        ids = ids[:limit]
    return ids


def resolve_sources(args) -> List[Tuple[str, str]]:
    """Return a list of (split_name, jsonl_path). ``--files`` wins over ``--splits``/
    ``--data-dir``; each file's split name is its basename without the .jsonl extension."""
    if args.files:
        out = []
        for p in [p for p in args.files.split(",") if p]:
            name = os.path.splitext(os.path.basename(p))[0]
            out.append((name, p))
        return out
    return [(s, os.path.join(args.data_dir, f"{s}.jsonl"))
            for s in args.splits.split(",") if s]


def run_split(name: str, path: str, args) -> Optional[dict]:
    if not os.path.exists(path):
        print(f"[{name}] SKIP — no such file: {path}", flush=True)
        return None
    by_id = load_raw(path)
    ids = select_ids(by_id, args.sample, args.seed, args.limit)
    rows = [audit_record(by_id[i]) for i in ids]
    agg = aggregate(rows, name)
    print(format_report(agg), flush=True)
    print("", flush=True)

    out_path = os.path.join(args.out_dir, f"results_identifiability_{name}.json")
    payload = dict(agg)
    payload["source_file"] = os.path.abspath(path)
    payload["sample"] = args.sample
    payload["seed"] = args.seed
    payload["limit"] = args.limit
    if args.dump_records:
        payload["records"] = rows
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{name}] wrote {out_path}", flush=True)
    return agg


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Label identifiability / ambiguity audit (pure geometry + json).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--splits", default="test,ood",
                    help="comma-separated splits to read from --data-dir (default: test,ood)")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dataset dir with <split>.jsonl (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--files", default="",
                    help="comma-separated JSONL paths to audit directly (each treated as its "
                         "own split named by filename); overrides --splits/--data-dir")
    ap.add_argument("--sample", type=int, default=0,
                    help="deterministically sample N ids per split (0 = all); use with --seed "
                         "to reproduce eval_tuned_coords.py's exact ids")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"sample seed (default: {DEFAULT_SEED}, matches eval_tuned_coords.py)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per split after sampling (0 = no cap); cheap smoke test")
    ap.add_argument("--out-dir", default=".", dest="out_dir",
                    help="directory for results_identifiability_<split>.json (default: .)")
    ap.add_argument("--dump-records", action="store_true", dest="dump_records",
                    help="also embed the per-record audit rows in the JSON output")
    ap.add_argument("--selftest", action="store_true",
                    help="run built-in sanity checks (identifiable + constructed aliasing) "
                         "and exit; no data files needed")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.selftest:
        return selftest()

    os.makedirs(args.out_dir, exist_ok=True)
    sources = resolve_sources(args)
    aggs = []
    for name, path in sources:
        agg = run_split(name, path, args)
        if agg is not None:
            aggs.append(agg)

    if len(aggs) > 1:  # combined line so the ceiling can be read against the eval's test+ood
        total = sum(a["n"] for a in aggs)
        ident = sum(a["n_identifiable"] for a in aggs)
        bayes = (sum(a["implied_ceiling_bayes"] * a["n"] for a in aggs) / total) if total else 0.0
        print(f"=== COMBINED {[a['split'] for a in aggs]} (n={total}) ===", flush=True)
        print(f"identifiable : {ident}/{total} ({_rate(ident, total) * 100:.2f}%)  "
              f"IMPLIED CEILING (bayes): {bayes:.4f}", flush=True)
    if not aggs:
        print("No splits audited (nothing found). Try --files or check --data-dir/--splits.",
              flush=True)
        return 1
    return 0


# --------------------------------------------------------------------------------------
# Self-test: proves the aliasing check is real (not vacuous) without any data files.
# --------------------------------------------------------------------------------------

def _make_record(original, correct_seq, student_seq, label=None) -> dict:
    """Build a record dict (observable + oracle fields) from transform sequences, exactly
    as dataset.py would compute the images."""
    correct_text = [tc.describe_transform(t) for t in correct_seq]
    student_text = [tc.describe_transform(t) for t in student_seq]
    return {
        "id": 0,
        "split": "selftest",
        "original": [list(p) for p in original],
        "correct_transform": correct_text,
        "correct_image": [list(p) for p in tc.compose(correct_seq).apply(original)],
        "student_transform": student_text,
        "student_image": [list(p) for p in tc.compose(student_seq).apply(original)],
        "label": label if label is not None
        else tc.diagnose(original, correct_seq, student_seq),
    }


def selftest() -> int:
    print("audit_identifiability selftest\n" + "-" * 32)

    # (1) IDENTIFIABLE: an asymmetric pre-image forces a unique net map for GREEN and BLUE,
    #     so exactly one label is consistent with the observation.
    asym = [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2)]
    assert tc.is_asymmetric(asym), "fixture must be asymmetric"
    rec_ok = _make_record(asym, [tc.rotate(90, "ccw")], [tc.rotate(180, "ccw")])
    a_ok = audit_record(rec_ok)
    print(f"[identifiable case] label={rec_ok['label']} identifiable={a_ok['identifiable']} "
          f"consistent={a_ok['consistent_labels']}")
    assert a_ok["identifiable"] is True
    assert a_ok["consistent_labels"] == [rec_ok["label"]]
    assert a_ok["n_candidate_correct_maps"] == 1 and a_ok["n_candidate_student_maps"] == 1
    assert a_ok["sanity_ok"] is True

    # (2) AMBIGUOUS: a MIRROR-SYMMETRIC pre-image (invariant under reflect y). Then GREEN and
    #     BLUE are each reachable by BOTH a rotation and a reflection, so the whole confusable
    #     set becomes consistent with the single observation -> must be flagged ambiguous.
    sym = [(-2, 0), (2, 0), (3, 2), (0, 4), (-3, 2)]
    assert not tc.is_asymmetric(sym), "fixture must be symmetric to induce aliasing"
    rec_amb = _make_record(sym, [tc.rotate(90, "ccw")], [tc.rotate(180, "ccw")])
    a_amb = audit_record(rec_amb)
    print(f"[ambiguous case]   label={rec_amb['label']} identifiable={a_amb['identifiable']} "
          f"consistent={a_amb['consistent_labels']}")
    assert a_amb["identifiable"] is False
    assert set(a_amb["consistent_labels"]) == set(CONFUSABLE), a_amb["consistent_labels"]
    assert rec_amb["label"] in a_amb["consistent_labels"]
    assert a_amb["sanity_ok"] is True  # stored maps/label are still among the candidates

    # (3) Aggregation surfaces both: 1 identifiable, 1 ambiguous, one confusable collision.
    agg = aggregate([a_ok, a_amb], "selftest")
    assert agg["n_identifiable"] == 1 and agg["n_ambiguous"] == 1
    assert agg["confusable_set"]["collision_records"] == 1
    assert agg["diagnostics"]["preimage_symmetric"] == 1
    assert agg["aliasing_map"], "ambiguous record must produce an aliasing entry"
    print("\n" + format_report(agg))
    print("\nselftest OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
