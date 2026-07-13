"""make_v4_data.py — assemble the v4 STRUCTURED-CoT training set (train_v4_cot_chat.jsonl).

v4 mixes three record pools, ALL emitted with the structured chain-of-thought target
(reasoning trace + an operation-type check line + the final JSON, whose scored keys
``label``/``correct_transform``/``hint`` are unchanged and are followed by the structured
type fields ``expected_operation_types`` / ``student_operation_types`` / ``main_mismatch``):

  * NORMAL       (~50%) : ordinary two-step records, sampled from an EXISTING oracle split
                          (default ``<data-dir>/train.jsonl``) so they REUSE renders already
                          on disk -- no re-render, no image upload.
  * CONTRASTIVE  (~30%) : hard matched quadruplets (transform_diagnosis.contrastive) -- one
                          shared RED + shared translation, the four confusable labels.
  * CURRICULUM   (~20%) : easy single-step transforms with a single-step error.

The mix ratio is a CLI knob (``--mix normal,contrastive,curriculum``). Contrastive +
curriculum records are NEW, so they are finalized (asserted like every dataset record) and
their images rendered under ``<out-dir>/<render-subdir>/`` (default ``renders_v4/``), with
the chat JSONL image paths matching the rendered files. A matching ``val_v4_cot_chat.jsonl``
is emitted for in-training eval-loss monitoring.

No GPU, no model, no image decode for the JSONL itself. Rendering needs matplotlib (guarded;
run with ``--no-render`` to produce JSONL only, then render separately).

Examples:
  # cluster: full set next to the existing data (reuses ~/…/renders for NORMAL)
  python make_v4_data.py --n 9600 --val-n 400
  # local smoke: tiny, reuse the committed sample as the NORMAL source, render a few
  python make_v4_data.py --data-dir dataset_sample --normal-source dataset_sample/train_sample.jsonl \
      --out-dir /tmp/v4 --n 20 --val-n 8 --max-render 4 --print 2
  python make_v4_data.py --n 9600 --val-n 400 --no-render   # JSONL only; render later
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HOME = os.path.expanduser("~")

# Import resolution: this dir + repo root first so it runs from any cwd; prefer the in-repo
# ``transform_diagnosis`` (the full generator package -- contrastive/dataset/render live only
# there, and are synced to ~/transform_diagnosis on the cluster), fall back to ``slm_eval``.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), ".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from transform_diagnosis import chat_format, contrastive, cot, dataset, eval as ev
except ModuleNotFoundError:  # cluster fallback (full package also synced under slm_eval)
    from slm_eval import chat_format, contrastive, cot, dataset, eval as ev

DEFAULT_DATA_DIR = os.path.join(HOME, "transform_diagnosis_data")

# Deterministic per-pool RNG salts (train vs val kept disjoint so the val monitor does not
# leak from train). XORed with the master seed.
_SALT = {
    ("normal", "train"): 0x1111, ("contrastive", "train"): 0x2222, ("curriculum", "train"): 0x3333,
    ("normal", "val"): 0x4444, ("contrastive", "val"): 0x5555, ("curriculum", "val"): 0x6666,
    ("mix", "train"): 0x7777, ("mix", "val"): 0x8888,
}
# New records get ids in a high, split-specific range so they never collide with the reused
# NORMAL ids (0..N) in the mixed file, and train/val renders never share a filename.
_ID_BASE = {"train": 1_000_000, "val": 2_000_000}
_RENDER_SUBDIR = {"train": "renders_v4", "val": "renders_v4_val"}


def load_records(path, limit=0):
    recs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                recs.append(json.loads(line))
                if limit and len(recs) >= limit:
                    break
    return recs


def _atomic_write_lines(path, lines):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(lines))
    os.replace(tmp, path)


def parse_mix(text):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--mix needs three comma-separated fractions, e.g. 0.5,0.3,0.2")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("--mix fractions must sum to a positive number")
    return tuple(p / total for p in parts)  # normalized


def plan_counts(n, mix):
    """Split ``n`` into (normal, contrastive, curriculum). Contrastive is rounded DOWN to a
    multiple of 4 (whole quadruplets); normal absorbs the remainder so the total is exactly n."""
    _, f_con, f_cur = mix
    n_con = (round(f_con * n) // 4) * 4
    n_cur = round(f_cur * n)
    n_norm = n - n_con - n_cur
    if n_norm < 0:  # extreme ratios: shrink curriculum first, then contrastive
        n_cur = max(0, n_cur + n_norm)
        n_norm = n - n_con - n_cur
    if n_norm < 0:
        n_con = max(0, (n_con + n_norm) // 4 * 4)
        n_norm = n - n_con - n_cur
    return max(0, n_norm), n_con, n_cur


def build_split(split, n, mix, seed, normal_source, out_dir):
    """Build one split's full records (finalized) + the pool it came from, ready for
    conversation building + rendering. Returns (records, new_records, counts)."""
    n_norm, n_con, n_cur = plan_counts(n, mix)

    # NORMAL: sample from an existing oracle split (already finalized; reuse its renders).
    src = load_records(normal_source)
    src.sort(key=lambda r: r.get("id", 0))
    if not src:
        raise SystemExit(f"no normal-source records in {normal_source}")
    rng_norm = random.Random(seed ^ _SALT[("normal", split)])
    k = min(n_norm, len(src))
    if k < n_norm:
        print(f"[{split}] WARNING: normal-source has only {len(src)} records; using {k} "
              f"(requested {n_norm}).", flush=True)
    normal_records = rng_norm.sample(src, k) if k < len(src) else list(src)

    # CONTRASTIVE + CURRICULUM: NEW partials -> finalize with high, split-specific ids.
    rng_con = random.Random(seed ^ _SALT[("contrastive", split)])
    con_partials, _groups = contrastive.build_contrastive_partials(rng_con, n_con // 4)
    rng_cur = random.Random(seed ^ _SALT[("curriculum", split)])
    cur_partials = contrastive.build_curriculum_partials(rng_cur, n_cur)

    subdir = _RENDER_SUBDIR[split]
    rid = _ID_BASE[split]
    new_records = []
    for p in con_partials + cur_partials:
        new_records.append(dataset.finalize_record(p, rid, split, subdir))
        rid += 1

    records = list(normal_records) + new_records
    counts = {"normal": len(normal_records), "contrastive": len(con_partials),
              "curriculum": len(cur_partials), "total": len(records)}
    return records, new_records, counts


def build_and_verify_rows(records, split):
    """Structured-CoT conversation rows for ``records`` (order preserved), each verified to
    round-trip: the final JSON's scored keys parse back to gold, the trace concludes with the
    label, and the structured fields are present + consistent with the oracle."""
    rows = []
    for rec in records:
        conv = cot.to_cot_conversation(rec, structured=True)
        target = conv["messages"][1]["content"][0]["text"]
        parsed = ev.parse_pred(target)
        if parsed is None:
            raise SystemExit(f"v4 target for id={rec.get('id')} did not parse")
        scored = {k: parsed.get(k) for k in ("label", "correct_transform", "hint")}
        gold = {"label": rec["label"], "correct_transform": rec["correct_transform"], "hint": rec["hint"]}
        if scored != gold:
            raise SystemExit(f"v4 target id={rec.get('id')} scored keys != gold:\n {scored}\n {gold}")
        if f"the diagnosis is {rec['label']}" not in target:
            raise SystemExit(f"v4 trace id={rec.get('id')} does not conclude with label {rec['label']!r}")
        # structured fields present + consistent with the oracle transforms
        if parsed.get("expected_operation_types") != cot.operation_types(rec["correct_transform"]):
            raise SystemExit(f"v4 target id={rec.get('id')} expected_operation_types inconsistent")
        if parsed.get("student_operation_types") != cot.operation_types(rec["student_transform"]):
            raise SystemExit(f"v4 target id={rec.get('id')} student_operation_types inconsistent")
        if parsed.get("main_mismatch") != cot.main_mismatch(rec):
            raise SystemExit(f"v4 target id={rec.get('id')} main_mismatch inconsistent")
        rows.append(conv)
    return rows


def label_counts(records):
    from collections import Counter
    return dict(Counter(r["label"] for r in records))


def _print_examples(records, n):
    print(f"\n===== {n} example v4 structured-CoT target(s) =====", flush=True)
    for rec in records[:n]:
        print(f"\n--- id={rec.get('id')}  split={rec.get('split')}  label={rec['label']} ---")
        print(cot.cot_target(rec, structured=True))
    print("=" * 56, flush=True)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Assemble the v4 structured-CoT training set.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dataset dir (image base + default source/out; default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--out-dir", default=None, dest="out_dir",
                    help="dir to write *_v4_cot_chat.jsonl + new renders (default: --data-dir)")
    ap.add_argument("--normal-source", default=None, dest="normal_source",
                    help="oracle JSONL for the NORMAL pool (default: <data-dir>/train.jsonl)")
    ap.add_argument("--normal-val-source", default=None, dest="normal_val_source",
                    help="oracle JSONL for the NORMAL pool of val (default: <data-dir>/val.jsonl)")
    ap.add_argument("--n", type=int, default=9600, help="target train record count (default: 9600)")
    ap.add_argument("--val-n", type=int, default=400, dest="val_n",
                    help="target val record count for monitoring (0 disables; default: 400)")
    ap.add_argument("--mix", type=parse_mix, default=(0.5, 0.3, 0.2),
                    help="normal,contrastive,curriculum fractions (default: 0.5,0.3,0.2)")
    ap.add_argument("--seed", type=int, default=20260711, help="master seed (default: 20260711)")
    ap.add_argument("--no-render", action="store_true", help="skip PNG rendering of new records")
    ap.add_argument("--max-render", type=int, default=0, dest="max_render",
                    help="render at most N new images per split (0 = all; smoke-test convenience)")
    ap.add_argument("--print", type=int, default=0, dest="print_n",
                    help="print the first N built structured-CoT targets")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + verify (+ --print) but write NO files and render nothing")
    return ap


def _render_new(new_records, out_dir, max_render):
    try:  # lazy: only import matplotlib (via render) when actually rendering
        from transform_diagnosis import render as render_mod
    except ModuleNotFoundError:
        from slm_eval import render as render_mod
    to_render = new_records if not max_render else new_records[:max_render]
    made = render_mod.render_all(to_render, out_dir, skip_existing=True)
    return made, len(to_render)


def process_split(split, n, args, out_dir, normal_source):
    records, new_records, counts = build_split(split, n, args.mix, args.seed, normal_source, out_dir)
    rows = build_and_verify_rows(records, split)

    # Deterministic interleave so the file mixes the three pools.
    order = list(range(len(rows)))
    random.Random(args.seed ^ _SALT[("mix", split)]).shuffle(order)
    rows = [rows[i] for i in order]

    out_name = f"{split}_v4_cot_chat.jsonl"
    out_path = os.path.join(out_dir, out_name)
    print(f"[{split}] n={counts['total']}  normal={counts['normal']} "
          f"contrastive={counts['contrastive']} curriculum={counts['curriculum']}", flush=True)
    print(f"[{split}] label counts: {label_counts(records)}", flush=True)

    if args.print_n and split == "train":
        # Show at least one contrastive + one curriculum example (they are the NEW records).
        _print_examples(new_records + records, args.print_n)

    if args.dry_run:
        print(f"[{split}] DRY RUN — {len(rows)} rows built + verified, not written.", flush=True)
        return
    os.makedirs(out_dir, exist_ok=True)
    _atomic_write_lines(out_path, [json.dumps(r, ensure_ascii=False) + "\n" for r in rows])
    print(f"[{split}] wrote {len(rows)} rows -> {out_path}", flush=True)

    if args.no_render:
        print(f"[{split}] rendering skipped (--no-render); {len(new_records)} new renders pending "
              f"under {_RENDER_SUBDIR[split]}/.", flush=True)
    elif new_records:
        made, attempted = _render_new(new_records, out_dir, args.max_render)
        print(f"[{split}] rendered {made}/{attempted} new images under {_RENDER_SUBDIR[split]}/ "
              f"(of {len(new_records)} total new records).", flush=True)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = args.out_dir or args.data_dir
    normal_source = args.normal_source or os.path.join(args.data_dir, "train.jsonl")
    normal_val_source = args.normal_val_source or os.path.join(args.data_dir, "val.jsonl")

    if not os.path.exists(normal_source):
        raise SystemExit(f"normal source not found: {normal_source}")

    print(f"v4 datagen: seed={args.seed} mix(normal,contrastive,curriculum)={tuple(round(m,3) for m in args.mix)} "
          f"out={out_dir}", flush=True)
    process_split("train", args.n, args, out_dir, normal_source)
    if args.val_n:
        if not os.path.exists(normal_val_source):
            print(f"[val] normal-val-source {normal_val_source} missing — reusing train source.", flush=True)
            normal_val_source = normal_source
        process_split("val", args.val_n, args, out_dir, normal_val_source)
    print("\nDONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
