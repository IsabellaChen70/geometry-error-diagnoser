"""make_cot_data.py — generate the chain-of-thought training file ``train_cot_chat.jsonl``.

For CoT fine-tuning the assistant target becomes a short reasoning trace followed by the
SAME final JSON the non-CoT training uses (see ``transform_diagnosis.cot``). This script
reads the FULL oracle records for a split (``<data-dir>/<split>.jsonl`` — which carry
``correct_transform`` / ``student_transform`` / ``label``), builds one CoT conversation per
record (image + instruction UNCHANGED; assistant = trace + JSON), and writes
``<out-dir>/<split>_cot_chat.jsonl``.

No GPU, no model, no image decode — the trace is derived deterministically from the stored
ground truth via ``transform_core``. Cheap enough to run on a login node.

Only TRAIN needs traces (the fine-tune learns to reason); val/test/ood stay as-is and are
scored on the final JSON only. A small ``val`` CoT file is optionally emitted for
in-training validation monitoring (``--splits train,val``); it does NOT touch the frozen
``val.jsonl`` / ``val_chat.jsonl``.

Examples:
  python make_cot_data.py                                  # ~/…/train.jsonl -> train_cot_chat.jsonl
  python make_cot_data.py --splits train,val              # also val_cot_chat.jsonl (monitoring)
  python make_cot_data.py --data-dir ~/transform_diagnosis_data --out-dir ~/transform_diagnosis_data
  python make_cot_data.py --input dataset_sample/train_sample.jsonl --print 3 --dry-run  # local smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HOME = os.path.expanduser("~")

# Import resolution (mirrors the eval scripts): this dir + repo root first so the harness
# resolves from any cwd; prefer the cluster package name ``slm_eval`` and fall back to the
# in-repo ``transform_diagnosis`` so it also runs from the repo locally.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), ".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from slm_eval import cot, eval as ev
except ModuleNotFoundError:
    from transform_diagnosis import cot, eval as ev

DEFAULT_DATA_DIR = os.path.join(HOME, "transform_diagnosis_data")


def load_records(path, limit=0):
    """Load full oracle records from a JSONL (one per line)."""
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


def cot_out_name(input_basename):
    """``train.jsonl`` -> ``train_cot_chat.jsonl`` (stem + ``_cot_chat.jsonl``)."""
    stem = input_basename
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return f"{stem}_cot_chat.jsonl"


def build_and_write(in_path, out_dir, *, limit=0, print_n=0, dry_run=False):
    """Build CoT rows for one input JSONL and (unless --dry-run) write the ``*_cot_chat``.

    Returns (n_records, out_path_or_None). Verifies each built target round-trips back to
    the gold JSON via the eval parser (fails loudly on any inconsistency)."""
    recs = load_records(in_path, limit=limit)
    if not recs:
        raise SystemExit(f"no records in {in_path}")

    rows = []
    for rec in recs:
        conv = cot.to_cot_conversation(rec)
        # Safety: the built target's FINAL JSON must parse back to the exact gold object,
        # and the concluded label must match the record (never contradict the ground truth).
        target = conv["messages"][1]["content"][0]["text"]
        parsed = ev.parse_pred(target)
        gold = {"label": rec["label"], "correct_transform": rec["correct_transform"],
                "hint": rec["hint"]}
        if parsed != gold:
            raise SystemExit(f"CoT target for id={rec.get('id')} does not round-trip to gold "
                             f"JSON:\n parsed={parsed}\n gold={gold}")
        if f"the diagnosis is {rec['label']}" not in target:
            raise SystemExit(f"CoT trace for id={rec.get('id')} does not conclude with its "
                             f"label {rec['label']!r}")
        rows.append(json.dumps(conv, ensure_ascii=False) + "\n")

    if print_n:
        _print_examples(recs, min(print_n, len(recs)))

    out_path = None
    if not dry_run:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, cot_out_name(os.path.basename(in_path)))
        _atomic_write_lines(out_path, rows)
    return len(rows), out_path


def _print_examples(recs, n):
    print(f"\n===== {n} example CoT target(s) (trace + final JSON) =====", flush=True)
    for rec in recs[:n]:
        print(f"\n--- id={rec.get('id')}  label={rec['label']} ---")
        print(cot.cot_target(rec))
    print("=" * 56, flush=True)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Generate <split>_cot_chat.jsonl for CoT fine-tuning.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dir with <split>.jsonl oracle files (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--out-dir", default=None, dest="out_dir",
                    help="dir to write <split>_cot_chat.jsonl (default: same as --data-dir)")
    ap.add_argument("--splits", default="train",
                    help="comma-separated splits to convert (default: train). Only TRAIN needs "
                         "traces; add 'val' to emit a val CoT file for in-training monitoring.")
    ap.add_argument("--input", default=None,
                    help="explicit input JSONL (overrides --splits); writes <stem>_cot_chat.jsonl")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per input (0 = all); cheap smoke test")
    ap.add_argument("--print", type=int, default=0, dest="print_n",
                    help="print the first N built (trace + JSON) targets to stdout")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + verify (+ --print) but write NO files")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = args.out_dir or args.data_dir

    if args.input:
        inputs = [args.input]
    else:
        inputs = [os.path.join(args.data_dir, f"{s}.jsonl")
                  for s in (s for s in args.splits.split(",") if s)]

    for in_path in inputs:
        if not os.path.exists(in_path):
            raise SystemExit(f"input not found: {in_path}")
        n, out_path = build_and_write(in_path, out_dir, limit=args.limit,
                                      print_n=args.print_n, dry_run=args.dry_run)
        where = "(dry-run, not written)" if args.dry_run else out_path
        print(f"[make_cot_data] {n} records from {os.path.relpath(in_path) if os.path.exists(in_path) else in_path} -> {where}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
