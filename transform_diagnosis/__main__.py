"""Command-line entrypoint: ``python3 -m transform_diagnosis``.

Generates the balanced student-error diagnosis dataset (JSONL) and renders PNGs.

Examples::

    python3 -m transform_diagnosis --seed 0 --n 400 --out data_out
    python3 -m transform_diagnosis --out data_out --min-count 40 --split 0.8,0.1,0.1
    python3 -m transform_diagnosis --out data_out --no-render
"""

from __future__ import annotations

import argparse
import json
import os

from . import dataset, transform_core as tc


def _parse_split(text: str):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split needs three comma-separated fractions, e.g. 0.8,0.1,0.1")
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("--split fractions must sum to a positive number")
    return tuple(p / total for p in parts)  # normalized


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python3 -m transform_diagnosis", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0, help="master seed (default: 0)")
    ap.add_argument("--n", type=int, default=400,
                    help="target total record count (default: 400); floored by min-count*labels")
    ap.add_argument("--out", default="transform_diagnosis_data",
                    help="output directory for JSONL + renders (default: transform_diagnosis_data)")
    ap.add_argument("--render", metavar="DIR", default="renders",
                    help="render subdirectory name under --out (default: renders)")
    ap.add_argument("--split", type=_parse_split, default=(0.8, 0.1, 0.1),
                    help="train,val,test fractions (default: 0.8,0.1,0.1)")
    ap.add_argument("--min-count", type=int, default=30, dest="min_count",
                    help="minimum records per diagnosis label (default: 30)")
    ap.add_argument("--no-render", action="store_true", help="skip PNG rendering")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    render_subdir = os.path.basename(args.render.rstrip("/")) or "renders"

    result = dataset.generate(
        out_dir=args.out,
        seed=args.seed,
        n=args.n,
        min_count=args.min_count,
        split_fracs=args.split,
        render_subdir=render_subdir,
        do_render=not args.no_render,
    )

    records = result["records"]
    summary = {
        "seed": args.seed,
        "requested_n": args.n,
        "min_count": args.min_count,
        "total_records": len(records),
        "split_fractions": list(args.split),
        "label_counts": result["label_counts"],
        "split_counts": result["split_counts"],
        "targets": result["targets"],
        "images_rendered_now": result["rendered"],
        "out_dir": os.path.abspath(args.out),
        "render_subdir": render_subdir,
        "rendered": not args.no_render,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print(f"[transform_diagnosis] wrote {len(records)} records to {os.path.abspath(args.out)}")
    print(f"  data.jsonl + train/val/test.jsonl; splits: {result['split_counts']}")
    print("  label counts:")
    for label in tc.DIAGNOSIS_LABELS:
        print(f"    {label:34} {result['label_counts'][label]}")
    if args.no_render:
        print("  rendering skipped (--no-render)")
    else:
        print(f"  images: {result['rendered']} newly rendered under {render_subdir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
