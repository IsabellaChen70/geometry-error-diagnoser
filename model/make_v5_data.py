"""make_v5_data.py — assemble the v5 ENUM-CoT training set (train_v5_cot_chat.jsonl).

v5 historically targeted ordered-step recovery. CoT fine-tuning (v3cot) lifted LABEL
accuracy a lot, but exact step readout stayed floored even when the model was handed exact
coordinates. The v5 hypothesis was to reframe ordered-step recovery
recovery from a free-text GENERATION problem into a CLASSIFICATION over a SMALL DISCRETE
vocabulary (:mod:`transform_diagnosis.enum_transform`). The current evaluator keeps semantic
net-map equality as the comparable ``transform_match`` headline and reports v5's stricter
ordered result separately as ``step_sequence_exact_match_rate``.

v5 holds the v4 DATA FIXED and changes ONLY the transform target representation, so the
reframing is the single variable (a clean ablation). Concretely it REUSES v4's assembler
(``make_v4_data.build_split`` — same seed, salts, ids, render subdirs, and the same three
pools NORMAL / CONTRASTIVE / CURRICULUM), so every image, split and id is byte-identical to
v4; only the assistant target differs:

  * ``correct_transform`` is emitted as the DISCRETE enum schema
    (``{"type","param"}`` for rotations/reflections, ``{"type","dx","dy"}`` for translations)
    instead of prose, built deterministically from the oracle transforms.
  * the reasoning trace gains a "transform readout" line naming the RED->GREEN step
    types/params in the enum vocabulary.
  * every v4 field is kept (label, hint, expected/student operation types, main_mismatch).

Curriculum knobs (transform-first emphasis; neither forced):
  * ``--mix normal,contrastive,curriculum`` upweights the transform-DIVERSE pools
    (contrastive quadruplets vary rotate/reflect/angle/line; curriculum is single-step).
  * ``--transform-first`` foregrounds the classification target by emitting
    ``correct_transform`` FIRST in the JSON (scored by key, so ordering is safe).

Renders: NORMAL reuses the existing ``~/…/renders`` (from the base dataset); the NEW
contrastive+curriculum records render under v4's subdirs (identical geometry -> idempotent,
a no-op if v4 already generated them). No GPU / no image decode for the JSONL itself.

Examples:
  # cluster: full set next to the existing data (reuses ~/…/renders for NORMAL)
  python make_v5_data.py --n 9600 --val-n 400
  # local smoke: tiny, reuse the committed sample as the NORMAL source, render a few
  python make_v5_data.py --data-dir dataset_sample --normal-source dataset_sample/train_sample.jsonl \
      --out-dir /tmp/v5 --n 20 --val-n 8 --max-render 4 --print 2
  python make_v5_data.py --n 9600 --val-n 400 --transform-first --no-render  # JSONL only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HOME = os.path.expanduser("~")

# Import resolution: this dir + repo root first so it runs from any cwd; prefer the in-repo
# ``transform_diagnosis`` (the full generator package), fall back to ``slm_eval``. The v4
# assembler is imported as a sibling script (same dir on the cluster and in model/ locally).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), ".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from transform_diagnosis import cot, enum_transform as et, eval as ev, transform_core as tc
except ModuleNotFoundError:  # cluster fallback (full package also synced under slm_eval)
    from slm_eval import cot, enum_transform as et, eval as ev, transform_core as tc

# Reuse the v4 assembler VERBATIM so v5 data == v4 data (same seed/salts/ids/renders/pools);
# only the assistant TARGET differs. Importing (not re-implementing) is what guarantees it.
from make_v4_data import (  # noqa: E402  (after sys.path setup)
    DEFAULT_DATA_DIR,
    build_split,
    label_counts,
    parse_mix,
    _atomic_write_lines,
    _render_new,
    _RENDER_SUBDIR,
)


def build_and_verify_rows(records, split, transform_first):
    """v5 enum-CoT conversation rows for ``records`` (order preserved), each verified to
    round-trip end-to-end:

      * the final JSON parses; label + hint equal the oracle; ``correct_transform`` equals
        the deterministic enum of the oracle transform (``enum_transform.seq_enum``);
      * the enum is LOSS-LESS — it composes back to the same net map as the oracle prose;
      * the harness scores parse/label/transform/hint ALL True against the ORACLE (prose)
        record (exercises the real eval path: enum prediction vs prose gold);
      * the trace concludes with the label; every v4 structured field is present + consistent.
    """
    rows = []
    for rec in records:
        conv = cot.to_cot_conversation(rec, enum_transform=True, transform_first=transform_first)
        target = conv["messages"][1]["content"][0]["text"]
        parsed = ev.parse_pred(target)
        if parsed is None:
            raise SystemExit(f"v5 target for id={rec.get('id')} did not parse")

        gold_enum = et.seq_enum(rec["correct_transform"])
        if parsed.get("label") != rec["label"]:
            raise SystemExit(f"v5 target id={rec.get('id')} label != gold")
        if parsed.get("hint") != rec["hint"]:
            raise SystemExit(f"v5 target id={rec.get('id')} hint != gold")
        if parsed.get("correct_transform") != gold_enum:
            raise SystemExit(f"v5 target id={rec.get('id')} enum correct_transform != gold enum:\n"
                             f" {parsed.get('correct_transform')}\n {gold_enum}")
        # enum must reconstruct the SAME net map as the oracle prose transform (loss-less).
        if tc.compose(et.enum_to_transforms(parsed["correct_transform"])) != tc.compose(rec["correct_transform"]):
            raise SystemExit(f"v5 target id={rec.get('id')} enum does not round-trip to the oracle net map")
        # Full harness score against the ORACLE (prose gold): the real v5 eval path.
        row = ev.score_record(target, rec)
        if not (row["parse_ok"] and row["label_ok"] and row["transform_ok"] and row["hint_ok"]):
            raise SystemExit(f"v5 target id={rec.get('id')} did not score all-pass: {row['failure_reason']!r}")
        if f"the diagnosis is {rec['label']}" not in target:
            raise SystemExit(f"v5 trace id={rec.get('id')} does not conclude with label {rec['label']!r}")
        # v4 structured fields present + consistent with the oracle transforms.
        if parsed.get("expected_operation_types") != cot.operation_types(rec["correct_transform"]):
            raise SystemExit(f"v5 target id={rec.get('id')} expected_operation_types inconsistent")
        if parsed.get("student_operation_types") != cot.operation_types(rec["student_transform"]):
            raise SystemExit(f"v5 target id={rec.get('id')} student_operation_types inconsistent")
        if parsed.get("main_mismatch") != cot.main_mismatch(rec):
            raise SystemExit(f"v5 target id={rec.get('id')} main_mismatch inconsistent")
        rows.append(conv)
    return rows


def _print_examples(records, n, transform_first):
    print(f"\n===== {n} example v5 enum-CoT target(s) =====", flush=True)
    for rec in records[:n]:
        print(f"\n--- id={rec.get('id')}  split={rec.get('split')}  label={rec['label']} ---")
        print(cot.cot_target(rec, enum_transform=True, transform_first=transform_first))
    print("=" * 56, flush=True)


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Assemble the v5 enum-CoT training set.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dataset dir (image base + default source/out; default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--out-dir", default=None, dest="out_dir",
                    help="dir to write *_v5_cot_chat.jsonl + new renders (default: --data-dir)")
    ap.add_argument("--normal-source", default=None, dest="normal_source",
                    help="oracle JSONL for the NORMAL pool (default: <data-dir>/train.jsonl)")
    ap.add_argument("--normal-val-source", default=None, dest="normal_val_source",
                    help="oracle JSONL for the NORMAL pool of val (default: <data-dir>/val.jsonl)")
    ap.add_argument("--n", type=int, default=9600, help="target train record count (default: 9600)")
    ap.add_argument("--val-n", type=int, default=400, dest="val_n",
                    help="target val record count for monitoring (0 disables; default: 400)")
    ap.add_argument("--mix", type=parse_mix, default=(0.5, 0.3, 0.2),
                    help="normal,contrastive,curriculum fractions (default: 0.5,0.3,0.2). Raise "
                         "contrastive/curriculum to upweight transform-DIVERSE examples.")
    ap.add_argument("--transform-first", action="store_true", dest="transform_first",
                    help="curriculum knob: emit correct_transform FIRST in the JSON to foreground "
                         "the classification target (scored by key; ordering is safe). Off by default.")
    ap.add_argument("--seed", type=int, default=20260711,
                    help="master seed (default: 20260711; MATCH v4 to reuse its exact renders)")
    ap.add_argument("--no-render", action="store_true", help="skip PNG rendering of new records")
    ap.add_argument("--max-render", type=int, default=0, dest="max_render",
                    help="render at most N new images per split (0 = all; smoke-test convenience)")
    ap.add_argument("--print", type=int, default=0, dest="print_n",
                    help="print the first N built enum-CoT targets")
    ap.add_argument("--dry-run", action="store_true",
                    help="build + verify (+ --print) but write NO files and render nothing")
    return ap


def process_split(split, n, args, out_dir, normal_source):
    records, new_records, counts = build_split(split, n, args.mix, args.seed, normal_source, out_dir)
    rows = build_and_verify_rows(records, split, args.transform_first)

    # Deterministic interleave so the file mixes the three pools (reuse v4's mix salt so the
    # ORDER matches v4 too; only the target text differs).
    from make_v4_data import _SALT
    order = list(range(len(rows)))
    random.Random(args.seed ^ _SALT[("mix", split)]).shuffle(order)
    rows = [rows[i] for i in order]

    out_name = f"{split}_v5_cot_chat.jsonl"
    out_path = os.path.join(out_dir, out_name)
    print(f"[{split}] n={counts['total']}  normal={counts['normal']} "
          f"contrastive={counts['contrastive']} curriculum={counts['curriculum']}  "
          f"transform_first={args.transform_first}", flush=True)
    print(f"[{split}] label counts: {label_counts(records)}", flush=True)

    if args.print_n and split == "train":
        _print_examples(new_records + records, args.print_n, args.transform_first)

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
              f"(of {len(new_records)} total new records; identical to v4, so 0 if v4 ran).", flush=True)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    out_dir = args.out_dir or args.data_dir
    normal_source = args.normal_source or os.path.join(args.data_dir, "train.jsonl")
    normal_val_source = args.normal_val_source or os.path.join(args.data_dir, "val.jsonl")

    if not os.path.exists(normal_source):
        raise SystemExit(f"normal source not found: {normal_source}")

    print(f"v5 datagen: seed={args.seed} mix(normal,contrastive,curriculum)={tuple(round(m,3) for m in args.mix)} "
          f"transform_first={args.transform_first} out={out_dir}", flush=True)
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
