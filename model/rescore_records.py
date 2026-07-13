"""rescore_records.py — re-score saved per-record eval outputs with the FIXED metrics.

Generalizes ``rescore_probe.py`` to ARBITRARY record JSONLs written by the eval scripts
(``records_tuned_test.jsonl``, ``records_tuned_ood.jsonl``, ``records_tuned_coords_*.jsonl``,
the v1/v2 tuned runs whose ``transform_match``/``hint_match`` were computed with the OLD
broken metrics, ...). No model or GPU is used: every row already stores ``raw_model_output``,
so we re-grade it against its oracle record with the current ``transform_diagnosis.eval`` and
write, next to each input, ``<name>_rescored.jsonl`` + the matching
``results_..._rescored.json`` aggregate, printing a before/after aggregate table per file.

Oracle transforms/hints come from the dataset split JSONL (``<data-dir>/<split>.jsonl``);
the split is read from each row (``eval.RECORD_FIELDS`` carries it). If a split file is
absent the records are rebuilt deterministically from the recorded seed/config (no
rendering), exactly as ``rescore_probe.py`` does.

Run locally (no GPU, no API key):
  python model/rescore_records.py records_tuned_test.jsonl records_tuned_ood.jsonl
  python model/rescore_records.py records_probe_coords.jsonl        # reproduces probe deltas
  python model/rescore_records.py records_tuned_coords_test.jsonl --data-dir ~/transform_diagnosis_data
  python model/rescore_records.py records_frontier_v6_opus_image_n150_test.jsonl \
      --task full --data-dir ~/transform_diagnosis_data

``--task`` overrides the scoring mode for every input row. Without it, a saved
``task_mode`` is reused when available, then the evaluator's legacy/v6 inference applies.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
HOME = os.path.expanduser("~")

# Import the fixed harness. Prefer the in-repo package; fall back to the cluster's synced
# copy (``slm_eval``), mirroring the eval scripts, so this runs in $HOME too.
for _cand in (_ROOT, ".", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from transform_diagnosis import eval as ev
except ModuleNotFoundError:
    from slm_eval import eval as ev

# Dataset config the records were produced from (see transform_diagnosis_data/summary.json).
# Used ONLY for the deterministic rebuild fallback when a split JSONL is absent.
GEN = dict(seed=0, n=24000, min_count=30, ood_per_label=500)
TASK_MODES = ("legacy", "correct", "student", "both", "full")

_TABLE_KEYS = (
    ("parse_rate", "parse_rate"),
    ("label_accuracy", "label_acc"),
    ("balanced_accuracy", "balanced_acc"),
    ("correct_net_match_rate", "correct_net"),
    ("student_net_match_rate", "student_net"),
    ("both_nets_match_rate", "both_nets"),
    ("derived_label_accuracy", "derived_label"),
    ("hint_match_rate", "hint_match"),
    ("hint_exact_match_rate", "hint_exact*"),
)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _default_data_dir():
    for c in ("transform_diagnosis_data",
              os.path.join(_ROOT, "transform_diagnosis_data"),
              os.path.join(HOME, "transform_diagnosis_data")):
        if os.path.isdir(c):
            return c
    return "transform_diagnosis_data"


def _rebuilt_oracle():
    """Deterministically rebuild every record (no rendering) from the recorded config."""
    from transform_diagnosis import dataset
    recs, _ = dataset.build_records(
        GEN["seed"], GEN["n"], GEN["min_count"], ood_per_label=GEN["ood_per_label"]
    )
    return {r["id"]: r for r in recs}


def oracle_by_id(rows, data_dir):
    """Map id -> full oracle record for every id in ``rows``.

    Loads only the split JSONLs the rows reference (from each row's ``split``); for any id
    still unresolved (missing split file, or a row with no split) it falls back once to a
    deterministic rebuild from the recorded seed/config.
    """
    by_id = {}
    for split in sorted({r.get("split") for r in rows if r.get("split")}):
        path = os.path.join(data_dir, f"{split}.jsonl")
        if os.path.exists(path):
            for r in _load_jsonl(path):
                by_id[r["id"]] = r
    need = {r["id"] for r in rows} - set(by_id)
    if need:
        by_id.update({i: r for i, r in _rebuilt_oracle().items() if i in need})
    return by_id


def out_paths(records_path):
    """``records_X.jsonl`` -> (``results_X_rescored.json``, ``records_X_rescored.jsonl``),
    mirroring rescore_probe.py's naming so this is a drop-in generalization."""
    d = os.path.dirname(records_path)
    stem = os.path.basename(records_path)
    if stem.endswith(".jsonl"):
        stem = stem[:-len(".jsonl")]
    rec_out = os.path.join(d, f"{stem}_rescored.jsonl")
    res_stem = "results" + stem[len("records"):] if stem.startswith("records") else stem
    res_out = os.path.join(d, f"{res_stem}_rescored.json")
    return res_out, rec_out


def _fmt(agg, key):
    v = agg.get(key)
    return f"{v:.3f}" if isinstance(v, float) else "  n/a"


def _row_task_mode(row, task_mode):
    """Use an explicit override, else preserve the mode saved with the eval row."""
    if task_mode is not None:
        return task_mode
    stored = row.get("task_mode")
    return stored if stored in TASK_MODES else None


def rescore_file(records_path, data_dir, task_mode=None):
    """Re-score one record JSONL: write the *_rescored files and print a before/after
    table. ``task_mode`` overrides saved per-row modes when provided. Returns
    ``(before_agg, after_agg)``. No model is called."""
    rows = _load_jsonl(records_path)
    if not rows:
        raise SystemExit(f"no records in {records_path}")

    oracle = oracle_by_id(rows, data_dir)
    missing = sorted({r["id"] for r in rows} - set(oracle))
    if missing:
        head = missing[:10]
        raise SystemExit(f"oracle records missing for ids {head}"
                         f"{'...' if len(missing) > 10 else ''}; pass --data-dir")

    # "before" = the stored (old-metric) booleans re-aggregated; "after" = re-graded now.
    before = ev.aggregate(rows)
    rescored = [
        ev.score_record(
            r["raw_model_output"],
            oracle[r["id"]],
            task_mode=_row_task_mode(r, task_mode),
        )
        for r in rows
    ]
    after = ev.aggregate(rescored)

    res_out, rec_out = out_paths(records_path)
    ev.save_results(after, res_out, rescored, rec_out)

    print(f"\n=== {os.path.basename(records_path)}  (n={after['n']}, no model called) ===")
    print(f"{'metric':<18}{'before':>9}{'after':>9}{'delta':>9}")
    print("-" * 45)
    for key, label in _TABLE_KEYS:
        b, a = before.get(key), after.get(key)
        delta = f"{a - b:+.3f}" if isinstance(a, float) and isinstance(b, float) else "    -"
        print(f"{label:<18}{_fmt(before, key):>9}{_fmt(after, key):>9}{delta:>9}")
    print("* hint_exact_match_rate = strict, exploratory-only (exact canonical tokens).")

    # How many rows the fixed metrics flipped from fail->pass (the recovered credit).
    prev = {r["id"]: r for r in rows}
    for field in (
        "parse_ok",
        "label_ok",
        "correct_net_ok",
        "student_net_ok",
        "both_nets_ok",
        "derived_label_ok",
        "hint_ok",
    ):
        flipped = 0
        for row in rescored:
            old = prev[row["id"]]
            prior = old.get(field)
            if field == "correct_net_ok" and field not in old:
                prior = old.get("transform_ok")
            if row.get(field) and not prior:
                flipped += 1
        if flipped:
            print(f"  {field}: {flipped} row(s) fail->pass under the fixed metrics")
    print(f"wrote {os.path.relpath(res_out)} and {os.path.relpath(rec_out)}")
    return before, after


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Offline re-scorer: re-grade saved eval records with the fixed metrics.")
    ap.add_argument("records", nargs="+",
                    help="one or more records_*.jsonl written by the eval scripts")
    ap.add_argument("--data-dir", default=_default_data_dir(), dest="data_dir",
                    help="dir holding the <split>.jsonl oracle files (default: first of "
                         "./transform_diagnosis_data, repo root, ~/transform_diagnosis_data)")
    ap.add_argument("--task", choices=TASK_MODES,
                    help="override scoring task for all rows (for v6 full records, pass "
                         "'--task full'); default: reuse each saved task_mode, then infer")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    for path in args.records:
        if not os.path.exists(path):
            raise SystemExit(f"records file not found: {path}")
        rescore_file(path, args.data_dir, task_mode=args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
