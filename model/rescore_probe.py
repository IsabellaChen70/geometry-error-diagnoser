"""rescore_probe.py — re-score the saved coordinate-probe responses with the FIXED metrics.

No model is called. We read the raw model responses already saved in
``records_probe_coords.jsonl`` and re-grade each one against its oracle record with the
current (fixed) ``transform_diagnosis.eval`` code, then write a NEW aggregate
(``results_probe_coords_rescored.json``) and per-record file
(``records_probe_coords_rescored.jsonl``). The originals are never overwritten.

Oracle transforms (``correct_transform`` / ``student_transform``) are not stored in the
probe JSONL, so we recover them from the dataset the probe ran on
(``transform_diagnosis_data/val.jsonl``, seed 0). If that file is absent we rebuild the
records deterministically from the same seed/config recorded in its ``summary.json``.

Run locally:  python model/rescore_probe.py
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from transform_diagnosis import eval as ev  # noqa: E402

PROBE_RECORDS = os.path.join(_ROOT, "records_probe_coords.jsonl")
PROBE_RESULTS = os.path.join(_ROOT, "results_probe_coords.json")
VAL_JSONL = os.path.join(_ROOT, "transform_diagnosis_data", "val.jsonl")

OUT_RESULTS = os.path.join(_ROOT, "results_probe_coords_rescored.json")
OUT_RECORDS = os.path.join(_ROOT, "records_probe_coords_rescored.jsonl")

# Dataset config the probe ran on (see transform_diagnosis_data/summary.json).
GEN = dict(seed=0, n=24000, min_count=30, ood_per_label=500)


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _oracle_by_id():
    """Map id -> full oracle record for every id the probe touched."""
    if os.path.exists(VAL_JSONL):
        return {r["id"]: r for r in _load_jsonl(VAL_JSONL)}
    # Fallback: rebuild deterministically from the recorded seed/config (no rendering).
    from transform_diagnosis import dataset
    recs, _ = dataset.build_records(
        GEN["seed"], GEN["n"], GEN["min_count"], ood_per_label=GEN["ood_per_label"]
    )
    return {r["id"]: r for r in recs}


def main() -> int:
    probe_rows = _load_jsonl(PROBE_RECORDS)
    oracle = _oracle_by_id()

    missing = [r["id"] for r in probe_rows if r["id"] not in oracle]
    if missing:
        raise SystemExit(f"oracle records missing for ids {missing}; cannot re-score")

    rescored = [ev.score_record(r["raw_model_output"], oracle[r["id"]]) for r in probe_rows]
    agg = ev.aggregate(rescored)
    ev.save_results(agg, OUT_RESULTS, rescored, OUT_RECORDS)

    before = json.load(open(PROBE_RESULTS)) if os.path.exists(PROBE_RESULTS) else {}

    def _fmt(d, k):
        return f"{d.get(k, 0.0):.3f}" if k in d else "  n/a"

    keys = [
        ("parse_rate", "parse_rate"),
        ("label_accuracy", "label_acc"),
        ("balanced_accuracy", "balanced_acc"),
        ("transform_match_rate", "transform_match"),
        ("hint_match_rate", "hint_match"),
        ("hint_exact_match_rate", "hint_exact*"),
    ]
    print(f"Re-scored {agg['n']} saved probe responses (no model called).\n")
    print(f"{'metric':<18}{'before':>9}{'after':>9}{'delta':>9}")
    print("-" * 45)
    for key, label in keys:
        b = before.get(key)
        a = agg.get(key)
        delta = f"{a - b:+.3f}" if (isinstance(a, float) and isinstance(b, float)) else "    -"
        bs = f"{b:.3f}" if isinstance(b, float) else "  n/a"
        as_ = f"{a:.3f}" if isinstance(a, float) else "  n/a"
        print(f"{label:<18}{bs:>9}{as_:>9}{delta:>9}")
    print("\n* hint_exact_match_rate = strict, exploratory-only (exact canonical tokens).")

    # Show the transform rows that the parser fix recovered.
    before_rows = {r["id"]: r for r in probe_rows}
    flipped = [r for r in rescored
               if not before_rows[r["id"]]["transform_ok"] and r["transform_ok"]]
    if flipped:
        print(f"\ntransform_ok recovered by the parser fix ({len(flipped)}):")
        for r in flipped:
            print(f"  id={r['id']:<6} {r['true_label']}")
    still_false = [r for r in rescored if not r["transform_ok"]]
    if still_false:
        print(f"\ntransform_ok still false ({len(still_false)}) — genuine, not brittleness:")
        for r in still_false:
            print(f"  id={r['id']:<6} true={r['true_label']} pred={r['pred_label']}")

    print(f"\nwrote {os.path.relpath(OUT_RESULTS, _ROOT)} and "
          f"{os.path.relpath(OUT_RECORDS, _ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
