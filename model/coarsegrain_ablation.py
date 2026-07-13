"""coarsegrain_ablation.py — POST-HOC "coarse-grain" ablation of a fine-grained
geometry-misconception classifier. No model, no GPU, no retraining.

The vision fine-tune predicts one of the 8 fine-grained diagnosis labels
(``transform_core.DIAGNOSIS_LABELS``). This script re-reads eval records that were
ALREADY saved (``records_*.jsonl`` written by the eval scripts / re-scorers), collapses
both the TRUE and the PREDICTED label into a small set of COARSE buckets, and reports how
well the model does at that easier task. It is the "when the task is simplified, the 4B
does much better" ablation, computed offline from what we already have on disk.

The coarse taxonomy (8 fine labels -> 3 coarse classes)
-------------------------------------------------------
    correct           <- correct
    wrong_type        <- reflection_instead_of_rotation, rotation_instead_of_reflection
                         (the model chose the wrong OPERATION TYPE / orientation:
                          rotation vs reflection)
    wrong_parameter   <- wrong_rotation_angle, wrong_reflection_line,
                         wrong_translation, opposite_translation
                         (right operation type, wrong PARAMETER: angle / line / vector)

``completely_wrong`` is special: per :func:`transform_core.diagnose` it is only ever
emitted when the net LINEAR part differs AND the translation ALSO differs — i.e. it is
never a single-parameter slip; it is at minimum a structural (type-level) error stacked
with a wrong translation. So the two defensible options are:

  * ``3class`` (default, "strict 3-class"): fold ``completely_wrong`` into ``wrong_type``.
    Rationale above — it always carries a wrong linear/structural part, so it belongs with
    the type errors rather than the parameter errors, and it keeps the headline at a clean
    3 classes ("did it get the operation right, the parameters right, or neither").
  * ``4class``: keep ``completely_wrong`` as its own fourth bucket, for readers who want to
    see the "everything wrong" mass reported separately.

The single knob is :data:`FINE_TO_COARSE_BASE` plus :data:`COMPLETELY_WRONG_BUCKET`; edit
those to change the taxonomy. Both variants are cheap, so BOTH are always computed and
written to the JSON summary; ``--mapping`` only selects which one is printed as the
headline table.

What it reports (per coarse variant)
------------------------------------
  * overall accuracy   — fraction of records whose coarse pred == coarse true
  * balanced accuracy  — mean per-coarse-class RECALL over the classes actually present
                         (so it is comparable to eval.py's balanced_accuracy on OOD, which
                          only carries a subset of labels)
  * per-class recall + support, and a coarse confusion matrix (with a ``PARSE_FAIL``
    prediction column for outputs that never parsed to a known fine label — those are
    always misses, exactly as in eval.py).

Where the labels come from
--------------------------
Each ``records_*.jsonl`` row (see ``eval.RECORD_FIELDS``) stores ``true_label`` and the
already-parsed ``pred_label`` ("PARSE_FAIL" when the output didn't parse to a known label).
We use those directly; if a row lacks ``pred_label`` we fall back to re-parsing its
``raw_model_output`` with the SAME parser eval uses (``eval.parse_pred``) — we never
re-implement parsing.

Pure Python / json (no torch / PIL). Reuses ``transform_diagnosis.eval`` only for the
label-parsing helper, via the same import-fallback as ``model/eval_tuned_coords.py`` so it
runs both in-repo (``transform_diagnosis``) and on the cluster (synced ``slm_eval``).

Usage
-----
  # on the cluster, against the v3cot records you already have:
  python coarsegrain_ablation.py records_v3cot_test.jsonl records_v3cot_ood.jsonl
  python coarsegrain_ablation.py records_v4_test.jsonl --mapping 4class
  python coarsegrain_ablation.py records_tuned_s500_test.jsonl --out my_summary.json

Writes ``results_<tag>_coarse.json`` next to each input (``records_<tag>.jsonl`` ->
``results_<tag>_coarse.json``) unless ``--out`` is given (single input only).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
HOME = os.path.expanduser("~")

# Resolve the label-parsing helper the SAME way model/eval_tuned_coords.py does: prefer the
# cluster package name (``slm_eval``), fall back to the in-repo ``transform_diagnosis``.
# The extra candidates (repo root, HOME) just make the import work from any CWD.
for _cand in (".", "..", _ROOT, HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from slm_eval import eval as ev
except ModuleNotFoundError:
    from transform_diagnosis import eval as ev

# --------------------------------------------------------------------------------------
# The coarse taxonomy — THE knob. Change these two constants to change the buckets.
# --------------------------------------------------------------------------------------

# The 8 fine labels (order == transform_core.DIAGNOSIS_LABELS; kept explicit here so this
# script is self-contained and the mapping is obvious at a glance).
FINE_LABELS: Tuple[str, ...] = (
    "correct",
    "reflection_instead_of_rotation",
    "rotation_instead_of_reflection",
    "wrong_rotation_angle",
    "wrong_reflection_line",
    "wrong_translation",
    "opposite_translation",
    "completely_wrong",
)

# Fine -> coarse for the 7 unambiguous labels. ``completely_wrong`` is added per-variant.
FINE_TO_COARSE_BASE: Dict[str, str] = {
    "correct": "correct",
    # operation-TYPE errors (rotation vs reflection: the orientation is wrong)
    "reflection_instead_of_rotation": "wrong_type",
    "rotation_instead_of_reflection": "wrong_type",
    # right type, wrong PARAMETER (angle / line / translation vector)
    "wrong_rotation_angle": "wrong_parameter",
    "wrong_reflection_line": "wrong_parameter",
    "wrong_translation": "wrong_parameter",
    "opposite_translation": "wrong_parameter",
}

# How ``completely_wrong`` is bucketed in each variant (see module docstring for why).
COMPLETELY_WRONG_BUCKET: Dict[str, str] = {
    "3class": "wrong_type",         # strict 3-class: fold into the type-error bucket
    "4class": "completely_wrong",   # keep as its own fourth class
}

# Class display order per variant (drives table columns / rows).
COARSE_CLASSES: Dict[str, List[str]] = {
    "3class": ["correct", "wrong_type", "wrong_parameter"],
    "4class": ["correct", "wrong_type", "wrong_parameter", "completely_wrong"],
}

MAPPINGS = tuple(COARSE_CLASSES)  # ("3class", "4class")

# Predicted "label" for an output that never parsed to a known fine label. Mirrors
# eval._PARSE_FAIL: it is a prediction column that can never match a true class (always a miss).
PARSE_FAIL = "PARSE_FAIL"

# Fail loudly if the mapping ever drifts out of sync with the 8-label contract.
assert set(FINE_TO_COARSE_BASE) | {"completely_wrong"} == set(FINE_LABELS), (
    "FINE_TO_COARSE_BASE must cover exactly the 8 fine labels (minus completely_wrong)."
)
for _m in MAPPINGS:
    assert COMPLETELY_WRONG_BUCKET[_m] in COARSE_CLASSES[_m], _m


def build_mapping(mapping: str) -> Tuple[Dict[str, str], List[str]]:
    """Return ``(fine_label -> coarse_class, ordered_class_list)`` for ``mapping``."""
    if mapping not in COARSE_CLASSES:
        raise ValueError(f"unknown mapping {mapping!r}; choose from {list(MAPPINGS)}")
    fine_to_coarse = dict(FINE_TO_COARSE_BASE)
    fine_to_coarse["completely_wrong"] = COMPLETELY_WRONG_BUCKET[mapping]
    return fine_to_coarse, list(COARSE_CLASSES[mapping])


# --------------------------------------------------------------------------------------
# Label extraction from a saved record row
# --------------------------------------------------------------------------------------

def extract_labels(row: dict) -> Tuple[str, str]:
    """Pull ``(true_fine_label, pred_fine_label)`` out of one saved record row.

    True label: ``true_label`` (records schema), falling back to ``label`` (oracle schema).
    Predicted label: the already-parsed ``pred_label`` when present (this includes the
    sentinel ``"PARSE_FAIL"``); otherwise re-parse ``raw_model_output`` with the SAME helper
    eval uses (``eval.parse_pred``) and take its ``label``, or ``PARSE_FAIL`` if that yields
    no string label. We never re-implement parsing here.
    """
    true_label = row.get("true_label", row.get("label"))

    pred = row.get("pred_label")
    if isinstance(pred, str) and pred:
        return true_label, pred

    raw = row.get("raw_model_output")
    if raw is not None:
        parsed = ev.parse_pred(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("label"), str):
            return true_label, parsed["label"]
    return true_label, PARSE_FAIL


def to_coarse(fine_label: Optional[str], fine_to_coarse: Dict[str, str]) -> str:
    """Coarse bucket for a fine label; any unknown/parse-fail prediction -> ``PARSE_FAIL``."""
    return fine_to_coarse.get(fine_label, PARSE_FAIL)


# --------------------------------------------------------------------------------------
# Coarse confusion + metrics
# --------------------------------------------------------------------------------------

def coarse_confusion(
    rows: Sequence[dict], mapping: str
) -> Tuple[Dict[str, Counter], List[str], int, Counter]:
    """Build the coarse ``confusion[true][pred]`` matrix for ``rows`` under ``mapping``.

    Returns ``(confusion, classes, n_parse_fail, unknown_true)`` where ``confusion`` keys
    are true coarse classes and each ``Counter`` counts predicted coarse classes (a
    predicted ``PARSE_FAIL`` column captures unparsed/unknown outputs). ``unknown_true``
    counts rows whose TRUE label isn't a known fine label (should be empty for real data;
    such rows are skipped and surfaced rather than silently mis-bucketed).
    """
    fine_to_coarse, classes = build_mapping(mapping)
    confusion: Dict[str, Counter] = {c: Counter() for c in classes}
    n_parse_fail = 0
    unknown_true: Counter = Counter()

    for row in rows:
        true_fine, pred_fine = extract_labels(row)
        if true_fine not in fine_to_coarse:
            unknown_true[true_fine] += 1
            continue
        true_coarse = fine_to_coarse[true_fine]
        pred_coarse = to_coarse(pred_fine, fine_to_coarse)
        if pred_coarse == PARSE_FAIL:
            n_parse_fail += 1
        confusion[true_coarse][pred_coarse] += 1

    return confusion, classes, n_parse_fail, unknown_true


def metrics_from_confusion(
    confusion: Dict[str, Counter], classes: Sequence[str]
) -> dict:
    """Overall accuracy, balanced accuracy, per-class recall + support from a confusion matrix.

    ``balanced_accuracy`` is the mean per-class recall over ONLY the classes present
    (support > 0), matching ``eval.aggregate``'s definition so coarse numbers stay
    comparable to the fine-grained ones (important on OOD, which lacks some classes).
    """
    total = sum(sum(c.values()) for c in confusion.values())
    n_correct = sum(confusion[c].get(c, 0) for c in classes)
    overall = n_correct / total if total else 0.0

    per_class_recall: Dict[str, Optional[float]] = {}
    support: Dict[str, int] = {}
    present_recalls: List[float] = []
    for c in classes:
        row_total = sum(confusion[c].values())
        support[c] = row_total
        if row_total:
            recall = confusion[c].get(c, 0) / row_total
            per_class_recall[c] = recall
            present_recalls.append(recall)
        else:
            per_class_recall[c] = None
    balanced = sum(present_recalls) / len(present_recalls) if present_recalls else 0.0

    return {
        "n": total,
        "overall_accuracy": overall,
        "balanced_accuracy": balanced,
        "per_class_recall": per_class_recall,
        "support": support,
    }


def summarize_variant(rows: Sequence[dict], mapping: str) -> dict:
    """Full coarse summary for one variant: mapping, metrics, and the confusion matrix."""
    confusion, classes, n_parse_fail, unknown_true = coarse_confusion(rows, mapping)
    fine_to_coarse, _ = build_mapping(mapping)
    metrics = metrics_from_confusion(confusion, classes)
    return {
        "classes": classes,
        "completely_wrong_bucket": COMPLETELY_WRONG_BUCKET[mapping],
        "label_to_coarse": fine_to_coarse,
        "n_parse_fail": n_parse_fail,
        "unknown_true_labels": dict(unknown_true),
        # confusion as plain nested dicts (JSON-friendly); PARSE_FAIL column kept if present.
        "confusion": {t: dict(c) for t, c in confusion.items()},
        **metrics,
    }


def summarize_file(rows: Sequence[dict], selected_mapping: str, input_path: str) -> dict:
    """Compute BOTH coarse variants for one records file (both are cheap) and package the
    JSON summary; ``selected_mapping`` is recorded as the printed headline."""
    variants = {m: summarize_variant(rows, m) for m in MAPPINGS}
    fine_true = Counter()
    for row in rows:
        true_fine, _ = extract_labels(row)
        fine_true[true_fine] += 1
    return {
        "input": input_path,
        "n": len(rows),
        "selected_mapping": selected_mapping,
        "fine_true_distribution": dict(fine_true),
        "variants": variants,
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    return "  n/a" if x is None else f"{x:.3f}"


def format_confusion(variant: dict) -> str:
    """Render the coarse confusion matrix as an aligned grid with a numeric legend.

    Rows are true coarse classes; columns are the coarse classes plus ``PF`` (PARSE_FAIL)
    when any unparsed prediction exists. Mirrors ``eval.format_confusion``'s compact style.
    """
    classes = variant["classes"]
    confusion = variant["confusion"]
    has_pf = any(PARSE_FAIL in confusion.get(c, {}) for c in classes)
    cols = list(classes) + ([PARSE_FAIL] if has_pf else [])

    codes = {c: str(i) for i, c in enumerate(classes)}
    codes[PARSE_FAIL] = "PF"
    legend = "  ".join(f"{codes[c]}={c}" for c in classes) + ("   PF=parse_fail" if has_pf else "")

    width = max(5, *(len(codes[c]) + 1 for c in cols))
    header = "true\\pred".ljust(14) + "".join(codes[c].rjust(width) for c in cols)
    lines = [legend, "", header]
    for c in classes:
        cells = "".join(str(confusion.get(c, {}).get(col, 0)).rjust(width) for col in cols)
        lines.append(codes[c].ljust(14) + cells)
    return "\n".join(lines)


def format_report(summary: dict) -> str:
    """A readable, self-contained text report for one records file."""
    selected = summary["selected_mapping"]
    variant = summary["variants"][selected]
    other = next(m for m in MAPPINGS if m != selected)
    other_v = summary["variants"][other]

    lines: List[str] = []
    lines.append(f"=== {os.path.basename(summary['input'])}  "
                 f"(n={summary['n']}, mapping={selected}) ===")
    lines.append(f"completely_wrong -> {variant['completely_wrong_bucket']}")
    lines.append("fine -> coarse: "
                 "correct->correct | {reflection_instead_of_rotation, "
                 "rotation_instead_of_reflection}->wrong_type | "
                 "{wrong_rotation_angle, wrong_reflection_line, wrong_translation, "
                 "opposite_translation}->wrong_parameter")
    lines.append("")
    lines.append("coarse metrics")
    lines.append(f"  overall_accuracy   {variant['overall_accuracy']:.3f}")
    lines.append(f"  balanced_accuracy  {variant['balanced_accuracy']:.3f}  "
                 f"(mean recall over present classes)")
    if variant["n_parse_fail"]:
        lines.append(f"  parse_fail preds   {variant['n_parse_fail']} "
                     f"(counted as misses)")
    lines.append("")
    lines.append("  per-class recall (support):")
    for c in variant["classes"]:
        recall = variant["per_class_recall"][c]
        support = variant["support"][c]
        lines.append(f"    {c:<16}{_fmt_pct(recall):>7}   (support {support})")
    lines.append("")
    lines.append("confusion (true rows x pred cols):")
    lines.append(format_confusion(variant))
    lines.append("")
    lines.append(f"[{other}] overall_accuracy={other_v['overall_accuracy']:.3f}  "
                 f"balanced_accuracy={other_v['balanced_accuracy']:.3f}  "
                 f"(completely_wrong -> {other_v['completely_wrong_bucket']})")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------
# I/O + CLI
# --------------------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def out_path(records_path: str) -> str:
    """``records_<tag>.jsonl`` -> ``results_<tag>_coarse.json`` (next to the input).

    Non ``records_``-prefixed names get a ``<stem>_coarse.json`` sibling.
    """
    d = os.path.dirname(records_path)
    stem = os.path.basename(records_path)
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    if stem.startswith("records"):
        stem = "results" + stem[len("records"):]
    return os.path.join(d, f"{stem}_coarse.json")


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Post-hoc coarse-grain ablation of the fine-grained diagnosis "
                    "classifier (offline; no model / GPU).")
    ap.add_argument("records", nargs="+",
                    help="one or more saved records_*.jsonl (e.g. records_v3cot_test.jsonl)")
    ap.add_argument("--mapping", choices=list(MAPPINGS), default="3class",
                    help="which coarse variant to PRINT as the headline (both are always "
                         "written to the JSON summary). 3class folds completely_wrong into "
                         "wrong_type; 4class keeps it separate. (default: 3class)")
    ap.add_argument("--out", default=None,
                    help="explicit output JSON path (single input only); default writes "
                         "results_<tag>_coarse.json next to each input")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if args.out is not None and len(args.records) != 1:
        raise SystemExit("--out is only valid with a single records file; omit it to write "
                         "results_<tag>_coarse.json next to each input.")

    for path in args.records:
        if not os.path.exists(path):
            raise SystemExit(f"records file not found: {path}")
        rows = load_jsonl(path)
        if not rows:
            raise SystemExit(f"no records in {path}")

        summary = summarize_file(rows, args.mapping, path)
        dest = args.out if args.out is not None else out_path(path)
        with open(dest, "w") as f:
            json.dump(summary, f, indent=2)

        print(format_report(summary))
        print(f"wrote {os.path.relpath(dest)}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
