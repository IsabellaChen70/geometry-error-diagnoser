"""eval — programmatic, judge-free scoring of model diagnoses against the oracle.

Legacy fine-tunes emit ``{"label", "correct_transform", "hint"}``; v6 emits one or both
canonical ``correct_net`` / ``student_net`` objects and optionally label + hint. Because
every record carries a ground-truth answer verified by :mod:`transform_core`, grading is
fully deterministic — no LLM-as-judge or hand-labeling.

    parse_ok      a syntactically valid/recoverable JSON object. For historical
                  comparability only, legacy v1-v5 tasks still classify a missing or
                  unknown required label as a parse failure; v6 scores it as a field error.
    label_ok      direct predicted label == oracle label, when requested
    transform_ok  predicted RED->GREEN NET affine map equals the oracle map. This is the
                  original v1-v4 semantic metric and remains the comparable headline for
                  every version, including v5 enum sequences and v6 canonical net JSON.
    student_net_ok / both_nets_ok score the second observable map when requested
    derived_label_ok scores transform_core.diagnose(predicted maps), independent of the
                  model's direct label token
    step_sequence_exact_ok is the optional stricter ordered-decomposition diagnostic
    hint_ok       hint references the right OPERATION FAMILY for the error
                  (rotation/reflection/translation, per hints.expected_hint_families)
                  AND leaks no coordinates it wasn't sanctioned to state

``hint_ok`` deliberately measures what the model is actually instructed to do — nudge at
the right operation "WITHOUT stating the correct coordinates" (see chat_format) — so a
correct, instruction-following hint can pass. The older, stricter check (the exact
canonical schema strings from hints.expected_hint_tokens must ALL appear) is kept as a
secondary/exploratory field ``hint_exact_ok`` (and ``hint_exact_match_rate``); it is NOT
part of the headline table because it is unachievable for translation labels, whose exact
token effectively IS the answer.

We report the fraction passing each check plus balanced accuracy and an 8x8 confusion
matrix. This module is pure Python (no torch / PIL) so it unit-tests locally and imports
cleanly inside the Colab notebook.

Split discipline lives in the caller: run this over the frozen ``test`` + ``ood`` splits
for the headline base-vs-tuned number, and over ``val`` for iteration. This module does not
know or care which split it is handed.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from . import enum_transform as et
from . import hints
from . import net_transform as nt
from . import transform_core as tc

# Per-record row schema saved to JSONL for error analysis.
RECORD_FIELDS = (
    "id",
    "split",
    "true_label",
    "pred_label",
    "derived_label",
    "task_mode",
    "parse_ok",
    "label_ok",
    "transform_ok",
    "correct_net_ok",
    "student_net_ok",
    "both_nets_ok",
    "step_sequence_exact_ok",
    "derived_label_ok",
    "hint_ok",
    "hint_exact_ok",
    "raw_model_output",
    "failure_reason",
)

_PARSE_FAIL = "PARSE_FAIL"
# A coordinate pair literal, e.g. "(3, -4)" — used only for residual-leak detection.
_COORD_RE = re.compile(r"\(-?\d+,-?\d+\)")


# --------------------------------------------------------------------------------------
# Prediction parsing
# --------------------------------------------------------------------------------------

def parse_pred(text: str) -> Optional[dict]:
    """Extract the diagnosis JSON object from a model output, or ``None``.

    Tolerates surrounding prose and ```` ``` ```` code fences, AND a chain-of-thought
    reasoning PREFIX before the answer: it tries a direct ``json.loads`` first (the plain
    single-object case), then falls back to the LAST brace-balanced ``{...}`` span that
    parses to an object. Taking the LAST object (not the first) is what lets CoT outputs
    — "reasoning text ... {JSON}" — score correctly: the final JSON is the answer. A plain
    single-object output is unchanged (``json.loads`` matches it directly, first == last).
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        obj = _last_json_object(s)
    return obj if isinstance(obj, dict) else None


def _last_json_object(s: str) -> Optional[dict]:
    """Return the LAST top-level brace-balanced ``{...}`` span that parses to a dict.

    Scans left-to-right tracking brace depth to isolate each top-level ``{...}`` span,
    ``json.loads`` each, and keeps the last one that is a dict. This robustly pulls the
    FINAL JSON object out of a response that begins with a reasoning prefix (and handles a
    fenced or one-line answer at the very end). Returns ``None`` if nothing parses.
    """
    last: Optional[dict] = None
    depth = 0
    start = -1
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start != -1:
                    try:
                        obj = json.loads(s[start : i + 1])
                    except ValueError:
                        obj = None
                    if isinstance(obj, dict):
                        last = obj
                    start = -1
    return last


# --------------------------------------------------------------------------------------
# Field checks
# --------------------------------------------------------------------------------------

def _norm(text: str) -> str:
    """Lowercase + collapse whitespace — for tolerant substring matching."""
    return " ".join(str(text).lower().split())


def _sequence_transforms(value: object) -> List[tc.Transform]:
    """Normalize a legacy prose/enum ordered sequence to canonical step maps."""
    if et.is_enum_seq(value):
        return et.enum_to_transforms(value)  # type: ignore[arg-type]
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"expected an ordered transform sequence, got {type(value).__name__}")
    return [tc.as_transform(step) for step in value]


def _affine_from_value(value: object) -> tc.Transform:
    """Normalize v1-v6 transform representations to one canonical affine map."""
    if isinstance(value, Mapping):
        # A bare dict is a v6 net object.  v5 uses a *list* of step dicts.
        return nt.net_to_affine(value)
    return tc.compose(_sequence_transforms(value))


def _transform_match(pred_transform: object, gold_transform: object) -> bool:
    """Semantic NET-map equality used by the apples-to-apples headline metric.

    v1-v4 prose, v5 enum step sequences, and v6 canonical net objects all
    normalize to :class:`transform_core.Transform` and compare exactly.  Thus two
    different ordered decompositions of the same observable map pass here.
    """
    try:
        return _affine_from_value(pred_transform) == _affine_from_value(gold_transform)
    except (KeyError, TypeError, ValueError):
        pass

    # Preserve the legacy normalized-string fallback for an unparseable prose
    # spelling.  It is intentionally unavailable to structured/v6 objects.
    pred = [pred_transform] if isinstance(pred_transform, str) else pred_transform
    gold = [gold_transform] if isinstance(gold_transform, str) else gold_transform
    if not isinstance(pred, (list, tuple)) or not isinstance(gold, (list, tuple)):
        return False
    if len(pred) != len(gold) or not all(isinstance(x, str) for x in [*pred, *gold]):
        return False
    return all(_norm(a) == _norm(b) for a, b in zip(pred, gold))


def _step_sequence_exact_match(pred_transform: object, gold_transform: object) -> bool:
    """Exact ordered primitive-step equality, reported separately from net equality."""
    try:
        pred = _sequence_transforms(pred_transform)
        gold = _sequence_transforms(gold_transform)
    except (KeyError, TypeError, ValueError):
        return False
    return pred == gold


def _hint_tokens_present(pred_hint: str, tokens: Sequence[str]) -> bool:
    hint = _norm(pred_hint)
    return all(_norm(tok) in hint for tok in tokens)


# Vocabulary that counts as "naming" each operation family in a free-form hint. Kept
# deliberately narrow (the concept words only) so a generic "you made a mistake" hint,
# or one that names the WRONG family, does not pass.
_HINT_FAMILY_KEYWORDS = {
    "rotation": ("rotat", "turn", "clockwise", "degree", "angle"),
    "reflection": ("reflect", "mirror", "flip"),
    "translation": ("translat", "slide", "slid", "shift", "move"),
}


def _hint_mentions_family(pred_hint: str, families: Sequence[str]) -> bool:
    """True iff the hint references at least one of the required operation families."""
    hint = _norm(pred_hint)
    return any(
        kw in hint
        for fam in families
        for kw in _HINT_FAMILY_KEYWORDS.get(fam, ())
    )


def _hint_has_leak(pred_hint: str, tokens: Sequence[str]) -> bool:
    """True iff the hint states a coordinate pair it was NOT sanctioned to name.

    Sanctioned pairs (e.g. the ``(dx, dy)`` net-translation tokens in a
    ``completely_wrong`` hint) come from ``expected_hint_tokens`` and are removed before
    scanning, so only *extra* coordinates — the kind that would give the answer away —
    count as a leak.
    """
    residual = re.sub(r"\s+", "", pred_hint.lower())
    for tok in tokens:
        residual = residual.replace(re.sub(r"\s+", "", tok.lower()), "")
    return bool(_COORD_RE.search(residual))


# --------------------------------------------------------------------------------------
# Per-record scoring
# --------------------------------------------------------------------------------------

_TASK_EXPECTATIONS = {
    # v1-v5 behavior: correct transform + direct label + hint; no student map target.
    "legacy": {"correct", "label", "hint", "steps"},
    "correct": {"correct"},
    "student": {"student"},
    "both": {"correct", "student"},
    "full": {"correct", "student", "label", "hint"},
}


def _resolve_task_mode(rec: Mapping, task_mode: Optional[str]) -> str:
    if task_mode is None:
        stored = rec.get("task")
        if stored in _TASK_EXPECTATIONS:
            task_mode = str(stored)
        elif str(rec.get("schema_version", "")).startswith("v6"):
            task_mode = "full"
        else:
            task_mode = "legacy"
    if task_mode not in _TASK_EXPECTATIONS:
        raise ValueError(f"unknown eval task mode {task_mode!r}")
    return task_mode


def _gold_affine(rec: Mapping, which: str) -> tc.Transform:
    net_key = f"{which}_net"
    transform_key = f"{which}_transform"
    if net_key in rec:
        return nt.net_to_affine(rec[net_key])
    if transform_key in rec:
        return _affine_from_value(rec[transform_key])
    image_key = "correct_image" if which == "correct" else "student_image"
    recovered = tc.recover_map(rec["original"], rec[image_key])
    if recovered is None:
        raise ValueError(f"oracle record has no recoverable {which} map")
    return recovered


def _score_predicted_map(
    pred: Mapping,
    rec: Mapping,
    which: str,
    *,
    expected: bool,
) -> Tuple[Optional[bool], Optional[tc.Transform], bool]:
    """Return ``(ok, parsed_affine, field_present)`` for one observable map."""
    net_key = f"{which}_net"
    transform_key = f"{which}_transform"
    if net_key in pred:
        value = pred[net_key]
    elif transform_key in pred:
        value = pred[transform_key]
    else:
        return (False if expected else None), None, False
    try:
        affine = _affine_from_value(value)
        gold = _gold_affine(rec, which)
    except (KeyError, TypeError, ValueError):
        return False, None, True
    return affine == gold, affine, True


def _parse_failure_row(base: dict, expected: set) -> Dict[str, object]:
    correct = False if "correct" in expected else None
    student = False if "student" in expected else None
    return {
        **base,
        "pred_label": _PARSE_FAIL,
        "derived_label": None,
        "parse_ok": False,
        "label_ok": False if "label" in expected else None,
        "transform_ok": correct,
        "correct_net_ok": correct,
        "student_net_ok": student,
        "both_nets_ok": (
            False if "correct" in expected and "student" in expected else None
        ),
        "step_sequence_exact_ok": False if "steps" in expected else None,
        "derived_label_ok": None,
        "hint_ok": False if "hint" in expected else None,
        "hint_exact_ok": False if "hint" in expected else None,
        "failure_reason": "parse_fail",
    }


def score_record(
    pred_text: str,
    rec: dict,
    *,
    task_mode: Optional[str] = None,
) -> Dict[str, object]:
    """Score one v1-v6 output against an oracle record.

    ``task_mode`` is optional for backward compatibility.  Legacy records infer
    ``legacy``; v6 records infer ``full`` unless a mode is stored.  Map-only v6
    evaluations pass ``correct``/``student``/``both`` explicitly, so fields not
    requested by that task are ``None`` rather than false and are excluded from
    aggregate denominators.
    """
    mode = _resolve_task_mode(rec, task_mode)
    expected = _TASK_EXPECTATIONS[mode]
    true_label = rec.get("label")
    base = {
        "id": rec.get("id"),
        "split": rec.get("split"),
        "true_label": true_label,
        "task_mode": mode,
        "raw_model_output": pred_text,
    }

    pred = parse_pred(pred_text)
    if pred is None:
        return _parse_failure_row(base, expected)

    emitted_label = pred.get("label")
    # Keep an unknown emitted string for diagnosis/confusion analysis. Non-string JSON
    # values are deliberately normalized to None so aggregate Counter keys stay hashable.
    pred_label = emitted_label if isinstance(emitted_label, str) else None
    label_present = "label" in pred
    label_valid = pred_label in tc.DIAGNOSIS_LABELS if pred_label is not None else False
    # Preserve the established v1-v5 contract: their required label is part of parsing.
    # v6's structured modes intentionally continue below and score every field separately.
    if mode == "legacy" and "label" in expected and not label_valid:
        return _parse_failure_row(base, expected)

    label_ok: Optional[bool]
    if label_valid:
        label_ok = pred_label == true_label
    elif "label" in expected:
        label_ok = False
    else:
        label_ok = None

    correct_ok, correct_affine, correct_present = _score_predicted_map(
        pred, rec, "correct", expected="correct" in expected
    )
    student_ok, student_affine, student_present = _score_predicted_map(
        pred, rec, "student", expected="student" in expected
    )
    both_nets_ok: Optional[bool]
    if correct_ok is None or student_ok is None:
        both_nets_ok = None
    else:
        both_nets_ok = bool(correct_ok and student_ok)

    # Ordered decomposition is an explicitly separate metric. It is expected for
    # legacy v1-v5 schemas; canonical v6 map tasks have no privileged sequence.
    if "correct_transform" in pred and "correct_transform" in rec:
        step_sequence_exact_ok: Optional[bool] = _step_sequence_exact_match(
            pred["correct_transform"], rec["correct_transform"]
        )
    elif "steps" in expected:
        step_sequence_exact_ok = False
    else:
        step_sequence_exact_ok = None

    derived_label = None
    derived_label_ok: Optional[bool] = None
    if correct_affine is not None and student_affine is not None:
        derived_label = tc.diagnose(
            rec.get("original", []), [correct_affine], [student_affine]
        )
        derived_label_ok = derived_label == true_label

    hint_present = "hint" in pred
    score_hint = "hint" in expected or hint_present
    hint_ok: Optional[bool] = None
    hint_exact_ok: Optional[bool] = None
    hint_leak = False
    hint_mentions_op = False
    if score_hint:
        pred_hint = pred.get("hint")
        if isinstance(pred_hint, str) and true_label in tc.DIAGNOSIS_LABELS:
            tokens = hints.expected_hint_tokens(str(true_label), rec)
            families = hints.expected_hint_families(str(true_label), rec)
            hint_leak = _hint_has_leak(pred_hint, tokens)
            hint_mentions_op = _hint_mentions_family(pred_hint, families)
            hint_ok = hint_mentions_op and not hint_leak
            hint_exact_ok = _hint_tokens_present(pred_hint, tokens) and not hint_leak
        else:
            hint_ok = False
            hint_exact_ok = False

    # Primary failure, by priority. Once JSON syntax has parsed, distinguish malformed or
    # missing v6 fields from valid-but-wrong predictions instead of reporting parse_fail.
    # Legacy mismatch reason strings remain unchanged for historical consumers.
    if label_ok is False:
        if not label_present:
            reason = "label_missing"
        elif not label_valid:
            reason = f"invalid_label:{pred_label}" if pred_label is not None else "invalid_label"
        else:
            reason = f"wrong_label:{true_label}->{pred_label}"
    elif correct_ok is False and ("correct" in expected or correct_present):
        if mode == "legacy" or correct_affine is not None:
            reason = "transform_mismatch"
        elif not correct_present:
            reason = "correct_net_missing"
        else:
            reason = "correct_net_invalid"
    elif student_ok is False and ("student" in expected or student_present):
        if student_affine is not None:
            reason = "student_net_mismatch"
        elif not student_present:
            reason = "student_net_missing"
        else:
            reason = "student_net_invalid"
    elif hint_ok is False and score_hint:
        if mode != "legacy" and not hint_present:
            reason = "hint_missing"
        elif mode != "legacy" and (
            not isinstance(pred.get("hint"), str) or not pred.get("hint", "").strip()
        ):
            reason = "hint_invalid"
        else:
            reason = "hint_leak" if hint_leak else "hint_missing_token"
    else:
        reason = ""

    return {
        **base,
        "pred_label": pred_label,
        "derived_label": derived_label,
        "parse_ok": True,
        "label_ok": label_ok,
        # Historical alias: always the RED->GREEN semantic net result.
        "transform_ok": correct_ok,
        "correct_net_ok": correct_ok,
        "student_net_ok": student_ok,
        "both_nets_ok": both_nets_ok,
        "step_sequence_exact_ok": step_sequence_exact_ok,
        "derived_label_ok": derived_label_ok,
        "hint_ok": hint_ok,
        "hint_exact_ok": hint_exact_ok,
        "failure_reason": reason,
    }


def score_all(
    pred_texts: Sequence[str],
    recs: Sequence[dict],
    *,
    task_mode: Optional[str] = None,
) -> List[Dict[str, object]]:
    """Score parallel sequences of model outputs and oracle records."""
    if len(pred_texts) != len(recs):
        raise ValueError(f"length mismatch: {len(pred_texts)} preds vs {len(recs)} records")
    return [score_record(p, r, task_mode=task_mode) for p, r in zip(pred_texts, recs)]


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------

def _mean(flags: Sequence[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0


def _metric_values(results: Sequence[dict], field: str, fallback: str | None = None) -> List[bool]:
    values: List[bool] = []
    for row in results:
        value = row.get(field)
        if value is None and fallback is not None and field not in row:
            value = row.get(fallback)
        if value is not None:
            values.append(bool(value))
    return values


def _metric_summary(
    results: Sequence[dict], field: str, fallback: str | None = None
) -> Tuple[Optional[float], float, int]:
    values = _metric_values(results, field, fallback)
    n = len(results)
    return (
        (_mean(values) if values else None),
        (len(values) / n if n else 0.0),
        len(values),
    )


def aggregate(results: Sequence[dict], labels: Optional[Sequence[str]] = None) -> dict:
    """Roll per-record rows up into the reported metrics.

    ``balanced_accuracy`` is the mean per-label recall over only the labels actually
    present in ``results`` (so it's meaningful on the OOD split, which carries 4 of 8
    labels). ``confusion[true][pred]`` counts, with ``pred == "PARSE_FAIL"`` for outputs
    that didn't parse to a known label.
    """
    labels = list(labels or tc.DIAGNOSIS_LABELS)
    n = len(results)
    if n == 0:
        return {"n": 0}

    per_label_recall: Dict[str, Optional[float]] = {}
    for lab in labels:
        subset = [
            r for r in results
            if r.get("true_label") == lab and r.get("label_ok") is not None
        ]
        per_label_recall[lab] = _mean([bool(r["label_ok"]) for r in subset]) if subset else None

    present = [v for v in per_label_recall.values() if v is not None]
    balanced_accuracy = sum(present) / len(present) if present else None

    confusion: Dict[str, Counter] = {lab: Counter() for lab in labels}
    for r in results:
        if r.get("label_ok") is None or r.get("true_label") is None:
            continue
        pred_label = r.get("pred_label") or _PARSE_FAIL
        confusion.setdefault(r["true_label"], Counter())[pred_label] += 1

    label_rate, label_coverage, label_available = _metric_summary(results, "label_ok")
    correct_rate, correct_coverage, correct_available = _metric_summary(
        results, "correct_net_ok", fallback="transform_ok"
    )
    student_rate, student_coverage, student_available = _metric_summary(
        results, "student_net_ok"
    )
    both_rate, both_coverage, both_available = _metric_summary(results, "both_nets_ok")
    step_rate, step_coverage, step_available = _metric_summary(
        results, "step_sequence_exact_ok"
    )
    derived_rate, derived_coverage, derived_available = _metric_summary(
        results, "derived_label_ok"
    )
    hint_rate, hint_coverage, hint_available = _metric_summary(results, "hint_ok")
    hint_exact_rate, hint_exact_coverage, hint_exact_available = _metric_summary(
        results, "hint_exact_ok"
    )

    return {
        "n": n,
        "parse_rate": _mean([r["parse_ok"] for r in results]),
        "label_accuracy": label_rate,
        "label_coverage": label_coverage,
        "label_available": label_available,
        "balanced_accuracy": balanced_accuracy,
        # Historical headline, restored to semantic RED->GREEN NET-map equality.
        "transform_match_rate": correct_rate,
        "transform_coverage": correct_coverage,
        "transform_available": correct_available,
        "correct_net_match_rate": correct_rate,
        "correct_net_coverage": correct_coverage,
        "correct_net_available": correct_available,
        "student_net_match_rate": student_rate,
        "student_net_coverage": student_coverage,
        "student_net_available": student_available,
        "both_nets_match_rate": both_rate,
        "both_nets_coverage": both_coverage,
        "both_nets_available": both_available,
        "step_sequence_exact_rate": step_rate,
        "step_sequence_exact_match_rate": step_rate,
        "step_sequence_exact_coverage": step_coverage,
        "step_sequence_exact_available": step_available,
        "derived_label_accuracy": derived_rate,
        "derived_label_coverage": derived_coverage,
        "derived_label_available": derived_available,
        "hint_match_rate": hint_rate,
        "hint_coverage": hint_coverage,
        "hint_available": hint_available,
        # Secondary/exploratory (strict exact-token hint match); not in the headline table.
        "hint_exact_match_rate": hint_exact_rate,
        "hint_exact_coverage": hint_exact_coverage,
        "hint_exact_available": hint_exact_available,
        "per_label_recall": per_label_recall,
        "confusion": {t: dict(c) for t, c in confusion.items()},
    }


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

_TABLE_METRICS = (
    ("parse_rate", "parse_rate"),
    ("label_accuracy", "label_acc"),
    ("balanced_accuracy", "balanced_acc"),
    ("transform_match_rate", "transform_match"),
    ("hint_match_rate", "hint_match"),
)


def format_table(base: dict, tuned: dict) -> str:
    """A base-vs-tuned metrics table (the headline deliverable)."""
    rows = [f"{'metric':<18}{'base':>8}{'tuned':>8}{'delta':>8}"]
    rows.append("-" * 42)
    for key, label in _TABLE_METRICS:
        b, t = base.get(key), tuned.get(key)
        b_text = "--" if b is None else f"{b:.3f}"
        t_text = "--" if t is None else f"{t:.3f}"
        delta = "--" if b is None or t is None else f"{t - b:+.3f}"
        rows.append(f"{label:<18}{b_text:>8}{t_text:>8}{delta:>8}")
    return "\n".join(rows)


def format_confusion(agg: dict, labels: Optional[Sequence[str]] = None) -> str:
    """Render the confusion matrix as a text grid (true rows x predicted cols).

    Columns are the label set plus ``PF`` (PARSE_FAIL). Uses short numeric codes for
    labels (legend printed above) so the grid stays readable in a notebook cell.
    """
    labels = list(labels or tc.DIAGNOSIS_LABELS)
    confusion = agg.get("confusion", {})
    cols = labels + [_PARSE_FAIL]
    codes = {lab: str(i) for i, lab in enumerate(labels)}
    codes[_PARSE_FAIL] = "PF"

    legend = "  ".join(f"{codes[lab]}={lab}" for lab in labels)
    header = "true\\pred".ljust(12) + "".join(codes[c].rjust(5) for c in cols)
    lines = [legend, "", header]
    for lab in labels:
        row = confusion.get(lab, {})
        cells = "".join(str(row.get(c, 0)).rjust(5) for c in cols)
        lines.append(codes[lab].ljust(12) + cells)
    return "\n".join(lines)


def save_results(agg: dict, agg_path: str, records: Sequence[dict], records_path: str) -> None:
    """Write the aggregate metrics (JSON) and the per-record rows (JSONL).

    The per-record file is what turns "the model improved" into "*this* behavior improved,
    and it still fails *here*" — one line per record with the ``RECORD_FIELDS`` schema.
    """
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)
    with open(records_path, "w") as f:
        for r in records:
            row = {k: r.get(k) for k in RECORD_FIELDS}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
