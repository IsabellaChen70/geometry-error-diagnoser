"""confidence_intervals.py — post-hoc statistical confidence for saved v6 eval.

No model, no GPU, no retraining, no network. This re-reads eval outputs that were
ALREADY produced (``results_*.json`` aggregates and/or ``records_*.jsonl`` per-record
rows written by ``transform_diagnosis.eval.save_results`` / the eval scripts) and puts
uncertainty on the reported accuracy RATES:

  1. A two-sided Wilson score confidence interval on every accuracy PROPORTION
     (``k / n``), e.g. ``both_nets_match_rate``. This is the SAME method already used to
     produce the ``wilson_95`` fields in ``results/v6_final/FINAL_RESULTS_SUMMARY.json``
     and the intervals quoted in ``FINAL_RESULTS.md`` — see :func:`wilson_interval`.
  2. Paired significance between two model arms that scored the SAME record ids
     (e.g. base vs tuned, tuned vs hintfix, image_hintfix vs coords_hintfix): the paired
     rate delta, McNemar's test on the discordant pairs (exact binomial for small
     discordant counts, chi-square with continuity correction otherwise), and a
     bootstrap CI on the paired delta with a FIXED seed for determinism.

WHY WILSON (matching the frozen artifacts). The existing independent auditor
``model/audit_v6_predictions.wilson`` computes, with ``z = 1.959963984540054`` (the
0.975 standard-normal quantile, i.e. two-sided 95%):

    p      = k / n
    denom  = 1 + z^2/n
    center = (p + z^2/(2n)) / denom
    margin = z * sqrt( (p(1-p) + z^2/(4n)) / n ) / denom
    (low, high) = (max(0, center-margin), min(1, center+margin))

:func:`wilson_interval` reproduces that formula exactly; at the default 95% level it
uses the identical ``z`` literal, so it is byte-for-byte consistent with the stored
``wilson_95``. ``--confidence`` selects other levels via ``statistics.NormalDist``.

DENOMINATOR SEMANTICS. A per-record boolean may be ``null`` when the metric was not
requested/applicable for that row; those rows are EXCLUDED from ``n`` (identical to
``transform_diagnosis.eval.aggregate``, whose ``*_available`` count is the ``n`` used
here). When reading a ``results_*.json`` aggregate we take ``n`` from the metric's
``*_available`` field (falling back to top-level ``n``) and recover ``k = round(rate*n)``.

CAVEAT (kept consistent with FINAL_RESULTS.md's honesty rail). These intervals quantify
finite-SAMPLE (evaluation) variability at a FIXED model checkpoint and training seed.
They do NOT capture training-seed or checkpoint variability — a different fine-tune of
the "same" recipe could land outside them.

Pure standard library (``math``/``statistics``/``random``/``json``/``argparse``); no
scipy, torch, or PIL, so it imports/tests cleanly and runs on a CPU login node.

Usage
-----
  # Wilson CIs for one or more saved cells (records preferred; results.json works too):
  python3 model/confidence_intervals.py results/v6_final/results_v6_4b_image_coords_test.json
  python3 model/confidence_intervals.py results/v6_final/records_v6_4b_*_test.jsonl

  # Paired base-vs-tuned McNemar + bootstrap on shared ids (repeat --pair as needed):
  python3 model/confidence_intervals.py \
    --pair results/v6_final/records_v6_4b_base_image_coords_test.jsonl \
           results/v6_final/records_v6_4b_image_coords_test.jsonl \
    --metric both_nets_ok --metric label_ok

  # Wilson CI for a raw k/n from a separate audit (e.g. strict hint-leak 1919/2000):
  python3 model/confidence_intervals.py --count strict_leak 1919 2000

  # Machine-readable dump:
  python3 model/confidence_intervals.py results_*.json --json-out ci_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
from typing import Dict, List, Optional, Sequence, Tuple

# Two-sided 95% z (0.975 standard-normal quantile). Identical literal to
# model/audit_v6_predictions.wilson so the default output reproduces the frozen
# ``wilson_95`` fields exactly; NormalDist().inv_cdf(0.975) equals this to <1e-15.
Z_95 = 1.959963984540054

# Default bootstrap RNG seed (distinct from data seeds: 20260709 eval sample,
# 20260711 v6 train/val, 20260712 golden). Fixed so paired-delta CIs are deterministic.
DEFAULT_SEED = 20260712
DEFAULT_BOOTSTRAP = 10000
# b + c (discordant total) at or below this uses the exact binomial McNemar test;
# above it, the chi-square approximation with continuity correction.
DEFAULT_EXACT_MAX = 40


# --------------------------------------------------------------------------------------
# Metric registry — canonical per-record boolean field <-> results_*.json aggregate keys.
# --------------------------------------------------------------------------------------

class MetricSpec:
    """One scoreable proportion: its records-row field and its results.json aliases."""

    __slots__ = ("field", "rate_keys", "n_keys")

    def __init__(self, field: str, rate_keys: Sequence[str], n_keys: Sequence[str]):
        self.field = field
        self.rate_keys = tuple(rate_keys)
        self.n_keys = tuple(n_keys)


# Order here is the display order. ``rate_keys`` / ``n_keys`` list results.json aliases in
# preference order (first present wins); every ``n_keys`` ends with "n" as a last resort.
METRIC_SPECS: Tuple[MetricSpec, ...] = (
    MetricSpec("parse_ok", ("parse_rate",), ("n",)),
    MetricSpec("correct_net_ok",
               ("correct_net_match_rate", "transform_match_rate"),
               ("correct_net_available", "transform_available", "n")),
    MetricSpec("student_net_ok", ("student_net_match_rate",),
               ("student_net_available", "n")),
    MetricSpec("both_nets_ok", ("both_nets_match_rate",),
               ("both_nets_available", "n")),
    MetricSpec("label_ok", ("label_accuracy",), ("label_available", "n")),
    MetricSpec("derived_label_ok", ("derived_label_accuracy",),
               ("derived_label_available", "n")),
    MetricSpec("hint_ok", ("hint_match_rate",), ("hint_available", "n")),
    MetricSpec("hint_exact_ok", ("hint_exact_match_rate",),
               ("hint_exact_available", "n")),
    MetricSpec("step_sequence_exact_ok",
               ("step_sequence_exact_rate", "step_sequence_exact_match_rate"),
               ("step_sequence_exact_available", "n")),
)

METRIC_BY_FIELD: Dict[str, MetricSpec] = {m.field: m for m in METRIC_SPECS}
ALL_METRIC_FIELDS: Tuple[str, ...] = tuple(m.field for m in METRIC_SPECS)

# Default metrics for a PAIRED comparison (the geometry + diagnosis + hint headline);
# any not present in both files are skipped with a note. Single-file mode reports every
# present metric instead.
DEFAULT_PAIR_METRICS: Tuple[str, ...] = (
    "correct_net_ok", "student_net_ok", "both_nets_ok",
    "label_ok", "derived_label_ok", "hint_ok",
)


# --------------------------------------------------------------------------------------
# Core statistics (closed-form; no scipy)
# --------------------------------------------------------------------------------------

def z_for_confidence(confidence: float) -> float:
    """Two-sided z (standard-normal quantile) for a confidence level in (0, 1).

    Returns the exact :data:`Z_95` literal at 0.95 so the default output is identical to
    the frozen ``wilson_95`` artifacts; other levels use ``statistics.NormalDist``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if confidence == 0.95:
        return Z_95
    return statistics.NormalDist().inv_cdf(0.5 + confidence / 2.0)


def wilson_interval(k: int, n: int, z: float = Z_95) -> Tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion ``k/n``.

    Identical formula and clamping to ``model/audit_v6_predictions.wilson`` (the source
    of the frozen ``wilson_95`` values). Returns ``(low, high)`` clamped to ``[0, 1]``;
    an empty sample (``n == 0``) returns ``(0.0, 0.0)``.
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def mcnemar_test(b: int, c: int, exact_max: int = DEFAULT_EXACT_MAX) -> dict:
    """McNemar's paired test on discordant counts ``b`` and ``c``.

    ``b`` = pairs the FIRST arm got right and the second wrong; ``c`` = the reverse.
    Concordant pairs (both right / both wrong) carry no information and are ignored. With
    ``m = b + c``:

    * ``m == 0``: no discordance -> ``p = 1.0`` (no evidence of a difference).
    * ``m <= exact_max``: exact two-sided binomial test, ``b ~ Binomial(m, 0.5)`` under
      H0, ``p = min(1, 2 * P(X <= min(b, c)))``.
    * otherwise: chi-square with continuity correction, ``chi2 = (|b-c|-1)^2 / m``,
      ``p = erfc(sqrt(chi2/2))`` (the exact upper tail of a 1-df chi-square).
    """
    m = b + c
    if m == 0:
        return {"method": "none", "p_value": 1.0, "statistic": 0.0,
                "b": b, "c": c, "n_discordant": 0}
    if m <= exact_max:
        k = min(b, c)
        tail = sum(math.comb(m, i) for i in range(k + 1)) * (0.5 ** m)
        return {"method": "exact_binomial", "p_value": min(1.0, 2.0 * tail),
                "statistic": float(k), "b": b, "c": c, "n_discordant": m}
    chi2 = (abs(b - c) - 1) ** 2 / m
    return {"method": "chi2_cc", "p_value": math.erfc(math.sqrt(chi2 / 2.0)),
            "statistic": chi2, "b": b, "c": c, "n_discordant": m}


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence (``q`` in [0, 1])."""
    if not sorted_vals:
        return float("nan")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = q * (len(sorted_vals) - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return float(sorted_vals[int(pos)])
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def bootstrap_paired_delta(
    a_flags: Sequence[int],
    b_flags: Sequence[int],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP,
    confidence: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Bootstrap CI for the paired rate delta ``mean(b) - mean(a)`` over shared records.

    ``a_flags`` / ``b_flags`` are aligned 0/1 outcomes for the SAME record ids (arm A and
    arm B). We resample the per-record DIFFERENCES ``d_i = b_i - a_i`` (each in
    ``{-1, 0, 1}``) with replacement ``n_boot`` times using a ``random.Random(seed)`` —
    fully deterministic and stdlib-only — and take the empirical percentile interval. The
    point ``delta`` equals ``(c - b) / N`` exactly.
    """
    n = len(a_flags)
    if n != len(b_flags):
        raise ValueError("a_flags and b_flags must be the same length")
    delta = (sum(b_flags) - sum(a_flags)) / n if n else 0.0
    if n == 0 or n_boot <= 0:
        return {"n": n, "n_boot": max(0, n_boot), "seed": seed,
                "delta": delta, "ci": [delta, delta]}
    diffs = [b_flags[i] - a_flags[i] for i in range(n)]
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(diffs, k=n)) / n for _ in range(n_boot))
    alpha = (1.0 - confidence) / 2.0
    lo = _percentile(means, alpha)
    hi = _percentile(means, 1.0 - alpha)
    return {"n": n, "n_boot": n_boot, "seed": seed, "delta": delta, "ci": [lo, hi]}


# --------------------------------------------------------------------------------------
# Reading saved eval outputs
# --------------------------------------------------------------------------------------

def load_jsonl(path: str) -> List[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_json(path: str) -> dict:
    with open(path) as handle:
        return json.load(handle)


def is_records_path(path: str) -> bool:
    """True for a per-record ``*.jsonl`` file, False for a ``*.json`` aggregate.

    Extension decides; an ambiguous name is sniffed (a records file's first non-blank
    line parses to a dict carrying an ``id``).
    """
    lower = path.lower()
    if lower.endswith(".jsonl"):
        return True
    if lower.endswith(".json"):
        return False
    try:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    obj = json.loads(line)
                    return isinstance(obj, dict) and "id" in obj
    except (OSError, ValueError):
        pass
    return False


def record_flag(row: dict, field: str) -> Optional[bool]:
    """The metric's boolean for one record row, or ``None`` when not applicable.

    ``correct_net_ok`` falls back to the historical ``transform_ok`` alias if the newer
    field is absent (older record files).
    """
    value = row.get(field)
    if value is None and field == "correct_net_ok" and "correct_net_ok" not in row:
        value = row.get("transform_ok")
    return None if value is None else bool(value)


def counts_from_rows(rows: Sequence[dict], field: str) -> Tuple[int, int]:
    """``(k, n)`` for a metric over record rows, excluding ``None`` (not-applicable)."""
    flags = [record_flag(r, field) for r in rows]
    present = [f for f in flags if f is not None]
    return sum(1 for f in present if f), len(present)


def counts_from_results(agg: dict, spec: MetricSpec) -> Optional[Tuple[int, int]]:
    """Recover ``(k, n)`` from a results.json aggregate, or ``None`` if not present.

    ``n`` is the first available ``*_available`` count (else top-level ``n``); ``k`` is
    ``round(rate * n)`` since the aggregate stores the rate, not the raw hit count.
    """
    rate = None
    for key in spec.rate_keys:
        if agg.get(key) is not None:
            rate = float(agg[key])
            break
    if rate is None:
        return None
    n = None
    for key in spec.n_keys:
        if agg.get(key) is not None:
            n = int(agg[key])
            break
    if n is None or n <= 0:
        return None
    return int(round(rate * n)), n


def present_metrics_records(rows: Sequence[dict]) -> List[str]:
    """Metric fields with at least one non-None value across ``rows`` (display order)."""
    out = []
    for field in ALL_METRIC_FIELDS:
        _, n = counts_from_rows(rows, field)
        if n > 0:
            out.append(field)
    return out


def present_metrics_results(agg: dict) -> List[str]:
    """Metric fields recoverable from a results.json aggregate (display order)."""
    return [m.field for m in METRIC_SPECS if counts_from_results(agg, m) is not None]


# --------------------------------------------------------------------------------------
# Per-file (single-cell) Wilson report
# --------------------------------------------------------------------------------------

def wilson_report(path: str, metrics: Optional[Sequence[str]], z: float,
                  confidence: float) -> dict:
    """Wilson CI for every requested (or every present) metric in one saved file."""
    records = is_records_path(path)
    if records:
        rows = load_jsonl(path)
        n_rows = len(rows)
        chosen = list(metrics) if metrics else present_metrics_records(rows)
        source = "records"
    else:
        agg = load_json(path)
        n_rows = int(agg.get("n", 0))
        chosen = list(metrics) if metrics else present_metrics_results(agg)
        source = "results"

    out: Dict[str, dict] = {}
    for field in chosen:
        if field not in METRIC_BY_FIELD:
            continue
        if records:
            k, n = counts_from_rows(rows, field)
        else:
            got = counts_from_results(agg, METRIC_BY_FIELD[field])
            if got is None:
                continue
            k, n = got
        if n == 0:
            continue
        low, high = wilson_interval(k, n, z)
        out[field] = {
            "k": k, "n": n, "rate": k / n,
            "wilson": [low, high],
            "wilson_pct": [round(100.0 * low, 1), round(100.0 * high, 1)],
        }
    return {"path": path, "source": source, "n_rows": n_rows,
            "confidence": confidence, "metrics": out}


# --------------------------------------------------------------------------------------
# Paired report (McNemar + bootstrap on shared ids)
# --------------------------------------------------------------------------------------

def _key(row: dict) -> Tuple[object, object]:
    return (row.get("split"), row.get("id"))


def paired_flags(rows_a: Sequence[dict], rows_b: Sequence[dict], field: str
                 ) -> Tuple[List[int], List[int], List[Tuple[object, object]]]:
    """Align arm-A/arm-B outcomes by ``(split, id)`` for one metric.

    Returns ``(a_flags, b_flags, keys)`` over ids present in BOTH files where the metric
    is non-None in BOTH. ``keys`` are the matched ``(split, id)`` pairs (sorted, stable).
    """
    amap = {_key(r): r for r in rows_a}
    bmap = {_key(r): r for r in rows_b}
    shared = sorted(set(amap) & set(bmap), key=lambda t: (str(t[0]), _id_sort(t[1])))
    a_flags: List[int] = []
    b_flags: List[int] = []
    keys: List[Tuple[object, object]] = []
    for key in shared:
        fa = record_flag(amap[key], field)
        fb = record_flag(bmap[key], field)
        if fa is None or fb is None:
            continue
        a_flags.append(1 if fa else 0)
        b_flags.append(1 if fb else 0)
        keys.append(key)
    return a_flags, b_flags, keys


def _id_sort(value: object):
    """Sort ids numerically when possible, else lexicographically (mixed-safe)."""
    return (0, value) if isinstance(value, (int, float)) else (1, str(value))


def paired_report(path_a: str, path_b: str, metrics: Optional[Sequence[str]], *,
                  z: float, confidence: float, n_boot: int, seed: int,
                  exact_max: int) -> dict:
    """McNemar + bootstrap paired-delta CI for two records files over shared ids."""
    if not is_records_path(path_a) or not is_records_path(path_b):
        raise SystemExit(
            f"--pair requires two records_*.jsonl files (per-record booleans needed for "
            f"pairing); got {path_a!r} and {path_b!r}")
    rows_a = load_jsonl(path_a)
    rows_b = load_jsonl(path_b)
    keys_a = {_key(r) for r in rows_a}
    keys_b = {_key(r) for r in rows_b}
    shared = keys_a & keys_b

    if metrics:
        chosen = list(metrics)
    else:
        present_a = set(present_metrics_records(rows_a))
        present_b = set(present_metrics_records(rows_b))
        chosen = [m for m in DEFAULT_PAIR_METRICS if m in present_a and m in present_b]

    out: Dict[str, dict] = {}
    for field in chosen:
        if field not in METRIC_BY_FIELD:
            continue
        a_flags, b_flags, keys = paired_flags(rows_a, rows_b, field)
        n = len(keys)
        if n == 0:
            continue
        both = sum(1 for i in range(n) if a_flags[i] and b_flags[i])
        a_only = sum(1 for i in range(n) if a_flags[i] and not b_flags[i])
        b_only = sum(1 for i in range(n) if b_flags[i] and not a_flags[i])
        neither = n - both - a_only - b_only
        rate_a = sum(a_flags) / n
        rate_b = sum(b_flags) / n
        mcnemar = mcnemar_test(a_only, b_only, exact_max=exact_max)
        boot = bootstrap_paired_delta(a_flags, b_flags, n_boot=n_boot,
                                      confidence=confidence, seed=seed)
        out[field] = {
            "n": n,
            "rate_a": rate_a,
            "rate_b": rate_b,
            "delta": rate_b - rate_a,
            "delta_pp": round(100.0 * (rate_b - rate_a), 2),
            "discordant": {"a_only": a_only, "b_only": b_only},
            "table": {"both": both, "a_only": a_only, "b_only": b_only,
                      "neither": neither},
            "mcnemar": mcnemar,
            "bootstrap": {
                "n_boot": boot["n_boot"], "seed": boot["seed"],
                "ci": boot["ci"],
                "ci_pp": [round(100.0 * boot["ci"][0], 2),
                          round(100.0 * boot["ci"][1], 2)],
            },
        }
    return {
        "a": path_a, "b": path_b,
        "n_shared_ids": len(shared),
        "n_a": len(rows_a), "n_b": len(rows_b),
        "confidence": confidence,
        "metrics": out,
    }


# --------------------------------------------------------------------------------------
# Text rendering
# --------------------------------------------------------------------------------------

def _fmt_p(x: float) -> str:
    return f"{x:.5f}"


def format_wilson(report: dict) -> str:
    conf_pct = f"{report['confidence'] * 100:g}"
    lines = [
        f"=== {os.path.basename(report['path'])}  "
        f"(source={report['source']}, n={report['n_rows']}, {conf_pct}% Wilson) ===",
        f"{'metric':<24}{'k':>6}{'n':>6}{'rate':>10}"
        f"{'  ' + conf_pct + '% CI (Wilson)':>22}{'    as %':>9}{'   CI %':>16}",
    ]
    if not report["metrics"]:
        lines.append("  (no scoreable proportions found)")
        return "\n".join(lines)
    for field, m in report["metrics"].items():
        lo, hi = m["wilson"]
        plo, phi = m["wilson_pct"]
        lines.append(
            f"{field:<24}{m['k']:>6}{m['n']:>6}{m['rate']:>10.5f}"
            f"   [{_fmt_p(lo)}, {_fmt_p(hi)}]"
            f"{100.0 * m['rate']:>8.1f}%"
            f"   [{plo:.1f}\u2013{phi:.1f}]"
        )
    return "\n".join(lines)


def format_paired(report: dict) -> str:
    conf_pct = f"{report['confidence'] * 100:g}"
    a = os.path.basename(report["a"])
    b = os.path.basename(report["b"])
    lines = [
        f"=== paired: A={a}",
        f"           B={b}",
        f"    shared ids={report['n_shared_ids']}  (A n={report['n_a']}, B n={report['n_b']})"
        f"   delta = rate(B) - rate(A)",
        f"{'metric':<20}{'rate_A':>9}{'rate_B':>9}{'delta_pp':>10}"
        f"{'  disc(A/B)':>12}{'  McNemar p (method)':>26}"
        f"{'  boot ' + conf_pct + '% CI(pp)':>22}",
    ]
    if not report["metrics"]:
        lines.append("  (no shared scoreable metric; check ids/fields)")
        return "\n".join(lines)
    for field, m in report["metrics"].items():
        mc = m["mcnemar"]
        blo, bhi = m["bootstrap"]["ci_pp"]
        disc = f"{m['discordant']['a_only']}/{m['discordant']['b_only']}"
        pstr = f"{mc['p_value']:.2e} ({mc['method']})"
        lines.append(
            f"{field:<20}{m['rate_a']:>9.4f}{m['rate_b']:>9.4f}"
            f"{m['delta_pp']:>+10.2f}{disc:>12}{pstr:>26}"
            f"   [{blo:+.2f}, {bhi:+.2f}]"
        )
    return "\n".join(lines)


NOTE = (
    "note: Wilson + McNemar + bootstrap quantify finite-SAMPLE (evaluation) variability "
    "at a FIXED\n      checkpoint and training seed; they do NOT capture "
    "training-seed / checkpoint variance."
)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Post-hoc confidence intervals + paired significance for saved v6 "
                    "eval outputs (offline; no model / GPU / scipy).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Metrics: " + ", ".join(ALL_METRIC_FIELDS))
    ap.add_argument("inputs", nargs="*",
                    help="results_*.json and/or records_*.jsonl for per-cell Wilson CIs "
                         "(records preferred; results.json recovers k=round(rate*n)).")
    ap.add_argument("--pair", nargs=2, action="append", metavar=("A", "B"), default=[],
                    help="two records_*.jsonl sharing ids (A then B); repeatable. Reports "
                         "delta=rate(B)-rate(A), McNemar, and a bootstrap delta CI.")
    ap.add_argument("--count", nargs=3, action="append", metavar=("LABEL", "K", "N"),
                    default=[],
                    help="Wilson CI for a raw k/n proportion (e.g. a strict hint-leak "
                         "rate from a separate audit); repeatable.")
    ap.add_argument("--metric", action="append", default=None, dest="metrics",
                    help="restrict to this metric (repeatable). Default: all present "
                         "(single file) / the headline set (paired).")
    ap.add_argument("--confidence", type=float, default=0.95,
                    help="two-sided confidence level for Wilson and the bootstrap "
                         "(default: 0.95, matching the frozen wilson_95).")
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                    help=f"bootstrap resamples for the paired delta CI, 0 disables "
                         f"(default: {DEFAULT_BOOTSTRAP}).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"bootstrap RNG seed for determinism (default: {DEFAULT_SEED}).")
    ap.add_argument("--exact-max", type=int, default=DEFAULT_EXACT_MAX,
                    help=f"discordant total b+c at/below which McNemar uses the exact "
                         f"binomial; above it, chi-square w/ continuity correction "
                         f"(default: {DEFAULT_EXACT_MAX}).")
    ap.add_argument("--json-out", default=None,
                    help="also write a machine-readable JSON summary here "
                         "(must NOT be under a results/ directory).")
    return ap


def _validate_metrics(metrics: Optional[Sequence[str]]) -> Optional[List[str]]:
    if not metrics:
        return None
    unknown = [m for m in metrics if m not in METRIC_BY_FIELD]
    if unknown:
        raise SystemExit(
            f"unknown metric(s): {', '.join(unknown)}\n"
            f"choose from: {', '.join(ALL_METRIC_FIELDS)}")
    return list(metrics)


def _check_json_out(path: Optional[str]) -> None:
    if path is None:
        return
    parts = os.path.normpath(os.path.abspath(path)).split(os.sep)
    if "results" in parts:
        raise SystemExit(
            f"refusing to write --json-out under a results/ directory (read-only): {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = _validate_metrics(args.metrics)
    _check_json_out(args.json_out)
    if not args.inputs and not args.pair and not args.count:
        raise SystemExit(
            "nothing to do: pass results_*/records_* inputs, --pair A B, or --count L K N")

    z = z_for_confidence(args.confidence)

    summary = {
        "confidence": args.confidence,
        "z": z,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "exact_max": args.exact_max,
        "wilson": {},
        "counts": {},
        "paired": [],
    }

    if args.count:
        conf_pct = f"{args.confidence * 100:g}"
        print(f"=== raw k/n proportions ({conf_pct}% Wilson) ===")
        print(f"{'label':<24}{'k':>8}{'n':>8}{'rate':>10}"
              f"{'  ' + conf_pct + '% CI (Wilson)':>22}{'   CI %':>16}")
        for label, k_str, n_str in args.count:
            k, n = int(k_str), int(n_str)
            low, high = wilson_interval(k, n, z)
            summary["counts"][label] = {
                "k": k, "n": n, "rate": (k / n if n else 0.0),
                "wilson": [low, high],
                "wilson_pct": [round(100.0 * low, 1), round(100.0 * high, 1)],
            }
            rate = k / n if n else 0.0
            print(f"{label:<24}{k:>8}{n:>8}{rate:>10.5f}"
                  f"   [{_fmt_p(low)}, {_fmt_p(high)}]"
                  f"   [{100.0 * low:.1f}\u2013{100.0 * high:.1f}]")
        print()

    for path in args.inputs:
        if not os.path.exists(path):
            raise SystemExit(f"input not found: {path}")
        report = wilson_report(path, metrics, z, args.confidence)
        summary["wilson"][path] = report
        print(format_wilson(report))
        print()

    for path_a, path_b in args.pair:
        for path in (path_a, path_b):
            if not os.path.exists(path):
                raise SystemExit(f"--pair file not found: {path}")
        report = paired_report(path_a, path_b, metrics, z=z,
                               confidence=args.confidence, n_boot=args.bootstrap,
                               seed=args.seed, exact_max=args.exact_max)
        summary["paired"].append(report)
        print(format_paired(report))
        print()

    print(NOTE)

    if args.json_out is not None:
        with open(args.json_out, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        print(f"\nwrote {os.path.relpath(args.json_out)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
