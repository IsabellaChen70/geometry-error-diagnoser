"""Tests for model/confidence_intervals.py.

Locks the Wilson interval to the frozen artifact (it must equal
``model.audit_v6_predictions.wilson`` and reproduce the stored ``wilson_95`` values),
and checks the McNemar branches, bootstrap determinism, and the results-reading helpers.
"""

from __future__ import annotations

import json
import math

import pytest

from model import audit_v6_predictions as audit
from model import confidence_intervals as ci


# --------------------------------------------------------------------------------------
# Wilson — must MATCH the existing method/artifacts
# --------------------------------------------------------------------------------------

def test_z_for_confidence_matches_audit_literal():
    assert ci.z_for_confidence(0.95) == audit.wilson.__defaults__[0] == ci.Z_95


def test_wilson_equals_audit_wilson():
    for k, n in [(0, 500), (2, 500), (192, 500), (231, 500), (493, 500),
                 (497, 500), (498, 500), (500, 500), (21, 50), (1, 1)]:
        assert ci.wilson_interval(k, n) == audit.wilson(k, n)


def test_wilson_reproduces_frozen_summary_values():
    # From results/v6_final/FINAL_RESULTS_SUMMARY.json (rounded to 6 dp there).
    cases = {
        (493, 500): [0.971387, 0.993202],
        (192, 500): [0.342408, 0.427361],
        (500, 500): [0.992376, 1.0],
        (412, 500): [0.788186, 0.854874],
    }
    for (k, n), expected in cases.items():
        low, high = ci.wilson_interval(k, n)
        assert round(low, 6) == expected[0]
        assert round(high, 6) == expected[1]


def test_wilson_other_confidence_level_is_wider():
    lo95, hi95 = ci.wilson_interval(450, 500, ci.z_for_confidence(0.95))
    lo99, hi99 = ci.wilson_interval(450, 500, ci.z_for_confidence(0.99))
    assert lo99 < lo95 < hi95 < hi99


def test_wilson_empty_sample():
    assert ci.wilson_interval(0, 0) == (0.0, 0.0)


# --------------------------------------------------------------------------------------
# McNemar — exact / chi-square / degenerate branches
# --------------------------------------------------------------------------------------

def test_mcnemar_no_discordance_is_one():
    r = ci.mcnemar_test(0, 0)
    assert r["method"] == "none" and r["p_value"] == 1.0


def test_mcnemar_exact_binomial_small_counts():
    # b=3,c=0 -> 2 * 0.5^3 = 0.25 ; b=2,c=0 -> 2 * 0.5^2 = 0.5
    assert ci.mcnemar_test(3, 0)["method"] == "exact_binomial"
    assert ci.mcnemar_test(3, 0)["p_value"] == pytest.approx(0.25)
    assert ci.mcnemar_test(2, 0)["p_value"] == pytest.approx(0.5)
    # symmetric in b, c
    assert ci.mcnemar_test(3, 5)["p_value"] == ci.mcnemar_test(5, 3)["p_value"]


def test_mcnemar_exact_two_sided_capped_at_one():
    assert ci.mcnemar_test(5, 5)["p_value"] == pytest.approx(1.0)


def test_mcnemar_chisq_branch_overwhelming():
    r = ci.mcnemar_test(0, 493, exact_max=ci.DEFAULT_EXACT_MAX)
    assert r["method"] == "chi2_cc"
    assert r["statistic"] == pytest.approx((abs(0 - 493) - 1) ** 2 / 493)
    assert r["p_value"] < 1e-100


def test_mcnemar_exact_max_threshold_switch():
    # exactly at the threshold -> exact; one above -> chi-square
    assert ci.mcnemar_test(20, 20, exact_max=40)["method"] == "exact_binomial"
    assert ci.mcnemar_test(20, 21, exact_max=40)["method"] == "chi2_cc"


# --------------------------------------------------------------------------------------
# Bootstrap — determinism + correctness of the point estimate
# --------------------------------------------------------------------------------------

def test_bootstrap_is_deterministic_with_seed():
    a = [0] * 500
    b = [1] * 493 + [0] * 7
    r1 = ci.bootstrap_paired_delta(a, b, n_boot=1000, seed=20260712)
    r2 = ci.bootstrap_paired_delta(a, b, n_boot=1000, seed=20260712)
    assert r1["ci"] == r2["ci"]
    assert r1["delta"] == pytest.approx(0.986)
    lo, hi = r1["ci"]
    assert lo < 0.986 < hi


def test_bootstrap_different_seed_changes_draws():
    a = [0, 1] * 100
    b = [1, 0] * 100
    r1 = ci.bootstrap_paired_delta(a, b, n_boot=500, seed=1)
    r2 = ci.bootstrap_paired_delta(a, b, n_boot=500, seed=2)
    assert r1["ci"] != r2["ci"]  # same point estimate, different resample draws


def test_bootstrap_disabled_returns_point():
    r = ci.bootstrap_paired_delta([0, 0], [1, 1], n_boot=0)
    assert r["ci"] == [1.0, 1.0]


def test_percentile_endpoints_and_interp():
    vals = [0.0, 1.0, 2.0, 3.0, 4.0]
    assert ci._percentile(vals, 0.0) == 0.0
    assert ci._percentile(vals, 1.0) == 4.0
    assert ci._percentile(vals, 0.5) == 2.0


# --------------------------------------------------------------------------------------
# Reading helpers — counts from records vs results aggregate
# --------------------------------------------------------------------------------------

def test_counts_from_rows_excludes_none():
    rows = [
        {"both_nets_ok": True},
        {"both_nets_ok": False},
        {"both_nets_ok": None},   # not applicable -> excluded from n
        {},                        # missing -> excluded from n
    ]
    assert ci.counts_from_rows(rows, "both_nets_ok") == (1, 2)


def test_correct_net_falls_back_to_transform_ok():
    rows = [{"transform_ok": True}, {"transform_ok": False}]
    assert ci.counts_from_rows(rows, "correct_net_ok") == (1, 2)


def test_counts_from_results_recovers_k_round_rate_times_n():
    agg = {"both_nets_match_rate": 0.986, "both_nets_available": 500, "n": 500}
    spec = ci.METRIC_BY_FIELD["both_nets_ok"]
    assert ci.counts_from_results(agg, spec) == (493, 500)


def test_counts_from_results_missing_metric_is_none():
    spec = ci.METRIC_BY_FIELD["hint_ok"]
    assert ci.counts_from_results({"n": 500}, spec) is None


def test_paired_flags_align_by_split_and_id():
    rows_a = [
        {"split": "test", "id": 1, "both_nets_ok": True},
        {"split": "test", "id": 2, "both_nets_ok": False},
        {"split": "test", "id": 3, "both_nets_ok": None},   # dropped (None)
        {"split": "test", "id": 9, "both_nets_ok": True},   # unmatched
    ]
    rows_b = [
        {"split": "test", "id": 1, "both_nets_ok": True},
        {"split": "test", "id": 2, "both_nets_ok": True},
        {"split": "test", "id": 3, "both_nets_ok": True},
    ]
    a_flags, b_flags, keys = ci.paired_flags(rows_a, rows_b, "both_nets_ok")
    assert keys == [("test", 1), ("test", 2)]
    assert a_flags == [1, 0]
    assert b_flags == [1, 1]


# --------------------------------------------------------------------------------------
# End-to-end reports over tiny synthetic files
# --------------------------------------------------------------------------------------

def _write_records(path, rows):
    with open(path, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def test_wilson_report_from_records(tmp_path):
    p = tmp_path / "records_toy_test.jsonl"
    _write_records(p, [{"split": "test", "id": i, "both_nets_ok": i < 9}
                       for i in range(10)])
    rep = ci.wilson_report(str(p), ["both_nets_ok"], ci.Z_95, 0.95)
    assert rep["metrics"]["both_nets_ok"]["k"] == 9
    assert rep["metrics"]["both_nets_ok"]["n"] == 10
    lo, hi = rep["metrics"]["both_nets_ok"]["wilson"]
    assert (lo, hi) == ci.wilson_interval(9, 10)


def test_paired_report_end_to_end(tmp_path):
    a = tmp_path / "records_base_test.jsonl"
    b = tmp_path / "records_tuned_test.jsonl"
    _write_records(a, [{"split": "test", "id": i, "both_nets_ok": False}
                       for i in range(100)])
    _write_records(b, [{"split": "test", "id": i, "both_nets_ok": i < 90}
                       for i in range(100)])
    rep = ci.paired_report(str(a), str(b), ["both_nets_ok"], z=ci.Z_95,
                           confidence=0.95, n_boot=500, seed=ci.DEFAULT_SEED,
                           exact_max=ci.DEFAULT_EXACT_MAX)
    m = rep["metrics"]["both_nets_ok"]
    assert m["discordant"] == {"a_only": 0, "b_only": 90}
    assert m["delta"] == pytest.approx(0.90)
    assert m["mcnemar"]["method"] == "chi2_cc"
    assert m["mcnemar"]["p_value"] < 1e-20


def test_paired_report_requires_records(tmp_path):
    j = tmp_path / "results_toy.json"
    j.write_text(json.dumps({"n": 1, "both_nets_match_rate": 1.0}))
    with pytest.raises(SystemExit):
        ci.paired_report(str(j), str(j), ["both_nets_ok"], z=ci.Z_95,
                         confidence=0.95, n_boot=10, seed=1, exact_max=40)


# --------------------------------------------------------------------------------------
# CLI guards
# --------------------------------------------------------------------------------------

def test_cli_rejects_unknown_metric():
    with pytest.raises(SystemExit):
        ci.main(["--metric", "not_a_metric",
                 "--pair", "a.jsonl", "b.jsonl"])


def test_cli_refuses_json_out_under_results():
    with pytest.raises(SystemExit):
        ci._check_json_out("results/v6_final/x.json")


def test_cli_needs_something_to_do():
    with pytest.raises(SystemExit):
        ci.main([])


def test_cli_count_writes_wilson_json(tmp_path):
    out = tmp_path / "ci.json"
    # strict hint-leak headline: 1919/2000 (~0.96) from the safety audit
    rc = ci.main(["--count", "leak", "1919", "2000", "--json-out", str(out)])
    assert rc == 0
    data = json.loads(out.read_text())
    leak = data["counts"]["leak"]
    assert leak["k"] == 1919 and leak["n"] == 2000
    assert leak["wilson"] == list(ci.wilson_interval(1919, 2000))
