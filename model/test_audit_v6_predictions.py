"""Tests for the independent v6 prediction audit."""

from __future__ import annotations

import json

from model import audit_v6_predictions as audit


def _oracle():
    return {
        "id": 7,
        "split": "test",
        "label": "completely_wrong",
        "original": [[0, 0], [1, 0], [0, 1]],
        "correct_image": [[2, 0], [2, 1], [1, 0]],
        "student_image": [[3, 0], [2, 0], [3, 1]],
    }


def _prediction():
    return {
        "correct_net": {"linear": "rot_ccw_90", "tx": 2, "ty": 0},
        "student_net": {"linear": "reflect_y_axis", "tx": 3, "ty": 0},
        "label": "completely_wrong",
        "hint": "Check both maps.",
    }


def test_extracts_last_json_object():
    text = 'scratch {"decoy": true}\nfinal ' + json.dumps(_prediction())
    assert audit.extract_last_object(text) == _prediction()


def test_direct_geometry_audit_matches_observed_vertices():
    saved = {
        "id": 7,
        "split": "test",
        "raw_model_output": json.dumps(_prediction()),
        "correct_net_ok": True,
        "student_net_ok": True,
        "both_nets_ok": True,
        "label_ok": True,
        "derived_label_ok": True,
    }
    row = audit.audit_row(saved, _oracle())
    assert row["correct_net_ok"] is True
    assert row["student_net_ok"] is True
    assert row["both_nets_ok"] is True
    assert row["derived_label"] == "completely_wrong"
    assert row["stored_metric_disagreements"] == {}


def test_wrong_translation_is_rejected_and_catches_evaluator_disagreement():
    pred = _prediction()
    pred["correct_net"] = {"linear": "rot_ccw_90", "tx": 3, "ty": 0}
    saved = {
        "id": 7,
        "split": "test",
        "raw_model_output": json.dumps(pred),
        "correct_net_ok": True,
        "student_net_ok": True,
        "both_nets_ok": True,
        "label_ok": True,
        "derived_label_ok": True,
    }
    row = audit.audit_row(saved, _oracle())
    assert row["correct_net_ok"] is False
    assert row["both_nets_ok"] is False
    assert "correct_net_ok" in row["stored_metric_disagreements"]


def test_wilson_interval_contains_observed_rate():
    low, high = audit.wilson(21, 50)
    assert low < 21 / 50 < high


def test_paired_summary_counts_disagreements():
    left = [
        {"id": 1, "split": "test", "correct_net_ok": True},
        {"id": 2, "split": "test", "correct_net_ok": False},
    ]
    right = [
        {"id": 1, "split": "test", "correct_net_ok": False},
        {"id": 2, "split": "test", "correct_net_ok": False},
    ]
    result = audit.paired_summary("left", left, "right", right)
    assert result["metrics"]["correct_net_ok"]["left_only"] == 1
    assert result["metrics"]["correct_net_ok"]["right_only"] == 0


def test_leakage_audit_detects_exact_geometry_overlap(tmp_path):
    v6 = tmp_path / "v6"
    v6.mkdir()
    (v6 / "train_v6.jsonl").write_text(json.dumps(_oracle()) + "\n")
    result = audit.leakage_audit(str(v6), [_oracle()])
    assert result["exact_geometry_overlap_count"] == 1
