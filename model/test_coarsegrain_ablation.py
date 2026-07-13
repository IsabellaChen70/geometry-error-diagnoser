"""Focused tests for coarsegrain_ablation.py — the label->coarse mapping and the coarse
accuracy/confusion math (the reusable logic that turns saved fine-grained records into the
coarse-grain ablation numbers). Pure Python; no model, no GPU, no dataset on disk.

Run:  python -m pytest model/test_coarsegrain_ablation.py -q
(kept OUT of transform_diagnosis/ so `pytest transform_diagnosis/ -q` stays at 92.)
"""

from __future__ import annotations

import json

import coarsegrain_ablation as cg


def _row(true_label, pred_label=None, raw=None):
    r = {"true_label": true_label}
    if pred_label is not None:
        r["pred_label"] = pred_label
    if raw is not None:
        r["raw_model_output"] = raw
    return r


# --------------------------------------------------------------------------------------
# The mapping itself
# --------------------------------------------------------------------------------------

def test_mapping_covers_all_eight_fine_labels():
    for mapping in ("3class", "4class"):
        fine_to_coarse, classes = cg.build_mapping(mapping)
        assert set(fine_to_coarse) == set(cg.FINE_LABELS)
        assert set(fine_to_coarse.values()) <= set(classes)


def test_type_vs_parameter_buckets_are_as_specified():
    fine_to_coarse, _ = cg.build_mapping("3class")
    assert fine_to_coarse["correct"] == "correct"
    # operation-type errors
    assert fine_to_coarse["reflection_instead_of_rotation"] == "wrong_type"
    assert fine_to_coarse["rotation_instead_of_reflection"] == "wrong_type"
    # right type, wrong parameter
    for lab in ("wrong_rotation_angle", "wrong_reflection_line",
                "wrong_translation", "opposite_translation"):
        assert fine_to_coarse[lab] == "wrong_parameter", lab


def test_completely_wrong_bucketing_differs_by_variant():
    assert cg.build_mapping("3class")[0]["completely_wrong"] == "wrong_type"
    assert cg.build_mapping("4class")[0]["completely_wrong"] == "completely_wrong"
    assert cg.build_mapping("3class")[1] == ["correct", "wrong_type", "wrong_parameter"]
    assert cg.build_mapping("4class")[1] == [
        "correct", "wrong_type", "wrong_parameter", "completely_wrong"]


# --------------------------------------------------------------------------------------
# The mapping SANITY example spelled out in the task
# --------------------------------------------------------------------------------------

def _coarse_correct(true_label, pred_label, mapping="3class"):
    """True iff this single (true, pred) row is counted CORRECT under the coarse mapping."""
    confusion, classes, _, _ = cg.coarse_confusion([_row(true_label, pred_label)], mapping)
    fine_to_coarse, _ = cg.build_mapping(mapping)
    tc = fine_to_coarse[true_label]
    return confusion[tc].get(tc, 0) == 1


def test_same_parameter_bucket_counts_as_correct():
    # right type, wrong parameter vs a DIFFERENT wrong parameter -> same coarse bucket -> CORRECT
    assert _coarse_correct("wrong_rotation_angle", "wrong_reflection_line")


def test_parameter_error_predicted_as_type_error_is_wrong():
    # wrong_parameter (true) predicted as a type error -> different coarse bucket -> WRONG
    assert not _coarse_correct("wrong_rotation_angle", "reflection_instead_of_rotation")


def test_completely_wrong_row_flips_between_variants():
    # true=completely_wrong predicted as a type error:
    #   3class: both fold to wrong_type -> CORRECT
    #   4class: completely_wrong vs wrong_type -> WRONG
    rows = [_row("completely_wrong", "reflection_instead_of_rotation")]
    assert cg.summarize_variant(rows, "3class")["overall_accuracy"] == 1.0
    assert cg.summarize_variant(rows, "4class")["overall_accuracy"] == 0.0


# --------------------------------------------------------------------------------------
# Coarse accuracy / balanced accuracy / confusion math
# --------------------------------------------------------------------------------------

def test_coarse_metrics_match_hand_computation():
    rows = [
        _row("correct", "correct"),                                  # correct -> correct  OK
        _row("correct", "reflection_instead_of_rotation"),           # correct -> wrong_type
        _row("reflection_instead_of_rotation",
             "rotation_instead_of_reflection"),                      # wrong_type -> wrong_type OK
        _row("wrong_rotation_angle", "wrong_reflection_line"),       # wrong_parameter -> " OK
        _row("wrong_rotation_angle", "reflection_instead_of_rotation"),  # wrong_parameter->type
        _row("wrong_translation", "PARSE_FAIL"),                     # wrong_parameter -> PF
    ]
    v = cg.summarize_variant(rows, "3class")

    # overall = 3/6 diagonal (correct/correct, type/type, param/param)
    assert abs(v["overall_accuracy"] - 0.5) < 1e-9
    # per-class recall: correct 1/2, wrong_type 1/1, wrong_parameter 1/3
    assert abs(v["per_class_recall"]["correct"] - 0.5) < 1e-9
    assert abs(v["per_class_recall"]["wrong_type"] - 1.0) < 1e-9
    assert abs(v["per_class_recall"]["wrong_parameter"] - (1 / 3)) < 1e-9
    # balanced = mean of the three present recalls
    assert abs(v["balanced_accuracy"] - (0.5 + 1.0 + 1 / 3) / 3) < 1e-9
    # support + parse-fail bookkeeping
    assert v["support"] == {"correct": 2, "wrong_type": 1, "wrong_parameter": 3}
    assert v["n_parse_fail"] == 1
    # PARSE_FAIL shows up as a prediction column and never on the diagonal
    assert v["confusion"]["wrong_parameter"]["PARSE_FAIL"] == 1


def test_balanced_accuracy_excludes_absent_classes():
    # Only two coarse classes present; the absent one must not drag balanced_accuracy down.
    rows = [_row("correct", "correct"), _row("wrong_rotation_angle", "wrong_translation")]
    v = cg.summarize_variant(rows, "3class")
    assert v["per_class_recall"]["wrong_type"] is None
    assert v["support"]["wrong_type"] == 0
    assert abs(v["balanced_accuracy"] - 1.0) < 1e-9   # (1.0 + 1.0) / 2, present only
    assert abs(v["overall_accuracy"] - 1.0) < 1e-9


# --------------------------------------------------------------------------------------
# Label extraction: prefer the parsed field, fall back to eval.parse_pred on raw output
# --------------------------------------------------------------------------------------

def test_extract_prefers_pred_label_field():
    assert cg.extract_labels(_row("correct", "wrong_translation")) == ("correct", "wrong_translation")
    # the stored PARSE_FAIL sentinel is honored as-is
    assert cg.extract_labels(_row("correct", "PARSE_FAIL")) == ("correct", "PARSE_FAIL")


def test_extract_falls_back_to_reparsing_raw_output():
    raw = json.dumps({"label": "opposite_translation", "correct_transform": [], "hint": "x"})
    true, pred = cg.extract_labels(_row("opposite_translation", pred_label=None, raw=raw))
    assert (true, pred) == ("opposite_translation", "opposite_translation")


def test_extract_returns_parse_fail_when_nothing_usable():
    assert cg.extract_labels(_row("correct")) == ("correct", "PARSE_FAIL")
    assert cg.extract_labels(_row("correct", raw="not json at all")) == ("correct", "PARSE_FAIL")


def test_extract_supports_oracle_label_key():
    # oracle rows use "label" for the ground truth; records use "true_label".
    assert cg.extract_labels({"label": "correct", "pred_label": "correct"}) == ("correct", "correct")


# --------------------------------------------------------------------------------------
# Summary packaging + output naming
# --------------------------------------------------------------------------------------

def test_summarize_file_reports_both_variants():
    rows = [_row("completely_wrong", "reflection_instead_of_rotation")]
    summary = cg.summarize_file(rows, "3class", "records_demo_test.jsonl")
    assert summary["selected_mapping"] == "3class"
    assert set(summary["variants"]) == {"3class", "4class"}
    assert summary["n"] == 1
    assert summary["fine_true_distribution"] == {"completely_wrong": 1}


def test_out_path_naming():
    assert cg.out_path("records_v3cot_test.jsonl").endswith("results_v3cot_test_coarse.json")
    assert cg.out_path("/a/b/records_v4_ood.jsonl") == "/a/b/results_v4_ood_coarse.json"
