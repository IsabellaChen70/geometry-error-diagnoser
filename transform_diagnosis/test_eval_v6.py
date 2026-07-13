"""Regression tests for v6 net-map scoring and legacy comparability."""

from __future__ import annotations

import copy
import json

from transform_diagnosis import dataset, enum_transform as et
from transform_diagnosis import eval as ev
from transform_diagnosis import net_transform as nt
from transform_diagnosis import transform_core as tc
from transform_diagnosis import v6_format


def _record():
    records, _ = dataset.build_records(20260708, 80, 10, ood_per_label=0)
    return v6_format.augment_record(records[0])


def test_v6_full_gold_scores_both_maps_and_derived_label():
    rec = _record()
    row = ev.score_record(v6_format.target_json(rec, "full"), rec, task_mode="full")
    assert row["parse_ok"]
    assert row["correct_net_ok"] is True
    assert row["student_net_ok"] is True
    assert row["both_nets_ok"] is True
    assert row["transform_ok"] is True
    assert row["label_ok"] is True
    assert row["derived_label"] == rec["label"]
    assert row["derived_label_ok"] is True
    assert row["step_sequence_exact_ok"] is None

    agg = ev.aggregate([row])
    assert agg["transform_match_rate"] == 1.0
    assert agg["correct_net_match_rate"] == 1.0
    assert agg["student_net_match_rate"] == 1.0
    assert agg["both_nets_match_rate"] == 1.0
    assert agg["derived_label_accuracy"] == 1.0
    assert agg["step_sequence_exact_rate"] is None


def test_v6_invalid_and_wrong_maps_fail_exactly():
    rec = _record()
    bad = v6_format.target_obj(rec, "both")
    bad["correct_net"] = {**bad["correct_net"], "tx": bad["correct_net"]["tx"] + 1}
    bad["student_net"] = {**bad["student_net"], "ty": "0"}
    row = ev.score_record(json.dumps(bad), rec, task_mode="both")
    assert row["parse_ok"]
    assert row["correct_net_ok"] is False
    assert row["student_net_ok"] is False
    assert row["both_nets_ok"] is False
    assert row["derived_label"] is None  # malformed student map cannot feed the oracle


def test_v6_invalid_label_does_not_erase_independently_valid_student_map():
    rec = {
        "id": 150,
        "split": "test",
        "schema_version": v6_format.SCHEMA_VERSION,
        "label": "wrong_rotation_angle",
        "correct_transform": [tc.rotate(0), tc.translate(0, 6)],
        "student_transform": [tc.rotate(90), tc.translate(0, 6)],
        "correct_net": {"linear": "identity", "tx": 0, "ty": 6},
        "student_net": {"linear": "rot_ccw_90", "tx": 0, "ty": 6},
    }
    raw = (
        '{"correct_net":{"linear":"reflect_y_eq_x","tx":0,"ty":6},'
        '"student_net":{"linear":"rot_ccw_90","tx":0,"ty":6},'
        '"label":"rotation_instead_of_rotation","hint":"..."}'
    )
    row = ev.score_record(raw, rec, task_mode="full")

    assert row["parse_ok"] is True
    assert row["pred_label"] == "rotation_instead_of_rotation"
    assert row["label_ok"] is False
    assert row["correct_net_ok"] is False
    assert row["student_net_ok"] is True
    assert row["both_nets_ok"] is False
    assert row["derived_label"] == "rotation_instead_of_reflection"
    assert row["derived_label_ok"] is False
    assert row["hint_ok"] is False
    assert row["failure_reason"] == "invalid_label:rotation_instead_of_rotation"


def test_v6_invalid_student_map_does_not_erase_valid_correct_map_or_label():
    rec = _record()
    pred = v6_format.target_obj(rec, "full")
    pred["student_net"] = {**pred["student_net"], "tx": "0"}
    row = ev.score_record(json.dumps(pred), rec, task_mode="full")

    assert row["parse_ok"] is True
    assert row["correct_net_ok"] is True
    assert row["student_net_ok"] is False
    assert row["both_nets_ok"] is False
    assert row["label_ok"] is True
    assert row["hint_ok"] is True
    assert row["derived_label"] is None
    assert row["derived_label_ok"] is None
    assert row["failure_reason"] == "student_net_invalid"

    pred = v6_format.target_obj(rec, "full")
    pred.pop("student_net")
    row = ev.score_record(json.dumps(pred), rec, task_mode="full")
    assert row["parse_ok"] is True
    assert row["correct_net_ok"] is True
    assert row["student_net_ok"] is False
    assert row["both_nets_ok"] is False
    assert row["label_ok"] is True
    assert row["derived_label"] is None
    assert row["failure_reason"] == "student_net_missing"


def test_v6_missing_label_and_invalid_hint_are_field_local_failures():
    rec = _record()
    missing_label = v6_format.target_obj(rec, "full")
    missing_label.pop("label")
    label_row = ev.score_record(json.dumps(missing_label), rec, task_mode="full")
    assert label_row["parse_ok"] is True
    assert label_row["pred_label"] is None
    assert label_row["label_ok"] is False
    assert label_row["correct_net_ok"] is True
    assert label_row["student_net_ok"] is True
    assert label_row["derived_label_ok"] is True
    assert label_row["hint_ok"] is True
    assert label_row["failure_reason"] == "label_missing"

    invalid_hint = v6_format.target_obj(rec, "full")
    invalid_hint["hint"] = {"text": rec["hint"]}
    hint_row = ev.score_record(json.dumps(invalid_hint), rec, task_mode="full")
    assert hint_row["parse_ok"] is True
    assert hint_row["label_ok"] is True
    assert hint_row["correct_net_ok"] is True
    assert hint_row["student_net_ok"] is True
    assert hint_row["derived_label_ok"] is True
    assert hint_row["hint_ok"] is False
    assert hint_row["failure_reason"] == "hint_invalid"


def test_v6_genuinely_malformed_json_keeps_parse_failure_behavior():
    rec = _record()
    row = ev.score_record('{"correct_net":', rec, task_mode="full")
    assert row["parse_ok"] is False
    assert row["pred_label"] == "PARSE_FAIL"
    assert row["label_ok"] is False
    assert row["correct_net_ok"] is False
    assert row["student_net_ok"] is False
    assert row["both_nets_ok"] is False
    assert row["failure_reason"] == "parse_fail"


def test_map_only_tasks_do_not_penalize_unrequested_optional_fields():
    rec = _record()
    row = ev.score_record(v6_format.target_json(rec, "student"), rec, task_mode="student")
    assert row["parse_ok"] and row["student_net_ok"]
    assert row["correct_net_ok"] is None
    assert row["transform_ok"] is None
    assert row["label_ok"] is None
    assert row["hint_ok"] is None
    agg = ev.aggregate([row])
    assert agg["student_net_match_rate"] == 1.0
    assert agg["student_net_coverage"] == 1.0
    assert agg["transform_match_rate"] is None
    assert agg["transform_coverage"] == 0.0
    assert agg["label_accuracy"] is None
    assert agg["label_coverage"] == 0.0


def test_both_maps_derive_label_without_direct_label():
    rec = _record()
    row = ev.score_record(v6_format.target_json(rec, "both"), rec, task_mode="both")
    assert row["label_ok"] is None
    assert row["derived_label"] == rec["label"]
    assert row["derived_label_ok"] is True
    agg = ev.aggregate([row])
    assert agg["derived_label_accuracy"] == 1.0
    assert agg["derived_label_coverage"] == 1.0
    assert agg["label_accuracy"] is None


def test_net_equivalent_v5_decomposition_passes_headline_but_fails_steps():
    gold = ["reflect across x axis", "rotate 90 degrees counterclockwise"]
    alternate = ["reflect across y axis", "rotate 270 degrees counterclockwise"]
    assert nt.sequence_to_net(gold) == nt.sequence_to_net(alternate)
    rec = {
        "id": 1,
        "split": "ood",
        "label": "correct",
        "correct_transform": gold,
    }
    pred = {"correct_transform": et.seq_enum(alternate)}
    row = ev.score_record(json.dumps(pred), rec, task_mode="correct")
    assert row["transform_ok"] is True
    assert row["correct_net_ok"] is True
    assert row["step_sequence_exact_ok"] is False


def test_legacy_prose_and_v5_enum_share_the_same_headline():
    rec = _record()
    legacy = copy.deepcopy(rec)
    legacy.pop("correct_net")
    legacy.pop("student_net")
    legacy.pop("schema_version")
    prose = {
        "label": rec["label"],
        "correct_transform": rec["correct_transform"],
        "hint": rec["hint"],
    }
    enum = {
        **prose,
        "correct_transform": et.seq_enum(rec["correct_transform"]),
    }
    prose_row = ev.score_record(json.dumps(prose), legacy)
    enum_row = ev.score_record(json.dumps(enum), legacy)
    assert prose_row["transform_ok"] is True
    assert enum_row["transform_ok"] is True
    assert prose_row["correct_net_ok"] == enum_row["correct_net_ok"] == True
    assert prose_row["step_sequence_exact_ok"] == enum_row["step_sequence_exact_ok"] == True
