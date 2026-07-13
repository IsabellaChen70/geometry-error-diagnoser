"""Tests for the v4 STRUCTURED chain-of-thought target (`cot.py` structured=True).

The v4 target adds three deterministic type fields to the final JSON
(``expected_operation_types`` / ``student_operation_types`` / ``main_mismatch``) and one
"operation-type check" line to the trace. Two properties matter and are asserted here:

  1. Consistency — the structured fields are computed from the same ground-truth transforms
     as the label, so they cannot contradict it.
  2. Eval-safety — the EXTRA JSON keys must not change scoring: ``label``/``correct_transform``/
     ``hint`` stay present and unchanged, ``parse_pred`` still recovers the final object, and
     every field check scores IDENTICALLY to the bare (v3cot) target.

Records are built in-memory via ``dataset.build_records`` (deterministic, no disk), the same
fixture style as ``test_cot.py``.
"""

from __future__ import annotations

import json

from transform_diagnosis import chat_format, cot, dataset
from transform_diagnosis import eval as ev
from transform_diagnosis import transform_core as tc

SEED = 20260708
N = 200
MIN_COUNT = 20
OOD = 12

_SCORED_KEYS = ("label", "correct_transform", "hint")
_STRUCTURED_KEYS = ("expected_operation_types", "student_operation_types", "main_mismatch")


def _records():
    recs, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    return recs


def _one(label: str) -> dict:
    for r in _records():
        if r["label"] == label:
            return r
    raise AssertionError(f"no record with label {label!r}")


# --------------------------------------------------------------------------------------
# Structured fields are consistent with the oracle (never contradict the label)
# --------------------------------------------------------------------------------------

def test_structured_fields_match_perstep_families_and_label():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        fields = cot.structured_fields(rec)
        assert fields["expected_operation_types"] == cot.operation_types(rec["correct_transform"])
        assert fields["student_operation_types"] == cot.operation_types(rec["student_transform"])
        assert fields["main_mismatch"] == cot._MAIN_MISMATCH[label]
        # families are drawn only from the three concept words
        for fam in fields["expected_operation_types"] + fields["student_operation_types"]:
            assert fam in ("rotation", "reflection", "translation"), fam


def test_type_confusion_labels_show_a_type_swap():
    # reflection_instead_of_rotation: some step is a rotation in expected but a reflection in
    # student (and vice versa for the mirror label).
    rir = cot.structured_fields(_one("reflection_instead_of_rotation"))
    assert "rotation" in rir["expected_operation_types"]
    assert "reflection" in rir["student_operation_types"]
    rr = cot.structured_fields(_one("rotation_instead_of_reflection"))
    assert "reflection" in rr["expected_operation_types"]
    assert "rotation" in rr["student_operation_types"]


def test_wrong_parameter_labels_keep_the_same_types():
    # wrong_rotation_angle / wrong_reflection_line are RIGHT type, wrong parameter: the
    # expected and student operation-type lists are identical.
    for label in ("wrong_rotation_angle", "wrong_reflection_line"):
        f = cot.structured_fields(_one(label))
        assert f["expected_operation_types"] == f["student_operation_types"], label


# --------------------------------------------------------------------------------------
# Structured target shape: scored keys FIRST + unchanged, structured keys AFTER
# --------------------------------------------------------------------------------------

def test_structured_json_key_order_and_scored_values():
    rec = _one("wrong_rotation_angle")
    obj = json.loads(cot.structured_json(rec))
    assert list(obj.keys()) == list(_SCORED_KEYS) + list(_STRUCTURED_KEYS)
    # the three scored keys are byte-identical to the bare target object
    assert {k: obj[k] for k in _SCORED_KEYS} == chat_format.target_obj(rec)


def test_structured_target_is_trace_plus_structured_json():
    rec = _one("completely_wrong")
    target = cot.cot_target(rec, structured=True)
    assert target.endswith(cot.structured_json(rec))
    assert target == cot.reasoning_trace(rec, structured=True) + "\n" + cot.structured_json(rec)
    assert "Operation-type check:" in target
    assert f"the diagnosis is {rec['label']}" in target


# --------------------------------------------------------------------------------------
# Eval-safety — extra keys never change scoring vs the bare JSON
# --------------------------------------------------------------------------------------

def test_structured_parse_pred_recovers_final_object_with_all_keys():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        parsed = ev.parse_pred(cot.cot_target(rec, structured=True))
        assert isinstance(parsed, dict)
        assert {k: parsed.get(k) for k in _SCORED_KEYS} == chat_format.target_obj(rec)
        for k in _STRUCTURED_KEYS:
            assert k in parsed, (label, k)


def test_structured_scores_identically_to_bare_for_every_label():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        bare = ev.score_record(chat_format.target_json(rec), rec)
        struct = ev.score_record(cot.cot_target(rec, structured=True), rec)
        for field in ("parse_ok", "label_ok", "transform_ok", "hint_ok",
                      "hint_exact_ok", "pred_label", "failure_reason"):
            assert bare[field] == struct[field], (label, field, bare[field], struct[field])
        # and the structured target is a clean all-pass on the gold record
        assert struct["parse_ok"] and struct["label_ok"] and struct["transform_ok"]
        assert struct["hint_ok"] and struct["failure_reason"] == "", label


# --------------------------------------------------------------------------------------
# Backward compatibility — structured=False (v3cot) output is UNCHANGED
# --------------------------------------------------------------------------------------

def test_default_cot_target_is_unchanged_v3cot_behavior():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        # default (structured=False) == the exact v3cot target: bare trace + bare JSON
        assert cot.cot_target(rec) == cot.reasoning_trace(rec) + "\n" + chat_format.target_json(rec)
        # the default trace carries NO structured line
        assert "Operation-type check:" not in cot.reasoning_trace(rec)
        # default final JSON has exactly the three scored keys
        assert list(json.loads(chat_format.target_json(rec)).keys()) == list(_SCORED_KEYS)


def test_structured_conversation_user_turn_matches_step_count_instruction():
    rec = _one("reflection_instead_of_rotation")  # two-step -> canonical instruction
    conv = cot.to_cot_conversation(rec, structured=True)
    instr = conv["messages"][0]["content"][1]["text"]
    assert instr == chat_format.INSTRUCTION
    assistant = conv["messages"][1]["content"][0]["text"]
    assert assistant == cot.cot_target(rec, structured=True)
