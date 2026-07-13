"""Tests for the chain-of-thought trace builder (`cot.py`).

The CoT target is a reasoning trace followed by the SAME final JSON the non-CoT training
uses. Two properties matter and are asserted here:

  1. Consistency — the trace is generated from the same ground truth as the label/JSON, so
     it must NEVER contradict them: the label it concludes with equals ``diagnose`` of the
     stored transforms, and it names the correct/student steps for that record.
  2. Eval-safety — appending the trace before the JSON does not change how the record
     scores: ``parse_pred`` recovers the exact gold JSON and every field check passes,
     just like scoring the bare JSON.

Records are built in-memory via ``dataset.build_records`` (deterministic, no disk), the
same fixture style as ``test_eval.py``.
"""

from __future__ import annotations

from transform_diagnosis import chat_format, cot, dataset
from transform_diagnosis import eval as ev
from transform_diagnosis import transform_core as tc

SEED = 20260708
N = 200
MIN_COUNT = 20
OOD = 12


def _records():
    recs, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    return recs


def _one(label: str) -> dict:
    for r in _records():
        if r["label"] == label:
            return r
    raise AssertionError(f"no record with label {label!r}")


# --------------------------------------------------------------------------------------
# Consistency with the gold label / JSON
# --------------------------------------------------------------------------------------

def test_cot_target_ends_with_exact_gold_json():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        target = cot.cot_target(rec)
        gold_json = chat_format.target_json(rec)
        assert target.endswith(gold_json), label
        # The trace precedes the JSON, separated by a newline.
        assert target == cot.reasoning_trace(rec) + "\n" + gold_json


def test_trace_concludes_with_the_gold_label_token():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        trace = cot.reasoning_trace(rec)
        assert f"the diagnosis is {label}" in trace, (label, trace)


def test_trace_never_contradicts_diagnose_across_all_records():
    # Every record's trace concludes with the label an INDEPENDENT diagnose() returns.
    for rec in _records():
        d = tc.diagnose(rec["original"], rec["correct_transform"], rec["student_transform"])
        assert d == rec["label"]  # dataset invariant, re-asserted for safety
        assert f"the diagnosis is {d}" in cot.reasoning_trace(rec)


def test_trace_names_correct_and_student_steps():
    # For a single-step error, the trace names both the correct and the student step
    # (canonical wording), and states which ordinal step is wrong.
    rec = _one("wrong_reflection_line")
    trace = cot.reasoning_trace(rec)
    correct = [tc.describe_transform(tc.as_transform(s)) for s in rec["correct_transform"]]
    student = [tc.describe_transform(tc.as_transform(s)) for s in rec["student_transform"]]
    assert all(c in trace for c in correct)
    # the differing student step must appear
    diff = [i for i in range(2) if tc.as_transform(rec["correct_transform"][i])
            != tc.as_transform(rec["student_transform"][i])]
    assert len(diff) == 1
    assert student[diff[0]] in trace


def test_trace_is_deterministic():
    rec = _one("completely_wrong")
    assert cot.reasoning_trace(rec) == cot.reasoning_trace(rec)


# --------------------------------------------------------------------------------------
# Eval-safety — the trace prefix does not change scoring
# --------------------------------------------------------------------------------------

def test_cot_target_parses_back_to_gold_object():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        assert ev.parse_pred(cot.cot_target(rec)) == chat_format.target_obj(rec)


def test_cot_target_scores_all_pass_for_every_label():
    recs = _records()
    seen = set()
    for rec in recs:
        if rec["label"] in seen:
            continue
        seen.add(rec["label"])
        row = ev.score_record(cot.cot_target(rec), rec)
        assert row["parse_ok"] and row["label_ok"], rec["label"]
        assert row["transform_ok"], rec["label"]
        assert row["hint_ok"], (rec["label"], rec["hint"])
        assert row["failure_reason"] == "", rec["label"]
    assert seen == set(tc.DIAGNOSIS_LABELS)


# --------------------------------------------------------------------------------------
# Conversation shape — the CoT row's user turn is unchanged; only the assistant differs
# --------------------------------------------------------------------------------------

def test_cot_conversation_user_turn_is_unchanged_and_assistant_is_trace_plus_json():
    rec = _one("opposite_translation")
    conv = cot.to_cot_conversation(rec)
    assert conv["id"] == rec["id"] and conv["split"] == rec["split"]
    # user turn identical to the non-CoT chat_format user turn (image + instruction)
    base_user = chat_format.to_messages(rec)[0]
    assert conv["messages"][0] == base_user
    assert conv["messages"][0]["content"][0] == {"type": "image", "image": rec["render_path"]}
    # assistant turn is the CoT target (trace + JSON)
    assistant_text = conv["messages"][1]["content"][0]["text"]
    assert assistant_text == cot.cot_target(rec)
