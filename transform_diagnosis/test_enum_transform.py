"""Tests for the v5 DISCRETE transform vocabulary (`enum_transform.py`) and its wiring into
the eval harness (`eval._transform_match`) and CoT target builder (`cot`).

Four properties matter and are asserted here:

  1. Vocabulary + bijection — the enum is derived from ``transform_core`` (never invented),
     canonicalizes wording variants (270 cw == 90 ccw) to one param, and round-trips
     loss-lessly on EVERY primitive step actually present in the data.
  2. Exact per-step matching — ``steps_match`` scores a prediction by type + param (or
     dx/dy), passing an exact answer and failing any wrong param / wrong dx (both
     directions), across multiple labels including a translation.
  3. Eval dispatch + backward-compat — ``eval._transform_match`` composes both enum and prose
     to the same semantic NET map, preserving the v1-v4 apples-to-apples headline.
  4. Exact ordered steps remain available as the separate
     ``eval._step_sequence_exact_match`` diagnostic.

Records are built in-memory via ``dataset.build_records`` (deterministic, no disk), the same
fixture style as ``test_eval.py`` / ``test_cot.py``.
"""

from __future__ import annotations

import copy
import json

from transform_diagnosis import chat_format, contrastive, cot, dataset
from transform_diagnosis import enum_transform as et
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
# 1. Vocabulary + bijection
# --------------------------------------------------------------------------------------

def test_vocabulary_is_the_actual_set_from_transform_core():
    # Exactly the 3 rotation matrices and 4 reflection lines transform_core defines.
    assert et.STEP_TYPES == ("rotation", "reflection", "translation")
    assert set(et.ROTATION_PARAMS) == {"rot_ccw_90", "rot_180", "rot_ccw_270"}
    assert set(et.REFLECTION_PARAMS) == {"reflect_x", "reflect_y", "reflect_y=x", "reflect_y=-x"}
    assert len(et.ROTATION_PARAMS) == 3 and len(et.REFLECTION_PARAMS) == 4


def test_step_enum_shapes_and_canonicalization():
    # Rotations/reflections -> {type, param}; translations -> {type, dx, dy}.
    assert et.step_enum("rotate 90 degrees counterclockwise") == {"type": "rotation", "param": "rot_ccw_90"}
    assert et.step_enum("rotate 180 degrees counterclockwise") == {"type": "rotation", "param": "rot_180"}
    # 270 cw is the SAME matrix as 90 ccw -> canonicalizes to the same param.
    assert et.step_enum("rotate 270 degrees clockwise") == {"type": "rotation", "param": "rot_ccw_90"}
    assert et.step_enum("reflect across line y = -x") == {"type": "reflection", "param": "reflect_y=-x"}
    assert et.step_enum("translate 7 left") == {"type": "translation", "dx": -7, "dy": 0}
    assert et.step_enum("translate by (-2, 3)") == {"type": "translation", "dx": -2, "dy": 3}


def test_enum_round_trips_on_every_primitive_in_the_space():
    # step_enum then enum_step_to_transform is the identity on the whole primitive space.
    prims = [tc.rotate(d) for d in (90, 180, 270)]
    prims += [tc.reflect(l) for l in ("x", "y", "y=x", "y=-x")]
    prims += [tc.translate(dx, dy) for dx in range(-8, 9) for dy in range(-8, 9)]
    for t in prims:
        assert et.enum_step_to_transform(et.step_enum(t)) == t


def test_enum_covers_and_round_trips_the_whole_dataset_vocabulary():
    # Over in-distribution + OOD + contrastive + curriculum: every correct_transform step
    # maps to a vocabulary value AND the enum reconstructs the SAME net map (loss-less).
    import random
    recs = _records()
    con, _ = contrastive.build_contrastive_partials(random.Random(SEED), 8)
    cur = contrastive.build_curriculum_partials(random.Random(SEED), 18)
    for rec in recs + con + cur:
        enum_seq = et.seq_enum(rec["correct_transform"])
        for step in enum_seq:
            assert step["type"] in et.STEP_TYPES
            if step["type"] == "rotation":
                assert step["param"] in et.ROTATION_PARAMS
            elif step["type"] == "reflection":
                assert step["param"] in et.REFLECTION_PARAMS
            else:
                assert isinstance(step["dx"], int) and isinstance(step["dy"], int)
        assert tc.compose(et.enum_to_transforms(enum_seq)) == tc.compose(rec["correct_transform"])


# --------------------------------------------------------------------------------------
# 2. Detection + exact per-step matching (both directions)
# --------------------------------------------------------------------------------------

def test_is_enum_seq_distinguishes_enum_from_prose():
    assert et.is_enum_seq(et.seq_enum(["rotate 90 degrees counterclockwise", "translate 7 left"]))
    assert not et.is_enum_seq(["rotate 90 degrees counterclockwise", "translate 7 left"])  # prose
    assert not et.is_enum_seq([])            # empty is not the enum format
    assert not et.is_enum_seq("rotate 90")   # bare string
    assert not et.is_enum_seq([{"foo": "bar"}])  # dict without a known type


def test_steps_match_exact_both_directions_rotation():
    gold = et.seq_enum(["rotate 90 degrees counterclockwise", "translate 7 left"])
    assert et.steps_match(copy.deepcopy(gold), gold)                                   # correct -> True
    wrong_param = [{"type": "rotation", "param": "rot_180"}, {"type": "translation", "dx": -7, "dy": 0}]
    assert not et.steps_match(wrong_param, gold)                                       # wrong step1 param -> False


def test_steps_match_exact_both_directions_translation():
    gold = et.seq_enum(["reflect across x axis", "translate by (3, -4)"])
    assert et.steps_match(copy.deepcopy(gold), gold)                                   # correct -> True
    wrong_dx = [{"type": "reflection", "param": "reflect_x"}, {"type": "translation", "dx": 2, "dy": -4}]
    assert not et.steps_match(wrong_dx, gold)                                          # wrong dx -> False
    wrong_dy = [{"type": "reflection", "param": "reflect_x"}, {"type": "translation", "dx": 3, "dy": -5}]
    assert not et.steps_match(wrong_dy, gold)                                          # wrong dy -> False


def test_steps_match_rejects_wrong_length_and_malformed():
    gold = et.seq_enum(["rotate 90 degrees counterclockwise", "translate 7 left"])
    assert not et.steps_match(gold[:1], gold)                          # wrong length
    assert not et.steps_match("not a list", gold)                      # not a sequence
    assert not et.steps_match([{"param": "rot_ccw_90"}, gold[1]], gold)  # missing type key


# --------------------------------------------------------------------------------------
# 3. Eval dispatch + backward compatibility
# --------------------------------------------------------------------------------------

def test_eval_prose_regime_unchanged_backward_compat():
    # v1–v4 prose net-map metric is untouched: a sanctioned cw wording variant still passes,
    # a genuinely different motion still fails.
    gold = ["rotate 90 degrees counterclockwise", "translate 7 right"]
    assert ev._transform_match(["rotate 270 degrees clockwise", "translate 7 right"], gold)
    assert not ev._transform_match(["rotate 180 degrees counterclockwise", "translate 7 right"], gold)


def test_eval_enum_regime_recovers_semantic_net_map():
    gold_enum = et.seq_enum(["rotate 90 degrees counterclockwise", "translate 7 left"])
    assert ev._transform_match(copy.deepcopy(gold_enum), gold_enum)
    wrong = [{"type": "rotation", "param": "rot_180"}, {"type": "translation", "dx": -7, "dy": 0}]
    assert not ev._transform_match(wrong, gold_enum)


def test_eval_enum_prediction_vs_prose_oracle_is_the_v5_eval_path():
    # The REAL v5 eval: the model emits ENUM, the frozen oracle record is PROSE. The harness
    # composes both representations and compares the canonical NET affine map.
    gold_prose = ["rotate 90 degrees counterclockwise", "translate 7 left"]
    good_enum = et.seq_enum(gold_prose)
    assert ev._transform_match(good_enum, gold_prose)
    bad_enum = [{"type": "rotation", "param": "rot_180"}, {"type": "translation", "dx": -7, "dy": 0}]
    assert not ev._transform_match(bad_enum, gold_prose)


# --------------------------------------------------------------------------------------
# 4. Exact, NOT looser — the transparency point for the writeup
# --------------------------------------------------------------------------------------

def test_net_headline_and_step_exact_are_explicitly_separate():
    # Two DIFFERENT per-step decompositions with the SAME net map (an OOD reflect∘rotate
    # composition). The prose net-map metric accepts the alternate; the enum metric does not.
    gold = ["reflect across x axis", "rotate 90 degrees counterclockwise"]
    alt = ["reflect across y axis", "rotate 270 degrees counterclockwise"]
    assert tc.compose(gold) == tc.compose(alt)                     # same net map
    assert ev._transform_match(alt, gold)                          # prose net headline passes
    assert ev._transform_match(et.seq_enum(alt), et.seq_enum(gold))  # enum net headline also passes
    assert ev._transform_match(et.seq_enum(alt), gold)             # v5 stays comparable to v1-v4
    assert not ev._step_sequence_exact_match(et.seq_enum(alt), gold)  # strict diagnostic fails
    assert ev._step_sequence_exact_match(et.seq_enum(gold), gold)


# --------------------------------------------------------------------------------------
# 5. v5 CoT target — scores all-pass against the PROSE oracle; wrong param is caught
# --------------------------------------------------------------------------------------

def test_v5_target_scores_all_pass_for_every_label_against_prose_oracle():
    recs = _records()
    seen = set()
    for rec in recs:
        if rec["label"] in seen:
            continue
        seen.add(rec["label"])
        row = ev.score_record(cot.cot_target(rec, enum_transform=True), rec)
        assert row["parse_ok"] and row["label_ok"], rec["label"]
        assert row["transform_ok"], rec["label"]           # enum pred vs prose oracle -> exact match
        assert row["hint_ok"], (rec["label"], rec["hint"])
        assert row["failure_reason"] == "", rec["label"]
    assert seen == set(tc.DIAGNOSIS_LABELS)


def test_v5_wrong_param_prediction_fails_transform_but_not_label():
    for label in ("wrong_rotation_angle", "wrong_translation"):
        rec = _one(label)
        gold = ev.parse_pred(cot.cot_target(rec, enum_transform=True))
        bad = copy.deepcopy(gold)
        for step in bad["correct_transform"]:
            if step["type"] in ("rotation", "reflection"):
                step["param"] = "rot_180" if step["param"] != "rot_180" else "rot_ccw_90"
            elif step["type"] == "translation":
                step["dx"] = step["dx"] + 1
        row = ev.score_record(json.dumps(bad), rec)
        assert row["label_ok"], label                       # label untouched
        assert not row["transform_ok"], label               # exact enum mismatch
        assert row["failure_reason"] == "transform_mismatch", label


# --------------------------------------------------------------------------------------
# 6. v5 target structure — enum correct_transform, kept v4 fields, transform_first ordering
# --------------------------------------------------------------------------------------

def test_v5_target_obj_swaps_transform_to_enum_and_keeps_v4_fields():
    rec = _one("reflection_instead_of_rotation")
    obj = cot.enum_target_obj(rec)
    assert list(obj.keys())[:3] == ["label", "correct_transform", "hint"]
    assert et.is_enum_seq(obj["correct_transform"])
    assert obj["correct_transform"] == et.seq_enum(rec["correct_transform"])
    # v4 structured fields retained and consistent with the oracle transforms.
    assert obj["expected_operation_types"] == cot.operation_types(rec["correct_transform"])
    assert obj["student_operation_types"] == cot.operation_types(rec["student_transform"])
    assert obj["main_mismatch"] == cot.main_mismatch(rec)
    # label + hint identical to the prose target's.
    assert obj["label"] == rec["label"] and obj["hint"] == rec["hint"]


def test_transform_first_knob_foregrounds_correct_transform():
    rec = _one("wrong_reflection_line")
    default = cot.enum_target_obj(rec, transform_first=False)
    first = cot.enum_target_obj(rec, transform_first=True)
    assert list(default.keys())[0] == "label"
    assert list(first.keys())[0] == "correct_transform"
    # Reordering is content-preserving (scored by key), so the two carry identical content.
    assert dict(first) == dict(default)


def test_v5_trace_has_readout_line_and_concludes_with_label():
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        trace = cot.reasoning_trace(rec, enum_transform=True)
        assert "Transform readout:" in trace, label
        assert f"the diagnosis is {label}" in trace, label
        # The readout names the enum params for this record's correct transform.
        for step in et.seq_enum(rec["correct_transform"]):
            assert et.describe_step(step) in trace, (label, step)


def test_v5_target_parses_to_final_json_with_enum_transform():
    rec = _one("completely_wrong")
    target = cot.cot_target(rec, enum_transform=True)
    parsed = ev.parse_pred(target)                       # pulls the LAST JSON object
    assert parsed is not None
    assert parsed["label"] == rec["label"]
    assert et.is_enum_seq(parsed["correct_transform"])
    assert parsed["correct_transform"] == et.seq_enum(rec["correct_transform"])


# --------------------------------------------------------------------------------------
# 7. Backward-compat guard — prose v3cot / v4 targets still score as before
# --------------------------------------------------------------------------------------

def test_prose_cot_targets_still_score_all_pass():
    # The enum wiring must not disturb the existing prose (net-map) scoring for any label.
    for label in tc.DIAGNOSIS_LABELS:
        rec = _one(label)
        for target in (chat_format.target_json(rec),         # v1/v3cot JSON
                       cot.cot_target(rec),                   # v3cot trace + prose JSON
                       cot.cot_target(rec, structured=True)): # v4 structured + prose JSON
            row = ev.score_record(target, rec)
            assert row["transform_ok"], (label, target[:60])
            assert row["failure_reason"] == "", label
