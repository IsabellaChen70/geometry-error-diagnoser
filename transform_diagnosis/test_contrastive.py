"""Tests for the v4 generators (`contrastive.py`): hard contrastive quadruplets + single-step
curriculum records.

Both reuse the verified ``errors.inject`` loop, so the key guarantees are inherited (label ==
diagnose of the stored transforms). These tests assert the v4-specific structure:

  * Contrastive quadruplet — ONE shared RED, ONE shared translation, the FOUR confusable
    labels, with each rotate/reflect FAMILY sharing its intended transform (matched sets).
  * Curriculum — genuinely SIMPLER: single-step correct transform + single-step error.
  * Both — records finalize (pass ``dataset._assert_record``) and their structured-CoT
    targets score all-pass through the fixed eval harness.
"""

from __future__ import annotations

import json
import random

from transform_diagnosis import chat_format, contrastive, cot, dataset, errors
from transform_diagnosis import eval as ev
from transform_diagnosis import transform_core as tc

SEED = 20260711


def _finalize(partials, split="train", subdir="renders_v4", base=500000):
    return [dataset.finalize_record(p, base + i, split, subdir) for i, p in enumerate(partials)]


def _net_translation(transform_text):
    """Net translation vector of a stored transform-text list (via the single oracle)."""
    return tc.compose(transform_text).vec


# --------------------------------------------------------------------------------------
# Contrastive quadruplets
# --------------------------------------------------------------------------------------

def test_contrastive_group_shares_red_and_covers_four_confusable_labels():
    g = contrastive.build_contrastive_group(random.Random(SEED))
    assert len(g.partials) == 4
    red = [list(p) for p in g.original]
    assert all(p["original"] == red for p in g.partials), "RED not shared across the quadruplet"
    assert {p["label"] for p in g.partials} == set(contrastive.CONFUSABLE_LABELS)


def test_contrastive_families_share_intended_transform():
    g = contrastive.build_contrastive_group(random.Random(SEED))
    by_label = {p["label"]: p for p in g.partials}
    # rotation family (reflection_instead_of_rotation, wrong_rotation_angle) shares correct
    rot = [by_label[l]["correct_transform"] for l in contrastive._ROTATION_BASE_LABELS]
    assert rot[0] == rot[1], "rotation-family pair does not share the intended transform"
    ref = [by_label[l]["correct_transform"] for l in contrastive._REFLECTION_BASE_LABELS]
    assert ref[0] == ref[1], "reflection-family pair does not share the intended transform"
    # the two families' correct nets differ in orientation (rotation vs reflection) -- the
    # geometric reason a single correct transform cannot realize all four labels.
    assert tc.compose(rot[0]).det() == 1 and tc.compose(ref[0]).det() == -1


def test_contrastive_shares_one_translation_across_all_four():
    g = contrastive.build_contrastive_group(random.Random(SEED))
    vecs = {_net_translation(p["correct_transform"]) for p in g.partials}
    assert len(vecs) == 1, f"the four records do not share one translation: {vecs}"


def test_contrastive_uses_in_distribution_patterns_only():
    # The correct transforms are ("rotate","translate") / ("reflect","translate"), both
    # IN-distribution -- so contrastive records never leak into the OOD (held-out) slice.
    g = contrastive.build_contrastive_group(random.Random(SEED))
    for p in g.partials:
        kinds = tuple(_kind(s) for s in p["correct_transform"])
        assert kinds not in errors.HELD_OUT_PATTERNS, kinds
        assert kinds in errors.IN_DISTRIBUTION_PATTERNS, kinds


def _kind(text):
    t = tc.as_transform(text)
    if t.matrix == tc.IDENTITY_MATRIX:
        return "translate"
    return "rotate" if t.det() == 1 else "reflect"


def test_contrastive_records_finalize_and_score_all_pass():
    flat, groups = contrastive.build_contrastive_partials(random.Random(SEED), 3)
    assert len(groups) == 3 and len(flat) == 12
    recs = _finalize(flat)                       # finalize_record runs _assert_record on each
    for rec in recs:
        row = ev.score_record(cot.cot_target(rec, structured=True), rec)
        assert row["parse_ok"] and row["label_ok"] and row["transform_ok"]
        assert row["hint_ok"] and row["failure_reason"] == "", rec["label"]


def test_contrastive_is_deterministic():
    a = contrastive.build_contrastive_group(random.Random(SEED))
    b = contrastive.build_contrastive_group(random.Random(SEED))
    assert [json.dumps(p, sort_keys=True) for p in a.partials] == \
           [json.dumps(p, sort_keys=True) for p in b.partials]


# --------------------------------------------------------------------------------------
# Curriculum (single-step) records
# --------------------------------------------------------------------------------------

def test_curriculum_records_are_single_step():
    parts = contrastive.build_curriculum_partials(random.Random(SEED), 18)
    assert len(parts) == 18
    for p in parts:
        assert len(p["correct_transform"]) == 1, p["correct_transform"]
        assert len(p["student_transform"]) == 1, p["student_transform"]


def test_curriculum_labels_are_the_expected_simple_set():
    parts = contrastive.build_curriculum_partials(random.Random(SEED), 27)  # 3 full round-robins
    labels = {p["label"] for p in parts}
    expected = {label for _pat, label in contrastive.CURRICULUM_SPECS}
    assert labels == expected
    assert "completely_wrong" not in labels          # excluded: not a "simpler" warm-up
    assert "correct" in labels


def test_curriculum_records_finalize_and_score_all_pass():
    parts = contrastive.build_curriculum_partials(random.Random(SEED), 18)
    recs = _finalize(parts)                          # _assert_record must pass for 1-step too
    for rec in recs:
        row = ev.score_record(cot.cot_target(rec, structured=True), rec)
        assert row["parse_ok"] and row["label_ok"] and row["transform_ok"]
        assert row["hint_ok"] and row["failure_reason"] == "", rec["label"]


def test_curriculum_uses_the_one_or_two_instruction():
    parts = contrastive.build_curriculum_partials(random.Random(SEED), 9)
    rec = dataset.finalize_record(parts[0], 500000, "train", "renders_v4")
    assert chat_format.instruction_for(rec) == chat_format.CURRICULUM_INSTRUCTION
    assert "one or two" in chat_format.CURRICULUM_INSTRUCTION
    assert "exactly two" not in chat_format.CURRICULUM_INSTRUCTION
    # and the two-step canonical instruction is unchanged (byte-stable for v1/v2/v3cot)
    assert "exactly two" in chat_format.INSTRUCTION and "two-step" in chat_format.INSTRUCTION


def test_curriculum_is_deterministic():
    a = contrastive.build_curriculum_partials(random.Random(SEED), 12)
    b = contrastive.build_curriculum_partials(random.Random(SEED), 12)
    assert [json.dumps(p, sort_keys=True) for p in a] == [json.dumps(p, sort_keys=True) for p in b]


# --------------------------------------------------------------------------------------
# finalize_record wrapper (schema + assertions), used by the v4 assembler
# --------------------------------------------------------------------------------------

def test_finalize_record_enforces_schema_and_render_path():
    parts = contrastive.build_curriculum_partials(random.Random(SEED), 3)
    rec = dataset.finalize_record(parts[0], 1234, "train", "renders_v4")
    assert list(rec.keys()) == dataset._SCHEMA_KEYS
    assert rec["id"] == 1234 and rec["split"] == "train"
    assert rec["render_path"] == "renders_v4/001234.png"
