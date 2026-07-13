"""Tests for the shared transform-first v6 prompts and targets."""

from __future__ import annotations

import json

import pytest

from transform_diagnosis import dataset, v6_format


def _record():
    records, _ = dataset.build_records(20260711, 80, 10, ood_per_label=0)
    return v6_format.augment_record(records[0])


@pytest.mark.parametrize(
    "task,keys",
    [
        ("correct", ("correct_net",)),
        ("student", ("student_net",)),
        ("both", ("correct_net", "student_net")),
        ("full", ("correct_net", "student_net", "label", "hint")),
    ],
)
def test_task_targets_have_exact_schema_and_round_trip(task, keys):
    rec = _record()
    target = v6_format.target_obj(rec, task)
    assert tuple(target) == keys
    assert json.loads(v6_format.target_json(rec, task)) == target
    assert v6_format.validate_target(target, task) == target


def test_augment_record_maps_match_images_and_label():
    rec = _record()
    assert rec["schema_version"] == v6_format.SCHEMA_VERSION
    assert set(rec["correct_net"]) == {"linear", "tx", "ty"}
    assert set(rec["student_net"]) == {"linear", "tx", "ty"}


def test_image_only_and_image_coords_are_distinct_in_distribution_prompts():
    rec = _record()
    image = v6_format.user_message(rec, "both", "image")
    coords = v6_format.user_message(rec, "both", "image_coords")
    assert [part["type"] for part in image["content"]] == ["image", "text"]
    assert [part["type"] for part in coords["content"]] == ["image", "text"]
    assert "Exact vertices in corresponding order" not in image["content"][1]["text"]
    assert "Exact vertices in corresponding order" in coords["content"][1]["text"]
    assert str(rec["original"][0]) in coords["content"][1]["text"]


def test_coordinates_only_frontier_prompt_has_no_image_part():
    rec = _record()
    message = v6_format.user_message(rec, "correct", "coords")
    assert [part["type"] for part in message["content"]] == ["text"]
    assert "Exact vertices" in message["content"][0]["text"]


def test_prompts_explicitly_reject_privileged_step_decomposition():
    rec = _record()
    for task in v6_format.TASK_MODES:
        prompt = v6_format.instruction(rec, task, "image")
        assert "Do not guess an ordered sequence of steps" in prompt
        assert "rot_ccw_90" in prompt and "reflect_y_eq_neg_x" in prompt


def test_target_validation_rejects_extra_or_misordered_fields():
    rec = _record()
    target = v6_format.target_obj(rec, "correct")
    with pytest.raises(ValueError):
        v6_format.validate_target({**target, "label": rec["label"]}, "correct")
    with pytest.raises(ValueError):
        v6_format.validate_target(
            {"student_net": rec["student_net"], "correct_net": rec["correct_net"]},
            "both",
        )
