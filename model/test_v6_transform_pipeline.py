"""GPU/API-free tests for the v6 data, train, eval, and gateway plumbing."""

from __future__ import annotations

import json
import random

import pytest

from model import eval_frontier_gateway as gateway
from model import eval_transform
from model import eval_tuned_coords
from model import make_v6_transform_data as make_v6
from model import train_transform


def test_mix_plan_is_exact_and_contrastive_uses_whole_groups():
    for n in range(40):
        counts = make_v6.plan_counts(n, (0.5, 0.2, 0.15, 0.15))
        assert sum(counts.values()) == n
        assert counts["contrastive"] % 4 == 0
        assert all(value >= 0 for value in counts.values())


def test_nonempty_output_requires_matching_safe_resume(tmp_path):
    config = {"schema_version": "v6.net-affine.1", "seed": 1}
    source = {"train.jsonl": {"sha256": "abc"}}
    (tmp_path / "manifest_v6.json").write_text(json.dumps({
        "generation_config": config,
        "source": {"before": source},
    }))
    with pytest.raises(SystemExit):
        make_v6._guard_output(str(tmp_path), config, source, False)
    make_v6._guard_output(str(tmp_path), config, source, True)
    with pytest.raises(SystemExit):
        make_v6._guard_output(
            str(tmp_path), config, {"train.jsonl": {"sha256": "changed"}}, True
        )


def test_hard_focus_cycle_covers_requested_parameter_boundaries():
    built = make_v6.build_hard_partials(random.Random(20260711), len(make_v6.HARD_SPECS))
    names = [name for _, name in built]
    assert names == [spec[0] for spec in make_v6.HARD_SPECS]
    assert {"rotation_angle_90", "rotation_angle_180", "rotation_angle_270"} <= set(names)
    assert "translation_parameter_off_by_one" in names
    assert "operation_type_rotation_to_reflection" in names
    assert "operation_type_reflection_to_rotation" in names
    assert sum(name.startswith("reflection_line_") for name in names) == 4


def test_train_rehearsal_assembly_is_deterministic(tmp_path):
    data_dir = str(tmp_path)
    for stage in train_transform.STAGES:
        path = tmp_path / f"train_v6_image_{stage}_chat.jsonl"
        rows = [
            {"id": f"{stage}-{i}", "messages": []}
            for i in range(10)
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    full = str(tmp_path / "train_v6_image_full_chat.jsonl")
    first, counts1 = train_transform.assemble_rows(
        full, data_dir, "image", "full",
        limit=8, rehearsal_ratio=0.25, seed=7,
    )
    second, counts2 = train_transform.assemble_rows(
        full, data_dir, "image", "full",
        limit=8, rehearsal_ratio=0.25, seed=7,
    )
    assert [row["id"] for row in first] == [row["id"] for row in second]
    assert counts1 == counts2
    assert counts1["full"] == 8
    assert len(first) > 8


def test_eval_sampling_uses_fixed_seed():
    records = {index: {"id": index} for index in range(100)}
    first = eval_transform.select_ids(records, 12, 20260709, 0)
    second = eval_transform.select_ids(records, 12, 20260709, 0)
    assert first == second and len(first) == 12
    assert eval_transform.select_ids(records, 12, 20260709, 3) == first[:3]
    assert first == eval_tuned_coords.select_ids(records, 12, 20260709, 0)


def test_gateway_v6_payload_uses_canonical_schema_and_image_coords(tmp_path):
    image = tmp_path / "one.png"
    image.write_bytes(b"not-decoded-by-payload-builder")
    rec = {
        "original": [[0, 0], [2, 0], [1, 1]],
        "correct_image": [[1, 0], [3, 0], [2, 1]],
        "student_image": [[0, 1], [2, 1], [1, 2]],
    }
    chat = {
        "messages": [{
            "content": [{"type": "image", "image": image.name}],
        }],
    }
    payload = gateway.build_responses_input(
        rec, chat, str(tmp_path), "image_coords", "v6", "both"
    )
    text = payload[0]["content"][0]["text"]
    assert "correct_net" in text and "student_net" in text
    assert "rot_ccw_90" in text
    assert "Exact vertices in corresponding order" in text
    assert payload[0]["content"][1]["image_url"].startswith("data:image/png;base64,")


def test_gateway_image_coords_keeps_render_bearing_chat_row():
    chat = {17: {"id": 17, "messages": []}}
    assert gateway._chat_row_for_input(chat, 17, "image") is chat[17]
    assert gateway._chat_row_for_input(chat, 17, "image_coords") is chat[17]
    assert gateway._chat_row_for_input(chat, 17, "coords") is None


def test_gateway_legacy_builder_default_is_unchanged_for_coords():
    rec = {
        "original": [[0, 0]],
        "correct_image": [[1, 0]],
        "student_image": [[0, 1]],
    }
    assert gateway.build_responses_input(rec, None, ".", "coords") == gateway.cf.coords_prompt(rec)
