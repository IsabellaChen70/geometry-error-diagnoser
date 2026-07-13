"""Offline v6 re-score regression tests (no model, GPU, or API)."""

from __future__ import annotations

import json
from pathlib import Path

from model import rescore_records
from transform_diagnosis import dataset, v6_format


def _record() -> dict:
    records, _ = dataset.build_records(20260711, 80, 10, ood_per_label=0)
    record = v6_format.augment_record(records[0])
    record["split"] = "test"
    return record


def test_rescore_v6_full_uses_raw_output_and_preserves_input(tmp_path):
    record = _record()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.jsonl").write_text(json.dumps(record) + "\n")

    pred = v6_format.target_obj(record, "full")
    pred["label"] = "rotation_instead_of_rotation"
    saved = {
        "id": record["id"],
        "split": "test",
        "true_label": record["label"],
        "pred_label": "PARSE_FAIL",
        "task_mode": "legacy",
        "parse_ok": False,
        "label_ok": False,
        "transform_ok": False,
        "correct_net_ok": False,
        "student_net_ok": False,
        "both_nets_ok": False,
        "derived_label_ok": None,
        "hint_ok": False,
        "raw_model_output": json.dumps(pred),
        "failure_reason": "parse_fail",
    }
    records_path = tmp_path / "records_frontier_v6_opus_image_n150_test.jsonl"
    original = json.dumps(saved) + "\n"
    records_path.write_text(original)

    _, after = rescore_records.rescore_file(
        str(records_path), str(data_dir), task_mode="full"
    )

    assert records_path.read_text() == original
    results_path, rescored_path = rescore_records.out_paths(str(records_path))
    rescored = json.loads(Path(rescored_path).read_text().splitlines()[0])
    aggregate = json.loads(Path(results_path).read_text())
    assert rescored["task_mode"] == "full"
    assert rescored["parse_ok"] is True
    assert rescored["correct_net_ok"] is True
    assert rescored["student_net_ok"] is True
    assert rescored["both_nets_ok"] is True
    assert rescored["label_ok"] is False
    assert rescored["pred_label"] == "rotation_instead_of_rotation"
    assert rescored["failure_reason"] == "invalid_label:rotation_instead_of_rotation"
    assert aggregate == after
    assert after["parse_rate"] == 1.0
    assert after["student_net_match_rate"] == 1.0
    assert after["label_accuracy"] == 0.0


def test_rescore_reuses_saved_task_mode_without_override():
    assert rescore_records._row_task_mode({"task_mode": "full"}, None) == "full"
    assert rescore_records._row_task_mode({"task_mode": "legacy"}, "full") == "full"
    assert rescore_records._row_task_mode({}, None) is None
