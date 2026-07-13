"""CPU-only tests for the fast hint-rewrite path (no GPU, no rendering)."""

from __future__ import annotations

import json
import os

from model import rewrite_hints_fast as rhf
from transform_diagnosis import dataset, v6_format

# A deliberately leaky "old" gold hint: names the exact canonical maps AND a raw
# coordinate pair, so both hints.is_strict_leak and eval._hint_has_leak fire.
LEAKY_HINT = (
    "The correct step is rotate 90 degrees counterclockwise, not reflect across the "
    "x axis; slide by (3, 4)."
)


def _write_v6_dir(tmp_path, *, leaky: bool = True):
    records, _ = dataset.build_records(20260711, 8, 1, ood_per_label=0)
    records = records[:4]
    v6_dir = tmp_path / "v6"
    v6_dir.mkdir()
    (v6_dir / "renders_v6").mkdir()
    (v6_dir / "renders_v6" / "train").mkdir()

    prepared = []
    for new_id, source in enumerate(records):
        rec = v6_format.augment_record(source)
        rec = dict(rec)
        rec["id"] = new_id
        rec["split"] = "train"
        rec["render_path"] = f"renders_v6/train/{new_id:06d}.png"
        if leaky:
            rec["hint"] = LEAKY_HINT
        prepared.append(rec)

    with open(v6_dir / "train_v6.jsonl", "w", encoding="utf-8") as handle:
        for rec in prepared:
            handle.write(json.dumps(rec, ensure_ascii=False) + "\n")

    for modality in rhf.MODALITIES:
        # full targets (carry the leaky hint) + one non-hint task file (must be replicated).
        with open(v6_dir / rhf.full_chat_name("train", modality), "w", encoding="utf-8") as handle:
            for rec in prepared:
                row = v6_format.conversation(rec, "full", modality)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(v6_dir / f"train_v6_{modality}_both_chat.jsonl", "w", encoding="utf-8") as handle:
            for rec in prepared:
                row = v6_format.conversation(rec, "both", modality)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(v6_dir)


def test_canonical_serialization_matches_make_v6_separators():
    inner = {"correct_net": {"linear": "rot_180", "tx": 0, "ty": 1}, "hint": "x"}
    assert rhf.canonical_inner(inner) == json.dumps(
        inner, ensure_ascii=False, separators=(",", ":")
    )
    row = {"id": 0, "messages": []}
    assert rhf.canonical_outer(row) == json.dumps(row, ensure_ascii=False)


def test_rewrite_changes_only_hint_and_removes_leak(tmp_path):
    v6_dir = _write_v6_dir(tmp_path, leaky=True)
    out_dir = str(tmp_path / "out")

    # The leaky source must fail verification.
    assert rhf.verify(v6_dir) is False

    assert rhf.main(["--v6-dir", v6_dir, "--out-dir", out_dir,
                     "--rewrite-jsonl-hints", "--paranoid", "--verify"]) == 0

    # renders are reused via a symlink (never copied / re-rendered).
    assert os.path.islink(os.path.join(out_dir, "renders_v6"))
    # the non-hint task file is replicated.
    assert os.path.exists(os.path.join(out_dir, "train_v6_image_both_chat.jsonl"))

    for modality in rhf.MODALITIES:
        name = rhf.full_chat_name("train", modality)
        with open(os.path.join(v6_dir, name), encoding="utf-8") as handle:
            orig = [line for line in handle if line.strip()]
        with open(os.path.join(out_dir, name), encoding="utf-8") as handle:
            new = [line for line in handle if line.strip()]
        assert len(orig) == len(new) and orig  # non-empty, aligned
        for before, after in zip(orig, new):
            assert before != after  # hint actually changed
            assert rhf.diff_is_hint_only(before, after)  # ONLY the hint changed

    # The rewritten output passes the strict + eval leak checks.
    assert rhf.verify(out_dir) is True


def test_refuses_nonempty_out_dir_without_overwrite(tmp_path):
    v6_dir = _write_v6_dir(tmp_path, leaky=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "sentinel").write_text("keep")
    try:
        rhf.main(["--v6-dir", v6_dir, "--out-dir", str(out_dir)])
    except SystemExit as exc:
        assert exc.code != 0
    else:  # pragma: no cover - should not reach
        raise AssertionError("expected SystemExit for nonempty out-dir")
