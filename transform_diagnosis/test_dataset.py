"""Acceptance tests #2-#6 for the dataset generator.

  #2  same seed -> byte-identical JSONL (generate twice, diff)
  #3  2000 problems: correct grades True; integer + in-bounds; simple + asymmetric;
      image != original
  #4  uniqueness: recover_map(original, correct_image) == intended net map
  #5  every record: diagnose(original, correct_transform, student_transform) == label,
      and is_correct == (label == 'correct')
  #6  every label reaches its configured minimum count

(Acceptance test #1 is ``test_transform_core.py``.)
"""

from __future__ import annotations

import json
import random

from transform_diagnosis import dataset, geometry, problems, transform_core as tc

SEED = 20260708
MIN_COUNT = 20
N = 200


# --------------------------------------------------------------------------------------
# #2  Byte-identical re-run
# --------------------------------------------------------------------------------------

def test_acceptance_2_byte_identical_records():
    a, _ = dataset.build_records(SEED, N, MIN_COUNT)
    b, _ = dataset.build_records(SEED, N, MIN_COUNT)
    assert [json.dumps(r) for r in a] == [json.dumps(r) for r in b]


def test_acceptance_2_byte_identical_files(tmp_path):
    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    recs1, _ = dataset.build_records(SEED, N, MIN_COUNT)
    recs2, _ = dataset.build_records(SEED, N, MIN_COUNT)
    dataset.write_jsonl(recs1, str(d1))
    dataset.write_jsonl(recs2, str(d2))
    for name in ("data.jsonl", "train.jsonl", "val.jsonl", "test.jsonl"):
        assert (d1 / name).read_bytes() == (d2 / name).read_bytes(), name


# --------------------------------------------------------------------------------------
# #3 + #4  2000 problems: invariants and unique recovery
# --------------------------------------------------------------------------------------

def test_acceptance_3_and_4_two_thousand_problems():
    rng = random.Random(SEED)
    for _ in range(2000):
        p = problems.make_problem(rng)

        # correct transform grades True (via both Transform objects and schema strings)
        assert tc.grade(p.original, p.image, p.answer)
        assert tc.grade(p.original, p.image, p.answer_text)

        # all vertices integer and in-bounds [-10, 10]
        for poly in (p.original, p.image):
            for x, y in poly:
                assert isinstance(x, int) and isinstance(y, int)
                assert -10 <= x <= 10 and -10 <= y <= 10

        # original is a single simple, asymmetric polygon
        assert geometry.is_simple(p.original)
        assert tc.is_asymmetric(p.original)

        # image differs from original
        assert p.image != p.original

        # #4 uniqueness: recovered net map equals the intended net map
        assert tc.recover_map(p.original, p.image) == p.net()


# --------------------------------------------------------------------------------------
# #5 + #6  Record-level label agreement and balance
# --------------------------------------------------------------------------------------

def _iter_written_records(tmp_path):
    recs, _ = dataset.build_records(SEED, N, MIN_COUNT)
    dataset.write_jsonl(recs, str(tmp_path))
    with open(tmp_path / "data.jsonl") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_acceptance_5_every_record_label_matches_diagnosis(tmp_path):
    records = _iter_written_records(tmp_path)
    assert records
    for rec in records:
        d = tc.diagnose(rec["original"], rec["correct_transform"], rec["student_transform"])
        assert d == rec["label"], (rec["id"], d, rec["label"])
        assert rec["is_correct"] == (rec["label"] == "correct")
        # correct transform reproduces correct_image; student reproduces student_image
        assert tc.grade(rec["original"], rec["correct_image"], rec["correct_transform"])
        assert tc.grade(rec["original"], rec["student_image"], rec["student_transform"])
        assert rec["label"] in tc.DIAGNOSIS_LABELS


def test_acceptance_5_vertices_integer_and_in_bounds(tmp_path):
    records = _iter_written_records(tmp_path)
    for rec in records:
        for key in ("original", "correct_image", "student_image"):
            for x, y in rec[key]:
                assert isinstance(x, int) and isinstance(y, int)
                assert -10 <= x <= 10 and -10 <= y <= 10


def test_acceptance_6_every_label_reaches_min_count():
    records, targets = dataset.build_records(SEED, N, MIN_COUNT)
    counts = dataset.label_counts(records)
    for label in tc.DIAGNOSIS_LABELS:
        assert counts[label] >= MIN_COUNT, (label, counts[label])
        assert counts[label] == targets[label]
    assert sum(counts.values()) == len(records)


def test_split_is_deterministic_function_of_seed():
    # Same (seed, id) -> same split, independent of anything else.
    for rid in range(50):
        s1 = dataset.split_of(rid, SEED, (0.8, 0.1, 0.1))
        s2 = dataset.split_of(rid, SEED, (0.8, 0.1, 0.1))
        assert s1 == s2 and s1 in ("train", "val", "test")


def test_all_splits_present_and_cover_labels():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT)
    by_split = {}
    for r in records:
        by_split.setdefault(r["split"], set()).add(r["label"])
    assert set(by_split) == {"train", "val", "test"}
    # the training split should exercise every diagnosis label
    assert by_split["train"] == set(tc.DIAGNOSIS_LABELS)
