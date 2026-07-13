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

from transform_diagnosis import (
    chat_format, dataset, errors, geometry, hints, problems, transform_core as tc,
)
from transform_diagnosis import eval as ev

SEED = 20260708
MIN_COUNT = 20
N = 200
OOD = 12  # per OOD-eligible label; small for test speed (real runs use ~120)


def _move_kind(t: tc.Transform) -> str:
    """Classify a single primitive step as rotate / reflect / translate (via the math)."""
    if t.matrix == tc.IDENTITY_MATRIX:
        return "translate"
    return "rotate" if t.det() == 1 else "reflect"


def _pattern_of(rec: dict):
    """Recover the correct-answer composition pattern straight from the stored text."""
    return tuple(_move_kind(tc.parse_transform(s)) for s in rec["correct_transform"])


# --------------------------------------------------------------------------------------
# #2  Byte-identical re-run
# --------------------------------------------------------------------------------------

def test_acceptance_2_byte_identical_records():
    a, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    b, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    assert [json.dumps(r) for r in a] == [json.dumps(r) for r in b]


def test_acceptance_2_byte_identical_files(tmp_path):
    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    recs1, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    recs2, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    dataset.write_jsonl(recs1, str(d1))
    dataset.write_jsonl(recs2, str(d2))
    for name in ("data.jsonl", "train.jsonl", "val.jsonl", "test.jsonl", "ood.jsonl"):
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
    recs, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
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
    records, targets = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    # Balance is enforced over the IN-DISTRIBUTION set only; the OOD slice is
    # intentionally unbalanced (it covers only the held-out-compatible labels).
    id_records = [r for r in records if r["split"] != "ood"]
    counts = dataset.label_counts(id_records)
    for label in tc.DIAGNOSIS_LABELS:
        assert counts[label] >= MIN_COUNT, (label, counts[label])
        assert counts[label] == targets[label]
    assert sum(counts.values()) == len(id_records)


def test_split_is_deterministic_function_of_seed():
    # Same (seed, id) -> same split, independent of anything else.
    for rid in range(50):
        s1 = dataset.split_of(rid, SEED, (0.8, 0.1, 0.1))
        s2 = dataset.split_of(rid, SEED, (0.8, 0.1, 0.1))
        assert s1 == s2 and s1 in ("train", "val", "test")


def test_all_splits_present_and_cover_labels():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    by_split = {}
    for r in records:
        by_split.setdefault(r["split"], set()).add(r["label"])
    assert set(by_split) == {"train", "val", "test", "ood"}
    # the training split should exercise every diagnosis label
    assert by_split["train"] == set(tc.DIAGNOSIS_LABELS)


# --------------------------------------------------------------------------------------
# HINT field — the flipped contract. Every gold hint is a coordinate-free Socratic nudge
# that (a) references at least one correct operation family, (b) is NOT a strict answer
# leak (same definition as results/overnight/audit_hint_safety.py), and (c) is not a
# residual-coordinate leak per the eval scorer. This is the enforceable hint contract,
# checked across ALL 8 labels. (It replaces the old contract that required the exact
# answer tokens to be substrings of the hint — the very thing that leaked ~96% of the
# time in the frozen v6 run.)
# --------------------------------------------------------------------------------------

def test_every_gold_hint_is_family_relevant_and_not_a_leak():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    assert records
    seen = set()
    for rec in records:
        label = rec["label"]
        seen.add(label)
        assert isinstance(rec["hint"], str) and rec["hint"].strip(), rec["id"]
        families = hints.expected_hint_families(label, rec)
        assert families, (rec["id"], label)
        # (a) mentions at least one required operation family (the primary eval metric)
        assert ev._hint_mentions_family(rec["hint"], families), (rec["id"], label, rec["hint"])
        # (b) not a strict leak (the audit-equivalent check)
        leak = hints.strict_leak_reasons(rec["hint"], label, rec)
        assert not leak, (rec["id"], label, leak, rec["hint"])
        # (c) not a residual-coordinate leak per eval._hint_has_leak
        tokens = hints.expected_hint_tokens(label, rec)
        assert not ev._hint_has_leak(rec["hint"], tokens), (rec["id"], label, rec["hint"])
    # the sample must actually exercise every one of the eight labels
    assert seen == set(tc.DIAGNOSIS_LABELS)


def test_gold_hint_passes_family_metric_but_not_exact_token_metric():
    # The crux of the fix: gold hints pass the achievable family metric (hint_ok) but no
    # longer contain the exact answer tokens, so the strict hint_exact_ok is now False for
    # every label (it was True under the leaky contract).
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    for label in tc.DIAGNOSIS_LABELS:
        rec = next(r for r in records if r["label"] == label)
        row = ev.score_record(chat_format.target_json(rec), rec)
        assert row["hint_ok"] is True, (label, rec["hint"])
        assert row["hint_exact_ok"] is False, (label, rec["hint"])


def test_hint_is_after_label_in_schema_order():
    # "hint" is part of the schema and sits immediately after "label".
    assert "hint" in dataset._SCHEMA_KEYS
    assert dataset._SCHEMA_KEYS.index("hint") == dataset._SCHEMA_KEYS.index("label") + 1
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    assert list(records[0].keys()) == dataset._SCHEMA_KEYS


# --------------------------------------------------------------------------------------
# Compositional OOD split — train/val/test hold zero held-out compositions; the OOD
# slice holds ONLY held-out compositions and covers exactly the 4 expected labels.
# --------------------------------------------------------------------------------------

def test_train_val_test_have_no_held_out_compositions():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    id_records = [r for r in records if r["split"] in ("train", "val", "test")]
    assert id_records
    for rec in id_records:
        pat = _pattern_of(rec)
        assert pat not in errors.HELD_OUT_PATTERNS, (rec["id"], pat)
        assert pat in errors.IN_DISTRIBUTION_PATTERNS, (rec["id"], pat)


def test_ood_contains_only_held_out_compositions():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    ood = [r for r in records if r["split"] == "ood"]
    assert ood
    for rec in ood:
        assert _pattern_of(rec) in errors.HELD_OUT_PATTERNS, (rec["id"], _pattern_of(rec))


def test_all_labels_in_train_and_ood_covers_four():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    train_labels = {r["label"] for r in records if r["split"] == "train"}
    ood_labels = {r["label"] for r in records if r["split"] == "ood"}
    assert train_labels == set(tc.DIAGNOSIS_LABELS)
    assert ood_labels == {
        "correct", "rotation_instead_of_reflection",
        "wrong_reflection_line", "completely_wrong",
    }
    assert ood_labels == set(errors.OOD_ELIGIBLE_LABELS)


def test_ood_split_count_matches_configuration():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    ood = [r for r in records if r["split"] == "ood"]
    assert len(ood) == OOD * len(errors.OOD_ELIGIBLE_LABELS)
    counts = dataset.label_counts(ood)
    for label in errors.OOD_ELIGIBLE_LABELS:
        assert counts[label] == OOD, (label, counts[label])


# --------------------------------------------------------------------------------------
# completely_wrong must span BOTH flavors (guards against a monoculture). `diagnose` is
# unchanged (frozen contract in test_transform_core.py); these only assert its behavior
# on flavor-(b) inputs and that the generated class actually contains both flavors.
# --------------------------------------------------------------------------------------

_ANY_ORIGINAL = [[0, 0], [2, 0], [0, 1]]  # diagnose ignores original; net maps carry all info


def test_diagnose_completely_wrong_flavor_b_rotations():
    # Two proper rotations, different angle, different net slide -> same det (+1), so this
    # is a flavor-(b) same-orientation compound, which must diagnose as completely_wrong.
    correct = ["rotate 90 degrees counterclockwise", "translate 7 right"]
    student = ["rotate 180 degrees counterclockwise", "translate 2 up"]
    assert tc.diagnose(_ANY_ORIGINAL, correct, student) == "completely_wrong"


def test_diagnose_completely_wrong_flavor_b_reflections():
    # Two reflections, different mirror line, different net slide -> same det (-1);
    # flavor-(b) same-orientation compound, must diagnose as completely_wrong.
    correct = ["reflect across x axis", "translate 7 right"]
    student = ["reflect across y axis", "translate 2 up"]
    assert tc.diagnose(_ANY_ORIGINAL, correct, student) == "completely_wrong"


def test_completely_wrong_spans_both_flavors_in_dataset():
    # Among generated completely_wrong records, BOTH must appear:
    #   - at least one where correct and student net maps share a determinant (flavor b), and
    #   - at least one where their determinants differ (flavor a).
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    cw = [r for r in records if r["label"] == "completely_wrong"]
    assert cw, "no completely_wrong records generated"
    same_det = diff_det = 0
    for r in cw:
        dc = tc.compose(r["correct_transform"]).det()
        ds = tc.compose(r["student_transform"]).det()
        if dc == ds:
            same_det += 1
        else:
            diff_det += 1
    assert same_det >= 1, "no flavor-(b) same-orientation completely_wrong (monoculture!)"
    assert diff_det >= 1, "no flavor-(a) cross-orientation completely_wrong"


def test_student_image_never_equals_original():
    # A student's answer must always visibly differ from the untouched original (guards
    # against degenerate attempts whose net collapses to the identity).
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    offenders = [r["id"] for r in records if r["student_image"] == r["original"]]
    assert not offenders, f"student_image == original for ids {offenders[:10]}"


# --------------------------------------------------------------------------------------
# CHAT FORMAT — the Qwen3-VL training conversation built from each record. Every
# assistant turn must be valid JSON that round-trips to the record's own fields, and the
# instruction must advertise exactly the DIAGNOSIS_LABELS vocabulary (no drift).
# --------------------------------------------------------------------------------------

def test_chat_instruction_lists_exactly_the_diagnosis_labels():
    # The label vocabulary is derived from transform_core, so it cannot drift.
    for label in tc.DIAGNOSIS_LABELS:
        assert label in chat_format.INSTRUCTION, label
    # And it names the three-colour key the model relies on to read the image.
    for token in ("RED", "GREEN", "BLUE"):
        assert token in chat_format.INSTRUCTION, token


def test_chat_conversation_shape_and_image_reference():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    for rec in records:
        conv = chat_format.to_conversation(rec)
        assert conv["id"] == rec["id"] and conv["split"] == rec["split"]
        msgs = conv["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant"]
        # user turn = one image part (pointing at the record's render) + one text part
        parts = msgs[0]["content"]
        img = [p for p in parts if p["type"] == "image"]
        txt = [p for p in parts if p["type"] == "text"]
        assert len(img) == 1 and img[0]["image"] == rec["render_path"], rec["id"]
        assert len(txt) == 1 and txt[0]["text"] == chat_format.INSTRUCTION
        # assistant turn = exactly one text part
        ans = msgs[1]["content"]
        assert len(ans) == 1 and ans[0]["type"] == "text"


def test_chat_assistant_target_parses_and_matches_record():
    records, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    for rec in records:
        target = chat_format.to_messages(rec)[1]["content"][0]["text"]
        obj = json.loads(target)  # must be valid JSON and nothing else
        assert list(obj.keys()) == list(chat_format.TARGET_KEYS), rec["id"]
        assert obj["label"] == rec["label"] and obj["label"] in tc.DIAGNOSIS_LABELS
        assert obj["correct_transform"] == rec["correct_transform"]
        assert isinstance(obj["hint"], str) and obj["hint"].strip()


def test_chat_splits_written_and_counts_match(tmp_path):
    recs, _ = dataset.build_records(SEED, N, MIN_COUNT, ood_per_label=OOD)
    written = chat_format.write_chat_splits(recs, str(tmp_path))
    for split in ("train", "val", "test", "ood"):
        raw = sum(1 for r in recs if r["split"] == split)
        with open(written[f"{split}_chat"]) as f:
            rows = [json.loads(line) for line in f if line.strip()]
        assert len(rows) == raw, (split, len(rows), raw)
        assert all(r["split"] == split for r in rows), split
        assert all("messages" in r for r in rows), split
