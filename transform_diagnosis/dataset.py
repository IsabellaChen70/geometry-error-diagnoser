"""dataset — assemble diagnosis records, balance labels, split, and write JSONL.

Every record is built from a verified error injection, so the stored ``label`` always
equals ``transform_core.diagnose(original, correct_transform, student_transform)``.
This is re-asserted for every record at write time (``_assert_record``) so a bad record
fails loudly rather than silently shipping.

Determinism: all randomness flows through one ``random.Random(seed)`` (plus a second
RNG, deterministically derived from the same seed, used only to interleave labels).
The train/val/test split is a pure integer function of ``(seed, id)``. No global
``random``, no ``time``, no ``uuid``. Given a seed the JSONL is byte-for-byte reproducible.
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, List, Sequence, Tuple

from . import chat_format, errors, eval as eval_mod, geometry, hints, problems, transform_core as tc

BOUND = geometry.DEFAULT_BOUND

# Field order matches the record schema exactly (dicts preserve insertion order).
# "hint" sits right after "label" — the record's assistant target is conceptually
# label + hint.
_SCHEMA_KEYS = [
    "id", "num_vertices", "original", "correct_transform", "correct_image",
    "student_transform", "student_image", "label", "hint", "is_correct", "split",
    "render_path",
]

# A second RNG stream, derived deterministically from the seed, interleaves the
# label-grouped records so the file (and every split) mixes all labels. A third,
# separately salted stream interleaves the OOD slice the same way.
_SHUFFLE_SALT = 0x5F3759DF
_OOD_SHUFFLE_SALT = 0x9E3779B9
_SPLIT_SALT = 1013904223


def compute_targets(n: int, min_count: int, labels: Sequence[str]) -> Dict[str, int]:
    """Per-label target counts: at least ``min_count`` each, extra distributed round-robin
    (in label order) up to a total of ``n``. If ``n < min_count * len(labels)`` the floor
    wins and the effective total is ``min_count * len(labels)``."""
    targets = {l: min_count for l in labels}
    extra = max(0, n - min_count * len(labels))
    i = 0
    while extra > 0:
        targets[labels[i % len(labels)]] += 1
        i += 1
        extra -= 1
    return targets


def split_of(rid: int, seed: int, fracs: Tuple[float, float, float]) -> str:
    """Deterministic train/val/test assignment — pure integer function of (seed, id)."""
    f_train, f_val, _ = fracs
    t1 = round(f_train * 100)
    t2 = round((f_train + f_val) * 100)
    h = (rid * 2654435761 + seed * 40503 + _SPLIT_SALT) % 100
    if h < t1:
        return "train"
    if h < t2:
        return "val"
    return "test"


def _partial_record(problem: problems.Problem, student_seq, student_text, label: str) -> dict:
    correct_image = tc.compose(problem.answer).apply(problem.original)
    student_image = tc.compose(student_seq).apply(problem.original)
    correct_text = list(problem.answer_text)
    student_text = list(student_text)
    # Hint wording is derived from the stored transforms via transform_core (in hints.py),
    # so it always names the correct/student axis, angle, or translation for this record.
    hint = hints.hint_for(label, {
        "label": label,
        "correct_transform": correct_text,
        "student_transform": student_text,
    })
    return {
        "num_vertices": problem.num_vertices,
        "original": [[x, y] for x, y in problem.original],
        "correct_transform": correct_text,
        "correct_image": [[x, y] for x, y in correct_image],
        "student_transform": student_text,
        "student_image": [[x, y] for x, y in student_image],
        "label": label,
        "hint": hint,
        "is_correct": (label == "correct"),
    }


def _assert_record(rec: dict) -> None:
    """Fail loudly if any invariant is violated (acceptance-grade checks, per record)."""
    assert rec["label"] in tc.DIAGNOSIS_LABELS, rec["label"]
    # Label must equal an independent diagnosis of the STORED text lists.
    d = tc.diagnose(rec["original"], rec["correct_transform"], rec["student_transform"])
    assert d == rec["label"], f"label {rec['label']} != diagnose {d}"
    assert rec["is_correct"] == (rec["label"] == "correct")
    # Hint contract (flipped from the old leaky one that required the exact answer
    # tokens): the gold hint must be a non-empty Socratic nudge that references the
    # correct operation family/families for this label AND must NOT leak the answer.
    # This is the SAME family metric eval scores (hint_ok) and the SAME strict-leak
    # definition as results/overnight/audit_hint_safety.py, so gold data cannot ship
    # a hint that would fail the audit.
    assert isinstance(rec["hint"], str) and rec["hint"].strip(), "empty hint"
    assert eval_mod._hint_mentions_family(
        rec["hint"], hints.expected_hint_families(rec["label"], rec)
    ), ("hint misses required operation family", rec["label"], rec["hint"])
    leak_reasons = hints.strict_leak_reasons(rec["hint"], rec["label"], rec)
    assert not leak_reasons, ("hint leaks the answer", rec["label"], leak_reasons, rec["hint"])
    # Correct transform actually maps original -> correct_image.
    assert tc.grade(rec["original"], rec["correct_image"], rec["correct_transform"])
    # Student transform actually maps original -> student_image.
    assert tc.grade(rec["original"], rec["student_image"], rec["student_transform"])
    # Integer + in-bounds vertices everywhere.
    for key in ("original", "correct_image", "student_image"):
        for x, y in rec[key]:
            assert isinstance(x, int) and isinstance(y, int), (key, x, y)
            assert -BOUND <= x <= BOUND and -BOUND <= y <= BOUND, (key, x, y)
    # Original is a valid single irregular polygon and uniquely identifiable.
    assert geometry.is_simple(rec["original"])
    assert tc.is_asymmetric(rec["original"])
    assert rec["correct_image"] != rec["original"]
    # A student's answer must never coincide with the untouched original (degenerate).
    assert rec["student_image"] != rec["original"], "student answer coincides with original"
    # Uniqueness: the recovered net map equals the intended net map.
    rec_map = tc.recover_map(rec["original"], rec["correct_image"])
    assert rec_map == tc.compose(rec["correct_transform"]), "recover_map mismatch"


def _inject_partials(
    label: str,
    need: int,
    patterns: Sequence[Tuple[str, ...]],
    rng: random.Random,
    max_attempts_factor: int,
) -> List[dict]:
    """Generate ``need`` verified partial records for ``label`` from ``patterns``.

    Shared by the in-distribution and OOD generation loops — same verify-or-retry
    behaviour, just a different pattern pool. Deterministic given ``rng``.
    """
    out: List[dict] = []
    attempts = 0
    cap = max_attempts_factor * max(need, 1)
    while len(out) < need:
        attempts += 1
        if attempts > cap:
            raise RuntimeError(
                f"could not inject {need} '{label}' records after {attempts} attempts"
            )
        pattern = rng.choice(list(patterns))
        problem = problems.make_problem(rng, pattern=pattern)
        injected = errors.inject(problem, label, rng)
        if injected is None:
            continue
        student_seq, student_text = injected
        out.append(_partial_record(problem, student_seq, student_text, label))
    return out


def finalize_record(partial: dict, rid: int, split: str, render_subdir: str) -> dict:
    """Wrap a partial (from :func:`_partial_record`) into a full, schema-ordered, asserted
    record with an ``id``, ``split`` and ``render_path``.

    This is the exact per-record finishing step ``build_records`` performs inline, exposed
    so the v4 assembler (contrastive + curriculum + normal mix) produces records that pass
    the identical invariants (``_assert_record``) as the balanced generator.
    """
    rec = {"id": rid, **partial, "split": split,
           "render_path": f"{render_subdir}/{rid:06d}.png"}
    rec = {k: rec[k] for k in _SCHEMA_KEYS}  # enforce schema key order
    _assert_record(rec)
    return rec


def build_records(
    seed: int,
    n: int,
    min_count: int,
    split_fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    render_subdir: str = "renders",
    max_attempts_factor: int = 400,
    ood_per_label: int = 120,
) -> Tuple[List[dict], Dict[str, int]]:
    """Build the full asserted record list (no file I/O, no rendering).

    Two disjoint groups, both deterministic functions of ``seed``:

    * In-distribution: balanced across all 8 labels (``compute_targets``), built ONLY
      from ``errors.IN_DISTRIBUTION_PATTERNS`` and assigned train/val/test via
      ``split_of``.
    * OOD (held-out compositions): ``ood_per_label`` records for each label realizable
      from ``errors.HELD_OUT_PATTERNS`` (the 4 OOD-eligible labels), all tagged
      ``split="ood"``. Intentionally unbalanced.

    IDs are contiguous across the whole dataset (in-distribution first, then OOD), so
    ``render_path`` stays ``"<subdir>/<id:06d>.png"`` for every record.
    """
    rng = random.Random(seed)
    labels = tc.DIAGNOSIS_LABELS
    targets = compute_targets(n, min_count, labels)

    # 1) In-distribution partials (balanced; in-distribution patterns only).
    id_partials: List[dict] = []
    for label in labels:  # grouped by label; interleaved afterwards
        id_partials += _inject_partials(
            label, targets[label], errors.ID_COMPATIBLE_PATTERNS[label],
            rng, max_attempts_factor,
        )

    # 2) OOD partials (held-out compositions only; naturally covers 4 labels).
    ood_partials: List[dict] = []
    if ood_per_label > 0:
        for label in errors.OOD_ELIGIBLE_LABELS:
            ood_partials += _inject_partials(
                label, ood_per_label, errors.OOD_COMPATIBLE_PATTERNS[label],
                rng, max_attempts_factor,
            )

    # 3) Deterministic interleave within each group (separately salted).
    random.Random(seed ^ _SHUFFLE_SALT).shuffle(id_partials)
    random.Random(seed ^ _OOD_SHUFFLE_SALT).shuffle(ood_partials)

    # 4) Assign contiguous ids; in-distribution gets a computed split, OOD is forced.
    records: List[dict] = []
    rid = 0
    for partial in id_partials:
        rec = {"id": rid, **partial,
               "split": split_of(rid, seed, split_fracs),
               "render_path": f"{render_subdir}/{rid:06d}.png"}
        rec = {k: rec[k] for k in _SCHEMA_KEYS}  # enforce schema key order
        _assert_record(rec)
        records.append(rec)
        rid += 1
    for partial in ood_partials:
        rec = {"id": rid, **partial,
               "split": "ood",
               "render_path": f"{render_subdir}/{rid:06d}.png"}
        rec = {k: rec[k] for k in _SCHEMA_KEYS}  # enforce schema key order
        _assert_record(rec)
        records.append(rec)
        rid += 1
    return records, targets


def label_counts(records: Sequence[dict]) -> Dict[str, int]:
    counts = {l: 0 for l in tc.DIAGNOSIS_LABELS}
    for r in records:
        counts[r["label"]] += 1
    return counts


def split_counts(records: Sequence[dict]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in records:
        counts[r["split"]] = counts.get(r["split"], 0) + 1
    return dict(sorted(counts.items()))


def _atomic_write_lines(path: str, lines: Sequence[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(lines))
    os.replace(tmp, path)


def write_jsonl(records: Sequence[dict], out_dir: str) -> Dict[str, str]:
    """Write ``data.jsonl`` (all records, in-distribution + OOD) plus per-split files
    ``train`` / ``val`` / ``test`` / ``ood``. Returns paths written. Writes are atomic
    (tmp + os.replace)."""
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}

    all_path = os.path.join(out_dir, "data.jsonl")
    _atomic_write_lines(all_path, [json.dumps(r) + "\n" for r in records])
    written["all"] = all_path

    for split in ("train", "val", "test", "ood"):
        rows = [json.dumps(r) + "\n" for r in records if r["split"] == split]
        path = os.path.join(out_dir, f"{split}.jsonl")
        _atomic_write_lines(path, rows)
        written[split] = path
    return written


def generate(
    out_dir: str,
    seed: int,
    n: int,
    min_count: int,
    split_fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    render_subdir: str = "renders",
    do_render: bool = True,
    skip_existing_renders: bool = True,
    ood_per_label: int = 120,
) -> dict:
    """End-to-end (in-process): build + assert records, write JSONL, optionally render.

    Idempotent/resumable: JSONL is rewritten atomically each call (cheap, deterministic);
    renders skip files that already exist. Renders every record, in-distribution AND OOD.
    """
    records, targets = build_records(
        seed, n, min_count, split_fracs, render_subdir, ood_per_label=ood_per_label
    )
    written = write_jsonl(records, out_dir)
    # Qwen3-VL chat conversations (image + instruction -> target JSON) per split.
    written.update(chat_format.write_chat_splits(records, out_dir))
    rendered = 0
    if do_render:
        from . import render as render_mod  # lazy: only touch matplotlib when rendering
        rendered = render_mod.render_all(records, out_dir, skip_existing=skip_existing_renders)
    id_records = [r for r in records if r["split"] != "ood"]
    ood_records = [r for r in records if r["split"] == "ood"]
    return {
        "records": records,
        "targets": targets,
        "written": written,
        "rendered": rendered,
        "label_counts": label_counts(records),
        "id_label_counts": label_counts(id_records),
        "ood_label_counts": label_counts(ood_records),
        "split_counts": split_counts(records),
        "id_count": len(id_records),
        "ood_count": len(ood_records),
    }
