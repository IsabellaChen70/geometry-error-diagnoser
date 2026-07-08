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

from . import errors, geometry, problems, transform_core as tc

BOUND = geometry.DEFAULT_BOUND

# Field order matches the record schema exactly (dicts preserve insertion order).
_SCHEMA_KEYS = [
    "id", "num_vertices", "original", "correct_transform", "correct_image",
    "student_transform", "student_image", "label", "is_correct", "split", "render_path",
]

# A second RNG stream, derived deterministically from the seed, interleaves the
# label-grouped records so the file (and every split) mixes all labels.
_SHUFFLE_SALT = 0x5F3759DF
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
    return {
        "num_vertices": problem.num_vertices,
        "original": [[x, y] for x, y in problem.original],
        "correct_transform": list(problem.answer_text),
        "correct_image": [[x, y] for x, y in correct_image],
        "student_transform": list(student_text),
        "student_image": [[x, y] for x, y in student_image],
        "label": label,
        "is_correct": (label == "correct"),
    }


def _assert_record(rec: dict) -> None:
    """Fail loudly if any invariant is violated (acceptance-grade checks, per record)."""
    assert rec["label"] in tc.DIAGNOSIS_LABELS, rec["label"]
    # Label must equal an independent diagnosis of the STORED text lists.
    d = tc.diagnose(rec["original"], rec["correct_transform"], rec["student_transform"])
    assert d == rec["label"], f"label {rec['label']} != diagnose {d}"
    assert rec["is_correct"] == (rec["label"] == "correct")
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
    # Uniqueness: the recovered net map equals the intended net map.
    rec_map = tc.recover_map(rec["original"], rec["correct_image"])
    assert rec_map == tc.compose(rec["correct_transform"]), "recover_map mismatch"


def build_records(
    seed: int,
    n: int,
    min_count: int,
    split_fracs: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    render_subdir: str = "renders",
    max_attempts_factor: int = 400,
) -> Tuple[List[dict], Dict[str, int]]:
    """Build the full, balanced, split, asserted record list (no file I/O, no rendering)."""
    rng = random.Random(seed)
    labels = tc.DIAGNOSIS_LABELS
    targets = compute_targets(n, min_count, labels)

    partials: List[dict] = []
    for label in labels:  # grouped by label; interleaved afterwards
        need = targets[label]
        made = 0
        attempts = 0
        cap = max_attempts_factor * max(need, 1)
        while made < need:
            attempts += 1
            if attempts > cap:
                raise RuntimeError(
                    f"could not inject {need} '{label}' records after {attempts} attempts"
                )
            pattern = rng.choice(errors.COMPATIBLE_PATTERNS[label])
            problem = problems.make_problem(rng, pattern=pattern)
            injected = errors.inject(problem, label, rng)
            if injected is None:
                continue
            student_seq, student_text = injected
            partials.append(_partial_record(problem, student_seq, student_text, label))
            made += 1

    # Deterministic interleave, then assign id / split / render_path from the final id.
    random.Random(seed ^ _SHUFFLE_SALT).shuffle(partials)

    records: List[dict] = []
    for rid, partial in enumerate(partials):
        rec = {"id": rid, **partial,
               "split": split_of(rid, seed, split_fracs),
               "render_path": f"{render_subdir}/{rid:06d}.png"}
        rec = {k: rec[k] for k in _SCHEMA_KEYS}  # enforce schema key order
        _assert_record(rec)
        records.append(rec)
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
    """Write ``data.jsonl`` (all records) plus per-split files. Returns paths written.
    Writes are atomic (tmp + os.replace)."""
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}

    all_path = os.path.join(out_dir, "data.jsonl")
    _atomic_write_lines(all_path, [json.dumps(r) + "\n" for r in records])
    written["all"] = all_path

    for split in ("train", "val", "test"):
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
) -> dict:
    """End-to-end (in-process): build + assert records, write JSONL, optionally render.

    Idempotent/resumable: JSONL is rewritten atomically each call (cheap, deterministic);
    renders skip files that already exist.
    """
    records, targets = build_records(seed, n, min_count, split_fracs, render_subdir)
    written = write_jsonl(records, out_dir)
    rendered = 0
    if do_render:
        from . import render as render_mod  # lazy: only touch matplotlib when rendering
        rendered = render_mod.render_all(records, out_dir, skip_existing=skip_existing_renders)
    return {
        "records": records,
        "targets": targets,
        "written": written,
        "rendered": rendered,
        "label_counts": label_counts(records),
        "split_counts": split_counts(records),
    }
