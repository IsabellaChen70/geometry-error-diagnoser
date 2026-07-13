"""Build a larger, balanced, committable v6 public sample from local source data.

This does NOT touch the frozen ``results/`` tree and never modifies the read-only
source dataset. It selects a deterministic, label-balanced draw from the local
source records, re-derives and verifies the canonical net maps through the
production ``transform_diagnosis.v6_format`` helpers, and writes:

* ``train_v6.jsonl``          -- N balanced v6 records (full schema, provenance)
* ``train_v6_coords_*_chat.jsonl`` -- text-only chat for all N records (4 tasks)
* ``images/``                 -- a small visual subset of byte-for-byte PNGs
* ``train_v6_image_coords_full_chat.jsonl`` -- multimodal example for the subset
* ``manifest_public.json``    -- counts, distributions, checksums, verification

The coordinates-only chat files are fully self-contained (no image dependency),
so the text sample is valid regardless of how many PNGs are shipped. Images are
intentionally limited to a handful to keep the repository small.

Run ``python3 model/build_public_sample.py --dry-run`` to validate without writing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Mapping, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
HOME = os.path.expanduser("~")
for candidate in (REPO_ROOT, HERE):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from transform_diagnosis import transform_core as tc  # noqa: E402
from transform_diagnosis import v6_format  # noqa: E402

TASKS = v6_format.TASK_MODES
DEFAULT_OUT = os.path.join(REPO_ROOT, "dataset_public")


def _default_source_dir() -> str:
    local = os.path.join(REPO_ROOT, "transform_diagnosis_data")
    if os.path.isdir(local):
        return local
    return os.path.join(HOME, "transform_diagnosis_data")


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: str, rows: Sequence[Mapping]) -> dict:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)
    return {"bytes": os.path.getsize(path), "records": len(rows), "sha256": _sha256(path)}


def _write_json(path: str, value: Mapping) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def _augment_source_record(rec: Mapping) -> dict:
    """Attach provenance, then verify + attach canonical nets via production path."""
    copied = dict(rec)
    copied["source_id"] = rec.get("id")
    copied["source_split"] = rec.get("split")
    copied["v6_pool"] = "source"
    return v6_format.augment_record(copied)


def select_balanced(
    records: Sequence[Mapping],
    per_label: int,
) -> tuple[List[dict], Dict[str, int], List[str]]:
    """Deterministic label-balanced selection with production verification."""
    by_label: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        by_label[str(rec.get("label"))].append(rec)

    selected: List[dict] = []
    skipped: List[str] = []
    got: Counter = Counter()
    for label in tc.DIAGNOSIS_LABELS:
        pool = sorted(by_label.get(label, []), key=lambda r: int(r.get("id", 0)))
        for rec in pool:
            if got[label] >= per_label:
                break
            try:
                aug = _augment_source_record(rec)
            except Exception as exc:  # keep going; record why a record was rejected
                skipped.append(f"{label}#{rec.get('id')}: {exc}")
                continue
            selected.append(aug)
            got[label] += 1
    return selected, dict(got), skipped


def build(args: argparse.Namespace) -> int:
    source_dir = os.path.abspath(args.source_dir)
    train_path = args.source_train or os.path.join(source_dir, "train.jsonl")
    if not os.path.isfile(train_path):
        raise SystemExit(f"source train JSONL not found: {train_path}")

    source_rows = load_jsonl(train_path)
    selected, per_label_counts, skipped = select_balanced(source_rows, args.per_label)

    short = [
        label
        for label in tc.DIAGNOSIS_LABELS
        if per_label_counts.get(label, 0) < args.per_label
    ]
    if short:
        raise SystemExit(
            f"could not fill {args.per_label}/label for {short}; "
            f"got {per_label_counts}; first skips: {skipped[:5]}"
        )

    # Renumber deterministically; keep source identity for provenance.
    for new_id, rec in enumerate(selected):
        rec["id"] = new_id

    # Image subset: first ``image_per_label`` of each label, in selection order.
    image_seen: Counter = Counter()
    image_subset: List[dict] = []
    for rec in selected:
        label = str(rec["label"])
        if image_seen[label] < args.image_per_label:
            image_subset.append(rec)
            image_seen[label] += 1

    # Verify every task target for every record before writing anything.
    verification = {"targets_parsed_and_exact": 0, "conversations_built": 0}
    coords_chat: Dict[str, List[dict]] = {task: [] for task in TASKS}
    for rec in selected:
        for task in TASKS:
            target = v6_format.target_json(rec, task)
            if v6_format.target_obj(rec, task) != json.loads(target):
                raise RuntimeError(f"id={rec['id']} task={task} target round-trip failed")
            verification["targets_parsed_and_exact"] += 1
            coords_chat[task].append(v6_format.conversation(rec, task, "coords"))
            verification["conversations_built"] += 1

    image_full_chat: List[dict] = []
    for rec in image_subset:
        basename = os.path.basename(str(rec["render_path"]))
        image_full_chat.append(
            v6_format.conversation(
                rec, "full", "image_coords", image_path=f"images/{basename}"
            )
        )

    label_dist = dict(sorted(Counter(str(r["label"]) for r in selected).items()))

    if args.dry_run:
        print("DRY RUN: nothing written")
        print(f"source: {train_path} ({len(source_rows)} rows)")
        print(f"selected: {len(selected)} records ({args.per_label}/label)")
        print(f"label distribution: {label_dist}")
        print(f"image subset: {len(image_subset)} records / PNGs")
        print(f"coords chat rows: {sum(len(v) for v in coords_chat.values())}")
        print(f"verification: {verification}")
        if skipped:
            print(f"skipped {len(skipped)} source records (first 3): {skipped[:3]}")
        return 0

    out_dir = os.path.abspath(args.out_dir)
    images_dir = os.path.join(out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    written: Dict[str, dict] = {}
    written["train_v6.jsonl"] = _write_jsonl(
        os.path.join(out_dir, "train_v6.jsonl"), selected
    )
    for task in TASKS:
        name = f"train_v6_coords_{task}_chat.jsonl"
        written[name] = _write_jsonl(os.path.join(out_dir, name), coords_chat[task])
    written["train_v6_image_coords_full_chat.jsonl"] = _write_jsonl(
        os.path.join(out_dir, "train_v6_image_coords_full_chat.jsonl"), image_full_chat
    )

    image_files: Dict[str, dict] = {}
    for rec in image_subset:
        basename = os.path.basename(str(rec["render_path"]))
        src_img = os.path.join(source_dir, str(rec["render_path"]))
        if not os.path.isfile(src_img):
            raise SystemExit(f"source render missing: {src_img}")
        dst_img = os.path.join(images_dir, basename)
        shutil.copyfile(src_img, dst_img)
        image_files[f"images/{basename}"] = {
            "bytes": os.path.getsize(dst_img),
            "sha256": _sha256(dst_img),
            "copied_byte_for_byte": True,
            "source": os.path.relpath(src_img, REPO_ROOT),
        }

    manifest = {
        "artifact": "dataset_public",
        "schema_version": v6_format.SCHEMA_VERSION,
        "description": (
            "Larger label-balanced v6 sample built from local source records via the "
            "production v6_format helpers; coordinates chat is text-only and covers all "
            "records, images are a small visual subset. Not a byte-for-byte sample of "
            "the ORCD mixed curriculum."
        ),
        "source": {
            "train_jsonl": os.path.relpath(train_path, REPO_ROOT),
            "train_records": len(source_rows),
            "sha256": _sha256(train_path),
        },
        "sample": {
            "records": len(selected),
            "per_label": args.per_label,
            "label_distribution": label_dist,
            "tasks": list(TASKS),
            "coords_chat_rows_per_task": len(selected),
            "image_subset_records": len(image_subset),
            "image_per_label": args.image_per_label,
        },
        "generation": {
            "record_augmentation": "transform_diagnosis.v6_format.augment_record",
            "conversation_builder": "transform_diagnosis.v6_format.conversation",
            "assistant_target_builder": "transform_diagnosis.v6_format.target_json",
            "selection": "deterministic: sorted by source id within each label",
            "builder": "model/build_public_sample.py",
        },
        "verification": {
            **verification,
            "skipped_source_records": len(skipped),
            "unique_ids": len({r["id"] for r in selected}),
        },
        "known_hint_limitation": {
            "preserved": True,
            "summary": (
                "Full-task hints preserve current training targets and may disclose "
                "exact operations, map parameters, or translation values."
            ),
            "audit": "../results/overnight/HINT_SAFETY_AUDIT.md",
        },
        "files": {**written, **image_files},
    }
    _write_json(os.path.join(out_dir, "manifest_public.json"), manifest)

    print(f"wrote v6 public sample to {out_dir}")
    print(f"records: {len(selected)} ({args.per_label}/label); images: {len(image_subset)}")
    if skipped:
        print(f"skipped {len(skipped)} source records during verification")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=_default_source_dir())
    parser.add_argument("--source-train", default=None,
                        help="default: <source-dir>/train.jsonl")
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--per-label", type=int, default=30,
                        help="records per diagnosis label (8 labels)")
    parser.add_argument("--image-per-label", type=int, default=3,
                        help="PNGs to copy per label for the visual subset")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and verify in memory; write nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return build(build_arg_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
