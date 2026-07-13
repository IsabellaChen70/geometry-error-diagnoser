#!/usr/bin/env python3
"""rewrite_hints_fast.py — regenerate v6 hint-fix data WITHOUT re-rendering.

Login-node, CPU-only. The leaky-hint fix only changed the ``hint`` STRING inside
each ``full`` assistant target; the geometry (maps/label) and every render are
byte-identical. So instead of re-running ``make_v6_transform_data.py`` (which
re-renders new pool images and takes ~1-3h), this script reuses an existing v6
tree and rewrites ONLY the ``hint`` field of the 4 ``full`` chat files
(train/val x image/image_coords), re-deriving each hint with the fixed
``transform_diagnosis.hints.hint_for()``.

What it produces (default ``~/transform_diagnosis_data_v6_hintfix``):
  * renders_v6/ (or images/) and the ``source_data`` symlink: replicated as
    symlinks — NO copy, NO render.
  * the 12 correct/student/both chat files + manifest: replicated unchanged
    (symlinks by default; ``--copy-unchanged`` to copy).
  * the 4 ``full`` chat files: rewritten so ONLY the assistant ``hint`` differs,
    re-serialized with the SAME ``json.dumps(..., ensure_ascii=False)`` (outer)
    and ``separators=(",", ":")`` (inner target) that make_v6 uses, so the output
    is byte-for-byte identical to the input except within each hint value.
  * ``--rewrite-jsonl-hints`` additionally cleans the (unused-at-train-time)
    ``hint`` field inside ``{train,val}_v6.jsonl`` the same way.

``--verify`` asserts, across ALL rewritten full rows: the hint is family-relevant
(``eval._hint_mentions_family``) AND NOT ``hints.is_strict_leak`` AND NOT
``eval._hint_has_leak``. It prints counts and exits nonzero on any leak.

Runtime on the full 10k-record set (train 9600 + val 400 -> 20000 full-chat rows
plus 10000 jsonl rows): pure JSON parse/dump + regex, no I/O of renders, so a few
seconds to well under a minute on a login node.

FALLBACK: if ``~/transform_diagnosis_data_v6`` was deleted on the cluster there is
nothing to reuse — regenerate with make_v6_transform_data.py instead (that path is
the ~1-3h one). See model/HINT_FIX_FAST.md.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _candidate in (REPO_ROOT, HERE, HOME):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)

try:  # local checkout
    from transform_diagnosis import eval as ev, hints
except ModuleNotFoundError:  # cluster package fallback (synced to ~/slm_eval)
    from slm_eval import eval as ev, hints  # type: ignore

DEFAULT_V6_DIR = os.path.join(HOME, "transform_diagnosis_data_v6")
DEFAULT_OUT_DIR = os.path.join(HOME, "transform_diagnosis_data_v6_hintfix")
SPLITS: Tuple[str, ...] = ("train", "val")
MODALITIES: Tuple[str, ...] = ("image", "image_coords")

RecordKey = Tuple[Optional[str], object]


# --------------------------------------------------------------------------------------
# Serialization helpers — MUST match make_v6_transform_data / v6_format byte-for-byte.
# --------------------------------------------------------------------------------------

def canonical_outer(row: dict) -> str:
    """Row serialization used by make_v6._atomic_write_jsonl (default separators)."""
    return json.dumps(row, ensure_ascii=False)


def canonical_inner(obj: dict) -> str:
    """Assistant-target serialization used by v6_format.target_json (compact)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def full_chat_name(split: str, modality: str) -> str:
    return f"{split}_v6_{modality}_full_chat.jsonl"


def jsonl_name(split: str) -> str:
    return f"{split}_v6.jsonl"


def _assistant_text_part(row: dict) -> dict:
    """Return the mutable ``{"type":"text","text":...}`` part of the assistant turn."""
    for message in row.get("messages", []):
        if message.get("role") == "assistant":
            for part in message.get("content", []):
                if part.get("type") == "text":
                    return part
    raise ValueError("chat row has no assistant text part")


# --------------------------------------------------------------------------------------
# Record loading + hint derivation
# --------------------------------------------------------------------------------------

def load_records(v6_dir: str) -> Dict[RecordKey, dict]:
    """Map ``(split, id) -> record`` from ``{train,val}_v6.jsonl``.

    ids restart at 0 per split, so the split is part of the key.
    """
    records: Dict[RecordKey, dict] = {}
    for split in SPLITS:
        path = os.path.join(v6_dir, jsonl_name(split))
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rec = json.loads(line)
                records[(rec.get("split", split), rec["id"])] = rec
    return records


def new_hint(rec: dict) -> str:
    """The fixed, coordinate-free Socratic hint for this record's label."""
    return hints.hint_for(rec["label"], rec)


def diff_is_hint_only(orig_line: str, new_line: str) -> bool:
    """True iff two full-chat lines are identical except the inner ``hint`` value."""
    orig_row = json.loads(orig_line)
    new_row = json.loads(new_line)
    orig_part = _assistant_text_part(orig_row)
    new_part = _assistant_text_part(new_row)
    orig_inner = json.loads(orig_part["text"])
    new_inner = json.loads(new_part["text"])
    if list(orig_inner.keys()) != list(new_inner.keys()):
        return False
    for key in orig_inner:
        if key == "hint":
            continue
        if orig_inner[key] != new_inner[key]:
            return False
    # Blank the assistant text on both and compare the entire remaining structure.
    orig_part["text"] = ""
    new_part["text"] = ""
    return orig_row == new_row


# --------------------------------------------------------------------------------------
# Rewriters
# --------------------------------------------------------------------------------------

def rewrite_full_chat(
    src: str, dst: str, records: Dict[RecordKey, dict], *, paranoid: bool
) -> dict:
    """Rewrite ONLY the assistant hint in a full chat file; return stats.

    ``noncanonical`` counts source rows whose exact bytes are not reproduced by
    ``canonical_outer`` (0 means the output is byte-identical to the input except
    inside the hint value). ``struct_fail`` counts rows where more than the hint
    changed (always 0 by construction; checked only when ``paranoid``).
    """
    total = changed = noncanonical = struct_fail = 0
    tmp = dst + ".tmp"
    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            total += 1
            line = raw[:-1] if raw.endswith("\n") else raw
            row = json.loads(line)
            if canonical_outer(row) != line:
                noncanonical += 1
            key = (row.get("split"), row.get("id"))
            rec = records.get(key)
            if rec is None:
                raise SystemExit(
                    f"{os.path.basename(src)}: no {jsonl_name(str(key[0]))} record for id={key[1]}"
                )
            part = _assistant_text_part(row)
            inner = json.loads(part["text"])
            if "hint" not in inner:
                raise SystemExit(f"{os.path.basename(src)}: row id={key} is not a full target")
            label = rec["label"]
            if inner.get("label") != label:
                raise SystemExit(
                    f"{os.path.basename(src)}: label mismatch id={key}: "
                    f"chat={inner.get('label')!r} record={label!r}"
                )
            hint = new_hint(rec)
            if inner.get("hint") != hint:
                changed += 1
            inner["hint"] = hint
            part["text"] = canonical_inner(inner)
            new_line = canonical_outer(row)
            if paranoid and not diff_is_hint_only(line, new_line):
                struct_fail += 1
            fout.write(new_line + "\n")
    os.replace(tmp, dst)
    return {
        "total": total,
        "changed": changed,
        "noncanonical": noncanonical,
        "struct_fail": struct_fail,
    }


def rewrite_jsonl_hints(src: str, dst: str, *, paranoid: bool) -> dict:
    """Rewrite ONLY the top-level ``hint`` field of a v6 records jsonl file."""
    total = changed = noncanonical = struct_fail = 0
    tmp = dst + ".tmp"
    with open(src, encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for raw in fin:
            if not raw.strip():
                continue
            total += 1
            line = raw[:-1] if raw.endswith("\n") else raw
            rec = json.loads(line)
            if canonical_outer(rec) != line:
                noncanonical += 1
            if "hint" not in rec:
                raise SystemExit(f"{os.path.basename(src)}: record id={rec.get('id')} has no hint")
            hint = new_hint(rec)
            if rec.get("hint") != hint:
                changed += 1
            before = copy.deepcopy(rec) if paranoid else None
            rec["hint"] = hint
            new_line = canonical_outer(rec)
            if paranoid and before is not None:
                after = json.loads(new_line)
                before["hint"] = None
                after["hint"] = None
                if before != after:
                    struct_fail += 1
            fout.write(new_line + "\n")
    os.replace(tmp, dst)
    return {
        "total": total,
        "changed": changed,
        "noncanonical": noncanonical,
        "struct_fail": struct_fail,
    }


# --------------------------------------------------------------------------------------
# Output directory replication (symlink renders + unchanged files)
# --------------------------------------------------------------------------------------

def _place_symlink(target: str, linkpath: str, *, is_dir: bool) -> None:
    if os.path.lexists(linkpath):
        if os.path.islink(linkpath) or os.path.isfile(linkpath):
            os.remove(linkpath)
        else:
            shutil.rmtree(linkpath)
    os.symlink(target, linkpath, target_is_directory=is_dir)


def replicate_entry(name: str, v6_dir: str, out_dir: str, *, copy_unchanged: bool) -> str:
    """Replicate one unchanged top-level entry into ``out_dir``.

    Directories and symlinks are always linked (never bulk-copied, so renders are
    never duplicated). Regular files are symlinked unless ``copy_unchanged``.
    """
    src = os.path.join(v6_dir, name)
    dst = os.path.join(out_dir, name)
    if os.path.islink(src):
        target = os.readlink(src)
        if not os.path.isabs(target):
            target = os.path.abspath(os.path.join(v6_dir, target))
        _place_symlink(target, dst, is_dir=os.path.isdir(src))
        return "symlink->" + target
    if os.path.isdir(src):
        _place_symlink(os.path.abspath(src), dst, is_dir=True)
        return "symlink-dir"
    if copy_unchanged:
        shutil.copy2(src, dst)
        return "copy"
    _place_symlink(os.path.abspath(src), dst, is_dir=False)
    return "symlink-file"


def prepare_out_dir(out_dir: str, *, overwrite: bool) -> None:
    if os.path.exists(out_dir) and os.listdir(out_dir):
        if not overwrite:
            raise SystemExit(
                f"refusing nonempty out-dir {out_dir}; pass --overwrite or choose a new --out-dir"
            )
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


# --------------------------------------------------------------------------------------
# Verification
# --------------------------------------------------------------------------------------

def verify(target_dir: str) -> bool:
    """Assert every rewritten full-chat hint is family-relevant and non-leaking.

    Returns True iff there are 0 leaks and 0 non-family hints across all rows.
    """
    records = load_records(target_dir)
    if not records:
        raise SystemExit(f"no {{train,val}}_v6.jsonl records found under {target_dir}")

    grand = {"n": 0, "family": 0, "strict_leak": 0, "eval_leak": 0, "safe": 0}
    per_file: List[Tuple[str, dict]] = []
    examples: List[str] = []
    for split in SPLITS:
        for modality in MODALITIES:
            path = os.path.join(target_dir, full_chat_name(split, modality))
            if not os.path.isfile(path):
                continue
            stat = {"n": 0, "family": 0, "strict_leak": 0, "eval_leak": 0, "safe": 0}
            with open(path, encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    key = (row.get("split"), row.get("id"))
                    rec = records.get(key)
                    if rec is None:
                        raise SystemExit(f"{os.path.basename(path)}: no record for id={key}")
                    inner = json.loads(_assistant_text_part(row)["text"])
                    hint = inner.get("hint")
                    hint = hint if isinstance(hint, str) else ""
                    label = rec["label"]
                    families = hints.expected_hint_families(label, rec)
                    tokens = hints.expected_hint_tokens(label, rec)
                    fam_ok = bool(hint.strip()) and ev._hint_mentions_family(hint, families)
                    strict = hints.is_strict_leak(hint, label, rec)
                    eval_leak = ev._hint_has_leak(hint, tokens)
                    safe = fam_ok and not strict and not eval_leak
                    for name, flag in (
                        ("n", True), ("family", fam_ok), ("strict_leak", strict),
                        ("eval_leak", eval_leak), ("safe", safe),
                    ):
                        stat[name] += int(flag)
                    if not safe and len(examples) < 5:
                        examples.append(
                            f"  {os.path.basename(path)} id={key[1]} label={label} "
                            f"fam={fam_ok} strict={strict} eval_leak={eval_leak} hint={hint!r}"
                        )
            per_file.append((os.path.basename(path), stat))
            for name in grand:
                grand[name] += stat[name]

    def rate(num: int, den: int) -> str:
        return f"{(num / den):.3f}" if den else "--"

    print("=== rewrite_hints_fast --verify ===")
    for name, stat in per_file:
        n = stat["n"]
        print(
            f"  {name:44s} n={n:5d} family={rate(stat['family'], n)} "
            f"strict_leak={rate(stat['strict_leak'], n)} "
            f"eval_leak={rate(stat['eval_leak'], n)} safe_useful={rate(stat['safe'], n)}"
        )
    n = grand["n"]
    print(
        f"  {'TOTAL':44s} n={n:5d} family={rate(grand['family'], n)} "
        f"strict_leak={rate(grand['strict_leak'], n)} "
        f"eval_leak={rate(grand['eval_leak'], n)} safe_useful={rate(grand['safe'], n)}"
    )
    leaks = grand["strict_leak"] + grand["eval_leak"]
    non_family = n - grand["family"]
    if examples:
        print("  first unsafe rows:")
        print("\n".join(examples))
    ok = leaks == 0 and non_family == 0 and n > 0
    print(
        f"  RESULT: {'PASS' if ok else 'FAIL'} "
        f"(leaks={leaks} non_family={non_family} rows={n})"
    )
    return ok


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--v6-dir", default=DEFAULT_V6_DIR,
                        help="existing v6 dataset dir to reuse (renders reused via symlink)")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help="hint-fixed output dir to create")
    parser.add_argument("--rewrite-jsonl-hints", action="store_true",
                        help="also rewrite the unused hint field inside {train,val}_v6.jsonl")
    parser.add_argument("--copy-unchanged", action="store_true",
                        help="copy unchanged small files instead of symlinking them")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace a nonempty --out-dir")
    parser.add_argument("--paranoid", action="store_true",
                        help="independently re-diff every rewritten row (hint-only proof)")
    parser.add_argument("--verify", action="store_true",
                        help="after building, assert 0 leaks over all rewritten full rows")
    parser.add_argument("--verify-only", action="store_true",
                        help="skip building; only verify an existing --out-dir")
    return parser


def build(args: argparse.Namespace) -> None:
    if not os.path.isdir(args.v6_dir):
        raise SystemExit(
            f"--v6-dir not found: {args.v6_dir}\n"
            "If ~/transform_diagnosis_data_v6 was deleted, there is nothing to reuse; "
            "regenerate with make_v6_transform_data.py (see model/HINT_FIX_FAST.md fallback)."
        )
    records = load_records(args.v6_dir)
    if not records:
        raise SystemExit(f"no {{train,val}}_v6.jsonl found under {args.v6_dir}")

    full_targets = [
        full_chat_name(split, modality)
        for split in SPLITS
        for modality in MODALITIES
        if os.path.isfile(os.path.join(args.v6_dir, full_chat_name(split, modality)))
    ]
    if not full_targets:
        raise SystemExit(f"no *_full_chat.jsonl files under {args.v6_dir}")
    jsonl_targets = (
        [jsonl_name(split) for split in SPLITS
         if os.path.isfile(os.path.join(args.v6_dir, jsonl_name(split)))]
        if args.rewrite_jsonl_hints else []
    )
    rewrite_set = set(full_targets) | set(jsonl_targets)

    prepare_out_dir(args.out_dir, overwrite=args.overwrite)

    replicated = 0
    for name in sorted(os.listdir(args.v6_dir)):
        if name in rewrite_set:
            continue
        replicate_entry(name, args.v6_dir, args.out_dir, copy_unchanged=args.copy_unchanged)
        replicated += 1

    print(f"=== rewrite_hints_fast: {args.v6_dir} -> {args.out_dir} ===")
    print(f"  replicated {replicated} unchanged entries (renders symlinked, no re-render)")
    for name in full_targets + jsonl_targets:
        src = os.path.join(args.v6_dir, name)
        dst = os.path.join(args.out_dir, name)
        if name.endswith("_full_chat.jsonl"):
            stat = rewrite_full_chat(src, dst, records, paranoid=args.paranoid)
        else:
            stat = rewrite_jsonl_hints(src, dst, paranoid=args.paranoid)
        flag = ""
        if stat["noncanonical"]:
            flag += f"  [WARN {stat['noncanonical']} rows not byte-canonical: hint-only guaranteed structurally, not byte-for-byte]"
        if stat["struct_fail"]:
            flag += f"  [ERROR {stat['struct_fail']} rows changed beyond the hint]"
        print(
            f"  rewrote {name:44s} rows={stat['total']:5d} hints_changed={stat['changed']:5d}{flag}"
        )
        if stat["struct_fail"]:
            raise SystemExit(f"{name}: {stat['struct_fail']} rows changed beyond the hint")


def main(argv: Optional[List[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    start = time.time()
    if not args.verify_only:
        build(args)
    ok = True
    if args.verify or args.verify_only:
        ok = verify(args.out_dir)
    print(f"  elapsed: {time.time() - start:.2f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
