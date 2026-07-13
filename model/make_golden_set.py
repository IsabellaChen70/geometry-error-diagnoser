"""Build a held-out, oracle-verified, provably-disjoint "golden" evaluation set.

Everything in this project is deterministically generated and verified by the
geometry oracle (:mod:`transform_diagnosis.transform_core`), so there are no human
labels.  "Golden" here therefore means: a *fresh*, *balanced*, oracle-verified set
of v6 canonical-net diagnosis records that is **provably disjoint** from every
existing split (v6 train/val AND the frozen source train/val/test/ood), written in
the exact v6 format so it can be scored by the existing harness (``eval_transform.py``).

Method — identical primitives, nothing new
------------------------------------------
Records are produced by the SAME verified pipeline the other splits use:

    problems.make_problem  ->  errors.inject  ->  dataset._partial_record
      ->  dataset.finalize_record (runs dataset._assert_record, the oracle contract)
      ->  v6_format.augment_record (attaches + re-verifies the canonical net maps)

No new geometry, no new oracle, no new label path.  Every record is a two-step
in-distribution problem (``errors.ID_COMPATIBLE_PATTERNS``), exactly like the
balanced in-distribution generator (``dataset.build_records``), balanced across all
eight :data:`transform_core.DIAGNOSIS_LABELS`.

Canonical dedup key (the disjointness guarantee)
------------------------------------------------
The pipeline's own leakage audit (``model/audit_v6_predictions.geometry_fingerprint``)
and the v6 source-pool splitter (``model/make_v6_transform_data._prepare_source_pools``)
both key a record by the *observable geometry*::

    json.dumps([original, correct_image, student_image], separators=(",", ":"))

i.e. RED pre-image + GREEN correct image + BLUE student image (the student image
encodes the injected error).  This is the problem geometry + injected error, NOT the
numeric id, so two records with the same key are the *same example* even with
different ids.  We reject any golden candidate whose key collides with ANY existing
record and assert 0 overlap at the end.

The "existing" key universe is built authoritatively, not from small samples:

* Frozen source ``train/val/test/ood`` are loaded in full from ``--source-dir``.
* v6 ``train``/``val`` are *reconstructed deterministically* in memory by calling the
  real generator (``make_v6_transform_data._build_split`` with the production
  ``--v6-seed``/``--v6-*-n``/``--v6-mix``); this reproduces the exact cluster v6
  geometry keys (the fresh contrastive/curriculum/hard pools plus the source-pool
  copies) without any rendering or file I/O.
* Any local v6 sample record files found in the repo are unioned in as a belt-and-
  suspenders extra.

Local/CPU only.  No ssh/sbatch/GPU/git/network.  Rendering uses the shared
``transform_diagnosis.render`` (matplotlib); if matplotlib is unavailable the jsonl /
chat files are still produced and the exact cluster render command is printed.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from collections import Counter
from typing import Dict, List, Mapping, Sequence, Set, Tuple

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for _cand in (REPO_ROOT, HERE, HOME):
    if _cand not in sys.path:
        sys.path.insert(0, _cand)

from transform_diagnosis import dataset, errors, problems, render
from transform_diagnosis import transform_core as tc
from transform_diagnosis import v6_format

# make_v6_transform_data supplies the shared IO helpers AND the deterministic v6
# reconstruction primitives, so the golden set is checked against the true v6 keys.
# Repo layout imports it as a package; the cluster syncs it flat into $HOME.
try:
    import model.make_v6_transform_data as mv6  # noqa: E402
except ModuleNotFoundError:  # cluster: script lives at $HOME
    import make_v6_transform_data as mv6  # type: ignore  # noqa: E402

DEFAULT_SEED = 20260712  # distinct from 0 (source), 20260709 (eval sample), 20260711 (v6)
DEFAULT_N_PER_LABEL = 20
GOLDEN_SPLIT = "golden"
RENDER_SUBDIR = os.path.join("renders_v6", GOLDEN_SPLIT)

# v6 production generation config (matches V6_TRANSFORM_RUNBOOK / make_v6 defaults),
# used only to *reconstruct* the existing v6 geometry keys for the disjointness check.
V6_SEED = mv6.DEFAULT_SEED  # 20260711
V6_TRAIN_N = 9600
V6_VAL_N = 400
V6_MIX = (0.50, 0.20, 0.15, 0.15)

SOURCE_SPLITS = ("train", "val", "test", "ood")


# --------------------------------------------------------------------------------------
# Canonical dedup key — identical to make_v6._prepare_source_pools.geometry_key and
# audit_v6_predictions.geometry_fingerprint.
# --------------------------------------------------------------------------------------

def geometry_key(rec: Mapping) -> str:
    """Observable-geometry fingerprint: RED pre-image + GREEN + BLUE (the injected error)."""
    return json.dumps(
        [rec.get("original"), rec.get("correct_image"), rec.get("student_image")],
        separators=(",", ":"),
        sort_keys=False,
    )


# --------------------------------------------------------------------------------------
# Default source-dir resolver (works both on the laptop and on the cluster).
# --------------------------------------------------------------------------------------

def default_source_dir() -> str:
    cluster = os.path.join(HOME, "transform_diagnosis_data")
    if os.path.isdir(cluster):
        return cluster
    return os.path.join(REPO_ROOT, "transform_diagnosis_data")


# --------------------------------------------------------------------------------------
# Build the "existing key" universe.
# --------------------------------------------------------------------------------------

def load_source_split_keys(source_dir: str) -> Dict[str, Set[str]]:
    """Full geometry-key sets for the frozen source train/val/test/ood splits."""
    out: Dict[str, Set[str]] = {}
    for split in SOURCE_SPLITS:
        path = os.path.join(source_dir, f"{split}.jsonl")
        if not os.path.isfile(path):
            raise SystemExit(f"source split not found: {path}")
        out[split] = {geometry_key(rec) for rec in mv6.load_jsonl(path)}
    return out


def reconstruct_v6_keys(
    source_dir: str, seed: int, train_n: int, val_n: int, mix: Sequence[float]
) -> Dict[str, Set[str]]:
    """Deterministically rebuild the exact cluster v6 train/val geometry keys in memory.

    Calls the real generator functions (no rendering, no file writes), so the returned
    keys are byte-for-byte the keys the production v6 data has for this config.
    """
    train_path = os.path.join(source_dir, "train.jsonl")
    val_path = os.path.join(source_dir, "val.jsonl")
    train_source, val_source = mv6._prepare_source_pools(train_path, val_path, seed)
    train_records, _, _ = mv6._build_split("train", train_n, mix, seed, train_source)
    val_records, _, _ = mv6._build_split("val", val_n, mix, seed, val_source)
    return {
        "v6_train": {geometry_key(rec) for rec in train_records},
        "v6_val": {geometry_key(rec) for rec in val_records},
    }


def _candidate_sample_files(out_dir: str) -> List[str]:
    """Curated + globbed local v6 *record* files (never the golden out-dir, never chat)."""
    out_abs = os.path.abspath(out_dir)
    names = ("train_v6.jsonl", "val_v6.jsonl", "test_v6.jsonl", "ood_v6.jsonl", "data_v6.jsonl")
    curated = [
        os.path.join(REPO_ROOT, "dataset_sample_v6", "train_v6.jsonl"),
        os.path.join(REPO_ROOT, "dataset_public", "train_v6.jsonl"),
    ]
    globbed: List[str] = []
    for pattern in ("_quarantine/.scratch_v6_verify", ".scratch_v6_verify", ".tmp_hintfix_*"):
        for base in glob.glob(os.path.join(REPO_ROOT, pattern)):
            for name in names:
                globbed.append(os.path.join(base, name))
    seen: Set[str] = set()
    files: List[str] = []
    for path in [*curated, *globbed]:
        ap = os.path.abspath(path)
        if not os.path.isfile(ap) or ap in seen:
            continue
        if os.path.commonpath((out_abs, ap)) == out_abs:  # skip anything under out-dir
            continue
        seen.add(ap)
        files.append(ap)
    return files


def load_sample_keys(out_dir: str) -> Tuple[Set[str], List[str]]:
    keys: Set[str] = set()
    used: List[str] = []
    for path in _candidate_sample_files(out_dir):
        try:
            rows = mv6.load_jsonl(path)
        except (ValueError, OSError):
            continue
        good = {geometry_key(r) for r in rows if r.get("original") is not None}
        if good:
            keys |= good
            used.append(path)
    return keys, used


def build_existing_keys(args) -> Tuple[Dict[str, Set[str]], List[str]]:
    """Return an ordered map ``split_name -> key set`` plus the sample files used."""
    existing: Dict[str, Set[str]] = {}
    for split, keys in load_source_split_keys(args.source_dir).items():
        existing[f"source_{split}"] = keys
    if not args.no_v6_reconstruct:
        existing.update(
            reconstruct_v6_keys(
                args.source_dir, args.v6_seed, args.v6_train_n, args.v6_val_n, args.mix_v6
            )
        )
    sample_keys, sample_files = load_sample_keys(args.out_dir)
    if sample_keys:
        existing["v6_local_samples"] = sample_keys
    return existing, sample_files


# --------------------------------------------------------------------------------------
# Golden generation — same verified injection loop as dataset._inject_partials.
# --------------------------------------------------------------------------------------

def generate_label_partials(
    label: str,
    need: int,
    rng: random.Random,
    seen: Set[str],
    golden_keys: Set[str],
    *,
    patterns: Sequence[Tuple[str, ...]],
    max_attempts_factor: int = 2000,
) -> List[dict]:
    """Generate ``need`` verified, non-colliding partial records for ``label``."""
    out: List[dict] = []
    attempts = 0
    cap = max_attempts_factor * max(need, 1)
    while len(out) < need:
        attempts += 1
        if attempts > cap:
            raise RuntimeError(
                f"could not generate {need} disjoint '{label}' records after {attempts} "
                "attempts (try a different --seed or fewer --n-per-label)"
            )
        pattern = rng.choice(list(patterns))
        problem = problems.make_problem(rng, pattern=pattern)
        injected = errors.inject(problem, label, rng)
        if injected is None:
            continue
        student_seq, student_text = injected
        partial = dataset._partial_record(problem, student_seq, student_text, label)
        key = geometry_key(partial)
        if key in seen or key in golden_keys:  # reject any existing/duplicate geometry
            continue
        golden_keys.add(key)
        out.append(partial)
    return out


def build_golden_records(
    labels: Sequence[str],
    n_per_label: int,
    seed: int,
    seen: Set[str],
) -> List[dict]:
    """Balanced, interleaved, fully finalized + augmented golden records (ids 0..N-1)."""
    rng = random.Random(seed)
    golden_keys: Set[str] = set()
    per_label: Dict[str, List[dict]] = {}
    for label in labels:
        per_label[label] = generate_label_partials(
            label, n_per_label, rng, seen, golden_keys,
            patterns=errors.ID_COMPATIBLE_PATTERNS[label],
        )

    # Round-robin interleave so the file mixes all labels (like the other splits).
    ordered: List[dict] = []
    for index in range(n_per_label):
        for label in labels:
            ordered.append(per_label[label][index])

    records: List[dict] = []
    for rid, partial in enumerate(ordered):
        rec = dataset.finalize_record(partial, rid, GOLDEN_SPLIT, RENDER_SUBDIR)
        rec["v6_pool"] = GOLDEN_SPLIT
        records.append(v6_format.augment_record(rec))
    return records


# --------------------------------------------------------------------------------------
# Verification (disjointness + oracle + balance) — used after build AND in --verify.
# --------------------------------------------------------------------------------------

def check_disjointness(records: Sequence[Mapping], existing: Mapping[str, Set[str]]) -> dict:
    golden_keys = [geometry_key(rec) for rec in records]
    key_set = set(golden_keys)
    duplicates = len(golden_keys) - len(key_set)
    per_split = {}
    for name, keys in existing.items():
        overlap = sorted(key_set & keys)
        per_split[name] = {"records_checked": len(keys), "overlap_count": len(overlap)}
    total_overlap = sum(v["overlap_count"] for v in per_split.values())
    return {
        "golden_records": len(golden_keys),
        "golden_unique_keys": len(key_set),
        "internal_duplicate_keys": duplicates,
        "total_overlap_all_splits": total_overlap,
        "per_split": per_split,
    }


def check_oracle(records: Sequence[Mapping]) -> dict:
    """Independently re-verify each record through the oracle and the v6 target scorer."""
    label_mismatches: List[object] = []
    net_mismatches: List[object] = []
    target_failures: List[object] = []
    for rec in records:
        diagnosed = tc.diagnose(
            rec["original"], rec["correct_transform"], rec["student_transform"]
        )
        if diagnosed != rec["label"]:
            label_mismatches.append(rec.get("id"))
        # augment_record re-derives + verifies the canonical nets against GREEN/BLUE.
        try:
            reaug = v6_format.augment_record(
                {k: rec[k] for k in rec if k not in ("correct_net", "student_net", "schema_version")}
            )
            if reaug["correct_net"] != rec["correct_net"] or reaug["student_net"] != rec["student_net"]:
                net_mismatches.append(rec.get("id"))
        except ValueError:
            net_mismatches.append(rec.get("id"))
        # Every task target must round-trip and score 1.0 (same gate as make_v6).
        try:
            for task in v6_format.TASK_MODES:
                mv6._validate_gold(rec, task)
        except (ValueError, KeyError):
            target_failures.append(rec.get("id"))
    return {
        "records_checked": len(records),
        "label_mismatches": len(label_mismatches),
        "net_mismatches": len(net_mismatches),
        "target_scoring_failures": len(target_failures),
        "label_mismatch_ids": label_mismatches[:20],
        "net_mismatch_ids": net_mismatches[:20],
        "target_failure_ids": target_failures[:20],
    }


def label_balance(records: Sequence[Mapping]) -> Dict[str, int]:
    return dict(sorted(Counter(str(rec["label"]) for rec in records).items()))


def format_report(disjoint: dict, oracle: dict, balance: Dict[str, int], n_per_label: int) -> str:
    lines = ["=== GOLDEN SET VERIFICATION REPORT ==="]
    lines.append(f"records: {disjoint['golden_records']}  unique_geometry_keys: {disjoint['golden_unique_keys']}")
    lines.append("")
    lines.append("per-label balance (target %d each):" % n_per_label)
    for label in tc.DIAGNOSIS_LABELS:
        count = balance.get(label, 0)
        flag = "" if count == n_per_label else "  <-- IMBALANCED"
        lines.append(f"  {label:<34} {count:>4}{flag}")
    balanced = all(balance.get(l, 0) == n_per_label for l in tc.DIAGNOSIS_LABELS)
    lines.append(f"balanced: {balanced}")
    lines.append("")
    lines.append("disjointness vs every existing split (overlap must be 0):")
    for name, info in disjoint["per_split"].items():
        flag = "" if info["overlap_count"] == 0 else "  <-- OVERLAP!"
        lines.append(
            f"  {name:<20} checked={info['records_checked']:>6}  overlap={info['overlap_count']}{flag}"
        )
    lines.append(f"  internal_duplicate_keys: {disjoint['internal_duplicate_keys']}")
    lines.append(f"  TOTAL overlap across all splits: {disjoint['total_overlap_all_splits']}")
    lines.append("")
    lines.append("oracle re-verification (all must be 0):")
    lines.append(f"  label_mismatches:        {oracle['label_mismatches']}")
    lines.append(f"  net_mismatches:          {oracle['net_mismatches']}")
    lines.append(f"  target_scoring_failures: {oracle['target_scoring_failures']}")
    lines.append("")
    ok = (
        balanced
        and disjoint["total_overlap_all_splits"] == 0
        and disjoint["internal_duplicate_keys"] == 0
        and oracle["label_mismatches"] == 0
        and oracle["net_mismatches"] == 0
        and oracle["target_scoring_failures"] == 0
    )
    lines.append(f"RESULT: {'PASS' if ok else 'FAIL'}")
    return "\n".join(lines)


def assert_clean(disjoint: dict, oracle: dict, balance: Dict[str, int], n_per_label: int) -> None:
    """Fail loudly if disjointness / oracle / balance is not perfect."""
    problems_found: List[str] = []
    if disjoint["total_overlap_all_splits"] != 0:
        problems_found.append(f"geometry overlap with existing splits: {disjoint['per_split']}")
    if disjoint["internal_duplicate_keys"] != 0:
        problems_found.append(f"internal duplicate keys: {disjoint['internal_duplicate_keys']}")
    if oracle["label_mismatches"] != 0:
        problems_found.append(f"label mismatches: {oracle['label_mismatch_ids']}")
    if oracle["net_mismatches"] != 0:
        problems_found.append(f"net mismatches: {oracle['net_mismatch_ids']}")
    if oracle["target_scoring_failures"] != 0:
        problems_found.append(f"target scoring failures: {oracle['target_failure_ids']}")
    for label in tc.DIAGNOSIS_LABELS:
        if label in balance and balance[label] != n_per_label:
            problems_found.append(f"label {label} count {balance[label]} != {n_per_label}")
    if problems_found:
        raise SystemExit("GOLDEN SET VERIFICATION FAILED:\n  - " + "\n  - ".join(problems_found))


# --------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------

CHAT_MODALITIES = ("image", "image_coords")
CHAT_TASK = "full"


def write_outputs(
    records: Sequence[dict],
    out_dir: str,
    *,
    render_images: bool,
) -> Tuple[Dict[str, dict], int, dict]:
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, dict] = {}

    records_path = os.path.join(out_dir, "golden_v6.jsonl")
    mv6._atomic_write_jsonl(records_path, records)
    written["golden_v6.jsonl"] = mv6._file_metadata(records_path)

    # Chat files reuse make_v6's exact serialization (incl. _validate_gold per row).
    for modality in CHAT_MODALITIES:
        rows = mv6._chat_rows(records, CHAT_TASK, modality)
        name = f"golden_v6_{modality}_{CHAT_TASK}_chat.jsonl"
        path = os.path.join(out_dir, name)
        mv6._atomic_write_jsonl(path, rows)
        written[name] = mv6._file_metadata(path)

    rendered = 0
    image_validation = {"checked": 0, "missing": 0, "decoded": 0, "available": render_images}
    if render_images:
        rendered = render.render_all(records, out_dir, skip_existing=True, progress_every=50)
        resolved = [os.path.join(out_dir, str(rec["render_path"])) for rec in records]
        missing = [p for p in resolved if not os.path.isfile(p)]
        image_validation["checked"] = len(resolved)
        image_validation["missing"] = len(missing)
        if missing:
            raise RuntimeError(f"golden render paths do not resolve, e.g. {missing[:3]}")
        from PIL import Image
        for path in resolved[: min(3, len(resolved))]:
            with Image.open(path) as image:
                image.verify()
            image_validation["decoded"] += 1
    return written, rendered, image_validation


def write_manifest(
    out_dir: str,
    args,
    records: Sequence[dict],
    disjoint: dict,
    oracle: dict,
    balance: Dict[str, int],
    written: Dict[str, dict],
    rendered: int,
    image_validation: dict,
    sample_files: Sequence[str],
) -> str:
    manifest = {
        "artifact": "dataset_golden_v6",
        "schema_version": v6_format.SCHEMA_VERSION,
        "what_this_is": (
            "A held-out, oracle-verified, balanced v6 canonical-net diagnosis set that is "
            "provably disjoint from every existing split (v6 train/val and frozen source "
            "train/val/test/ood). Same primitives, same oracle, same format as the other "
            "splits; nothing was trained or previously evaluated on it."
        ),
        "generator": "model/make_golden_set.py",
        "generation_config": {
            "seed": args.seed,
            "n_per_label": args.n_per_label,
            "labels": list(args.labels),
            "total_records": len(records),
            "split": GOLDEN_SPLIT,
            "patterns": "errors.ID_COMPATIBLE_PATTERNS (two-step in-distribution)",
            "render_subdir": RENDER_SUBDIR,
        },
        "seeds_in_project": {
            "source_train_val_test_ood": 0,
            "eval_sample": 20260709,
            "v6_train_val": V6_SEED,
            "golden": args.seed,
        },
        "canonical_dedup_key": {
            "definition": 'json.dumps([original, correct_image, student_image], separators=(",", ":"))',
            "meaning": "RED pre-image + GREEN correct image + BLUE student image (encodes the injected error)",
            "matches": [
                "model/make_v6_transform_data.py::_prepare_source_pools.geometry_key",
                "model/audit_v6_predictions.py::geometry_fingerprint",
            ],
        },
        "disjointness": disjoint,
        "oracle_verification": oracle,
        "label_balance": balance,
        "existing_key_universe": {
            "source_dir": os.path.abspath(args.source_dir),
            "v6_reconstructed": not args.no_v6_reconstruct,
            "v6_reconstruct_config": {
                "seed": args.v6_seed,
                "train_n": args.v6_train_n,
                "val_n": args.v6_val_n,
                "mix": list(args.mix_v6),
            },
            "local_sample_files": [os.path.abspath(p) for p in sample_files],
        },
        "rendering": {
            "matplotlib_available": render.MATPLOTLIB_AVAILABLE,
            "rendered_now": rendered,
            "image_validation": image_validation,
        },
        "output_files": written,
        "how_to_eval": (
            "Drop golden_v6.jsonl + renders_v6/golden/ into a data dir and run "
            "eval_transform.py --data-dir <dir> --splits golden (see model/GOLDEN_SET.md)."
        ),
    }
    path = os.path.join(out_dir, "manifest_golden.json")
    mv6._atomic_write_json(path, manifest)
    return path


def write_readme(out_dir: str, args, records: Sequence[dict], rendered_ok: bool) -> str:
    render_note = (
        "Images are rendered under `renders_v6/golden/` and this directory is a complete, "
        "self-contained data dir."
        if rendered_ok
        else "Images were NOT rendered locally (matplotlib unavailable). Render them on the "
        "cluster with the command in `model/GOLDEN_SET.md` before evaluating."
    )
    text = f"""# dataset_golden_v6 — held-out golden evaluation set

{len(records)} oracle-verified v6 canonical-net diagnosis records ({args.n_per_label}
per each of the 8 labels), seed `{args.seed}`.

## What it is / why it's disjoint

Every record is generated by the SAME verified pipeline as train/val/test/ood
(`problems.make_problem` -> `errors.inject` -> `dataset._partial_record` ->
`dataset.finalize_record` -> `v6_format.augment_record`) and diagnosed by the SAME
geometry oracle (`transform_core.diagnose`). It is **not** a new task.

Disjointness is guaranteed on the pipeline's own canonical dedup key —
`json.dumps([original, correct_image, student_image], separators=(",", ":"))` (RED +
GREEN + BLUE, i.e. problem geometry + injected error). Each candidate is rejected if
its key collides with ANY record in v6 train/val (reconstructed deterministically from
seed {V6_SEED}) or the frozen source train/val/test/ood; the generator asserts
0 overlap and 0 oracle mismatches before writing. Re-verify anytime:

```bash
python3 model/make_golden_set.py --verify --out-dir dataset_golden_v6 \\
  --source-dir <path to transform_diagnosis_data>
```

## Files

- `golden_v6.jsonl` — full v6 records (same schema as `train_v6.jsonl`).
- `golden_v6_image_full_chat.jsonl` — `task=full`, `input_mode=image` chat rows.
- `golden_v6_image_coords_full_chat.jsonl` — `task=full`, `input_mode=image_coords`.
- `renders_v6/golden/<id>.png` — one RED/GREEN/BLUE figure per record.
- `manifest_golden.json` — config + disjointness + oracle report + checksums.

{render_note}

## Evaluate

See `model/GOLDEN_SET.md` for the exact base / tuned / hintfix cluster commands.
"""
    path = os.path.join(out_dir, "README.md")
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        handle.write(text)
    os.replace(tmp, path)
    return path


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------

def parse_labels(text: str) -> Tuple[str, ...]:
    labels = tuple(item for item in text.split(",") if item)
    unknown = [l for l in labels if l not in tc.DIAGNOSIS_LABELS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown labels {unknown}; choose from {tc.DIAGNOSIS_LABELS}")
    return labels or tuple(tc.DIAGNOSIS_LABELS)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=os.path.join(REPO_ROOT, "dataset_golden_v6"))
    parser.add_argument("--source-dir", default=default_source_dir(),
                        help="dir with frozen train/val/test/ood.jsonl")
    parser.add_argument("--n-per-label", type=int, default=DEFAULT_N_PER_LABEL)
    parser.add_argument("--labels", type=parse_labels, default=tuple(tc.DIAGNOSIS_LABELS),
                        help="comma-separated subset of the 8 diagnosis labels")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-v6-reconstruct", action="store_true",
                        help="skip deterministic v6 train/val reconstruction (rely on samples only)")
    parser.add_argument("--v6-seed", type=int, default=V6_SEED)
    parser.add_argument("--v6-train-n", type=int, default=V6_TRAIN_N)
    parser.add_argument("--v6-val-n", type=int, default=V6_VAL_N)
    parser.add_argument("--v6-mix", type=mv6.parse_mix, default=V6_MIX, dest="mix_v6",
                        help="source,contrastive,curriculum,hard fractions for v6 reconstruction")
    parser.add_argument("--no-render", action="store_true", help="do not render PNGs locally")
    parser.add_argument("--verify", action="store_true",
                        help="re-verify an existing golden_v6.jsonl in --out-dir and exit")
    parser.add_argument("--print", type=int, default=2, dest="print_n")
    return parser


def run_verify(args) -> int:
    path = os.path.join(args.out_dir, "golden_v6.jsonl")
    if not os.path.isfile(path):
        raise SystemExit(f"--verify: no golden set at {path}")
    records = mv6.load_jsonl(path)
    existing, _ = build_existing_keys(args)
    disjoint = check_disjointness(records, existing)
    oracle = check_oracle(records)
    balance = label_balance(records)
    n_per_label = min(balance.values()) if balance else args.n_per_label
    print(format_report(disjoint, oracle, balance, n_per_label))
    assert_clean(disjoint, oracle, balance, n_per_label)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.verify:
        return run_verify(args)

    print(f"building existing-key universe from {os.path.abspath(args.source_dir)} ...", flush=True)
    existing, sample_files = build_existing_keys(args)
    seen: Set[str] = set()
    for keys in existing.values():
        seen |= keys
    print(
        "existing keys: "
        + ", ".join(f"{name}={len(keys)}" for name, keys in existing.items())
        + f"  (union={len(seen)})",
        flush=True,
    )
    if sample_files:
        print(f"local v6 sample files unioned: {len(sample_files)}", flush=True)

    print(
        f"generating golden set: {len(args.labels)} labels x {args.n_per_label} "
        f"= {len(args.labels) * args.n_per_label} records (seed {args.seed}) ...",
        flush=True,
    )
    records = build_golden_records(args.labels, args.n_per_label, args.seed, seen)

    disjoint = check_disjointness(records, existing)
    oracle = check_oracle(records)
    balance = label_balance(records)
    print("\n" + format_report(disjoint, oracle, balance, args.n_per_label) + "\n")
    assert_clean(disjoint, oracle, balance, args.n_per_label)

    for rec in records[: args.print_n]:
        print(f"[golden id={rec['id']} label={rec['label']}] full={v6_format.target_json(rec, 'full')}")

    render_images = not args.no_render and render.MATPLOTLIB_AVAILABLE
    written, rendered, image_validation = write_outputs(
        records, args.out_dir, render_images=render_images
    )
    manifest_path = write_manifest(
        args.out_dir, args, records, disjoint, oracle, balance,
        written, rendered, image_validation, sample_files,
    )
    readme_path = write_readme(args.out_dir, args, records, rendered_ok=render_images)

    print(f"\nwrote golden set to {os.path.abspath(args.out_dir)}")
    print(f"  records: golden_v6.jsonl ({len(records)})")
    print(f"  chat:    golden_v6_image_full_chat.jsonl, golden_v6_image_coords_full_chat.jsonl")
    print(f"  manifest:{manifest_path}")
    print(f"  readme:  {readme_path}")
    if render_images:
        print(f"  renders: {rendered} new PNG(s) under {RENDER_SUBDIR}/")
    else:
        reason = "matplotlib unavailable" if not render.MATPLOTLIB_AVAILABLE else "--no-render"
        print(f"  renders: SKIPPED ({reason}). Render on the cluster; see model/GOLDEN_SET.md:")
        print(
            "    python make_golden_set.py --out-dir ~/transform_diagnosis_data_golden "
            f"--source-dir ~/transform_diagnosis_data --seed {args.seed} "
            f"--n-per-label {args.n_per_label}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
