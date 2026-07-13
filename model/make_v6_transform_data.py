"""Build the non-destructive v6 canonical-net transform curriculum.

The source v1-v5 dataset is read only.  v6 is written to a separate directory,
reuses source renders through a read-only symlink, and places every newly
generated curriculum render under the v6 output tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from collections import Counter, defaultdict
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for candidate in (REPO_ROOT, HERE, HOME):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from transform_diagnosis import contrastive, dataset, errors, eval as ev
    from transform_diagnosis import net_transform as nt
    from transform_diagnosis import problems, render, transform_core as tc, v6_format
except ModuleNotFoundError:  # cluster package fallback
    from slm_eval import eval as ev, net_transform as nt, transform_core as tc, v6_format
    from transform_diagnosis import contrastive, dataset, errors, problems, render

DEFAULT_SOURCE = os.path.join(HOME, "transform_diagnosis_data")
DEFAULT_OUTPUT = os.path.join(HOME, "transform_diagnosis_data_v6")
DEFAULT_SEED = 20260711
SOURCE_LINK = "source_data"
POOL_NAMES = ("source", "contrastive", "curriculum", "hard")
MODALITIES = ("image", "image_coords")
TASKS = v6_format.TASK_MODES
_SPLIT_SALTS = {"train": 0x61A7E, "val": 0xB4C3D}
_TEMP_ID_BASE = {"train": 6_000_000, "val": 7_000_000}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_metadata(path: str) -> dict:
    return {
        "path": os.path.abspath(path),
        "bytes": os.path.getsize(path),
        "sha256": _sha256(path),
        "records": len(load_jsonl(path)),
    }


def _source_jsonl_metadata(
    source_dir: str,
    explicit_paths: Sequence[str],
) -> Dict[str, dict]:
    """Checksum every top-level source JSONL, including frozen test/OOD files."""
    candidates = {
        os.path.abspath(os.path.join(source_dir, name))
        for name in os.listdir(source_dir)
        if name.endswith(".jsonl") and os.path.isfile(os.path.join(source_dir, name))
    }
    candidates.update(os.path.abspath(path) for path in explicit_paths)
    metadata: Dict[str, dict] = {}
    source_abs = os.path.abspath(source_dir)
    for path in sorted(candidates):
        key = (
            os.path.relpath(path, source_abs)
            if os.path.commonpath((source_abs, path)) == source_abs
            else path
        )
        metadata[key] = _file_metadata(path)
    return metadata


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _atomic_write_jsonl(path: str, rows: Sequence[Mapping]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _atomic_write_json(path: str, value: Mapping) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


def parse_mix(text: str) -> Tuple[float, float, float, float]:
    parts = [float(item) for item in text.split(",")]
    if len(parts) != 4 or any(item < 0 for item in parts):
        raise argparse.ArgumentTypeError(
            "--mix needs four nonnegative fractions: source,contrastive,curriculum,hard"
        )
    total = sum(parts)
    if total <= 0:
        raise argparse.ArgumentTypeError("--mix fractions must sum to a positive value")
    return tuple(item / total for item in parts)  # type: ignore[return-value]


def plan_counts(n: int, mix: Sequence[float]) -> Dict[str, int]:
    """Allocate an exact total; contrastive count is a whole number of quadruplets."""
    if n < 0:
        raise ValueError("record count cannot be negative")
    counts = {name: round(n * fraction) for name, fraction in zip(POOL_NAMES, mix)}
    counts["contrastive"] = (counts["contrastive"] // 4) * 4
    counts["source"] += n - sum(counts.values())
    if counts["source"] < 0:
        for name in ("hard", "curriculum", "contrastive"):
            take = min(counts[name], -counts["source"])
            if name == "contrastive":
                take = (take // 4) * 4
            counts[name] -= take
            counts["source"] += take
            if counts["source"] >= 0:
                break
    assert sum(counts.values()) == n and all(value >= 0 for value in counts.values())
    return counts


def _balanced_source_sample(records: Sequence[dict], n: int, rng: random.Random) -> List[dict]:
    """Sample without replacement over D4 class × translation-bucket strata."""
    groups: Dict[Tuple[str, str], List[dict]] = defaultdict(list)
    for rec in records:
        net = nt.sequence_to_net(rec["correct_transform"])
        groups[(str(net["linear"]), _translation_bucket(net))].append(rec)
    for group in groups.values():
        group.sort(key=lambda row: (str(row.get("id")), row.get("render_path", "")))
        rng.shuffle(group)
    order = [
        key
        for linear in nt.D4_LINEAR_NAMES
        for key in sorted(groups)
        if key[0] == linear and groups.get(key)
    ]
    selected: List[dict] = []
    while len(selected) < min(n, len(records)):
        progressed = False
        for key in order:
            if groups[key] and len(selected) < n:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    return selected


def _primitive_degree(transform: tc.Transform) -> int | None:
    for degree in (90, 180, 270):
        if transform == tc.rotate(degree):
            return degree
    return None


def _primitive_line(transform: tc.Transform) -> str | None:
    for line in problems.REFLECTION_LINES:
        if transform == tc.reflect(line):
            return line
    return None


HardPredicate = Callable[[problems.Problem, Sequence[tc.Transform]], bool]


def _hard_specs() -> List[Tuple[str, Tuple[str, ...], str, HardPredicate]]:
    specs: List[Tuple[str, Tuple[str, ...], str, HardPredicate]] = []
    for degree in (90, 180, 270):
        specs.append((
            f"rotation_angle_{degree}",
            ("rotate",),
            "wrong_rotation_angle",
            lambda problem, _student, degree=degree:
                _primitive_degree(problem.answer[0]) == degree,
        ))
    specs.append((
        "translation_opposite_direction",
        ("translate",),
        "opposite_translation",
        lambda _problem, _student: True,
    ))
    for line in problems.REFLECTION_LINES:
        safe = line.replace("=", "_eq_").replace("-", "neg_")
        specs.append((
            f"reflection_line_{safe}",
            ("reflect",),
            "wrong_reflection_line",
            lambda problem, _student, line=line: _primitive_line(problem.answer[0]) == line,
        ))

    def differs_by_one(problem: problems.Problem, student: Sequence[tc.Transform]) -> bool:
        cdx, cdy = problem.answer[0].vec
        sdx, sdy = student[0].vec
        return (
            (abs(cdx - sdx) == 1 and cdy == sdy)
            or (abs(cdy - sdy) == 1 and cdx == sdx)
        )

    specs.append((
        "translation_parameter_off_by_one",
        ("translate",),
        "wrong_translation",
        differs_by_one,
    ))
    specs.extend([
        (
            "operation_type_rotation_to_reflection",
            ("rotate",),
            "reflection_instead_of_rotation",
            lambda _problem, _student: True,
        ),
        (
            "operation_type_reflection_to_rotation",
            ("reflect",),
            "rotation_instead_of_reflection",
            lambda _problem, _student: True,
        ),
    ])
    return specs


HARD_SPECS = _hard_specs()


def build_hard_partials(
    rng: random.Random,
    n: int,
    *,
    max_attempts_per_record: int = 4000,
) -> List[Tuple[dict, str]]:
    """Select parameter-focused records produced by the existing generator/injectors."""
    output: List[Tuple[dict, str]] = []
    for index in range(n):
        name, pattern, label, predicate = HARD_SPECS[index % len(HARD_SPECS)]
        for _ in range(max_attempts_per_record):
            try:
                problem = problems.make_problem(rng, pattern=pattern)
            except RuntimeError:
                continue
            injected = errors.inject(problem, label, rng)
            if injected is None:
                continue
            student, student_text = injected
            if not predicate(problem, student):
                continue
            partial = dataset._partial_record(problem, student, student_text, label)
            output.append((partial, name))
            break
        else:
            raise RuntimeError(
                f"could not produce hard-focus record {name!r} after "
                f"{max_attempts_per_record} attempts"
            )
    return output


def _source_image_path(source_dir: str, rec: Mapping) -> str:
    path = str(rec["render_path"])
    return path if os.path.isabs(path) else os.path.join(source_dir, path)


def _copy_source_record(rec: Mapping, split: str) -> dict:
    copied = dict(rec)
    copied["source_id"] = rec.get("id")
    copied["source_split"] = rec.get("split")
    copied["split"] = split
    render_path = str(rec["render_path"])
    copied["render_path"] = (
        render_path if os.path.isabs(render_path)
        else os.path.join(SOURCE_LINK, render_path)
    )
    copied["v6_pool"] = "source"
    return copied


def _finalize_new(
    partials: Sequence[Tuple[dict, str | None]],
    split: str,
    pool: str,
    start_id: int,
) -> List[dict]:
    records: List[dict] = []
    render_subdir = os.path.join("renders_v6", split)
    for offset, (partial, focus) in enumerate(partials):
        rec = dataset.finalize_record(partial, start_id + offset, split, render_subdir)
        rec["v6_pool"] = pool
        if focus:
            rec["v6_focus"] = focus
        records.append(rec)
    return records


def _build_split(
    split: str,
    n: int,
    mix: Sequence[float],
    seed: int,
    source_records: Sequence[dict],
) -> Tuple[List[dict], List[dict], Dict[str, int]]:
    counts = plan_counts(n, mix)
    rng = random.Random(seed ^ _SPLIT_SALTS[split])
    source = _balanced_source_sample(source_records, counts["source"], rng)
    if len(source) < counts["source"]:
        short = counts["source"] - len(source)
        counts["source"] = len(source)
        counts["curriculum"] += short

    con_partials, _ = contrastive.build_contrastive_partials(
        random.Random(rng.getrandbits(64)), counts["contrastive"] // 4
    )
    cur_partials = contrastive.build_curriculum_partials(
        random.Random(rng.getrandbits(64)), counts["curriculum"]
    )
    hard = build_hard_partials(random.Random(rng.getrandbits(64)), counts["hard"])

    next_id = _TEMP_ID_BASE[split]
    curriculum_records = _finalize_new(
        [(partial, None) for partial in cur_partials],
        split,
        "curriculum",
        next_id,
    )
    next_id += len(curriculum_records)
    hard_records = _finalize_new(hard, split, "hard", next_id)
    next_id += len(hard_records)
    contrastive_records = _finalize_new(
        [(partial, None) for partial in con_partials],
        split,
        "contrastive",
        next_id,
    )
    source_copies = [_copy_source_record(rec, split) for rec in source]

    # Easy one-step records come first, followed by focused contrasts, matched
    # two-step contrasts, and ordinary two-step source examples.
    records = curriculum_records + hard_records + contrastive_records + source_copies
    for new_id, rec in enumerate(records):
        rec["id"] = new_id
        augmented = v6_format.augment_record(rec)
        rec.clear()
        rec.update(augmented)
    return records, curriculum_records + hard_records + contrastive_records, counts


def _translation_bucket(net: Mapping) -> str:
    tx, ty = int(net["tx"]), int(net["ty"])
    magnitude = max(abs(tx), abs(ty))
    if (tx, ty) == (0, 0):
        return "zero"
    shape = "axis" if tx == 0 or ty == 0 else "diagonal"
    size = "small_1_3" if magnitude <= 3 else "medium_4_6" if magnitude <= 6 else "large_7_plus"
    return f"{shape}_{size}"


def distributions(records: Sequence[Mapping]) -> dict:
    return {
        "label": dict(sorted(Counter(str(rec["label"]) for rec in records).items())),
        "pool": dict(sorted(Counter(str(rec.get("v6_pool", "unknown")) for rec in records).items())),
        "focus": dict(sorted(Counter(
            str(rec["v6_focus"]) for rec in records if rec.get("v6_focus")
        ).items())),
        "step_count": dict(sorted(Counter(
            str(len(rec["correct_transform"])) for rec in records
        ).items())),
        "correct_linear": dict(sorted(Counter(
            str(rec["correct_net"]["linear"]) for rec in records
        ).items())),
        "student_linear": dict(sorted(Counter(
            str(rec["student_net"]["linear"]) for rec in records
        ).items())),
        "correct_translation_bucket": dict(sorted(Counter(
            _translation_bucket(rec["correct_net"]) for rec in records
        ).items())),
        "student_translation_bucket": dict(sorted(Counter(
            _translation_bucket(rec["student_net"]) for rec in records
        ).items())),
    }


def _validate_gold(record: Mapping, task: str) -> None:
    target = v6_format.target_json(record, task)
    parsed = ev.parse_pred(target)
    if parsed != v6_format.target_obj(record, task):
        raise ValueError(f"id={record.get('id')} task={task} target failed JSON round trip")
    row = ev.score_record(target, dict(record), task_mode=task)
    expected_fields = {
        "correct": ("correct_net_ok",),
        "student": ("student_net_ok",),
        "both": ("correct_net_ok", "student_net_ok", "both_nets_ok", "derived_label_ok"),
        "full": (
            "correct_net_ok", "student_net_ok", "both_nets_ok",
            "label_ok", "derived_label_ok", "hint_ok",
        ),
    }[task]
    if not all(row[field] is True for field in expected_fields):
        raise ValueError(f"id={record.get('id')} task={task} gold did not score 1.0: {row}")


def _chat_rows(records: Sequence[Mapping], task: str, modality: str) -> List[dict]:
    rows = []
    for rec in records:
        _validate_gold(rec, task)
        rows.append(v6_format.conversation(rec, task, modality))
    return rows


def _prepare_source_pools(
    train_path: str,
    val_path: str,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    train = load_jsonl(train_path)
    val = load_jsonl(val_path)
    for path, rows in ((train_path, train), (val_path, val)):
        leaked = [rec.get("id") for rec in rows if rec.get("split") in ("test", "ood")]
        if leaked:
            raise ValueError(f"{path} contains frozen test/OOD records (e.g. {leaked[:5]})")
    if os.path.realpath(train_path) == os.path.realpath(val_path):
        # Local/sample convenience without leakage: deterministically partition the
        # source file into disjoint geometry pools.
        all_rows = sorted(train, key=lambda rec: str(rec.get("id")))
        random.Random(seed ^ 0xD15A).shuffle(all_rows)
        cut = max(1, round(0.8 * len(all_rows)))
        if cut >= len(all_rows):
            cut = max(0, len(all_rows) - 1)
        train, val = all_rows[:cut], all_rows[cut:]
    def geometry_key(rec: Mapping) -> str:
        return json.dumps(
            [rec.get("original"), rec.get("correct_image"), rec.get("student_image")],
            separators=(",", ":"),
        )

    train_keys = {geometry_key(rec) for rec in train}
    val_keys = {geometry_key(rec) for rec in val}
    overlap = train_keys & val_keys
    if overlap:
        raise ValueError("source train/val geometry overlap detected")
    return train, val


def _config(args, train_path: str, val_path: str) -> dict:
    return {
        "schema_version": v6_format.SCHEMA_VERSION,
        "seed": args.seed,
        "source_dir": os.path.abspath(args.source_dir),
        "source_train": os.path.abspath(train_path),
        "source_val": os.path.abspath(val_path),
        "train_n": args.train_n,
        "val_n": args.val_n,
        "mix_source_contrastive_curriculum_hard": list(args.mix),
        "modalities": list(MODALITIES),
        "tasks": list(TASKS),
    }


def _guard_output(
    out_dir: str,
    config: Mapping,
    source_metadata: Mapping,
    resume: bool,
) -> None:
    if not os.path.isdir(out_dir):
        return
    entries = os.listdir(out_dir)
    if not entries:
        return
    if not resume:
        raise SystemExit(
            f"refusing nonempty output directory {out_dir}; choose a new --out-dir or pass "
            "--resume-existing-output to resume the same deterministic generation"
        )
    manifest_path = os.path.join(out_dir, "manifest_v6.json")
    if os.path.exists(manifest_path):
        with open(manifest_path) as handle:
            old = json.load(handle)
        if old.get("generation_config") != dict(config):
            raise SystemExit(
                "--resume-existing-output config does not match the existing v6 manifest; "
                "use a new output directory"
            )
        prior_source = old.get("source", {}).get("before")
        if prior_source is not None and prior_source != dict(source_metadata):
            raise SystemExit(
                "--resume-existing-output source checksums differ from the existing manifest; "
                "use a new output directory"
            )


def _ensure_source_link(out_dir: str, source_dir: str) -> None:
    link = os.path.join(out_dir, SOURCE_LINK)
    target = os.path.abspath(source_dir)
    if os.path.lexists(link):
        if not os.path.islink(link) or os.path.realpath(link) != os.path.realpath(target):
            raise RuntimeError(f"existing {link} is not the expected source symlink")
        return
    os.symlink(target, link, target_is_directory=True)


def _legacy_artifacts(source_dir: str) -> dict:
    adapter_paths = [
        os.path.join(HOME, "lora_adapters"),
        os.path.join(HOME, "lora_adapters_v2"),
        os.path.join(HOME, "lora_adapters_v3cot"),
        os.path.join(HOME, "lora_adapters_v4"),
        os.path.join(HOME, "lora_adapters_v5"),
    ]
    checkpoint_paths = [
        os.path.join(HOME, name)
        for name in ("outputs", "outputs_v2", "outputs_v3cot", "outputs_v4", "outputs_v5")
    ]
    return {
        "dataset_directory": os.path.abspath(source_dir),
        "checkpoint_directories": [
            {"path": os.path.abspath(path), "exists": os.path.exists(path)}
            for path in [*adapter_paths, *checkpoint_paths]
        ],
        "result_search_directories": [
            {"path": HOME, "exists": os.path.isdir(HOME)},
            {"path": REPO_ROOT, "exists": os.path.isdir(REPO_ROOT)},
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default=DEFAULT_SOURCE)
    parser.add_argument("--source-train", default=None,
                        help="default: <source-dir>/train.jsonl")
    parser.add_argument("--source-val", default=None,
                        help="default: <source-dir>/val.jsonl")
    parser.add_argument("--out-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--train-n", type=int, default=9600)
    parser.add_argument("--val-n", type=int, default=400)
    parser.add_argument("--mix", type=parse_mix, default=(0.50, 0.20, 0.15, 0.15),
                        help="source,contrastive,curriculum,hard fractions")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resume-existing-output", action="store_true",
                        help="resume only when an existing manifest has the identical config")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-render", type=int, default=0,
                        help="render at most N new records per split (0=all)")
    parser.add_argument("--print", type=int, default=2, dest="print_n")
    parser.add_argument("--dry-run", action="store_true",
                        help="build and validate in memory; write/render nothing")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not os.path.isdir(args.source_dir):
        raise SystemExit(f"source directory not found: {args.source_dir}")
    source_abs = os.path.realpath(args.source_dir)
    output_abs = os.path.realpath(args.out_dir)
    common = os.path.commonpath((source_abs, output_abs))
    if common in (source_abs, output_abs):
        raise SystemExit(
            "--out-dir and --source-dir must be disjoint directories; v6 never writes "
            "inside or above the legacy source tree"
        )
    train_path = args.source_train or os.path.join(args.source_dir, "train.jsonl")
    val_path = args.source_val or os.path.join(args.source_dir, "val.jsonl")
    for path in (train_path, val_path):
        if not os.path.isfile(path):
            raise SystemExit(f"source JSONL not found: {path}")

    source_before = _source_jsonl_metadata(
        args.source_dir, (train_path, val_path)
    )
    config = _config(args, train_path, val_path)
    if not args.dry_run:
        _guard_output(
            args.out_dir, config, source_before, args.resume_existing_output
        )

    train_source, val_source = _prepare_source_pools(train_path, val_path, args.seed)
    missing_source_images = [
        _source_image_path(args.source_dir, rec)
        for rec in [*train_source, *val_source]
        if not os.path.isfile(_source_image_path(args.source_dir, rec))
    ]
    if missing_source_images:
        raise SystemExit(
            f"{len(missing_source_images)} source render(s) are missing, e.g. "
            f"{missing_source_images[:3]}"
        )
    train_records, train_new, train_counts = _build_split(
        "train", args.train_n, args.mix, args.seed, train_source
    )
    val_records, val_new, val_counts = _build_split(
        "val", args.val_n, args.mix, args.seed, val_source
    )

    # Verify every target for every task before any output is committed.
    chat: Dict[Tuple[str, str, str], List[dict]] = {}
    for split, records in (("train", train_records), ("val", val_records)):
        for modality in MODALITIES:
            for task in TASKS:
                chat[(split, modality, task)] = _chat_rows(records, task, modality)

    for rec in train_records[:args.print_n]:
        print(f"\n[v6 sample id={rec['id']} pool={rec['v6_pool']} label={rec['label']}]")
        for task in TASKS:
            print(f"  {task:7s} {v6_format.target_json(rec, task)}")

    written: Dict[str, dict] = {}
    rendered = {"train": 0, "val": 0}
    image_validation = {"checked": 0, "missing": 0, "decoded": 0}
    if not args.dry_run:
        os.makedirs(args.out_dir, exist_ok=True)
        _ensure_source_link(args.out_dir, args.source_dir)
        for split, records in (("train", train_records), ("val", val_records)):
            path = os.path.join(args.out_dir, f"{split}_v6.jsonl")
            _atomic_write_jsonl(path, records)
            written[os.path.basename(path)] = _file_metadata(path)
        for (split, modality, task), rows in chat.items():
            name = f"{split}_v6_{modality}_{task}_chat.jsonl"
            path = os.path.join(args.out_dir, name)
            _atomic_write_jsonl(path, rows)
            written[name] = _file_metadata(path)

        if not args.no_render:
            for split, new_records in (("train", train_new), ("val", val_new)):
                selected = new_records[:args.max_render] if args.max_render else new_records
                rendered[split] = render.render_all(
                    selected, args.out_dir, skip_existing=True, progress_every=100
                )
        # A full/default generation must leave every chat image resolvable.  A
        # deliberate --no-render/--max-render smoke run may leave new renders pending.
        if not args.no_render and not args.max_render:
            all_records = [*train_records, *val_records]
            resolved = [
                path if os.path.isabs(path) else os.path.join(args.out_dir, path)
                for path in (str(rec["render_path"]) for rec in all_records)
            ]
            missing = [path for path in resolved if not os.path.isfile(path)]
            image_validation["checked"] = len(resolved)
            image_validation["missing"] = len(missing)
            if missing:
                raise RuntimeError(f"v6 chat image paths do not resolve, e.g. {missing[:3]}")
            from PIL import Image
            for path in resolved[: min(3, len(resolved))]:
                with Image.open(path) as image:
                    image.verify()
                image_validation["decoded"] += 1

    source_after = _source_jsonl_metadata(
        args.source_dir, (train_path, val_path)
    )
    source_unchanged = {
        key: source_before[key]["sha256"] == source_after[key]["sha256"]
        for key in source_before
    }
    if not all(source_unchanged.values()):
        raise RuntimeError("source JSONL checksum changed during v6 generation")

    manifest = {
        "schema_version": v6_format.SCHEMA_VERSION,
        "generation_config": config,
        "source": {
            "before": source_before,
            "after": source_after,
            "byte_for_byte_unchanged": source_unchanged,
            "render_reuse": (
                f"{SOURCE_LINK} is a read-only-by-convention symlink to the source; "
                "the generator never writes through it"
            ),
        },
        "counts": {
            "train": len(train_records),
            "val": len(val_records),
            "train_pool_plan": train_counts,
            "val_pool_plan": val_counts,
            "chat_rows_per_modality_task": {
                f"{split}/{modality}/{task}": len(rows)
                for (split, modality, task), rows in chat.items()
            },
            "new_renders_created_now": rendered,
            "image_validation": image_validation,
        },
        "distributions": {
            "train": distributions(train_records),
            "val": distributions(val_records),
        },
        "hard_focus_spec_order": [spec[0] for spec in HARD_SPECS],
        "output_files": written,
        "legacy_artifacts": _legacy_artifacts(args.source_dir),
        "frozen_splits": {
            "test": "checksummed only; not loaded into generation or written",
            "ood": "checksummed only; not loaded into generation or written",
        },
    }
    if not args.dry_run:
        manifest_path = os.path.join(args.out_dir, "manifest_v6.json")
        _atomic_write_json(manifest_path, manifest)
        print(f"\nwrote v6 dataset to {os.path.abspath(args.out_dir)}")
        print(f"manifest: {manifest_path}")
    else:
        print("\nDRY RUN: no files or renders written")
    print(f"records: train={len(train_records)} val={len(val_records)}")
    print(
        f"source JSONL checksums unchanged: "
        f"{sum(source_unchanged.values())}/{len(source_unchanged)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
