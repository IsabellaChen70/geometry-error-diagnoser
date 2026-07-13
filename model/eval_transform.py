"""Evaluate v6 canonical-net tasks with image or image+coordinates inputs."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from typing import Dict, List, Sequence

HOME = os.path.expanduser("~")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
for candidate in (REPO_ROOT, HERE, HOME):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

try:
    from transform_diagnosis import eval as ev, v6_format
except ModuleNotFoundError:
    from slm_eval import eval as ev, v6_format

DEFAULT_SOURCE_DATA = os.path.join(HOME, "transform_diagnosis_data")
DEFAULT_BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
DEFAULT_SEED = 20260709
DEFAULT_SAMPLE = 500
DEFAULT_ADAPTERS = {
    "image": os.path.join(HOME, "lora_adapters_v6_image"),
    "image_coords": os.path.join(HOME, "lora_adapters_v6_coords"),
}


def _split_path(data_dir: str, split: str) -> str:
    v6 = os.path.join(data_dir, f"{split}_v6.jsonl")
    return v6 if os.path.isfile(v6) else os.path.join(data_dir, f"{split}.jsonl")


def load_records(data_dir: str, split: str) -> Dict[object, dict]:
    path = _split_path(data_dir, split)
    records: Dict[object, dict] = {}
    with open(path) as handle:
        for line in handle:
            if line.strip():
                source = json.loads(line)
                rec = source if "correct_net" in source else v6_format.augment_record(source)
                records[rec["id"]] = rec
    return records


def select_ids(records: Dict[object, dict], sample: int, seed: int, limit: int) -> List[object]:
    ids = list(records)
    try:
        stable = sorted(ids)
    except TypeError:
        stable = sorted(ids, key=lambda value: (str(type(value)), str(value)))
    if sample:
        ids = random.Random(seed).sample(stable, min(sample, len(stable)))
        try:
            ids = sorted(ids)
        except TypeError:
            ids = sorted(ids, key=lambda value: (str(type(value)), str(value)))
    else:
        ids = stable
    return ids[:limit] if limit else ids


def _resolve_image(data_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(data_dir, path)


def materialize_user_message(record: dict, task: str, input_mode: str, data_dir: str) -> dict:
    """Build the shared v6 prompt and decode its one image."""
    from PIL import Image

    message = v6_format.user_message(record, task, input_mode)
    message = copy.deepcopy(message)
    for part in message["content"]:
        if part.get("type") == "image" and isinstance(part.get("image"), str):
            path = _resolve_image(data_dir, part["image"])
            with Image.open(path) as image:
                part["image"] = image.convert("RGB").copy()
    return message


def load_model(adapter: str | None, base_model: str):
    """Load an adapter when supplied, otherwise the requested base checkpoint."""
    from unsloth import FastVisionModel

    source = adapter or base_model
    model, tokenizer = FastVisionModel.from_pretrained(source, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    print(f"loaded {'adapter' if adapter else 'base model'}: {source}", flush=True)
    return model, tokenizer


def run_model(model, tokenizer, user_message: dict, max_new_tokens: int) -> str:
    image = next(
        part["image"] for part in user_message["content"] if part.get("type") == "image"
    )
    prompt = tokenizer.apply_chat_template([user_message], add_generation_prompt=True)
    inputs = tokenizer(
        image,
        prompt,
        add_special_tokens=False,
        return_tensors="pt",
    ).to("cuda")
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    return tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    ).strip()


def _rate(value) -> str:
    return "--" if value is None else f"{value:.3f}"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", choices=("image", "image_coords"), default="image")
    parser.add_argument("--task", choices=v6_format.TASK_MODES, default="full")
    parser.add_argument("--splits", default="test,ood")
    parser.add_argument("--data-dir", default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--sample", type=int, default=DEFAULT_SAMPLE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--adapter", default=None,
                        help="LoRA adapter path (default selected by --input); use --base-only to omit")
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="verified base model ID/path; also used for --base-only")
    parser.add_argument("--base-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--dry-run", action="store_true",
                        help="no model/GPU: build prompts and score oracle v6 targets")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    splits = [split for split in args.splits.split(",") if split]
    adapter = None if args.base_only else (args.adapter or DEFAULT_ADAPTERS[args.input])
    tag = args.tag or f"v6_{'coords' if args.input == 'image_coords' else 'image'}_{args.task}"

    if not args.dry_run:
        if adapter and os.path.isabs(adapter) and not os.path.isdir(adapter):
            raise SystemExit(f"adapter directory not found: {adapter}")
        model, tokenizer = load_model(adapter, args.base_model)
    else:
        model = tokenizer = None
        tag += "_dryrun"
        print("DRY RUN: no model/GPU; oracle targets exercise prompt + scoring paths")

    os.makedirs(args.output_dir, exist_ok=True)
    for split in splits:
        records = load_records(args.data_dir, split)
        ids = select_ids(records, args.sample, args.seed, args.limit)
        if not ids:
            raise SystemExit(f"no records selected for split {split}")
        scored = []
        for index, record_id in enumerate(ids):
            rec = records[record_id]
            user = v6_format.user_message(rec, args.task, args.input)
            image_part = next(
                (part for part in user["content"] if part.get("type") == "image"),
                None,
            )
            if image_part is None:
                raise RuntimeError("v6 image arms require one image content part")
            image_path = _resolve_image(args.data_dir, image_part["image"])
            if not os.path.isfile(image_path):
                raise FileNotFoundError(f"image does not resolve: {image_path}")
            if args.dry_run:
                output = v6_format.target_json(rec, args.task)
                if index == 0:
                    printable = copy.deepcopy(user)
                    print(
                        f"\n[{split}] first id={record_id} payload:\n"
                        + json.dumps(printable, indent=2)
                    )
            else:
                materialized = materialize_user_message(
                    rec, args.task, args.input, args.data_dir
                )
                output = run_model(
                    model, tokenizer, materialized, args.max_new_tokens
                )
            scored.append(ev.score_record(output, rec, task_mode=args.task))
            if not args.dry_run and (index + 1) % 100 == 0:
                print(f"[{split}] {index + 1}/{len(ids)}", flush=True)

        aggregate = ev.aggregate(scored)
        aggregate.update({
            "schema_version": v6_format.SCHEMA_VERSION,
            "task": args.task,
            "input_mode": args.input,
            "sample_seed": args.seed,
            "sample_requested": args.sample,
            "ids": ids,
            "adapter": adapter,
            "base_model": args.base_model,
        })
        aggregate_path = os.path.join(args.output_dir, f"results_{tag}_{split}.json")
        records_path = os.path.join(args.output_dir, f"records_{tag}_{split}.jsonl")
        ev.save_results(aggregate, aggregate_path, scored, records_path)
        print(
            f"[{split}] n={aggregate['n']} parse={_rate(aggregate['parse_rate'])} "
            f"correct={_rate(aggregate['correct_net_match_rate'])} "
            f"student={_rate(aggregate['student_net_match_rate'])} "
            f"both={_rate(aggregate['both_nets_match_rate'])} "
            f"label={_rate(aggregate['label_accuracy'])} "
            f"derived={_rate(aggregate['derived_label_accuracy'])}"
        )
        print(f"  -> {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
