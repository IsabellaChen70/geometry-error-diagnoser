"""Sequential transform-first QLoRA training for the v6 canonical-net tasks.

Heavy GPU dependencies are imported only for a real run.  ``--dry-run`` checks
stage files, image resolution, adapter/checkpoint paths, and schedule math on a
laptop/login node without loading a model.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
from typing import Dict, Iterable, List, Sequence

HOME = os.path.expanduser("~")
DEFAULT_DATA_DIR = os.path.join(HOME, "transform_diagnosis_data_v6")
DEFAULT_BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
STAGES = ("correct", "student", "both", "full")
MODALITIES = ("image", "image_coords")
_MODALITY_OUT = {
    "image": os.path.join(HOME, "lora_adapters_v6_image"),
    "image_coords": os.path.join(HOME, "lora_adapters_v6_coords"),
}


def _stage_file(data_dir: str, split: str, modality: str, stage: str) -> str:
    return os.path.join(data_dir, f"{split}_v6_{modality}_{stage}_chat.jsonl")


def _default_final_out(modality: str, stage: str) -> str:
    root = _MODALITY_OUT[modality]
    return root if stage == "full" else f"{root}_{stage}"


def _default_checkpoint_dir(modality: str, stage: str) -> str:
    suffix = "coords" if modality == "image_coords" else "image"
    return os.path.join(HOME, f"outputs_v6_{suffix}_{stage}")


def _read_rows(path: str, limit: int = 0) -> List[dict]:
    rows: List[dict] = []
    with open(path) as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


def _resolve_image_path(data_dir: str, path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(data_dir, path)


def _image_paths(row: dict, data_dir: str) -> Iterable[str]:
    for message in row["messages"]:
        for part in message["content"]:
            if part.get("type") == "image" and isinstance(part.get("image"), str):
                yield _resolve_image_path(data_dir, part["image"])


def _materialize_rows(rows: Sequence[dict], data_dir: str) -> List[dict]:
    """Decode chat image paths to independent RGB PIL images."""
    from PIL import Image

    output: List[dict] = []
    for source in rows:
        row = copy.deepcopy(source)
        for message in row["messages"]:
            for part in message["content"]:
                if part.get("type") == "image" and isinstance(part.get("image"), str):
                    path = _resolve_image_path(data_dir, part["image"])
                    with Image.open(path) as image:
                        part["image"] = image.convert("RGB").copy()
        output.append({"messages": row["messages"]})
    return output


def _stage_index(stage: str) -> int:
    return STAGES.index(stage)


def _rehearsal_paths(data_dir: str, modality: str, stage: str) -> List[str]:
    return [
        _stage_file(data_dir, "train", modality, earlier)
        for earlier in STAGES[: _stage_index(stage)]
    ]


def assemble_rows(
    train_path: str,
    data_dir: str,
    modality: str,
    stage: str,
    *,
    limit: int,
    rehearsal_ratio: float,
    seed: int,
) -> tuple[List[dict], Dict[str, int]]:
    """Load the current stage plus a deterministic sample of earlier-task rehearsal."""
    main = _read_rows(train_path, limit=limit)
    counts = {stage: len(main)}
    if not main:
        return [], counts
    previous_paths = _rehearsal_paths(data_dir, modality, stage)
    if rehearsal_ratio and previous_paths:
        requested = round(len(main) * rehearsal_ratio / (1.0 - rehearsal_ratio))
        per_file = math.ceil(requested / len(previous_paths))
        rehearsal: List[dict] = []
        for index, path in enumerate(previous_paths):
            if not os.path.isfile(path):
                raise FileNotFoundError(f"rehearsal stage file missing: {path}")
            candidates = _read_rows(path)
            random.Random(seed ^ (index + 1) * 0x9E3779B1).shuffle(candidates)
            chosen = candidates[:per_file]
            rehearsal.extend(chosen)
        rehearsal = rehearsal[:requested]
        for earlier in STAGES[: _stage_index(stage)]:
            counts[earlier] = sum(1 for row in rehearsal if row.get("task") == earlier)
        rows = main + rehearsal
    else:
        rows = main
    random.Random(seed).shuffle(rows)
    return rows, counts


def _adapter_base(adapter_dir: str) -> str | None:
    config_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.isfile(config_path):
        return None
    with open(config_path) as handle:
        return json.load(handle).get("base_model_name_or_path")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, default="full")
    parser.add_argument("--modality", choices=MODALITIES, default="image")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--val-file", default=None)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL,
                        help="verified base model ID/path for a fresh LoRA")
    parser.add_argument("--init-adapter", default=None,
                        help="prior-stage adapter to continue; loaded directly and never PEFT-wrapped again")
    parser.add_argument("--out", default=None)
    parser.add_argument("--output-dir", default=None,
                        help="Trainer checkpoint directory")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--val-sample", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=300)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--rehearsal-ratio", type=float, default=None,
                        help="fraction of assembled train rows from earlier tasks (default: 0.15 after stage 1)")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true",
                        help="resume the latest checkpoint in --output-dir")
    parser.add_argument("--allow-existing-output", action="store_true",
                        help="allow replacing an existing final adapter directory")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve(args: argparse.Namespace) -> dict:
    train_path = args.train_file or _stage_file(
        args.data_dir, "train", args.modality, args.stage
    )
    val_path = args.val_file or _stage_file(
        args.data_dir, "val", args.modality, args.stage
    )
    out = args.out or _default_final_out(args.modality, args.stage)
    output_dir = args.output_dir or _default_checkpoint_dir(args.modality, args.stage)
    rehearsal_ratio = (
        args.rehearsal_ratio
        if args.rehearsal_ratio is not None
        else (0.0 if args.stage == "correct" else 0.15)
    )
    if not 0.0 <= rehearsal_ratio < 1.0:
        raise SystemExit("--rehearsal-ratio must be in [0,1)")
    return {
        "train_path": train_path,
        "val_path": val_path,
        "out": out,
        "output_dir": output_dir,
        "rehearsal_ratio": rehearsal_ratio,
    }


def _preflight(args: argparse.Namespace, resolved: dict) -> tuple[List[dict], List[dict], dict]:
    for key in ("train_path", "val_path"):
        if not os.path.isfile(resolved[key]):
            raise SystemExit(f"{key.replace('_', ' ')} not found: {resolved[key]}")
    if args.init_adapter and not os.path.isdir(args.init_adapter):
        raise SystemExit(f"--init-adapter directory not found: {args.init_adapter}")
    if (
        os.path.isdir(resolved["out"])
        and os.listdir(resolved["out"])
        and not args.allow_existing_output
        and not args.resume
    ):
        raise SystemExit(
            f"refusing nonempty final adapter directory {resolved['out']}; choose --out or "
            "pass --allow-existing-output"
        )
    train_rows, mix_counts = assemble_rows(
        resolved["train_path"],
        args.data_dir,
        args.modality,
        args.stage,
        limit=args.limit,
        rehearsal_ratio=resolved["rehearsal_ratio"],
        seed=args.seed,
    )
    val_rows = (
        _read_rows(resolved["val_path"], limit=args.val_sample)
        if args.val_sample
        else []
    )
    if not train_rows:
        raise SystemExit("no training rows")
    for row in [*train_rows[:2], *val_rows[:2]]:
        paths = list(_image_paths(row, args.data_dir))
        if len(paths) != 1:
            raise SystemExit(f"expected exactly one image content part, found {len(paths)}")
        if not os.path.isfile(paths[0]):
            raise SystemExit(f"chat image does not resolve: {paths[0]}")
    return train_rows, val_rows, mix_counts


def dry_run(args: argparse.Namespace, resolved: dict) -> int:
    train_rows, val_rows, mix_counts = _preflight(args, resolved)
    effective_batch = args.batch_size * args.grad_accum
    steps_per_epoch = math.ceil(len(train_rows) / effective_batch)
    print("=== train_transform.py DRY RUN (no GPU/model load) ===")
    print(f"stage/modality   : {args.stage} / {args.modality}")
    print(f"base model       : {args.base_model}")
    print(f"initial adapter  : {args.init_adapter or '(fresh LoRA from base)'}")
    if args.init_adapter:
        print(f"adapter base     : {_adapter_base(args.init_adapter) or '(config not locally readable)'}")
    print(f"train file       : {resolved['train_path']}")
    print(f"val file         : {resolved['val_path']}")
    print(f"train composition: {mix_counts} (rehearsal={resolved['rehearsal_ratio']:.3f})")
    print(f"rows             : train={len(train_rows)} val={len(val_rows)}")
    print(f"final adapters   : {resolved['out']}")
    print(f"checkpoints      : {resolved['output_dir']} (resume={args.resume})")
    print(f"schedule         : cosine lr={args.lr} warmup={args.warmup_ratio} epochs={args.epochs}")
    print(f"effective batch  : {effective_batch}; ~{steps_per_epoch} steps/epoch")
    print("DRY RUN OK")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    resolved = _resolve(args)
    if args.dry_run:
        return dry_run(args, resolved)

    raw_train, raw_val, mix_counts = _preflight(args, resolved)

    # Heavy imports remain below the dry-run path.
    from unsloth import FastVisionModel, is_bf16_supported

    model_source = args.init_adapter or args.base_model
    model, tokenizer = FastVisionModel.from_pretrained(
        model_source,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)
    if args.init_adapter:
        # FastVisionModel.from_pretrained(adapter_dir) restores the existing PEFT
        # adapter.  Re-running get_peft_model here would double-wrap it.
        if not getattr(model, "peft_config", None):
            raise RuntimeError(
                "--init-adapter did not load as a PEFT model; refusing to double-wrap or "
                "silently start a fresh adapter"
            )
        print(f"continuing existing adapter without PEFT re-wrapping: {args.init_adapter}")
    else:
        model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            bias="none",
            random_state=args.seed,
            use_rslora=False,
        )

    print(f"decoding {len(raw_train)} train and {len(raw_val)} val images ...", flush=True)
    train_dataset = _materialize_rows(raw_train, args.data_dir)
    eval_dataset = _materialize_rows(raw_val, args.data_dir) if raw_val else None

    try:
        from unsloth import UnslothVisionDataCollator
    except Exception:
        from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTConfig, SFTTrainer

    config = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=args.seed,
        data_seed=args.seed,
        output_dir=resolved["output_dir"],
        report_to="none",
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_seq_length=args.max_seq_length,
        save_total_limit=args.save_total_limit,
        save_strategy="steps",
        save_steps=args.eval_steps,
    )
    callbacks = []
    if eval_dataset:
        config.update(
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        if args.patience:
            try:
                from transformers import EarlyStoppingCallback
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
            except Exception as exc:
                print(f"early stopping unavailable; continuing without it: {exc}", flush=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        args=SFTConfig(**config),
    )
    print(
        f"training v6 {args.modality}/{args.stage}: rows={len(train_dataset)} "
        f"composition={mix_counts}",
        flush=True,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    model.save_pretrained(resolved["out"])
    tokenizer.save_pretrained(resolved["out"])
    print(f"saved adapters to {resolved['out']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
