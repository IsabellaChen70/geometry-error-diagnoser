"""train_cot.py — chain-of-thought QLoRA fine-tune from the BASE model (v3cot).

Mirrors ``train.py`` (Unsloth + TRL SFT, 4-bit QLoRA, r=16, same memory settings) but:

  * trains on ``train_cot_chat.jsonl`` — the assistant target is a step-by-step reasoning
    trace followed by the SAME final JSON (built by ``make_cot_data.py`` / ``cot.py``);
  * always starts from the BASE model (does not build on v1/v2) and saves the adapters to
    ``~/lora_adapters_v3cot`` (a NEW dir, so v1/v2 stay intact);
  * uses a COSINE LR schedule (train.py used linear);
  * adds VALIDATION monitoring on a small val slice (``eval_strategy="steps"``) with
    EARLY STOPPING (train.py had none), so we can tell when it plateaus;
  * exposes ``--epochs`` (default 2; bump to 3) and other knobs on the CLI;
  * checkpoints every ``--eval-steps`` and supports ``--resume`` so a long run can be split
    across the partition wall-clock cap.

Lazy imports: ``unsloth`` / ``torch`` / ``trl`` are imported INSIDE ``main`` (like
``eval_tuned_coords.py``), so this module byte-compiles and ``--help`` / ``--dry-run`` work
on a machine without a GPU. Run the real thing via sbatch on the GPU.

Examples (on the cluster, from $HOME):
  python train_cot.py                        # 2 epochs, val monitoring + early stopping
  python train_cot.py --epochs 3             # bump epochs
  python train_cot.py --resume               # continue from the latest checkpoint
  python train_cot.py --dry-run              # GPU-free plumbing + step/time estimate
"""

from __future__ import annotations

import argparse
import json
import math
import os

HOME = os.path.expanduser("~")
DEFAULT_DATA_DIR = os.path.join(HOME, "transform_diagnosis_data")
DEFAULT_OUT = os.path.join(HOME, "lora_adapters_v3cot")
DEFAULT_BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"


# --- Data loading (PIL imported lazily so the module imports without Pillow/GPU) ----------
def load_chat_rows(path, data_dir, limit=0):
    """Load a ``*_chat.jsonl`` into trainer rows, decoding each image PATH to a PIL image.

    Identical materialization to ``train.py``: only ``{"messages": ...}`` is kept, and the
    image content part's path string is swapped for a decoded RGB ``PIL.Image``.
    """
    from PIL import Image
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for m in rec["messages"]:
                for part in m["content"]:
                    if part.get("type") == "image" and isinstance(part.get("image"), str):
                        part["image"] = Image.open(
                            os.path.join(data_dir, part["image"])).convert("RGB")
            rows.append({"messages": rec["messages"]})
            if limit and len(rows) >= limit:
                break
    return rows


def pick_val_file(data_dir):
    """Prefer a CoT-format val file (so eval loss matches the training objective); fall back
    to the plain ``val_chat.jsonl`` (JSON-only target — still a valid plateau signal)."""
    for name in ("val_cot_chat.jsonl", "val_chat.jsonl"):
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            return p
    return None


def count_lines(path):
    n = 0
    with open(path) as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def build_arg_parser():
    ap = argparse.ArgumentParser(description="CoT QLoRA fine-tune (v3cot), sbatch-friendly.")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dir with train_cot_chat.jsonl + val files (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--train-file", default=None, dest="train_file",
                    help="explicit train JSONL (default: <data-dir>/train_cot_chat.jsonl); "
                         "e.g. train_v4_cot_chat.jsonl for v4")
    ap.add_argument("--val-file", default=None, dest="val_file",
                    help="explicit val JSONL for eval-loss monitoring (default: auto-pick "
                         "val_cot_chat.jsonl / val_chat.jsonl in --data-dir); e.g. "
                         "val_v4_cot_chat.jsonl for v4")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help=f"where to save the final adapters (default: {DEFAULT_OUT})")
    ap.add_argument("--output-dir", default="outputs_v3cot", dest="output_dir",
                    help="trainer working/checkpoint dir (default: outputs_v3cot); kept for --resume")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL, dest="base_model",
                    help=f"base model id (default: {DEFAULT_BASE_MODEL})")
    # --- training length / schedule ---
    ap.add_argument("--epochs", type=float, default=2.0,
                    help="num_train_epochs (default: 2; try 2-3)")
    ap.add_argument("--lr", type=float, default=2e-4, help="learning rate (default: 2e-4)")
    ap.add_argument("--warmup-ratio", type=float, default=0.03, dest="warmup_ratio",
                    help="warmup fraction for the cosine schedule (default: 0.03)")
    ap.add_argument("--batch-size", type=int, default=2, dest="batch_size",
                    help="per-device train batch size (default: 2, per train.py)")
    ap.add_argument("--grad-accum", type=int, default=4, dest="grad_accum",
                    help="gradient accumulation steps (default: 4, per train.py)")
    ap.add_argument("--max-seq-length", type=int, default=2048, dest="max_seq_length",
                    help="max sequence length (default: 2048; trace+JSON fits comfortably)")
    # --- LoRA ---
    ap.add_argument("--lora-r", type=int, default=16, dest="lora_r", help="LoRA rank (default: 16)")
    ap.add_argument("--lora-alpha", type=int, default=16, dest="lora_alpha",
                    help="LoRA alpha (default: 16)")
    # --- validation monitoring / early stopping ---
    ap.add_argument("--val-sample", type=int, default=200, dest="val_sample",
                    help="rows from the val file to monitor eval_loss on (0 disables; default: 200)")
    ap.add_argument("--eval-steps", type=int, default=300, dest="eval_steps",
                    help="eval + checkpoint interval in steps (default: 300)")
    ap.add_argument("--patience", type=int, default=3,
                    help="early-stopping patience in evals (default: 3; 0 disables early stop)")
    ap.add_argument("--save-total-limit", type=int, default=3, dest="save_total_limit",
                    help="max checkpoints to keep (default: 3)")
    # --- misc ---
    ap.add_argument("--limit", type=int, default=0,
                    help="cap train rows (0 = all); cheap smoke test")
    ap.add_argument("--resume", action="store_true",
                    help="resume from the latest checkpoint in --output-dir (for split runs)")
    ap.add_argument("--dry-run", action="store_true",
                    help="GPU-free: verify files + print the resolved config + step/time estimate")
    return ap


def _resolve_paths(args):
    train_path = args.train_file or os.path.join(args.data_dir, "train_cot_chat.jsonl")
    val_path = None
    if args.val_sample:
        val_path = args.val_file or pick_val_file(args.data_dir)
    return train_path, val_path


def dry_run(args):
    train_path, val_path = _resolve_paths(args)
    eff = args.batch_size * args.grad_accum
    print("=== train_cot.py DRY RUN (no GPU, no model load) ===", flush=True)
    print(f"base model      : {args.base_model}")
    print(f"train file      : {train_path}  {'(exists)' if os.path.exists(train_path) else '(MISSING)'}")
    print(f"val file        : {val_path or '(none — eval disabled)'}"
          f"{'  (exists)' if val_path and os.path.exists(val_path) else ''}")
    print(f"adapters out    : {args.out}")
    print(f"checkpoint dir  : {args.output_dir}  (resume={'on' if args.resume else 'off'})")
    print(f"LoRA            : r={args.lora_r} alpha={args.lora_alpha} (all layers, dropout 0)")
    print(f"schedule        : cosine  lr={args.lr}  warmup_ratio={args.warmup_ratio}  epochs={args.epochs}")
    print(f"batch           : per_device={args.batch_size} x accum={args.grad_accum} = eff {eff}")
    print(f"val monitor     : sample={args.val_sample}  eval/save every {args.eval_steps} steps  "
          f"patience={args.patience}")
    if os.path.exists(train_path):
        n = count_lines(train_path)
        spe = math.ceil(n / eff) if eff else 0
        total = math.ceil(spe * args.epochs)
        print(f"train examples  : {n}")
        print(f"steps/epoch     : ~{spe}   total (~{args.epochs} ep): ~{total}")
    print("DRY RUN OK — plumbing/config validated; run without --dry-run on the GPU.", flush=True)


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if args.dry_run:
        dry_run(args)
        return 0

    train_path, val_path = _resolve_paths(args)
    if not os.path.exists(train_path):
        raise SystemExit(f"train file not found: {train_path} (run make_cot_data.py first)")

    # --- Heavy imports (GPU); lazy so the module imports/byte-compiles without a GPU. ------
    from unsloth import FastVisionModel, is_bf16_supported

    model, tokenizer = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=True,
        use_gradient_checkpointing="unsloth",
    )
    FastVisionModel.for_training(model)
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=True, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.0, bias="none",
        random_state=3407, use_rslora=False,
    )

    print("loading + decoding train images ...", flush=True)
    train_dataset = load_chat_rows(train_path, args.data_dir, limit=args.limit)
    print("train examples:", len(train_dataset), flush=True)

    eval_dataset = None
    if val_path and args.val_sample:
        print(f"loading + decoding up to {args.val_sample} val images from "
              f"{os.path.basename(val_path)} ...", flush=True)
        eval_dataset = load_chat_rows(val_path, args.data_dir, limit=args.val_sample)
        print("val examples:", len(eval_dataset), flush=True)
    else:
        print("no val file / val monitoring disabled — training without eval.", flush=True)

    try:
        from unsloth import UnslothVisionDataCollator
    except Exception:
        from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig

    # Base config mirrors train.py's memory-safe settings; cosine schedule is the change.
    cfg = dict(
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir=args.output_dir,
        report_to="none",
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_seq_length=args.max_seq_length,
        save_total_limit=args.save_total_limit,
    )

    callbacks = []
    if eval_dataset is not None:
        # eval + checkpoint on the same cadence so load_best_model_at_end can pick the best.
        cfg.update(
            per_device_eval_batch_size=args.batch_size,
            eval_strategy="steps",
            eval_steps=args.eval_steps,
            save_strategy="steps",
            save_steps=args.eval_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
        )
        if args.patience:
            try:
                from transformers import EarlyStoppingCallback
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.patience))
            except Exception as e:  # never let a missing callback abort training
                print("EarlyStoppingCallback unavailable, continuing without it:", e, flush=True)
    else:
        # No eval: still checkpoint periodically so a capped run can --resume.
        cfg.update(save_strategy="steps", save_steps=args.eval_steps)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=UnslothVisionDataCollator(model, tokenizer),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
        args=SFTConfig(**cfg),
    )

    print(f"starting training: epochs={args.epochs} cosine lr={args.lr} "
          f"eff_batch={args.batch_size * args.grad_accum} "
          f"eval={'on' if eval_dataset is not None else 'off'} resume={args.resume}", flush=True)
    trainer.train(resume_from_checkpoint=args.resume)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    print("saved adapters to", args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
