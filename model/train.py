"""train.py — headless QLoRA retrain on the NUMBERED-VERTEX renders (v2).

Same config as the original run (Unsloth + TRL SFT, r=16, 1 epoch, lr 2e-4), but trains
on the re-rendered images (numbered vertices + higher resolution). Saves to
~/lora_adapters_v2 so the original ~/lora_adapters (v1) stays intact for comparison.

Run via sbatch (GPU). Images are decoded from disk into RAM up front, matching the
original notebook — the bigger renders need more host memory, so request --mem=128G.
"""

import os, json
from PIL import Image

HOME = os.path.expanduser("~")
DATA = os.path.join(HOME, "transform_diagnosis_data")
OUT = os.path.join(HOME, "lora_adapters_v2")

from unsloth import FastVisionModel, is_bf16_supported

model, tokenizer = FastVisionModel.from_pretrained(
    "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit",
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
)

FastVisionModel.for_training(model)
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True, finetune_language_layers=True,
    finetune_attention_modules=True, finetune_mlp_modules=True,
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none",
    random_state=3407, use_rslora=False,
)


def load_train():
    rows = []
    with open(os.path.join(DATA, "train_chat.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            for m in rec["messages"]:
                for part in m["content"]:
                    if part.get("type") == "image" and isinstance(part.get("image"), str):
                        part["image"] = Image.open(
                            os.path.join(DATA, part["image"])).convert("RGB")
            rows.append({"messages": rec["messages"]})
    return rows


print("loading + decoding train images ...", flush=True)
train_dataset = load_train()
print("train examples:", len(train_dataset), flush=True)

try:
    from unsloth import UnslothVisionDataCollator
except Exception:
    from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    data_collator=UnslothVisionDataCollator(model, tokenizer),
    train_dataset=train_dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=1,
        learning_rate=2e-4,
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs_v2",
        report_to="none",
        fp16=not is_bf16_supported(),
        bf16=is_bf16_supported(),
        remove_unused_columns=False,
        dataset_text_field="",
        dataset_kwargs={"skip_prepare_dataset": True},
        dataset_num_proc=1,
        max_seq_length=2048,
    ),
)

trainer.train()
model.save_pretrained(OUT)
tokenizer.save_pretrained(OUT)
print("saved adapters to", OUT, flush=True)
