"""eval_val.py — score the tuned adapters on the VAL split for Day-4 error analysis.

test/ood are FROZEN (scored only for the headline base-vs-tuned number). ALL iteration
happens here, on val. This runs the tuned model over val, writes the per-record rows
(records_tuned_val.jsonl) you read to diagnose failures, and prints a failure breakdown
so you can see the top (true_label, failure) modes immediately.

Assumes (already in $HOME from training): ~/lora_adapters, ~/transform_diagnosis_data,
~/slm_eval. Run via sbatch (needs the GPU). ~2400 val records ≈ 40-50 min on an L40S.
"""

import os, sys, json, copy, collections
from PIL import Image

HOME = os.path.expanduser("~")
sys.path.insert(0, HOME)
from slm_eval import eval as ev

DATA_DIR = os.path.join(HOME, "transform_diagnosis_data")
ADAPTERS = os.path.join(HOME, "lora_adapters")

from unsloth import FastVisionModel
model, tokenizer = FastVisionModel.from_pretrained(ADAPTERS, load_in_4bit=True)
FastVisionModel.for_inference(model)
print("loaded fine-tuned adapters from", ADAPTERS, flush=True)


def load_chat(split):
    with open(os.path.join(DATA_DIR, f"{split}_chat.jsonl")) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_raw(split):
    by_id = {}
    with open(os.path.join(DATA_DIR, f"{split}.jsonl")) as f:
        for l in f:
            if l.strip():
                r = json.loads(l)
                by_id[r["id"]] = r
    return by_id


def decode_user(row):
    msg = copy.deepcopy(row["messages"][0])
    for part in msg["content"]:
        if part.get("type") == "image" and isinstance(part.get("image"), str):
            part["image"] = Image.open(os.path.join(DATA_DIR, part["image"])).convert("RGB")
    return msg


def run_model(user_msg, image, max_new_tokens=256):
    it = tokenizer.apply_chat_template([user_msg], add_generation_prompt=True)
    inp = tokenizer(image, it, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


chat_rows = load_chat("val")
raw_by_id = load_raw("val")
scored = []
for i, row in enumerate(chat_rows):
    um = decode_user(row)
    image = next(p["image"] for p in um["content"] if p.get("type") == "image")
    scored.append(ev.score_record(run_model(um, image), raw_by_id[row["id"]]))
    if i % 200 == 0:
        print(f"  [val] {i}/{len(chat_rows)}", flush=True)
        if i:
            ev.save_results(ev.aggregate(scored), "results_tuned_val.json",
                            scored, "records_tuned_val.jsonl")

agg = ev.aggregate(scored)
ev.save_results(agg, "results_tuned_val.json", scored, "records_tuned_val.jsonl")

print(f"\n[val] n={agg['n']} label_acc={agg['label_accuracy']:.3f} "
      f"balanced_acc={agg['balanced_accuracy']:.3f} parse={agg['parse_rate']:.3f} "
      f"transform={agg['transform_match_rate']:.3f} hint={agg['hint_match_rate']:.3f}", flush=True)

# Immediate error analysis: top (true_label, failure kind) modes.
fails = collections.Counter()
for r in scored:
    if r["failure_reason"]:
        kind = r["failure_reason"].split(":")[0]      # collapse wrong_label:X->Y
        fails[(r["true_label"], kind)] += 1
print("\n== Top val failure modes (count, true_label, kind) ==")
for (lab, kind), c in fails.most_common(15):
    print(f"  {c:4d}  {lab:34s} {kind}")

# A few concrete transform_match failures to read (the ~6% mystery on test).
print("\n== Sample transform_mismatch cases (true vs model output) ==")
shown = 0
for r in scored:
    if r["parse_ok"] and r["label_ok"] and not r["transform_ok"]:
        print(f"  id={r['id']} label={r['true_label']}")
        print(f"    model: {r['raw_model_output'][:200]}")
        shown += 1
        if shown >= 5:
            break
print("\nDONE — read records_tuned_val.jsonl for full per-record analysis.", flush=True)
