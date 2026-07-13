"""eval_tuned.py — score the SAVED fine-tuned adapters on the IMAGE inputs (the tuned-4B +
image cell of the 2x2 {frontier, tuned-4B} x {image, coords} table), headless.

Originally a hardcoded recovery script (full test + ood, no CLI). It now has an argparse CLI
so the image cell can be run SAMPLED and cap-safe: the full test+ood is ~4.4k generations
(~4.5h on an L40S), which blows past the 4h job cap. With ``--sample`` it scores a comparable
N to the coordinate cells, and — crucially — it reuses ``eval_tuned_coords.select_ids`` with
the SAME default seed, so ``--sample N --seed S`` picks the IDENTICAL record ids that
``eval_tuned_coords.py`` picks for the same N/S. That makes tuned+image and tuned+coords a
PAIRED comparison over the exact same records.

Defaults preserve the original behavior: no args -> FULL test + ood, tag ``tuned`` ->
results_tuned_{test,ood}.json + records_tuned_{test,ood}.jsonl, then the base-vs-tuned table,
confusion matrices, and per-label recall (base metrics saved during the training session).
When sampled, the base column is re-aggregated on the SAME sampled ids (from the saved
records_base_{split}.jsonl) so the delta stays apples-to-apples.

Assumes these exist in $HOME (written during the training session):
  ~/lora_adapters/            the trained LoRA adapters (model.save_pretrained)  (--adapters)
  ~/transform_diagnosis_data/ the dataset (jsonl + renders)                      (--data-dir)
  ~/slm_eval/                 the scoring harness package
  ~/results_base_*.json       baseline metrics (optional; the comparison table)
  ~/records_base_*.jsonl      baseline per-record rows (optional; sampled comparison)

Examples (on the cluster):
  python eval_tuned.py                                  # FULL test+ood (original behavior)
  python eval_tuned.py --sample 500                     # 500/split, SAME ids as the coords cell
  python eval_tuned.py --limit 8                        # cheap smoke test
  python eval_tuned.py --dry-run --sample 4 --splits test  # no model/no image decode; plumbing
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys

HOME = os.path.expanduser("~")

# Import resolution. This script's OWN dir first so the sibling ``eval_tuned_coords`` import
# resolves from any cwd (repo/model locally, $HOME on the cluster); then the harness, preferring
# the cluster package name (``slm_eval``, exactly as before) and falling back to the in-repo
# ``transform_diagnosis`` so the non-GPU logic imports and smoke-tests locally.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), ".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from slm_eval import eval as ev
except ModuleNotFoundError:
    from transform_diagnosis import eval as ev

# Reuse the tuned-coords cell's id selection, raw loader, and seed/dir defaults so the image
# and coords cells sample the SAME record ids under the same seed (the whole point of the
# paired comparison). Reusing the exact functions — not re-implementing — is what guarantees it.
from eval_tuned_coords import (  # noqa: E402  (after sys.path setup)
    DEFAULT_ADAPTERS,
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    load_raw,
    select_ids,
)


def load_chat(data_dir, split):
    """Chat rows (image + instruction messages) for a split, keyed by id."""
    with open(os.path.join(data_dir, f"{split}_chat.jsonl")) as f:
        return {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}


# --- Model (loaded lazily so the module imports without unsloth/torch for smoke tests) -----
def load_model(adapters):
    """Load the fine-tuned model from the saved adapters (unsloth reloads base + LoRA)."""
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(adapters, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    print("loaded fine-tuned adapters from", adapters, flush=True)
    return model, tokenizer


def decode_user(row, data_dir):
    """Materialize the user message, swapping the image PATH for a decoded PIL image."""
    from PIL import Image
    msg = copy.deepcopy(row["messages"][0])
    for part in msg["content"]:
        if part.get("type") == "image" and isinstance(part.get("image"), str):
            part["image"] = Image.open(os.path.join(data_dir, part["image"])).convert("RGB")
    return msg


def run_model(model, tokenizer, user_msg, image, max_new_tokens=512):
    it = tokenizer.apply_chat_template([user_msg], add_generation_prompt=True)
    inp = tokenizer(image, it, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_eval(tag, ids, raw_by_id, predict):
    """Score ``ids`` with ``predict(id) -> raw_output`` and write records + aggregate.

    Flushes partial results every 200 records (insurance against a wall-clock timeout)."""
    scored = []
    for i, rid in enumerate(ids):
        scored.append(ev.score_record(predict(rid), raw_by_id[rid]))
        if i % 200 == 0:
            print(f"  [{tag}] {i}/{len(ids)}", flush=True)
            if i:
                ev.save_results(ev.aggregate(scored), f"results_{tag}.json",
                                scored, f"records_{tag}.jsonl")
    agg = ev.aggregate(scored)
    ev.save_results(agg, f"results_{tag}.json", scored, f"records_{tag}.jsonl")
    print(f"[{tag}] n={agg['n']} label_acc={agg['label_accuracy']:.3f} "
          f"balanced_acc={agg['balanced_accuracy']:.3f} parse={agg['parse_rate']:.3f} "
          f"transform={agg['transform_match_rate']:.3f} hint={agg['hint_match_rate']:.3f}",
          flush=True)
    return agg


# --- Base-vs-tuned comparison (baseline metrics were saved during the training session) ----
def _load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return None


def base_agg_for_table(split, ids, sampling):
    """Base metrics to sit beside the tuned column in the table.

    Full run: the saved full aggregate (results_base_<split>.json) — the original behavior.
    Sampled run: re-aggregate the saved base PER-RECORD rows (records_base_<split>.jsonl) on
    the SAME sampled ids so the base-vs-tuned delta is apples-to-apples; fall back to the full
    aggregate (the caller flags the mismatch) if those rows aren't present locally.
    """
    if sampling:
        path = f"records_base_{split}.jsonl"
        if os.path.exists(path):
            keep = set(ids)
            with open(path) as f:
                rows = [json.loads(l) for l in f if l.strip()]
            rows = [r for r in rows if r.get("id") in keep]
            if rows:
                return ev.aggregate(rows)
    return _load_json(f"results_base_{split}.json")


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Tuned-4B IMAGE eval (2x2 cell), sampleable + cap-safe.")
    ap.add_argument("--splits", default="test,ood",
                    help="comma-separated splits (default: test,ood)")
    ap.add_argument("--sample", type=int, default=0,
                    help="deterministically sample N ids per split (0 = FULL; preserves the "
                         "original behavior); with --seed reproduces eval_tuned_coords.py's ids")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"sample seed (default {DEFAULT_SEED}, matches eval_tuned_coords.py so "
                         f"image & coords cells score the SAME ids)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per split after sampling (0 = no cap); cheap smoke test")
    ap.add_argument("--max-new-tokens", type=int, default=512, dest="max_new_tokens",
                    help="generation length (default: 512). Sized for chain-of-thought "
                         "adapters (e.g. v3cot), whose output is a reasoning trace + the final "
                         "JSON (~250-300 tokens); JSON-only adapters stop at EOS well before "
                         "either cap, so their outputs/scores are unchanged.")
    ap.add_argument("--adapters", default=DEFAULT_ADAPTERS,
                    help=f"LoRA adapters dir (default: {DEFAULT_ADAPTERS})")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dataset dir with <split>_chat.jsonl (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--tag", default="tuned",
                    help="output tag: results_<tag>_<split>.json (default: tuned)")
    ap.add_argument("--dry-run", action="store_true",
                    help="select ids + score empty outputs (NO model load, NO image decode); "
                         "writes <tag>_dryrun_* files to verify the sampling/scoring pipeline")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]
    sampling = bool(args.sample)

    if args.dry_run:
        model = tokenizer = None
        tag_base = f"{args.tag}_dryrun"
        print(f"Tuned-4B image eval: DRY RUN (no model load, no image decode)"
              f"{f'  sample={args.sample}' if args.sample else ''}"
              f"{f'  limit={args.limit}' if args.limit else ''}\n", flush=True)
    else:
        model, tokenizer = load_model(args.adapters)
        tag_base = args.tag
        print(f"Tuned-4B image eval: adapters={args.adapters}  splits={splits}  "
              f"{'sample=' + str(args.sample) if sampling else 'FULL'}\n", flush=True)

    results = {}
    for split in splits:
        raw = load_raw(args.data_dir, split)
        ids = select_ids(raw, args.sample, args.seed, args.limit)
        chat_by_id = load_chat(args.data_dir, split)
        missing = [i for i in ids if i not in chat_by_id]
        if missing:
            raise KeyError(f"[{split}] {len(missing)} sampled ids missing from "
                           f"{split}_chat.jsonl (e.g. {missing[:5]})")

        if args.dry_run:
            predict = lambda rid: ""       # no model, no image decode (dataset-light, GPU-free)
        else:
            def predict(rid, _chat=chat_by_id):
                um = decode_user(_chat[rid], args.data_dir)
                image = next(p["image"] for p in um["content"] if p.get("type") == "image")
                return run_model(model, tokenizer, um, image, args.max_new_tokens)

        print(f"[{split}] scoring {len(ids)} records "
              f"({'dry-run' if args.dry_run else 'image'}) ...", flush=True)
        results[split] = (ids, run_eval(f"{tag_base}_{split}", ids, raw, predict))

    # --- Tables: base vs tuned, confusion, per-label recall (guarded for sampled subsets) ---
    print("\n================ RESULTS ================", flush=True)
    for split in splits:
        ids, tuned = results[split]
        base = base_agg_for_table(split, ids, sampling)
        note = "" if not sampling else (
            "  [base re-aggregated on the same sampled ids]"
            if os.path.exists(f"records_base_{split}.jsonl")
            else "  [base = FULL split; tuned = sampled — not directly comparable]")
        if base:
            print(f"\n== {split.upper()} (base vs tuned){note} ==")
            print(ev.format_table(base, tuned))
        else:
            print(f"\n== {split.upper()} (tuned only — no results_base_{split}.json found) ==")
        print(f"\n== Tuned {split.upper()} confusion (true rows x predicted cols; PF = parse fail) ==")
        print(ev.format_confusion(tuned))
        print(f"\n== Per-label recall (tuned {split}) ==")
        for lab, rec in tuned["per_label_recall"].items():
            print(f"  {lab:34s} {'--' if rec is None else f'{rec:.3f}'}")

    if args.dry_run:
        print("\nDRY RUN complete — sampling + scoring pipeline OK; no model loaded.")
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
