"""eval_tuned_coords.py — the tuned 4B on COORDINATES-AS-TEXT (the 4B+coords cell).

Runs the SAME saved QLoRA checkpoint as eval_tuned.py, but feeds it the three polygons as
integer (x, y) vertex lists (``chat_format.coords_prompt`` — no image) instead of the
rendered grid, and asks for the same JSON diagnosis. Scored by the same fixed harness
(``slm_eval.eval`` on the cluster / ``transform_diagnosis.eval`` locally).

This is the fourth cell of the 2x2 {frontier, tuned-4B} x {image, coords} diagnostic table:

    * frontier + coords : probe_coords.py            (GPT-5.3-codex, coordinates-as-text)
    * frontier + image  : eval_frontier.py           (Claude Opus, rendered images)
    * tuned-4B + image  : eval_tuned.py              (this checkpoint, rendered images)
    * tuned-4B + coords : eval_tuned_coords.py (HERE) (this checkpoint, coordinates-as-text)

The coordinate prompt is the SAME shared ``coords_prompt`` the frontier probe uses, so this
cell is directly comparable to frontier+coords; it is the same checkpoint eval_tuned.py runs,
so it is directly comparable to tuned+image. Every cell scores through the same fixed metrics.

Run via sbatch (needs the GPU). Assumes (from the training session, in $HOME):
  ~/lora_adapters/             the trained LoRA adapters  (override with --adapters)
  ~/transform_diagnosis_data/  the dataset jsonl          (override with --data-dir)
  ~/slm_eval/                  the scoring harness package

Examples (on the cluster):
  python eval_tuned_coords.py                                  # test+ood, full
  python eval_tuned_coords.py --limit 8                        # cheap smoke test
  python eval_tuned_coords.py --dry-run --limit 8              # no model load; plumbing only
  python eval_tuned_coords.py --splits val --sample 30 --seed 20260709   # exact probe ids
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HOME = os.path.expanduser("~")

# Harness + shared prompt. Prefer the cluster package name (``slm_eval``, exactly as
# eval_tuned.py imports it); fall back to the in-repo ``transform_diagnosis`` so the
# non-GPU logic (prompt build, scoring, arg parsing) imports and smoke-tests locally.
for _cand in (".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from slm_eval import eval as ev, chat_format as cf
except ModuleNotFoundError:
    from transform_diagnosis import eval as ev, chat_format as cf

DEFAULT_DATA_DIR = os.path.join(HOME, "transform_diagnosis_data")
DEFAULT_ADAPTERS = os.path.join(HOME, "lora_adapters")
DEFAULT_SEED = 20260709          # matches probe_coords.py's sample seed for head-to-heads


def load_raw(data_dir, split):
    """Full oracle records for a split, keyed by id (in file order)."""
    by_id = {}
    with open(os.path.join(data_dir, f"{split}.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r
    return by_id


def select_ids(by_id, sample, seed, limit):
    """Which ids to score. ``sample>0`` -> deterministic seeded sample (identical selection
    to probe_coords.py when seed/N match); otherwise all ids in file order. ``limit>0``
    then truncates (cheap smoke tests)."""
    if sample:
        ids = sorted(random.Random(seed).sample(sorted(by_id), min(sample, len(by_id))))
    else:
        ids = list(by_id)
    if limit:
        ids = ids[:limit]
    return ids


# --- Model (loaded lazily so the module imports without unsloth/torch for smoke tests) -----
def load_model(adapters):
    """Load the fine-tuned model from the saved adapters — same call as eval_tuned.py."""
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(adapters, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    print("loaded fine-tuned adapters from", adapters, flush=True)
    return model, tokenizer


def run_model(model, tokenizer, text, max_new_tokens=512):
    """Greedy-decode the model on a TEXT-ONLY prompt (no image). Mirrors eval_tuned.py's
    generation, but calls the processor with text only (images default to None)."""
    msg = {"role": "user", "content": [{"type": "text", "text": text}]}
    it = tokenizer.apply_chat_template([msg], add_generation_prompt=True)
    inp = tokenizer(text=it, add_special_tokens=False, return_tensors="pt").to("cuda")
    out = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
    return tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def run_eval(tag, ids, raw_by_id, generate, max_new_tokens):
    """Score ``ids`` with ``generate(text) -> raw_output`` and write records + aggregate.

    Flushes partial results every 200 records (insurance against a wall-clock timeout),
    exactly like eval_tuned.py."""
    scored = []
    for i, rid in enumerate(ids):
        rec = raw_by_id[rid]
        out = generate(cf.coords_prompt(rec))
        scored.append(ev.score_record(out, rec))
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


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Tuned-4B coordinates-as-text eval (2x2 cell).")
    ap.add_argument("--splits", default="test,ood",
                    help="comma-separated splits (default: test,ood)")
    ap.add_argument("--sample", type=int, default=0,
                    help="deterministically sample N ids per split (0 = all); "
                         "use with --seed to reproduce probe_coords.py's exact ids")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"sample seed (default: {DEFAULT_SEED}, matches probe_coords.py)")
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
                    help=f"dataset dir with <split>.jsonl (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--tag", default="tuned_coords",
                    help="output tag: results_<tag>_<split>.json (default: tuned_coords)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the coord prompts + score empty outputs (NO model load); "
                         "writes <tag>_dryrun_* files to verify the pipeline")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]

    if args.dry_run:
        generate = lambda text: ""            # no model; honest empty output -> parse_fail
        tag_base = f"{args.tag}_dryrun"
        print(f"Tuned-4B coords eval: DRY RUN (no model load){f'  limit={args.limit}' if args.limit else ''}\n",
              flush=True)
    else:
        model, tokenizer = load_model(args.adapters)
        generate = lambda text: run_model(model, tokenizer, text, args.max_new_tokens)
        tag_base = args.tag
        print(f"Tuned-4B coords eval: adapters={args.adapters}  splits={splits}\n", flush=True)

    for split in splits:
        raw = load_raw(args.data_dir, split)
        ids = select_ids(raw, args.sample, args.seed, args.limit)
        if args.dry_run:                       # exercise the new prompt path on real records
            for rid in ids:
                cf.coords_prompt(raw[rid])
        print(f"[{split}] scoring {len(ids)} records "
              f"({'dry-run' if args.dry_run else 'coords-as-text'}) ...", flush=True)
        run_eval(f"{tag_base}_{split}", ids, raw, generate, args.max_new_tokens)

    if args.dry_run:
        print("\nDRY RUN complete — prompt + scoring pipeline OK; no model loaded.")
    else:
        print(f"\nSaved results_{tag_base}_{{{','.join(splits)}}}.json + "
              f"records_{tag_base}_{{{','.join(splits)}}}.jsonl", flush=True)


if __name__ == "__main__":
    main()
