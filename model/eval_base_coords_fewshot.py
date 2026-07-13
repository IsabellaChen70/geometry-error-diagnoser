"""eval_base_coords_fewshot.py — the BASE (un-fine-tuned) 4B on COORDINATES-AS-TEXT with
K in-context examples: the decisive disambiguating experiment.

The 2x2 {frontier, tuned-4B} x {image, coords} table produced a puzzle: the image-fine-tuned
4B, fed coordinates-as-text, COLLAPSES (label ~0.24, transform ~0.00), while a frontier model
on the same coordinate text scores ~0.80 transform. The suspected confound is that the 4B was
fine-tuned on IMAGE inputs only, so coordinate TEXT is out-of-distribution for it — the
collapse may be a distribution-shift artifact, not a reasoning-capacity ceiling.

This script isolates reasoning capacity from that image-only fine-tune confound by asking a
different question:

    "Can the BASE (un-fine-tuned) 4B reason about transformation errors from
     coordinates-as-text, given a few in-context examples?"

It loads the SAME base checkpoint ``model/train.py`` fine-tunes from (NOT the LoRA adapters),
feeds it a K-shot coordinate prompt (K worked ``coords_prompt`` -> gold-JSON demonstrations,
then the query's ``coords_prompt``), and scores with the SAME fixed harness (``slm_eval.eval``
on the cluster / ``transform_diagnosis.eval`` locally) on the SAME sampled test/ood ids as the
other coordinate cells (shared seed).

Reading the result:
  * base+coords+few-shot reasons well (transform high, near frontier) -> the tuned-4B's
    coordinate collapse is the IMAGE-ONLY FINE-TUNE confound (text is OOD for it), NOT a
    capacity limit of the 4B family. The clean conclusion for the writeup.
  * base+coords+few-shot ALSO collapses -> the 4B family simply cannot do this from
    coordinates; the gap to frontier is genuine reasoning capacity, and the fine-tune is not
    the culprit.

METHODOLOGY CAVEAT (state this explicitly in the writeup):
  The frontier coordinate probe (``probe_coords.py`` / ``eval_frontier.py --input coords``)
  was ZERO-shot against a dedicated reasoning model. This cell deliberately hands the base 4B
  a K-shot crutch (default K=3). That is NOT an apples-to-apples "who is better" comparison —
  it is a deliberate handicap-leveling: the question here is capacity ("CAN the base 4B do it
  at all, given help?"), not a head-to-head score. The K-shot advantage is intentional and is
  stated so the number is not over-read.

NO LEAKAGE: the few-shot demonstrations are drawn ONLY from the ``train`` split (never
test/ood), selected deterministically (seeded) and spread across diverse error labels. Train
ids are disjoint from test/ood ids by construction, and the script asserts a few-shot id never
coincides with an evaluated id.

Run via sbatch (needs the GPU for the base model). Assumes (in $HOME):
  ~/transform_diagnosis_data/  the dataset jsonl (train.jsonl for the demos)  (--data-dir)
  ~/slm_eval/                  the scoring harness package
The base model is downloaded from the Hub on first use (one-time).

Examples (on the cluster):
  python eval_base_coords_fewshot.py                                 # test+ood, 500 sampled, K=3
  python eval_base_coords_fewshot.py --shots 5 --sample 500          # more in-context examples
  python eval_base_coords_fewshot.py --limit 8                       # cheap smoke test
  python eval_base_coords_fewshot.py --dry-run --limit 4 --splits test  # no model; plumbing only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

HOME = os.path.expanduser("~")

# Import resolution. Put this script's OWN dir first so the sibling ``eval_tuned_coords``
# import resolves from any cwd (repo/model locally, $HOME on the cluster); then the harness,
# preferring the cluster package name (``slm_eval``) and falling back to the in-repo
# ``transform_diagnosis`` so the non-GPU logic imports and smoke-tests locally (same fallback
# as eval_tuned_coords.py / eval_frontier.py).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), ".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from slm_eval import eval as ev, chat_format as cf
except ModuleNotFoundError:
    from transform_diagnosis import eval as ev, chat_format as cf

# Reuse the tuned-coords cell's id selection, raw loader, seed/data-dir defaults, and
# text-only greedy generation, so THIS cell scores the SAME sampled ids under the same seed
# (paired with eval_tuned_coords.py / eval_tuned.py) and decodes identically. Reusing the
# exact functions — not re-implementing them — is what guarantees the ids line up.
from eval_tuned_coords import (  # noqa: E402  (after sys.path setup)
    DEFAULT_DATA_DIR,
    DEFAULT_SEED,
    load_raw,
    run_model,
    select_ids,
)

# The base repo id ``model/train.py`` fine-tunes from (the 4-bit pre-quantized Qwen3-VL-4B).
# Kept in sync with train.py by hand; override with --base-model.
DEFAULT_BASE_MODEL = "unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit"
DEFAULT_SHOTS = 3
DEFAULT_SAMPLE = 500
DEFAULT_TAG = "base_coords_fewshot"

# Preference order for few-shot demonstration labels. The FIRST THREE deliberately span the
# three operation THEMES (rotation / reflection / translation), so even the default K=3 covers
# the space; the full order reaches all 8 labels by K=8 and then cycles (with fresh records)
# for larger K. Any label not listed is appended (sorted) so the selection is total.
_FEWSHOT_LABEL_ORDER = [
    "wrong_rotation_angle",             # rotation-type error
    "wrong_reflection_line",            # reflection-type error
    "wrong_translation",                # translation-type error
    "reflection_instead_of_rotation",   # rotation/reflection orientation confusion
    "opposite_translation",             # translation-type variant (reversed direction)
    "rotation_instead_of_reflection",   # orientation confusion (other direction)
    "completely_wrong",                 # linear part AND translation both wrong
    "correct",                          # no error at all
]


# --- Few-shot demonstration selection (TRAIN only, deterministic, diverse) -----------------
def select_fewshot_ids(train_by_id, shots, seed):
    """Pick ``shots`` DIVERSE, deterministic demonstration ids from the TRAIN pool.

    Groups train ids by label, seeded-shuffles each label's queue, then walks
    ``_FEWSHOT_LABEL_ORDER`` round-robin — so the demonstrations prefer distinct labels
    (K=3 -> rotation/reflection/translation) and only repeat a label, with a fresh record,
    once every label has been used. Returns ids in demonstration order. Never draws from
    test/ood, so it cannot leak.
    """
    rng = random.Random(seed)
    queues: dict = {}
    for rid, rec in train_by_id.items():
        queues.setdefault(rec["label"], []).append(rid)
    for ids in queues.values():
        ids.sort()          # reproducible base order before the seeded shuffle
        rng.shuffle(ids)

    order = [lab for lab in _FEWSHOT_LABEL_ORDER if queues.get(lab)]
    order += [lab for lab in sorted(queues) if lab not in order]

    chosen = []
    k = 0
    while len(chosen) < shots:
        live = [lab for lab in order if queues[lab]]
        if not live:
            break                        # train pool exhausted (never happens in practice)
        chosen.append(queues[live[k % len(live)]].pop())
        k += 1
    return chosen


def build_fewshot_prompt(demos, query_rec):
    """Assemble the K-shot coordinates prompt: K worked demonstrations then the query.

    Every demonstration and the query use the SAME shared ``cf.coords_prompt`` (so the wording
    can never drift from the other coordinate cells), and each demonstrated answer is exactly
    ``cf.target_json`` — the identical label/correct_transform/hint JSON the fine-tune trains
    on — so the schema shown to the model is byte-identical to the expected output.
    """
    blocks = [
        "You will diagnose a student's geometry-transformation mistake from vertex "
        "coordinates. First study the worked examples, each ending in the correct JSON "
        "answer, then answer the final case in that exact same JSON format."
    ]
    for i, demo in enumerate(demos, 1):
        blocks.append(f"===== EXAMPLE {i} =====\n{cf.coords_prompt(demo)}\n\nANSWER:\n"
                      f"{cf.target_json(demo)}")
    blocks.append(f"===== YOUR TURN =====\n{cf.coords_prompt(query_rec)}\n\nANSWER:\n")
    return "\n\n".join(blocks)


# --- Model (loaded lazily so the module imports without unsloth/torch for smoke tests) -----
def load_base_model(base_model):
    """Load the BASE model (NOT the LoRA adapters) — the un-fine-tuned checkpoint train.py
    starts from. Same unsloth call shape as eval_tuned_coords.load_model, but the plain base
    repo id in 4-bit, so this measures the base 4B's reasoning, free of the image-only tune."""
    from unsloth import FastVisionModel
    model, tokenizer = FastVisionModel.from_pretrained(base_model, load_in_4bit=True)
    FastVisionModel.for_inference(model)
    print("loaded BASE model (no adapters):", base_model, flush=True)
    return model, tokenizer


def run_eval(tag, ids, raw_by_id, demos, generate, max_new_tokens):
    """Score ``ids`` with the K-shot prompt via ``generate(text) -> raw_output``.

    Flushes partial results every 200 records (insurance against a wall-clock timeout),
    exactly like eval_tuned_coords.run_eval."""
    scored = []
    for i, rid in enumerate(ids):
        out = generate(build_fewshot_prompt(demos, raw_by_id[rid]))
        scored.append(ev.score_record(out, raw_by_id[rid]))
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


def _print_dryrun_checks(demos, demo_ids, train_by_id, raw, eval_ids, split):
    """Verify (a) no leakage, (b) demonstrated JSON schema, (c) assembled prompt structure —
    on real records, with no model loaded."""
    demo_set, eval_set = set(demo_ids), set(eval_ids)
    all_train = all(train_by_id.get(i, {}).get("split") == "train" for i in demo_ids)
    overlap = demo_set & eval_set
    print(f"  (a) few-shot ids {demo_ids}")
    print(f"      all from train split : {all_train}")
    print(f"      disjoint from the {len(eval_ids)} evaluated {split} ids : "
          f"{'OK (no leakage)' if not overlap else f'LEAK {sorted(overlap)}'}")

    demo_keys = list(json.loads(cf.target_json(demos[0])).keys())
    gold_keys = list(json.loads(cf.target_json(raw[eval_ids[0]])).keys())  # a real gold target
    schema_ok = demo_keys == gold_keys == list(cf.TARGET_KEYS)
    print(f"  (b) demonstrated JSON keys {demo_keys}")
    print(f"      == gold target keys  {gold_keys}  == TARGET_KEYS : "
          f"{'OK' if schema_ok else 'MISMATCH'}")

    prompt = build_fewshot_prompt(demos, raw[eval_ids[0]])
    n_demo = prompt.count("===== EXAMPLE ")
    has_query = "===== YOUR TURN =====" in prompt
    print(f"  (c) assembled prompt: {n_demo} demonstrations + query "
          f"(query block present: {has_query}), {len(prompt)} chars")
    print(f"      first 320 chars:\n        " + prompt[:320].replace("\n", "\n        "))
    print(f"      tail 220 chars:\n        " + prompt[-220:].replace("\n", "\n        "))


def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="BASE-4B coordinates-as-text few-shot eval (disambiguating experiment).")
    ap.add_argument("--splits", default="test,ood",
                    help="comma-separated splits to EVALUATE (default: test,ood)")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"deterministically sample N ids per split (0 = all; "
                         f"default {DEFAULT_SAMPLE}); with --seed reproduces the OTHER cells' ids")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"sample + few-shot seed (default {DEFAULT_SEED}, matches "
                         f"eval_tuned_coords.py so image/coords cells score the SAME ids)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per split after sampling (0 = no cap); cheap smoke test")
    ap.add_argument("--shots", type=int, default=DEFAULT_SHOTS,
                    help=f"number of in-context TRAIN examples to prepend (default {DEFAULT_SHOTS})")
    ap.add_argument("--max-new-tokens", type=int, default=256, dest="max_new_tokens",
                    help="generation length (default: 256)")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL, dest="base_model",
                    help=f"base repo id (default from train.py: {DEFAULT_BASE_MODEL})")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dataset dir with <split>.jsonl (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--tag", default=DEFAULT_TAG,
                    help=f"output tag: results_<tag>_<split>.json (default: {DEFAULT_TAG})")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the K-shot prompts + score empty outputs (NO model load); "
                         "prints the no-leakage / schema / prompt-structure checks and writes "
                         "<tag>_dryrun_* files to verify the pipeline")
    return ap


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]

    # Few-shot demonstrations ALWAYS come from TRAIN (never test/ood) -> no leakage by
    # construction. Selected ONCE (a fixed preamble reused for every query and every split).
    train_by_id = load_raw(args.data_dir, "train")
    demo_ids = select_fewshot_ids(train_by_id, args.shots, args.seed)
    demos = [train_by_id[i] for i in demo_ids]

    if args.dry_run:
        generate = lambda text: ""            # no model; honest empty output -> parse_fail
        tag_base = f"{args.tag}_dryrun"
        print(f"BASE-4B coords few-shot eval: DRY RUN (no model load)  shots={args.shots}"
              f"{f'  limit={args.limit}' if args.limit else ''}\n", flush=True)
    else:
        model, tokenizer = load_base_model(args.base_model)
        generate = lambda text: run_model(model, tokenizer, text, args.max_new_tokens)
        tag_base = args.tag
        print(f"BASE-4B coords few-shot eval: base={args.base_model}  shots={args.shots}  "
              f"splits={splits}\n", flush=True)

    print(f"Few-shot demonstrations ({len(demos)}, from TRAIN — no leakage):", flush=True)
    for j, d in enumerate(demos, 1):
        print(f"  [{j}] id={d['id']} split={d['split']} label={d['label']}", flush=True)
    print(flush=True)

    for split in splits:
        raw = load_raw(args.data_dir, split)
        ids = select_ids(raw, args.sample, args.seed, args.limit)
        # No-leakage guard: a demonstration id must NEVER be an evaluated id.
        overlap = set(demo_ids) & set(ids)
        assert not overlap, f"LEAKAGE: few-shot ids {sorted(overlap)} appear in {split} eval set"
        if args.dry_run:
            print(f"[{split}] dry-run checks on {len(ids)} evaluated records:", flush=True)
            _print_dryrun_checks(demos, demo_ids, train_by_id, raw, ids, split)
        print(f"[{split}] scoring {len(ids)} records "
              f"({args.shots}-shot coords{', dry-run' if args.dry_run else ''}) ...", flush=True)
        run_eval(f"{tag_base}_{split}", ids, raw, demos, generate, args.max_new_tokens)

    if args.dry_run:
        print("\nDRY RUN complete — few-shot prompt build + scoring pipeline OK; no model loaded.")
    else:
        print(f"\nSaved results_{tag_base}_{{{','.join(splits)}}}.json + "
              f"records_{tag_base}_{{{','.join(splits)}}}.jsonl", flush=True)


if __name__ == "__main__":
    main()
