"""eval_frontier.py — the FRONTIER row of the 2x2 table, ONE model (Claude Opus 4.8).

Runs a prompted frontier model over a fixed sample of the frozen test + ood splits, using
the SAME programmatic harness (``transform_diagnosis/eval.py``) that scored base and tuned,
then re-scores base and tuned on the *identical sampled ids* (from their saved per-record
JSONL) so all three columns compare like-for-like, and prints a base / tuned / frontier
table per split.

Two input modes fill BOTH frontier cells of the 2x2 {frontier, tuned-4B} x {image, coords}
table with the SAME model, so the frontier row is uniform (not Opus-image vs GPT-coords):

    --input image  (default): the PNG render + ``chat_format.INSTRUCTION`` — the vision cell.
    --input coords          : NO image; the shared ``chat_format.coords_prompt`` (the exact
                              coordinates-as-text prompt ``eval_tuned_coords.py`` feeds the
                              4B and the GPT probe uses), for the coordinates cell.

Same model, same output JSON schema, same fixed metrics in both modes, so the four cells
stay apples-to-apples. (``probe_coords.py`` is the historical GPT-5.3-codex coordinate
probe; the canonical frontier-coords cell is now ``eval_frontier.py --input coords``.)

Runs anywhere the data + harness are present (no GPU needed — it calls an API):
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
    # for the 3-way table, have the saved per-record files alongside (cluster $HOME, or
    # scp them down):  records_{base,tuned}_{test,ood}.jsonl        (image)
    #                  records_tuned_coords_{test,ood}.jsonl        (coords)
Then, e.g.:
    python model/eval_frontier.py                       # frontier+image (150 images/split)
    python model/eval_frontier.py --input coords        # frontier+coords, same model + ids
    python model/eval_frontier.py --limit 2             # cheap smoke test (2 API calls/split)
    python model/eval_frontier.py --input coords --dry-run --limit 4   # no API; pipeline only
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import random
import sys
import time

# --- Config (defaults; overridable on the CLI) --------------------------------------------
MODEL = "claude-opus-4-8"          # the frontier vision model to compare against
N_PER_SPLIT = 150                  # sampled images per split (each = one vision API call)
SEED = 20260709                    # deterministic sample choice
SPLITS = ("test", "ood")          # frozen evaluation splits only
MAX_TOKENS = 512
MAX_RETRIES = 4                    # transient API error backoff

HOME = os.path.expanduser("~")


# --- Make the harness importable, load it -------------------------------------------------
# Runs both from the repo (package ``transform_diagnosis``) and on the cluster (the synced
# copy is named ``slm_eval`` in $HOME, exactly as eval_tuned.py imports it).
def _add_path(cand: str) -> None:
    if cand and os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)


for _cand in (".", "..", HOME):
    _add_path(_cand)

try:
    from transform_diagnosis import eval as ev, chat_format as cf
except ModuleNotFoundError:
    from slm_eval import eval as ev, chat_format as cf


def _first_existing_dir(*cands: str) -> str:
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[0]


# Local checkout (``./transform_diagnosis_data``) or cluster ($HOME). Results/records are
# read from and written to the cwd (the sbatch does ``cd ~`` first, matching eval_tuned.py).
DATA_DIR = _first_existing_dir(
    "transform_diagnosis_data", os.path.join(HOME, "transform_diagnosis_data")
)
RESULTS_DIR = "."


def load_raw(split):
    """Full oracle records for a split, keyed by id (carry correct_transform,
    student_transform, hint — everything score_record needs)."""
    by_id = {}
    with open(os.path.join(DATA_DIR, f"{split}.jsonl")) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                by_id[r["id"]] = r
    return by_id


def sample_ids(by_id, n, seed):
    """Deterministic id sample (sorted for reproducibility, then seeded choice)."""
    ids = sorted(by_id)
    return sorted(random.Random(seed).sample(ids, min(n, len(ids))))


# --- Frontier model call ------------------------------------------------------------------
def _make_client():
    """Create the Anthropic client lazily (import + key read only when actually calling the
    API), so the module imports cleanly for smoke tests without anthropic or a key."""
    import anthropic
    return anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from the environment


def _b64_png(path):
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode()


def build_content(rec, input_mode):
    """The user-message content for one record.

    ``image``  : the base64 PNG render + the standard ``INSTRUCTION`` (unchanged).
    ``coords`` : NO image — the shared ``coords_prompt`` text (byte-identical to what
                 ``probe_coords.py`` / ``eval_tuned_coords.py`` build for the record), so
                 the SAME Opus model can fill both frontier cells.
    """
    if input_mode == "coords":
        return [{"type": "text", "text": cf.coords_prompt(rec)}]
    img_path = os.path.join(DATA_DIR, rec["render_path"])
    return [
        {"type": "image", "source": {
            "type": "base64", "media_type": "image/png", "data": _b64_png(img_path)}},
        {"type": "text", "text": cf.INSTRUCTION},
    ]


def run_frontier(rec, client, model, max_tokens, input_mode, max_retries=MAX_RETRIES):
    """Send one record to the frontier model (image+INSTRUCTION or coords-as-text, per
    ``input_mode``) and return its raw text. Retries with backoff on transient errors;
    returns "" on hard failure (scored as a parse_fail — an honest outcome, not a crash)."""
    content = build_content(rec, input_mode)
    for attempt in range(max_retries):
        try:
            msg = client.messages.create(
                model=model, max_tokens=max_tokens, temperature=0,
                messages=[{"role": "user", "content": content}],
            )
            return "".join(b.text for b in msg.content if b.type == "text").strip()
        except Exception as exc:                       # rate limit / transient network
            if attempt == max_retries - 1:
                print(f"    !! giving up on id={rec['id']}: {exc}", flush=True)
                return ""
            time.sleep(2 ** attempt)
    return ""


# --- Re-aggregate saved base/tuned rows on the SAME sampled ids ----------------------------
def reagg_saved(tag, split, keep_ids):
    """Load records_<tag>_<split>.jsonl, keep only sampled ids, re-aggregate. Returns the
    aggregate dict, or None if the file isn't present locally."""
    path = os.path.join(RESULTS_DIR, f"records_{tag}_{split}.jsonl")
    if not os.path.exists(path):
        return None
    keep = set(keep_ids)
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("id") in keep:
                    rows.append(r)
    return ev.aggregate(rows) if rows else None


# --- 3-column table -----------------------------------------------------------------------
def format_table3(base, tuned, frontier):
    def cell(agg, key):
        return f"{agg.get(key, 0.0):>9.3f}" if agg else f"{'--':>9}"
    rows = [f"{'metric':<18}{'base':>9}{'tuned':>9}{'frontier':>9}", "-" * 45]
    for key, label in ev._TABLE_METRICS:
        rows.append(f"{label:<18}{cell(base, key)}{cell(tuned, key)}{cell(frontier, key)}")
    return "\n".join(rows)


# --- CLI ----------------------------------------------------------------------------------
def build_arg_parser():
    ap = argparse.ArgumentParser(description="Frontier (image or coords) eval for the 2x2 table.")
    ap.add_argument("--input", choices=("image", "coords"), default="image",
                    help="frontier input modality: rendered PNG (default) or coords-as-text")
    ap.add_argument("--model", default=MODEL, help=f"frontier model (default: {MODEL})")
    ap.add_argument("--n-per-split", type=int, default=N_PER_SPLIT, dest="n_per_split",
                    help=f"sampled images per split (default: {N_PER_SPLIT})")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per split after sampling (0 = no cap); cheap smoke test")
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help=f"comma-separated splits (default: {','.join(SPLITS)})")
    ap.add_argument("--seed", type=int, default=SEED, help=f"sample seed (default: {SEED})")
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS, dest="max_tokens",
                    help=f"max output tokens (default: {MAX_TOKENS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="exercise the full pipeline (load, sample, encode PNG, score, save) "
                         "with NO API calls; writes *_dryrun.* files")
    return ap


# --- Main ---------------------------------------------------------------------------------
def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    splits = [s for s in args.splits.split(",") if s]
    suffix = "_dryrun" if args.dry_run else ""
    # Distinct output tag per input mode so a coords run never overwrites the image run;
    # compare against the SAME-modality base/tuned records so the table stays like-for-like.
    tag = "frontier_coords" if args.input == "coords" else "frontier"
    cmp_base, cmp_tuned = (("base_coords", "tuned_coords") if args.input == "coords"
                           else ("base", "tuned"))
    client = None if args.dry_run else _make_client()

    mode = "DRY RUN (no API calls)" if args.dry_run else f"model={args.model}"
    print(f"Frontier eval ({args.input}): {mode}  n_per_split={args.n_per_split}"
          f"{f'  limit={args.limit}' if args.limit else ''}\n", flush=True)

    for split in splits:
        raw = load_raw(split)
        ids = sample_ids(raw, args.n_per_split, args.seed)
        if args.limit:
            ids = ids[:args.limit]
        print(f"[{split}] scoring {len(ids)} {args.input} inputs "
              f"({'dry-run' if args.dry_run else args.model}) ...", flush=True)

        scored = []
        for i, rid in enumerate(ids):
            rec = raw[rid]
            if args.dry_run:
                build_content(rec, args.input)   # assert render exists / build coords prompt
                out = ""                         # no API output
            else:
                out = run_frontier(rec, client, args.model, args.max_tokens, args.input)
            scored.append(ev.score_record(out, rec))
            if (i + 1) % 25 == 0:
                print(f"  [{split}] {i + 1}/{len(ids)}", flush=True)

        front = ev.aggregate(scored)
        ev.save_results(front, f"results_{tag}_{split}{suffix}.json",
                        scored, f"records_{tag}_{split}{suffix}.jsonl")

        base = reagg_saved(cmp_base, split, ids)
        tuned = reagg_saved(cmp_tuned, split, ids)

        print(f"\n== {split.upper()} — base vs tuned vs frontier "
              f"(same {len(ids)} {args.input} inputs) ==")
        print(format_table3(base, tuned, front))
        if base is None or tuned is None:
            print(f"  (base/tuned columns blank — copy records_{cmp_base}_{split}.jsonl / "
                  f"records_{cmp_tuned}_{split}.jsonl here)")
        print(flush=True)

    if args.dry_run:
        print("DRY RUN complete — pipeline OK; no API calls made. Re-run without --dry-run "
              "(with ANTHROPIC_API_KEY set) for real numbers.")
    else:
        print("Saved results_%s_{%s}.json + records_%s_{%s}.jsonl"
              % (tag, ",".join(splits), tag, ",".join(splits)))


if __name__ == "__main__":
    main()
