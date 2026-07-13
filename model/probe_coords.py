"""probe_coords.py — is the transform failure PERCEPTION or REASONING?

NOTE: the canonical frontier-coords cell of the 2x2 table is now ``eval_frontier.py
--input coords`` (Claude Opus, so the whole frontier row is one model). This file is the
HISTORICAL GPT-5.3-codex coordinate probe kept for provenance (it produced the original
``results_probe_coords*.json``); its behaviour is unchanged.

Feeds a frontier reasoning model (GPT-5.3-codex, effort high) the exact vertex COORDINATES
as text (no image at all) for a small val sample, and asks for the same JSON diagnosis. The
coordinate prompt is the shared ``chat_format.coords_prompt`` (same one the tuned-4B
coordinate eval uses), and it is scored by the same harness (transform_diagnosis/eval.py).

Decision rule:
  * high transform/label here  -> the geometry is sufficient, task is doable; your 4B's
    failure is PERCEPTION -> a rendering fix (numbered vertices / higher res) is worth it.
  * still low here             -> the bottleneck is REASONING, not the image; a rendering
    fix won't help -> report the capability limit honestly.

Run locally.  pip install openai ; export TFY_API_KEY=tfy_...  ; python model/probe_coords.py
"""

from __future__ import annotations

import json, os, random, sys, time

# TrueFoundry gateway — GPT-5.3 (Responses API) via the EU gateway. (The claude-pro-traffic
# Anthropic account has an invalid upstream key; any strong frontier model answers this probe.)
MODEL = "codex-traffic/gpt-5.3-codex"
BASE_URL = "https://tfy-eu.promptlens.trilogy.com"
N = 30                    # val records to probe (each = one text API call)
SEED = 20260709
DATA_DIR = "transform_diagnosis_data"

for cand in (".", ".."):
    if os.path.isdir(os.path.join(cand, "transform_diagnosis")):
        sys.path.insert(0, cand); break

from transform_diagnosis import eval as ev, chat_format as cf

from openai import OpenAI
_client = OpenAI(api_key=os.environ["TFY_API_KEY"], base_url=BASE_URL)


def _extract(resp):
    """Pull the assistant text out of a Responses API result (SDK convenience first,
    then a manual walk of the output items as a fallback)."""
    text = getattr(resp, "output_text", None)
    if text:
        return text
    parts = []
    for item in (getattr(resp, "output", None) or []):
        for c in (getattr(item, "content", None) or []):
            t = getattr(c, "text", None)
            if t:
                parts.append(t)
    return "".join(parts)


def ask(rec):
    for attempt in range(4):
        try:
            resp = _client.responses.create(
                model=MODEL, input=cf.coords_prompt(rec), max_output_tokens=8000,
                reasoning={"effort": "high"},   # deterministic geometry: let it actually think
            )
            return (_extract(resp) or "").strip()
        except Exception as exc:
            if attempt == 3:
                print(f"  !! id={rec['id']} failed: {exc}", flush=True); return ""
            time.sleep(2 ** attempt)
    return ""


def main():
    raw = {}
    for l in open(os.path.join(DATA_DIR, "val.jsonl")):
        if l.strip():
            r = json.loads(l); raw[r["id"]] = r
    ids = sorted(random.Random(SEED).sample(sorted(raw), min(N, len(raw))))
    print(f"Probe: {MODEL} on {len(ids)} val records, COORDINATES-AS-TEXT (no image)\n", flush=True)

    scored = []
    for i, rid in enumerate(ids):
        scored.append(ev.score_record(ask(raw[rid]), raw[rid]))
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(ids)}", flush=True)

    agg = ev.aggregate(scored)
    ev.save_results(agg, "results_probe_coords.json", scored, "records_probe_coords.jsonl")

    print("\n================ PROBE RESULT (coords-as-text, frontier) ================")
    print(f"n={agg['n']}  label_acc={agg['label_accuracy']:.3f}  "
          f"balanced_acc={agg['balanced_accuracy']:.3f}  parse={agg['parse_rate']:.3f}  "
          f"transform={agg['transform_match_rate']:.3f}  hint={agg['hint_match_rate']:.3f}")
    print("\nCompare to your tuned 4B on val (from IMAGES): "
          "label=0.455  transform=0.056  hint=0.061")
    print("\nRead it:")
    print("  * transform HIGH here (say >0.6)  -> perception is the bottleneck; a rendering")
    print("    fix (numbered vertices / higher res) is worth the retrain.")
    print("  * transform still LOW here        -> reasoning ceiling; rendering won't help,")
    print("    report the limit and skip the re-render.")

    # A couple of concrete cases to eyeball.
    print("\n== Sample cases ==")
    for r in scored[:5]:
        print(f"  id={r['id']} label={r['true_label']} label_ok={r['label_ok']} "
              f"transform_ok={r['transform_ok']}")
        print(f"    {r['raw_model_output'][:180]}")


if __name__ == "__main__":
    main()
