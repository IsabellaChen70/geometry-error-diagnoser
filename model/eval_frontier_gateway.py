"""eval_frontier_gateway.py — the FRONTIER row of the 2x2 table via the TrueFoundry
(OpenAI-compatible) LLM gateway, for when there is NO direct Anthropic key.

This is the gateway twin of ``eval_frontier.py``. ``eval_frontier.py`` calls the direct
``anthropic`` SDK (needs ANTHROPIC_API_KEY); this script instead routes through the same
TrueFoundry gateway that ``probe_coords.py`` already uses successfully — the OpenAI
**Responses API** at the promptlens EU endpoint, authenticated with ``TFY_API_KEY``. Use
this when your only frontier access is that gateway.

It runs a prompted frontier model over a fixed sample of the frozen test + ood splits and
scores it with the SAME programmatic harness (``transform_diagnosis/eval.py`` locally /
``slm_eval.eval`` on the cluster) that scored base and tuned. Output schema and metrics are
byte-for-byte the fixed harness's, so this cell stays apples-to-apples with the other three.

Two input modes fill BOTH frontier cells with the SAME model (so the frontier row is
uniform), exactly like ``eval_frontier.py``:

    --input image  (default): the PNG render + ``chat_format.INSTRUCTION`` sent as an
                              OpenAI Responses-API vision message (an ``input_text`` part +
                              an ``input_image`` part carrying a base64 ``data:image/png``
                              data URL). THIS is the cell that could not run without an
                              Anthropic key before.
    --input coords          : NO image; the shared ``chat_format.coords_prompt`` string
                              (byte-identical to ``probe_coords.py`` / ``eval_tuned_coords``).

Client construction, the ``_extract`` output-walk, the retry/backoff loop, and the scoring
call mirror ``probe_coords.py`` exactly; only the request *input* changes (image+instruction
instead of coords text). Ids are chosen with ``eval_tuned_coords.select_ids`` under the same
default seed, so ``--sample N --seed S`` scores the IDENTICAL records the tuned/coords cells
score at that N/S — the whole 2x2 stays paired.

``--schema v6`` switches to the shared canonical-net prompt and supports task modes
``correct`` / ``student`` / ``both`` / ``full``. ``--input image_coords`` adds exact
corresponding vertices alongside the image. The default remains the legacy schema, so
existing invocations and output behavior are unchanged.

Runs anywhere the data + harness are present — it calls an API, so NO GPU is needed (login
node or laptop):
    pip install openai
    export TFY_API_KEY=tfy_...
    # the frontier model MUST be a VISION-capable route on your gateway (see --model note):
    python model/eval_frontier_gateway.py --sample 500 --seed 20260709 --model <vision-route>
    python model/eval_frontier_gateway.py --input coords --sample 500   # same model + ids
    python model/eval_frontier_gateway.py --dry-run --limit 2 --splits test  # no API; plumbing

Fallback: if your gateway rejects the Responses-API vision shape, pass ``--api chat`` to send
the OpenAI Chat-Completions vision shape instead (``image_url: {"url": ...}``); see the
``--api`` help and the ``build_chat_messages`` docstring.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import os
import sys
import time

# --- Config (defaults; overridable on the CLI) --------------------------------------------
# Default gateway route. NOTE: this MUST be a VISION-capable model on your gateway for the
# (default) image cell — see the --model help and the startup banner. Overridable.
MODEL = "codex-traffic/gpt-5.3-codex"
BASE_URL = "https://tfy-eu.promptlens.trilogy.com"   # promptlens EU gateway (same as probe)
SPLITS = ("test", "ood")                              # frozen evaluation splits only
DEFAULT_SAMPLE = 150                 # sampled inputs per split (each = one API call); 0 = ALL
MAX_OUTPUT_TOKENS = 8000             # generous, like probe_coords.py (reasoning needs room)
REASONING_EFFORT = "high"            # deterministic geometry: let it actually think
MAX_RETRIES = 4                      # transient API error backoff (mirrors probe_coords.py)
IMAGE_TAG = "frontier_image_gw"      # default output tag (image cell)
COORDS_TAG = "frontier_coords_gw"    # output tag when --input coords (so it never overwrites)

# --- Chain-of-thought (--cot) knobs (frontier PROMPT-only; nothing else changes) ----------
# Floor on max_output_tokens under --cot so reasoning + the final JSON both fit. The default
# MAX_OUTPUT_TOKENS (8000) already exceeds this, so the default run is unaffected; the floor
# only raises a smaller explicit --max-output-tokens. Applied ONLY when --cot is set.
COT_MIN_OUTPUT_TOKENS = 4000
# Directive appended AFTER the (unmodified) v6_format/legacy prompt when --cot is passed. It
# (a) explicitly overrides that prompt's trailing "Return one valid JSON object and nothing
# else." instruction so the model may reason first, and (b) requires the SAME final JSON
# object as the very last thing in the reply. transform_diagnosis.eval.parse_pred keeps the
# LAST brace-balanced object, so "reasoning ... {final JSON}" parses + scores byte-identically
# to the direct-output path — only the prompt changes, so GPT-CoT stays comparable to
# GPT-direct and to every model cell (both_nets etc.). We do NOT touch v6_format.py.
COT_DIRECTIVE = (
    "\n\n"
    "Disregard the instruction above to return only one JSON object and nothing else. "
    "First, work through the problem step by step: for each shape, read the vertices in "
    "corresponding order, determine the linear operation (which D4 rotation or reflection) "
    "and the exact integer translation (tx, ty) that maps RED->GREEN and RED->BLUE, checking "
    "a few vertices to confirm. After your reasoning, end your reply with EXACTLY the single "
    "JSON object specified above (the same schema and keys) as the very last thing, with "
    "nothing after it."
)

HOME = os.path.expanduser("~")


# --- Make the harness + sibling importable ------------------------------------------------
# This script's OWN dir first so the sibling ``eval_tuned_coords`` import resolves from any
# cwd (repo/model locally, $HOME on the cluster); then the harness, preferring the cluster
# package name (``slm_eval``) and falling back to the in-repo ``transform_diagnosis`` — the
# same import-fallback pattern as eval_tuned_coords.py, so the non-GPU logic imports and
# smoke-tests locally (openai stays UNimported until an actual API call).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _cand in (_HERE, os.path.dirname(_HERE), ".", "..", HOME):
    if _cand and os.path.isdir(_cand) and _cand not in sys.path:
        sys.path.insert(0, _cand)

try:
    from slm_eval import eval as ev, chat_format as cf, v6_format
except ModuleNotFoundError:
    from transform_diagnosis import eval as ev, chat_format as cf, v6_format

# Reuse the coords cell's id selection + raw loader + seed so image & coords cells (here and
# in the tuned scripts) sample the SAME record ids under the same seed. Reusing the exact
# function — not re-implementing — is what guarantees the paired comparison.
from eval_tuned_coords import (  # noqa: E402  (after sys.path setup)
    DEFAULT_SEED,
    load_raw,
    select_ids,
)


def _first_existing_dir(*cands: str) -> str:
    for c in cands:
        if c and os.path.isdir(c):
            return c
    return cands[0]


# Local checkout (``./transform_diagnosis_data``) or cluster ($HOME). Overridable via
# --data-dir. Results/records are written to the cwd (the sbatch does ``cd ~`` first).
DEFAULT_DATA_DIR = _first_existing_dir(
    "transform_diagnosis_data", os.path.join(HOME, "transform_diagnosis_data")
)


def load_raw_any(data_dir: str, split: str) -> dict:
    """Load legacy ``<split>.jsonl`` or a generated v6 ``<split>_v6.jsonl``."""
    legacy = os.path.join(data_dir, f"{split}.jsonl")
    if os.path.isfile(legacy):
        return load_raw(data_dir, split)
    v6 = os.path.join(data_dir, f"{split}_v6.jsonl")
    if not os.path.isfile(v6):
        raise FileNotFoundError(f"neither {legacy} nor {v6} exists")
    with open(v6) as handle:
        return {
            rec["id"]: rec
            for rec in (json.loads(line) for line in handle if line.strip())
        }


# --- Data loading (same render-path resolution eval_tuned.py uses) -------------------------
def load_chat(data_dir: str, split: str) -> dict:
    """Chat rows (image + instruction messages) for a split, keyed by id — exactly as
    eval_tuned.py loads them. The image content part holds the render's relative path."""
    with open(os.path.join(data_dir, f"{split}_chat.jsonl")) as f:
        return {r["id"]: r for r in (json.loads(l) for l in f if l.strip())}


def _image_rel(chat_row: dict) -> str:
    """The render's relative path from a chat row (the ``{"type":"image","image":...}``
    content part) — the same field eval_tuned.py's ``decode_user`` reads before decoding."""
    for part in chat_row["messages"][0]["content"]:
        if part.get("type") == "image" and isinstance(part.get("image"), str):
            return part["image"]
    raise KeyError(f"no image content part in chat row id={chat_row.get('id')}")


def _b64_png_data_url(path: str) -> str:
    """Read a PNG and return it as a base64 ``data:image/png;base64,...`` data URL (the form
    the OpenAI Responses/Chat vision APIs accept for an inline image)."""
    with open(path, "rb") as f:
        b64 = base64.standard_b64encode(f.read()).decode()
    return f"data:image/png;base64,{b64}"


# --- Request payload construction ---------------------------------------------------------
def build_responses_input(
    rec: dict,
    chat_row,
    data_dir: str,
    input_mode: str,
    schema: str = "legacy",
    task: str = "full",
    cot: bool = False,
):
    """The ``input`` for ``client.responses.create``.

    ``image``  : a single user message mixing an ``input_text`` part (the shared
                 ``chat_format.INSTRUCTION``) and an ``input_image`` part whose ``image_url``
                 is the base64 ``data:image/png;base64,...`` data URL of the render. This is
                 the OpenAI Responses-API vision shape.
    ``coords`` : the plain ``chat_format.coords_prompt`` STRING — byte-identical to what
                 probe_coords.py passes as ``input`` (no image), so the coords cell matches.

    ``cot`` appends :data:`COT_DIRECTIVE` AFTER the schema prompt (reasoning-then-JSON); it
    defaults False so the direct path is byte-identical.
    """
    if schema == "v6":
        prompt_mode = "coords" if input_mode == "coords" else input_mode
        prompt = v6_format.instruction(rec, task, prompt_mode)
    else:
        prompt = cf.coords_prompt(rec) if input_mode == "coords" else cf.INSTRUCTION
    if cot:
        prompt = prompt + COT_DIRECTIVE
    if input_mode == "coords":
        return prompt
    data_url = _b64_png_data_url(os.path.join(data_dir, _image_rel(chat_row)))
    return [
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": data_url},
            ],
        }
    ]


def build_chat_messages(
    rec: dict,
    chat_row,
    data_dir: str,
    input_mode: str,
    schema: str = "legacy",
    task: str = "full",
    cot: bool = False,
):
    """Chat-Completions ``messages`` fallback (used with ``--api chat``).

    Same content, the Chat-Completions vision shape instead of the Responses one: a ``text``
    part + an ``image_url`` part whose value is the OBJECT ``{"url": "data:image/png;..."}``
    (note: Chat wraps the URL in an object; Responses takes the URL string directly). Use
    this if your gateway serves vision through /chat/completions rather than /responses.

    ``cot`` appends :data:`COT_DIRECTIVE` AFTER the schema prompt (reasoning-then-JSON); it
    defaults False so the direct path is byte-identical.
    """
    if schema == "v6":
        prompt_mode = "coords" if input_mode == "coords" else input_mode
        prompt = v6_format.instruction(rec, task, prompt_mode)
    else:
        prompt = cf.coords_prompt(rec) if input_mode == "coords" else cf.INSTRUCTION
    if cot:
        prompt = prompt + COT_DIRECTIVE
    if input_mode == "coords":
        return [{"role": "user", "content": prompt}]
    data_url = _b64_png_data_url(os.path.join(data_dir, _image_rel(chat_row)))
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]


def _use_reasoning(effort: str) -> bool:
    """Whether to send a reasoning directive. ``--reasoning-effort none|off|''`` omits it
    (some vision routes reject a reasoning param); anything else is passed through."""
    return bool(effort) and effort.strip().lower() not in ("none", "off")


def build_request_kwargs(model, payload, api, max_output_tokens, reasoning_effort):
    """Assemble the exact kwargs handed to the create() call (also what --dry-run prints).

    Responses: ``input`` + ``max_output_tokens`` + optional ``reasoning={"effort":...}``
               (mirrors probe_coords.py). Chat: ``messages`` + ``max_completion_tokens``
               (the reasoning-model field; non-reasoning models want ``max_tokens`` instead)
               + optional ``reasoning_effort``.
    """
    if api == "responses":
        kw = {"model": model, "input": payload, "max_output_tokens": max_output_tokens}
        if _use_reasoning(reasoning_effort):
            kw["reasoning"] = {"effort": reasoning_effort}
    else:
        kw = {"model": model, "messages": payload,
              "max_completion_tokens": max_output_tokens}
        if _use_reasoning(reasoning_effort):
            kw["reasoning_effort"] = reasoning_effort
    return kw


# --- Gateway client + call (mirrors probe_coords.py) --------------------------------------
def make_client(base_url: str):
    """Create the OpenAI-compatible gateway client lazily (import + key read only when
    actually calling the API), so the module imports cleanly for smoke tests without the
    ``openai`` package or a key. Same construction as probe_coords.py."""
    from openai import OpenAI
    return OpenAI(api_key=os.environ["TFY_API_KEY"], base_url=base_url)


def _extract(resp) -> str:
    """Pull the assistant text out of a Responses API result (SDK convenience first, then a
    manual walk of the output items as a fallback). Copied verbatim from probe_coords.py."""
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


def _extract_chat(resp) -> str:
    """Pull the assistant text out of a Chat-Completions result (the --api chat path)."""
    try:
        return resp.choices[0].message.content or ""
    except (AttributeError, IndexError, TypeError):
        return ""


def ask(client, rid, kw, api, max_retries=MAX_RETRIES) -> str:
    """Send one request and return its raw text, retrying with exponential backoff on
    transient errors; returns "" on hard failure (scored as a parse_fail — an honest
    outcome, not a crash). Same ret/retry structure as probe_coords.py."""
    for attempt in range(max_retries):
        try:
            if api == "responses":
                resp = client.responses.create(**kw)
                return (_extract(resp) or "").strip()
            resp = client.chat.completions.create(**kw)
            return (_extract_chat(resp) or "").strip()
        except Exception as exc:                       # rate limit / transient network
            if attempt == max_retries - 1:
                print(f"  !! id={rid} failed: {exc}", flush=True)
                return ""
            time.sleep(2 ** attempt)
    return ""


# --- Dry-run: show the assembled request with the base64 blob truncated --------------------
def _redact(obj):
    """Deep copy of a request payload with any base64 data URL truncated, so the request
    SHAPE is printable/inspectable without dumping a multi-KB image blob."""
    if isinstance(obj, dict):
        return {k: _redact(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]
    if isinstance(obj, str) and obj.startswith("data:image"):
        return f"{obj[:40]}...<truncated, {len(obj)} chars total>"
    return obj


# --- CLI ----------------------------------------------------------------------------------
def build_arg_parser():
    ap = argparse.ArgumentParser(
        description="Frontier (image or coords) eval for the 2x2 table, via the TrueFoundry "
                    "OpenAI-compatible gateway (no Anthropic key needed).")
    ap.add_argument("--input", choices=("image", "coords", "image_coords"), default="image",
                    help="frontier input modality: rendered PNG (default), coordinates-only, "
                         "or image+coordinates (the latter is v6-only)")
    ap.add_argument("--schema", choices=("legacy", "v6"), default="legacy",
                    help="output/prompt schema; legacy preserves all existing behavior")
    ap.add_argument("--task", choices=v6_format.TASK_MODES, default="full",
                    help="v6 transform task (ignored for --schema legacy)")
    ap.add_argument("--model", default=MODEL,
                    help=f"gateway-routed frontier model (default: {MODEL}). For --input "
                         f"image this MUST be a VISION-capable route on your gateway.")
    ap.add_argument("--base-url", default=BASE_URL, dest="base_url",
                    help=f"OpenAI-compatible gateway base URL (default: {BASE_URL})")
    ap.add_argument("--api", choices=("responses", "chat"), default="responses",
                    help="gateway API surface: 'responses' (default; mirrors probe_coords.py) "
                         "or 'chat' (Chat-Completions vision fallback if the gateway rejects "
                         "the Responses image shape)")
    ap.add_argument("--splits", default=",".join(SPLITS),
                    help=f"comma-separated splits (default: {','.join(SPLITS)})")
    ap.add_argument("--data-dir", default=DEFAULT_DATA_DIR, dest="data_dir",
                    help=f"dataset dir with <split>.jsonl + <split>_chat.jsonl + renders/ "
                         f"(default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--sample", type=int, default=DEFAULT_SAMPLE,
                    help=f"deterministically sample N ids per split (default: {DEFAULT_SAMPLE}; "
                         f"0 = ALL). With --seed reproduces the SAME ids the tuned/coords "
                         f"cells score (via eval_tuned_coords.select_ids).")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"sample seed (default: {DEFAULT_SEED}, matches the other 2x2 cells)")
    ap.add_argument("--limit", type=int, default=0,
                    help="cap records per split after sampling (0 = no cap); cheap smoke test")
    ap.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS,
                    dest="max_output_tokens",
                    help=f"max output tokens (default: {MAX_OUTPUT_TOKENS}; generous so a "
                         f"reasoning model has room to think, like probe_coords.py)")
    ap.add_argument("--reasoning-effort", default=REASONING_EFFORT, dest="reasoning_effort",
                    help=f"reasoning effort (default: {REASONING_EFFORT}); 'none'/'off' omits "
                         f"the reasoning directive for routes that don't accept it")
    ap.add_argument("--cot", action="store_true",
                    help="chain-of-thought: append a directive AFTER the schema prompt that "
                         "overrides 'return only JSON', asks the model to reason step by step, "
                         "and requires the SAME final JSON object last (eval.parse_pred keeps "
                         "the LAST JSON object, so scoring/oracle/ids/schema are unchanged — "
                         "only the frontier prompt changes). Raises the effective "
                         f"--max-output-tokens floor to {COT_MIN_OUTPUT_TOKENS} for reasoning "
                         "room. Prompt-driven (independent of --reasoning-effort), so it works "
                         "for non-reasoning routes like gpt-4o. No effect when absent.")
    ap.add_argument("--tag", default=IMAGE_TAG,
                    help=f"output tag: results_<tag>_<split>.json (default: {IMAGE_TAG}; "
                         f"auto-switches to {COORDS_TAG} for --input coords unless overridden)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the base64 image payload + prompt and run the "
                         "score->aggregate->save pipeline with an EMPTY response (NO API "
                         "call, no key/network); prints the assembled request shape and "
                         "writes *_dryrun.* files")
    return ap


def _resolve_tag(args) -> str:
    """Coords runs get their own tag so they never overwrite the image run — unless the user
    explicitly set --tag to something other than the image default."""
    if args.tag != IMAGE_TAG:
        return args.tag
    if args.schema == "v6":
        modality = "coords" if args.input == "coords" else (
            "image_coords" if args.input == "image_coords" else "image"
        )
        return f"frontier_v6_{modality}_{args.task}"
    if args.input == "coords":
        return COORDS_TAG
    return args.tag


def _chat_row_for_input(chat_by_id: dict, rid, input_mode: str):
    """Return the render-bearing row for every input mode that includes an image."""
    if input_mode in ("image", "image_coords"):
        return chat_by_id.get(rid)
    return None


# --- Main ---------------------------------------------------------------------------------
def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    if args.schema == "legacy" and args.input == "image_coords":
        raise SystemExit("--input image_coords is defined only for --schema v6")
    splits = [s for s in args.splits.split(",") if s]
    tag = _resolve_tag(args)
    suffix = "_dryrun" if args.dry_run else ""
    data_dir = args.data_dir
    # Under --cot, floor max_output_tokens so reasoning + the final JSON both fit. No effect
    # without --cot (direct path uses args.max_output_tokens unchanged).
    max_output_tokens = (max(args.max_output_tokens, COT_MIN_OUTPUT_TOKENS)
                         if args.cot else args.max_output_tokens)

    mode = "DRY RUN (no API calls)" if args.dry_run else f"model={args.model}  api={args.api}"
    print(f"Frontier gateway eval ({args.schema}/{args.input}/{args.task}): {mode}", flush=True)
    print(f"  base_url={args.base_url}  sample={args.sample or 'ALL'}"
          f"{f'  limit={args.limit}' if args.limit else ''}  seed={args.seed}  "
          f"data_dir={data_dir}", flush=True)
    if args.cot:
        print(f"  CoT: reasoning-before-JSON directive appended to the prompt (overrides "
              f"'return only JSON'); effective max_output_tokens={max_output_tokens}",
              flush=True)
    if args.input in ("image", "image_coords"):
        print("  NOTE: --model must be a VISION-capable route on the gateway for the image "
              "cell.\n        The default is a starting guess; override with --model "
              "<vision-route>.\n        (Confirm a vision route with your gateway admin / "
              "the gateway's model list;\n        a text-only route will error or ignore the "
              "image.)", flush=True)
    client = None if args.dry_run else make_client(args.base_url)
    print(flush=True)

    for split in splits:
        raw = load_raw_any(data_dir, split)
        ids = select_ids(raw, args.sample, args.seed, args.limit)
        if args.input in ("image", "image_coords"):
            try:
                chat_by_id = load_chat(data_dir, split)
            except FileNotFoundError:
                if args.schema != "v6":
                    raise
                chat_by_id = {
                    rid: {
                        "id": rid,
                        "messages": [{
                            "content": [
                                {"type": "image", "image": rec["render_path"]}
                            ]
                        }],
                    }
                    for rid, rec in raw.items()
                }
        else:
            chat_by_id = {}
        if args.input in ("image", "image_coords"):
            missing = [i for i in ids if i not in chat_by_id]
            if missing:
                raise KeyError(f"[{split}] {len(missing)} sampled ids missing from "
                               f"{split}_chat.jsonl (e.g. {missing[:5]})")

        print(f"[{split}] scoring {len(ids)} {args.input} inputs "
              f"({'dry-run' if args.dry_run else args.model}) "
              f"= {0 if args.dry_run else len(ids)} API call(s) ...", flush=True)

        scored = []
        for i, rid in enumerate(ids):
            rec = raw[rid]
            chat_row = _chat_row_for_input(chat_by_id, rid, args.input)
            payload = (build_chat_messages(
                           rec, chat_row, data_dir, args.input, args.schema, args.task,
                           cot=args.cot)
                       if args.api == "chat"
                       else build_responses_input(
                           rec, chat_row, data_dir, args.input, args.schema, args.task,
                           cot=args.cot))
            kw = build_request_kwargs(args.model, payload, args.api,
                                      max_output_tokens, args.reasoning_effort)
            if args.dry_run and i == 0:
                key = "input" if args.api == "responses" else "messages"
                print(f"\n  ---- assembled {args.api} request for id={rid} "
                      f"(base64 truncated) ----", flush=True)
                print("  " + json.dumps(_redact(kw), indent=2).replace("\n", "\n  "),
                      flush=True)
                print(f"  ---- (the '{key}' above is what is sent to the gateway) ----\n",
                      flush=True)
            out = "" if args.dry_run else ask(client, rid, kw, args.api)
            scored.append(ev.score_record(
                out,
                rec,
                task_mode=args.task if args.schema == "v6" else None,
            ))
            if not args.dry_run and (i + 1) % 25 == 0:
                print(f"  [{split}] {i + 1}/{len(ids)}", flush=True)

        agg = ev.aggregate(scored)
        if args.schema == "v6":
            agg.update({
                "gateway_schema": args.schema,
                "task": args.task,
                "input_mode": args.input,
                "model": args.model,
                "sample_seed": args.seed,
                "ids": ids,
            })
        ev.save_results(agg, f"results_{tag}_{split}{suffix}.json",
                        scored, f"records_{tag}_{split}{suffix}.jsonl")
        fmt = lambda value: "--" if value is None else f"{value:.3f}"
        print(f"[{split}] n={agg['n']} label_acc={fmt(agg['label_accuracy'])} "
              f"balanced_acc={fmt(agg['balanced_accuracy'])} parse={fmt(agg['parse_rate'])} "
              f"transform={fmt(agg['transform_match_rate'])} hint={fmt(agg['hint_match_rate'])}"
              f"  -> results_{tag}_{split}{suffix}.json", flush=True)

    if args.dry_run:
        print("\nDRY RUN complete — payload build + score/aggregate/save pipeline OK; no API "
              "calls made. Re-run without --dry-run (with TFY_API_KEY set) for real numbers.")
    else:
        print("\nSaved results_%s_{%s}.json + records_%s_{%s}.jsonl"
              % (tag, ",".join(splits), tag, ",".join(splits)))


if __name__ == "__main__":
    main()
