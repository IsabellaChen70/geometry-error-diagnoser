# Golden-set model comparison (base vs tuned vs hintfix vs frontier)

A single, copy-pasteable command sheet that scores **six of your own model cells**
(BASE / TUNED / HINTFIX x {image, image+coords}) **and three frontier models**
(gpt / sonnet / gemini x {image, image+coords}) on the **held-out golden set**
(`n=160`, provably disjoint from every train/val/test/ood split — see
[`GOLDEN_SET.md`](GOLDEN_SET.md)), then puts confidence intervals on everything.

- Your models run on the **GPU** via [`eval_transform.py`](eval_transform.py).
- Frontier models run on the **login node / API** (no GPU, no SLURM queue) via
  [`eval_frontier_gateway.py`](eval_frontier_gateway.py) through the TrueFoundry
  OpenAI-compatible gateway (`TFY_API_KEY`).
- Confidence intervals come from [`confidence_intervals.py`](confidence_intervals.py)
  (CPU, stdlib-only) — see [`CONFIDENCE.md`](CONFIDENCE.md).

Everything runs from `$HOME` on ORCD (`orcd-login.mit.edu`, user `ikchen`). The frozen
`~/transform_diagnosis_data` stays read-only; nothing here touches `results/v6_final/`
or any adapter. All result/record files land in `$HOME`.

> **Fill in the gateway routes yourself.** Every `--model '<...>'` below is a
> **placeholder**. Substitute a route you have **verified is vision-capable** on your
> gateway (via your gateway admin / the gateway model list). Do **not** paste a guessed
> route — a text-only route will error or silently ignore the image. Verify each with the
> one-call smoke in §3 before spending the full 160 calls.

---

## Cheat-sheet: exact output files this run produces

All written to `$HOME`. `eval_transform.py` and `eval_frontier_gateway.py` both build the
filename as `results_<tag>_<split>.json` / `records_<tag>_<split>.jsonl` — the split
(`golden`) is **auto-appended**, so no tag contains the word `golden`.

**GPU cells (`eval_transform.py`, 6):**

| arm | modality | `--tag` | results file | records file |
|---|---|---|---|---|
| base | image | `v6_4b_image_base` | `results_v6_4b_image_base_golden.json` | `records_v6_4b_image_base_golden.jsonl` |
| tuned | image | `v6_4b_image_tuned` | `results_v6_4b_image_tuned_golden.json` | `records_v6_4b_image_tuned_golden.jsonl` |
| hintfix | image | `v6_4b_image_hintfix` | `results_v6_4b_image_hintfix_golden.json` | `records_v6_4b_image_hintfix_golden.jsonl` |
| base | image+coords | `v6_4b_image_coords_base` | `results_v6_4b_image_coords_base_golden.json` | `records_v6_4b_image_coords_base_golden.jsonl` |
| tuned | image+coords | `v6_4b_image_coords_tuned` | `results_v6_4b_image_coords_tuned_golden.json` | `records_v6_4b_image_coords_tuned_golden.jsonl` |
| hintfix | image+coords | `v6_4b_image_coords_hintfix` | `results_v6_4b_image_coords_hintfix_golden.json` | `records_v6_4b_image_coords_hintfix_golden.jsonl` |

**Frontier cells (`eval_frontier_gateway.py`, 6):** the gateway uses `--tag` **verbatim**
and appends only the split (it does **not** add the modality), so each (model, modality)
needs its **own** tag or the image+coords run overwrites the image run:

| model | modality | `--tag` | results file | records file |
|---|---|---|---|---|
| gpt | image | `frontier_gpt_image` | `results_frontier_gpt_image_golden.json` | `records_frontier_gpt_image_golden.jsonl` |
| gpt | image+coords | `frontier_gpt_image_coords` | `results_frontier_gpt_image_coords_golden.json` | `records_frontier_gpt_image_coords_golden.jsonl` |
| sonnet | image | `frontier_sonnet_image` | `results_frontier_sonnet_image_golden.json` | `records_frontier_sonnet_image_golden.jsonl` |
| sonnet | image+coords | `frontier_sonnet_image_coords` | `results_frontier_sonnet_image_coords_golden.json` | `records_frontier_sonnet_image_coords_golden.jsonl` |
| gemini | image | `frontier_gemini_image` | `results_frontier_gemini_image_golden.json` | `records_frontier_gemini_image_golden.jsonl` |
| gemini | image+coords | `frontier_gemini_image_coords` | `results_frontier_gemini_image_coords_golden.json` | `records_frontier_gemini_image_coords_golden.jsonl` |

---

## 0. Laptop: sync code to the cluster

`GOLDEN_COMPARISON.md` (this file), the eval scripts, the golden generator, and
`confidence_intervals.py` are all in `SCRIPT_FILES` in
[`sync_to_cluster.sh`](sync_to_cluster.sh), so one push carries everything:

```bash
cd /Users/isabellachen/projects/SLM
bash -n model/sync_to_cluster.sh          # syntax check
bash model/sync_to_cluster.sh --dry-run   # show exactly what WOULD transfer
bash model/sync_to_cluster.sh             # push (code only; no --delete, no data/adapters)
```

The sync pushes the harness to `~/slm_eval` + `~/transform_diagnosis` and the scripts +
docs to `$HOME`. It is code-only and idempotent.

---

## 1. Cluster: (re)generate + render the golden set

Deterministic (seed `20260712`), CPU-only, ~30s incl. rendering — the login node is fine.
This is the exact command from [`GOLDEN_SET.md`](GOLDEN_SET.md) §(a):

```bash
module load miniforge
conda activate slm 2>/dev/null || source activate slm
cd ~

python make_golden_set.py \
  --out-dir ~/transform_diagnosis_data_golden \
  --source-dir ~/transform_diagnosis_data \
  --seed 20260712 --n-per-label 20

python make_golden_set.py --verify \
  --out-dir ~/transform_diagnosis_data_golden \
  --source-dir ~/transform_diagnosis_data      # expect RESULT: PASS
```

Confirm the data dir is a complete `--data-dir` (records + chat rows + renders):

```bash
ls ~/transform_diagnosis_data_golden
#   golden_v6.jsonl  golden_v6_image_full_chat.jsonl
#   golden_v6_image_coords_full_chat.jsonl  manifest_golden.json  README.md  renders_v6/
ls ~/transform_diagnosis_data_golden/renders_v6/golden | head   # 000000.png ... 000159.png
wc -l ~/transform_diagnosis_data_golden/golden_v6.jsonl         # 160
```

> Deterministic alternative: the local `dataset_golden_v6/` is byte-identical and already
> rendered, so instead of regenerating you may copy it up:
> `rsync -a dataset_golden_v6/ ikchen@orcd-login.mit.edu:~/transform_diagnosis_data_golden/`

---

## 2. GPU evals — base / tuned / hintfix x {image, image+coords} (6 cells)

`task=full`, all 160 golden records (`--sample 0`), both modalities. n=160 is small so each
cell is a short single-GPU run. Adapters: base uses `--base-only` (no adapter); tuned uses
the frozen `~/lora_adapters_v6_*`; hintfix uses `~/lora_adapters_v6_*_hintfix`.

> `--sample 0` scores **all 160** records (stable-sorted ids). At `--sample 0` the `--seed`
> does not change *which* records are scored (all of them); it is set only for parity with
> the repo's standard eval seed. Every cell scores the same 160 ids, so all arms are paired.

```bash
GOLD=~/transform_diagnosis_data_golden
COMMON='-p mit_normal_gpu -G 1 -c 8 --mem=64G -t 02:00:00'
PRE='module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~;'

# ---- image-only arm ---------------------------------------------------------
sbatch -J g_img_base $COMMON -o g_img_base_%j.log --wrap "$PRE python eval_transform.py \
  --input image --task full --data-dir $GOLD --splits golden --sample 0 --seed 20260709 \
  --base-only --tag v6_4b_image_base"

sbatch -J g_img_tuned $COMMON -o g_img_tuned_%j.log --wrap "$PRE python eval_transform.py \
  --input image --task full --data-dir $GOLD --splits golden --sample 0 --seed 20260709 \
  --adapter ~/lora_adapters_v6_image --tag v6_4b_image_tuned"

sbatch -J g_img_hintfix $COMMON -o g_img_hintfix_%j.log --wrap "$PRE python eval_transform.py \
  --input image --task full --data-dir $GOLD --splits golden --sample 0 --seed 20260709 \
  --adapter ~/lora_adapters_v6_image_hintfix --tag v6_4b_image_hintfix"

# ---- image + coordinates arm ------------------------------------------------
sbatch -J g_crd_base $COMMON -o g_crd_base_%j.log --wrap "$PRE python eval_transform.py \
  --input image_coords --task full --data-dir $GOLD --splits golden --sample 0 --seed 20260709 \
  --base-only --tag v6_4b_image_coords_base"

sbatch -J g_crd_tuned $COMMON -o g_crd_tuned_%j.log --wrap "$PRE python eval_transform.py \
  --input image_coords --task full --data-dir $GOLD --splits golden --sample 0 --seed 20260709 \
  --adapter ~/lora_adapters_v6_coords --tag v6_4b_image_coords_tuned"

sbatch -J g_crd_hintfix $COMMON -o g_crd_hintfix_%j.log --wrap "$PRE python eval_transform.py \
  --input image_coords --task full --data-dir $GOLD --splits golden --sample 0 --seed 20260709 \
  --adapter ~/lora_adapters_v6_coords_hintfix --tag v6_4b_image_coords_hintfix"
```

**Produces** (in `$HOME`): the six `results_v6_4b_*_golden.json` + `records_v6_4b_*_golden.jsonl`
pairs listed in the GPU cheat-sheet table above.

> **One-GPU alternative.** These are tiny, so you can run all six on a single interactive
> allocation instead of six jobs — or grab two GPUs and run the arms concurrently:
>
> ```bash
> salloc -p mit_normal_gpu -G 2 -c 16 --mem=128G -t 02:00:00
> module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~
> GOLD=~/transform_diagnosis_data_golden
> # then run the six `python eval_transform.py ...` bodies above (drop the sbatch/--wrap wrapper),
> # e.g. backgrounding the image arm on GPU 0 and the coords arm on GPU 1.
> ```

Optional live 6-cell table once the jobs finish (login node, no GPU): see
[`GOLDEN_SET.md`](GOLDEN_SET.md) §(c) for the ready-made reader snippet.

---

## 3. Frontier evals — gpt / sonnet / gemini x {image, image+coords} (gateway)

These call the API (login node, **no GPU, no SLURM queue**), so they run **in parallel with
the GPU jobs** in §2. Each model is ~15-20 min for 160 image calls; **run the three models
concurrently in three separate shells/`tmux` panes**. `--schema v6 --task full` makes the
prompt + scoring byte-for-byte the same harness the GPU cells use, so the cells stay
comparable. The gateway loads the golden split from `--data-dir` (`golden_v6.jsonl`) and,
because there is no `golden_chat.jsonl`, resolves each image from the record's `render_path`
under the data dir — this fallback only exists under `--schema v6`, so keep `--schema v6`.

```bash
export TFY_API_KEY='tfy_...'          # your one gateway key (vision routes enabled)
GOLD=~/transform_diagnosis_data_golden
cd ~
```

### 3a. Verify each vision route FIRST (do this before the full runs)

For **every** route you plan to use, run a no-cost dry-run (payload/plumbing only, no key
needed) and then a single real call. This is the route-verification pattern from
[`V6_TRANSFORM_RUNBOOK.md`](V6_TRANSFORM_RUNBOOK.md) §5, pointed at the golden data dir:

```bash
# (i) dry-run: builds the base64 image payload + prompt, no API call, no key required
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits golden --data-dir $GOLD --sample 1 --limit 1 --seed 20260709 --dry-run \
  --model '<PASTE-VERIFIED-VISION-ROUTE>'

# (ii) one REAL call: confirms the route actually accepts an image and returns text
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits golden --data-dir $GOLD --sample 1 --limit 1 --seed 20260709 \
  --model '<PASTE-VERIFIED-VISION-ROUTE>' --tag frontier_smoke
#   -> writes results_frontier_smoke_golden.json (n=1, disposable). If this errors on the
#      image, the route is not vision-capable (pick another) OR the gateway wants the
#      Chat-Completions shape: add `--api chat`. If it rejects the reasoning param, add
#      `--reasoning-effort none`. Delete the smoke file before the CI step: rm results_frontier_smoke_golden.json records_frontier_smoke_golden.jsonl
```

### 3b. Full frontier runs (all 160), one block per model

Substitute each `<PASTE-...-VISION-ROUTE>` with the verified route for that model. Add
`--api chat` and/or `--reasoning-effort none` here too if the smoke told you the route needs
them.

```bash
# ===== GPT (shell 1) =========================================================
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model '<PASTE-GPT-VISION-ROUTE>' --tag frontier_gpt_image
python eval_frontier_gateway.py --schema v6 --input image_coords --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model '<PASTE-GPT-VISION-ROUTE>' --tag frontier_gpt_image_coords

# ===== Sonnet (shell 2) ======================================================
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model '<PASTE-SONNET-VISION-ROUTE>' --tag frontier_sonnet_image
python eval_frontier_gateway.py --schema v6 --input image_coords --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model '<PASTE-SONNET-VISION-ROUTE>' --tag frontier_sonnet_image_coords

# ===== Gemini (shell 3) ======================================================
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model '<PASTE-GEMINI-VISION-ROUTE>' --tag frontier_gemini_image
python eval_frontier_gateway.py --schema v6 --input image_coords --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model '<PASTE-GEMINI-VISION-ROUTE>' --tag frontier_gemini_image_coords
```

**Produces** (in `$HOME`): the six `results_frontier_*_golden.json` +
`records_frontier_*_golden.jsonl` pairs in the frontier cheat-sheet table above.

> Gateway knobs you may need to set per route (defaults shown; only change if the smoke
> fails): `--base-url` defaults to `https://tfy-eu.promptlens.trilogy.com` (the promptlens
> EU gateway — change if your tenant differs); `--api responses` is the default, `--api chat`
> is the Chat-Completions vision fallback; `--reasoning-effort high` is the default, use
> `none` for routes that reject a reasoning directive; `--max-output-tokens 8000` is generous
> headroom for a reasoning model.

---

## 4. Confidence intervals (login node, CPU — no GPU/API)

Run from `$HOME` after §2 and §3 finish. Reads the saved files **read-only**; writes nothing
unless you pass `--json-out`. See [`CONFIDENCE.md`](CONFIDENCE.md).

### 4a. Wilson 95% CI for every rate in every golden cell

```bash
cd ~
python confidence_intervals.py results_v6_4b_*_golden.json       # 6 GPU cells
python confidence_intervals.py results_frontier_*_golden.json    # 6 frontier cells
# (if you left a smoke file around it will show as n=1 — delete it or ignore that row)
```

### 4b. Paired comparisons on the SAME 160 golden ids (records files)

`--pair A B` reports `delta = rate(B) - rate(A)`, McNemar's test, and a bootstrap delta CI.

**(1) Learned-behavior jump: base -> tuned** (expect large, significant positive deltas):

```bash
python confidence_intervals.py \
  --pair records_v6_4b_image_base_golden.jsonl         records_v6_4b_image_tuned_golden.jsonl \
  --pair records_v6_4b_image_coords_base_golden.jsonl  records_v6_4b_image_coords_tuned_golden.jsonl \
  --metric both_nets_ok --metric correct_net_ok --metric student_net_ok --metric label_ok
```

**(2) Does hintfix keep geometry within the CI of tuned?** (expect delta ~ 0 with a CI that
**straddles 0** and a **non-significant** McNemar on the geometry metrics; `hint_ok` may
move because that is the text the fix changed):

```bash
python confidence_intervals.py \
  --pair records_v6_4b_image_tuned_golden.jsonl        records_v6_4b_image_hintfix_golden.jsonl \
  --pair records_v6_4b_image_coords_tuned_golden.jsonl records_v6_4b_image_coords_hintfix_golden.jsonl \
  --metric correct_net_ok --metric student_net_ok --metric both_nets_ok \
  --metric label_ok --metric derived_label_ok --metric hint_ok
```

Read "within CI of tuned" as: the tuned->hintfix geometry delta CI **includes 0** and
McNemar is **not** significant (the maps/labels are unchanged; only the hint text differs).

**(3) Hintfix vs each frontier — the fair geometry comparison (`both_nets_ok`).** Order is
A=frontier, B=hintfix, so `delta = rate(hintfix) - rate(frontier)` and a **positive** delta
means your model leads. image+coords is the hero arm; the image block is below it:

```bash
# hero arm: image+coords
python confidence_intervals.py \
  --pair records_frontier_gpt_image_coords_golden.jsonl     records_v6_4b_image_coords_hintfix_golden.jsonl \
  --pair records_frontier_sonnet_image_coords_golden.jsonl  records_v6_4b_image_coords_hintfix_golden.jsonl \
  --pair records_frontier_gemini_image_coords_golden.jsonl  records_v6_4b_image_coords_hintfix_golden.jsonl \
  --metric both_nets_ok

# image-only arm
python confidence_intervals.py \
  --pair records_frontier_gpt_image_golden.jsonl     records_v6_4b_image_hintfix_golden.jsonl \
  --pair records_frontier_sonnet_image_golden.jsonl  records_v6_4b_image_hintfix_golden.jsonl \
  --pair records_frontier_gemini_image_golden.jsonl  records_v6_4b_image_hintfix_golden.jsonl \
  --metric both_nets_ok
```

> Prefer `records_*.jsonl` for the paired step (per-record booleans are required to pair).
> The Wilson step accepts either; `results_*.json` recovers `k = round(rate*n)`, which is
> exact at n=160.

---

## 5. Results table (n=160, golden set)

Exact two-map geometry (`both_nets`) is the headline; `label_acc` and `hint` are secondary
and carry the caveats below. Rates read off the §4a Wilson output; 95% CIs live in
`results/v6_golden/*.json`. (sonnet/gemini gateway routes could not be scored in budget.)

| model / arm | image `both_nets` | image+coords `both_nets` | label_acc (image+coords) | hint (image+coords) | source files (image / image+coords) |
|---|---|---|---|---|---|
| base    | 0.0% | 0.0% | 25.6% | 44.4% | `results_v6_4b_image_base_golden.json` / `results_v6_4b_image_coords_base_golden.json` |
| tuned   | 36.2% | 98.75% | 99.4% | 96.2% | `results_v6_4b_image_tuned_golden.json` / `results_v6_4b_image_coords_tuned_golden.json` |
| hintfix | 33.75% | 98.75% | 99.4% | 100.0% | `results_v6_4b_image_hintfix_golden.json` / `results_v6_4b_image_coords_hintfix_golden.json` |
| gpt-4o  | 0.0% | 0.6% | 30.6% | 67.5% | `results_frontier_gpt_image_golden.json` / `results_frontier_gpt_image_coords_golden.json` |
| sonnet  | — | — | — | — | not scored — gateway returned empty completions |
| gemini  | — | — | — | — | not scored — gateway ~1 min/call, low parse |

Metric keys per cell: `both_nets` = `both_nets_match_rate`; `label_acc` = `label_accuracy`;
`hint` = `hint_match_rate` (family-relevant + no coordinate leak). Add a `hint_exact_match_rate`
column too if you want to show the leak collapse (hintfix is ~0 there **by design** — the gold
hint no longer contains the exact answer tokens).

### Honesty caveats (keep these attached to the table)

- **Lead with the geometry (`both_nets`) comparison — it is the fair one.** All cells are
  scored by the identical `full`/`v6.net-affine.1` harness on the identical 160 records, so
  exact two-map correctness is directly comparable across your models and the frontier models.
- **Frontier `label_accuracy` is apples-to-oranges.** The frontier models are **zero-shot on
  the bespoke 8-label diagnosis taxonomy**; they were never shown the label vocabulary or the
  Socratic hint schema, so their `label_acc` and `hint` numbers understate them and should not
  be read as a like-for-like classification result. Do not headline them — use them only as
  context beside the geometry map comparison.
- **CIs cover eval-sampling variance, not training-seed variance.** Wilson + McNemar + the
  bootstrap quantify finite-**sample** (evaluation) uncertainty at a **fixed** checkpoint and
  training seed. A different fine-tune of the same recipe could land outside them.
- **n=160 -> wider intervals** than the 500-case test/ood cells in `results/v6_final/`; expect
  visibly wider bars. An observed 100% is not proof of perfection: for 160/160 the 95% Wilson
  lower bound is ~97.6%.
- **`hint_exact_match_rate` ~ 0 for hintfix is intended**, not a regression — that field
  measures exact-answer disclosure, which the hint-leak fix deliberately removed while keeping
  `hint_match_rate` (family relevance) high. See [`HINT_FIX_RUNBOOK.md`](HINT_FIX_RUNBOOK.md).

## 6. CoT bulletproofing — does letting GPT-4o *reason* close the gap?

A skeptic can object that §3 ran the frontier models in **direct-output** mode: the v6 prompt
ends with *"Return one valid JSON object and nothing else."*, leaving no room to think. This
section re-runs **GPT-4o with an explicit chain-of-thought (CoT)** and scores it with the
**same oracle, metrics, golden ids, and schema** — the *only* thing that changes is the
frontier **prompt**. So GPT-CoT is directly comparable to GPT-direct (§3) and to every model
cell (`both_nets` etc.).

**What `--cot` does** (lives ONLY in [`eval_frontier_gateway.py`](eval_frontier_gateway.py);
`transform_diagnosis/v6_format.py` and the scoring in `transform_diagnosis/eval.py` are
untouched, so the tuned/base/hintfix cells stay byte-comparable): it appends a directive
*after* the built schema prompt that (a) explicitly **overrides** *"return only JSON and
nothing else"* and (b) requires the model to reason step by step and then emit the **same**
final JSON object as the **last** thing in the reply. `transform_diagnosis.eval.parse_pred`
keeps the **last** brace-balanced JSON object, so "reasoning … {final JSON}" parses and scores
byte-identically to the direct path. `--cot` also floors `--max-output-tokens` at 4000 (the
default 8000 already clears this) so reasoning + JSON fit. The CoT is **prompt-driven**, so it
works on non-reasoning routes like gpt-4o regardless of `--reasoning-effort`. Without `--cot`
the direct path is unchanged.

> Verified offline before shipping (no API/GPU): a `--cot --dry-run` shows both the schema
> **and** the CoT directive in the payload; `eval.parse_pred` extracts the final JSON from a
> reasoning-then-fenced-JSON reply (and ignores an earlier scratch object); `pytest
> transform_diagnosis/ model/` stays green. The actual GPT-CoT **accuracy** still requires the
> gateway run below.

**Output files this section adds** (distinct tags so they never clobber the §3 direct files):

| model | modality | `--tag` | results file | records file |
|---|---|---|---|---|
| gpt-CoT | image | `frontier_gpt_cot_image` | `results_frontier_gpt_cot_image_golden.json` | `records_frontier_gpt_cot_image_golden.jsonl` |
| gpt-CoT | image+coords | `frontier_gpt_cot_image_coords` | `results_frontier_gpt_cot_image_coords_golden.json` | `records_frontier_gpt_cot_image_coords_golden.jsonl` |

### 6a. Laptop: push the updated gateway

`eval_frontier_gateway.py` is already in `SCRIPT_FILES`, so the standard sync carries the
`--cot` change (code only; no data/adapters/results):

```bash
cd /Users/isabellachen/projects/SLM
bash model/sync_to_cluster.sh
```

### 6b. Cluster (login node): GPT-4o **with `--cot`**, both modalities

Use the **same** verified GPT-4o vision route + flags you used for the direct run in §3b (here
`openai-group/gpt-4o`, `--reasoning-effort none` since gpt-4o is not a reasoning model), and
**add only `--cot`** plus a distinct `--tag`. Everything else (`--data-dir $GOLD`, `--splits
golden`, `--sample 0`, `--seed 20260709`, `--schema v6`, `--task full`) is identical to §3, so
all 160 golden ids stay paired and the CoT-vs-direct delta isolates *reasoning*.

```bash
export TFY_API_KEY='tfy_...'          # your gateway key (vision routes enabled)
GOLD=~/transform_diagnosis_data_golden
cd ~

# image-only (pane 1)
nohup python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model 'openai-group/gpt-4o' --reasoning-effort none --cot --max-output-tokens 8000 \
  --tag frontier_gpt_cot_image > cot_gpt_image.log 2>&1 &

# image+coords (pane 2) — the hero arm
nohup python eval_frontier_gateway.py --schema v6 --input image_coords --task full \
  --splits golden --data-dir $GOLD --sample 0 --seed 20260709 \
  --model 'openai-group/gpt-4o' --reasoning-effort none --cot --max-output-tokens 8000 \
  --tag frontier_gpt_cot_image_coords > cot_gpt_image_coords.log 2>&1 &
```

> **Timing.** gpt-4o was fast in §3 (~5-15 s/call), so 160 calls ≈ **15-25 min per modality**.
> The two modalities are independent API loops — run them concurrently. `nohup … &` keeps them
> alive if SSH drops (`tail -f cot_gpt_image*.log` to watch, `wait` to block). Match whatever
> `--api` / `--reasoning-effort` your §3 GPT run needed so CoT is the only change. `8000` tokens
> is ample (the JSON answer is tiny); if you ever see parse-fail rows from replies truncated
> before the final JSON, raise it (e.g. `--max-output-tokens 12000`) and re-run.

### 6c. Cluster CI (after both finish): did reasoning rescue GPT-4o?

All CPU/stdlib, read-only. Wilson CIs on the two new cells, then the two paired questions on
the shared 160 golden ids (`--pair A B` reports `delta = rate(B) − rate(A)`).

```bash
cd ~
python confidence_intervals.py results_frontier_gpt_cot_image_golden.json \
                               results_frontier_gpt_cot_image_coords_golden.json
```

**(1) How much did CoT lift GPT?** A=direct, B=CoT ⇒ `delta = CoT − direct`. Expect a small
delta whose CI **straddles 0** — reasoning does not manufacture exact integer maps:

```bash
# hero arm: image+coords
python confidence_intervals.py \
  --pair records_frontier_gpt_image_coords_golden.jsonl records_frontier_gpt_cot_image_coords_golden.jsonl \
  --metric both_nets_ok --metric correct_net_ok
# image-only arm
python confidence_intervals.py \
  --pair records_frontier_gpt_image_golden.jsonl records_frontier_gpt_cot_image_golden.jsonl \
  --metric both_nets_ok --metric correct_net_ok
```

**(2) Does the fine-tune still win big even against GPT-CoT?** A=CoT, B=hintfix ⇒ `delta =
hintfix − CoT`. Expect a **large, significant, positive** delta:

```bash
# hero arm: image+coords
python confidence_intervals.py \
  --pair records_frontier_gpt_cot_image_coords_golden.jsonl records_v6_4b_image_coords_hintfix_golden.jsonl \
  --metric both_nets_ok
# image-only arm
python confidence_intervals.py \
  --pair records_frontier_gpt_cot_image_golden.jsonl records_v6_4b_image_hintfix_golden.jsonl \
  --metric both_nets_ok
```

> **`--metric` takes ONE value per flag** (it is `action="append"`); to score several metrics
> **repeat the flag** — `--metric both_nets_ok --metric correct_net_ok`, NOT `--metric
> both_nets_ok correct_net_ok` (argparse would mis-read the second token as a stray input path).

### 6d. Honest framing for the writeup

> Even when it is allowed to reason step by step (CoT), zero-shot GPT-4o scores **X%** both-maps
> on the held-out golden set (n=160) versus the fine-tune's **98.75%** — and the paired McNemar
> stays significant. The gap is a genuine **exact-computation** gap, not a prompting artifact:
> the *only* difference between GPT-direct and GPT-CoT is the prompt (identical oracle, metrics,
> ids, and schema), and reasoning room did not close it.

*(Fill in **X%** from the §6c Wilson `both_nets_match_rate` on
`results_frontier_gpt_cot_image_coords_golden.json`. This number needs the gateway run in §6b —
it cannot be produced offline.)*

---

## Related docs

- Golden set (disjointness proof + regeneration): [`GOLDEN_SET.md`](GOLDEN_SET.md)
- v6 curriculum + training/eval + frontier §5: [`V6_TRANSFORM_RUNBOOK.md`](V6_TRANSFORM_RUNBOOK.md)
- Hint-leak fix + hintfix adapters: [`HINT_FIX_RUNBOOK.md`](HINT_FIX_RUNBOOK.md)
- Confidence-interval tool: [`CONFIDENCE.md`](CONFIDENCE.md)
- Frozen base-vs-tuned results + honesty rail: [`results/v6_final/FINAL_RESULTS.md`](../results/v6_final/FINAL_RESULTS.md)
