# v6 hint-leak fix — FAST PATH (fits well under 9h)

Same outcome as `model/HINT_FIX_RUNBOOK.md` (drop the ~96% hint leak, keep the
geometry), but engineered for a hard 9h real wall-clock budget **including SLURM
queue wait**. The three time sinks in the safe runbook are removed or shrunk:

| Safe runbook | Fast path | Why it is safe |
|---|---|---|
| Re-render v6 data with a queued CPU job (~1-3h) | Rewrite ONLY the hint text of the existing v6 tree on the **login node** (~seconds) | The fix changed only the hint STRING; every map/label/render is byte-identical. |
| Train `full` from the `both` adapter, full epoch (~2-5h each) | Continue the same `both` adapter, **capped** (`--epochs 0.5`), **both modalities concurrent in ONE `salloc`** | Geometry is already in the `both` adapter; the new hint is 1 of 8 fixed templates — trivially learned. |
| Eval `--sample 500` in a **second** queued GPU phase | Eval `--sample 200 --max-new-tokens 256`, **same allocation** (no re-queue) | The leak is deterministic per hint, so 200 shows it unambiguously; targets are ~80 tokens. |

Net structural win: the fast path needs **one** queued GPU allocation (train +
eval share it); the data step has **no** queue at all. Everything writes to
`_hintfix` names, so `results/v6_final/` and the frozen v6 adapters are untouched.

> Only real, verified flags are used below. `train_transform.py` has **no**
> `--max-steps`; the step cap is `--epochs` (float, fractional OK) and/or
> `--limit` (subsets the main `full` rows). Neither `train_transform.py` nor
> `eval_transform.py` has a device flag — they honor `CUDA_VISIBLE_DEVICES`
> (both use the default `cuda` device), which is how the two arms are pinned to
> separate GPUs in one allocation.

---

## Step 0 — Laptop: test + sync (~10 min, no queue)

```bash
cd /Users/isabellachen/projects/SLM
python3 -m pytest transform_diagnosis/ -q
python3 -m pytest model/ -q
bash -n model/sync_to_cluster.sh
bash model/sync_to_cluster.sh --dry-run
bash model/sync_to_cluster.sh
```

The sync pushes the fixed harness (`hints.py`, `dataset.py`, `eval.py`,
`v6_format.py`, ...) plus the scripts (`rewrite_hints_fast.py`,
`train_transform.py`, `eval_transform.py`) to `$HOME` and `~/transform_diagnosis`
/ `~/slm_eval`. Run everything below from `$HOME` on ORCD.

---

## Step 1 — ORCD login node (CPU, NO sbatch): rewrite hints (~seconds)

Reuse the existing `~/transform_diagnosis_data_v6` (renders reused via symlink):

```bash
module load miniforge; conda activate slm 2>/dev/null || source activate slm
cd ~
python rewrite_hints_fast.py \
  --v6-dir ~/transform_diagnosis_data_v6 \
  --out-dir ~/transform_diagnosis_data_v6_hintfix \
  --rewrite-jsonl-hints --verify
```

- Symlinks `renders_v6/`, `source_data`, and the 12 correct/student/both chat
  files unchanged; rewrites ONLY the assistant `hint` inside the 4 `full` chat
  files (train/val × image/image_coords) using the fixed `hints.hint_for()`,
  re-serialized with make_v6's exact `json.dumps(..., ensure_ascii=False)` /
  `separators=(",",":")` — byte-for-byte identical except each hint value.
- `--verify` asserts across ALL rewritten full rows: family-relevant
  (`eval._hint_mentions_family`) AND NOT `hints.is_strict_leak` AND NOT
  `eval._hint_has_leak`, prints per-file counts, and **exits nonzero on any
  leak**. Expect `strict_leak=0.000 safe_useful=1.000`.
- Full 10k-record set (20000 full-chat + 10000 jsonl rows) rewrites+verifies in
  ~2-3s (measured); budget a couple minutes with shell/import overhead.
- `--paranoid` additionally re-diffs every row to independently prove only the
  hint changed. `--copy-unchanged` copies instead of symlinking the small files.

**FALLBACK — only if `~/transform_diagnosis_data_v6` was deleted** (nothing to
reuse): regenerate with the queued CPU job from `HINT_FIX_RUNBOOK.md §1`
(`make_v6_transform_data.py --source-dir ... --out-dir
~/transform_diagnosis_data_v6_hintfix --train-n 9600 --val-n 400 --mix
0.50,0.20,0.15,0.15 --seed 20260711`). That path re-renders new pool images and
costs ~1-3h (still within budget). A `--dry-run --train-n 16 --val-n 8` smoke
runs in ~1 min. (`make_v6` also supports `--no-render` / `--max-render N` /
`--resume-existing-output`, but those leave renders pending and are not a
substitute for the reuse above.)

---

## Step 2 — ORCD: capped `full` training, BOTH modalities concurrent in ONE salloc

One interactive allocation for two GPUs = **one** queue wait for the whole GPU
phase. Continue the existing `both` adapters; write to NEW `_hintfix` dirs.

```bash
salloc -p mit_normal_gpu -G 2 -c 16 --mem=256G -t 04:00:00
# ---- inside the allocation ----
module load miniforge; conda activate slm 2>/dev/null || source activate slm
cd ~

# (a) instant CPU preflight for both arms (paths / schedule / adapter base; no GPU):
python train_transform.py --modality image --stage full \
  --data-dir ~/transform_diagnosis_data_v6_hintfix \
  --init-adapter ~/lora_adapters_v6_image_both \
  --out ~/lora_adapters_v6_image_hintfix --output-dir ~/outputs_v6_image_hintfix \
  --epochs 0.5 --dry-run
python train_transform.py --modality image_coords --stage full \
  --data-dir ~/transform_diagnosis_data_v6_hintfix \
  --init-adapter ~/lora_adapters_v6_coords_both \
  --out ~/lora_adapters_v6_coords_hintfix --output-dir ~/outputs_v6_coords_hintfix \
  --epochs 0.5 --dry-run

# (b) 1-record GPU smoke on GPU 0 (confirms model load + memory), throwaway out:
CUDA_VISIBLE_DEVICES=0 python train_transform.py --modality image --stage full \
  --data-dir ~/transform_diagnosis_data_v6_hintfix \
  --init-adapter ~/lora_adapters_v6_image_both \
  --out ~/smoke_img --output-dir ~/smoke_img_out \
  --epochs 1 --limit 1 --val-sample 1 --allow-existing-output

# (c) real runs — one per GPU, concurrent, capped at half an epoch:
CUDA_VISIBLE_DEVICES=0 nohup python train_transform.py --modality image --stage full \
  --data-dir ~/transform_diagnosis_data_v6_hintfix \
  --init-adapter ~/lora_adapters_v6_image_both \
  --out ~/lora_adapters_v6_image_hintfix --output-dir ~/outputs_v6_image_hintfix \
  --epochs 0.5 > ~/v6hf_i_full.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python train_transform.py --modality image_coords --stage full \
  --data-dir ~/transform_diagnosis_data_v6_hintfix \
  --init-adapter ~/lora_adapters_v6_coords_both \
  --out ~/lora_adapters_v6_coords_hintfix --output-dir ~/outputs_v6_coords_hintfix \
  --epochs 0.5 > ~/v6hf_c_full.log 2>&1 &
wait
```

- `--epochs 0.5` roughly halves the schedule (cosine LR still decays fully over
  the shortened run). The `full` stage auto-rehearses the earlier tasks (default
  `--rehearsal-ratio 0.15`) from the same `--data-dir`, preserving geometry —
  keep the default; do not drop it to 0.
- Target ~1-1.5h each, concurrent → ~1.5h wall for both.
- If a run is cut off, resubmit the identical command with `--resume` (checkpoints
  in `~/outputs_v6_{image,coords}_hintfix`).

**FALLBACK if the `both` adapters were deleted:** omit `--init-adapter` and add
`--base-model unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit` (the 15% rehearsal
supplies the earlier tasks in one run); see `HINT_FIX_RUNBOOK.md §2` FALLBACK.
**Fallback if under-trained** (eval shows geometry regression or weak hint):
rerun the full-epoch version (`--epochs 1`, `HINT_FIX_RUNBOOK.md §2`), ~+1.5h.

---

## Step 3 — Same allocation: trimmed eval, both arms concurrent (no re-queue)

Eval reads the FROZEN `~/transform_diagnosis_data` test/ood (only training data
changed). Same `--seed 20260709` → both arms score the SAME ids (image vs coords
paired).

```bash
# still inside the salloc, after training finishes:
CUDA_VISIBLE_DEVICES=0 nohup python eval_transform.py --input image --task full \
  --data-dir ~/transform_diagnosis_data \
  --adapter ~/lora_adapters_v6_image_hintfix \
  --sample 200 --seed 20260709 --max-new-tokens 256 \
  --tag v6_4b_image_hintfix > ~/v6hf_i_eval.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python eval_transform.py --input image_coords --task full \
  --data-dir ~/transform_diagnosis_data \
  --adapter ~/lora_adapters_v6_coords_hintfix \
  --sample 200 --seed 20260709 --max-new-tokens 256 \
  --tag v6_4b_image_coords_hintfix > ~/v6hf_c_eval.log 2>&1 &
wait
exit   # release the allocation
```

- `--sample`: **the** eval speed lever (linear in generations); 200 halves+ the
  work vs 500 and still cleanly shows the leak drop and geometry hold.
- `--max-new-tokens 256`: greedy decode already stops at EOS (~80-token targets),
  so this mainly caps runaway/non-EOS stragglers — a safe trim from the wasteful
  default 512, near-zero truncation risk (and `eval.parse_pred` takes the last
  JSON object anyway).
- Writes `results_v6_4b_{image,image_coords}_hintfix_{test,ood}.json` + matching
  `records_*.jsonl` to `$HOME`.
- Optional if time remains: rerun `--sample 500` (strictly matches the frozen
  500-run distribution) — the leak conclusion will not change.

---

## Step 4 — ORCD login node (CPU): confirm the leak dropped (~minutes)

Headline metric + the strict audit port (reuses `hints.is_strict_leak`). This is
the same snippet as `HINT_FIX_RUNBOOK.md §4b`:

```bash
cd ~
python - <<'PY'
import json, os
from transform_diagnosis import eval as ev, hints
HOME = os.path.expanduser("~")
oracle = {}
for split in ("test", "ood"):
    for line in open(f"{HOME}/transform_diagnosis_data/{split}.jsonl"):
        r = json.loads(line); oracle[(split, r["id"])] = r
for tag in ("v6_4b_image_hintfix", "v6_4b_image_coords_hintfix"):
    n = fam = leak = safe = 0
    for split in ("test", "ood"):
        for line in open(f"{HOME}/records_{tag}_{split}.jsonl"):
            row = json.loads(line); rec = oracle[(split, row["id"])]; label = rec["label"]
            pred = ev.parse_pred(row.get("raw_model_output") or "") or {}
            hint = pred.get("hint") if isinstance(pred.get("hint"), str) else ""
            n += 1
            fam_ok = bool(hint.strip()) and ev._hint_mentions_family(hint, hints.expected_hint_families(label, rec))
            leaked = (not hint.strip()) or hints.is_strict_leak(hint, label, rec)
            fam += fam_ok; leak += leaked; safe += (fam_ok and not leaked)
    print(f"{tag}: n={n} family_relevant={fam/n:.3f} strict_leak={leak/n:.3f} safe_useful={safe/n:.3f}")
PY
```

Expect `strict_leak` ≈ 0.0 (was ~0.96) and `safe_useful` ≈ 1.0 (was ~0.014),
with geometry (`both_nets_match_rate`, `label_accuracy`) ~unchanged vs
`results/v6_final/`. Optional deeper checks: `HINT_FIX_RUNBOOK.md §4a/§4c`.

---

## Wall-clock budget (hard cap 9h, includes queue wait)

| # | Step | Where / parallelism | Queue | Run | Cum. |
|---|---|---|---|---|---|
| 0 | pytest + rsync sync | laptop | — | 0.2h | 0.2h |
| 1 | hint rewrite + verify | ORCD login CPU (no sbatch) | 0 | ~0.1h | ~0.3h |
| — | acquire 1× `salloc -G 2` | queue | ≤1.5h | — | ≤1.8h |
| 2 | train `full` ×2 (`--epochs 0.5`) | 2 GPUs concurrent, in salloc | 0 | ~1.5h | ≤3.3h |
| 3 | eval ×2 (`--sample 200`) | 2 GPUs concurrent, same salloc | 0 | ~0.75h | ≤4.05h |
| 4 | leak verify (§4b) | ORCD login CPU | 0 | ~0.2h | ≤4.25h |

- **Typical ≈ 3.5h; pessimistic ≈ 4.25h** (1.5h queue + capped) → **~4.75h
  margin** under 9h.
- Even the all-fallbacks case (regen ~2h + full-epoch train ~3h + `--sample 500`
  eval ~1.5h + ~2h queue) ≈ ~8.5h, still under 9h but tight — prefer the fast
  levers and keep the fallbacks in reserve.

### Riskiest time-savers → safe fallback (one line each)

1. **Reuse renders (no regen)** — breaks only if `~/transform_diagnosis_data_v6`
   is gone → Step 1 FALLBACK: `make_v6_transform_data.py` regen (~1-3h).
2. **`--epochs 0.5` cap** — risk of under-training/geometry drift → rerun
   `--epochs 1` (or `--resume`), `HINT_FIX_RUNBOOK.md §2` (~+1.5h).
3. **`--sample 200` eval** — noisier geometry estimate → rerun `--sample 500`
   (leak drop is deterministic, so 200 already proves it).
4. **Old user-prompt wording retained** — the fast rewrite changes ONLY the
   assistant target hint; the user turn still shows the pre-fix
   `"hint":"<short Socratic hint>"` schema line (eval uses the new wording). The
   model learns the coordinate-free hint from the TARGET, so this is cosmetic →
   if you want the prompt refreshed too, use the full `make_v6` regen (Step 1
   FALLBACK).
5. **`CUDA_VISIBLE_DEVICES` device pinning** — valid for the current code (no
   device flag; default `cuda`) → if a future version hardcodes a device, fall
   back to two separate `sbatch` GPU jobs (`HINT_FIX_RUNBOOK.md §2`, two queue
   waits, still fits).
6. **`--max-new-tokens 256`** — truncation risk on a verbose output → raise back
   to 512 (targets are ~80 tokens, so risk is near zero).

See `model/HINT_FIX_RUNBOOK.md` for the full, maximally-safe version of every
step.
