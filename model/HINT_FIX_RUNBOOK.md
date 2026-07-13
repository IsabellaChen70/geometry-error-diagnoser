# v6 hint-leak fix runbook

## What changed and why

The frozen v6 tuned model reproduced the exact answer inside its tutoring `hint`
~96% of the time (see `results/overnight/HINT_SAFETY_AUDIT.md`, "any exact
answer/map/value disclosure: 1919/2000 (96.0%)"). Root cause was in the DATA, not
the model: `transform_diagnosis/hints.py` built the gold hint by embedding the
exact `expected_hint_tokens` (the canonical axis/angle/translation strings), and
`dataset._assert_record` *required* those tokens to be present. So the model was
trained to state the answer in the hint.

The code fix (already committed on your laptop, verified by the test suites)
flips the contract so every gold hint is a coordinate-free Socratic nudge:

- `transform_diagnosis/hints.py` — `hint_for()` now returns one concept-only
  template per label (names the operation family and which aspect is wrong; no
  coordinates/axes/angles/translation values). Adds `is_strict_leak()` /
  `strict_leak_reasons()`, a faithful port of `results/overnight/audit_hint_safety.py`.
- `transform_diagnosis/dataset.py` — `_assert_record` now requires the hint to be
  family-relevant (`eval._hint_mentions_family`) AND not a strict leak, instead of
  requiring the exact answer tokens.
- `transform_diagnosis/v6_format.py` — the `full` prompt hint schema now says
  "naming the kind of mistake WITHOUT stating any coordinates, axes, angles, or
  translation values".

The geometry (correct_net/student_net/label and their scoring) is unchanged; only
the hint text differs. Regenerating with the SAME seed/mix/train-n/val-n produces
byte-identical maps/labels and only new hint strings, so geometry metrics should
land ~unchanged while the hint leak rate drops toward 0.

Everything below runs on ORCD (`orcd-login.mit.edu`, user `ikchen`). The frozen
source `~/transform_diagnosis_data` stays read-only; new data goes to a NEW
directory and new adapters/results use `_hintfix` names so nothing in
`results/v6_final` or the original v6 adapters is touched.

---

## 0. Laptop: run tests, then sync the fixed code

```bash
cd /Users/isabellachen/projects/SLM
python3 -m pytest transform_diagnosis/ -q
python3 -m pytest model/ -q
bash -n model/sync_to_cluster.sh
bash model/sync_to_cluster.sh --dry-run
bash model/sync_to_cluster.sh
```

The sync is code-only (no `--delete`); it pushes the fixed harness (`hints.py`,
`dataset.py`, `eval.py`, `net_transform.py`, `v6_format.py`, ...) to both
`~/slm_eval` and `~/transform_diagnosis` and the scripts to `$HOME`. Run every
remaining command from `$HOME` on ORCD.

---

## 1. ORCD CPU job: regenerate v6 data (hint text only) to a NEW dir

Same seed / mix / sizes as the original v6 run, so only the hints change:

```bash
sbatch -J v6hf_data -p mit_normal -c 8 --mem=32G -t 03:00:00 -o v6hf_data_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python make_v6_transform_data.py --source-dir ~/transform_diagnosis_data --out-dir ~/transform_diagnosis_data_v6_hintfix --train-n 9600 --val-n 400 --mix 0.50,0.20,0.15,0.15 --seed 20260711'
```

- Reuses the frozen source renders through a read-only symlink; only the new
  curriculum/contrastive/hard pool images are rendered, so this is ~1-3h.
- Outputs to `~/transform_diagnosis_data_v6_hintfix/`:
  `{train,val}_v6.jsonl`, the 16 `{train,val}_v6_{image,image_coords}_{correct,student,both,full}_chat.jsonl`
  files, `renders_v6/`, and `manifest_v6.json`.
- `make_v6_transform_data.py` calls `_validate_gold()` on every `full` target, which
  now enforces `hint_ok` (family-relevant + no leak) at generation time, so a
  leaking hint cannot be written.

Quick CPU smoke first if you want (uses the source data, ~1 min):

```bash
python make_v6_transform_data.py --dry-run --train-n 16 --val-n 8 \
  --source-dir ~/transform_diagnosis_data --out-dir ~/transform_diagnosis_data_v6_hintfix
```

---

## 2. ORCD GPU jobs: retrain ONLY the `full` stage, BOTH modalities IN PARALLEL

Submit both at once so they run on two GPUs concurrently — that parallelism is
what keeps the whole flow inside ~9h. Preferred path continues from the existing
intermediate `both`-stage adapters and points `--data-dir` at the new hintfix data,
writing to NEW adapter/checkpoint dirs (never the frozen `~/lora_adapters_v6_*`).

```bash
# image arm
sbatch -J v6hf_i_full -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6hf_i_full_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image --stage full --data-dir ~/transform_diagnosis_data_v6_hintfix --init-adapter ~/lora_adapters_v6_image_both --out ~/lora_adapters_v6_image_hintfix --output-dir ~/outputs_v6_image_hintfix --epochs 1'

# image+coordinates arm (submit at the same time -> runs in parallel)
sbatch -J v6hf_c_full -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6hf_c_full_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image_coords --stage full --data-dir ~/transform_diagnosis_data_v6_hintfix --init-adapter ~/lora_adapters_v6_coords_both --out ~/lora_adapters_v6_coords_hintfix --output-dir ~/outputs_v6_coords_hintfix --epochs 1'
```

- The `full` stage auto-rehearses the earlier tasks (default 15%) from the same
  `--data-dir`, so the hintfix `correct/student/both` files are used too.
- Expected ~2-5h each on one L40S-class GPU. If a job hits the 6h cap, resubmit
  the identical command with `--resume` appended (checkpoints are in
  `~/outputs_v6_{image,coords}_hintfix`).

FALLBACK (only if the intermediate `~/lora_adapters_v6_{image,coords}_both` were
deleted): the section-6 one-run mixed curriculum — train `full` from base (omit
`--init-adapter`); the 15% rehearsal from the hintfix data provides the earlier
tasks in a single run.

```bash
sbatch -J v6hf_i_full -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6hf_i_full_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image --stage full --data-dir ~/transform_diagnosis_data_v6_hintfix --base-model unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit --out ~/lora_adapters_v6_image_hintfix --output-dir ~/outputs_v6_image_hintfix --epochs 1'
sbatch -J v6hf_c_full -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6hf_c_full_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image_coords --stage full --data-dir ~/transform_diagnosis_data_v6_hintfix --base-model unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit --out ~/lora_adapters_v6_coords_hintfix --output-dir ~/outputs_v6_coords_hintfix --epochs 1'
```

(A one-record smoke, `... --limit 1 --val-sample 1`, confirms model load + memory
before committing the full run.)

---

## 3. ORCD GPU jobs: evaluate both new adapters on frozen test + ood

Eval reads the FROZEN `~/transform_diagnosis_data` test/ood splits (only training
data changed), with the same `--sample 500 --seed 20260709` as the original v6 run
so the IDs match and the numbers are directly comparable. Submit both in parallel.

```bash
sbatch -J v6hf_i_eval -p mit_normal_gpu -G 1 -c 8 --mem=64G -t 04:00:00 -o v6hf_i_eval_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python eval_transform.py --input image --task full --data-dir ~/transform_diagnosis_data --adapter ~/lora_adapters_v6_image_hintfix --sample 500 --seed 20260709 --tag v6_4b_image_hintfix'

sbatch -J v6hf_c_eval -p mit_normal_gpu -G 1 -c 8 --mem=64G -t 04:00:00 -o v6hf_c_eval_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python eval_transform.py --input image_coords --task full --data-dir ~/transform_diagnosis_data --adapter ~/lora_adapters_v6_coords_hintfix --sample 500 --seed 20260709 --tag v6_4b_image_coords_hintfix'
```

- Writes to `$HOME`: `results_v6_4b_image_hintfix_{test,ood}.json` +
  `records_v6_4b_image_hintfix_{test,ood}.jsonl` (and the `_coords_` pair).
- ~1-4h each; run them after (or alongside) training via `sbatch --dependency=afterok:<jobid>`
  or just submit once the adapters exist.

---

## 4. ORCD login node (CPU, no GPU): confirm the leak dropped

### 4a. Headline hint metric from the aggregates

```bash
cd ~
python - <<'PY'
import json, os
HOME = os.path.expanduser("~")
for tag in ("v6_4b_image_hintfix", "v6_4b_image_coords_hintfix"):
    for split in ("test", "ood"):
        a = json.load(open(f"{HOME}/results_{tag}_{split}.json"))
        print(f"{tag:28s} {split:4s}  hint_match={a.get('hint_match_rate')}  "
              f"hint_exact={a.get('hint_exact_match_rate')}  "
              f"both_nets={a.get('both_nets_match_rate')}  label_acc={a.get('label_accuracy')}")
PY
```

### 4b. Strict safety audit over the NEW records (drift-free)

This reuses `transform_diagnosis.hints.is_strict_leak` — the exact port of
`results/overnight/audit_hint_safety.py` — so it is the audit's "strict" definition
without touching the frozen script or `results/`:

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

Expected: `strict_leak` drops from ~0.96 toward ~0.0 and `safe_useful` rises from
~0.014 toward ~1.0 (family relevance stays ~0.99+).

### 4c. Offline re-score sanity (no GPU/API)

```bash
cd ~
python rescore_records.py \
  records_v6_4b_image_hintfix_test.jsonl records_v6_4b_image_hintfix_ood.jsonl \
  records_v6_4b_image_coords_hintfix_test.jsonl records_v6_4b_image_coords_hintfix_ood.jsonl \
  --task full --data-dir ~/transform_diagnosis_data
```

Prints a before/after table per file (parse/label/correct_net/student_net/both_nets/
derived_label/hint_match/hint_exact) and writes `*_rescored.{json,jsonl}` next to
each input.

(Optional) To reproduce the frozen audit's exact markdown over the new records,
copy `results/overnight/audit_hint_safety.py` to `~/audit_hint_safety_hintfix.py`,
repoint its `CELL_FILES` to the four `records_v6_4b_*_hintfix_{test,ood}.jsonl`
files, and run it with `--json-out ~/HINT_SAFETY_AUDIT_hintfix.json --markdown-out
~/HINT_SAFETY_AUDIT_hintfix.md`. Do NOT edit the original script or anything under
`results/`.

---

## Expected outcome

- Geometry metrics (`both_nets_match_rate`, `correct_net`, `student_net`,
  `label_accuracy`, `derived_label_accuracy`) should be ~unchanged vs the frozen v6
  results in `results/v6_final/` — same seed/mix/sizes means byte-identical maps and
  labels; only the hint text changed.
- Hint leak (strict `exact answer/map/value disclosure` in the audit) should fall
  from ~96% toward ~0%; `safe_useful` should rise from ~1.4% toward ~near-100%.
- `hint_match_rate` (family-relevant + no coordinate leak) should stay high
  (~0.9-1.0); the strict `hint_exact_match_rate` will now be ~0 by design (gold
  hints no longer contain the exact tokens — that was the leak).

## Wall-clock budget (fits in ~9h with the two training jobs in parallel)

| Stage | Command | Parallelism | Time |
|---|---|---|---|
| 1. Data regen | `make_v6_transform_data.py` | 1 CPU job | ~1-3h |
| 2. Train `full` | `train_transform.py` x2 | 2 GPUs, concurrent | ~2-5h |
| 3. Eval `full` | `eval_transform.py` x2 | 2 GPUs, concurrent | ~1-4h |
| 4. Audit/rescore | login-node Python | CPU | ~minutes |

Typical total ≈ 2h + 3.5h + 2.5h ≈ **8h** wall-clock. The two train jobs (and the
two eval jobs) MUST run on separate GPUs concurrently for this to fit; run
sequentially on a single GPU it would be ~2x(train)+2x(eval) and overflow ~9h.
