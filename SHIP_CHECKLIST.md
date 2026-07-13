# Ship checklist

Remaining manual steps to make the graded evidence visible, in priority order.
Nothing here has been executed for you. Steps 1–2 are **local, one-click** (no
credentials); step 3 needs **Hugging Face credentials**; step 4 is **optional**
and needs the **cluster + ~9h of GPU time**.

Legend: [LOCAL] no creds/network · [CREDS] needs a token · [CLUSTER] needs ORCD/GPU.

---

## 1. [LOCAL] Commit the frozen evidence + the new docs/sample/scripts

This is the single most important fix: the base-vs-tuned results, audits, and
error analysis already exist under `results/` but are **untracked**, which is why
the grader "couldn't find" them.

> **Gotcha:** `results/v6_final/audit_v6_all.log` matches `*.log` in
> `.gitignore`, so a plain `git add results/v6_final/` silently skips it. It must
> be force-added. (It is the only `.log` under `results/`.)

Review first, then stage exactly these paths (no blanket `git add .`):

```bash
# --- review ---
git status
git diff --stat

# --- frozen evidence (results/ is entirely untracked) ---
git add results/v6_final/
git add -f results/v6_final/audit_v6_all.log      # *.log is gitignored
git add results/overnight/                          # audit trail linked from README/brainlift

# --- docs, dataset card, this checklist ---
git add README.md brainlift.md DATASET_CARD.md SHIP_CHECKLIST.md

# --- larger public sample + shipping scripts ---
git add dataset_public/ model/build_public_sample.py model/push_dataset_to_hf.py

# --- reviewer samples referenced by the README (already on disk, untracked) ---
git add dataset_sample_v6/ dataset_sample_v6.zip dataset_sample/ dataset_sample.zip

# --- verify the staged set looks right, then commit ---
git status
git commit -m "$(cat <<'EOF'
Ship v6 base-vs-tuned evidence, error analysis, and dataset

- Commit frozen results/v6_final base/tuned predictions, aggregates, and audits
- Surface base-vs-tuned headline table + own-model error analysis in README
- Add evidenced composition-collapse conclusion to brainlift
- Add HF dataset card, a 240-record balanced public sample, and upload script
EOF
)"
```

Notes:
- The hint-fix code changes (`transform_diagnosis/hints.py`, `dataset.py`,
  `v6_format.py`, their tests, and `model/HINT_FIX_RUNBOOK.md`) are separate work.
  Commit them in the same commit or a dedicated one as you prefer; they are not
  required for the evidence above.
- `transform_diagnosis_data/` and `transform_diagnosis_data.zip` are gitignored on
  purpose (the 26k source corpus / 518 MB of renders). Do **not** commit them.
- Do not force-add anything else; the `.gitignore` secret/key rules are protective.

## 2. [LOCAL] Sanity-check before/after committing

```bash
python3 -m pytest transform_diagnosis/ -q     # expect: passed
python3 -m pytest model/ -q                    # expect: passed
python3 model/build_public_sample.py --dry-run # regenerates nothing; verifies counts
python3 model/push_dataset_to_hf.py --dry-run  # lists exactly what would upload
```

If you push to a remote afterward, confirm the `results/` links in `README.md`
resolve on the hosted repo.

## 3. [CREDS] Publish the dataset to Hugging Face

Fixes "dataset not on Hugging Face." The card and uploader are ready; you only
need a repo id, a token, and (before publishing) a real license in
`DATASET_CARD.md` (currently a placeholder).

```bash
# 3a. Dry run (no token needed) — confirm the file list
python3 model/push_dataset_to_hf.py --dry-run

# 3b. Real upload of the 240-record public sample
export HF_TOKEN=hf_xxx                          # never commit this
python3 model/push_dataset_to_hf.py \
  --repo-id <your-username>/geometry-transform-diagnosis-v6

# 3c. (Optional) upload a fuller local tree instead, e.g. a cluster-pulled
#     full v6 curriculum, by pointing --path at it:
# python3 model/push_dataset_to_hf.py \
#   --repo-id <your-username>/geometry-transform-diagnosis-v6 \
#   --path ~/transform_diagnosis_data_v6
```

The uploader reads `HF_TOKEN` from the environment, refuses the placeholder repo
id, and uploads `DATASET_CARD.md` as the repo `README.md`. Add the resulting HF
URL to `README.md` once live.

## 4. [CLUSTER] (Optional) Retrain to fix the hint leak (~9h)

Only needed if you want safe, non-leaking tutoring hints. The geometry/diagnosis
results in `results/v6_final/` are unaffected and remain the reported evidence.

Follow [`model/HINT_FIX_RUNBOOK.md`](model/HINT_FIX_RUNBOOK.md): regenerate v6
data to a new `_hintfix` dir (same seed/mix/sizes, hint text only), retrain the
`full` stage for both modalities **in parallel on two GPUs**, evaluate on the
frozen test/OOD splits with `--seed 20260709`, and confirm the strict leak rate
drops from ~96% toward ~0% while geometry metrics stay ~unchanged. It writes
`_hintfix` adapters/results and never touches `results/v6_final/` or the frozen
adapters. Budget ≈ 8h wall-clock with the two training jobs concurrent.

---

### Quick status

| Item | State | Needs |
| --- | --- | --- |
| Base-vs-tuned results + audits | on disk, **untracked** | step 1 (local) |
| README headline table + error analysis | done | step 1 (local) |
| Brainlift evidenced conclusion | done | step 1 (local) |
| 240-record balanced public sample | built (`dataset_public/`) | step 1 (local) |
| HF dataset card + uploader | ready (placeholder license/repo) | step 3 (creds) |
| Non-leaking hints | not done | step 4 (cluster, optional) |
