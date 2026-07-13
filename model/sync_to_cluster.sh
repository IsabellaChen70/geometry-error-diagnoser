#!/usr/bin/env bash
#
# sync_to_cluster.sh -- push the fixed eval harness + eval scripts to the ORCD
# (MIT SLURM) cluster over SSH so you can run the 2x2 {frontier,tuned-4B} x
# {image,coords} evaluation there, and (optionally) trigger the GPU-free
# re-score step remotely.
#
# It ONLY pushes code. It is idempotent, uses no --delete, and does nothing
# destructive on the remote. It never touches the dataset, the LoRA adapters,
# images, notebooks, or any results_*/records_* files -- those are large and/or
# produced ON the cluster.
#
# Auth: uses your EXISTING SSH keys / ssh-agent. NEVER put passwords or private
# keys in this file. The CONFIG block below holds only non-secret host/user/path
# values.
#
# Usage:
#   bash model/sync_to_cluster.sh            # sync harness + scripts to the cluster
#   bash model/sync_to_cluster.sh --dry-run  # show what WOULD transfer (rsync -n); no changes
#   bash model/sync_to_cluster.sh --rescore  # sync, then run the GPU-free re-score remotely
#   bash model/sync_to_cluster.sh --help
#
set -euo pipefail

# ===========================================================================
# CONFIG -- fill these in (NON-SECRET values only). Each may also be supplied
# from the environment, e.g.:
#     ORCD_HOST=login.mit.edu ORCD_USER=you bash model/sync_to_cluster.sh --dry-run
# ===========================================================================
ORCD_HOST="${ORCD_HOST:-orcd-login.mit.edu}"   # MIT ORCD Engaging login node
ORCD_USER="${ORCD_USER:-ikchen}"    # your MIT Kerberos username (the part before @mit.edu)

# Remote base directory. Leave EMPTY to use your remote login $HOME -- that is the
# intended layout: the eval_*.py scripts import ~/slm_eval and read
# ~/transform_diagnosis_data / ~/lora_adapters, all relative to $HOME. If you do
# set this, use the ABSOLUTE path of that same home (e.g. "/home/you"), NOT a
# different directory, or the scripts' ~-relative imports/paths will not resolve.
REMOTE_HOME="${REMOTE_HOME:-}"

# Name of the harness package directory under the remote home. Keep "slm_eval":
# eval_tuned.py / eval_val.py do `from slm_eval import eval`, so renaming this
# would break their imports unless you also edit those files.
REMOTE_SLM_EVAL="${REMOTE_SLM_EVAL:-slm_eval}"
# ===========================================================================

usage() {
    cat <<'EOF'
sync_to_cluster.sh -- push the eval harness + scripts to the ORCD SLURM cluster.

USAGE:
  bash model/sync_to_cluster.sh [--dry-run|-n] [--rescore] [--help|-h]

FLAGS:
  -n, --dry-run   rsync in dry-run mode: print exactly what WOULD transfer and
                  make no changes on the cluster.
  --rescore       After syncing, SSH in and run the GPU-free re-score
                  (rescore_records.py over any records_tuned*.jsonl already on
                  the cluster from a prior eval run). No GPU / no API key needed.
                  If no such files exist yet, it prints a note and exits cleanly.
  -h, --help      Show this help and exit.

CONFIG (edit the top of this file, or pass as env vars -- NON-SECRET only):
  ORCD_HOST         login-node hostname
  ORCD_USER         your ORCD username
  REMOTE_HOME       absolute remote home path (default: use the login $HOME)
  REMOTE_SLM_EVAL   harness package dir name (default: slm_eval)

Auth uses your existing SSH keys / ssh-agent. Never store secrets in this file.
EOF
}

# --------------------------- parse flags -----------------------------------
DRY_RUN=false
RESCORE=false
while [ $# -gt 0 ]; do
    case "$1" in
        -n|--dry-run) DRY_RUN=true ;;
        --rescore)    RESCORE=true ;;
        -h|--help)    usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; echo >&2; usage >&2; exit 2 ;;
    esac
    shift
done

# --------------------------- preflight -------------------------------------
if [ -z "$ORCD_HOST" ] || [ -z "$ORCD_USER" ]; then
    echo "ERROR: ORCD_HOST and ORCD_USER must be set (edit the CONFIG block, or pass as env vars)." >&2
    echo "       e.g. ORCD_HOST=login.mit.edu ORCD_USER=you bash model/sync_to_cluster.sh --dry-run" >&2
    exit 1
fi

# Repo root, derived from this script's own location so it works from any CWD.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOCAL_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"

DEST="${ORCD_USER}@${ORCD_HOST}"

# rsync destination base: relative (empty) => remote login $HOME. mkdir/cd need a
# concrete base, so use a literal $HOME (expanded on the remote) when unset.
if [ -n "$REMOTE_HOME" ]; then
    RBASE="${REMOTE_HOME%/}/"        # e.g. /home/you/
    REMOTE_BASE_EXPR="${REMOTE_HOME%/}"
else
    RBASE=""                         # rsync relative to the remote login $HOME
    REMOTE_BASE_EXPR="\$HOME"        # kept literal; expanded on the remote side
fi

# --------------------------- file lists ------------------------------------
# Harness package. It is self-contained: eval.py -> {enum_transform, hints, transform_core};
# chat_format.py -> transform_core; enum_transform.py -> transform_core;
# cot.py -> {chat_format, enum_transform, transform_core};
# __init__.py -> transform_core (all relative imports, so the same __init__.py is valid
# under either package name). Synced to BOTH ~/slm_eval AND ~/transform_diagnosis so every
# script resolves its import:
#   eval_tuned.py / eval_val.py     : from slm_eval import eval            (needs slm_eval)
#   eval_tuned_coords.py            : try slm_eval -> transform_diagnosis  (needs chat_format too)
#   eval_frontier.py                : try transform_diagnosis -> slm_eval
#   rescore_records.py              : try transform_diagnosis -> slm_eval
#   make_cot_data.py                : try slm_eval -> transform_diagnosis  (needs cot + eval)
#   make_v5_data.py                 : needs cot + enum_transform + eval (v5 enum-CoT targets)
# enum_transform.py is the v5 discrete-transform vocabulary; eval.py + cot.py both import it,
# so it MUST ship with the harness (both package names).
HARNESS_DIR="$LOCAL_REPO/transform_diagnosis"
HARNESS_FILES=(__init__.py eval.py hints.py transform_core.py chat_format.py cot.py enum_transform.py net_transform.py v6_format.py)

# Dataset GENERATOR modules. These are NOT needed by the eval harness, but ARE needed to
# generate v4/v6 data ON the cluster (the assemblers import
# contrastive, dataset, render, ...`). Synced ONLY to ~/transform_diagnosis/ (the full
# generator package), so that + HARNESS_FILES makes the package complete there. Rendering
# needs matplotlib in the `slm` env (guarded import in render.py; see the v4 note below).
GENERATOR_FILES=(geometry.py problems.py errors.py dataset.py render.py contrastive.py __main__.py)

# Eval + train scripts + the SLURM batch file -> remote home. eval_base_coords_fewshot.py
# imports the sibling eval_tuned_coords.py (for select_ids/load_raw/run_model), and
# eval_tuned.py now does too, so all three must land in the same remote dir. train_cot.py has
# no harness import (reads jsonl + trains) but must still be present to sbatch; make_cot_data.py
# imports the synced cot + eval harness. eval_frontier_gateway.py imports the sibling
# eval_tuned_coords.py (select_ids/load_raw) and the harness, so it lands in the same dir too.
# coarsegrain_ablation.py is a GPU-free post-hoc coarse-grain ablation over saved
# records_*.jsonl; it imports ONLY the eval label-parser (same slm_eval->transform_diagnosis
# fallback), so it runs on the login node next to the records a prior eval produced.
#
# make_v5_data.py imports the sibling make_v4_data.py (it REUSES v4's assembler verbatim so
# v5 data == v4 data, only the target differs) plus the synced cot + enum_transform + eval
# harness, so it must land in the same remote dir as make_v4_data.py.
#
# confidence_intervals.py is a GPU-free post-hoc statistics tool (Wilson CIs + paired
# McNemar/bootstrap over saved results_*/records_*). Stdlib-only, so it runs on the login
# node next to the eval outputs; CONFIDENCE.md documents it.
MODEL_DIR="$LOCAL_REPO/model"
SCRIPT_FILES=(eval_tuned.py eval_tuned_coords.py eval_base_coords_fewshot.py eval_frontier.py eval_frontier_gateway.py eval_transform.py eval_val.py rescore_records.py coarsegrain_ablation.py confidence_intervals.py audit_identifiability.py audit_v6_predictions.py make_cot_data.py make_v4_data.py make_v5_data.py make_v6_transform_data.py make_golden_set.py snapshot_v6_artifacts.py train_cot.py train_transform.py rewrite_hints_fast.py run_eval.sbatch V6_TRANSFORM_RUNBOOK.md HINT_FIX_RUNBOOK.md HINT_FIX_FAST.md GOLDEN_SET.md GOLDEN_COMPARISON.md CONFIDENCE.md)

harness_srcs=()
for f in "${HARNESS_FILES[@]}"; do harness_srcs+=("$HARNESS_DIR/$f"); done
generator_srcs=()
for f in "${GENERATOR_FILES[@]}"; do generator_srcs+=("$HARNESS_DIR/$f"); done
script_srcs=()
for f in "${SCRIPT_FILES[@]}"; do script_srcs+=("$MODEL_DIR/$f"); done

# Fail early if any listed local source is missing.
missing=0
for f in "${harness_srcs[@]}" "${generator_srcs[@]}" "${script_srcs[@]}"; do
    if [ ! -f "$f" ]; then echo "ERROR: missing local source file: $f" >&2; missing=1; fi
done
if [ "$missing" -ne 0 ]; then
    echo "Aborting: fix the missing source file(s) above." >&2
    exit 1
fi

# --------------------------- rsync options ---------------------------------
RSYNC_OPTS=(-a -v)
if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS+=(-n)
    echo "== DRY RUN == (rsync -n; nothing will be changed on the cluster)"
fi

echo "Local repo : $LOCAL_REPO"
echo "Remote     : ${DEST}:${RBASE:-<login \$HOME>}"
echo "Harness    -> ${REMOTE_SLM_EVAL}/ and transform_diagnosis/   |   scripts -> home"
echo

# --------------------------- create remote dirs ----------------------------
# Skipped in dry-run so we make no changes; rsync -n below still prints the plan.
if [ "$DRY_RUN" = true ]; then
    echo "[dry-run] would run: ssh $DEST 'mkdir -p <home>/$REMOTE_SLM_EVAL <home>/transform_diagnosis'"
else
    echo ">> ensuring remote package directories exist"
    # REMOTE_BASE_EXPR is intentionally expanded on the remote when it is \$HOME.
    # shellcheck disable=SC2029
    ssh "$DEST" "mkdir -p \"${REMOTE_BASE_EXPR}/${REMOTE_SLM_EVAL}\" \"${REMOTE_BASE_EXPR}/transform_diagnosis\""
fi

# --------------------------- sync ------------------------------------------
echo ">> syncing harness -> ${DEST}:${RBASE}${REMOTE_SLM_EVAL}/"
rsync "${RSYNC_OPTS[@]}" "${harness_srcs[@]}" "${DEST}:${RBASE}${REMOTE_SLM_EVAL}/"

echo ">> syncing harness -> ${DEST}:${RBASE}transform_diagnosis/"
rsync "${RSYNC_OPTS[@]}" "${harness_srcs[@]}" "${DEST}:${RBASE}transform_diagnosis/"

echo ">> syncing generator -> ${DEST}:${RBASE}transform_diagnosis/  (v4/v6 datagen)"
rsync "${RSYNC_OPTS[@]}" "${generator_srcs[@]}" "${DEST}:${RBASE}transform_diagnosis/"

echo ">> syncing scripts -> ${DEST}:${RBASE:-<login \$HOME>}"
rsync "${RSYNC_OPTS[@]}" "${script_srcs[@]}" "${DEST}:${RBASE}"

# --------------------------- optional re-score -----------------------------
if [ "$RESCORE" = true ]; then
    if [ "$DRY_RUN" = true ]; then
        echo
        echo "[dry-run] skipping --rescore (it would run rescore_records.py on the cluster)."
    else
        echo
        echo ">> running the GPU-free re-score on the cluster"
        if [ -n "$REMOTE_HOME" ]; then rescore_dir="${REMOTE_HOME%/}"; else rescore_dir=""; fi
        # Login shell (-l) so `module`/conda are set up; -s reads the script below
        # from stdin; the base dir is passed as $1 (empty => remote $HOME).
        ssh "$DEST" bash -l -s -- "$rescore_dir" <<'REMOTE'
set -euo pipefail
target="${1:-$HOME}"
cd "$target"
shopt -s nullglob
files=( records_tuned*.jsonl )
if [ "${#files[@]}" -eq 0 ]; then
    echo "[rescore] No records_tuned*.jsonl found in $(pwd) on the cluster."
    echo "[rescore] Those per-record files are produced by a prior GPU eval run."
    echo "[rescore] Run it first:  sbatch run_eval.sbatch   then re-run this with --rescore."
    exit 0
fi
module load miniforge
conda activate slm 2>/dev/null || source activate slm
echo "[rescore] re-scoring ${#files[@]} file(s): ${files[*]}"
python rescore_records.py "${files[@]}"
echo "[rescore] done -- wrote *_rescored.jsonl + results_*_rescored.json next to each input."
REMOTE
    fi
fi

# --------------------------- summary ---------------------------------------
echo
echo "============================================================"
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN complete -- nothing was changed on the cluster."
else
    echo "Sync complete."
fi
cat <<EOF
Synced (code only; NOT the dataset / adapters / images / notebooks / results):
  transform_diagnosis/{$(IFS=,; echo "${HARNESS_FILES[*]}")}
      -> ${DEST}:${RBASE}${REMOTE_SLM_EVAL}/
      -> ${DEST}:${RBASE}transform_diagnosis/
  transform_diagnosis/{$(IFS=,; echo "${GENERATOR_FILES[*]}")}   # v4/v6 datagen generator
      -> ${DEST}:${RBASE}transform_diagnosis/
  model/{$(IFS=,; echo "${SCRIPT_FILES[*]}")}
      -> ${DEST}:${RBASE:-<login \$HOME>}

Next, on the cluster:
  ssh ${DEST}

  # --- v6 canonical NET-map pipeline (full staged commands/resources are in
  #     ~/V6_TRANSFORM_RUNBOOK.md, synced above). Source v1-v5 data is read-only;
  #     output is a separate ~/transform_diagnosis_data_v6 directory. ---
  python snapshot_v6_artifacts.py --out ~/slm_v6_snapshot                  # dry-run
  python make_v6_transform_data.py --dry-run --train-n 16 --val-n 8       # CPU smoke
  # Then follow V6_TRANSFORM_RUNBOOK.md for generation, both 4B arms, eval, and frontier.

  # --- v4: generate structured-CoT data ON the cluster (CPU login node; reuses existing
  #     ~/transform_diagnosis_data/renders for the NORMAL 50%, renders only new records) ---
  python -c "import matplotlib" || pip install --user matplotlib   # render.py preflight
  python make_v4_data.py --n 9600 --val-n 400                      # -> train/val_v4_cot_chat.jsonl + renders_v4/
  # then QLoRA-train v4 from BASE (6h partition cap -> --resume to continue a capped run):
  sbatch --wrap "python train_cot.py --train-file ~/transform_diagnosis_data/train_v4_cot_chat.jsonl \\
     --val-file ~/transform_diagnosis_data/val_v4_cot_chat.jsonl --out ~/lora_adapters_v4 \\
     --output-dir outputs_v4 --epochs 3" -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00
  python eval_tuned.py --adapters ~/lora_adapters_v4 --sample 500 --seed 20260709 --tag v4

  # --- v5: ENUM-CoT data targeting transform_match (reframe exact-transform recovery as a
  #     CLASSIFICATION over a small discrete vocabulary). v5 == v4 DATA re-targeted to the
  #     enum schema (same seed/ids/splits/renders), so it reuses ~/…/renders for NORMAL and
  #     the renders_v4/ images for the new records (a no-op if v4 already ran). The fixed
  #     eval harness keeps transform_match as semantic NET-map equality for comparability
  #     and reports exact ordered steps separately as step_sequence_exact_match_rate. ---
  python make_v5_data.py --n 9600 --val-n 400                      # -> train/val_v5_cot_chat.jsonl
  #   (optional curriculum knobs: --transform-first foregrounds correct_transform in the JSON;
  #    --mix 0.4,0.4,0.2 upweights the transform-diverse contrastive/curriculum pools.)
  # QLoRA-train v5 from BASE, mirroring train_cot resources (6h partition cap):
  sbatch --wrap "python train_cot.py --train-file ~/transform_diagnosis_data/train_v5_cot_chat.jsonl \\
     --val-file ~/transform_diagnosis_data/val_v5_cot_chat.jsonl --out ~/lora_adapters_v5 \\
     --output-dir outputs_v5 --epochs 3" -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00
  #   if the 6h cap hits before finishing, resubmit the SAME command with --resume appended
  #   to continue from the latest checkpoint in outputs_v5.
  python eval_tuned.py --adapters ~/lora_adapters_v5 --sample 500 --seed 20260709 --tag v5

  sbatch run_eval.sbatch            # GPU: tuned+image, tuned+coords (+ frontier+image if a key is set)

  # frontier row on the LOGIN node (API-bound, no GPU) -- needs an API key:
  export ANTHROPIC_API_KEY=sk-ant-...
  python eval_frontier.py                 # frontier + image  (direct Anthropic SDK)
  python eval_frontier.py --input coords  # frontier + coords (same model & ids)

  # ... OR, if you only have the TrueFoundry (OpenAI-compatible) gateway, no Anthropic key:
  export TFY_API_KEY=tfy_...
  python eval_frontier_gateway.py --sample 500 --model <vision-route>  # frontier + image
  python eval_frontier_gateway.py --input coords --sample 500          # frontier + coords

  # GPU-free re-score of saved per-record files (or re-run this script with --rescore):
  python rescore_records.py records_tuned*.jsonl
EOF
