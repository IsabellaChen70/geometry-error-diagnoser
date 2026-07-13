#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o eval_%j.log

# Fills three cells of the 2x2 {frontier, tuned-4B} x {image, coords} table with the FIXED
# metrics (frontier+coords is already done locally via probe_coords.py + rescore_probe.py).
# Assumes in $HOME: the fixed harness synced as ~/slm_eval, the eval_*.py scripts,
# ~/transform_diagnosis_data, and ~/lora_adapters.
module load miniforge
conda activate slm 2>/dev/null || source activate slm
cd ~

# tuned-4B + image: re-run under the fixed harness (this is the 4B+image "re-score").
python eval_tuned.py

# tuned-4B + coords: same checkpoint, coordinates-as-text input.
python eval_tuned_coords.py

# frontier + image: API-bound (no GPU). Runs only if a key is set, so the GPU cells still
# finish cleanly otherwise. If the compute node has no outbound internet, run this on the
# login node instead:  python eval_frontier.py
if [ -n "$ANTHROPIC_API_KEY" ]; then
    python eval_frontier.py
else
    echo "[run_eval] ANTHROPIC_API_KEY not set -- skipping frontier+image (run it on the login node)."
fi
