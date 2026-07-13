# v6 canonical-net transform runbook

v6 predicts the unique composed affine map (`linear`, `tx`, `ty`) for RED→GREEN
and RED→BLUE. It never treats one ordered step decomposition as identifiable.
The source `~/transform_diagnosis_data` remains read-only; all v6 data goes to
`~/transform_diagnosis_data_v6`.

## 0. Laptop: verify and sync code

```bash
cd /Users/isabellachen/projects/SLM
python3 -m pytest transform_diagnosis/ -q
bash -n model/sync_to_cluster.sh
bash model/sync_to_cluster.sh --dry-run
bash model/sync_to_cluster.sh
```

The sync is code-only and uses no `--delete`. Run the remaining commands on ORCD
from `$HOME`.

## 1. ORCD login node: snapshot old metadata

Dry-run first, then explicitly execute:

```bash
module load miniforge
conda activate slm 2>/dev/null || source activate slm
cd ~
python snapshot_v6_artifacts.py --out ~/slm_v6_snapshot
python snapshot_v6_artifacts.py --out ~/slm_v6_snapshot --execute
```

This copies result JSON/JSONL and adapter metadata only, never datasets, renders,
weights, or checkpoints.

## 2. ORCD CPU job: generate separate v6 data

```bash
sbatch -J v6_data -p mit_normal -c 8 --mem=32G -t 03:00:00 \
  -o v6_data_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python make_v6_transform_data.py --source-dir ~/transform_diagnosis_data --out-dir ~/transform_diagnosis_data_v6 --train-n 9600 --val-n 400 --mix 0.50,0.20,0.15,0.15 --seed 20260711'
```

If the site uses a differently named CPU partition, change only `-p mit_normal`.
Expected wall time is roughly 1–3 hours, dominated by new PNG rendering. Outputs:

- `~/transform_diagnosis_data_v6/{train,val}_v6.jsonl`
- 16 task/modality chat files named
  `{train,val}_v6_{image,image_coords}_{correct,student,both,full}_chat.jsonl`
- `~/transform_diagnosis_data_v6/renders_v6/`
- `~/transform_diagnosis_data_v6/manifest_v6.json`

An interrupted identical run may use `--resume-existing-output`. A nonempty
output is otherwise refused, and a resume with a different config is refused.

## 3. ORCD GPU jobs: 4B image-only arm

Each stage is a separate job under the six-hour cap. Later stages load the prior
adapter directly and do not create a second PEFT wrapper. They also rehearse
earlier task files (default 15%). Submit each command only after the preceding
stage finishes successfully (or add an equivalent SLURM `afterok` dependency).

```bash
sbatch -J v6i_correct -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6i_correct_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image --stage correct --epochs 1'

sbatch -J v6i_student -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6i_student_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image --stage student --init-adapter ~/lora_adapters_v6_image_correct --epochs 1'

sbatch -J v6i_both -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6i_both_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image --stage both --init-adapter ~/lora_adapters_v6_image_student --epochs 1'

sbatch -J v6i_full -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6i_full_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image --stage full --init-adapter ~/lora_adapters_v6_image_both --epochs 1'
```

Typical stage time is approximately 2–5 hours on one L40S-class GPU. If a stage
hits the cap, resubmit the identical command with `--resume`; its checkpoints
are in `~/outputs_v6_image_<stage>`. Final adapters are
`~/lora_adapters_v6_image`.

Evaluate a paired 500-record sample from frozen test and OOD:

```bash
sbatch -J v6i_eval -p mit_normal_gpu -G 1 -c 8 --mem=64G -t 04:00:00 -o v6i_eval_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python eval_transform.py --input image --task full --data-dir ~/transform_diagnosis_data --adapter ~/lora_adapters_v6_image --sample 500 --seed 20260709 --tag v6_4b_image'
```

This writes `results_v6_4b_image_{test,ood}.json` and matching record JSONL.

## 4. ORCD GPU jobs: 4B image + coordinates arm

Repeat the four stages with `--modality image_coords`:
As above, submit each stage only after its predecessor succeeds.

```bash
sbatch -J v6c_correct -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6c_correct_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image_coords --stage correct --epochs 1'
sbatch -J v6c_student -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6c_student_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image_coords --stage student --init-adapter ~/lora_adapters_v6_coords_correct --epochs 1'
sbatch -J v6c_both -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6c_both_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image_coords --stage both --init-adapter ~/lora_adapters_v6_coords_student --epochs 1'
sbatch -J v6c_full -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6c_full_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality image_coords --stage full --init-adapter ~/lora_adapters_v6_coords_both --epochs 1'

sbatch -J v6c_eval -p mit_normal_gpu -G 1 -c 8 --mem=64G -t 04:00:00 -o v6c_eval_%j.log \
  --wrap 'module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python eval_transform.py --input image_coords --task full --data-dir ~/transform_diagnosis_data --adapter ~/lora_adapters_v6_coords --sample 500 --seed 20260709 --tag v6_4b_image_coords'
```

The same seed/sample produces the same IDs as the image arm. Final adapters are
`~/lora_adapters_v6_coords`.

## 5. ORCD login node: frontier same-schema smoke and paired sample

First inspect one complete payload without a key or API call:

```bash
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits test --sample 1 --limit 1 --seed 20260709 --dry-run \
  --model '<verified-vision-route>'
```

Then run one real image smoke and paired image/image+coordinates samples. The
model route must be verified as vision-capable on the gateway.

```bash
export TFY_API_KEY='...'
python eval_frontier_gateway.py --schema v6 --input image --task full \
  --splits test --sample 1 --limit 1 --seed 20260709 \
  --model '<verified-vision-route>' --tag frontier_v6_image_smoke

python eval_frontier_gateway.py --schema v6 --input image --task full \
  --sample 150 --seed 20260709 --model '<verified-vision-route>' \
  --tag frontier_v6_image
python eval_frontier_gateway.py --schema v6 --input image_coords --task full \
  --sample 150 --seed 20260709 --model '<verified-vision-route>' \
  --tag frontier_v6_image_coords
```

These are API-bound login-node commands and need no GPU.

## 6. Larger same-family run on the winning modality

Do not guess a model identifier. Set a model ID that has been independently
verified to exist, be accessible, be Qwen3-VL-compatible, and fit the allocated
GPU. A one-run mixed curriculum is available by training `full` from base;
the default rehearsal adds samples from all three earlier task files.

```bash
VERIFIED_BASE_MODEL='<verified-larger-same-family-model-id>'
BEST_MODALITY='image'  # or image_coords, based on the paired 4B results

sbatch -J v6_large -p mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00 -o v6_large_%j.log \
  --wrap "module load miniforge; conda activate slm 2>/dev/null || source activate slm; cd ~; python train_transform.py --modality ${BEST_MODALITY} --stage full --base-model '${VERIFIED_BASE_MODEL}' --out ~/lora_adapters_v6_large_best --output-dir ~/outputs_v6_large_best --epochs 1"
```

Memory and wall time must be adjusted after a one-record/model-load smoke because
the larger model ID and quantization are intentionally not assumed here.

## Local-only validation boundary

`--dry-run` validates imports, paths, prompts, scoring, stage composition, and
adapter metadata without CUDA. Sequential adapter continuation follows the
Unsloth convention of loading the prior adapter directory via
`FastVisionModel.from_pretrained` and verifies `peft_config` before training, but
the actual Unsloth/PEFT continuation and GPU memory footprint cannot be proven
without a GPU job. Gateway quality/availability likewise requires the real API.
