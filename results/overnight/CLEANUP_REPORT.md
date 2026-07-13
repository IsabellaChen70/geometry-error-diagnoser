# Conservative cleanup and quarantine report

Policy applied: every candidate was searched across the repository with `rg`, excluding the
quarantine/output areas by interpretation. A candidate moved only when no protected source or
final-pipeline file referred to it. Ambiguous and historically useful files remain in place.
Nothing was deleted.

## Actions

| Path | Size | Action | `rg` evidence | Why |
|---|---:|---|---|---|
| `.scratch_verify/` | 0 B; empty at move time | quarantined | Only named by the overnight instruction; no protected/final-pipeline inbound reference | Empty generated scratch directory |
| `.scratch_v6_verify/` | 700 KB; 61 files | quarantined | Matches were self-paths inside its own manifest plus the overnight instruction | Generated v6 verification fixture/output |
| `results/overnight/__pycache__/` | 20 KB | quarantined | Generated only by this run; no source reference | Regenerable bytecode created while validating the report helper |
| `model/make_v4_data.py` | 16 KB | review-manually / kept | Referenced by protected `transform_diagnosis/contrastive.py`, `BUILD_LOG.md`, and `model/sync_to_cluster.sh` | Hard rule forbids moving anything with protected/final-pipeline inbound references |
| `model/make_v5_data.py` | 16 KB | review-manually / kept | Referenced by protected `BUILD_LOG.md` and `model/sync_to_cluster.sh`; imports v4 builder | Historical v5 pipeline remains internally connected |
| `model/make_cot_data.py` | 8 KB | review-manually / kept | Referenced by protected `model/sync_to_cluster.sh` | Still included in protected cluster sync manifest |
| `model/coarsegrain_ablation.py` | 20 KB | review-manually / kept | Referenced by protected `model/sync_to_cluster.sh`; imported by its focused test | Still included in protected cluster sync manifest |
| `model/test_coarsegrain_ablation.py` | 8 KB | review-manually / kept | Imports `coarsegrain_ablation` | Moving only the test would damage a still-referenced module pair |
| `model/train_cot.py` | 16 KB | review-manually / kept | Referenced by protected `BUILD_LOG.md` and `model/sync_to_cluster.sh` | Preserved historical training provenance |
| `model/train.py` | 4 KB | review-manually / kept | Referenced by `model/eval_base_coords_fewshot.py` | Inbound reference exists; conservative rule requires keeping |
| `model/eval_tuned.py` | 12 KB | review-manually / kept | Referenced repeatedly by protected `model/eval_tuned_coords.py` and `model/sync_to_cluster.sh` | Explicit caution case; not dead |
| `model/eval_val.py` | 4 KB | review-manually / kept | Referenced by protected `model/sync_to_cluster.sh` | Still included in protected cluster sync manifest |
| `model/eval_base_coords_fewshot.py` | 16 KB | review-manually / kept | Referenced by protected `model/sync_to_cluster.sh` | Still included in protected cluster sync manifest |
| `model/probe_coords.py` | 8 KB | review-manually / kept | Referenced by protected `model/eval_tuned_coords.py` and historical provenance comments | Historical frontier-coordinate provenance remains linked |
| `model/rescore_probe.py` | 8 KB | review-manually / kept | Referenced by protected `model/rescore_records.py` documentation and `run_eval.sbatch` | General re-scorer identifies it as its predecessor/drop-in reference |
| `model/run_eval.sbatch` | 4 KB | review-manually / kept | Referenced by protected `model/sync_to_cluster.sh` | Protected sync workflow still points to it |
| `model/test_cot*.py` | not present | kept / not applicable | No matching files under `model/` | The similarly named tests under protected `transform_diagnosis/` were untouched |
| `records_probe_coords.jsonl` | 16 KB | review-manually / kept | Referenced by protected `model/rescore_records.py` and historical probe scripts | Historical raw artifact is still named by a protected final utility |
| `records_probe_coords_rescored.jsonl` | 16 KB | review-manually / kept | Referenced by historical probe/re-score documentation | Small provenance artifact; ambiguity favors keeping |
| `results_probe_coords.json` | 4 KB | review-manually / kept | Referenced by probe and re-score scripts | Small historical aggregate |
| `results_probe_coords_rescored.json` | 4 KB | review-manually / kept | Referenced by probe/re-score documentation | Small historical aggregate |
| `model/01_base_model_inference.ipynb` | 16 KB | review-manually / kept | Ambiguous notebook/deliverable | User explicitly designated manual review |
| `model/02_vision_base_model.ipynb` | 20 KB | review-manually / kept | Ambiguous notebook with existing modifications | User explicitly designated manual review |
| `model/colab_train_eval.ipynb` | 64 KB | review-manually / kept | Potential inference/publication deliverable | User explicitly designated manual review |
| `hf_dataset_topic_swarm.md` | 24 KB | review-manually / kept | Research note | User explicitly designated manual review |
| `reasoning_gaps_by_subject.md` | 76 KB | review-manually / kept | Research note | User explicitly designated manual review |
| `transformation_diagnoser_research.md` | 80 KB | review-manually / kept | Research note | User explicitly designated manual review |
| `dataset_sample/` | 880 KB | review-manually / kept | Dataset publication/sample artifact | Likely final deliverable |
| `dataset_sample.zip` | 728 KB | review-manually / kept | Dataset publication/sample archive | Likely final deliverable |
| `transform_diagnosis_data.zip` | not present | kept / not applicable | No matching file | No backup existed to review or move |
| `manual_check/` | 632 KB | review-manually / kept | Human inspection evidence | Useful audit evidence, explicitly designated manual review |
| `rlhf/` | not present | kept / not applicable | No matching directory | Nothing to move |
| `inspect_record.py` | 8 KB | kept | Used successfully overnight to validate five oracle cases | Active, useful audit tool |

## Summary

- **Quarantined:** 3 paths (2 pre-existing scratch directories, 1 overnight-generated bytecode directory).
- **Review manually / kept:** 27 existing paths.
- **Not present / no action:** 3 candidate patterns.
- **Hard-deleted:** 0.

Quarantined paths preserve their original relative layout beneath `_quarantine/`.

