# Authorized Dead-Path Deletion Report

- Timestamp: `2026-07-12T14:14:55-05:00`
- Repository: `/Users/isabellachen/projects/SLM`
- Authorization: explicit user-authorized hard deletion after two independent audits

## Audit agreement and safety checks

Both audits classified generated Python bytecode caches and empty placeholder directories as dead. The artifact audit also verified that `.pytest_cache/` and `.ruff_cache/` were standard regenerable caches.

Immediately before deletion, every exact target was re-checked with `lstat`-style, no-follow inspection. Every target was a real directory, no target or descendant was a symlink or special file, each `__pycache__/` contained only generated `.pyc` files (or was empty), the pytest and Ruff directories contained their standard cache markers and layouts, and every empty-directory candidate was empty.

## Deleted audited targets

All 10 audited targets were deleted; none were skipped.

1. `/Users/isabellachen/projects/SLM/transform_diagnosis/__pycache__/` — 845,082 logical bytes; 913,408 allocated bytes
2. `/Users/isabellachen/projects/SLM/model/__pycache__/` — 347,013 logical bytes; 380,928 allocated bytes
3. `/Users/isabellachen/projects/SLM/.pytest_cache/` — 22,897 logical bytes; 40,960 allocated bytes
4. `/Users/isabellachen/projects/SLM/.ruff_cache/` — 209 logical bytes; 12,288 allocated bytes
5. `/Users/isabellachen/projects/SLM/results/overnight/__pycache__/` — 0 logical bytes; 0 allocated bytes
6. `/Users/isabellachen/projects/SLM/_quarantine/results/overnight/__pycache__/` — 18,498 logical bytes; 20,480 allocated bytes
7. `/Users/isabellachen/projects/SLM/_quarantine/.scratch_verify/` — 0 logical bytes; 0 allocated bytes
8. `/Users/isabellachen/projects/SLM/rlhf/` — 0 logical bytes; 0 allocated bytes
9. `/Users/isabellachen/projects/SLM/src/rlhf/` — 0 logical bytes; 0 allocated bytes
10. `/Users/isabellachen/projects/SLM/results/overnight/verify/brainlift_build/src/rlhf/` — 0 logical bytes; 0 allocated bytes

Total reclaimed from pre-deletion measurements: **1,233,699 logical bytes (1.18 MiB)** and **1,368,064 allocated bytes (1.30 MiB)**. Logical size is the sum of regular-file sizes; allocated size is the sum of `st_blocks × 512` for each deleted tree.

The following newly empty quarantine parents were also removed with non-recursive empty-directory removal and are not included in the 10-target count:

- `/Users/isabellachen/projects/SLM/_quarantine/results/overnight/`
- `/Users/isabellachen/projects/SLM/_quarantine/results/`

## Skipped paths

None.

## Retained manual-review and critical items

- `/Users/isabellachen/projects/SLM/_quarantine/.scratch_v6_verify/` was retained because it contains unique dry-run evidence requiring manual review.
- `/Users/isabellachen/projects/SLM/_quarantine/` and all other remaining quarantine contents were retained.
- `/Users/isabellachen/projects/SLM/results/overnight/verify/brainlift_build/` was retained; only its verified-empty `src/rlhf/` child was removed.
- `/Users/isabellachen/projects/SLM/results/v6_final/`, `/Users/isabellachen/projects/SLM/dataset_sample/`, `/Users/isabellachen/projects/SLM/dataset_sample_v6/`, and the gitignored `/Users/isabellachen/projects/SLM/transform_diagnosis_data/` tree were retained.
- Existing overnight reports, including `LLM_JUDGE_REPORT.md`, `OVERNIGHT_REPORT.md`, and `CLEANUP_REPORT.md`, were retained.
- The 407 pre-existing staged deletions were not touched.

## Git and artifact integrity

Read-only pre-deletion snapshots:

- Worktree/index status SHA-256: `8f84cd0b99d5c9e6b007d78a1fbee6658c72a82eee9492548e0d18ba881b70fb`
- Index entries SHA-256: `d7dc4f4b26afe0a9987a1ccf885427e4ac4b04ff1ebe982c674378c660e08151`
- Staged raw diff SHA-256: `12667e2064f26a3de7afa77dee31ffb588cb20dd291461bec68ab840dbc8dbfb`
- Staged name/status SHA-256: `394866fc804b9f86b8f7161460e6c5f69936723fb8131c9f89e1b52a657fa20e`

The same four hashes were observed after deletion and again after this report was created. Path-scoped status identifies this report as untracked (`?`) and not staged. The Git index and all 407 staged deletions were therefore byte-for-byte unchanged by the cleanup. No Git write command was used.

No source files, staged files, frozen evidence, notebooks, datasets, results, samples, archives, or user-authored artifacts were touched or deleted. This report is the only new artifact created by the operation.
