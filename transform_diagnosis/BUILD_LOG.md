# BUILD LOG — transform_diagnosis

Composed-transformation **student-error diagnosis** dataset generator.

This log is appended to throughout the (unattended) build. Timestamps are ISO-8601 UTC.

---

## 2026-07-08T04:39:58Z — Start

- Workspace: `/Users/isabellachen/projects/SLM`. Not a git repo (verified `git status` → fatal). No init performed (out of scope).
- Environment: `python3` 3.13.7, `matplotlib` 3.10.6, `pytest` 8.4.1 all present and importable (matplotlib with `Agg` backend). No venv needed.
- Grounding check (as the task warned): the spec claims `transform_core.py`, `reference_generator.py`, `test_transform_core.py` already exist. They do **not** exist in this workspace. Confirmed the only pre-existing Python is:
  - `model/generate_diagnosis_data.py` (owner's earlier misconception generator — DIFFERENT label taxonomy)
  - `model/generate_dataset.py` (visual two-step MCQ identification generator)
  - `GeoSym127K/` files
- Decision: create the whole package fresh at repo root `transform_diagnosis/` with ONE canonical `transform_core.py`; everything imports it. Recorded chosen paths below.

### Chosen paths (package at repo root)
- Package dir: `/Users/isabellachen/projects/SLM/transform_diagnosis/`
- Files: `__init__.py`, `__main__.py`, `transform_core.py`, `reference_generator.py`, `geometry.py`, `problems.py`, `errors.py`, `dataset.py`, `render.py`, `run_all.py`, `test_transform_core.py`, `test_dataset.py`, `BUILD_LOG.md`
- Runnable as `python3 -m transform_diagnosis` from `/Users/isabellachen/projects/SLM`.

### Key design decisions (rationale)
- **Transform representation**: affine map `x -> M x + t` with `M` an integer 2x2 orthogonal matrix `((a,b),(c,d))` and `t=(e,f)` an integer vector. Pure integer arithmetic end-to-end (no floats in the transform/grade/label path). `det(M)` sign is orientation: `+1` rotation/identity, `-1` reflection.
- **`compose(seq)`**: `seq[0]` applies first. Fold: `combined <- T ∘ combined`, i.e. `M' = T.M @ M`, `t' = T.M @ t + T.t`.
- **Text <-> Transform**: `describe_transform` / `parse_transform` round-trip. `grade`/`diagnose`/`compose` accept EITHER `Transform` objects OR schema strings (strings are parsed). This lets the acceptance tests call `diagnose(original, record["correct_transform"], record["student_transform"])` directly on the stored text lists. Rotations can be re-worded (e.g. `rotate 270 degrees clockwise` == `rotate 90 degrees counterclockwise`) — same matrix, different text.
- **`diagnose`** compares NET maps only (independent of `original`), total & deterministic. See `transform_core.diagnose` docstring for the exact decision tree.
- **`is_asymmetric`**: shape has trivial symmetry under the 8-element lattice isometry group (identity + 3 rotations + 4 reflections) modulo translation — guarantees the net map is uniquely recoverable (`recover_map`).
- **Error injection (`errors.py`)**: mutate the correct sequence for a target category, then VERIFY via `diagnose`; keep only if it equals the target, else signal "cannot inject" and the caller retries with a fresh (compatible-pattern) problem. Injected label and independent diagnosis therefore always agree.
- 2026-07-08T04:45:23Z — ======================================================================
- 2026-07-08T04:45:23Z — run_all START — seed=0 n=400 min_count=30 out=/Users/isabellachen/projects/SLM/transform_diagnosis_data render=True
- 2026-07-08T04:45:23Z — stage 'import_check': attempt 1 starting
- 2026-07-08T04:45:23Z — import_check: matplotlib_available=True
- 2026-07-08T04:45:23Z — stage 'import_check': OK on attempt 1
- 2026-07-08T04:45:23Z — stage 'generate': attempt 1 starting
- 2026-07-08T04:45:38Z — generate: 400 records; labels={'correct': 50, 'reflection_instead_of_rotation': 50, 'rotation_instead_of_reflection': 50, 'wrong_rotation_angle': 50, 'wrong_reflection_line': 50, 'wrong_translation': 50, 'opposite_translation': 50, 'completely_wrong': 50}; splits={'test': 40, 'train': 320, 'val': 40}; new_renders=400
- 2026-07-08T04:45:38Z — stage 'generate': OK on attempt 1
- 2026-07-08T04:45:38Z — stage 'determinism_check': attempt 1 starting
- 2026-07-08T04:45:38Z — determinism: re-run byte-identical (400 lines match)
- 2026-07-08T04:45:38Z — stage 'determinism_check': OK on attempt 1
- 2026-07-08T04:45:38Z — stage 'acceptance_tests': attempt 1 starting
- 2026-07-08T04:45:38Z — acceptance_tests: running /Library/Frameworks/Python.framework/Versions/3.13/bin/python3 -m pytest /Users/isabellachen/projects/SLM/transform_diagnosis -q
- 2026-07-08T04:45:39Z — acceptance_tests: PASSED — 45 passed in 0.31s
- 2026-07-08T04:45:39Z — stage 'acceptance_tests': OK on attempt 1
- 2026-07-08T04:45:39Z — run_all DONE — 400 records at /Users/isabellachen/projects/SLM/transform_diagnosis_data

---

## 2026-07-08T04:46:08Z — COMPLETE (definition of done met)

**All six acceptance tests pass and the dataset + renders are written to `--out`.**

### Final artifacts (`--out = /Users/isabellachen/projects/SLM/transform_diagnosis_data`)
- `data.jsonl` (400 records), `train.jsonl` (320), `val.jsonl` (40), `test.jsonl` (40)
- `renders/000000.png` … `renders/000399.png` (400 PNGs), `reference_example.png`
- `summary.json`

### Per-label counts (perfectly balanced; min-count 30, n 400 -> 50 each)
`correct`, `reflection_instead_of_rotation`, `rotation_instead_of_reflection`,
`wrong_rotation_angle`, `wrong_reflection_line`, `wrong_translation`,
`opposite_translation`, `completely_wrong` = **50 each** (400 total).
Splits: train 320 / val 40 / test 40.

### Acceptance tests (45 tests total across the two files)
1. `test_transform_core.py` — passes (the contract test, kept green, never weakened).
2. Same seed -> byte-identical JSONL — verified in-process (`run_all` determinism stage),
   in `test_dataset.py`, AND explicitly via CLI (`--out` twice + `diff` == diff-clean).
3. 2000 problems: correct grades True; integer & in-bounds; simple & asymmetric;
   image != original — passes.
4. `recover_map(original, correct_image)` == intended net map — passes.
5. Every record: `diagnose(original, correct_transform, student_transform) == label`
   and `is_correct == (label == "correct")` — passes (checked from the written JSONL).
6. Every label reaches its configured minimum count — passes.

### Reconciliation with the pre-existing `model/generate_diagnosis_data.py`
- Left **in place, superseded**. It is a different, earlier generator with a DIFFERENT
  taxonomy (`swapped_order`, `skipped_first/second_step`, `wrong_reflection_axis`,
  `wrong_translation_direction`, `wrong_rotation_direction`, `reflection_instead_of_rotation`,
  `rotation_instead_of_reflection`, `none`) and its OWN inline transform functions.
- This package does NOT import it and does NOT copy its transforms/grader. Instead we
  MINED its *ideas*: the angular-sweep `generate_irregular_polygon`, the simple-polygon
  test, integer snapping/validation, the identifiability idea, and the matplotlib grid
  render style. All of those were re-expressed in `geometry.py` / `render.py` and routed
  through the single canonical `transform_core` for every transform / symmetry call.
- Net result: exactly ONE implementation of transforms + grade + diagnose + recover_map
  in the whole package (`transform_core.py`), used by `geometry`, `problems`, `errors`,
  `dataset`, `render`, `reference_generator`, and both test files.

### Ambiguities resolved (and how)
- **Spec claimed files already existed** — they did not; created the package fresh.
- **`diagnose` precedence when orientation AND translation both wrong** — resolved to
  `completely_wrong` (matches the spec's "orientation AND translation both wrong" rule).
  The orientation-confusion / angle / line / opposite / wrong-translation labels are only
  assigned when the *other* axis matches, keeping the function total and each label
  reachable. Error injection uses compatible patterns + a verify-or-discard loop so the
  injected label always equals the independent diagnosis.
- **Text carries wording, math carries meaning** — rotations may be re-worded
  (`90 ccw` == `270 cw`) for variety; equality/diagnosis compare only the matrix+vector,
  so a re-worded correct answer still diagnoses as `correct`.
- **`--render DIR`** — interpreted as the render subdirectory NAME under `--out`
  (default `renders`); `render_path` in each record is `"<subdir>/<id:06d>.png"`.
- **`--n` vs `--min-count`** — `min-count` is a hard floor per label; `--n` is the target
  total (extra distributed round-robin). If `n < min_count * 8`, the floor wins.
- **Split** — pure integer function of `(seed, id)` (no floats in the path), fractions
  from `--split` (default 0.8/0.1/0.1).

### Exact commands to regenerate
```
cd /Users/isabellachen/projects/SLM
python3 transform_diagnosis/run_all.py                 # full resilient pipeline + tests
python3 -m transform_diagnosis --seed 0 --n 400 --min-count 30 --out transform_diagnosis_data
python3 -m pytest transform_diagnosis -q               # 45 passed
```

