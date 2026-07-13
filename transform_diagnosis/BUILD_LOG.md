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

---

## 2026-07-08 — broaden `completely_wrong`, extend tests, repo cleanup (interactive)

### Task 1 — `errors.py`: broadened the `completely_wrong` injector
Defect: `completely_wrong` (per `diagnose`: net linear part AND translation both wrong)
covers two flavors, but the old injector only produced flavor (a) — a monoculture.

- Replaced `_inject_completely_wrong` with three sub-kind helpers, all still funneling
  through the `_accept` verify-or-discard loop (so `diagnose` remains the sole oracle;
  `diagnose` / `DIAGNOSIS_LABELS` / `test_transform_core.py` were NOT touched):
  * `_cw_cross_orientation`   — flavor (a): student net orientation flips (det differs) + wrong slide.
  * `_cw_wrong_rotation_angle` — flavor (b): correct net is a rotation, student a DIFFERENT rotation + wrong slide.
  * `_cw_wrong_reflection_line`— flavor (b): correct net is a reflection, student a DIFFERENT reflection + wrong slide.
- Sub-kind is chosen with weights = inverse of each sub-kind's realizability, derived from
  `PATTERNS` (not hardcoded), so the three come out roughly even even though
  cross-orientation is realizable on every problem while each flavor-(b) sub-kind needs a
  matching net orientation. If a sub-kind is not realizable for a problem it returns
  `None` and the caller retries with a fresh problem (existing behavior).
- No other category's injector changed.

Decision: chose weighted sub-kind selection (targets ~1/3 each) over per-record round-robin
to avoid threading sub-kind state through `inject`. Verified the asymptotic split over
4000 injections = 32.6% / 33.0% / 34.4% (cross / wrong_rotation_angle / wrong_reflection_line).

### Task 2 — tests (extended only; `test_transform_core.py` left frozen and green)
Added to `test_dataset.py`:
- `test_diagnose_completely_wrong_flavor_b_rotations` and `_reflections` — unit asserts that
  `diagnose` returns `completely_wrong` for the two flavor-(b) example inputs from the spec.
- `test_completely_wrong_spans_both_flavors_in_dataset` — builds the dataset and asserts the
  `completely_wrong` class contains BOTH at least one same-determinant (flavor b) and one
  different-determinant (flavor a) record (guards against regressing to a monoculture).
  Decision: made this test hermetic (builds via `dataset.build_records`) rather than reading
  a committed `data.jsonl`, so it needs no on-disk artifact.

### Task 3 — regenerate + verify
- Regenerated: `python3 -m transform_diagnosis --seed 0 --n 400 --min-count 30 --out transform_diagnosis_data`
  (deleted the stale output dir first so the 400 renders match the new records).
- Balance held: 50 per label (400 total); splits train 320 / val 40 / test 40.
- Byte-identical on a second run (data.jsonl `diff` clean), before and after cleanup.
- Full suite: **48 passed** (was 45; +3 new).
- `completely_wrong` sub-kind counts in the shipped n=50 slice: cross_orientation (flavor a) 23,
  wrong_rotation_angle (flavor b) 17, wrong_reflection_line (flavor b) 10 — i.e. flavor (a) 23 /
  flavor (b) 27; small-sample noise around the ~1/3 asymptotic each (see Task 1).

### Task 4 — repo cleanup
- **Backup (before any deletion):**
  * `git init` + `git -c user.…  commit -m "pre-cleanup snapshot"` (commit `1b95083`) — captures
    all normally-tracked source (incl. both old generators). NOTE: `GeoSym127K/` was an embedded
    clone, so git stored only a gitlink, not its contents.
  * Therefore also wrote a timestamped tarball: `../SLM_backup_2026-07-08.tar.gz` (7.8 MB),
    which DOES contain the GeoSym source with my earlier smoke-test edits
    (`GeoSym127K/scripts/main_mt_v1_test.py`, `config_smoke.json`), both old generators, and the
    package. Excluded regenerables: `.venv`, `node_modules`, `dist`, `results_smoke` (100 MB), `.git`, caches.
- **Reference check** (`generate_dataset` / `generate_diagnosis_data` / `GeoSym`): the only hits
  in the kept package are prose/comment mentions in `BUILD_LOG.md`, and "modeled on / adapted
  from" credits in `render.py` (line 8) and `geometry.py` (line 9). No imports. Safe to delete.
- **Deleted:**
  * `GeoSym127K/` (confirmed unused; its `.git` needed sandbox-disabled removal).
  * `model/generate_diagnosis_data.py` (superseded).
  * `model/generate_dataset.py` — **confirmed dead** (no imports/references anywhere except
    itself; not used by the kept package or notebooks) → deleted (not archived).
  * All `__pycache__/` dirs and `*.pyc`.
- **Left in place (not in scope):** `model/01_base_model_inference.ipynb`,
  `model/02_vision_base_model.ipynb`, and the now-orphaned sample outputs
  `model/dataset_sample/` and `model/diagnosis_sample/` (outputs of the deleted generators;
  harmless, can be removed on request).
- **Post-cleanup verification:** `python3 -m pytest transform_diagnosis -q` → **48 passed**;
  regeneration still byte-identical to the shipped `data.jsonl`. Nothing removed was needed.

### `transform_core` invariant
Untouched. Still the one and only implementation of transforms / grade / diagnose /
recover_map; no forks introduced.

---

## 2026-07-08 — reject degenerate student attempts (student_image == original)

### Defect (found during manual review of a `rotation_instead_of_reflection` render)
Some student attempts had a net map equal to the identity, so the student's answer sat
exactly on the untouched original (e.g. correct = "rotate 270 ccw, then reflect y=x";
injected student = "rotate 270 ccw, then rotate 90 ccw" → 360° = identity). The label was
still arithmetically correct (student did an orientation-preserving move where a reflection
was required), but the example is degenerate/confusing (the original looks "missing" under
the student shape). Audit: **17/400 records**, all `rotation_instead_of_reflection`; and
**0** real bugs (no record had `student_image == correct_image` with a non-`correct` label).

### Fix (generation + write-time + test; `diagnose`/labels/contract untouched)
- `errors.py::_accept` — now also rejects any candidate whose student image equals the
  original (`[tuple(p) for p in img] == [tuple(p) for p in problem.original]`). Since this
  is the single funnel all injectors pass through, it applies to every label; the
  verify-or-discard loop simply retries. (`correct` is unaffected: its image is the
  correct image, already guaranteed `!= original`.)
- `dataset.py::_assert_record` — added `assert rec["student_image"] != rec["original"]`
  next to the existing `correct_image != original` assertion (write-time enforcement).
- `test_dataset.py::test_student_image_never_equals_original` — regression test over
  `build_records` asserting no record has `student_image == original`.

### Regenerate + verify
- `python3 -m transform_diagnosis --seed 0 --n 400 --min-count 30 --out transform_diagnosis_data`
  (deleted stale output dir first so renders match).
- Degenerate `student_image == original`: **0/400** (was 17).
- Balance held: 50 per label; splits 320/40/40. Byte-identical on re-run (diff clean).
- **49 tests pass** (was 48; +1 regression test).
- All-labels spot check: 50 records each, MATCH=100 / agrees=50 / ok=50 / fails=0 — 400/400.

---

## 2026-07-08T18:55:02Z — per-record HINT field + compositional OOD split

Extended the generator with (1) a deterministic per-record tutor **hint** and (2) a
**compositional OOD split** that holds out the two rotation∘reflection compositions as a
test-only generalization slice. `transform_core` was **NOT** forked — all hint wording is
built through `transform_core.describe_transform`, and `transform_core.py` /
`test_transform_core.py` are byte-for-byte untouched (verified via `git status`).

### Part 1 — HINT field (new `hints.py`; deterministic, templated, token-graded)
- New module `transform_diagnosis/hints.py` with two functions, both taking
  `(label, rec)` where `rec` exposes the schema lists `correct_transform` /
  `student_transform`:
  * `hint_for` — builds the hint string.
  * `expected_hint_tokens` — the deterministically-expected substring token(s) a correct
    hint MUST contain, recomputed from the record's transforms.
  They agree by construction: `hint_for` embeds exactly the strings
  `expected_hint_tokens` recomputes. **No hand-typed axis/angle/vector literals** — every
  wording flows through `transform_core.describe_transform` (canonical ccw for rotations),
  and step identification uses the exact `Transform` math (parse → compare), not text.
- Design: the single-step error injectors mutate exactly one step, so the hint finds the
  one differing step (by math) and names the correct vs student step. `correct` restates
  both steps; `completely_wrong` names the correct vs student **net** map (orientation +
  net translation), since that class replaces both steps.
- Per-label hint templates (examples, one per label, from the shipped data):
  * `correct`: "Correct. Both steps are right: first translate 4 right, then rotate 180 degrees counterclockwise. You applied each move in the right order."
  * `reflection_instead_of_rotation`: "You reflected where a rotation was required. The step should have been rotate 180 degrees counterclockwise, not reflect across line y = x."
  * `rotation_instead_of_reflection`: "You rotated where a reflection was required. The step should have been reflect across y axis, not rotate 180 degrees counterclockwise."
  * `wrong_rotation_angle`: "Check the angle: the task used rotate 180 degrees counterclockwise, but you used rotate 90 degrees counterclockwise."
  * `wrong_reflection_line`: "Check the line of reflection: the task used reflect across x axis, but you used reflect across line y = x."
  * `wrong_translation`: "Check the translation: it should be translate 4 left, not translate 2 up."
  * `opposite_translation`: "You translated in the opposite direction: it should be translate 5 left, not translate 5 right."
  * `completely_wrong`: "Your whole answer is off: the correct net map is a rotation (rotate 270 degrees counterclockwise) with translation (0, -4), but yours is a rotation (rotate 180 degrees counterclockwise) with translation (5, -6). Both the transformation and the translation are wrong."
- **Schema change**: added `"hint"` immediately after `"label"` in `_SCHEMA_KEYS` (and thus
  in `_partial_record` and the key-order enforcement in `build_records`). Record order is
  now: id, num_vertices, original, correct_transform, correct_image, student_transform,
  student_image, label, **hint**, is_correct, split, render_path.
- **Write-time enforcement**: `dataset._assert_record` now asserts `hint` is a non-empty
  string AND contains every `hints.expected_hint_tokens(label, rec)` token, so a
  mis-worded hint fails loudly per record.

### Part 2 — compositional OOD split (held-out compositions)
- Held-out patterns (OOD, test-only): `("rotate","reflect")` and `("reflect","rotate")`.
- **Rationale / label-coverage constraint**: neither held-out pattern is the *exclusive*
  compatible pattern of any label, so all 8 labels keep ≥1 in-distribution pattern and
  stay trainable. (We must NOT hold out `("rotate","translate")` because it is the sole
  pattern for `reflection_instead_of_rotation` and `wrong_rotation_angle`.) A runtime
  `assert` in `errors.py` guards this invariant.
- In-distribution generation (train/val/test) uses ONLY the other 4 patterns:
  `("rotate","translate")`, `("reflect","translate")`, `("translate","rotate")`,
  `("translate","reflect")`. The OOD slice uses ONLY the 2 held-out patterns and therefore
  naturally covers only the 4 compatible labels: `correct`,
  `rotation_instead_of_reflection`, `wrong_reflection_line`, `completely_wrong`
  (unbalanced OOD is expected/fine).
- Implementation: `errors.py` derives `ID_COMPATIBLE_PATTERNS` / `OOD_COMPATIBLE_PATTERNS`
  (a disjoint split of `COMPATIBLE_PATTERNS`), `HELD_OUT_PATTERNS`,
  `IN_DISTRIBUTION_PATTERNS`, and `OOD_ELIGIBLE_LABELS` (all derived, not hardcoded).
  `dataset.build_records` builds the balanced in-distribution set (assigned train/val/test
  via `split_of`) then a separate OOD set (`split="ood"`) from the held-out patterns; a
  shared `_inject_partials` helper keeps the verify-or-retry logic identical for both.
  IDs are contiguous across the whole dataset (in-distribution first, then OOD), so
  `render_path` stays `renders/{id:06d}.png` and every record (incl. OOD) is rendered.
- **OOD size chosen: 120 per OOD-eligible label** (`--ood-per-label`, default 120) → 480
  OOD records. Within the recommended 100–150 range; a clean multiple. Deterministic
  function of the seed (separate `_OOD_SHUFFLE_SALT` interleave).
- `write_jsonl` now also emits `ood.jsonl`; `summary.json` / CLI printout emit id vs OOD
  counts, held-out & in-distribution patterns, and OOD-eligible labels.

### Part 3 — tests (extended; `test_transform_core.py` left frozen and green)
Added to `test_dataset.py`: hint presence + expected-token containment for every record;
hint schema position (after `label`); train/val/test contain ZERO held-out compositions;
OOD contains ONLY held-out compositions; all 8 labels in train + OOD covers exactly the 4
expected labels; OOD per-label count matches configuration. Updated the existing
split/balance/determinism tests to account for the new `ood` split (balance is asserted
over the in-distribution set only; the byte-identical file test now also diffs
`ood.jsonl`). Tests use a small `ood_per_label=12` for speed. **55 pass** (was 49; +6).

### Part 4 — regenerate + verify (seed 0, n 4000, min-count 30)
- Exact command (deleted stale `transform_diagnosis_data/` first so renders match):
  `MPLCONFIGDIR=/tmp/mpl python3 -m transform_diagnosis --seed 0 --n 4000 --min-count 30 --out transform_diagnosis_data`
- **Counts** — total **4480** records / **4480** renders:
  * splits: train 3200 / val 400 / test 400 / **ood 480**.
  * in-distribution per label: **500 each** (8 labels = 4000).
  * OOD per label: `correct` 120, `rotation_instead_of_reflection` 120,
    `wrong_reflection_line` 120, `completely_wrong` 120 (= 480); other 4 labels 0.
- **Verification — all PASS** (22 checks):
  data.jsonl == train+val+test+ood; split files match `split`; in-distribution balanced
  (500×8); OOD 120×4; **train/val/test have ZERO held-out compositions**; **OOD contains
  ONLY held-out compositions**; all 8 labels in train; OOD covers exactly the 4 expected
  labels; **0 degenerate** (student_image == original); every record label == `diagnose`
  of stored text; correct/student transforms grade to stored images;
  `is_correct == (label=='correct')`; all vertices integer & in-bounds; originals simple &
  asymmetric; `recover_map(original, correct_image)` == intended net; correct_image !=
  original; **every hint non-empty & contains its expected token(s)**; **render count ==
  total records (4480)**; ids contiguous 0..4479; every render_path exists;
  **byte-identical determinism** (fresh `build_records` == shipped `data.jsonl`).
- `MPLCONFIGDIR=/tmp/mpl python3 -m pytest transform_diagnosis -q` → **55 passed**.

### `transform_core` invariant
Untouched. Still the one and only implementation of transforms / grade / diagnose /
recover_map; hints and the OOD split are built strictly on top of it (hints via
`describe_transform`, the split via `COMPATIBLE_PATTERNS`). No forks introduced.

## Part 5 — Qwen3-VL chat format + scale-up (seed 0, n 12000, ood-per-label 300)

**Why:** the fine-tune target (`unsloth/Qwen3-VL-4B-Instruct-unsloth-bnb-4bit`) consumes a
two-turn conversation per record, and the notebook still carried a *stale* 7-label
taxonomy + a different JSON schema. Added the format bridge and regenerated larger.

- **New module `chat_format.py`** (dependency-light — no PIL/matplotlib). Emits per record:
  * user turn = `[{type:image, image:"<render_path>"}, {type:text, text:INSTRUCTION}]`
  * assistant turn = compact JSON `{"label","correct_transform","hint"}` (the training target).
  * `INSTRUCTION` states the RED/GREEN/BLUE colour key and lists the label vocabulary built
    from `transform_core.DIAGNOSIS_LABELS` (cannot drift). The rendered PNG already contains
    the GREEN correct image, so `correct_transform` is a genuine RED→GREEN output, not input.
- **Wiring:** `dataset.generate()` now also writes `train_chat / val_chat / test_chat /
  ood_chat .jsonl` (one conversation per line, image referenced by path; the notebook swaps
  each path for a decoded PIL image at load time).
- **Tests:** +4 in `test_dataset.py` (instruction lists exactly the 8 labels + colour key;
  conversation shape + image reference; assistant target parses & round-trips to the record;
  chat split files written with matching counts). **59 pass** (was 55). `transform_core.py`
  and the frozen `test_transform_core.py` untouched.
- **Regenerated** (wiped stale `renders/` first so ids match):
  `python3 -m transform_diagnosis --n 12000 --ood-per-label 300 --seed 0`
  * total **13200** records: train 9600 / val 1200 / test 1200 / **ood 1200**.
  * in-distribution **1500 each** (8 labels = 12000); OOD 300×4 (`correct`,
    `rotation_instead_of_reflection`, `wrong_reflection_line`, `completely_wrong`).
  * chat files mirror the split counts exactly; loader spot-check decoded 300 test
    conversations to RGB with valid targets.
- **Notebook `model/02_vision_base_model.ipynb` rewritten** to consume the format: load
  `*_chat.jsonl` → PIL, `apply_chat_template` format-fit proof, QLoRA
  (`get_peft_model` + `UnslothVisionDataCollator` + `SFTTrainer`), baseline + post-tune eval
  on test/OOD (parse-failure = wrong, parse-rate separate). GPU cells not run locally.

---

## v5 — enum/structured `correct_transform` (reframe transform recovery as CLASSIFICATION)

**Why:** CoT fine-tuning (v3cot) lifted LABEL accuracy a lot (balanced 0.45→0.63) but
`transform_match` (does the model name the EXACT correct transform?) stayed floored
(~0.05–0.08). A prior ablation showed transform fails even when the model is *given exact
coordinates*, so exact-transform recovery is reasoning/capacity bound, not perception/parse
bound. v5 hypothesis: reframe exact-transform recovery from a free-text GENERATION problem
into a CLASSIFICATION over a SMALL DISCRETE vocabulary, which this 4B does much better.

**Derived enum vocabulary** (enumerated from the generator, NOT invented — verified over the
in-distribution + OOD + contrastive + curriculum records):
- `step_type` ∈ {`rotation`, `reflection`, `translation`}
- rotation `param` ∈ {`rot_ccw_90`, `rot_180`, `rot_ccw_270`} — the 3 rotation matrices, all
  about the origin, canonicalized to CCW (so a stored "270 degrees clockwise" collapses to
  `rot_ccw_90`; `rot_ccw_270` == a 90° clockwise turn).
- reflection `param` ∈ {`reflect_x`, `reflect_y`, `reflect_y=x`, `reflect_y=-x`} — the 4 lines.
- translation → two integers `dx`, `dy` (observed [-8, 8] for correct answers).
- step counts: 1 (single-step curriculum) or 2 (the two-step problems).

**Schema:** `correct_transform` becomes a list of per-step dicts —
`{"type":"rotation","param":"rot_ccw_90"}` / `{"type":"reflection","param":"reflect_x"}` /
`{"type":"translation","dx":-7,"dy":0}`. `label` + `hint` are unchanged, and every v4
structured field is kept (`expected_operation_types` / `student_operation_types` /
`main_mismatch`). Built deterministically from the oracle transforms; the trace gains a
brace-free "Transform readout" line naming the RED→GREEN step types/params in the enum vocab.

- **New module `enum_transform.py`** — the single source of the enum vocabulary and a
  loss-less bridge to `transform_core`: `step_enum`/`seq_enum` (Transform→enum, derived from
  the public factories so it can never drift), `enum_step_to_transform`/`enum_to_transforms`
  (the inverse, for round-trip verification), `is_enum_seq` (format detection), and
  `steps_match` (EXACT per-step comparison). No new geometry.
- **`eval.py`** — `_transform_match` now dispatches on format: PROSE (v1–v4) keeps the exact
  semantic net-map comparison; ENUM (v5, when either side is enum) scores by EXACT per-step
  type+param / dx+dy, coercing a prose side to its canonical enum. So the SAME headline
  `transform_match_rate` measures "recovered the transform" across every version, and the
  real v5 eval path (enum prediction vs the frozen PROSE oracle) is handled automatically.
  **Transparency:** the enum metric is EXACT and *never looser* than net-map — a prediction
  that composes to the same net map via different per-step ops (possible for the OOD
  rotate∘reflect patterns) passes the prose metric but FAILS the enum metric. `label`/`hint`
  scoring is untouched.
- **`cot.py`** — added `enum_transform=True` (v5) and `transform_first` flags to
  `reasoning_trace`/`cot_target`/`to_cot_conversation`/`build_cot_rows`, plus
  `enum_target_obj`/`enum_json` (v4 object with `correct_transform` swapped to enum). v3cot
  (`structured=False`) and v4 (`structured=True`) paths are byte-for-byte unchanged.
- **`model/make_v5_data.py`** — assembles `train/val_v5_cot_chat.jsonl` by REUSING
  `make_v4_data.build_split` verbatim (same seed/salts/ids/renders/pools), so v5 == v4 DATA
  re-targeted to the enum schema (a clean ablation isolating the reframing). Each row is
  verified end-to-end: enum == `seq_enum(oracle)`, enum round-trips to the oracle net map,
  and the harness scores parse/label/transform/hint ALL True against the PROSE oracle.
- **Curriculum knobs (transform-first emphasis; neither forced):** `--transform-first`
  foregrounds `correct_transform` (emits it FIRST in the JSON; scored by key, so safe);
  `--mix` upweights the transform-DIVERSE contrastive/curriculum pools.
- **`model/sync_to_cluster.sh`** — `enum_transform.py` added to `HARNESS_FILES` (eval + cot
  import it); `make_v5_data.py` added to `SCRIPT_FILES`; v5 generate→train→eval block added
  (`bash -n` clean).
- **Tests:** new `test_enum_transform.py` (+19): vocabulary is the actual set; canonicalize +
  round-trip on every primitive and over the whole dataset; `steps_match` exact both
  directions (incl. a translation); eval dispatch + backward-compat; exact-not-looser
  demonstration; v5 target scores all-pass vs the prose oracle for every label; wrong-param
  caught; transform-first ordering; prose v3cot/v4 targets still score as before. Suite:
  **132 pass** (was 113). `transform_core.py` and the frozen `test_transform_core.py`
  untouched — no fork of the transform/grade/diagnose math.
- **Cluster sequence:** `python make_v5_data.py --n 9600 --val-n 400` →
  `sbatch --wrap "python train_cot.py --train-file …/train_v5_cot_chat.jsonl --val-file
  …/val_v5_cot_chat.jsonl --out ~/lora_adapters_v5 --output-dir outputs_v5 --epochs 3" -p
  mit_normal_gpu -G 1 -c 8 --mem=128G -t 06:00:00` (resubmit with `--resume` if the 6h cap
  hits) → `python eval_tuned.py --adapters ~/lora_adapters_v5 --sample 500 --seed 20260709
  --tag v5`.

