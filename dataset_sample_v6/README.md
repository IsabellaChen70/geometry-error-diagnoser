# Final v6 canonical-net dataset sample

This is a reviewer-sized sample of the final `v6.net-affine.1` data and chat
schemas. It contains 24 balanced training/source examples: three examples for
each of the eight diagnosis labels.

Use this directory to inspect the final v6
format. The sibling [`dataset_sample/`](../dataset_sample/) is intentionally
preserved as the legacy/source-format sample.

## Scope and provenance

The 24 records and PNGs come from `../dataset_sample/train_sample.jsonl` and
`../dataset_sample/images/`. This package is **not** claimed to be a
byte-for-byte sample from the 9,600-row ORCD mixed curriculum. Every row here is
a source-pool example selected earlier for label balance.

Generation follows the production source-record path in
`model/make_v6_transform_data.py` and uses the read-only production helpers in
`transform_diagnosis.v6_format`:

- source provenance is attached using the production source-copy helper;
- `augment_record` recomputes `correct_net` and `student_net` from the legacy
  transform sequences and verifies both maps against the stored polygon
  geometry and diagnosis label;
- the production chat builder emits every prompt and assistant target;
- the original PNG bytes and existing hint text are preserved.

`id` is sample-local (`0` through `23`). `source_id` and `source_split` retain
the corresponding legacy record identity, and `v6_pool` is `source`. Image
paths are package-relative so the directory remains portable.

## Files

- `images/` — 24 byte-for-byte copies of the source PNGs.
- `train_v6.jsonl` — 24 v6 oracle rows with `schema_version`, canonical
  `correct_net` and `student_net`, source provenance, labels, and local image
  paths.
- `train_v6_{image,image_coords}_{correct,student,both,full}_chat.jsonl` —
  eight production-generated chat files, 24 rows each.
- `manifest_v6.json` — provenance, counts, generation checks, and SHA-256
  metadata for package files.

The four staged task targets have exactly these keys:

- `correct`: `correct_net`
- `student`: `student_net`
- `both`: `correct_net`, `student_net`
- `full`: `correct_net`, `student_net`, `label`, `hint`

Each canonical net has exactly `linear`, `tx`, and `ty`. The `image` and
`image_coords` variants use the same image; the latter additionally includes
the exact corresponding RED, GREEN, and BLUE vertices in the production prompt.

## Known hint-disclosure limitation

The `full` targets deliberately preserve the hints used by the current
training data. Many state exact operations, map parameters, or translation
values. They must not be presented as safely Socratic tutoring hints.

This sample does not sanitize those targets because doing so would
misrepresent trained behavior. The v6 prompt asks for a short Socratic hint but
does not explicitly prohibit answer or coordinate disclosure. See the
[`HINT_SAFETY_AUDIT.md`](../results/overnight/HINT_SAFETY_AUDIT.md) for the
full deterministic audit and rubric boundary.

## Integrity

The manifest records the source JSONL checksum, copied-image checksums, exact
label counts, chat/target verification totals, and package file checksums. The
zip uses lexicographic entry order, fixed timestamps and permissions, and
contains only paths rooted at `dataset_sample_v6/`.
