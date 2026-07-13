# Deterministic v6 hint-safety audit

Status: **PASS (audit completed)**. This status means the audit ran over all 2,000 frozen tuned `full` outputs and passed its consistency assertions; it does not mean every hint was safe.

## Contract and rubric boundary

- The actual v6 `full` prompt asks for a **short Socratic hint**, but unlike the legacy prompt it does **not** say “without stating coordinates.” This is a specification gap, so coordinate disclosures below are not mislabeled as direct v6 prompt violations.
- The existing `hint_ok` rubric requires at least one oracle-relevant operation family and rejects only coordinate pairs left after removing exact expected tokens. Exact expected tokens are therefore *sanctioned* by that implementation. For `completely_wrong`, this explicitly sanctions both oracle net-translation pairs; other labels may sanction diagonal primitive-translation pairs.
- This audit additionally flags any exact oracle operation/map parameter or translation value repeated in the hint. `safe_useful` is the conservative tutoring measure: valid hint + relevant family + no rubric-unsanctioned coordinate pair + no exact answer/map/value disclosure. It is intentionally stricter than stored `hint_ok` and is not a replacement for exact geometry scoring.

## Overall findings

- Final JSON parse: **2000/2000 (100.0%)**; valid hint field: **2000/2000 (100.0%)**.
- Operation-family relevance: **1993/2000 (99.7%)**.
- Any explicit coordinate pair: **527/2000 (26.4%)**; rubric-unsanctioned coordinate leak: **253/2000 (12.7%)**.
- Exact canonical-linear disclosure: **1750/2000 (87.5%)**; exact translation-value disclosure: **492/2000 (24.6%)**; exact full canonical-map disclosure: **312/2000 (15.6%)**; any exact answer/map/value disclosure: **1919/2000 (96.0%)**.
- Conservative combined safe/useful: **28/2000 (1.4%)**.
- Stored `hint_ok` agreement: **2000/2000 (100.0%)**; stored `hint_exact_ok` agreement: **2000/2000 (100.0%)**.

Exact geometry metrics remain authoritative and separate; this report never uses hint quality to override map correctness.

## By modality and split

| Cell | Family relevant | Any coordinate pair | Unsanctioned pair leak | Exact answer/value | Safe/useful | Stored hint_ok |
|---|---:|---:|---:|---:|---:|---:|
| `image_test` | 498/500 (99.6%) | 148/500 (29.6%) | 133/500 (26.6%) | 439/500 (87.8%) | 19/500 (3.8%) | 366/500 (73.2%) |
| `image_ood` | 495/500 (99.0%) | 96/500 (19.2%) | 69/500 (13.8%) | 495/500 (99.0%) | 0/500 (0.0%) | 426/500 (85.2%) |
| `image_coords_test` | 500/500 (100.0%) | 158/500 (31.6%) | 49/500 (9.8%) | 485/500 (97.0%) | 9/500 (1.8%) | 451/500 (90.2%) |
| `image_coords_ood` | 500/500 (100.0%) | 125/500 (25.0%) | 2/500 (0.4%) | 500/500 (100.0%) | 0/500 (0.0%) | 498/500 (99.6%) |

## By cell and true label

| Cell | Label | n | Family relevant | Exact answer/value | Safe/useful |
|---|---|---:|---:|---:|---:|
| `image_test` | `correct` | 66 | 66/66 (100.0%) | 66/66 (100.0%) | 0/66 (0.0%) |
| `image_test` | `reflection_instead_of_rotation` | 59 | 59/59 (100.0%) | 59/59 (100.0%) | 0/59 (0.0%) |
| `image_test` | `rotation_instead_of_reflection` | 65 | 64/65 (98.5%) | 64/65 (98.5%) | 1/65 (1.5%) |
| `image_test` | `wrong_rotation_angle` | 76 | 76/76 (100.0%) | 76/76 (100.0%) | 0/76 (0.0%) |
| `image_test` | `wrong_reflection_line` | 57 | 56/57 (98.2%) | 56/57 (98.2%) | 0/57 (0.0%) |
| `image_test` | `wrong_translation` | 66 | 66/66 (100.0%) | 30/66 (45.5%) | 4/66 (6.1%) |
| `image_test` | `opposite_translation` | 47 | 47/47 (100.0%) | 26/47 (55.3%) | 13/47 (27.7%) |
| `image_test` | `completely_wrong` | 64 | 64/64 (100.0%) | 62/64 (96.9%) | 1/64 (1.6%) |
| `image_ood` | `correct` | 111 | 110/111 (99.1%) | 111/111 (100.0%) | 0/111 (0.0%) |
| `image_ood` | `rotation_instead_of_reflection` | 134 | 132/134 (98.5%) | 132/134 (98.5%) | 0/134 (0.0%) |
| `image_ood` | `wrong_reflection_line` | 129 | 128/129 (99.2%) | 128/129 (99.2%) | 0/129 (0.0%) |
| `image_ood` | `completely_wrong` | 126 | 125/126 (99.2%) | 124/126 (98.4%) | 0/126 (0.0%) |
| `image_coords_test` | `correct` | 66 | 66/66 (100.0%) | 66/66 (100.0%) | 0/66 (0.0%) |
| `image_coords_test` | `reflection_instead_of_rotation` | 59 | 59/59 (100.0%) | 59/59 (100.0%) | 0/59 (0.0%) |
| `image_coords_test` | `rotation_instead_of_reflection` | 65 | 65/65 (100.0%) | 65/65 (100.0%) | 0/65 (0.0%) |
| `image_coords_test` | `wrong_rotation_angle` | 76 | 76/76 (100.0%) | 76/76 (100.0%) | 0/76 (0.0%) |
| `image_coords_test` | `wrong_reflection_line` | 57 | 57/57 (100.0%) | 57/57 (100.0%) | 0/57 (0.0%) |
| `image_coords_test` | `wrong_translation` | 66 | 66/66 (100.0%) | 60/66 (90.9%) | 1/66 (1.5%) |
| `image_coords_test` | `opposite_translation` | 47 | 47/47 (100.0%) | 38/47 (80.9%) | 8/47 (17.0%) |
| `image_coords_test` | `completely_wrong` | 64 | 64/64 (100.0%) | 64/64 (100.0%) | 0/64 (0.0%) |
| `image_coords_ood` | `correct` | 111 | 111/111 (100.0%) | 111/111 (100.0%) | 0/111 (0.0%) |
| `image_coords_ood` | `rotation_instead_of_reflection` | 134 | 134/134 (100.0%) | 134/134 (100.0%) | 0/134 (0.0%) |
| `image_coords_ood` | `wrong_reflection_line` | 129 | 129/129 (100.0%) | 129/129 (100.0%) | 0/129 (0.0%) |
| `image_coords_ood` | `completely_wrong` | 126 | 126/126 (100.0%) | 126/126 (100.0%) | 0/126 (0.0%) |

## Representative flagged hints

- `image_coords_ood` ID `24006` (`completely_wrong`): states an exact oracle linear map; states an exact oracle/primitive translation value; repeats at least one exact oracle hint token. Excerpt: “Your whole answer is off: the correct net map is a reflection (reflect across line y = -x) with translation (0, 0), but yours is a rotation (rotate 90 degrees counterclockwise) wit”
- `image_coords_ood` ID `24007` (`wrong_reflection_line`): states an exact oracle linear map. Excerpt: “Check the line of reflection: the task used reflect across line y = -x, but you used reflect across line y = x.”
- `image_coords_ood` ID `24010` (`rotation_instead_of_reflection`): states an exact oracle linear map. Excerpt: “You rotated where a reflection was required. The step should have been reflect across line y = x, not rotate 180 degrees counterclockwise.”
- `image_coords_ood` ID `24018` (`completely_wrong`): states an exact oracle linear map; states an exact oracle/primitive translation value; repeats at least one exact oracle hint token. Excerpt: “Your whole answer is off: the correct net map is a reflection (reflect across x axis) with translation (0, 0), but yours is a reflection (reflect across line y = x) with translatio”
- `image_coords_ood` ID `24023` (`completely_wrong`): states an exact oracle linear map; states an exact oracle/primitive translation value; repeats at least one exact oracle hint token. Excerpt: “Your whole answer is off: the correct net map is a reflection (reflect across x axis) with translation (0, 0), but yours is a rotation (rotate 180 degrees counterclockwise) with tr”
- `image_coords_ood` ID `25622` (`completely_wrong`): contains coordinate pair not sanctioned by existing exact tokens; states an exact oracle linear map; states an exact oracle/primitive translation value; repeats at least one exact oracle hint token. Excerpt: “Your whole answer is off: the correct net map is a reflection (reflect across y axis) with translation (0, 0), but yours is a reflection (reflect across x axis) with translation (8”
- `image_coords_ood` ID `25853` (`completely_wrong`): contains coordinate pair not sanctioned by existing exact tokens; states an exact oracle linear map; states an exact oracle/primitive translation value; repeats at least one exact oracle hint token. Excerpt: “Your whole answer is off: the correct net map is a reflection (reflect across x axis) with translation (0, 0), but yours is a rotation (rotate 90 degrees counterclockwise) with tra”
- `image_coords_test` ID `52` (`wrong_reflection_line`): contains coordinate pair not sanctioned by existing exact tokens; states an exact oracle linear map; states an exact oracle/primitive translation value; repeats at least one exact oracle hint token. Excerpt: “Your whole answer is off: the correct net map is a reflection (reflect across line y = x) with translation (7, 0), but yours is a reflection (reflect across line y = -x) with trans”
- `image_ood` ID `24049` (`correct`): does not mention a required operation family; states an exact oracle/primitive translation value. Excerpt: “Correct. Your work is right: translate 0 left and 0 down. You applied each move in the right order.”
- `image_ood` ID `24091` (`wrong_reflection_line`): does not mention a required operation family. Excerpt: “Check the angle: the task used rotate 90 degrees counterclockwise, but you used rotate 180 degrees counterclockwise.”
- `image_ood` ID `24335` (`rotation_instead_of_reflection`): does not mention a required operation family. Excerpt: “You translated in the opposite direction: it should be translate 1 down, not translate 1 up.”

## Stored-metric agreement

- Independent `hint_ok` disagreements: **0 / 2000**.
- Independent `hint_exact_ok` disagreements: **0 / 2000**.
- The independent parser also agreed with stored `parse_ok` on every row.

## Reproduction

```bash
PYTHONDONTWRITEBYTECODE=1 python3 results/overnight/audit_hint_safety.py
```

The script asserts four 500-row cells, unique IDs, paired ID order by split, oracle coverage, independent/stored parse agreement, summary arithmetic, and JSON round-trip validity. Frozen predictions and source oracles are read-only.
