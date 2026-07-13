# Confidence estimates for the v6 eval

`model/confidence_intervals.py` puts statistical uncertainty on the already-saved eval
numbers. It is **post-hoc, CPU-only, stdlib-only** (no model, GPU, network, or scipy),
reads `results_*.json` / `records_*.jsonl` **read-only**, and writes nothing unless you
pass `--json-out`.

It reports two things:

1. **Wilson score CI** on every accuracy **rate** (`k/n`), e.g. `both_nets_match_rate`.
2. **Paired significance** between two arms that scored the **same record ids** (base vs
   tuned vs hintfix; image vs image+coords): the paired delta `rate(B) - rate(A)`,
   **McNemar's test** (exact binomial for small discordant counts, else chi-square with
   continuity correction), and a **bootstrap CI** on the delta (fixed seed).

### Method note (matches the frozen artifacts)

The Wilson interval is the **same** one that produced the `wilson_95` fields in
`results/v6_final/FINAL_RESULTS_SUMMARY.json` and the intervals quoted in
`FINAL_RESULTS.md`: the two-sided 95% formula in
[`audit_v6_predictions.wilson`](audit_v6_predictions.py) with `z = 1.959963984540054`.
`confidence_intervals.wilson_interval` reproduces it exactly (a test asserts equality),
and `--confidence` selects other levels. McNemar and the bootstrap are **new** (the audit
only stored the discordance counts they consume).

> **CAVEAT (honesty rail).** These intervals quantify **finite-sample (evaluation)
> variability at a FIXED model checkpoint and training seed**. They do **not** capture
> training-seed or checkpoint variance: a different fine-tune of the same recipe could
> land outside them. An observed 100% is not proof of perfection — for 500/500 the Wilson
> lower bound is 99.2%.

## CLI

```
python3 model/confidence_intervals.py [INPUTS ...] [--pair A B] [--count LABEL K N] [options]
```

- `INPUTS`: `results_*.json` and/or `records_*.jsonl` -> per-metric Wilson CIs. Records are
  preferred (exact `k`); a results aggregate recovers `k = round(rate * n)`, `n` from the
  metric's `*_available` field. Detected by extension.
- `--pair A B` (repeatable): two `records_*.jsonl` sharing ids -> delta, McNemar, bootstrap.
- `--count LABEL K N` (repeatable): Wilson CI for a raw `k/n` from a separate audit (e.g.
  the strict hint-leak rate), no file needed.
- `--metric M` (repeatable): restrict metrics. Metrics: `parse_ok`, `correct_net_ok`,
  `student_net_ok`, `both_nets_ok`, `label_ok`, `derived_label_ok`, `hint_ok`,
  `hint_exact_ok`, `step_sequence_exact_ok`.
- `--confidence 0.95` | `--bootstrap 10000` | `--seed 20260712` | `--exact-max 40`.
- `--json-out PATH`: machine-readable dump (refuses to write under any `results/` dir).

`n` excludes `null`/not-applicable rows, identical to `eval.aggregate`'s `*_available`.

## Validated locally on `results/v6_final/` (frozen base-vs-tuned)

These match `FINAL_RESULTS.md` to the published digits.

```
$ python3 model/confidence_intervals.py results/v6_final/results_v6_4b_image_coords_test.json
metric                       k     n      rate       95% CI (Wilson)     as %            CI %
correct_net_ok             493   500   0.98600   [0.97139, 0.99320]    98.6%   [97.1–99.3]
student_net_ok             498   500   0.99600   [0.98553, 0.99890]    99.6%   [98.6–99.9]
both_nets_ok               493   500   0.98600   [0.97139, 0.99320]    98.6%   [97.1–99.3]
label_ok                   498   500   0.99600   [0.98553, 0.99890]    99.6%   [98.6–99.9]

# image-only ablation (records_v6_4b_image_test.jsonl): both_nets 192/500 -> 38.4% [34.2–42.7]
# OOD hero (results_v6_4b_image_coords_ood.json):       correct  500/500 -> 100.0% [99.2–100.0]
```

Paired base -> tuned (image+coords, test) is overwhelmingly significant:

```
$ python3 model/confidence_intervals.py \
    --pair results/v6_final/records_v6_4b_base_image_coords_test.jsonl \
           results/v6_final/records_v6_4b_image_coords_test.jsonl --metric both_nets_ok
metric                 rate_A   rate_B  delta_pp   disc(A/B)        McNemar p (method)       boot 95% CI(pp)
both_nets_ok           0.0000   0.9860    +98.60       0/493       8.63e-109 (chi2_cc)   [+97.40, +99.60]
```

(base 0/500 -> tuned 493/500; 493 discordant pairs all favor tuned.)

---

## Hand-off: run it on the hintfix + golden results

Both run on the **ORCD login node (CPU, no GPU)** after `confidence_intervals.py` is synced
to `$HOME` by `model/sync_to_cluster.sh`, **or** locally after copying the result files
back. Nothing here touches `results/v6_final/` or any adapter.

### (a) Hintfix results (after the [HINT_FIX_RUNBOOK](HINT_FIX_RUNBOOK.md) eval finishes)

On the cluster (`cd ~`), Wilson CI for every rate in all four hintfix cells:

```bash
python confidence_intervals.py \
  results_v6_4b_image_hintfix_test.json  results_v6_4b_image_hintfix_ood.json \
  results_v6_4b_image_coords_hintfix_test.json results_v6_4b_image_coords_hintfix_ood.json
```

Headline **hint** metric with a CI (family-relevant + no coordinate leak):

```bash
python confidence_intervals.py \
  records_v6_4b_image_coords_hintfix_test.jsonl records_v6_4b_image_coords_hintfix_ood.jsonl \
  --metric hint_ok --metric hint_exact_ok
```

**The leak-rate headline (`~0.96 -> ~0`) with CIs.** The strict-leak rate is defined by the
safety audit ([HINT_FIX_RUNBOOK](HINT_FIX_RUNBOOK.md) step 4b), so feed its `k/n` directly
(frozen leak was 1919/2000 = 96.0%; use the hintfix run's printed `strict_leak` x n):

```bash
python confidence_intervals.py \
  --count strict_leak_frozen  1919 2000 \
  --count strict_leak_hintfix "$K_LEAK" "$N"     # K_LEAK = round(strict_leak * N)
# frozen -> 95.95% [95.0–96.7];  a ~0 hintfix rate lands with an upper bound near ~0.6%.
```

**Does hintfix keep geometry within the CI of tuned?** The frozen tuned records live in
this repo (`results/v6_final/`), so copy the four hintfix `records_*.jsonl` back and pair
them locally (per-record ids match: same `--seed 20260709`, same test/ood):

```bash
# from your laptop, copy the new per-record files next to the frozen ones (your usual auth):
rsync -a ikchen@orcd-login.mit.edu:'~/records_v6_4b_*_hintfix_{test,ood}.jsonl' /tmp/hintfix/

cd /Users/isabellachen/projects/SLM
# tuned (A) vs hintfix (B): expect delta ~ 0 with a CI straddling 0 and a NON-significant
# McNemar -> geometry preserved within noise; the hint text changed, not the maps.
python3 model/confidence_intervals.py \
  --pair results/v6_final/records_v6_4b_image_coords_test.jsonl /tmp/hintfix/records_v6_4b_image_coords_hintfix_test.jsonl \
  --pair results/v6_final/records_v6_4b_image_test.jsonl        /tmp/hintfix/records_v6_4b_image_hintfix_test.jsonl \
  --metric correct_net_ok --metric student_net_ok --metric both_nets_ok \
  --metric label_ok --metric derived_label_ok
# base (A) vs hintfix (B) sanity (expect large, significant deltas like base-vs-tuned):
python3 model/confidence_intervals.py \
  --pair results/v6_final/records_v6_4b_base_image_coords_test.jsonl /tmp/hintfix/records_v6_4b_image_coords_hintfix_test.jsonl \
  --metric both_nets_ok
```

Read "within CI of tuned" as: the tuned-vs-hintfix geometry delta CI **includes 0** and
McNemar is **not** significant. (The numbers themselves need the hintfix files, which come
from the cluster run — not verifiable on the laptop yet.)

### (b) Golden-set results (see [GOLDEN_SET.md](GOLDEN_SET.md), 6 cells, n=160)

On the cluster (`cd ~`) after the golden eval writes `results_/records_*_golden.*`:

```bash
# Wilson CI per rate for all six golden cells:
python confidence_intervals.py results_v6_4b_*_golden.json

# base vs tuned vs hintfix on the SAME 160 golden ids (image+coords shown; repeat for image):
python confidence_intervals.py \
  --pair records_v6_4b_image_coords_base_golden.jsonl   records_v6_4b_image_coords_tuned_golden.jsonl \
  --pair records_v6_4b_image_coords_tuned_golden.jsonl  records_v6_4b_image_coords_hintfix_golden.jsonl \
  --metric both_nets_ok --metric label_ok --metric hint_ok
```

The first `--pair` is the learned-behavior jump (base -> tuned); the second asks whether
hintfix holds geometry vs tuned (delta CI includes 0, McNemar n.s.) while `hint_ok` stays
high. At n=160 the CIs are wider than the 500-case test/ood cells — expected.

## Related

- Frozen results + honesty rail: [`results/v6_final/FINAL_RESULTS.md`](../results/v6_final/FINAL_RESULTS.md)
- Wilson source / geometry audit: [`audit_v6_predictions.py`](audit_v6_predictions.py)
- Hint-leak fix + safety audit: [`HINT_FIX_RUNBOOK.md`](HINT_FIX_RUNBOOK.md)
- Golden held-out set: [`GOLDEN_SET.md`](GOLDEN_SET.md)
