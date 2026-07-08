"""run_all — resilient, unattended end-to-end pipeline.

Stages (each wrapped in retry-with-backoff and logged to BUILD_LOG.md):

  1. import_check      — import every module + a tiny transform_core smoke test
  2. generate          — build + assert records, write JSONL, render PNGs (renders skip
                         files that already exist, so a restart never re-renders)
  3. determinism_check — rebuild in-process and diff against the written data.jsonl
  4. acceptance_tests  — run the full pytest suite (all six acceptance tests)

Transient / billing / rate errors ("unpaid invoice", "insufficient credits", "payment
required", "quota exceeded", "rate limit", timeouts, network) are retried indefinitely
with exponential backoff (5s, 15s, 45s, ... capped ~5min). Other errors are retried a
few times and then re-raised so a genuine bug surfaces. Safe to leave running overnight.

Usage::

    python3 transform_diagnosis/run_all.py
    python3 -m transform_diagnosis.run_all --out data_out --n 400 --min-count 30
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import subprocess
import sys
import time
import traceback

# Allow running as a loose script (`python3 transform_diagnosis/run_all.py`).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transform_diagnosis import dataset, transform_core as tc  # noqa: E402

BUILD_LOG = os.path.join(_THIS_DIR, "BUILD_LOG.md")

_TRANSIENT_MARKERS = (
    "unpaid invoice", "insufficient credits", "payment required", "quota exceeded",
    "rate limit", "ratelimit", "too many requests", "timed out", "timeout",
    "temporarily unavailable", "connection reset", "connection aborted",
    "connection refused", "network is unreachable", "service unavailable",
)

_BACKOFF = (5, 15, 45, 135, 300)


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"- {_now()} — {msg}"
    print(f"[run_all] {msg}", flush=True)
    try:
        with open(BUILD_LOG, "a") as f:
            f.write(line + "\n")
    except Exception:  # logging must never crash the pipeline
        pass


def _is_transient(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def retry(stage: str, fn, *, max_nontransient: int = 4):
    """Run ``fn`` with exponential backoff. Transient/billing errors retry forever;
    other errors retry up to ``max_nontransient`` times then re-raise."""
    attempt = 0
    while True:
        attempt += 1
        try:
            log(f"stage '{stage}': attempt {attempt} starting")
            result = fn()
            log(f"stage '{stage}': OK on attempt {attempt}")
            return result
        except Exception as exc:  # noqa: BLE001 - resilience is the whole point
            transient = _is_transient(exc)
            delay = _BACKOFF[min(attempt - 1, len(_BACKOFF) - 1)]
            tb = traceback.format_exc().strip().splitlines()[-1]
            log(f"stage '{stage}': attempt {attempt} FAILED "
                f"({'transient' if transient else 'error'}): {tb}")
            if not transient and attempt >= max_nontransient:
                log(f"stage '{stage}': giving up after {attempt} non-transient failures")
                raise
            log(f"stage '{stage}': backing off {delay}s then retrying")
            time.sleep(delay)


# --------------------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------------------

def import_check() -> None:
    from transform_diagnosis import (  # noqa: F401
        dataset as _d, errors as _e, geometry as _g, problems as _p,
        reference_generator as _r, render as _rn, transform_core as _t,
    )
    # tiny smoke test of the canonical semantics
    assert _t.compose([_t.rotate(90, "ccw"), _t.translate(3, -2)]).apply([(1, 0)]) == [(3, -1)]
    assert _t.diagnose([(0, 0)], [_t.rotate(90, "ccw")], [_t.reflect("x")]) == "reflection_instead_of_rotation"
    assert set(_t.DIAGNOSIS_LABELS) == {
        "correct", "reflection_instead_of_rotation", "rotation_instead_of_reflection",
        "wrong_rotation_angle", "wrong_reflection_line", "wrong_translation",
        "opposite_translation", "completely_wrong",
    }
    log(f"import_check: matplotlib_available={_rn.MATPLOTLIB_AVAILABLE}")


def generate_stage(args) -> dict:
    result = dataset.generate(
        out_dir=args.out, seed=args.seed, n=args.n, min_count=args.min_count,
        split_fracs=args.split, render_subdir=args.render_subdir,
        do_render=not args.no_render,
    )
    lc = result["label_counts"]
    below = {l: c for l, c in lc.items() if c < args.min_count}
    if below:
        raise RuntimeError(f"labels below min_count={args.min_count}: {below}")
    log(f"generate: {len(result['records'])} records; labels={lc}; "
        f"splits={result['split_counts']}; new_renders={result['rendered']}")
    return result


def determinism_stage(args) -> None:
    path = os.path.join(args.out, "data.jsonl")
    with open(path) as f:
        on_disk = [line.rstrip("\n") for line in f if line.strip()]
    records, _ = dataset.build_records(
        args.seed, args.n, args.min_count, args.split, args.render_subdir)
    rebuilt = [json.dumps(r) for r in records]
    if on_disk != rebuilt:
        raise RuntimeError("determinism check FAILED: rebuilt records differ from data.jsonl")
    log(f"determinism: re-run byte-identical ({len(rebuilt)} lines match)")


def tests_stage() -> None:
    cmd = [sys.executable, "-m", "pytest", _THIS_DIR, "-q"]
    log(f"acceptance_tests: running {' '.join(cmd)}")
    proc = subprocess.run(cmd, cwd=_REPO_ROOT, capture_output=True, text=True)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-15:])
    print(proc.stdout, flush=True)
    if proc.stderr:
        print(proc.stderr, flush=True)
    if proc.returncode != 0:
        log(f"acceptance_tests: FAILED (rc={proc.returncode})\n{tail}")
        raise RuntimeError(f"pytest failed with rc={proc.returncode}")
    summary = (proc.stdout.strip().splitlines() or ["(no output)"])[-1]
    log(f"acceptance_tests: PASSED — {summary}")


# --------------------------------------------------------------------------------------
# Entry
# --------------------------------------------------------------------------------------

def _parse_split(text: str):
    parts = [float(p) for p in text.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("--split needs three fractions, e.g. 0.8,0.1,0.1")
    total = sum(parts)
    return tuple(p / total for p in parts)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_all", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--out", default=os.path.join(_REPO_ROOT, "transform_diagnosis_data"))
    ap.add_argument("--render", dest="render_subdir", default="renders",
                    help="render subdirectory name under --out")
    ap.add_argument("--split", type=_parse_split, default=(0.8, 0.1, 0.1))
    ap.add_argument("--min-count", type=int, default=30, dest="min_count")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--skip-tests", action="store_true", help="skip the pytest stage")
    return ap


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    args.render_subdir = os.path.basename(args.render_subdir.rstrip("/")) or "renders"
    os.makedirs(args.out, exist_ok=True)

    log("=" * 70)
    log(f"run_all START — seed={args.seed} n={args.n} min_count={args.min_count} "
        f"out={os.path.abspath(args.out)} render={not args.no_render}")

    retry("import_check", import_check)
    result = retry("generate", lambda: generate_stage(args))
    retry("determinism_check", lambda: determinism_stage(args))
    if not args.skip_tests:
        retry("acceptance_tests", tests_stage)
    else:
        log("acceptance_tests: skipped (--skip-tests)")

    # Persist a machine-readable summary next to the data.
    summary = {
        "when": _now(), "seed": args.seed, "n": args.n, "min_count": args.min_count,
        "total_records": len(result["records"]),
        "label_counts": result["label_counts"], "split_counts": result["split_counts"],
        "out_dir": os.path.abspath(args.out),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    log(f"run_all DONE — {summary['total_records']} records at {summary['out_dir']}")
    print("\n[run_all] SUCCESS. Dataset + renders written; all acceptance tests green "
          "(unless --skip-tests).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
