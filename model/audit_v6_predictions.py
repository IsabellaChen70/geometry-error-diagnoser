#!/usr/bin/env python3
"""Independent geometric audit for saved v6 prediction records.

This intentionally does not call ``transform_diagnosis.eval``.  It parses the
last JSON object itself, applies each predicted affine map directly to the
observable RED vertices, and compares the produced points with GREEN/BLUE.
It also checks saved evaluator booleans, exact train/eval geometry overlap,
Wilson confidence intervals, and paired model disagreements.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple


MATRICES = {
    "identity": ((1, 0), (0, 1)),
    "rot_ccw_90": ((0, -1), (1, 0)),
    "rot_180": ((-1, 0), (0, -1)),
    "rot_ccw_270": ((0, 1), (-1, 0)),
    "reflect_x_axis": ((1, 0), (0, -1)),
    "reflect_y_axis": ((-1, 0), (0, 1)),
    "reflect_y_eq_x": ((0, 1), (1, 0)),
    "reflect_y_eq_neg_x": ((0, -1), (-1, 0)),
}
NET_KEYS = frozenset(("linear", "tx", "ty"))
AUDIT_TO_STORED = {
    "correct_net_ok": "correct_net_ok",
    "student_net_ok": "student_net_ok",
    "both_nets_ok": "both_nets_ok",
    "label_ok": "label_ok",
    "derived_label_ok": "derived_label_ok",
}


def load_jsonl(path: str) -> list:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_last_object(text: str) -> Optional[dict]:
    """Independently extract the last valid JSON object from generated text."""
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except (TypeError, json.JSONDecodeError):
        pass
    if not isinstance(text, str):
        return None
    decoder = json.JSONDecoder()
    found = None
    best = (-1, -1)
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        rank = (index + consumed, -index)
        if isinstance(value, dict) and rank > best:
            found = value
            best = rank
    return found


def validate_net(value: object) -> dict:
    if not isinstance(value, Mapping) or frozenset(value) != NET_KEYS:
        raise ValueError("net must contain exactly linear/tx/ty")
    linear = value["linear"]
    tx, ty = value["tx"], value["ty"]
    if linear not in MATRICES:
        raise ValueError(f"unknown linear map {linear!r}")
    if isinstance(tx, bool) or not isinstance(tx, int):
        raise ValueError("tx must be an integer")
    if isinstance(ty, bool) or not isinstance(ty, int):
        raise ValueError("ty must be an integer")
    return {"linear": linear, "tx": tx, "ty": ty}


def apply_net(value: object, points: Sequence[Sequence[int]]) -> list:
    net = validate_net(value)
    (a, b), (c, d) = MATRICES[net["linear"]]
    tx, ty = net["tx"], net["ty"]
    return [
        [a * int(x) + b * int(y) + tx, c * int(x) + d * int(y) + ty]
        for x, y in points
    ]


def geometry_matches(value: object, source: Sequence, target: Sequence) -> bool:
    try:
        produced = apply_net(value, source)
    except (TypeError, ValueError):
        return False
    return produced == [[int(x), int(y)] for x, y in target]


def determinant(linear: str) -> int:
    (a, b), (c, d) = MATRICES[linear]
    return a * d - b * c


def diagnose_nets(correct: object, student: object) -> str:
    """Independent copy of the documented net-map diagnosis decision table."""
    cnet, snet = validate_net(correct), validate_net(student)
    mc, ms = cnet["linear"], snet["linear"]
    tc = (cnet["tx"], cnet["ty"])
    ts = (snet["tx"], snet["ty"])
    if mc == ms and tc == ts:
        return "correct"
    if mc == ms:
        if tc != (0, 0) and ts == (-tc[0], -tc[1]):
            return "opposite_translation"
        return "wrong_translation"
    if determinant(mc) != determinant(ms):
        if tc != ts:
            return "completely_wrong"
        return (
            "reflection_instead_of_rotation"
            if determinant(mc) == 1
            else "rotation_instead_of_reflection"
        )
    if tc != ts:
        return "completely_wrong"
    return "wrong_rotation_angle" if determinant(mc) == 1 else "wrong_reflection_line"


def audit_row(saved: Mapping, oracle: Mapping) -> dict:
    pred = extract_last_object(saved.get("raw_model_output", ""))
    correct = pred.get("correct_net") if pred else None
    student = pred.get("student_net") if pred else None
    correct_ok = geometry_matches(correct, oracle["original"], oracle["correct_image"])
    student_ok = geometry_matches(student, oracle["original"], oracle["student_image"])
    both_ok = correct_ok and student_ok
    pred_label = pred.get("label") if pred else None
    label_ok = pred_label == oracle.get("label")
    derived = None
    derived_ok = None
    if correct is not None and student is not None:
        try:
            derived = diagnose_nets(correct, student)
            derived_ok = derived == oracle.get("label")
        except (TypeError, ValueError):
            pass
    audited = {
        "id": saved.get("id"),
        "split": saved.get("split"),
        "parse_ok": pred is not None,
        "correct_net_ok": correct_ok,
        "student_net_ok": student_ok,
        "both_nets_ok": both_ok,
        "label_ok": label_ok,
        "derived_label": derived,
        "derived_label_ok": derived_ok,
        "predicted": pred,
        "true_label": oracle.get("label"),
    }
    audited["stored_metric_disagreements"] = {
        stored_key: {"stored": saved.get(stored_key), "independent": audited[audit_key]}
        for audit_key, stored_key in AUDIT_TO_STORED.items()
        if saved.get(stored_key) != audited[audit_key]
        and not (stored_key == "derived_label_ok" and saved.get(stored_key) is None)
    }
    return audited


def wilson(hits: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = hits / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def summarize(rows: Sequence[Mapping]) -> dict:
    metrics = {}
    for key in (
        "parse_ok",
        "correct_net_ok",
        "student_net_ok",
        "both_nets_ok",
        "label_ok",
        "derived_label_ok",
    ):
        values = [row.get(key) is True for row in rows]
        hits = sum(values)
        low, high = wilson(hits, len(values))
        metrics[key] = {
            "hits": hits,
            "n": len(values),
            "rate": hits / len(values) if values else 0.0,
            "wilson_95": [low, high],
        }
    disagreements = sum(bool(row["stored_metric_disagreements"]) for row in rows)
    return {"n": len(rows), "metrics": metrics, "evaluator_disagreement_rows": disagreements}


def geometry_fingerprint(record: Mapping) -> str:
    payload = [
        record.get("original"),
        record.get("correct_image"),
        record.get("student_image"),
    ]
    return json.dumps(payload, separators=(",", ":"), sort_keys=False)


def leakage_audit(v6_dir: str, oracles: Iterable[Mapping]) -> dict:
    train_path = os.path.join(v6_dir, "train_v6.jsonl")
    if not os.path.isfile(train_path):
        return {"available": False, "reason": f"missing {train_path}"}
    train = load_jsonl(train_path)
    train_geometry = {geometry_fingerprint(row) for row in train}
    evaluated = list(oracles)
    overlaps = [
        row.get("id") for row in evaluated
        if geometry_fingerprint(row) in train_geometry
    ]
    bad_source_splits = [
        row.get("id") for row in train
        if row.get("source_split") in ("test", "ood")
    ]
    return {
        "available": True,
        "train_records": len(train),
        "evaluated_records": len(evaluated),
        "exact_geometry_overlap_count": len(overlaps),
        "overlap_ids": overlaps[:20],
        "train_rows_sourced_from_test_or_ood": len(bad_source_splits),
        "bad_source_ids": bad_source_splits[:20],
    }


def paired_summary(left_name: str, left: Sequence[Mapping],
                   right_name: str, right: Sequence[Mapping]) -> Optional[dict]:
    lmap = {(row["split"], row["id"]): row for row in left}
    rmap = {(row["split"], row["id"]): row for row in right}
    if set(lmap) != set(rmap):
        return None
    metrics = {}
    for key in ("correct_net_ok", "student_net_ok", "both_nets_ok", "label_ok", "derived_label_ok"):
        both = left_only = right_only = neither = 0
        for rid in lmap:
            lv, rv = lmap[rid].get(key) is True, rmap[rid].get(key) is True
            if lv and rv:
                both += 1
            elif lv:
                left_only += 1
            elif rv:
                right_only += 1
            else:
                neither += 1
        metrics[key] = {
            "both_correct": both,
            f"{left_name}_only": left_only,
            f"{right_name}_only": right_only,
            "neither": neither,
            "rate_difference_left_minus_right": (left_only - right_only) / len(lmap),
        }
    return {"left": left_name, "right": right_name, "n": len(lmap), "metrics": metrics}


def load_oracles(data_dir: str, saved_rows: Sequence[Mapping]) -> Dict[Tuple[str, object], dict]:
    splits = sorted({str(row.get("split")) for row in saved_rows})
    result = {}
    for split in splits:
        path = os.path.join(data_dir, f"{split}.jsonl")
        for row in load_jsonl(path):
            result[(split, row["id"])] = row
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records", nargs="+")
    parser.add_argument("--data-dir", default=os.path.expanduser("~/transform_diagnosis_data"))
    parser.add_argument("--v6-dir", default=os.path.expanduser("~/transform_diagnosis_data_v6"))
    parser.add_argument("--cases", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260711)
    args = parser.parse_args(argv)

    all_audits = {}
    all_oracles = {}
    for path in args.records:
        saved = load_jsonl(path)
        oracle_map = load_oracles(args.data_dir, saved)
        audited = []
        for row in saved:
            key = (str(row.get("split")), row.get("id"))
            if key not in oracle_map:
                raise KeyError(f"{path}: no oracle for {key}")
            audited.append(audit_row(row, oracle_map[key]))
            all_oracles[key] = oracle_map[key]
        name = Path(path).stem
        summary = summarize(audited)
        failures = [
            row for row in audited
            if not row["both_nets_ok"] or not row["label_ok"]
        ]
        random.Random(args.seed).shuffle(failures)
        summary["sample_failures"] = failures[:args.cases]
        output = str(Path(path).with_name(f"audit_{name}.json"))
        with open(output, "w") as handle:
            json.dump(summary, handle, indent=2)
            handle.write("\n")
        all_audits[name] = audited
        print(f"\n{name}  n={summary['n']}  evaluator_disagreements="
              f"{summary['evaluator_disagreement_rows']}")
        for key, metric in summary["metrics"].items():
            low, high = metric["wilson_95"]
            print(f"  {key:22s} {metric['rate']:.3f}  "
                  f"95% CI [{low:.3f}, {high:.3f}]")
        print(f"  wrote {output}")

    leak = leakage_audit(args.v6_dir, all_oracles.values())
    print("\nleakage audit:", json.dumps(leak, indent=2))

    names = list(all_audits)
    pairs = []
    for index, left_name in enumerate(names):
        for right_name in names[index + 1:]:
            paired = paired_summary(
                left_name, all_audits[left_name],
                right_name, all_audits[right_name],
            )
            if paired is not None:
                pairs.append(paired)
                print(f"\npaired: {left_name} vs {right_name} (n={paired['n']})")
                for key, counts in paired["metrics"].items():
                    print(f"  {key:22s} {counts}")
    with open("audit_v6_paired_summary.json", "w") as handle:
        json.dump({"leakage": leak, "pairs": pairs}, handle, indent=2)
        handle.write("\n")
    print("\nwrote audit_v6_paired_summary.json")


if __name__ == "__main__":
    main()
