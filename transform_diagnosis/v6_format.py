"""Shared v6 transform-first prompts and targets.

All v6 data, tuned-model evaluation, and frontier evaluation use these builders,
so task wording and JSON schemas cannot drift.  Heavy image/model dependencies
are intentionally absent; image content parts contain a path string until a
trainer/evaluator materializes them.
"""

from __future__ import annotations

import json
from typing import Dict, Mapping, Sequence, Tuple

from . import net_transform as nt
from . import transform_core as tc

SCHEMA_VERSION = "v6.net-affine.1"
TASK_MODES: Tuple[str, ...] = ("correct", "student", "both", "full")
INPUT_MODES: Tuple[str, ...] = ("image", "image_coords", "coords")

_MAP_SCHEMA = (
    '{"linear":"<D4 enum>","tx":<integer>,"ty":<integer>}'
)


def _require_task(task: str) -> str:
    if task not in TASK_MODES:
        raise ValueError(f"task must be one of {TASK_MODES}, got {task!r}")
    return task


def _require_input(input_mode: str) -> str:
    if input_mode not in INPUT_MODES:
        raise ValueError(f"input mode must be one of {INPUT_MODES}, got {input_mode!r}")
    return input_mode


def augment_record(record: Mapping) -> dict:
    """Copy a legacy oracle record and attach verified canonical net maps."""
    rec = dict(record)
    correct = nt.sequence_to_net(rec["correct_transform"])
    student = nt.sequence_to_net(rec["student_transform"])
    if tc.recover_map(rec["original"], rec["correct_image"]) != nt.net_to_affine(correct):
        raise ValueError(f"record id={rec.get('id')} correct image disagrees with correct transform")
    if tc.recover_map(rec["original"], rec["student_image"]) != nt.net_to_affine(student):
        raise ValueError(f"record id={rec.get('id')} student image disagrees with student transform")
    derived = nt.diagnose_nets(correct, student)
    if derived != rec["label"]:
        raise ValueError(
            f"record id={rec.get('id')} label {rec['label']!r} disagrees with net oracle {derived!r}"
        )
    rec["correct_net"] = correct
    rec["student_net"] = student
    rec["schema_version"] = SCHEMA_VERSION
    return rec


def target_obj(record: Mapping, task: str = "full") -> Dict[str, object]:
    """Return the exact assistant target for one transform-first task."""
    task = _require_task(task)
    rec = record if "correct_net" in record and "student_net" in record else augment_record(record)
    if task == "correct":
        obj = {"correct_net": nt.validate_net(rec["correct_net"])}
    elif task == "student":
        obj = {"student_net": nt.validate_net(rec["student_net"])}
    elif task == "both":
        obj = {
            "correct_net": nt.validate_net(rec["correct_net"]),
            "student_net": nt.validate_net(rec["student_net"]),
        }
    else:
        obj = {
            "correct_net": nt.validate_net(rec["correct_net"]),
            "student_net": nt.validate_net(rec["student_net"]),
            "label": rec["label"],
            "hint": rec["hint"],
        }
    validate_target(obj, task)
    return obj


def target_json(record: Mapping, task: str = "full") -> str:
    return json.dumps(target_obj(record, task), ensure_ascii=False, separators=(",", ":"))


def validate_target(target: object, task: str) -> dict:
    """Strictly validate a task target and return a shallow canonical copy."""
    task = _require_task(task)
    if not isinstance(target, Mapping):
        raise ValueError("v6 target must be a JSON object")
    expected = {
        "correct": ("correct_net",),
        "student": ("student_net",),
        "both": ("correct_net", "student_net"),
        "full": ("correct_net", "student_net", "label", "hint"),
    }[task]
    if tuple(target.keys()) != expected:
        raise ValueError(f"{task} target keys must be {expected}, got {tuple(target.keys())}")
    out = dict(target)
    if "correct_net" in out:
        out["correct_net"] = nt.validate_net(out["correct_net"])
    if "student_net" in out:
        out["student_net"] = nt.validate_net(out["student_net"])
    if "label" in out and out["label"] not in tc.DIAGNOSIS_LABELS:
        raise ValueError(f"unknown diagnosis label: {out['label']!r}")
    if "hint" in out and (not isinstance(out["hint"], str) or not out["hint"].strip()):
        raise ValueError("full target hint must be a non-empty string")
    return out


def coordinates_block(record: Mapping) -> str:
    """Exact corresponding vertices used by the image+coordinates arm."""
    def pts(key: str) -> Sequence[list]:
        return [list(p) for p in record[key]]

    return (
        "Exact vertices in corresponding order:\n"
        f"  RED original: {pts('original')}\n"
        f"  GREEN correct image: {pts('correct_image')}\n"
        f"  BLUE student image: {pts('student_image')}"
    )


def instruction(record: Mapping, task: str, input_mode: str = "image") -> str:
    """Task-specific v6 prompt; never reuses the legacy step-sequence prompt."""
    task = _require_task(task)
    input_mode = _require_input(input_mode)
    scene = (
        "Recover observable NET affine maps on the integer coordinate grid. A net map sends "
        "(x,y) to M(x,y)+(tx,ty). RED is the original polygon, GREEN is the correct image, "
        "and BLUE is the student's image. Vertex numbers/order give correspondence. Do not "
        "guess an ordered sequence of steps: different step sequences can represent the same "
        "net map.\n\n"
    )
    if input_mode in ("image_coords", "coords"):
        scene += coordinates_block(record) + "\n\n"

    vocab = ", ".join(nt.D4_LINEAR_NAMES)
    schema = (
        f"Use exactly this map schema: {_MAP_SCHEMA}. The D4 linear enum is exactly "
        f"[{vocab}]. tx and ty are integers. Return one valid JSON object and nothing else."
    )
    asks = {
        "correct": (
            'Recover RED→GREEN. Return exactly {"correct_net":<map>}.'
        ),
        "student": (
            'Recover RED→BLUE. Return exactly {"student_net":<map>}.'
        ),
        "both": (
            'Recover both observable maps. Return exactly '
            '{"correct_net":<RED-to-GREEN map>,"student_net":<RED-to-BLUE map>}.'
        ),
        "full": (
            "Recover both maps first, then diagnose the mismatch. Return exactly "
            '{"correct_net":<RED-to-GREEN map>,"student_net":<RED-to-BLUE map>,'
            f'"label":"<one of {list(tc.DIAGNOSIS_LABELS)}>",'
            '"hint":"<short Socratic hint naming the kind of mistake WITHOUT stating any '
            'coordinates, axes, angles, or translation values>"}. '
            "The label must follow deterministically from the two maps; do not state an "
            "ordered step decomposition in place of either map."
        ),
    }
    return scene + asks[task] + "\n\n" + schema


def user_message(
    record: Mapping,
    task: str,
    input_mode: str = "image",
    *,
    image_path: str | None = None,
) -> dict:
    """Build one v6 user turn for image, image+coordinates, or coordinates-only."""
    input_mode = _require_input(input_mode)
    text = {"type": "text", "text": instruction(record, task, input_mode)}
    if input_mode == "coords":
        content = [text]
    else:
        path = image_path if image_path is not None else record["render_path"]
        content = [{"type": "image", "image": path}, text]
    return {"role": "user", "content": content}


def conversation(
    record: Mapping,
    task: str,
    input_mode: str = "image",
    *,
    image_path: str | None = None,
) -> dict:
    rec = record if "correct_net" in record and "student_net" in record else augment_record(record)
    return {
        "id": rec["id"],
        "split": rec.get("split"),
        "schema_version": SCHEMA_VERSION,
        "task": _require_task(task),
        "input_mode": _require_input(input_mode),
        "messages": [
            user_message(rec, task, input_mode, image_path=image_path),
            {
                "role": "assistant",
                "content": [{"type": "text", "text": target_json(rec, task)}],
            },
        ],
    }


__all__ = [
    "INPUT_MODES",
    "SCHEMA_VERSION",
    "TASK_MODES",
    "augment_record",
    "conversation",
    "coordinates_block",
    "instruction",
    "target_json",
    "target_obj",
    "user_message",
    "validate_target",
]
