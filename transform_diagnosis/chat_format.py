"""chat_format — turn verified diagnosis records into Qwen3-VL chat conversations.

The training target for the vision fine-tune (``unsloth/Qwen3-VL-4B-Instruct``) is a
two-turn conversation per record:

    user      : [ image , instruction text ]
    assistant : {"label": ..., "correct_transform": [...], "hint": ...}

The rendered PNG already contains three polygons (see :mod:`render`): the RED pre-image,
the GREEN dashed *correct* image, and the BLUE student answer. The model therefore has a
visual reference for "correct", so ``correct_transform`` is a genuine output (name the
RED->GREEN map) rather than something we must feed in as text.

This module is dependency-light on purpose (no PIL / matplotlib): the JSONL it emits
references each image by its relative ``render_path`` string, using the OpenAI/LLaVA-style
``{"type": "image", "image": "<path>"}`` content part. The training notebook swaps that
path for a decoded ``PIL.Image`` at load time (that is the one place Pillow is needed).

Single source of truth: the label vocabulary in ``INSTRUCTION`` is built from
``transform_core.DIAGNOSIS_LABELS`` at import, so it can never drift from the generator.
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence

from . import transform_core as tc

# The three fields the assistant must emit, in this order. "correct_transform" sits
# between label and hint because it is the evidence bridging the two.
TARGET_KEYS = ("label", "correct_transform", "hint")


def _build_instruction(
    *, seq_phrase: str, image_step_phrase: str, transform_phrase: str, list_phrase: str
) -> str:
    labels = ", ".join(tc.DIAGNOSIS_LABELS)
    return (
        "You are a geometry tutor's diagnostic assistant. You are shown a coordinate "
        "grid from -10 to 10 containing three polygons:\n"
        "  - RED (solid fill): the pre-image P, the starting shape.\n"
        "  - GREEN (dashed outline): the correct image -- where P should land after the "
        f"intended {image_step_phrase}.\n"
        "  - BLUE (solid fill): the student's submitted answer.\n\n"
        f"A student was asked to apply {seq_phrase} rigid transformations "
        "(rotations, reflections, and/or translations) to P. Compare the student's BLUE "
        "answer to the correct GREEN image and diagnose the student's mistake.\n\n"
        "Return a SINGLE valid JSON object and NOTHING else -- no prose, no markdown, no "
        "code fences -- with exactly these fields:\n"
        f'  "label": the diagnosis, exactly one of [{labels}].\n'
        f'  "correct_transform": the intended {transform_phrase} as a list of {list_phrase} '
        "strings (the moves that map the RED pre-image onto the GREEN correct image), "
        'e.g. ["rotate 90 degrees counterclockwise", "translate 7 left"].\n'
        '  "hint": a short Socratic nudge that points the student toward their error '
        "WITHOUT stating the correct coordinates.\n"
        'Use "correct" only when the BLUE answer coincides with the GREEN image.'
    )


# The canonical two-step instruction (unchanged wording -- v1/v2/v3cot conversations stay
# byte-for-byte identical). The curriculum variant only relaxes the step-count wording so
# single-step warm-up records are described accurately; everything else is identical.
INSTRUCTION = _build_instruction(
    seq_phrase="a sequence of exactly two",
    image_step_phrase="two-step transformation",
    transform_phrase="two-step transformation",
    list_phrase="two",
)
CURRICULUM_INSTRUCTION = _build_instruction(
    seq_phrase="a sequence of one or two",
    image_step_phrase="transformation",
    transform_phrase="transformation",
    list_phrase="one or two",
)


def instruction_for(record: dict) -> str:
    """Instruction text matching the record's step count.

    Two-step records (the default, and every v1/v2/v3cot record) get the exact canonical
    ``INSTRUCTION`` so their conversations are byte-identical; single-step curriculum
    records get ``CURRICULUM_INSTRUCTION`` (says "one or two" steps) so the prompt does not
    claim "exactly two" while showing a one-step answer.
    """
    n = len(record.get("correct_transform") or [])
    return CURRICULUM_INSTRUCTION if n == 1 else INSTRUCTION


def coords_prompt(record: dict) -> str:
    """Coordinates-as-text analogue of ``INSTRUCTION`` (no image).

    Gives the three polygons as integer ``(x, y)`` vertex lists in corresponding order
    instead of a rendered grid, but asks for the exact same output JSON schema
    (``label`` / ``correct_transform`` / ``hint``). This is the SINGLE source of the
    coordinate-input prompt, shared by both coordinates-input eval cells (the frontier
    probe and the tuned-4B coordinate eval) so their wording can never drift and the two
    stay directly comparable.
    """
    labels = ", ".join(tc.DIAGNOSIS_LABELS)
    return (
        "You are a geometry tutor's diagnostic assistant. A pre-image polygon P was to be "
        "transformed by a sequence of EXACTLY TWO rigid transformations (rotations of "
        "90/180/270 degrees, reflections across the x axis / y axis / line y = x / line "
        "y = -x, and/or translations).\n\n"
        "You are given three polygons as lists of integer (x, y) vertices, in corresponding "
        "vertex order:\n"
        f"  Pre-image P (RED):            {[list(p) for p in record['original']]}\n"
        f"  Correct image (GREEN):        {[list(p) for p in record['correct_image']]}\n"
        f"  Student's answer (BLUE):      {[list(p) for p in record['student_image']]}\n\n"
        "Compare the student's BLUE answer to the correct GREEN image and diagnose the "
        "student's mistake.\n\n"
        "Return a SINGLE valid JSON object and NOTHING else, with exactly these fields:\n"
        f'  "label": exactly one of [{labels}].\n'
        '  "correct_transform": the intended two-step transformation mapping RED onto GREEN, '
        'as a list of two strings, e.g. ["rotate 90 degrees counterclockwise", '
        '"translate 7 left"] or ["reflect across line y = x", "translate by (-2, 3)"].\n'
        '  "hint": a short Socratic nudge pointing at the error without stating coordinates.\n'
        'Use "correct" only when BLUE coincides with GREEN.'
    )


def target_obj(record: dict) -> Dict[str, object]:
    """The assistant's answer as a plain dict (label + correct_transform + hint)."""
    return {k: record[k] for k in TARGET_KEYS}


def target_json(record: dict) -> str:
    """The assistant's answer serialized compactly (fewer target tokens for training)."""
    return json.dumps(target_obj(record), ensure_ascii=False, separators=(",", ":"))


def to_messages(record: dict, *, image_path: str | None = None) -> List[dict]:
    """Two-turn Qwen3-VL message list for one record.

    ``image_path`` overrides the reference stored in the image content part (default:
    the record's ``render_path``). The assistant turn is the serialized target JSON.
    """
    path = image_path if image_path is not None else record["render_path"]
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": path},
                {"type": "text", "text": instruction_for(record)},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": target_json(record)}],
        },
    ]


def to_conversation(record: dict, *, image_path: str | None = None) -> dict:
    """One training row: ``{"id", "split", "messages"}``. ``id``/``split`` are kept for
    traceability and split filtering; trainers read only ``messages``."""
    return {
        "id": record["id"],
        "split": record["split"],
        "messages": to_messages(record, image_path=image_path),
    }


# Split -> chat filename. Mirrors the raw per-split JSONL names with a ``_chat`` suffix.
_CHAT_FILES = {
    "train": "train_chat.jsonl",
    "val": "val_chat.jsonl",
    "test": "test_chat.jsonl",
    "ood": "ood_chat.jsonl",
}


def _atomic_write_lines(path: str, lines: Sequence[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        f.write("".join(lines))
    os.replace(tmp, path)


def write_chat_splits(records: Sequence[dict], out_dir: str) -> Dict[str, str]:
    """Write ``<split>_chat.jsonl`` (one conversation per line) for each of the four
    splits. Returns the paths written, keyed ``"<split>_chat"``. Atomic writes."""
    os.makedirs(out_dir, exist_ok=True)
    written: Dict[str, str] = {}
    for split, fname in _CHAT_FILES.items():
        rows = [
            json.dumps(to_conversation(r), ensure_ascii=False) + "\n"
            for r in records
            if r["split"] == split
        ]
        path = os.path.join(out_dir, fname)
        _atomic_write_lines(path, rows)
        written[f"{split}_chat"] = path
    return written
