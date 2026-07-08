"""render — matplotlib rendering of diagnosis records.

Draws the original polygon (red), the student's image (blue), and optionally the correct
image (green) on a ``[-10, 10]`` grid with bold axes and integer ticks. The matplotlib
import is guarded so the rest of the package works even if matplotlib is unavailable
(rendering simply raises a clear error in that case).

Modeled on the grid rendering in the owner's earlier ``model/generate_diagnosis_data.py``.
"""

from __future__ import annotations

import os
from typing import Optional, Sequence

try:  # guarded import — the data pipeline does not require matplotlib
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - environment dependent
    MATPLOTLIB_AVAILABLE = False
    _IMPORT_ERROR = exc

LIM = 10
_RED = "#d1344e"     # original / pre-image
_BLUE = "#2b6cb0"    # student's answer
_GREEN = "#1f9d55"   # correct image


def _require_matplotlib() -> None:
    if not MATPLOTLIB_AVAILABLE:
        raise RuntimeError(
            "matplotlib is required for rendering but could not be imported: "
            f"{_IMPORT_ERROR!r}. Install it (e.g. `python3 -m pip install matplotlib`) "
            "or run with rendering disabled."
        )


def _draw_polygon(ax, pts: Sequence[Sequence[int]], color: str, *, fill: bool = True,
                  linestyle: str = "-", linewidth: float = 2.0) -> None:
    xs = [p[0] for p in pts] + [pts[0][0]]
    ys = [p[1] for p in pts] + [pts[0][1]]
    if fill:
        ax.fill(xs, ys, color=color, alpha=0.18)
    ax.plot(xs, ys, color=color, linewidth=linewidth, linestyle=linestyle,
            solid_joinstyle="round")


def render_to_path(
    original: Sequence[Sequence[int]],
    student_image: Sequence[Sequence[int]],
    path: str,
    correct_image: Optional[Sequence[Sequence[int]]] = None,
) -> str:
    """Render a single figure to ``path`` (created atomically). Returns ``path``."""
    _require_matplotlib()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    ax.set_xlim(-LIM, LIM)
    ax.set_ylim(-LIM, LIM)
    ax.set_aspect("equal")
    ax.set_xticks(range(-LIM, LIM + 1))
    ax.set_yticks(range(-LIM, LIM + 1))
    ax.grid(True, linewidth=0.4, alpha=0.4)
    ax.tick_params(labelsize=5)
    ax.axhline(0, color="black", linewidth=1.1)
    ax.axvline(0, color="black", linewidth=1.1)

    if correct_image is not None:
        _draw_polygon(ax, correct_image, _GREEN, fill=False, linestyle="--", linewidth=1.6)
    _draw_polygon(ax, original, _RED)
    _draw_polygon(ax, student_image, _BLUE)

    tmp = path + ".tmp.png"
    fig.savefig(tmp, bbox_inches="tight")
    plt.close(fig)
    os.replace(tmp, path)
    return path


def render_record(rec: dict, out_dir: str, *, skip_existing: bool = True,
                  show_correct: bool = True) -> str:
    """Render one record to ``<out_dir>/<render_path>``. Skips if the file already exists
    (idempotent / resumable)."""
    path = os.path.join(out_dir, rec["render_path"])
    if skip_existing and os.path.exists(path):
        return path
    return render_to_path(
        rec["original"], rec["student_image"], path,
        correct_image=rec["correct_image"] if show_correct else None,
    )


def render_all(records: Sequence[dict], out_dir: str, *, skip_existing: bool = True,
               show_correct: bool = True, progress_every: int = 100) -> int:
    """Render every record. Returns the count of freshly rendered files."""
    _require_matplotlib()
    made = 0
    for i, rec in enumerate(records):
        path = os.path.join(out_dir, rec["render_path"])
        if skip_existing and os.path.exists(path):
            continue
        render_to_path(
            rec["original"], rec["student_image"], path,
            correct_image=rec["correct_image"] if show_correct else None,
        )
        made += 1
        if progress_every and made % progress_every == 0:
            print(f"  rendered {made} new images ...", flush=True)
    return made
