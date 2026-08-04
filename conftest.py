"""Pytest path setup.

CI runs the suites from the repo root as ``pytest transform_diagnosis/`` and
``pytest model/``. The tests import both packaged modules (``from model import
...``, ``from transform_diagnosis import ...``) and a few bare module names
(``import coarsegrain_ablation``) that live under ``model/``. Put the repo root
and ``model/`` on ``sys.path`` so collection resolves in any invocation.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
for _path in (_ROOT, _ROOT / "model"):
    _entry = str(_path)
    if _entry not in sys.path:
        sys.path.insert(0, _entry)
