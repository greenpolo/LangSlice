"""Launcher shim for ``python -m langslice_rlvr`` / ``langslice-rlvr``.

The actual RLVR training code lives at
``models/langslice-gemma-4/training/rlvr/``, off the ``src/`` tree. This shim
adds that directory to ``sys.path`` and re-exports ``main`` so the CLI works
from the repo root without restructuring the model project.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_ROOT = _REPO_ROOT / "models" / "langslice-gemma-4" / "training"


def _bootstrap_paths() -> None:
    # Insert the training dir at the FRONT of sys.path so `import rlvr` finds
    # the RLVR package even if a same-named module lives elsewhere.
    if str(_TRAINING_ROOT) not in sys.path:
        sys.path.insert(0, str(_TRAINING_ROOT))


def main(argv: list[str] | None = None) -> None:
    _bootstrap_paths()
    from rlvr.train_grpo import main as _main  # noqa: PLC0415

    _main(argv)

__all__ = ["main"]
