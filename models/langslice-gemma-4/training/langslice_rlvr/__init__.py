"""Launcher shim for the parked RLVR trainer.

The actual RLVR training code lives next door at
``models/langslice-gemma-4/training/rlvr/``. This shim adds the training
directory to ``sys.path`` and re-exports ``main`` so callers that have
PYTHONPATH set to ``models/langslice-gemma-4/training`` can ``python -m
langslice_rlvr`` (or just ``python -m rlvr.train_grpo``).

RLVR is parked as of 2026-05-09 in favor of expert-iteration SFT; see
``models/langslice-gemma-4/training/rlvr/README.md`` before un-parking.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TRAINING_ROOT = Path(__file__).resolve().parent.parent


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
