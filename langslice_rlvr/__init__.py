"""Repo-root launcher for ``python -m langslice_rlvr``."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_TRAINING_ROOT = _REPO_ROOT / "models" / "langslice-gemma-4" / "training"


def _bootstrap_paths() -> None:
    """Add source roots needed by the off-src RLVR package."""
    for path in (str(_SRC_ROOT), str(_TRAINING_ROOT)):
        if path not in sys.path:
            sys.path.insert(0, path)


def main(argv: list[str] | None = None) -> None:
    """Import and run the real training driver after path bootstrapping."""
    _bootstrap_paths()
    from rlvr.train_grpo import main as _main  # noqa: PLC0415

    _main(argv)

__all__ = ["main"]
