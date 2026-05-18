"""Compatibility helpers for the Gemma training transition to shared packages."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    """Return repository root from this module location."""
    return Path(__file__).resolve().parents[3]


def add_transition_roots() -> None:
    """Add shared package roots, then legacy roots, to ``sys.path``.

    Order matters:
    1. Shared packages (if present)
    2. Legacy in-repo training/data roots (fallback)
    """
    root = repo_root()
    candidates = [
        root / "models" / "langslice-traces",
        root / "models" / "synthdata",
        root / "models" / "training-core",
        root / "models" / "data",
        root / "models" / "langslice-gemma-4" / "training",
        root / "models" / "langslice-gemma-4" / "data",
        root / "src",
    ]
    for path in candidates:
        if path.exists():
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

