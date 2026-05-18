"""Compatibility package for legacy ``augmentation`` imports."""

from __future__ import annotations

from pathlib import Path

_SHARED_AUGMENTATION = Path(__file__).resolve().parents[1] / "synthdata" / "augmentation"
__path__ = [str(_SHARED_AUGMENTATION)]  # type: ignore[name-defined]
