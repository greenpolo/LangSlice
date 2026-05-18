"""Legacy shim for ``augmentation`` during shared-package transition."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[4]
for _candidate in (
    _ROOT / "models" / "synthdata",
    _ROOT / "models" / "data",
):
    if _candidate.exists():
        _value = str(_candidate)
        if _value not in sys.path:
            sys.path.insert(0, _value)

# Expose submodules from shared ``synthdata`` first, then this legacy shim.
_shared_augmentation = _ROOT / "models" / "synthdata" / "synthdata" / "augmentation"
if _shared_augmentation.exists():
    __path__ = [str(_shared_augmentation), *__path__]  # type: ignore[name-defined]
