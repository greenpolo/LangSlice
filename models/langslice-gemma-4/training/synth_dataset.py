"""Legacy synth-dataset import shim.

Preferred target is ``synthdata.dataset`` from shared packages.
Fallback target is the legacy ``models/langslice-gemma-4/data/synth_dataset.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
for _candidate in (
    _ROOT / "models" / "synthdata",
    _ROOT / "models" / "data",
    _ROOT / "models" / "langslice-gemma-4" / "data",
):
    if _candidate.exists():
        _value = str(_candidate)
        if _value not in sys.path:
            sys.path.insert(0, _value)

_module = None
for _name in ("synthdata.dataset", "synthdata.synth_dataset"):
    try:
        _module = importlib.import_module(_name)
        break
    except ImportError:
        continue

if _module is None:
    _legacy = _ROOT / "models" / "langslice-gemma-4" / "data" / "synth_dataset.py"
    if not _legacy.exists():
        raise ImportError(
            "Unable to import synth_dataset from shared package or legacy location."
        )
    _spec = importlib.util.spec_from_file_location("_legacy_synth_dataset", _legacy)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"Unable to load legacy synth_dataset from {_legacy}")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_spec.name] = _module
    _spec.loader.exec_module(_module)

sys.modules[__name__] = _module
globals().update(_module.__dict__)
