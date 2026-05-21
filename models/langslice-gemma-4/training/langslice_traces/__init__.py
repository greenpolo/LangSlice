"""Compatibility shim for the canonical ``models/langslice-traces`` package."""

from __future__ import annotations

from pathlib import Path

_HERE = Path(__file__).resolve()
_CANONICAL_PACKAGE = _HERE.parents[3] / "langslice-traces" / "langslice_traces"
_CANONICAL_INIT = _CANONICAL_PACKAGE / "__init__.py"

if not _CANONICAL_INIT.is_file():
    raise ImportError(f"Could not locate canonical langslice_traces at {_CANONICAL_INIT}")

__file__ = str(_CANONICAL_INIT)
__path__ = [str(_CANONICAL_PACKAGE)]

exec(compile(_CANONICAL_INIT.read_text(encoding="utf-8"), str(_CANONICAL_INIT), "exec"))
