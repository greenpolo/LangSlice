"""Deprecated shim: moved to :mod:`langslice.core.space`.

Kept for SliceBench, LangSlice-Training (layered folder move, 2026-10-04).
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.core.space")
