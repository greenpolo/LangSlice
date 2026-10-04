"""Deprecated shim: moved to :mod:`langslice.core.nonlinear.border_refinement`.

Kept for SliceBench's local validation scripts (layered folder move, 2026-10-04).
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.core.nonlinear.border_refinement")
