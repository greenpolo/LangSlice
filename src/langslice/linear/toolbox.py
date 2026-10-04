"""Deprecated shim: moved to :mod:`langslice.doors.tools.toolbox`.

Kept for SliceBench (layered folder move, 2026-10-04).
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.doors.tools.toolbox")
