"""Deprecated shim: moved to :mod:`langslice.agent.trace`.

Kept for SliceBench (layered folder move, 2026-10-04).
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.agent.trace")
