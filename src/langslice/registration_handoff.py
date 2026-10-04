"""Deprecated shim: moved to :mod:`langslice.core.nonlinear.registration_handoff`.

Kept for SliceBench (layered folder move, 2026-10-04).
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.core.nonlinear.registration_handoff")
