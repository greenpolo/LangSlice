"""Deprecated shim: moved to :mod:`langslice.hosts.integrations.abba_linear`.

Kept for SliceBench (layered folder move, 2026-10-04).
"""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.hosts.integrations.abba_linear")
