"""Compatibility shim for sibling repos; use :mod:`langslice.doors.tools.toolbox`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.doors.tools.toolbox")
