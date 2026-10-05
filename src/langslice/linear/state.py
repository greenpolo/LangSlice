"""Compatibility shim for sibling repos; use :mod:`langslice.core.state`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.core.state")
