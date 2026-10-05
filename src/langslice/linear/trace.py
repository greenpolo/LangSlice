"""Compatibility shim for sibling repos; use :mod:`langslice.agent.trace`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.agent.trace")
