"""Compatibility shim for sibling repos; use
:mod:`langslice.core.nonlinear.registration_handoff`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.core.nonlinear.registration_handoff")
