"""Compatibility shim for sibling repos; use :mod:`langslice.core.nonlinear.image_gen_helpers`."""

import importlib
import sys

sys.modules[__name__] = importlib.import_module("langslice.core.nonlinear.image_gen_helpers")
