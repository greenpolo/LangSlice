"""Compatibility module for legacy ``import synth_dataset`` callers."""

from __future__ import annotations

import importlib
import sys

_module = importlib.import_module("synthdata.dataset")
sys.modules[__name__] = _module
