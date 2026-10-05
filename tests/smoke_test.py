"""Smoke checks that the public modules import."""

from __future__ import annotations

import importlib


def test_public_module_imports() -> None:
    importlib.import_module("langslice")
    importlib.import_module("langslice.core.atlas")
    importlib.import_module("langslice.providers.vlm_config")
    importlib.import_module("langslice.linear")
    importlib.import_module("langslice.core.nonlinear")
    importlib.import_module("langslice.job.quint")
    importlib.import_module("langslice.core.image_prep")
    importlib.import_module("langslice.cli")
