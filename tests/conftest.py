"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import os

import pytest

from tests.synthetic_stack import apply_patches

# Every test process fits in process at the coarse level, whatever ran before.
apply_patches()
# Tests never write the atlas disk cache in the user's home; the cache's own
# tests point it at a temporary folder.
os.environ.setdefault("LANGSLICE_ATLAS_CACHE", "off")
# A long verb's CLI call runs in its own process unless a test detaches it
# (whatever harness runs the tests: Claude Code sets CLAUDECODE).
os.environ["LANGSLICE_DETACH"] = "0"


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice.core.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")
