"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import pytest

from tests.synthetic_stack import apply_patches

# Every test process fits in process at the coarse level, whatever ran before.
apply_patches()


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice.core.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")
