"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice.core.atlas.core import load_atlas

    return load_atlas("allen_mouse_25um")
