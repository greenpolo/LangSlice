"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import os

import pytest

from tests.synthetic_stack import apply_patches

apply_patches()
os.environ.setdefault("LANGSLICE_ATLAS_CACHE", "off")
os.environ.setdefault("LANGSLICE_RECENT_JOBS", "off")
os.environ.setdefault("LANGSLICE_SETTINGS", "off")
os.environ.setdefault("LANGSLICE_CLAUDE_TRANSCRIPTS", "off")
os.environ["LANGSLICE_DETACH"] = "0"

if os.name == "nt":
    _setenv = pytest.MonkeyPatch.setenv

    def _setenv_home_too(self, name, value, prepend=None):
        _setenv(self, name, value, prepend)
        if name == "HOME":
            _setenv(self, "USERPROFILE", value, prepend)

    pytest.MonkeyPatch.setenv = _setenv_home_too


@pytest.fixture(scope="module")
def atlas() -> object:
    from langslice.core.atlas.load import load_atlas

    return load_atlas("allen_mouse_25um")
