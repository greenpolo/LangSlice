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
# Nor the recent jobs list (~/.langslice/recent_jobs.jsonl); its tests point
# it at a temporary file.
os.environ.setdefault("LANGSLICE_RECENT_JOBS", "off")
# Nor read or write the user's settings (~/.langslice/settings.json), which
# `linear run` reads for its model; their tests point it at a temporary file.
os.environ.setdefault("LANGSLICE_SETTINGS", "off")
# Nor Claude Code's transcripts (~/.claude/projects), which the window reads
# for a coding agent's words; their tests point it at a temporary folder.
os.environ.setdefault("LANGSLICE_CLAUDE_TRANSCRIPTS", "off")
# A long verb's CLI call runs in its own process unless a test detaches it
# (whatever harness runs the tests: Claude Code sets CLAUDECODE).
os.environ["LANGSLICE_DETACH"] = "0"

if os.name == "nt":
    # Tests give a test its own home folder by setting HOME; Python on Windows
    # finds the home folder through USERPROFILE instead, so setting HOME sets
    # both.
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
