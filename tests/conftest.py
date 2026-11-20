"""Keep the test jobs off the user's own LangSlice settings and caches, and
make the synthetic job they open (``images``).

The settings are switched off before any test imports LangSlice, so the
fixture imports LangSlice inside its body.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("LANGSLICE_ATLAS_CACHE", "off")
os.environ.setdefault("LANGSLICE_RECENT_JOBS", "off")
os.environ.setdefault("LANGSLICE_SETTINGS", "off")
os.environ.setdefault("LANGSLICE_CLAUDE_TRANSCRIPTS", "off")
os.environ["LANGSLICE_DETACH"] = "0"


@pytest.fixture
def images(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Three synthetic sections in a folder, opened on the synthetic atlas."""
    import langslice.doors.jobs as jobs
    from tests.deformable_synthetic import SyntheticAtlas
    from tests.synthetic_job import write_sections

    atlas = SyntheticAtlas()
    make_context = jobs.context

    def context_on_synthetic_atlas(*args: object, **kwargs: object) -> object:
        kwargs["atlas_loader"] = lambda _name: atlas
        return make_context(*args, **kwargs)

    monkeypatch.setattr(jobs, "context", context_on_synthetic_atlas)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    folder = tmp_path / "stack"
    write_sections(folder)
    return folder
