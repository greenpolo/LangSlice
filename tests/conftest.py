"""Keep the smoke run off the user's own LangSlice settings and caches."""

from __future__ import annotations

import os

os.environ.setdefault("LANGSLICE_ATLAS_CACHE", "off")
os.environ.setdefault("LANGSLICE_RECENT_JOBS", "off")
os.environ.setdefault("LANGSLICE_SETTINGS", "off")
os.environ.setdefault("LANGSLICE_CLAUDE_TRANSCRIPTS", "off")
os.environ["LANGSLICE_DETACH"] = "0"
