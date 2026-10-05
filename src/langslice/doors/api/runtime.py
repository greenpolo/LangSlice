"""Runtime wrappers for the engine API."""

from __future__ import annotations

from langslice.doors.api.models import VersionResult


def get_version() -> VersionResult:
    import langslice

    return VersionResult(version=langslice.__version__)
