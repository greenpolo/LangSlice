"""DeepSlice seeding — the seam only; the integration is not built yet.

DeepSlice ships as an optional extra (``pip install langslice[deepslice]``)
and is expected to place a whole coronal mouse stack in one shot. Nothing here
calls it: the package is not a dependency today, so :func:`deepslice_available`
reports False and the ``run_deepslice`` tool answers ``UNAVAILABLE``. When the
extra lands, :func:`run_deepslice` is the only function that has to grow a body.
"""

from __future__ import annotations

import importlib.util
import logging
from typing import Any

logger = logging.getLogger(__name__)

#: Import name of the optional package.
DEEPSLICE_PACKAGE = "DeepSlice"


def deepslice_available() -> bool:
    """True when the optional DeepSlice package can be imported."""
    try:
        return importlib.util.find_spec(DEEPSLICE_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def run_deepslice(
    state: Any, ctx: Any, *, slice_ids: list[str], allow_angle_change: bool
) -> dict[str, Any]:
    """Positions (and angles) for the named sections. NOT IMPLEMENTED.

    Returns the tool payload directly: on success it would carry
    ``{"status": "ok", "positions": {slice_id: mm}, "angles": {...}}``.
    """
    del state, ctx, slice_ids, allow_angle_change
    return {
        "status": "error",
        "error": "UNAVAILABLE",
        "installed": deepslice_available(),
        "message": "DeepSlice is not wired in this build.",
    }
