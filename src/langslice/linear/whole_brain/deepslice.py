"""DeepSlice seeding — the seam only; the integration is not built yet.

DeepSlice ships as an optional extra (``pip install langslice[deepslice]``)
and is expected to seed positions for a whole coronal mouse stack in one shot.
Nothing here calls it: the package is not a dependency today, so
:func:`deepslice_available` reports False and the seed node falls back to
anchor seeding. When the extra lands, :func:`run_deepslice` is the only
function that has to grow a body.
"""

from __future__ import annotations

import importlib.util
import logging

from langslice.linear.whole_brain.engine import EngineContext
from langslice.linear.whole_brain.state import StackState

logger = logging.getLogger(__name__)

#: Import name of the optional package.
DEEPSLICE_PACKAGE = "DeepSlice"


def deepslice_available() -> bool:
    """True when the optional DeepSlice package can be imported."""
    try:
        return importlib.util.find_spec(DEEPSLICE_PACKAGE) is not None
    except (ImportError, ValueError):
        return False


def run_deepslice(state: StackState, ctx: EngineContext) -> dict[str, float]:
    """Seed every section's position with DeepSlice. NOT IMPLEMENTED.

    Returns a mapping of slice id -> position in atlas-native millimetres,
    which the seed node writes with ``position_source="deepslice"``.
    """
    del state, ctx
    raise NotImplementedError(
        "DeepSlice seeding is not implemented yet. It ships later as the "
        "optional 'deepslice' extra; until then the seed node uses anchor "
        "seeding (a few single-slice estimates plus interpolation)."
    )
