"""Compatibility shim for sibling repos; use :mod:`langslice.core.sections` and
:mod:`langslice.core.canvas`."""

from langslice.core.canvas import canvas_geometry
from langslice.core.sections import PREVIEW_LONG_EDGE, canvas_um_per_px, render_slice

__all__ = ["PREVIEW_LONG_EDGE", "canvas_geometry", "canvas_um_per_px", "render_slice"]
