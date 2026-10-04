"""The fit after the image model's border correction: the deformable package's.

Every caller that turns the model's corrected lines into a deformation (both
border routes of ``nonlinear register`` and the ABBA registration plugin)
fits them here with :mod:`langslice.core.deformable`, route B: the extracted lines
against the colour-family borders the model was shown
(:func:`langslice.core.deformable.traced_settings`), the linear agent's
``fit_deformable`` with ``fit_section="traced_lines"``. The engine is Elastix
by default (``traced_lines``: lines against borders, mean squares, medium
stiffness): a core dependency, so the result does not change with the
optional ANTs install, and on the 2026-10-04 before/after check (a stubbed
reply on a real coronal section, the supplied placement shrunk, turned and
shifted) it followed the reply's lines closest, outline included, where
ANTs' named-region mode (``engine="ants"``, ``traced_borders``, the agent's
recommendation for a small correction of a good placement) left the
ventrolateral outline short, its regions named after a placement that far
off. This replaced the April 2026 Elastix borders B-spline residual fit on
2026-10-04 (layered refactor, phase 4). The model's request, prompt and
line extraction are untouched (:mod:`langslice.core.nonlinear.border_refinement`);
only the fit after them is this.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image

from langslice.core.atlas.render import atlas_um_per_px
from langslice.core.deformable import (
    DeformableRecord,
    Placement,
    draw_warped_borders,
    fit_section,
    traced_settings,
)
from langslice.core.deformable.atlas_images import native_labels
from langslice.core.deformable.geometry import sample_native
from langslice.core.deformable.settings import Engine
from langslice.core.space import Plane


@dataclass
class BorderFit:
    """One fitted correction, in the canvas frame the model saw."""

    #: (H, W, 2) canvas pixels: output pixel ``q`` lies on the placed atlas
    #: at ``q + field_px[q]`` (the record's millimetre field over its mm/px).
    field_px: np.ndarray
    #: Leaf atlas ids on the canvas under the fit (not clipped to tissue).
    fitted_labels: np.ndarray
    #: The fitted family borders drawn smoothly on the canvas, clipped to tissue
    #: (:func:`langslice.core.deformable.draw_warped_borders`).
    fitted_border_overlay: Image.Image
    record: DeformableRecord
    elapsed: float
    metadata: dict[str, Any] = field(default_factory=dict)


def placement_on_canvas(
    atlas: Any,
    atlas_to_canvas: np.ndarray,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    source: str = "supplied",
) -> Placement:
    """A deformable :class:`~langslice.core.deformable.Placement` of the atlas plane on a canvas.

    *atlas_to_canvas* maps native atlas-plane pixel centres (the plane as
    ``annotation_slice`` returns it: no image axes, no mirror) to canvas
    pixel centres. The canvas has no pixel size of its own, so the
    placement's scale gives it: an atlas pixel is ``atlas_um_per_px`` µm
    and covers ``sqrt(|det|)`` canvas pixels.
    """
    matrix = np.asarray(atlas_to_canvas, dtype=np.float64)
    scale = float(np.sqrt(abs(np.linalg.det(matrix[:2, :2]))))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The atlas placement on the canvas is singular")
    return Placement(
        atlas_name=atlas_name, position_mm=float(position_mm), plane=plane,
        pitch_deg=float(pitch_deg), yaw_deg=float(yaw_deg), atlas_to_section=matrix,
        section_mm_per_px=atlas_um_per_px(atlas) / 1000.0 / scale, source=source,
    )


def fit_border_lines(
    canvas: Image.Image,
    lines: np.ndarray,
    atlas: Any,
    placement: Placement,
    *,
    engine: Engine = "elastix",
    native: np.ndarray | None = None,
) -> BorderFit:
    """Fit the atlas placement's family borders onto the model's *lines*.

    *lines* is the boolean line mask extracted from the model's reply on the
    *canvas* grid, and *placement* the atlas on that canvas. *native*
    replaces the atlas plane's labels with the caller's own grid (ABBA's
    labels sampled at its own coordinates, placed by an identity).
    *engine* is ``elastix`` (lines against borders) or ``ants`` (the lines
    as named regions too, :func:`langslice.core.deformable.traced_settings`).
    Raises when the fit cannot run (no tissue, masks empty, an engine
    error).
    """
    mask = np.asarray(lines, dtype=bool)
    if mask.shape != (canvas.height, canvas.width):
        raise ValueError("The line mask must be on the canvas grid")
    if not mask.any():
        raise ValueError("Image model returned no usable yellow anatomical boundaries")
    settings = traced_settings(engine)
    started = time.perf_counter()
    record = fit_section(canvas, atlas, placement, settings, lines=mask, native=native)
    elapsed = time.perf_counter() - started
    labels = native if native is not None else native_labels(atlas, placement)
    fitted = sample_native(np.asarray(labels), record.native_coordinates())
    overlay = draw_warped_borders(canvas, record, atlas, native=native)
    field_px = (np.asarray(record.field_mm, dtype=np.float64) / record.mm_per_px)
    flags = record.diagnostics.get("flags", [])
    metadata: dict[str, Any] = {
        "fit": "deformable", "fit_inputs": "model lines against the atlas family borders",
        "settings": settings.to_dict(), "metric": settings.metric,
        "engine": {key: record.engine.get(key) for key in ("name", "version", "notes")},
        "section_mm_per_px": record.mm_per_px,
        "flags": flags, "diagnostics": record.diagnostics,
        "max_residual_displacement_px": float(np.linalg.norm(field_px, axis=2).max()),
        "scope": "residual deformation only; the initial placement is separate",
        "description": "Mechanical fit diagnostics, not an assessment of model anatomy",
    }
    return BorderFit(field_px=field_px, fitted_labels=fitted, fitted_border_overlay=overlay,
                     record=record, elapsed=elapsed,
                     metadata=json.loads(json.dumps(metadata, default=_plain)))


def _plain(value: Any) -> Any:
    """JSON for the numpy values in the fit's diagnostics."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")
