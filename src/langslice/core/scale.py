"""One scale for a section and its atlas, wherever the two are drawn apart.

The physical canvas (:func:`langslice.core.canvas.physical_views`) puts the
atlas on the section's frame at true scale. The pictures that show the two
as separate panels (the positioning picture, the opening strips) frame each
to its own anatomy; this module sizes both panels at ONE micrometres per
pixel, so the section reads at its true size against the atlas, as on the
canvas:

- the section's scale is the one its placement pictures draw it at: its
  calibration (:func:`langslice.core.transform.calibrate`: the file, the
  host, or the estimate) times the isotropic scale of its stored transform
  (:func:`stored_scale`), on the :data:`PREVIEW_LONG_EDGE` working frame;
- the atlas plane is at its voxel size (:func:`atlas_um_per_px`);
- the common scale puts the larger of the panels at the picture's long
  edge (:func:`pair_um_per_px`); the other is smaller, never stretched.
"""

from __future__ import annotations

import math
from typing import Any

from PIL import Image

from langslice.core.appearance import Look
from langslice.core.atlas.render import atlas_um_per_px
from langslice.core.atlas_fetch import atlas_section
from langslice.core.sections import PREVIEW_LONG_EDGE, render_cache_key, render_slice
from langslice.core.state import Angles, SliceState, StackState
from langslice.core.transform import calibrate
from langslice.core.workspace import Workspace


def stored_scale(record: SliceState) -> float:
    """The isotropic scale of the section's stored transform (1.0 without).

    The six stored numbers ``[a, b, tx, c, d, ty]`` map the working frame
    onto the canvas; ``sqrt(|ad - bc|)`` is how much larger the placement
    pictures draw each section pixel (the frame's width and height cancel).
    """
    values = (record.transform or {}).get("params")
    if values is None or len(values) != 6:
        return 1.0
    a, b, _tx, c, d, _ty = (float(v) for v in values)
    det = abs(a * d - b * c)
    return math.sqrt(det) if det > 0 and math.isfinite(det) else 1.0


def section_um_per_px(
    ws: Workspace, state: StackState, record: SliceState, um_per_px: float | None = None,
) -> float:
    """Micrometres per pixel of the section's working frame as its placement
    pictures draw it: *um_per_px* (None: :func:`~langslice.core.transform.calibrate`)
    times :func:`stored_scale`."""
    if um_per_px is None:
        section = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
        um_per_px, _source = calibrate(state, ws, record, section)
    return float(um_per_px) * stored_scale(record)


def framed_um_per_px(
    ws: Workspace, record: SliceState, working_um: float, *, long_edge: int, look: Look = None,
) -> float:
    """Micrometres per pixel of the tissue-framed render at *long_edge*,
    given *working_um*, that of the unframed working frame."""
    render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
    render_slice(ws, record, long_edge=long_edge, frame=True, look=look)
    working = ws.render_scale[render_cache_key(ws, record, long_edge=PREVIEW_LONG_EDGE,
                                               frame=False)]
    framed = ws.render_scale[render_cache_key(ws, record, long_edge=long_edge, frame=True,
                                              look=look)]
    return float(working_um) * framed / working


def section_extent_um(ws: Workspace, record: SliceState, working_um: float) -> float:
    """The long edge of the tissue-framed section, in micrometres."""
    picture = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE, frame=True)
    return max(picture.size) * framed_um_per_px(ws, record, working_um,
                                                long_edge=PREVIEW_LONG_EDGE)


def atlas_extent_um(
    ws: Workspace, state: StackState, position_mm: float, *, angles: Angles | None = None,
) -> float:
    """The long edge of the anatomy-framed atlas plane at *position_mm*, in micrometres."""
    picture = atlas_section(ws, state, position_mm, frame=True, angles=angles)
    return max(picture.size) * atlas_um_per_px(ws.atlas)


def finest_um_per_px(ws: Workspace, record: SliceState, working_um: float) -> float:
    """Micrometres per pixel of the section's working copy, given
    *working_um* (the working frame's): no picture draws the section finer."""
    render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
    _source, file_px_per_px = ws.working_source(record.id)
    working = ws.render_scale[render_cache_key(ws, record, long_edge=PREVIEW_LONG_EDGE,
                                               frame=False)]
    return float(working_um) * float(file_px_per_px) / working


def pair_um_per_px(extents_um: Any, long_edge: int, *, finest: float = 0.0) -> float:
    """The one scale at which the largest of *extents_um* (long edges in
    micrometres) fills *long_edge* pixels, never finer than *finest* (the
    section's own pixels: a small snapshot is not upsampled)."""
    return max(max(float(value) for value in extents_um) / max(1, int(long_edge)),
               float(finest))


def pair_scale(
    ws: Workspace, state: StackState, record: SliceState, position_mm: float, long_edge: int,
    *, um_per_px: float | None = None,
) -> tuple[float, float]:
    """``(picture um/px, working um/px)`` of *record* drawn with the atlas at
    *position_mm* (its own angles): the larger of the two framed panels at
    *long_edge*. *um_per_px* is the section's calibration when the caller
    holds it (None: :func:`section_um_per_px` calibrates)."""
    working_um = section_um_per_px(ws, state, record, um_per_px)
    shown = pair_um_per_px(
        (section_extent_um(ws, record, working_um),
         atlas_extent_um(ws, state, position_mm, angles=record.angles)),
        long_edge, finest=finest_um_per_px(ws, record, working_um),
    )
    return shown, working_um


def section_at(
    ws: Workspace, record: SliceState, um_per_px: float, *, working_um: float,
    long_edge: int, look: Look = None,
) -> Image.Image:
    """The section tissue-framed at exactly *um_per_px*.

    Drawn from the framed render at *long_edge* (the call's size, so the
    render cache keeps its usual entries), then resized by the one factor
    that brings it to *um_per_px*: never stretched, at most *long_edge*
    when *um_per_px* is :func:`pair_um_per_px` of a pair it belongs to.
    """
    picture = render_slice(ws, record, long_edge=int(long_edge), frame=True, look=look)
    have = framed_um_per_px(ws, record, working_um, long_edge=int(long_edge), look=look)
    return resized(picture, have / float(um_per_px))


def atlas_at(
    ws: Workspace, state: StackState, position_mm: float, um_per_px: float,
    *, angles: Angles | None = None,
) -> Image.Image:
    """The atlas template plane at *position_mm*, anatomy-framed, at exactly *um_per_px*."""
    picture = atlas_section(ws, state, position_mm, frame=True, angles=angles)
    return resized(picture, atlas_um_per_px(ws.atlas) / float(um_per_px))


def resized(picture: Image.Image, factor: float) -> Image.Image:
    """*picture* scaled by *factor* on both axes (the same object at 1)."""
    if abs(factor - 1.0) < 1e-6:
        return picture
    size = (max(1, round(picture.width * factor)), max(1, round(picture.height * factor)))
    if size == picture.size:
        return picture
    resample = Image.Resampling.LANCZOS if factor < 1.0 else Image.Resampling.BILINEAR
    return picture.resize(size, resample)


__all__ = [
    "atlas_at", "atlas_extent_um", "finest_um_per_px", "framed_um_per_px",
    "pair_scale", "pair_um_per_px", "resized", "section_at",
    "section_extent_um", "section_um_per_px", "stored_scale",
]
