"""Damage by atlas region: what every fit leaves out, and the picture of a mark.

A section's damage is the atlas regions it is missing, or has so badly
displaced that they would wreck a fit (``SliceState.damaged_regions``):
acronyms or ids, each optionally one side (``"CTX:left"``,
:mod:`langslice.core.atlas.sides`, the section's displayed left and right).
Named by atlas region, a mark moves with the registration. Every fit and the
image model's trace take their regions through :func:`exclusions`, so the
marked regions are left out without the caller repeating them.

:func:`damage_picture` shows a mark: the marked regions hatched on the
section under its current registration (stored transform and applied
deformation) and on the atlas at the section's plane, side by side, on one
canvas and one crop, so the two panels match pixel for pixel.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core.atlas.sides import SEPARATOR, split_side

if TYPE_CHECKING:
    from langslice.core.canvas import PanelFrame
    from langslice.core.state import SliceState, StackState
    from langslice.core.workspace import Workspace

#: The hatch inside a marked region: the second ink of the deformable fit's
#: pictures (excluded regions), so a marked region reads the same everywhere.
HATCH_COLOR = (255, 64, 160)
#: Distance between hatch lines and their width, in picture pixels.
HATCH_SPACING_PX = 9
HATCH_WIDTH_PX = 2
#: Strength of the hatch lines over the picture (the tissue shows between them).
HATCH_ALPHA = 0.8


class DamagePictureError(ValueError):
    """A mark that cannot be drawn, with its error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def entry_key(entry: str | int) -> tuple[str, str | None]:
    """``(region lowercased, side or None)``: two entries name the same thing
    exactly when their keys are equal. ``ValueError`` for an unknown side."""
    region, side = split_side(entry)
    return region.lower(), side


def normalized_entries(entries: Iterable[str | int]) -> list[str]:
    """Region entries stripped, the side lowercased (``"CTX:left"``), blanks
    and repeats (case-insensitive) dropped, in order. ``ValueError`` for an
    unknown side."""
    out: dict[tuple[str, str | None], str] = {}
    for entry in entries:
        if not str(entry).strip():
            continue
        region, side = split_side(entry)
        if not region:
            raise ValueError(f"Region entry {str(entry)!r} names no region")
        out.setdefault(entry_key(entry), region if side is None else f"{region}{SEPARATOR}{side}")
    return list(out.values())


def exclusions(
    record: SliceState, restrict_to: Sequence[str] = (), exclude: Sequence[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """``(restrict_to, exclude)`` one fit of *record* runs with.

    *restrict_to* (a fit by these regions only; empty: every region) is
    kept as given. The section's marked regions are added to *exclude* (the
    older per-call exclusions, still taken), except a marked entry that
    *restrict_to* names exactly (same region, same side or none): an
    explicit request for that region wins. Repeats are dropped
    (case-insensitive). Every fit (``ops.transforms.fit_affine``,
    ``ops.deformable.fit_deformable``) and the image model's trace
    (``ops.traces.trace_borders``) take their regions through here.
    """
    restrict = tuple(str(name).strip() for name in restrict_to if str(name).strip())
    named = {entry_key(name) for name in restrict}
    marked = [name for name in record.damaged_regions if entry_key(name) not in named]
    seen: set[tuple[str, str | None]] = set()
    excluded: list[str] = []
    for name in (*exclude, *marked):
        text = str(name).strip()
        if not text or entry_key(text) in seen:
            continue
        seen.add(entry_key(text))
        excluded.append(text)
    return restrict, tuple(excluded)


# --- the picture ----------------------------------------------------------------------


def _picture_mask(panel: PanelFrame, native: np.ndarray) -> np.ndarray:
    """*native* (a mask on the atlas plane grid) on the panel's picture
    pixels, nearest neighbour, through the mapping its borders are drawn with
    (as ``layers.labels_layer``); False in the caption band."""
    width, height = panel.size
    out = np.zeros((height, width), dtype=bool)
    x0, y0, x1, y1 = panel.content_box
    bx, by = panel.crop_box[0], panel.crop_box[1]
    scale = panel.geometry.atlas_scale
    ox, oy = panel.geometry.atlas_offset
    rows = np.rint(((np.arange(y0, y1) - y0) / panel.factor + by - oy) / scale).astype(np.int64)
    cols = np.rint(((np.arange(x0, x1) - x0) / panel.factor + bx - ox) / scale).astype(np.int64)
    keep_r = (rows >= 0) & (rows < native.shape[0])
    keep_c = (cols >= 0) & (cols < native.shape[1])
    block = native[np.ix_(rows[keep_r], cols[keep_c])]
    target = out[y0:y0 + len(rows), x0:x0 + len(cols)]
    target[np.ix_(np.nonzero(keep_r)[0], np.nonzero(keep_c)[0])] = block
    return out


def _hatched(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """A copy of *image* with diagonal :data:`HATCH_COLOR` lines inside *mask*."""
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
    rows, cols = np.indices(mask.shape)
    lines = mask & (((rows + cols) % HATCH_SPACING_PX) < HATCH_WIDTH_PX)
    ink = np.asarray(HATCH_COLOR, dtype=np.float32)
    pixels[lines] = pixels[lines] * (1.0 - HATCH_ALPHA) + ink * HATCH_ALPHA
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), mode="RGB")


def _native_marks(
    ws: Workspace, panel: PanelFrame, regions: Sequence[tuple[str, frozenset[int]]],
    position_mm: float, plane: str, pitch_deg: float, yaw_deg: float,
) -> np.ndarray:
    """The marked regions on the panel's native atlas plane, sides resolved
    through its placement exactly as ``canvas.physical_views`` resolves them
    for the highlighted outlines."""
    from langslice.core.atlas.sides import restrict
    from langslice.core.canvas import regions_left

    annotation = np.asarray(panel.geometry.annotation)
    linear = np.asarray(panel.section_matrix, dtype=np.float64)[:2, :2]
    left = regions_left(ws.atlas, regions, position_mm, plane, pitch_deg, yaw_deg,
                        np.linalg.inv(linear) * panel.geometry.atlas_scale)
    native = np.zeros(annotation.shape[:2], dtype=bool)
    for name, ids in regions:
        native |= restrict(np.isin(annotation, list(ids)), split_side(name)[1], left)
    return native


def damage_picture(
    ws: Workspace,
    state: StackState,
    record: SliceState,
    *,
    store: Any = None,
    long_edge: int | None = None,
    border_color: str = "#ffff00",
    border_thickness: float | None = None,
) -> Image.Image:
    """The section's marked regions, hatched, on the section and on the atlas.

    Left: the section under its current registration (the stored transform,
    identity without; the applied deformation from *store*, a
    ``deformation.RecordStore``, when it is current) with the atlas borders
    faint and the marked regions' outlines strong. Right: the atlas template
    at the section's position and angles, on the same canvas and crop, with
    the same lines. Inside the marked regions both panels carry diagonal
    :data:`HATCH_COLOR` lines, the tissue visible between them. Each panel
    is *long_edge* on its long side (``display.default_options``'s size
    when None). Captioned with the marked regions and the note.

    :class:`DamagePictureError` ``NO_POSITION`` for a section without a
    position, ``NO_REGIONS`` for one without marked regions;
    ``ValueError`` (unknown regions) and ``atlas.sides.SideError`` (a side
    the placement cannot define) pass through.
    """
    from langslice.core.captions import caption
    from langslice.core.deformable.atlas_images import resolve_entries
    from langslice.core.display import DEFAULT_BORDER_THICKNESS, default_options
    from langslice.core.layers import note
    from langslice.core.placement import current_warp, draw_canvas, stored_placement
    from langslice.core.sections import PREVIEW_LONG_EDGE, render_slice
    from langslice.core.sheets import beside
    from langslice.core.transform import calibrate

    if record.position_mm is None:
        raise DamagePictureError("NO_POSITION", "The section has no position, so its "
                                 "marked regions cannot be drawn on it.")
    entries = list(record.damaged_regions)
    if not entries:
        raise DamagePictureError("NO_REGIONS", "The section has no marked regions.")
    resolved = resolve_entries(ws.atlas, entries)
    regions = tuple((name, ids) for name, (_region, _side, ids) in zip(entries, resolved,
                                                                     strict=True))
    base = default_options("overlay") if long_edge is None else default_options(
        "overlay", long_edge=int(long_edge))
    options = replace(base, regions=regions, border_color=border_color,
                      border_thickness=float(border_thickness or DEFAULT_BORDER_THICKNESS))
    position = float(record.position_mm)
    section = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
    um_per_px, _source = calibrate(state, ws, record, section)
    params, kind = stored_placement(record, section)
    warp = current_warp(store, state, record) if store is not None else None
    shown = draw_canvas(
        ws, state, record, section, um_per_px, position, params, options, mode="overlay",
        label=f"{record.id} damage", warp=warp,
        matrix_label=f"{kind} transform" + (" + deformation" if warp is not None else ""),
    )
    atlas = draw_canvas(
        ws, state, record, section, um_per_px, position, params,
        replace(options, atlas_channels=("template", "borders")), mode="template",
        label=f"{record.id} damage", matrix_label=f"{kind} transform",
    )
    panels = []
    for canvas in (shown, atlas):
        image, panel = canvas.images[0], canvas.panels[0]
        marks = _native_marks(ws, panel, regions, position, state.plane,
                              record.pitch_deg, record.yaw_deg)
        panels.append(_hatched(image, _picture_mask(panel, marks)))
    text = ("marked damaged (hatched): " + ", ".join(entries)
            + (f"  note: {record.damage_note}" if record.damage_note else "")
            + "\nleft: the section under its current registration; right: the atlas")
    return note(caption(beside(*panels), text), sections=(record.id,), mode="damage")
