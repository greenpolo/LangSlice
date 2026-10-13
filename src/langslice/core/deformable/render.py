"""Smooth drawing of a fitted atlas on the ORIGINAL section.

``DeformableRecord.labels`` samples atlas labels nearest-neighbour from the
25 um atlas grid, so borders traced from it are stair-stepped on finer
section pixels. Here the borders are drawn from the final map itself (linear
placement composed with the residual field), the way
:func:`langslice.core.atlas.render.placed_border_coverage` draws a placement:
each region's indicator on the atlas grid is blurred, sampled bilinearly at
a supersampled grid of section positions pushed through the composed map,
every fine pixel takes the region that covers it most, shared edges become
one line, widened and area-averaged down so they are antialiased. Only
the grid differs (:func:`composed_native_grid` adds the residual field); the
sampler, the boundaries and the stroke are the shared
:func:`~langslice.core.atlas.render.smooth_owner_map`,
:func:`~langslice.core.atlas.render.region_boundaries` and
:func:`~langslice.core.atlas.render.stroke_boundaries`.

Selected regions (acronyms or ids, descendants included) can be highlighted:
their edges are drawn in a strong color over a faint outline of everything
else.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.atlas.levels import OVERLAY_DEPTH, labels_at
from langslice.core.atlas.render import (
    HIGHLIGHT_WIDTH,
    region_boundaries,
    smooth_owner_map,
    stroke_boundaries,
    supersampled_positions,
)
from langslice.core.deformable.atlas_images import native_labels, placement_left, regions_mask
from langslice.core.deformable.record import DeformableRecord

#: Atlas-grid blur of each region indicator before bilinear sampling.
#: 1.0 leaves a ripple along shallow edges (the annotation's own 25 um stair
#: steps); 2.0 closes thin regions into dotted blobs.
BORDER_SMOOTHING_PX = 1.4
STRONG_COLOR = (255, 255, 0)
#: Second ink, for a marked set of regions (those excluded from a fit).
MARKED_COLOR = (255, 64, 160)
FAINT_ALPHA = 0.35


def composed_native_grid(
    record: DeformableRecord, supersample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Native atlas-plane (x, y) of every supersampled section position, float32.

    Fine pixel centre (i, j) sits at section position ((i - (s-1)/2)/s, ...);
    the residual field is sampled there bilinearly, then the inverse linear
    placement maps the displaced position to the native plane (the order of
    ``core.maps.native_points``).
    """
    map_x, map_y = supersampled_positions(record.section_size, supersample)
    scale = 1.0 / record.mm_per_px
    dx = cv2.remap(np.ascontiguousarray(record.field_mm[..., 0], dtype=np.float32),
                   map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    dy = cv2.remap(np.ascontiguousarray(record.field_mm[..., 1], dtype=np.float32),
                   map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    px = map_x + dx * scale
    py = map_y + dy * scale
    inverse = np.linalg.inv(record.placement.atlas_to_section)
    nx = (inverse[0, 0] * px + inverse[0, 1] * py + inverse[0, 2]).astype(np.float32)
    ny = (inverse[1, 0] * px + inverse[1, 1] * py + inverse[1, 2]).astype(np.float32)
    return nx, ny


def warped_border_layers(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), marked: Iterable[str | int] = (),
    warped: bool = True, depth: int = OVERLAY_DEPTH,
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    native: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Border coverage in [0, 1] on the section grid, per layer.

    Layers: ``strong`` (edges touching a *highlight* region, stroked
    :data:`~langslice.core.atlas.render.HIGHLIGHT_WIDTH` times as thick;
    every edge, at *width_px*, when nothing is highlighted), ``marked``
    (edges touching a *marked* region, e.g. the regions excluded from a
    fit), ``faint`` (every other edge between regions at ontology *depth*,
    :func:`~langslice.core.atlas.levels.labels_at`) and
    ``outer`` (edges against the background: the atlas's outer boundary).
    With ``warped=False`` the residual field is ignored (the linear placement alone).
    """
    s = int(supersample)
    width, height = record.section_size
    leaf = native if native is not None else native_labels(atlas, record.placement)
    highlight, marked = list(highlight), list(marked)
    # One-sided entries ("CTX:left") are sides of the section as drawn here.
    left = placement_left(atlas, record.placement, (*highlight, *marked))
    flagged = (regions_mask(atlas, leaf, highlight, left) if highlight
               else np.zeros(leaf.shape, dtype=bool))
    flagged_marks = (regions_mask(atlas, leaf, marked, left) if marked
                     else np.zeros(leaf.shape, dtype=bool))
    # Codes: a highlighted leaf is leaf*4+1, a marked leaf leaf*4+3, any other
    # pixel its region at *depth* *4 (background stays 0), so the sets never merge.
    big = leaf.astype(np.int64)
    level = labels_at(leaf, atlas, depth).astype(np.int64)
    grouped = np.where(flagged_marks, big * 4 + 3, np.where(flagged, big * 4 + 1, level * 4))
    nx, ny = composed_native_grid(record if warped else _without_field(record), s)
    owner = smooth_owner_map(grouped, nx, ny, smoothing_px=smoothing_px)
    edges, h_edge, v_edge = region_boundaries(owner)

    def touching(flag: np.ndarray) -> np.ndarray:
        touch = np.zeros(owner.shape, dtype=bool)
        touch[:, 1:] |= h_edge & (flag[:, 1:] | flag[:, :-1])
        touch[1:, :] |= v_edge & (flag[1:, :] | flag[:-1, :])
        return touch

    nothing = np.zeros(owner.shape, dtype=bool)
    marked_edges = touching((owner & 3) == 3) if marked else nothing
    strong_edges = (touching((owner & 3) == 1) if highlight else edges) & ~marked_edges
    size = (width, height)
    return {
        "faint": stroke_boundaries(edges & ~strong_edges & ~marked_edges, size, s, width_px),
        "strong": stroke_boundaries(strong_edges, size, s,
                                    width_px * (HIGHLIGHT_WIDTH if highlight else 1.0)),
        "marked": stroke_boundaries(marked_edges, size, s, width_px),
    }


def _without_field(record: DeformableRecord) -> DeformableRecord:
    from dataclasses import replace

    return replace(record, field_mm=np.zeros_like(record.field_mm))


def _drawn_layers(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int], marked: Iterable[str | int], warped: bool, outlines: str,
    depth: int, width_px: float, smoothing_px: float, supersample: int,
    native: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(faint, strong, marked)``: the coverage :func:`draw_warped_borders`
    draws, each layer as the *outlines* choice leaves it."""
    if outlines not in ("all", "none"):
        raise ValueError("outlines must be all or none")
    highlight = list(highlight)
    layers = warped_border_layers(
        record, atlas, highlight=highlight, marked=marked, warped=warped, depth=depth,
        width_px=width_px, smoothing_px=smoothing_px, supersample=supersample, native=native,
    )
    blank = np.zeros_like(layers["faint"])
    if highlight:
        strong = layers["strong"]
        faint = layers["faint"] if outlines == "all" else blank
    else:
        faint = blank
        strong = layers["strong"] if outlines == "all" else blank
    return faint, strong, layers["marked"]


def drawn_border_coverage(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), marked: Iterable[str | int] = (),
    warped: bool = True, outlines: str = "all", depth: int = OVERLAY_DEPTH,
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    native: np.ndarray | None = None,
) -> np.ndarray:
    """Every line :func:`draw_warped_borders` draws with these arguments, as
    coverage in [0, 1] at full strength (the faint lines too): a picture's
    borders layer."""
    layers = _drawn_layers(record, atlas, highlight=highlight, marked=marked, warped=warped,
                           outlines=outlines, depth=depth, width_px=width_px,
                           smoothing_px=smoothing_px, supersample=supersample, native=native)
    return np.maximum.reduce(layers)


def draw_warped_borders(
    image: Image.Image, record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), marked: Iterable[str | int] = (),
    warped: bool = True, outlines: str = "all", depth: int = OVERLAY_DEPTH,
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    color: tuple[int, int, int] = STRONG_COLOR, marked_color: tuple[int, int, int] = MARKED_COLOR,
    native: np.ndarray | None = None,
) -> Image.Image:
    """The fitted atlas borders, smooth and antialiased, on the original section.

    Without *highlight* every border between regions at ontology *depth* is drawn in
    *color*. With it, only the selected structures' edges are strong, over a
    faint outline of the rest. *marked* regions' edges are drawn in
    *marked_color* (a second set, e.g. regions excluded from the fit).
    *outlines* limits the other lines: ``all`` or ``none``.
    """
    faint, strong, marked_lines = _drawn_layers(
        record, atlas, highlight=highlight, marked=marked, warped=warped, outlines=outlines,
        depth=depth, width_px=width_px, smoothing_px=smoothing_px, supersample=supersample,
        native=native)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if base.shape[:2] != strong.shape:
        raise ValueError("The image must be on the record's section grid")
    for coverage, ink, weight in (
        (faint, color, FAINT_ALPHA), (strong, color, 1.0), (marked_lines, marked_color, 1.0),
    ):
        alpha = (weight * coverage)[..., None]
        base = base * (1.0 - alpha) + np.array(ink, dtype=np.float32) * alpha
    return Image.fromarray(np.rint(base).astype(np.uint8))


def resampled_record(
    record: DeformableRecord, size: tuple[int, int],
    box: tuple[int, int, int, int] | None = None,
) -> DeformableRecord:
    """The same deformation on a resized (and optionally cropped) section grid.

    For drawing at picture size: the field is in millimetres, so it is only
    resampled; the placement is carried onto the new pixel centres and the
    footprint mask resized. *box* (x0, y0, x1, y1 on the resized grid) crops.
    Only what drawing reads is carried: no inverse, labels or parent.
    """
    from dataclasses import replace

    from langslice.core.affine import pixel_center_map

    width, height = record.section_size
    field = np.stack([
        cv2.resize(np.ascontiguousarray(record.field_mm[..., k], dtype=np.float32), size,
                   interpolation=cv2.INTER_LINEAR) for k in range(2)
    ], axis=-1)
    footprint = cv2.resize(record.footprint.astype(np.uint8), size,
                           interpolation=cv2.INTER_NEAREST) > 0
    matrix = pixel_center_map((width, height), size) @ record.placement.atlas_to_section
    mm = record.mm_per_px * width / float(size[0])
    if box is not None:
        x0, y0, x1, y1 = box
        field, footprint = field[y0:y1, x0:x1], footprint[y0:y1, x0:x1]
        matrix = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ matrix
        size = (x1 - x0, y1 - y0)
    placement = replace(record.placement, atlas_to_section=matrix, section_mm_per_px=mm)
    return replace(
        record, placement=placement, section_size=size, field_mm=field,
        inverse_field_mm=None, labels=np.zeros(footprint.shape, dtype=record.labels.dtype),
        footprint=footprint, parent=None,
    )


def warp_section_image(image: Image.Image, record: DeformableRecord) -> Image.Image:
    """The section resampled into its placed-atlas frame by the residual warp.

    *image* is the section on the record's frame at any size (the same
    oriented render the fit grid is a resize of). Pixel q of the
    result shows section point ``q + inverse_field(q)``, so drawing the result
    under the LINEAR placement shows the full registration: the placement's
    atlas lines then fall where the warp put them on the tissue.
    """
    width, height = image.size
    inverse, mm = _inverse_pixels(record, (width, height))
    yy, xx = np.indices((height, width), dtype=np.float32)
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    out = cv2.remap(pixels, xx + inverse[..., 0] / mm, yy + inverse[..., 1] / mm,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)


def warp_source_points(
    record: DeformableRecord, size: tuple[int, int], x: np.ndarray, y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """The section points that :func:`warp_section_image` shows at points
    ``(x, y)`` of its result for a render of *size*: ``q + inverse_field(q)``
    (pixels of that render, centres at integers), the field read between its
    samples as the image's warp reads it."""
    inverse, mm = _inverse_pixels(record, size)
    mx = np.asarray(x, dtype=np.float32)
    my = np.asarray(y, dtype=np.float32)
    shift = [cv2.remap(np.ascontiguousarray(inverse[..., k]), mx, my, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE) for k in range(2)]
    return (np.asarray(x, dtype=np.float64) + shift[0] / mm,
            np.asarray(y, dtype=np.float64) + shift[1] / mm)


def _inverse_pixels(record: DeformableRecord, size: tuple[int, int]
                    ) -> tuple[np.ndarray, float]:
    """``(inverse field resized to *size*, mm per pixel of that size)``."""
    if record.inverse_field_mm is None:
        raise ValueError("A record resampled for drawing has no inverse field")
    width, height = size
    mm = record.mm_per_px * record.section_size[0] / float(width)

    def resized(field: np.ndarray) -> np.ndarray:
        return np.stack([
            cv2.resize(np.ascontiguousarray(field[..., k], dtype=np.float32), (width, height),
                       interpolation=cv2.INTER_LINEAR) for k in range(2)
        ], axis=-1)

    return resized(record.inverse_field_mm), mm
