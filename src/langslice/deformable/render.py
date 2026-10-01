"""Smooth drawing of a fitted atlas on the ORIGINAL section.

``DeformableRecord.labels`` samples atlas labels nearest-neighbour from the
25 um atlas grid, so borders traced from it are stair-stepped on finer
section pixels. Here the borders are drawn from the final map itself (linear
placement composed with the residual field), the way
:func:`langslice.atlas.render.placed_border_coverage` draws a placement:
each region's indicator on the atlas grid is blurred, sampled bilinearly at
a supersampled grid of section positions pushed through the composed map,
every fine pixel takes the region that covers it most, shared edges become
one line, widened and area-averaged down so they are antialiased. That
function only knows affine maps, so the composed (non-affine) sampling lives
here; the stroke step is the same few lines.

Selected regions (acronyms or ids, descendants included) can be highlighted:
their edges are drawn in a strong color over a faint outline of everything
else (color-family regions, the set the image model was shown). Nothing is
drawn outside the section's tissue.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.atlas.render import family_labels
from langslice.deformable.atlas_images import native_labels, resolve_structures, with_descendants
from langslice.deformable.record import DeformableRecord

#: Atlas-grid blur of each region indicator before bilinear sampling.
#: 1.0 leaves a ripple along shallow edges (the annotation's own 25 um stair
#: steps); 2.0 closes thin regions into dotted blobs.
BORDER_SMOOTHING_PX = 1.4
STRONG_COLOR = (255, 255, 0)
FAINT_ALPHA = 0.35
#: Stride of the coarse coordinate grid that finds each region's fine-pixel box.
COARSE_STEP = 8


def composed_native_grid(
    record: DeformableRecord, supersample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Native atlas-plane (x, y) of every supersampled section position, float32.

    Fine pixel centre (i, j) sits at section position ((i - (s-1)/2)/s, ...);
    the residual field is sampled there bilinearly, then the inverse linear
    placement maps the displaced position to the native plane (the order of
    :meth:`DeformableRecord.native_coordinates`).
    """
    width, height = record.section_size
    s = int(supersample)
    xs = (np.arange(width * s, dtype=np.float32) - (s - 1) / 2) / s
    ys = (np.arange(height * s, dtype=np.float32) - (s - 1) / 2) / s
    map_x, map_y = np.meshgrid(xs, ys)
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


def smooth_owner_map(
    labels: np.ndarray, native_x: np.ndarray, native_y: np.ndarray,
    *, smoothing_px: float = BORDER_SMOOTHING_PX,
) -> np.ndarray:
    """Region of each fine pixel: the one whose blurred indicator covers it most.

    Each region is sampled only inside its own bounding box (plus the blur's
    reach) so the cost does not grow with the number of regions times the
    fine grid.
    """
    height, width = labels.shape
    owner = np.zeros(native_x.shape, dtype=labels.dtype)
    best = np.full(native_x.shape, -1.0, dtype=np.float32)
    pad = int(np.ceil(3 * smoothing_px)) + 2
    coarse_x = native_x[::COARSE_STEP, ::COARSE_STEP]
    coarse_y = native_y[::COARSE_STEP, ::COARSE_STEP]
    for region in np.unique(labels):
        mask = labels == region
        if region == 0:
            box = (0, width, 0, height)  # the background also lies outside the plane
        else:
            ys, xs = np.nonzero(mask)
            box = (max(xs.min() - pad, 0), min(xs.max() + pad + 1, width),
                   max(ys.min() - pad, 0), min(ys.max() + pad + 1, height))
        x0, x1, y0, y1 = box
        indicator = mask[y0:y1, x0:x1].astype(np.float32)
        if smoothing_px > 0:
            indicator = cv2.GaussianBlur(
                indicator, (0, 0), smoothing_px,
                borderType=cv2.BORDER_REPLICATE if region == 0 else cv2.BORDER_CONSTANT)
        near = ((coarse_x >= x0 - 1) & (coarse_x <= x1) & (coarse_y >= y0 - 1)
                & (coarse_y <= y1))
        rows, cols = np.nonzero(near.any(axis=1))[0], np.nonzero(near.any(axis=0))[0]
        if rows.size == 0 or cols.size == 0:
            continue
        # Coarse indices back to fine pixels, with one coarse step of slack.
        r0 = max(int(rows[0]) * COARSE_STEP - COARSE_STEP, 0)
        r1 = min(int(rows[-1]) * COARSE_STEP + 2 * COARSE_STEP, native_x.shape[0])
        c0 = max(int(cols[0]) * COARSE_STEP - COARSE_STEP, 0)
        c1 = min(int(cols[-1]) * COARSE_STEP + 2 * COARSE_STEP, native_x.shape[1])
        cover = cv2.remap(
            indicator, np.ascontiguousarray(native_x[r0:r1, c0:c1] - np.float32(x0)),
            np.ascontiguousarray(native_y[r0:r1, c0:c1] - np.float32(y0)),
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=float(region == 0),
        )
        view_best = best[r0:r1, c0:c1]
        wins = cover > view_best
        view_best[wins] = cover[wins]
        owner[r0:r1, c0:c1][wins] = region
    return owner


def _stroke(edges: np.ndarray, size: tuple[int, int], s: int, width_px: float) -> np.ndarray:
    radius = max(0.5, width_px * s / 2.0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(np.ceil(radius)) + 1,) * 2)
    stroke = cv2.dilate(edges.astype(np.uint8) * 255, kernel)
    return cv2.resize(stroke, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0


def _owner_edges(owner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels on a change of owner, drawn once per shared edge (right/down neighbours)."""
    horizontal = owner[:, 1:] != owner[:, :-1]
    vertical = owner[1:, :] != owner[:-1, :]
    return horizontal, vertical


def warped_border_coverage(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), warped: bool = True,
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    native: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(faint, strong) border coverage in [0, 1] on the section grid, tissue-clipped.

    *strong* is the edges that touch a highlighted region (leaf regions of the
    chosen structures and their descendants); with no *highlight* every edge is
    strong. *faint* is every other edge between color-family regions. With
    ``warped=False`` the residual field is ignored (the linear placement alone).
    """
    s = int(supersample)
    width, height = record.section_size
    leaf = native if native is not None else native_labels(atlas, record.placement)
    ids = with_descendants(atlas, resolve_structures(atlas, highlight)) if highlight else ()
    family = family_labels(leaf, atlas).astype(np.int64)
    flagged = np.isin(leaf, list(ids)) if ids else np.zeros(leaf.shape, dtype=bool)
    # Highlighted leaves keep their own identity (odd codes), everything else
    # is its color family (even codes), so the two sets never merge.
    grouped = np.where(flagged, leaf.astype(np.int64) * 2 + 1, family * 2)
    if warped:
        nx, ny = composed_native_grid(record, s)
    else:
        nx, ny = composed_native_grid(_without_field(record), s)
    owner = smooth_owner_map(grouped, nx, ny, smoothing_px=smoothing_px)
    strong_flag = (owner & 1).astype(bool)
    h_edge, v_edge = _owner_edges(owner)
    edges = np.zeros(owner.shape, dtype=bool)
    edges[:, 1:] |= h_edge
    edges[1:, :] |= v_edge
    touch = np.zeros(owner.shape, dtype=bool)
    touch[:, 1:] |= h_edge & (strong_flag[:, 1:] | strong_flag[:, :-1])
    touch[1:, :] |= v_edge & (strong_flag[1:, :] | strong_flag[:-1, :])
    strong_edges = touch if ids else edges
    fine_tissue = cv2.resize(record.tissue.astype(np.uint8), (width * s, height * s),
                             interpolation=cv2.INTER_NEAREST) > 0
    # One section pixel of slack so a border on the tissue edge itself shows.
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    fine_tissue = cv2.dilate(fine_tissue.astype(np.uint8), reach) > 0
    faint = _stroke(edges & ~strong_edges & fine_tissue, (width, height), s, width_px)
    strong = _stroke(strong_edges & fine_tissue, (width, height), s, width_px)
    return faint, strong


def _without_field(record: DeformableRecord) -> DeformableRecord:
    from dataclasses import replace

    return replace(record, field_mm=np.zeros_like(record.field_mm))


def draw_warped_borders(
    image: Image.Image, record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), warped: bool = True,
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    color: tuple[int, int, int] = STRONG_COLOR, native: np.ndarray | None = None,
) -> Image.Image:
    """The fitted atlas borders, smooth and antialiased, on the original section.

    Without *highlight* every border between color-family regions is drawn in
    *color*. With it, only the selected structures' edges are strong, over a
    faint outline of the rest. Clipped to the tissue.
    """
    faint, strong = warped_border_coverage(
        record, atlas, highlight=highlight, warped=warped, width_px=width_px,
        smoothing_px=smoothing_px, supersample=supersample, native=native,
    )
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if base.shape[:2] != faint.shape:
        raise ValueError("The image must be on the record's section grid")
    ink = np.array(color, dtype=np.float32)
    alpha = (FAINT_ALPHA * faint)[..., None]
    base = base * (1.0 - alpha) + ink * alpha
    alpha = strong[..., None]
    base = base * (1.0 - alpha) + ink * alpha
    return Image.fromarray(np.rint(base).astype(np.uint8))
