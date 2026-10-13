"""Atlas geometry both methods draw from: annotation slices and contours.

Shared on purpose. ``linear`` draws region outlines over a section in physical
space and ``nonlinear`` draws region lines over its generated maps from the
same annotation, so the two methods agree about where a boundary is. Regions
are grouped by ontology depth alone (:mod:`langslice.core.atlas.levels`).

Nothing here loads a model, writes a file, or knows about a run — an atlas, a
position and (optionally) the block's cutting angles go in, geometry comes out.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from langslice.core.atlas.core import _resolve_idx_axis, orient_slice_for_display
from langslice.core.atlas.levels import labels_at
from langslice.core.space import Plane

__all__ = [
    "annotation_slice",
    "atlas_um_per_px",
    "is_dark_background",
    "placed_border_coverage",
    "region_boundaries",
    "region_contours",
    "smooth_owner_map",
    "stroke_boundaries",
    "supersampled_positions",
    "region_outlines",
]

#: How much thicker than the other atlas lines a picture draws the regions
#: it highlights (``grep_atlas_view``'s regions, a restricted fit's
#: ``restrict_to``, marked damage), the others drawn faint.
HIGHLIGHT_WIDTH = 2.0


def annotation_slice(
    atlas: Any,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> np.ndarray:
    """The atlas annotation at *position_mm*, oriented for display.

    With a non-zero cutting angle the plane is resliced obliquely instead of
    taken flat off the voxel grid: on a block cut at 4 degrees, matching the
    plane is worth far more than any fit tuning.
    """
    if pitch_deg or yaw_deg:
        from langslice.core.oblique import sample_oblique_annotation

        return sample_oblique_annotation(atlas, position_mm, plane, pitch_deg, yaw_deg)
    idx, axis = _resolve_idx_axis(atlas, position_mm, plane)
    return orient_slice_for_display(
        np.asarray(np.take(atlas.annotation, idx, axis=axis)), plane
    )


def atlas_um_per_px(atlas: Any) -> float:
    """Micrometres per pixel of an atlas render at native resolution."""
    return float(max(atlas.resolution))


def placed_border_coverage(
    labels: np.ndarray,
    atlas_to_image: np.ndarray,
    size: tuple[int, int],
    *,
    width_px: float = 2.0,
    smoothing_px: float = 0.8,
    supersample: int = 3,
) -> np.ndarray:
    """Coverage in [0, 1] of the placed region boundaries on a *size* image.

    *labels* is an atlas-resolution region map and *atlas_to_image* the 3x3
    (or 2x3) map from its pixel centres to the image's. A nearest-neighbour
    warp of the label map magnifies the atlas grid into a staircase; instead
    each region's indicator is blurred by *smoothing_px* atlas pixels and
    sampled bilinearly on a *supersample*-times finer grid, every fine
    pixel taking the region that covers it most (:func:`smooth_owner_map`).
    Boundaries between neighbouring fine pixels are one shared line (tracing
    each region separately draws a shared edge twice, one atlas pixel apart),
    widened to *width_px* image pixels and area-averaged back down, so the
    edges are antialiased.
    """
    s = int(supersample)
    matrix = np.asarray(atlas_to_image, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    px, py = supersampled_positions(size, s)
    inverse = np.linalg.inv(matrix)
    native_x = (inverse[0, 0] * px + inverse[0, 1] * py + inverse[0, 2]).astype(np.float32)
    native_y = (inverse[1, 0] * px + inverse[1, 1] * py + inverse[1, 2]).astype(np.float32)
    owner = smooth_owner_map(labels, native_x, native_y, smoothing_px=smoothing_px)
    edges, _horizontal, _vertical = region_boundaries(owner)
    return stroke_boundaries(edges, size, s, width_px)


#: Stride of the coarse coordinate grid that finds each region's fine-pixel box.
COARSE_STEP = 8


def supersampled_positions(
    size: tuple[int, int], supersample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Image-pixel ``(x, y)`` of every pixel centre of the *supersample*-times
    finer grid over a *size* image, float32: fine pixel ``(i, j)`` sits at
    ``((i - (s-1)/2) / s, (j - (s-1)/2) / s)``."""
    width, height = size
    s = int(supersample)
    xs = (np.arange(width * s, dtype=np.float32) - (s - 1) / 2) / s
    ys = (np.arange(height * s, dtype=np.float32) - (s - 1) / 2) / s
    map_x, map_y = np.meshgrid(xs, ys)
    return map_x, map_y


def smooth_owner_map(
    labels: np.ndarray, native_x: np.ndarray, native_y: np.ndarray,
    *, smoothing_px: float,
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


def region_boundaries(labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Where neighbouring pixels of *labels* hold different ids.

    ``(edges, horizontal, vertical)``: *horizontal* ``[r, c]`` is True where
    pixels ``(r, c)`` and ``(r, c + 1)`` differ, *vertical* ``[r, c]`` where
    ``(r, c)`` and ``(r + 1, c)`` do, and *edges* (the shape of *labels*)
    marks the right or lower pixel of each such pair, so a boundary two
    regions share is one line one pixel wide. Every pixel boundary picture
    (:func:`placed_border_coverage`, the warped borders of
    :mod:`langslice.core.deformable.render`) starts here.
    """
    horizontal = labels[:, 1:] != labels[:, :-1]
    vertical = labels[1:, :] != labels[:-1, :]
    edges = np.zeros(labels.shape, dtype=bool)
    edges[:, 1:] |= horizontal
    edges[1:, :] |= vertical
    return edges, horizontal, vertical


def stroke_boundaries(
    edges: np.ndarray, size: tuple[int, int], supersample: int, width_px: float,
) -> np.ndarray:
    """Coverage in [0, 1] on a *size* image of *edges*, a boolean mask on the
    *supersample*-times finer grid: widened to *width_px* image pixels and
    area-averaged down, so the lines are antialiased."""
    radius = max(0.5, width_px * supersample / 2.0)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(np.ceil(radius)) + 1,) * 2,
    )
    stroke = cv2.dilate(edges.astype(np.uint8) * 255, kernel)
    coverage = cv2.resize(stroke, size, interpolation=cv2.INTER_AREA)
    return coverage.astype(np.float32) / 255.0


def is_dark_background(image: Any) -> bool:
    """True for fluorescence-style canvases, False for light brightfield.

    Line and text colors flip on this: the same near-black acronym that reads
    cleanly on a white slide disappears on a dark-field section.
    """
    from PIL import Image

    if isinstance(image, Image.Image):
        gray = np.asarray(image.convert("L"), dtype=np.uint8)
    else:
        arr = np.asarray(image)
        gray = arr if arr.ndim == 2 else cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    return float(np.median(gray)) < 110.0


def _smooth_closed(points: np.ndarray, window: int) -> np.ndarray:
    """Circular moving average over a closed polygon's vertices.

    A traced label boundary is a staircase of unit steps; a short circular
    box filter erases the steps while leaving anything larger than the window
    where it was. Cheaper and more stable than resampling to a spline, and it
    preserves the vertex count so both sides of a shared boundary land on the
    same curve.
    """
    count = len(points)
    if window < 3 or count < 2 * window:
        return points
    half = window // 2
    kernel = np.ones(2 * half + 1) / float(2 * half + 1)
    extended = np.concatenate([points[-half:], points, points[:half]])
    return np.stack(
        [np.convolve(extended[:, axis], kernel, mode="valid") for axis in (0, 1)],
        axis=1,
    )


def _on_pixel_edges(contour: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """*contour* (the centres of a region's boundary pixels, as OpenCV traces
    them) moved half a pixel outward, onto the edges the region shares with
    its neighbours.

    Traced through pixel centres, two adjacent regions' outlines run one
    pixel apart, one on each side of their common edge, and a zoomed picture
    shows every border as a double line. Moved out by half a pixel along the
    outline's normal, both land on the edge itself. The outward side of each
    outline (an outer boundary or a hole's) is the side where its points'
    neighbours lie off *mask*.
    """
    points = contour.astype(np.float64)
    if len(points) < 3:
        return points
    tangent = np.roll(points, -1, axis=0) - np.roll(points, 1, axis=0)
    normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
    length = np.hypot(normal[:, 0], normal[:, 1])
    normal = np.divide(normal, length[:, None], out=np.zeros_like(normal),
                       where=length[:, None] > 0)
    rows, cols = mask.shape
    probe = np.rint(points + normal).astype(np.int64)
    inside = mask[np.clip(probe[:, 1], 0, rows - 1), np.clip(probe[:, 0], 0, cols - 1)] > 0
    outward = 1.0 if inside.mean() <= 0.5 else -1.0
    return points + 0.5 * outward * normal


def region_contours(
    labels: np.ndarray,
    *,
    smooth_window: int = 9,
    min_area_px: float = 24.0,
) -> dict[int, list[np.ndarray]]:
    """Smoothed outlines per region id, as float ``(N, 2)`` x/y polygons.

    Includes hole boundaries (a region wrapping another is traced inside and
    out). Region id 0 is background and never traced. Components smaller than
    *min_area_px* are dropped — classification confetti, not anatomy.
    """
    from scipy import ndimage

    ids = np.union1d(np.array([0], dtype=labels.dtype), np.unique(labels))
    compact = np.searchsorted(ids, labels).astype(np.int32)
    boxes = ndimage.find_objects(compact)

    out: dict[int, list[np.ndarray]] = {}
    for index, box in enumerate(boxes):
        uid = int(ids[index + 1])
        if box is None or uid == 0:
            continue
        rows, cols = box
        # Pad by one so a region touching its own bounding box still traces a
        # closed contour; the offset below puts it back in image coordinates.
        sub = np.pad((labels[rows, cols] == uid).astype(np.uint8), 1)
        contours, _ = cv2.findContours(sub, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        offset = np.array([cols.start - 1, rows.start - 1], dtype=np.float64)
        polys = [
            _smooth_closed(_on_pixel_edges(contour.reshape(-1, 2), sub) + offset,
                           smooth_window)
            for contour in contours
            if cv2.contourArea(contour) >= min_area_px
        ]
        if polys:
            out[uid] = polys
    return out


def region_outlines(
    atlas: Any,
    position_mm: float,
    *,
    depth: int,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    smooth_window: int = 9,
    min_area_px: float = 24.0,
) -> list[np.ndarray]:
    """Polylines of every region at ontology *depth*
    (:func:`~langslice.core.atlas.levels.labels_at`), in atlas-native pixels."""
    labels = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    contours = region_contours(
        labels_at(labels, atlas, depth), smooth_window=smooth_window, min_area_px=min_area_px
    )
    return [poly for _uid, polys in sorted(contours.items()) for poly in polys]
