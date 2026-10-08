"""Atlas geometry both methods draw from: annotation slices, contours, colors.

Shared on purpose. ``linear`` draws family outlines over a section in physical
space and ``nonlinear`` draws the same lines over its generated maps; the lines
have to come from the same source, or the two methods disagree about where a
boundary is.

Nothing here loads a model, writes a file, or knows about a run — an atlas, a
position and (optionally) the block's cutting angles go in, geometry comes out.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from langslice.core.atlas.core import orient_slice_for_display, position_mm_to_index
from langslice.core.atlas.recolor import MERGE_EPS, color_lut
from langslice.core.space import Plane, atlas_space_context, slice_axis_index

__all__ = [
    "annotation_slice",
    "atlas_um_per_px",
    "border_color",
    "family_labels",
    "family_mapping",
    "family_outlines",
    "is_dark_background",
    "outer_outline",
    "placed_border_coverage",
    "region_contours",
]

Rgb = tuple[int, int, int]


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
    idx = position_mm_to_index(atlas, position_mm, plane=plane)
    axis = slice_axis_index(atlas_space_context(atlas), plane)
    return orient_slice_for_display(
        np.asarray(np.take(atlas.annotation, idx, axis=axis)), plane
    )


def atlas_um_per_px(atlas: Any) -> float:
    """Micrometres per pixel of an atlas render at native resolution."""
    return float(max(atlas.resolution))


def family_mapping(
    uids: Any, atlas: Any, merge_eps: float = MERGE_EPS
) -> dict[int, int]:
    """region id -> representative id of its merged color family.

    Deterministic in the id set it is given: pass the ids of every map that
    has to share one vocabulary (e.g. a generated map and the atlas render),
    or the two get different representatives for the same family.
    """
    lut = color_lut(atlas)
    reps: list[tuple[tuple[int, int, int], int]] = []
    mapping: dict[int, int] = {}
    for uid in sorted(int(u) for u in uids if int(u) != 0):
        color = lut.get(uid, (128, 128, 128))
        for rep_color, rep_id in reps:
            if sum((a - b) ** 2 for a, b in zip(color, rep_color, strict=False)) <= merge_eps**2:
                mapping[uid] = rep_id
                break
        else:
            reps.append((color, uid))
            mapping[uid] = uid
    return mapping


def family_labels(labels: np.ndarray, atlas: Any, merge_eps: float = MERGE_EPS) -> np.ndarray:
    """*labels* with every region replaced by its color family's representative.

    The merged set the image model is shown: the image tool's placed borders
    are drawn from exactly this map (``nonlinear``'s ``_merge_classified`` is
    the same mapping), so a fit against the model's lines must use it too.
    """
    merged = np.zeros_like(labels)
    for uid, rep_id in family_mapping(np.unique(labels), atlas, merge_eps).items():
        merged[labels == uid] = rep_id
    return merged


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
    each region's indicator is blurred by *smoothing_px* atlas pixels, warped
    bilinearly onto a *supersample*-times finer grid, and every fine pixel
    takes the region that covers it most. Boundaries between neighbouring
    fine pixels are one shared line (tracing each region separately draws a
    shared edge twice, one atlas pixel apart), widened to *width_px* image
    pixels and area-averaged back down, so the edges are antialiased.
    """
    width, height = size
    s = int(supersample)
    fine = np.array([[s, 0, (s - 1) / 2], [0, s, (s - 1) / 2], [0, 0, 1]], dtype=np.float64)
    matrix = np.asarray(atlas_to_image, dtype=np.float64)
    if matrix.shape == (2, 3):
        matrix = np.vstack([matrix, [0.0, 0.0, 1.0]])
    to_fine = (fine @ matrix)[:2]
    fine_size = (width * s, height * s)
    best = np.full((height * s, width * s), -1.0, dtype=np.float32)
    owner = np.zeros((height * s, width * s), dtype=np.int32)
    for index, region in enumerate(np.unique(labels)):
        indicator = (labels == region).astype(np.float32)
        if smoothing_px > 0:
            indicator = cv2.GaussianBlur(indicator, (0, 0), smoothing_px)
        cover = cv2.warpAffine(
            indicator, to_fine, fine_size, flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=float(region == 0),
        )
        wins = cover > best
        best[wins] = cover[wins]
        owner[wins] = index
    edges = np.zeros(owner.shape, dtype=np.uint8)
    edges[:, 1:] |= (owner[:, 1:] != owner[:, :-1]).astype(np.uint8)
    edges[1:, :] |= (owner[1:, :] != owner[:-1, :]).astype(np.uint8)
    radius = max(0.5, width_px * s / 2.0)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * int(np.ceil(radius)) + 1,) * 2,
    )
    stroke = cv2.dilate(edges * 255, kernel)
    coverage = cv2.resize(stroke, (width, height), interpolation=cv2.INTER_AREA)
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


def border_color(color: Rgb, *, dark_background: bool = False) -> Rgb:
    """A saturated, background-contrasting version of a region's own color.

    Keeping the hue is the whole point: a border tells you which structure it
    bounds. Only saturation and value move, and value moves toward whichever
    end the canvas is not.
    """
    hsv = cv2.cvtColor(np.array([[color]], dtype=np.uint8), cv2.COLOR_RGB2HSV)
    hue, sat, val = (int(v) for v in hsv[0, 0])
    # Gray regions (fiber tracts, root) stay gray: pushing saturation on an
    # achromatic color invents a hue out of rounding noise.
    sat = 0 if sat < 12 else min(255, int(sat * 1.9) + 40)
    val = min(255, int(val * 0.35) + 170) if dark_background else int(val * 0.52)
    out = cv2.cvtColor(np.array([[(hue, sat, val)]], dtype=np.uint8), cv2.COLOR_HSV2RGB)
    r, g, b = (int(v) for v in out[0, 0])
    return (r, g, b)


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


def outer_outline(
    atlas: Any,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    smooth_window: int = 9,
    min_area_px: float = 24.0,
) -> list[tuple[Rgb, np.ndarray]]:
    """``(color, polyline)`` for the ROOT silhouette alone, in atlas pixels.

    Same shape as :func:`family_outlines`, one layer down: the boundary of
    everything the annotation labels, with no internal region lines. What an
    overlay wants when the family lines read as busy.
    """
    labels = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    contours = region_contours(
        (labels > 0).astype(np.int32),
        smooth_window=smooth_window,
        min_area_px=min_area_px,
    )
    return [((128, 128, 128), poly) for poly in contours.get(1, [])]


def family_outlines(
    atlas: Any,
    position_mm: float,
    *,
    plane: Plane = "coronal",
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    smooth_window: int = 9,
    min_area_px: float = 24.0,
) -> list[tuple[Rgb, np.ndarray]]:
    """``(family color, polyline)`` per family region, in atlas-native pixels.

    Family level, not leaf level: the registration units the color LUT already
    merges at :data:`~langslice.core.atlas.recolor.MERGE_EPS`. Leaf boundaries are
    a dozen near-identical lines through one cortex and read as noise on an
    overlay.
    """
    labels = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    families = np.zeros_like(labels)
    for uid, rep_id in family_mapping(np.unique(labels), atlas).items():
        families[labels == uid] = rep_id
    lut = color_lut(atlas)
    contours = region_contours(
        families, smooth_window=smooth_window, min_area_px=min_area_px
    )
    return [
        (lut.get(uid, (128, 128, 128)), poly)
        for uid, polys in sorted(contours.items())
        for poly in polys
    ]
