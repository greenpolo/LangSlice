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

from langslice.atlas.core import orient_slice_for_display, position_mm_to_index
from langslice.atlas.recolor import MERGE_EPS, color_lut
from langslice.space import Plane, atlas_space_context, slice_axis_index

__all__ = [
    "BORDER_DARKEN",
    "annotation_slice",
    "atlas_um_per_px",
    "border_color",
    "darker",
    "family_mapping",
    "family_outlines",
    "is_dark_background",
    "outer_outline",
    "region_contours",
]

Rgb = tuple[int, int, int]


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
    taken flat off the voxel grid. Measured on the LSD_910 hand
    registrations, whose block was cut at 4 degrees: matching the plane is
    worth far more than any fit tuning (fit-only family dice 0.93 -> 0.96,
    boundary p95 34px -> 9px over 33 slices).
    """
    if pitch_deg or yaw_deg:
        from langslice.oblique import sample_oblique_annotation

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


#: Allen-Reference-Atlas plate look: a region's delineating line is its own
#: color at this fraction of its brightness — same hue and saturation, only
#: value moves, so the line reads as "the edge of THIS region" rather than as
#: a separate structure. (Scaling all three channels scales HSV value exactly.)
BORDER_DARKEN = 0.7


def darker(color: Rgb) -> Rgb:
    """*color* dimmed to :data:`BORDER_DARKEN` of its brightness."""
    return (
        int(color[0] * BORDER_DARKEN),
        int(color[1] * BORDER_DARKEN),
        int(color[2] * BORDER_DARKEN),
    )


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
            _smooth_closed(contour.reshape(-1, 2).astype(np.float64) + offset, smooth_window)
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
    merges at :data:`~langslice.atlas.recolor.MERGE_EPS`. Leaf boundaries are
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
