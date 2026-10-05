"""The physical canvas: a section and its atlas section at true scale, drawn.

:func:`canvas_geometry` places an atlas section on a section's frame at true
physical scale; :func:`physical_views` draws the alignment picture on it in
one of :data:`VIEW_MODES` and hands back, per picture, the
:class:`PanelFrame` its layers are computed from (:mod:`langslice.core.layers`).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageColor

from langslice.core.affine import (
    extract_slice_silhouette,
    physical_affine_matrix,
    resize_long_edge,
    silhouette_iou,
)
from langslice.core.atlas.core import get_reference_slice
from langslice.core.atlas.render import (
    annotation_slice,
    atlas_um_per_px,
    family_outlines,
    is_dark_background,
    outer_outline,
    region_contours,
)
from langslice.core.captions import _draw_scale_bar, caption
from langslice.core.image_prep import foreground_mask
from langslice.core.space import Plane

#: Black working space on each side of the larger of section and atlas, as a
#: fraction of that extent. Room for x/y moves, like ABBA's viewer.
WORKING_MARGIN = 0.12

@dataclass(frozen=True)
class CanvasGeometry:
    """Where the section and the atlas sit on one millimetre-true canvas.

    The canvas IS the section's frame, grown symmetrically when the atlas
    anatomy at true scale would not fit inside it. Both frames therefore share
    a centre, which is why a transform's rotation centre is the same in either
    (:func:`langslice.core.affine.normalized_physical_affine`).
    """

    size: tuple[int, int]
    um_per_px: float
    #: Canvas coordinates of the section's top-left corner.
    section_offset: tuple[int, int]
    #: Atlas pixels -> canvas pixels.
    atlas_scale: float
    #: Canvas coordinates of the atlas frame's (0, 0).
    atlas_offset: tuple[float, float]
    annotation: np.ndarray

    def atlas_to_section_frame(self) -> np.ndarray:
        """3x3 from native atlas-plane pixels to the section's own frame (the
        canvas less the section's offset)."""
        sx, sy = self.section_offset
        ax, ay = self.atlas_offset
        return np.array([[self.atlas_scale, 0.0, ax - sx],
                         [0.0, self.atlas_scale, ay - sy], [0.0, 0.0, 1.0]], dtype=np.float64)


def canvas_geometry(
    section_size: tuple[int, int],
    section_um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    *,
    pad_to_fit_atlas: bool = True,
    margin: float = WORKING_MARGIN,
) -> CanvasGeometry:
    """Place an atlas section on a section's frame at TRUE physical scale.

    The atlas is scaled by ``atlas um/px / canvas um/px`` — never fitted to
    the canvas, which is not a calibration: fit-to-canvas can put the atlas
    plate at two thirds of the tissue's size where true scale matches it.
    Its ANATOMY (the annotation's bounding box, not the atlas frame with its
    empty margins) is centred on the canvas centre, and
    with *pad_to_fit_atlas* the canvas grows to hold whichever of the section
    and the atlas anatomy is larger, plus *margin* of black on every side —
    ABBA's viewer leaves generous black space around both, and the alignment
    loop needs it: the atlas runs out of frame otherwise and every x/y move
    has to fit inside the tissue's own bounding box.
    """
    ann = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    scale = atlas_um_per_px(atlas) / float(section_um_per_px)
    ys, xs = np.nonzero(ann)
    if ys.size:
        cx = float(xs.min() + xs.max() + 1) / 2.0
        cy = float(ys.min() + ys.max() + 1) / 2.0
        extent = (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
    else:
        cx, cy = ann.shape[1] / 2.0, ann.shape[0] / 2.0
        extent = (float(ann.shape[1]), float(ann.shape[0]))

    width, height = section_size
    if pad_to_fit_atlas:
        body_w = max(width, extent[0] * scale)
        body_h = max(height, extent[1] * scale)
        width = int(np.ceil(body_w * (1.0 + 2.0 * margin)))
        height = int(np.ceil(body_h * (1.0 + 2.0 * margin)))
    return CanvasGeometry(
        size=(width, height),
        um_per_px=float(section_um_per_px),
        section_offset=((width - section_size[0]) // 2, (height - section_size[1]) // 2),
        atlas_scale=scale,
        atlas_offset=(width / 2.0 - cx * scale, height / 2.0 - cy * scale),
        annotation=ann,
    )


#: How the ONE alignment screen may be composed. ``overlay`` is the default and
#: what every earlier run saw.
VIEW_MODES = ("overlay", "side_by_side", "checkerboard", "outlines", "section", "template")

#: Which atlas lines a view draws: every family boundary, the root silhouette
#: alone, or none at all.
OUTLINE_LAYERS = ("all", "outer", "none")

#: Tiles across the width of a ``checkerboard`` view.
CHECKER_TILES = 8

#: Landmark glyph colors (RGB): the section's point, its atlas point, the
#: connector between them. Colored on purpose — a landmark is a click, not
#: anatomy, and the atlas hairlines keep their one neutral grey.
MARKER_SECTION = (80, 220, 255)
MARKER_ATLAS = (255, 170, 60)
MARKER_CONNECTOR = (170, 170, 170)


def _background_color(section: Image.Image) -> tuple[int, int, int]:
    """The section's own border color, so padding does not read as anatomy."""
    arr = np.asarray(section.convert("RGB"), dtype=np.uint8)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    r, g, b = (int(v) for v in np.median(border, axis=0))
    return (r, g, b)


def _shift(offset: tuple[float, float]) -> np.ndarray:
    return np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]])


def _as_3x3(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])


def normalize_border_style(
    color: str = "yellow", thickness: float = 0.5,
) -> tuple[tuple[int, int, int], float]:
    """Validate display-only atlas borders; width is in output-image pixels."""
    if not isinstance(color, str):
        raise ValueError("border_color must be a named color or #RRGGBB")
    value = color.strip().lower()
    if not re.fullmatch(r"[a-z]+|#[0-9a-f]{6}", value):
        raise ValueError("border_color must be a named color or #RRGGBB")
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError:
        raise ValueError("border_color must be a named color or #RRGGBB") from None
    if len(rgb) != 3:
        raise ValueError("border_color must be an opaque RGB color")
    if (
        isinstance(thickness, bool) or not isinstance(thickness, (int, float))
        or not math.isfinite(thickness) or not 0.25 <= thickness <= 8
    ):
        raise ValueError("border_thickness must be a number from 0.25 to 8 output pixels")
    return (rgb[0], rgb[1], rgb[2]), float(thickness)


def _tissue_color(dark: bool) -> tuple[int, int, int]:
    """Neutral grey for the section's silhouette, distinct from atlas borders."""
    return (150, 150, 150) if dark else (140, 140, 140)


def _draw_polys(
    canvas: np.ndarray,
    polys: list[np.ndarray],
    color: tuple[int, int, int],
    *,
    thickness: float = 1,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """Closed x/y polylines with anti-aliased thickness at OUTPUT resolution.

    Each point runs through the same chain the pixels did: its own frame ->
    canvas (*scale*, *offset*), minus the zoom crop's *origin*, times *factor*,
    the canvas-px -> output-px ratio. Sub-pixel via OpenCV's 4-bit shift.
    *alpha* below 1 draws the lines faint (context under highlighted regions).
    """
    coverage = line_coverage(
        canvas.shape[:2], polys, thickness=thickness, scale=scale, offset=offset,
        origin=origin, factor=factor,
    )[..., None] * float(alpha)
    blended = canvas * (1.0 - coverage) + np.asarray(color) * coverage
    canvas[:] = np.rint(blended).astype(np.uint8)


def line_coverage(
    shape: tuple[int, ...],
    polys: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    thickness: float = 1,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
) -> np.ndarray:
    """How much of each output pixel the polylines cover, float32 in [0, 1].

    The one rasteriser of every atlas border a picture draws
    (:func:`_draw_polys` blends it in) and of the picture's border layer
    (:func:`langslice.core.layers.picture_layers`), so the two cannot
    disagree. *shape* is ``(rows, cols)``; the chain is :func:`_draw_polys`'s.
    """
    ox, oy = float(origin[0]), float(origin[1])
    rows, cols = int(shape[0]), int(shape[1])
    # Supersample all stroke widths consistently, keeping tissue at its
    # original resolution and compositing each border pixel only once.
    supersample = 8
    target = np.zeros((rows * supersample, cols * supersample), dtype=np.uint8)
    for poly in polys:
        points = np.round(
            ((poly * scale + np.asarray(offset, dtype=np.float64)) - (ox, oy))
            * factor * supersample * 16.0
        ).astype(np.int32)
        cv2.polylines(
            target, [points], True, 255,
            max(1, round(thickness * supersample)), cv2.LINE_AA, 4,
        )
    return cv2.resize(
        target, (cols, rows), interpolation=cv2.INTER_AREA,
    ).astype(np.float32) / 255.0


def _draw_outlines(
    canvas: np.ndarray,
    outlines: list[tuple[tuple[int, int, int], np.ndarray]],
    geometry: CanvasGeometry,
    *,
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: float = 1,
    factor: float = 1.0,
    origin: tuple[int, int] = (0, 0),
    alpha: float = 1.0,
) -> None:
    """One atlas-border color, with width set after crop and display resizing."""
    _draw_polys(
        canvas,
        [poly for _color, poly in outlines],
        color,
        thickness=thickness,
        scale=geometry.atlas_scale,
        offset=geometry.atlas_offset,
        origin=origin,
        factor=factor,
        alpha=alpha,
    )


#: Strength of the context outlines drawn under highlighted regions.
REGION_CONTEXT_ALPHA = 0.35


def regions_left(
    atlas: Any, regions: Any, position_mm: float, plane: str, pitch_deg: float,
    yaw_deg: float, native_to_display: np.ndarray,
) -> np.ndarray | None:
    """Native pixels on the picture's left when a region names a side, else None.

    *regions* is ``[(name, ids)]``; *native_to_display* the linear map from the
    native atlas plane to the frame whose left and right the sides name
    (:func:`langslice.core.atlas.sides.native_left`).
    """
    from langslice.core.atlas.sides import has_sides, native_left

    if not has_sides([name for name, _ids in regions]):
        return None
    return native_left(atlas, position_mm, plane, pitch_deg, yaw_deg, native_to_display)


def region_polys(
    annotation: np.ndarray, regions: Any, left: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Smoothed outlines of each highlighted region, in atlas-native pixels.

    *regions* is ``[(name, ids)]``; each region is traced as the union of its
    ids (the region and its descendants), with the same tracer and smoothing
    as the family outlines. A name with a side (``"CTX:left"``) keeps only
    that side, *left* being the native pixels on the left (:func:`regions_left`).
    """
    from langslice.core.atlas.sides import restrict, split_side

    polys: list[np.ndarray] = []
    for name, ids in regions:
        mask = restrict(np.isin(annotation, list(ids)), split_side(name)[1], left)
        if mask.any():
            polys.extend(region_contours(mask.astype(np.int32)).get(1, []))
    return polys


def _patch_window(
    shape: tuple[int, int], patch: np.ndarray, offset: tuple[float, float]
) -> tuple[int, int, np.ndarray]:
    """``(y, x, clipped patch)`` for pasting *patch* at *offset* on *shape*.

    Clips both ways: the canvas grows to hold the ANATOMY, so the atlas
    frame's empty margins may still hang over any edge. An empty patch comes
    back when nothing lands.
    """
    x0, y0 = (int(round(v)) for v in offset)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    height, width = shape
    return y0, x0, patch[sy0 : sy0 + height - y0, sx0 : sx0 + width - x0]


def _template_patch(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    geometry: CanvasGeometry,
    picture: Image.Image | None = None,
) -> tuple[int, int, np.ndarray]:
    """The atlas image resized to the canvas's scale, ready to paste.

    *picture* is the atlas image on the native plane grid; None is the atlas
    reference template.
    """
    template = picture if picture is not None else get_reference_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    rows, cols = geometry.annotation.shape[:2]
    if template.size != (cols, rows):
        template = template.resize((cols, rows), Image.Resampling.BILINEAR)
    scale = geometry.atlas_scale
    resized = template.convert("RGB").resize(
        (max(1, round(cols * scale)), max(1, round(rows * scale))),
        Image.Resampling.LANCZOS,
    )
    width, height = geometry.size
    return _patch_window(
        (height, width), np.asarray(resized, dtype=np.uint8), geometry.atlas_offset
    )


def _blend_template(
    canvas: np.ndarray,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    geometry: CanvasGeometry,
    opacity: float = 0.35,
    picture: Image.Image | None = None,
) -> None:
    """Blend the atlas image under the outlines, at the same placement."""
    y0, x0, patch = _template_patch(
        atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, picture
    )
    if patch.size == 0:
        return
    window = canvas[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
    lit = patch.max(axis=2) > 0
    window[lit] = (
        window[lit] * (1.0 - opacity) + patch[lit] * opacity
    ).astype(np.uint8)


def template_canvas(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    geometry: CanvasGeometry,
) -> np.ndarray:
    """The atlas template alone on a black canvas, RGB uint8, at the placement."""
    plate = np.zeros((geometry.size[1], geometry.size[0], 3), dtype=np.uint8)
    _blend_template(plate, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, opacity=1.0)
    return plate


def atlas_mask_canvas(geometry: CanvasGeometry) -> np.ndarray:
    """The atlas anatomy's silhouette on the canvas, as a uint8 0/255 mask.

    The same placement the outlines are drawn at, so an overlap measured
    against it is measured against the lines the model is looking at.
    """
    scale = geometry.atlas_scale
    rows, cols = geometry.annotation.shape[:2]
    resized = cv2.resize(
        (geometry.annotation > 0).astype(np.uint8) * 255,
        (max(1, round(cols * scale)), max(1, round(rows * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    width, height = geometry.size
    mask = np.zeros((height, width), dtype=np.uint8)
    y0, x0, patch = _patch_window((height, width), resized, geometry.atlas_offset)
    if patch.size:
        mask[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] = patch
    return mask


def zoom_box(zoom: list[float] | None, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """``[x0, y0, x1, y1]`` fractions of the canvas as a pixel crop box.

    Anything that is not four numbers — an empty list, the default — is the
    whole canvas. The box is ordered, clamped to the canvas and never allowed
    to collapse below 8 px, so a mistyped fraction costs magnification, not a
    crash.
    """
    width, height = size
    if not zoom or len(zoom) != 4:
        return (0, 0, width, height)
    x0, x1 = sorted((float(zoom[0]), float(zoom[2])))
    y0, y1 = sorted((float(zoom[1]), float(zoom[3])))
    bx0 = int(np.clip(round(x0 * width), 0, width - 8))
    by0 = int(np.clip(round(y0 * height), 0, height - 8))
    bx1 = int(np.clip(round(x1 * width), bx0 + 8, width))
    by1 = int(np.clip(round(y1 * height), by0 + 8, height))
    return (bx0, by0, bx1, by1)


def _to_screen(
    canvas: np.ndarray, box: tuple[int, int, int, int], long_edge: int | None
) -> tuple[np.ndarray, float]:
    """Crop to *box* then resize, returning ``(rgb, canvas px -> output px)``.

    Cropping BEFORE the resize is what makes a zoom real magnification: the
    same output budget spent on fewer canvas pixels. Lines and the bar
    are drawn after this, at output resolution.
    """
    x0, y0, x1, y1 = box
    cropped = canvas[y0:y1, x0:x1]
    image = Image.fromarray(cropped, mode="RGB")
    factor = 1.0
    if long_edge is not None:
        image = resize_long_edge(image, long_edge)
        factor = image.width / float(cropped.shape[1])
    return np.asarray(image, dtype=np.uint8).copy(), factor


def _checkerboard(a: np.ndarray, b: np.ndarray, tiles: int = CHECKER_TILES) -> np.ndarray:
    """*a* and *b* in alternating tiles, *tiles* of them across the width."""
    height, width = a.shape[:2]
    size = max(1, int(np.ceil(width / float(tiles))))
    ys, xs = np.mgrid[0:height, 0:width]
    even = ((xs // size + ys // size) % 2 == 0)[..., None]
    return np.where(even, a, b)


def pivot_on_canvas(
    pivot: Any, section: Image.Image, geometry: CanvasGeometry
) -> tuple[float, float] | None:
    """A pivot spec as CANVAS pixels, or ``None`` for the canvas centre.

    ``"canvas"`` (or empty) is the centre, ``"tissue"`` the section's own
    tissue centroid placed on the canvas, and ``[fx, fy]`` fractions of the
    canvas. A section whose tissue cannot be detected falls back to the
    centre; the caller reports the pivot it actually got.

    Raises ``ValueError`` on anything else.
    """
    if pivot is None or (isinstance(pivot, str) and pivot.strip().lower() in ("", "canvas")):
        return None
    width, height = geometry.size
    if isinstance(pivot, str):
        if pivot.strip().lower() != "tissue":
            raise ValueError(f"pivot must be 'canvas', 'tissue' or [fx, fy]; got {pivot!r}")
        mask = foreground_mask(section)
        if mask is None or not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        # The mask is measured on a proxy; scale its centroid onto the section,
        # then place the section on the canvas.
        return (
            float(xs.mean()) * section.width / mask.shape[1] + geometry.section_offset[0],
            float(ys.mean()) * section.height / mask.shape[0] + geometry.section_offset[1],
        )
    values = [float(v) for v in pivot]
    if len(values) != 2:
        raise ValueError("pivot fractions must be [fx, fy] of the canvas")
    return (values[0] * width, values[1] * height)


def _draw_markers(
    canvas: np.ndarray,
    section_pts: np.ndarray,
    atlas_pts: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
) -> None:
    """Landmark pairs: a cross on each section point, a ring on its atlas point.

    Two glyphs and two colors rather than two greys — these are the model's
    own clicks, not anatomy, and they have to be findable against both the
    tissue and the hairlines. A connector runs between the pair, and the pair's
    1-based index is written by the cross so a glyph can be tied to the row in
    the payload.
    """
    def screen(points: np.ndarray) -> np.ndarray:
        return np.round(
            (np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64))
            * factor
        ).astype(np.int32)

    # Glyphs are drawn thicker than the atlas hairline on purpose: at the same
    # weight they read as another contour instead of as a marker.
    radius = max(6, round(min(canvas.shape[:2]) / 55))
    weight = max(1, round(min(canvas.shape[:2]) / 350))
    for index, (start, end) in enumerate(
        zip(screen(section_pts), screen(atlas_pts), strict=True), start=1
    ):
        sx, sy = int(start[0]), int(start[1])
        ax, ay = int(end[0]), int(end[1])
        cv2.line(canvas, (sx, sy), (ax, ay), MARKER_CONNECTOR, 1, cv2.LINE_AA)
        cv2.line(
            canvas, (sx - radius, sy), (sx + radius, sy), MARKER_SECTION, weight, cv2.LINE_AA
        )
        cv2.line(
            canvas, (sx, sy - radius), (sx, sy + radius), MARKER_SECTION, weight, cv2.LINE_AA
        )
        cv2.circle(canvas, (ax, ay), radius, MARKER_ATLAS, weight, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (sx + radius + 3, sy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            MARKER_SECTION,
            weight,
            cv2.LINE_AA,
        )


def _silhouette_polys(mask: np.ndarray) -> list[np.ndarray]:
    """Closed contours of the section's own tissue silhouette, in canvas px.

    Traced by the same smoothed tracer the atlas lines come from: a raw
    findContours boundary on speckled fluorescence is a stipple, and a
    stippled line next to a smooth one reads as texture, not as an edge.
    """
    return region_contours(
        (mask > 0).astype(np.int32), smooth_window=9, min_area_px=64.0
    ).get(1, [])


def placement_matrices(
    section_size: tuple[int, int],
    um_per_px: float,
    section_offset: tuple[int, int],
    params: dict[str, float] | np.ndarray,
    *,
    pivot: tuple[float, float] | None = None,
    pivot_in_section: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(section matrix, canvas matrix)`` of one placement on its canvas.

    The section matrix maps the section's own frame (pixels of the render of
    *section_size*, x/y) onto itself, the knobs about the pivot or a ready
    2x3; the canvas matrix (2x3) is the same map with the section pasted at
    *section_offset* on the canvas. *pivot* is on the canvas,
    *pivot_in_section* on the section's frame (it wins). The one place the
    placement is built: :func:`physical_views` draws with it and the
    picture's frame (:class:`PanelFrame`) carries what it returned.
    """
    ox, oy = (float(v) for v in section_offset)
    if isinstance(params, np.ndarray):
        section_matrix = np.asarray(params, dtype=np.float64)
    else:
        section_matrix = physical_affine_matrix(
            size=section_size,
            um_per_px=um_per_px,
            # The pivot arrives on the canvas; the matrix is built on the
            # section's frame and conjugated onto the canvas below.
            pivot=pivot_in_section if pivot_in_section is not None
            else None if pivot is None else (pivot[0] - ox, pivot[1] - oy),
            **params,
        )
    matrix = (_shift((ox, oy)) @ _as_3x3(section_matrix) @ _shift((-ox, -oy)))[:2]
    return section_matrix, matrix


@dataclass(frozen=True)
class PanelFrame:
    """Where one drawn placement panel's pixels sit, kept as it was drawn.

    What the picture's layers (:mod:`langslice.core.layers`) are computed
    from: the captioned picture's *size* ``(width, height)``, the
    *content_box* holding the canvas crop below the caption, the *crop_box*
    on the canvas and *factor* (canvas px -> picture px, one for both axes,
    as the borders are drawn), the canvas *geometry*, the placement's
    *section_matrix* (3x3, the section render's frame -> canvas,
    :func:`placement_matrices`), and the atlas *lines* and *highlighted*
    region outlines drawn on it (native plane pixels, x/y; empty when none
    were drawn) at *line_width*.
    """

    size: tuple[int, int]
    content_box: tuple[int, int, int, int]
    crop_box: tuple[int, int, int, int]
    factor: float
    geometry: CanvasGeometry
    section_matrix: np.ndarray
    lines: tuple[np.ndarray, ...]
    highlighted: tuple[np.ndarray, ...]
    line_width: float
    mode: str


def physical_views(
    section: Image.Image,
    section_um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    params: dict[str, float] | np.ndarray,
    *,
    mode: str = "overlay",
    zoom: list[float] | None = None,
    atlas_opacity: float = 0.0,
    border_color: str = "yellow",
    border_thickness: float = 0.5,
    outlines: str = "all",
    pad_to_fit_atlas: bool = True,
    pivot: tuple[float, float] | None = None,
    markers: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "",
    long_edge: int | None = None,
    frames: list[dict[str, Any]] | None = None,
    panel_frames: list[PanelFrame] | None = None,
    pivot_in_section: tuple[float, float] | None = None,
    atlas_picture: Image.Image | None = None,
    atlas_name: str = "template",
    regions: Any = (),
    matrix_label: str = "fitted matrix",
    template_lines: bool = False,
    left: np.ndarray | None = None,
) -> tuple[list[Image.Image], float]:
    """The alignment screen in one of :data:`VIEW_MODES`, plus the overlap.

    One composition, several ways of showing it. Every view is built on the
    SAME millimetre-true canvas (:func:`canvas_geometry`) and the same crop, so
    a number read off one is the number on the others:

    * ``overlay`` — the warped section with the atlas family outlines on it,
      and the atlas image blended under them at *atlas_opacity*.
    * ``side_by_side`` — two images, the warped section and the atlas template,
      at the same micrometres per pixel and the same crop, outlines on both.
    * ``checkerboard`` — section and template in alternating tiles.
    * ``outlines`` — the atlas lines and the section's own silhouette contour
      in a second grey, on black. No pixels.

    *zoom* is ``[x0, y0, x1, y1]`` in fractions of the CANVAS; the crop happens
    before the screen is sized, so it is real magnification up to the canvas's
    own pixels: each panel's long edge is *long_edge*, or the crop's when that
    is smaller (never upsampled; ``None`` is canvas pixels one to one). A
    caller wanting a magnified zoom draws the canvas from a larger render.
    The scale bar is redrawn for the magnified micrometres per pixel.

    *outlines* picks which atlas lines are drawn (:data:`OUTLINE_LAYERS`):
    every family boundary, the root silhouette alone, or none. *border_color*
    is a named color or #RRGGBB (default yellow); *border_thickness* is 0.25..8
    output-image pixels (default 0.5), unchanged by zoom or canvas resolution.

    *pivot* is the rotation/scale centre in CANVAS pixels (``None`` is the
    canvas centre), and *markers* is ``(section points, atlas points)`` in
    canvas pixels, drawn on every panel as landmark pairs. *pivot_in_section*
    gives the pivot on the SECTION's frame instead and wins over *pivot*: a
    picture drawn from a larger render (:func:`shown_section`) knows its pivot
    only relative to the section.

    *atlas_picture* is the atlas image on the native plane grid (None: the
    reference template), named *atlas_name* in captions. *regions*
    (``[(name, ids)]``) are drawn at full strength in every mode, the
    *outlines* layer then at :data:`REGION_CONTEXT_ALPHA` for context.
    *matrix_label* names a ready matrix in the caption. ``template`` draws
    the atlas image alone; *template_lines* adds the *outlines* layer to it.
    *left* is the section's displayed left on the native plane as resolved
    elsewhere (a fit's own split, :class:`langslice.core.transform.FitFrame`);
    a one-sided region then uses it instead of resolving the sides through
    this placement, which has none once it turns the midline past 45 degrees.

    *panel_frames*, when given, receives one :class:`PanelFrame` per image:
    where its pixels sit, for the picture's layers.

    Returns ``(images, silhouette_iou)`` — every image captioned, and the
    overlap between the warped section's tissue mask and the atlas anatomy at
    this placement.
    """
    line_color, line_width = normalize_border_style(border_color, border_thickness)
    geometry = canvas_geometry(
        section.size,
        section_um_per_px,
        atlas,
        position_mm,
        plane,
        pitch_deg,
        yaw_deg,
        pad_to_fit_atlas=pad_to_fit_atlas,
    )
    fill = _background_color(section)
    canvas = Image.new("RGB", geometry.size, fill)
    canvas.paste(section.convert("RGB"), geometry.section_offset)

    section_matrix, matrix = placement_matrices(
        section.size, geometry.um_per_px, geometry.section_offset, params,
        pivot=pivot, pivot_in_section=pivot_in_section,
    )
    warped = cv2.warpAffine(
        np.asarray(canvas, dtype=np.uint8),
        matrix,
        geometry.size,
        flags=cv2.INTER_LINEAR,
        borderValue=fill,
    )

    tissue = extract_slice_silhouette(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY))
    iou = silhouette_iou(tissue, atlas_mask_canvas(geometry))

    dark = is_dark_background(section)
    layer = str(outlines or "all").strip().lower()
    draw = outer_outline if layer == "outer" else family_outlines
    atlas_lines = (
        []
        if layer == "none"
        else draw(atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
    )

    highlighted: list[np.ndarray] = []
    sides_note = ""
    if regions:
        # A side is the SECTION's: carry the native plane onto the section
        # frame (native -> canvas is the atlas scale; section -> canvas the matrix).
        linear = _as_3x3(section_matrix)[:2, :2]
        if left is not None and left.shape == geometry.annotation.shape[:2]:
            from langslice.core.atlas.sides import has_sides

            sides = left if has_sides([name for name, _ids in regions]) else None
        else:
            sides = regions_left(atlas, regions, position_mm, plane, pitch_deg, yaw_deg,
                                 np.linalg.inv(linear) * geometry.atlas_scale)
        highlighted = region_polys(geometry.annotation, regions, sides)
        if sides is not None and np.linalg.det(linear) < 0:
            sides_note = " (the section's sides; this placement mirrors it)"
    atlas_head = f"atlas {atlas_name}"

    def _template_canvas() -> np.ndarray:
        plate = np.zeros_like(warped)
        _blend_template(
            plate, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, opacity=1.0,
            picture=atlas_picture,
        )
        return plate

    silhouette: list[np.ndarray] = []
    lines = True  # atlas outlines drawn on every panel...
    if mode == "section":
        panels = [(warped, label or "section")]
        lines = False  # ...except the clean views, which show one source alone
    elif mode == "template":
        panels = [(_template_canvas(), atlas_head)]
        lines = template_lines
    elif mode == "side_by_side":
        panels = [(warped, label or "section"), (_template_canvas(), atlas_head)]
    elif mode == "checkerboard":
        panels = [(_checkerboard(warped, _template_canvas()), label or "section")]
    elif mode == "outlines":
        silhouette = _silhouette_polys(tissue)
        black = np.zeros_like(warped)
        if atlas_opacity > 0.0:  # an atlas image the call listed, on the black
            _blend_template(
                black, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry,
                opacity=float(np.clip(atlas_opacity, 0.0, 1.0)), picture=atlas_picture,
            )
        panels = [(black, label or "section")]
        dark = True  # the canvas is black whatever the section is
    else:
        if atlas_opacity > 0.0:
            _blend_template(
                warped,
                atlas,
                position_mm,
                plane,
                pitch_deg,
                yaw_deg,
                geometry,
                opacity=float(np.clip(atlas_opacity, 0.0, 1.0)),
                picture=atlas_picture,
            )
        panels = [(warped, label or "section")]

    box = zoom_box(zoom, geometry.size)
    if isinstance(params, np.ndarray):
        knobs = matrix_label
    else:
        knobs = (
            f"rot {params['rotation_deg']:.1f}  "
            f"scale {params['scale_x']:.3f}/{params['scale_y']:.3f}  "
            f"shift {params['translate_x_mm']:+.2f}/{params['translate_y_mm']:+.2f} mm"
            + (f"  shear {params['shear']:+.3f}" if params.get("shear") else "")
        )

    # The crop happens first, then the screen is sized: *long_edge*, or the
    # crop's own pixels when fewer (never upsampled). No *long_edge* means
    # canvas pixels one to one (host-side use, never a model's screen).
    edge = None
    if long_edge is not None:
        edge = max(1, min(int(long_edge), max(box[2] - box[0], box[3] - box[1])))
    images: list[Image.Image] = []
    for panel, head in panels:
        screen, factor = _to_screen(panel, box, edge)
        if lines:
            _draw_outlines(
                screen, atlas_lines, geometry, color=line_color, thickness=line_width,
                factor=factor, origin=box[:2],
                alpha=REGION_CONTEXT_ALPHA if regions else 1.0,
            )
        if highlighted:
            _draw_polys(
                screen, highlighted, line_color, thickness=line_width,
                scale=geometry.atlas_scale, offset=geometry.atlas_offset,
                origin=box[:2], factor=factor,
            )
        if silhouette:
            _draw_polys(
                screen,
                silhouette,
                _tissue_color(dark),
                origin=box[:2],
                factor=factor,
            )
        if markers is not None and len(markers[0]):
            _draw_markers(screen, markers[0], markers[1], origin=box[:2], factor=factor)
        _draw_scale_bar(screen, geometry.um_per_px / factor, dark)
        # Two lines by meaning; `caption` wraps any line wider than the panel.
        text = (
            f"{head} @ {position_mm:.3f} mm  pitch {pitch_deg:.2f} yaw {yaw_deg:.2f}\n"
            f"{knobs}  canvas {geometry.um_per_px:.2f} um/px"
        ).strip()
        if mode != "overlay" or box != (0, 0, geometry.size[0], geometry.size[1]):
            zoomed = [
                round(box[0] / geometry.size[0], 3),
                round(box[1] / geometry.size[1], 3),
                round(box[2] / geometry.size[0], 3),
                round(box[3] / geometry.size[1], 3),
            ]
            text += (
                f"\n{mode}  zoom {zoomed}  "
                f"view {geometry.um_per_px / factor:.2f} um/px"
            )
        if layer == "outer" and lines:
            text += "  outlines outer"
        if regions:
            text += "  regions " + ",".join(str(name) for name, _ids in regions) + sides_note
        labelled = caption(Image.fromarray(screen, mode="RGB"), text)
        if panel_frames is not None:
            panel_frames.append(PanelFrame(
                size=labelled.size,
                content_box=(0, labelled.height - screen.shape[0], labelled.width,
                             labelled.height),
                crop_box=box, factor=float(factor), geometry=geometry,
                section_matrix=_shift(geometry.section_offset) @ _as_3x3(section_matrix),
                lines=tuple(poly for _color, poly in atlas_lines) if lines else (),
                highlighted=tuple(highlighted), line_width=float(line_width), mode=mode,
            ))
        if frames is not None:
            frames.append({
                "width": labelled.width, "height": labelled.height,
                "content_box": [0, labelled.height - screen.shape[0],
                                labelled.width, labelled.height],
                "canvas_box": list(box), "section_offset": list(geometry.section_offset),
                "section_size": list(section.size), "um_per_px": geometry.um_per_px,
            })
        images.append(labelled)
    return images, iou


def physical_overlay(
    section: Image.Image,
    section_um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    params: dict[str, float] | np.ndarray,
    *,
    atlas_opacity: float = 0.0,
    border_color: str = "yellow",
    border_thickness: float = 0.5,
    pad_to_fit_atlas: bool = True,
    pivot: tuple[float, float] | None = None,
    label: str = "",
    long_edge: int | None = None,
) -> Image.Image:
    """The ONE screen of the alignment loop: section and atlas in millimetres.

    *section* is the display render, *section_um_per_px* the micrometres one
    of its pixels covers. The atlas is drawn at true physical scale on that
    canvas (:func:`canvas_geometry`), as one smoothed outline per FAMILY
    region; leaf boundaries are visual noise at this size and are not drawn.

    *params* is either the five physical knobs (``rotation_deg``,
    ``scale_x``, ``scale_y``, ``translate_x_mm``, ``translate_y_mm``) or a
    ready 2x3 matrix in the SECTION's frame — what a closed-form fit hands
    back — so the interactive loop and ``fit_affine`` draw the same picture.
    The ``overlay`` view of :func:`physical_views`, which is where the other
    views live.
    """
    images, _iou = physical_views(
        section,
        section_um_per_px,
        atlas,
        position_mm,
        plane,
        pitch_deg,
        yaw_deg,
        params,
        atlas_opacity=atlas_opacity,
        border_color=border_color,
        border_thickness=border_thickness,
        pad_to_fit_atlas=pad_to_fit_atlas,
        pivot=pivot,
        label=label,
        long_edge=long_edge,
    )
    return images[0]


def estimate_um_per_px(
    section: Image.Image,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> float | None:
    """Micrometres per pixel guessed from the tissue's width, or None.

    The fallback when no file and no host says: the section's tissue is
    assumed to span the same millimetres as the atlas anatomy at this
    position, which is a SHAPE fit, not a calibration — every caller that
    uses it reports the calibration as "estimated".
    """
    mask = foreground_mask(section)
    if mask is None:
        return None
    xs = np.nonzero(mask.any(axis=0))[0]
    if xs.size == 0:
        return None
    # The mask is measured on a proxy; scale its width back onto the section.
    tissue_px = (xs.max() - xs.min() + 1) * section.width / float(mask.shape[1])
    ann = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    columns = np.nonzero(ann.any(axis=0))[0]
    if columns.size == 0 or tissue_px < 1:
        return None
    atlas_um = (columns.max() - columns.min() + 1) * atlas_um_per_px(atlas)
    return float(atlas_um / tissue_px)
