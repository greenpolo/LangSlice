"""The physical canvas: a section and its atlas section at true scale, drawn.

:func:`canvas_geometry` places an atlas section on a section's frame at true
physical scale; :func:`physical_views` draws the alignment picture on it in
one of :data:`VIEW_MODES` and hands back, per picture, the
:class:`PanelFrame` its layers are computed from (:mod:`langslice.core.layers`).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageColor

from langslice.core.affine import (
    physical_affine_matrix,
    resize_long_edge,
)
from langslice.core.atlas.core import get_reference_slice
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.atlas.render import (
    HIGHLIGHT_WIDTH,
    annotation_slice,
    atlas_um_per_px,
    is_dark_background,
    region_contours,
    region_outlines,
)
from langslice.core.captions import _draw_scale_bar, caption
from langslice.core.space import Plane

#: Black working space on each side of the larger of section and atlas, as a
#: fraction of that extent. Room for x/y moves, like ABBA's viewer.
WORKING_MARGIN = 0.12

#: A zoom's source of section pixels (:func:`physical_views`' *detail*):
#: points of the section image (x and y arrays) -> ``(rgb, inside,
#: enlarged)``, or None.
SectionDetail = Callable[[np.ndarray, np.ndarray],
                         "tuple[np.ndarray, np.ndarray, float] | None"]


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
    margin: float = WORKING_MARGIN,
) -> CanvasGeometry:
    """Place an atlas section on a section's frame at TRUE physical scale.

    The atlas is scaled by ``atlas um/px / canvas um/px`` — never fitted to
    the canvas, which is not a calibration: fit-to-canvas can put the atlas
    plate at two thirds of the tissue's size where true scale matches it.
    Its ANATOMY (the annotation's bounding box, not the atlas frame with its
    empty margins) is centred on the canvas centre, and
    the canvas grows to hold whichever of the section
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

    body_w = max(section_size[0], extent[0] * scale)
    body_h = max(section_size[1], extent[1] * scale)
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


#: How the ONE alignment screen may be composed: the section under its
#: placement with the atlas over it (``overlay``, the default), the placed
#: section alone (``section``) or the atlas image alone (``template``).
VIEW_MODES = ("overlay", "section", "template")

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


#: ``border_color`` sentinel: resolved per picture by :func:`resolve_border_color`
#: rather than a fixed colour (the default everywhere a picture is drawn).
AUTO_BORDER_COLOR = "auto"

#: What :func:`pick_border_color` chooses from, in tie-break order (first
#: wins a near-tie). Excludes every colour a picture's OTHER drawing already
#: gives a fixed meaning: cyan #00ffff and orange #ffaa00 (landmark markers,
#: fresh/stale), white and black (marker rings, label chips, fold/hold
#: outlines), magenta in both its shades here (FOLD_COLOUR #ff2bd6 in
#: core.landmarks and MARKED_COLOR/HATCH_COLOR (255,64,160) in
#: core.deformable.render/core.damage, the "second ink" for excluded
#: regions), and the zoom grid's lime green #7dff3a (picked in
#: core.zoom_grid specifically as a colour nothing else uses). Green here is
#: a darker, more saturated green than that lime so the two read as
#: distinct colours rather than two shades of the same one.
BORDER_CANDIDATES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("yellow", (255, 255, 0)),
    ("green", (0, 170, 0)),
    ("red", (255, 0, 0)),
    ("blue", (0, 0, 255)),
)
#: CIELAB dE76 gap under which the automatic choice keeps yellow (the first
#: candidate) rather than a marginally higher-scoring one, so sections that
#: already read fine are not changed for a sliver of extra contrast.
BORDER_COLOR_TIE_EPSILON = 4.0

#: Caption words for colours this module can produce or that a caller might
#: pass explicitly; anything else is named by its hex.
_COLOR_WORDS: dict[tuple[int, int, int], str] = {
    rgb: name for name, rgb in BORDER_CANDIDATES
} | {
    (0, 255, 255): "cyan", (255, 0, 255): "magenta",
    (255, 255, 255): "white", (0, 0, 0): "black", (255, 165, 0): "orange",
}


def color_word(rgb: tuple[int, int, int]) -> str:
    """A caption word for a concrete border colour: a known name, else ``#rrggbb``."""
    key = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    if key in _COLOR_WORDS:
        return _COLOR_WORDS[key]
    return rgb_to_hex(key)


def rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    """A concrete RGB as ``#rrggbb``, to hand an already-resolved colour to a
    second call as an explicit ``border_color`` (so a multi-panel picture
    draws every panel in the one colour its first panel picked)."""
    return f"#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}"


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float64)


def pick_border_color(
    rgb: np.ndarray, coverage: np.ndarray,
) -> tuple[tuple[int, int, int], str]:
    """The :data:`BORDER_CANDIDATES` entry with the largest CIELAB dE76 against
    *rgb*'s own pixels under and around where the lines fall (*coverage* in
    [0, 1], same height/width as *rgb*; dilated a little here for "around").

    dE76 (Euclidean distance in CIELAB) is used rather than plain RGB
    distance or a WCAG luminance-only contrast ratio because it separates
    colours by perceived hue and chroma as well as lightness: a pale pink
    Nissl section and yellow can sit at similar luminance, where a
    luminance-only measure would call them high-contrast when they read as
    washed out. Ties within :data:`BORDER_COLOR_TIE_EPSILON` of the best
    score keep yellow.
    """
    if coverage.shape != rgb.shape[:2]:
        coverage = cv2.resize(coverage.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    weights = cv2.dilate(coverage.astype(np.float32), np.ones((5, 5), np.float32))
    total = float(weights.sum())
    default_name, default_rgb = BORDER_CANDIDATES[0]
    if total <= 0:
        return default_rgb, default_name
    lab = _lab(rgb[..., :3])
    sample = (lab * weights[..., None]).sum(axis=(0, 1)) / total
    scores = {
        name: float(np.linalg.norm(_lab(np.array([[candidate]]))[0, 0] - sample))
        for name, candidate in BORDER_CANDIDATES
    }
    best_name = max(scores, key=lambda name: scores[name])
    if scores[best_name] - scores[default_name] < BORDER_COLOR_TIE_EPSILON:
        return default_rgb, default_name
    return dict(BORDER_CANDIDATES)[best_name], best_name


def validate_border_thickness(thickness: float) -> float:
    """*thickness* in 0.25..8 output-image pixels, or a ``ValueError``."""
    if (
        isinstance(thickness, bool) or not isinstance(thickness, (int, float))
        or not math.isfinite(thickness) or not 0.25 <= thickness <= 8
    ):
        raise ValueError("border_thickness must be a number from 0.25 to 8 output pixels")
    return float(thickness)


def normalize_border_style(
    color: str = AUTO_BORDER_COLOR, thickness: float = 0.5,
) -> tuple[tuple[int, int, int], float]:
    """Validate an EXPLICIT display-only atlas border colour; width is in
    output-image pixels. :data:`AUTO_BORDER_COLOR` is not valid here (there
    are no pixels to judge contrast from) -- resolve it with
    :func:`resolve_border_color` instead."""
    if not isinstance(color, str):
        raise ValueError("border_color must be a named color or #RRGGBB")
    value = color.strip().lower()
    if value == AUTO_BORDER_COLOR:
        raise ValueError("border_color 'auto' must be resolved with resolve_border_color")
    if not re.fullmatch(r"[a-z]+|#[0-9a-f]{6}", value):
        raise ValueError("border_color must be a named color or #RRGGBB")
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError:
        raise ValueError("border_color must be a named color or #RRGGBB") from None
    if len(rgb) != 3:
        raise ValueError("border_color must be an opaque RGB color")
    return (rgb[0], rgb[1], rgb[2]), validate_border_thickness(thickness)


def resolve_border_color(
    color: str, thickness: float, *,
    rgb: np.ndarray | None = None, coverage: np.ndarray | None = None,
) -> tuple[tuple[int, int, int], float, str]:
    """*color* resolved to a concrete RGB, a validated thickness, and the
    caption word for it. :data:`AUTO_BORDER_COLOR` picks the highest-contrast
    candidate (:func:`pick_border_color`) from *rgb*/*coverage* -- the
    picture's own pixels and where its lines fall; any other value is
    validated as :func:`normalize_border_style` does. *rgb*/*coverage* may be
    omitted only when no line will actually be drawn (falls back to yellow,
    unused).
    """
    if not isinstance(color, str):
        raise ValueError("border_color must be a named color, #RRGGBB or 'auto'")
    if color.strip().lower() != AUTO_BORDER_COLOR:
        rgb_out, valid_thickness = normalize_border_style(color, thickness)
        return rgb_out, valid_thickness, color_word(rgb_out)
    valid_thickness = validate_border_thickness(thickness)
    if rgb is None or coverage is None or not coverage.any():
        name, rgb_out = BORDER_CANDIDATES[0]
        return rgb_out, valid_thickness, name
    rgb_out, name = pick_border_color(rgb, coverage)
    return rgb_out, valid_thickness, name


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
    as the region outlines. A name with a side (``"CTX:left"``) keeps only
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


def pivot_on_canvas(pivot: Any, geometry: CanvasGeometry) -> tuple[float, float] | None:
    """A pivot spec as CANVAS pixels, or ``None`` for the canvas centre.

    ``"canvas"`` (or empty) is the centre and ``[fx, fy]`` fractions of the
    canvas.

    Raises ``ValueError`` on anything else.
    """
    if pivot is None or (isinstance(pivot, str) and pivot.strip().lower() in ("", "canvas")):
        return None
    width, height = geometry.size
    if isinstance(pivot, str):
        raise ValueError(f"pivot must be 'canvas' or [fx, fy]; got {pivot!r}")
    values = [float(v) for v in pivot]
    if len(values) != 2:
        raise ValueError("pivot fractions must be [fx, fy] of the canvas")
    return (values[0] * width, values[1] * height)


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
    *content_box* holding the canvas crop above the caption, the *crop_box*
    on the canvas and *factor* (canvas px -> picture px, one for both axes,
    as the borders are drawn), the canvas *geometry*, the placement's
    *section_matrix* (3x3, the section render's frame -> canvas,
    :func:`placement_matrices`), and the atlas *lines* and *highlighted*
    region outlines drawn on it (native plane pixels, x/y; empty when none
    were drawn) at *line_width*, in *border_color* (resolved from "auto" if
    that is what was asked for). *from_file*: a zoom whose section pixels
    were read from the image file, how many picture pixels one file pixel
    spans (None: drawn from the render).
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
    from_file: float | None = None
    #: The concrete RGB any drawn line used (``resolve_border_color``'s
    #: choice); None when nothing was drawn, so "auto" had nothing to judge.
    border_color: tuple[int, int, int] | None = None


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
    zoom_pixels: Sequence[float] = (),
    atlas_opacity: float = 0.0,
    border_color: str = AUTO_BORDER_COLOR,
    border_thickness: float = 0.5,
    outlines: str = "all",
    depth: int = OVERLAY_DEPTH,
    pivot: tuple[float, float] | None = None,
    label: str = "",
    long_edge: int | None = None,
    panel_frames: list[PanelFrame] | None = None,
    pivot_in_section: tuple[float, float] | None = None,
    atlas_picture: Image.Image | None = None,
    atlas_name: str = "template",
    regions: Any = (),
    matrix_label: str = "fitted matrix",
    template_lines: bool = False,
    detail: SectionDetail | None = None,
) -> list[Image.Image]:
    """The alignment screen in one of :data:`VIEW_MODES`.

    Every view is built on the SAME millimetre-true canvas
    (:func:`canvas_geometry`) and the same crop, so a number read off one is
    the number on the others:

    * ``overlay`` — the warped section with the atlas region outlines on it,
      and the atlas image blended under them at *atlas_opacity*.
    * ``section`` — the warped section alone, no lines.
    * ``template`` — the atlas image alone; *template_lines* adds the
      *outlines* layer to it.

    *zoom* is ``[x0, y0, x1, y1]`` in fractions of the CANVAS (the tools take
    pixels and convert, :func:`langslice.core.display.zoom_fractions`); the crop happens
    before the screen is sized, so it is real magnification up to the canvas's
    own pixels: each panel's long edge is *long_edge*, or the crop's when that
    is smaller (never upsampled; ``None`` is canvas pixels one to one). A
    caller wanting a magnified zoom draws the canvas from a larger render.
    The scale bar is redrawn for the magnified micrometres per pixel.
    *zoom_pixels* is the zoom as the caller gave it (pixels of the unzoomed
    picture), for the caption.

    *outlines* picks which atlas lines are drawn: every boundary between
    regions at ontology *depth* (``all``, :func:`langslice.core.atlas.levels.labels_at`)
    or none (``none``). *border_color*
    is a named color, #RRGGBB, or :data:`AUTO_BORDER_COLOR` ("auto", the default):
    resolved per picture to whichever candidate (:data:`BORDER_CANDIDATES`) has
    the highest CIELAB contrast against the picture's own pixels under the lines
    (:func:`resolve_border_color`). *border_thickness* is 0.25..8
    output-image pixels (default 0.5), unchanged by zoom or canvas resolution.

    *pivot* is the rotation/scale centre in CANVAS pixels (``None`` is the
    canvas centre). *pivot_in_section* gives the pivot on the SECTION's frame
    instead and wins over *pivot*: a picture drawn from a larger render
    (:func:`shown_section`) knows its pivot only relative to the section.

    *atlas_picture* is the atlas image on the native plane grid (None: the
    reference template), named *atlas_name* in captions. *regions*
    (``[(name, ids)]``) are drawn at full strength and
    :data:`~langslice.core.atlas.render.HIGHLIGHT_WIDTH` times as thick in
    every mode, the *outlines* layer then at :data:`REGION_CONTEXT_ALPHA` for
    context.
    *matrix_label* names a ready matrix in the caption.

    *detail*, for a zoom of the ``overlay`` or ``section`` view, gives the
    section's pixels at points of *section* (pixels, centres at integers):
    the zoom shows them in place of *section*'s own, the atlas image and the
    lines over them as usual (:func:`langslice.core.placement.draw_canvas`
    passes the image file's own pixels, :mod:`langslice.core.native`). It
    answers ``(rgb, inside, enlarged)`` or None (the section's own pixels
    then).

    *panel_frames*, when given, receives one :class:`PanelFrame` per image:
    where its pixels sit, for the picture's layers. Returns the images, each
    captioned.
    """
    geometry = canvas_geometry(
        section.size,
        section_um_per_px,
        atlas,
        position_mm,
        plane,
        pitch_deg,
        yaw_deg,
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

    dark = is_dark_background(section)
    layer = str(outlines or "all").strip().lower()
    atlas_lines = (
        [] if layer == "none"
        else region_outlines(atlas, position_mm, depth=depth, plane=plane, pitch_deg=pitch_deg,
                             yaw_deg=yaw_deg)
    )

    highlighted: list[np.ndarray] = []
    sides_note = ""
    if regions:
        # A side is the SECTION's: carry the native plane onto the section
        # frame (native -> canvas is the atlas scale; section -> canvas the matrix).
        linear = _as_3x3(section_matrix)[:2, :2]
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

    lines = True  # atlas outlines drawn over the section...
    if mode not in VIEW_MODES:
        raise ValueError(f"mode must be one of {VIEW_MODES}; got {mode!r}")
    box = zoom_box(zoom, geometry.size)
    zoomed = box != (0, 0, geometry.size[0], geometry.size[1])
    # The section alone on the canvas, before an atlas image is blended under
    # it: what the file's pixels replace in a zoom.
    bare = warped.copy() if detail is not None and zoomed and mode != "template" else None
    opacity = float(np.clip(atlas_opacity, 0.0, 1.0))
    if mode == "section":
        panels = [(warped, label or "section")]
        lines = False  # ...except the clean views, which show one source alone
    elif mode == "template":
        panels = [(_template_canvas(), atlas_head)]
        lines = template_lines
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
                opacity=opacity,
                picture=atlas_picture,
            )
        panels = [(warped, label or "section")]

    # One colour for the whole picture: resolved once, against the one
    # background panel this call draws (never per zoom crop or per line
    # layer), before anything is drawn on it.
    drawn_polys = (list(atlas_lines) if lines else []) + list(highlighted)
    coverage = (
        line_coverage(
            panels[0][0].shape[:2], drawn_polys, thickness=max(float(border_thickness), 1.0),
            scale=geometry.atlas_scale, offset=geometry.atlas_offset,
        )
        if drawn_polys else None
    )
    line_color, line_width, border_word = resolve_border_color(
        border_color, border_thickness, rgb=np.asarray(panels[0][0]), coverage=coverage,
    )

    knobs = matrix_label if isinstance(params, np.ndarray) else knob_words(params)

    # The crop happens first, then the screen is sized: *long_edge*, or, for
    # the whole canvas, its own pixels when fewer (never upsampled). A zoom
    # is always drawn at *long_edge*: a small region comes out at the size of
    # any other picture, enlarged past the canvas's pixels when it has fewer
    # (said in the caption). No *long_edge* means canvas pixels one to one
    # (host-side use, never a model's screen).
    edge = None
    if long_edge is not None:
        edge = (max(1, int(long_edge)) if zoomed else
                max(1, min(int(long_edge), max(box[2] - box[0], box[3] - box[1]))))
    images: list[Image.Image] = []
    for panel, head in panels:
        screen, factor = _to_screen(panel, box, edge)
        from_file: float | None = None
        if bare is not None and detail is not None:
            found = _detailed_screen(bare, box, edge, matrix, geometry.section_offset, detail)
            if found is not None:
                screen, from_file = found
                if mode == "overlay" and opacity > 0.0:
                    _blend_screen(screen, _template_canvas(), box, edge, opacity)
        if lines:
            _draw_polys(
                screen, atlas_lines, line_color, thickness=line_width,
                scale=geometry.atlas_scale, offset=geometry.atlas_offset,
                origin=box[:2], factor=factor,
                alpha=REGION_CONTEXT_ALPHA if regions else 1.0,
            )
        if highlighted:
            _draw_polys(
                screen, highlighted, line_color, thickness=line_width * HIGHLIGHT_WIDTH,
                scale=geometry.atlas_scale, offset=geometry.atlas_offset,
                origin=box[:2], factor=factor,
            )
        _draw_scale_bar(screen, geometry.um_per_px / factor, dark)
        # Two lines by meaning; `caption` wraps any line wider than the panel.
        text = (
            f"{head} @ {position_mm:.3f} mm  pitch {pitch_deg:.2f} yaw {yaw_deg:.2f}\n"
            f"{knobs}  canvas {geometry.um_per_px:.2f} um/px"
        ).strip()
        if mode != "overlay" or zoomed or abs(float(factor) - 1.0) > 0.02:
            # `view` is the picture's own pixel; `canvas` the canvas's (they differ
            # whenever the picture is drawn at other than canvas pixels).
            where = (f"  zoom {[round(float(v)) for v in zoom_pixels]} px"
                     if zoom_pixels else "")
            text += f"\n{mode}{where}  view {geometry.um_per_px / factor:.2f} um/px"
            if from_file is not None:
                text += "  section from its image file" + (
                    f" (file pixels enlarged x{from_file:.1f})" if from_file > 1.05 else "")
            elif factor > 1.05:
                text += f" (canvas pixels enlarged x{factor:.1f})"
        if regions:
            text += "  regions " + ",".join(str(name) for name, _ids in regions) + sides_note
        labelled = caption(Image.fromarray(screen, mode="RGB"), text,
                           min_width=edge if zoomed and edge is not None else 0)
        if drawn_polys:
            # Carried on the image itself (as `pixel_to_native` etc. already
            # are), so any caller building a caption after this call can
            # name the colour without re-threading it through every layer.
            labelled.info["border_word"] = border_word
        if panel_frames is not None:
            panel_frames.append(PanelFrame(
                size=labelled.size,
                content_box=(0, 0, screen.shape[1], screen.shape[0]),
                crop_box=box, factor=float(factor), geometry=geometry,
                section_matrix=_shift(geometry.section_offset) @ _as_3x3(section_matrix),
                lines=tuple(atlas_lines) if lines else (),
                highlighted=tuple(highlighted), line_width=float(line_width), mode=mode,
                from_file=from_file, border_color=line_color if drawn_polys else None,
            ))
        images.append(labelled)
    return images


def knob_words(params: Any, sep: str = "  ") -> str:
    """An in-plane transform's knobs for a caption: ``rot 0.3  scale
    0.990/1.086  shift -0.14/+0.00 mm`` (and the shear when not 0), the
    parts joined by *sep*."""
    parts = [
        f"rot {float(params['rotation_deg']):.1f}",
        f"scale {float(params['scale_x']):.3f}/{float(params['scale_y']):.3f}",
        f"shift {float(params['translate_x_mm']):+.2f}/"
        f"{float(params['translate_y_mm']):+.2f} mm",
    ]
    if params.get("shear"):
        parts.append(f"shear {float(params['shear']):+.3f}")
    return sep.join(parts)


def _detailed_screen(
    bare: np.ndarray, box: tuple[int, int, int, int], edge: int | None, matrix: np.ndarray,
    offset: tuple[int, int], detail: SectionDetail,
) -> tuple[np.ndarray, float] | None:
    """The zoom of *bare* (the section alone on the canvas) with the
    section's pixels from *detail* wherever it has them, and how far they
    are enlarged; None when *detail* has none."""
    screen, _factor = _to_screen(bare, box, edge)
    rows, cols = screen.shape[:2]
    x0, y0, x1, y1 = box
    cx = x0 + (np.arange(cols, dtype=np.float64) + 0.5) * (x1 - x0) / cols - 0.5
    cy = y0 + (np.arange(rows, dtype=np.float64) + 0.5) * (y1 - y0) / rows - 0.5
    grid_x, grid_y = np.meshgrid(cx, cy)
    inverse = np.linalg.inv(_as_3x3(matrix))
    sx = inverse[0, 0] * grid_x + inverse[0, 1] * grid_y + inverse[0, 2] - offset[0]
    sy = inverse[1, 0] * grid_x + inverse[1, 1] * grid_y + inverse[1, 2] - offset[1]
    found = detail(sx, sy)
    if found is None:
        return None
    rgb, inside, enlarged = found
    screen[inside] = rgb[inside]
    return screen, float(enlarged)


def _blend_screen(
    screen: np.ndarray, plate: np.ndarray, box: tuple[int, int, int, int], edge: int | None,
    opacity: float,
) -> None:
    """The atlas image *plate* (zero where it has none) blended under
    *screen* at *opacity*, as :func:`_blend_template` blends it on the canvas."""
    shown, _factor = _to_screen(plate, box, edge)
    x0, y0, x1, y1 = box
    lit = (plate[y0:y1, x0:x1].max(axis=2) > 0).astype(np.uint8)
    lit = cv2.resize(lit, (screen.shape[1], screen.shape[0]),
                     interpolation=cv2.INTER_NEAREST) > 0
    screen[lit] = (screen[lit] * (1.0 - opacity) + shown[lit] * opacity).astype(np.uint8)
