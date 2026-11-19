"""The physical canvas: a section and its atlas section at true scale, drawn.

:func:`canvas_geometry` places an atlas section on a section's frame at true
physical scale; :func:`physical_layers` draws the alignment picture's layers on
it in one of :data:`VIEW_MODES` (:class:`PlacementLayers`: the section alone,
the atlas borders' coverage, the :class:`PanelFrame` the picture's layers
are computed from in :mod:`langslice.core.pictures.layers`, the scale bar),
and :func:`physical_view` composes the finished, captioned picture from them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.atlas.borders import (
    HIGHLIGHT_WIDTH,
    annotation_slice,
    atlas_um_per_px,
    is_dark_background,
    region_contours,
    region_outlines,
)
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.atlas.load import get_reference_slice
from langslice.core.geometry.affine import (
    physical_affine_matrix,
    resize_long_edge,
)
from langslice.core.pictures.captions import ScaleBar, caption, draw_scale_bar, scale_bar
from langslice.core.space import Plane

WORKING_MARGIN = 0.12

SectionDetail = Callable[[np.ndarray, np.ndarray], "tuple[np.ndarray, np.ndarray, float] | None"]


@dataclass(frozen=True)
class CanvasGeometry:
    """Where the section and the atlas sit on one millimetre-true canvas.

    The canvas IS the section's frame, grown symmetrically when the atlas
    anatomy at true scale would not fit inside it. Both frames therefore share
    a centre, which is why a transform's rotation centre is the same in either
    (:func:`langslice.core.geometry.affine.normalized_physical_affine`).
    """

    size: tuple[int, int]
    um_per_px: float
    section_offset: tuple[int, int]
    atlas_scale: float
    atlas_offset: tuple[float, float]
    annotation: np.ndarray

    def atlas_to_section_frame(self) -> np.ndarray:
        """3x3 from native atlas-plane pixels to the section's own frame (the
        canvas less the section's offset)."""
        sx, sy = self.section_offset
        ax, ay = self.atlas_offset
        return np.array(
            [[self.atlas_scale, 0.0, ax - sx], [0.0, self.atlas_scale, ay - sy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


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
    ann = annotation_slice(atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
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


AUTO_BORDER_COLOR = "auto"

BORDER_CORES = ("black", "white")
_CORE_RGB = {"black": (0, 0, 0), "white": (255, 255, 255)}
CASING_PX = 1.0
CORE_LIGHTNESS_SPLIT = 50.0


def casing_of(core: str) -> str:
    """The outline colour of a core colour: the other of black and white."""
    return "white" if core == "black" else "black"


def border_words(core: str) -> str:
    """The caption words for a cased border: ``black, white-edged``."""
    return f"{core}, {casing_of(core)}-edged"


def _lab(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB).astype(
        np.float64
    )


def pick_border_core(rgb: np.ndarray, coverage: np.ndarray) -> str:
    """``black`` for a pale picture, ``white`` for a dark one.

    The mean CIELAB lightness (0-100) of *rgb*'s own pixels under and around
    where the lines fall (*coverage* in [0, 1], same height/width as *rgb*;
    dilated a little here for "around"), weighted by coverage: at or above
    :data:`CORE_LIGHTNESS_SPLIT` the core is black. No coverage: black, unused.
    """
    if coverage.shape != rgb.shape[:2]:
        coverage = cv2.resize(coverage.astype(np.float32), (rgb.shape[1], rgb.shape[0]))
    weights = cv2.dilate(coverage.astype(np.float32), np.ones((5, 5), np.float32))
    total = float(weights.sum())
    if total <= 0:
        return "black"
    lightness = _lab(rgb[..., :3])[..., 0] * (100.0 / 255.0)
    mean = float((lightness * weights).sum() / total)
    return "black" if mean >= CORE_LIGHTNESS_SPLIT else "white"


def validate_border_thickness(thickness: float) -> float:
    """*thickness* in 0.25..8 output-image pixels, or a ``ValueError``."""
    if (
        isinstance(thickness, bool)
        or not isinstance(thickness, (int, float))
        or not math.isfinite(thickness)
        or not 0.25 <= thickness <= 8
    ):
        raise ValueError("border_thickness must be a number from 0.25 to 8 output pixels")
    return float(thickness)


def resolve_border_color(
    color: str,
    thickness: float,
    *,
    rgb: np.ndarray | None = None,
    coverage: np.ndarray | None = None,
) -> tuple[str, float]:
    """``(core, thickness)``: the core colour (``black`` or ``white``) and a
    validated thickness. :data:`AUTO_BORDER_COLOR` picks the core from *rgb*
    and *coverage* (:func:`pick_border_core`); *rgb*/*coverage* may be omitted
    only when no line will be drawn."""
    if not isinstance(color, str):
        raise ValueError("border_color must be 'auto', 'black' or 'white'")
    value = color.strip().lower()
    valid_thickness = validate_border_thickness(thickness)
    if value == AUTO_BORDER_COLOR:
        if rgb is None or coverage is None or not coverage.any():
            return "black", valid_thickness
        return pick_border_core(rgb, coverage), valid_thickness
    if value not in BORDER_CORES:
        raise ValueError("border_color must be 'auto', 'black' or 'white'")
    return value, valid_thickness


def composite_cased(
    base: np.ndarray,
    core: str,
    *,
    casing: np.ndarray,
    line: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """*base* (float, rows x cols x 3) with a cased line laid on it: the
    casing coverage in the casing colour, then the core coverage in the core
    colour, both at *alpha*."""
    for coverage, ink in ((casing, casing_of(core)), (line, core)):
        weight = coverage[..., None] * float(alpha)
        base = base * (1.0 - weight) + np.asarray(_CORE_RGB[ink], dtype=np.float32) * weight
    return base


def cased_coverage(
    shape: tuple[int, ...],
    polys: list[np.ndarray] | tuple[np.ndarray, ...],
    *,
    thickness: float,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """``(line, casing)`` coverage of *polys*: the core line of *thickness*
    and the casing :data:`CASING_PX` wider on each side, both float32 in
    [0, 1] from :func:`line_coverage` (zeros without polylines)."""
    if not len(polys):
        empty = np.zeros(tuple(shape[:2]), dtype=np.float32)
        return empty, empty.copy()
    kwargs = {"scale": scale, "offset": offset, "origin": origin, "factor": factor}
    line = line_coverage(shape, list(polys), thickness=thickness, **kwargs)
    casing = line_coverage(shape, list(polys), thickness=thickness + 2 * CASING_PX, **kwargs)
    return line, casing


def lay_cased(
    canvas: np.ndarray,
    core: str,
    *,
    line: np.ndarray,
    casing: np.ndarray,
    alpha: float = 1.0,
) -> None:
    """The cased line (:func:`composite_cased`) laid on *canvas* (uint8, in
    place) from its coverage, *alpha* below 1 faint."""
    blended = composite_cased(
        canvas.astype(np.float32), core, casing=casing, line=line, alpha=alpha
    )
    canvas[:] = np.rint(blended).astype(np.uint8)


def draw_polys(
    canvas: np.ndarray,
    polys: list[np.ndarray],
    core: str,
    *,
    thickness: float = 1,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """Closed x/y polylines, cased, with anti-aliased thickness at OUTPUT resolution.

    Each point runs through the same chain the pixels did: its own frame ->
    canvas (*scale*, *offset*), minus the zoom crop's *origin*, times *factor*,
    the canvas-px -> output-px ratio. Sub-pixel via OpenCV's 4-bit shift.
    The line is the *core* colour (``black`` or ``white``) of *thickness*, over
    a casing of the other colour :data:`CASING_PX` wider on each side, both
    from :func:`cased_coverage`. *alpha* below 1 draws core and casing faint
    (context under highlighted regions).
    """
    line, casing = cased_coverage(
        canvas.shape[:2],
        polys,
        thickness=thickness,
        scale=scale,
        offset=offset,
        origin=origin,
        factor=factor,
    )
    lay_cased(canvas, core, line=line, casing=casing, alpha=alpha)


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
    (:func:`draw_polys` blends it in) and of the picture's border layer
    (:func:`langslice.core.pictures.layers.picture_layers`), so the two cannot
    disagree. *shape* is ``(rows, cols)``; the chain is :func:`draw_polys`'s.
    """
    ox, oy = float(origin[0]), float(origin[1])
    rows, cols = int(shape[0]), int(shape[1])
    supersample = 8
    target = np.zeros((rows * supersample, cols * supersample), dtype=np.uint8)
    for poly in polys:
        points = np.round(
            ((poly * scale + np.asarray(offset, dtype=np.float64)) - (ox, oy))
            * factor
            * supersample
            * 16.0
        ).astype(np.int32)
        cv2.polylines(
            target,
            [points],
            True,
            255,
            max(1, round(thickness * supersample)),
            cv2.LINE_AA,
            4,
        )
    return (
        cv2.resize(
            target,
            (cols, rows),
            interpolation=cv2.INTER_AREA,
        ).astype(np.float32)
        / 255.0
    )


REGION_CONTEXT_ALPHA = 0.35


def regions_left(
    atlas: Any,
    regions: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    native_to_display: np.ndarray,
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
    annotation: np.ndarray,
    regions: Any,
    left: np.ndarray | None = None,
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
    template = (
        picture
        if picture is not None
        else get_reference_slice(
            atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
        )
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
    opacity: float,
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
    window[lit] = (window[lit] * (1.0 - opacity) + patch[lit] * opacity).astype(np.uint8)


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
    placement is built: draw with it and the
    picture's frame (:class:`PanelFrame`) carries what it returned.
    """
    ox, oy = (float(v) for v in section_offset)
    if isinstance(params, np.ndarray):
        section_matrix = np.asarray(params, dtype=np.float64)
    else:
        section_matrix = physical_affine_matrix(
            size=section_size,
            um_per_px=um_per_px,
            pivot=pivot_in_section
            if pivot_in_section is not None
            else None
            if pivot is None
            else (pivot[0] - ox, pivot[1] - oy),
            **params,
        )
    matrix = (_shift((ox, oy)) @ _as_3x3(section_matrix) @ _shift((-ox, -oy)))[:2]
    return section_matrix, matrix


@dataclass(frozen=True)
class PanelFrame:
    """Where one drawn placement panel's pixels sit, kept as it was drawn.

    What the picture's layers (:mod:`langslice.core.pictures.layers`) are
    computed from: the picture's *size* ``(width, height)`` (the captioned
    picture's; the content's in :class:`PlacementLayers`, which has no
    caption), the *content_box* holding the canvas crop above the caption, the
    *crop_box* on the canvas and *factor* (canvas px -> picture px, one for
    both axes, as the borders are drawn), the canvas *geometry*, the
    placement's *section_matrix* (3x3, the section render's frame -> canvas,
    :func:`placement_matrices`), and the atlas *lines* and *highlighted*
    region outlines drawn on it (native plane pixels, x/y; empty when none
    were drawn) at *line_width*, in *border_color* (resolved from "auto" if
    that is what was asked for; None when no line is drawn).
    *from_file*: a zoom whose section pixels were read from the image file, how
    many picture pixels one file pixel spans (None: drawn from the render).
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
    border_color: str | None = None

    @property
    def border_word(self) -> str:
        """How the drawn lines are worded in a caption (``black,
        white-edged``); empty when none are drawn."""
        return border_words(self.border_color) if self.border_color else ""


@dataclass(frozen=True)
class PlacementLayers:
    """A placement picture as layers, before anything is burned into it.

    *section* is the section on the canvas (the atlas image under it at the
    asked opacity, an applied warp in it), cropped and sized to the picture's
    content: RGB, no atlas border, no scale bar, no caption. The atlas
    borders are coverage arrays on that same pixel grid, float32 in [0, 1]:
    *borders* the context lines and *highlighted* the highlighted regions'
    outlines (zeros without any), each at *frame*'s ``line_width`` (the
    highlighted ones :data:`~langslice.core.atlas.borders.HIGHLIGHT_WIDTH`
    times as thick), before colour; *borders_casing* and *highlighted_casing*
    the same lines :data:`CASING_PX` wider on each side. :func:`compose_view`
    lays them cased in ``frame.border_color`` (the core colour; the casing is
    the other of black and white, :func:`casing_of`), the context ones at
    *context_alpha*. *frame* is where the pixels sit on the canvas and in the
    atlas (its ``size`` is the content's); *scale_bar* where the bar is drawn
    (None when it does not fit) and *caption* the words under the picture, laid
    out at least *caption_width* wide.
    """

    section: Image.Image
    borders: np.ndarray
    borders_casing: np.ndarray
    highlighted: np.ndarray
    highlighted_casing: np.ndarray
    context_alpha: float
    frame: PanelFrame
    scale_bar: ScaleBar | None
    caption: str
    caption_width: int


@dataclass(frozen=True)
class PhysicalView:
    """One finished placement picture: the captioned *image* and the
    :class:`PanelFrame` its layers are computed from."""

    image: Image.Image
    frame: PanelFrame


def content_edge(long_edge: int | None, box: tuple[int, int, int, int], zoomed: bool) -> int | None:
    """The long edge a crop of the canvas at *box* is sized to: *long_edge*,
    or the crop's own when that is smaller (never upsampled) unless the crop is
    a zoom, which is enlarged to *long_edge*; None keeps canvas pixels."""
    if long_edge is None:
        return None
    if zoomed:
        return max(1, int(long_edge))
    return max(1, min(int(long_edge), max(box[2] - box[0], box[3] - box[1])))


def content_size(box: tuple[int, int, int, int], edge: int | None) -> tuple[int, int]:
    """The ``(width, height)`` a crop of the canvas at *box* has when sized to
    *edge* (:func:`content_edge`), the size
    :func:`langslice.core.geometry.affine.resize_long_edge` gives it."""
    width, height = box[2] - box[0], box[3] - box[1]
    if edge is None or max(width, height) == edge:
        return (width, height)
    scale = edge / float(max(width, height))
    return (max(1, round(width * scale)), max(1, round(height * scale)))


def unzoomed_content_size(canvas_size: tuple[int, int], long_edge: int) -> tuple[int, int]:
    """The content size of the whole canvas of *canvas_size* at *long_edge*,
    which a zoom's window is given against: computed, not drawn."""
    whole = (0, 0, canvas_size[0], canvas_size[1])
    return content_size(whole, content_edge(long_edge, whole, False))


def physical_layers(
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
    border_thickness: float,
    outlines: str = "all",
    depth: int = OVERLAY_DEPTH,
    pivot: tuple[float, float] | None = None,
    label: str = "",
    long_edge: int | None = None,
    pivot_in_section: tuple[float, float] | None = None,
    atlas_picture: Image.Image | None = None,
    atlas_name: str = "template",
    regions: Any = (),
    matrix_label: str = "fitted matrix",
    template_lines: bool = False,
    detail: SectionDetail | None = None,
) -> PlacementLayers:
    """The alignment screen in one of :data:`VIEW_MODES`, as layers.

    Every view is built on the SAME millimetre-true canvas
    (:func:`canvas_geometry`) and the same crop, so a number read off one is
    the number on the others:

    * ``overlay`` — the warped section with the atlas region outlines on it,
      and the atlas image blended under them at *atlas_opacity*.
    * ``section`` — the warped section alone, no lines.
    * ``template`` — the atlas image alone; *template_lines* adds the
      *outlines* layer to it.

    *zoom* is ``[x0, y0, x1, y1]`` in fractions of the CANVAS (the tools take
    pixels and convert,
    :func:`langslice.core.pictures.options.zoom_fractions`); the crop happens
    before the screen is sized, so it is real magnification up to the canvas's
    own pixels: the panel's long edge is *long_edge*, or the crop's when that
    is smaller (never upsampled; ``None`` is canvas pixels one to one). A
    caller wanting a magnified zoom draws the canvas from a larger render. The
    scale bar is for the magnified micrometres per pixel. *zoom_pixels*
    is the zoom as the caller gave it (pixels of the unzoomed picture), for the
    caption.

    *outlines* picks which atlas lines are drawn: every boundary between
    regions at ontology *depth* (``all``, :func:`langslice.core.atlas.levels.labels_at`)
    or none (``none``). *border_color*
    is the core colour of the cased line: "black" (white casing), "white"
    (black casing) or :data:`AUTO_BORDER_COLOR` ("auto", the default), resolved
    from the lightness of the picture's own pixels under the lines
    (:func:`resolve_border_color`). *border_thickness* (the core) is 0.25..8
    output-image pixels, unchanged by zoom or canvas resolution.

    *pivot* is the rotation/scale centre in CANVAS pixels (``None`` is the
    canvas centre). *pivot_in_section* gives the pivot on the SECTION's frame
    instead and wins over *pivot*: a picture drawn from a larger render
    (:func:`shown_section`) knows its pivot only relative to the section.

    *atlas_picture* is the atlas image on the native plane grid (None: the
    reference template), named *atlas_name* in captions. *regions* (``[(name,
    ids)]``) are drawn at full strength and
    :data:`~langslice.core.atlas.borders.HIGHLIGHT_WIDTH` times as thick in
    every mode, the *outlines* layer then at :data:`REGION_CONTEXT_ALPHA` for
    context. *matrix_label* names a ready matrix in the caption.

    *detail*, for a zoom of the ``overlay`` or ``section`` view, gives the
    section's pixels at points of *section* (pixels, centres at integers): the
    zoom shows them in place of *section*'s own, the atlas image and the lines
    over them as usual (:func:`langslice.core.pictures.placement.canvas_layers`
    passes the image file's own pixels, :mod:`langslice.core.sections.native`).
    It answers ``(rgb, inside, enlarged)`` or None (the section's own pixels
    then).
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
        section.size,
        geometry.um_per_px,
        geometry.section_offset,
        params,
        pivot=pivot,
        pivot_in_section=pivot_in_section,
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
        []
        if layer == "none"
        else region_outlines(
            atlas, position_mm, depth=depth, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
        )
    )

    highlighted: list[np.ndarray] = []
    sides_note = ""
    if regions:
        linear = _as_3x3(section_matrix)[:2, :2]
        sides = regions_left(
            atlas,
            regions,
            position_mm,
            plane,
            pitch_deg,
            yaw_deg,
            np.linalg.inv(linear) * geometry.atlas_scale,
        )
        highlighted = region_polys(geometry.annotation, regions, sides)
        if sides is not None and np.linalg.det(linear) < 0:
            sides_note = " (the section's sides; this placement mirrors it)"
    atlas_head = f"atlas {atlas_name}"

    def _template_canvas() -> np.ndarray:
        plate = np.zeros_like(warped)
        _blend_template(
            plate,
            atlas,
            position_mm,
            plane,
            pitch_deg,
            yaw_deg,
            geometry,
            opacity=1.0,
            picture=atlas_picture,
        )
        return plate

    lines = True
    if mode not in VIEW_MODES:
        raise ValueError(f"mode must be one of {VIEW_MODES}; got {mode!r}")
    box = zoom_box(zoom, geometry.size)
    zoomed = box != (0, 0, geometry.size[0], geometry.size[1])
    bare = warped.copy() if detail is not None and zoomed and mode != "template" else None
    opacity = float(np.clip(atlas_opacity, 0.0, 1.0))
    if mode == "section":
        drawn = warped
        head = label or "section"
        lines = False
    elif mode == "template":
        drawn = _template_canvas()
        head = atlas_head
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
        drawn = warped
        head = label or "section"

    drawn_polys = (list(atlas_lines) if lines else []) + list(highlighted)
    coverage = (
        line_coverage(
            drawn.shape[:2],
            drawn_polys,
            thickness=max(float(border_thickness), 1.0),
            scale=geometry.atlas_scale,
            offset=geometry.atlas_offset,
        )
        if drawn_polys
        else None
    )
    line_color, line_width = resolve_border_color(
        border_color,
        border_thickness,
        rgb=np.asarray(drawn),
        coverage=coverage,
    )

    knobs = matrix_label if isinstance(params, np.ndarray) else knob_words(params)

    edge = content_edge(long_edge, box, zoomed)
    screen, factor = _to_screen(drawn, box, edge)
    from_file: float | None = None
    if bare is not None and detail is not None:
        found = _detailed_screen(bare, box, edge, matrix, geometry.section_offset, detail)
        if found is not None:
            screen, from_file = found
            if mode == "overlay" and opacity > 0.0:
                _blend_screen(screen, _template_canvas(), box, edge, opacity)
    shape = screen.shape[:2]
    reach = {
        "scale": geometry.atlas_scale,
        "offset": geometry.atlas_offset,
        "origin": box[:2],
        "factor": factor,
    }
    borders, borders_casing = cased_coverage(
        shape, atlas_lines if lines else [], thickness=line_width, **reach
    )
    emphasised, emphasised_casing = cased_coverage(
        shape, highlighted, thickness=line_width * HIGHLIGHT_WIDTH, **reach
    )
    text = (
        f"{head} @ {position_mm:.3f} mm  pitch {pitch_deg:.2f} yaw {yaw_deg:.2f}\n"
        f"{knobs}  canvas {geometry.um_per_px:.2f} um/px"
    ).strip()
    if mode != "overlay" or zoomed or abs(float(factor) - 1.0) > 0.02:
        where = f"  zoom {[round(float(v)) for v in zoom_pixels]} px" if zoom_pixels else ""
        text += f"\n{mode}{where}  view {geometry.um_per_px / factor:.2f} um/px"
        if from_file is not None:
            text += "  section from its image file" + (
                f" (file pixels enlarged x{from_file:.1f})" if from_file > 1.05 else ""
            )
        elif factor > 1.05:
            text += f" (canvas pixels enlarged x{factor:.1f})"
    if regions:
        text += "  regions " + ",".join(str(name) for name, _ids in regions) + sides_note
    frame = PanelFrame(
        size=(shape[1], shape[0]),
        content_box=(0, 0, shape[1], shape[0]),
        crop_box=box,
        factor=float(factor),
        geometry=geometry,
        section_matrix=_shift(geometry.section_offset) @ _as_3x3(section_matrix),
        lines=tuple(atlas_lines) if lines else (),
        highlighted=tuple(highlighted),
        line_width=float(line_width),
        mode=mode,
        from_file=from_file,
        border_color=line_color if drawn_polys else None,
    )
    return PlacementLayers(
        section=Image.fromarray(screen, mode="RGB"),
        borders=borders,
        borders_casing=borders_casing,
        highlighted=emphasised,
        highlighted_casing=emphasised_casing,
        context_alpha=REGION_CONTEXT_ALPHA if regions else 1.0,
        frame=frame,
        scale_bar=scale_bar(shape, geometry.um_per_px / factor, dark),
        caption=text,
        caption_width=edge if zoomed and edge is not None else 0,
    )


def compose_view(layers: PlacementLayers) -> PhysicalView:
    """The finished picture of *layers*: the section with the cased atlas
    borders laid on it, then the scale bar, then the caption band. The one
    way a placement picture is drawn from its layers."""
    screen = np.asarray(layers.section, dtype=np.uint8).copy()
    core = layers.frame.border_color
    if core is not None:
        for line, casing, alpha in (
            (layers.borders, layers.borders_casing, layers.context_alpha),
            (layers.highlighted, layers.highlighted_casing, 1.0),
        ):
            if line.any():
                lay_cased(screen, core, line=line, casing=casing, alpha=alpha)
    draw_scale_bar(screen, layers.scale_bar)
    labelled = caption(
        Image.fromarray(screen, mode="RGB"), layers.caption, min_width=layers.caption_width
    )
    return PhysicalView(image=labelled, frame=replace(layers.frame, size=labelled.size))


def knob_words(params: Any, sep: str = "  ") -> str:
    """An in-plane transform's knobs for a caption: ``rot 0.3  scale
    0.990/1.086  shift -0.14/+0.00 mm`` (and the shear when not 0), the
    parts joined by *sep*."""
    parts = [
        f"rot {float(params['rotation_deg']):.1f}",
        f"scale {float(params['scale_x']):.3f}/{float(params['scale_y']):.3f}",
        f"shift {float(params['translate_x_mm']):+.2f}/{float(params['translate_y_mm']):+.2f} mm",
    ]
    if params.get("shear"):
        parts.append(f"shear {float(params['shear']):+.3f}")
    return sep.join(parts)


def _detailed_screen(
    bare: np.ndarray,
    box: tuple[int, int, int, int],
    edge: int | None,
    matrix: np.ndarray,
    offset: tuple[int, int],
    detail: SectionDetail,
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
    screen: np.ndarray,
    plate: np.ndarray,
    box: tuple[int, int, int, int],
    edge: int | None,
    opacity: float,
) -> None:
    """The atlas image *plate* (zero where it has none) blended under
    *screen* at *opacity*, as :func:`_blend_template` blends it on the canvas."""
    shown, _factor = _to_screen(plate, box, edge)
    x0, y0, x1, y1 = box
    lit = (plate[y0:y1, x0:x1].max(axis=2) > 0).astype(np.uint8)
    lit = cv2.resize(lit, (screen.shape[1], screen.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    screen[lit] = (screen[lit] * (1.0 - opacity) + shown[lit] * opacity).astype(np.uint8)
