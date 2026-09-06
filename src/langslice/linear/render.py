"""Rendering the stack: sections as the run sees them, plus the status table.

Every path that shows or measures a section goes through :func:`render_slice`,
so the pixels the agent judges are the pixels a fit is computed on. Corrections
are applied in one order everywhere: ROTATE first, then FLIP left-right.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from google.genai import types
from PIL import Image, ImageDraw, ImageFont

from langslice.affine import (
    extract_slice_silhouette,
    physical_affine_matrix,
    resize_long_edge,
    silhouette_iou,
)
from langslice.atlas.core import get_reference_slice
from langslice.atlas.render import (
    annotation_slice,
    atlas_um_per_px,
    family_outlines,
    is_dark_background,
    region_contours,
)
from langslice.image_prep import (
    adaptive_preprocess,
    crop_to_tissue,
    foreground_mask,
    normalize_image,
    prepare_image_for_vlm,
)
from langslice.linear.state import SliceState, StackState
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox that renders
    from langslice.linear.engine import EngineContext

#: Long edge for ``view_slices`` images: enough to zoom past the seed-message
#: stack images without paying full-resolution tokens.
VIEW_LONG_EDGE = 1024
#: Long edge for the per-section images in the seed message. Small on purpose:
#: the whole stack (40-odd sections) rides in one user message and stays in
#: context for the whole session.
SEED_IMAGE_LONG_EDGE = 512
#: Working frame for transform fits and their preview panels.
PREVIEW_LONG_EDGE = 512
#: Long edge of the ONE image the interactive alignment loop looks at.
OVERLAY_LONG_EDGE = 768
#: Font size of the label strip :func:`caption` burns into an image.
CAPTION_PX = 14

_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def image_to_part(img: Image.Image, *, quality: int = 85) -> types.Part:
    """One PIL image as a JPEG ``types.Part``."""
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return types.Part.from_bytes(mime_type="image/jpeg", data=buf.getvalue())


@lru_cache(maxsize=1)
def _caption_font() -> Any:
    try:
        return ImageFont.load_default(size=CAPTION_PX)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def caption(image: Image.Image, text: str) -> Image.Image:
    """A COPY of *image* with *text* burned into a dark strip, top-left.

    Tool images reach the model as bare attachments, so the text that binds an
    image to its section or its position has to ride in the pixels. Caption
    only what is SHOWN: an image a fit measures must never be captioned, since
    the strip changes the pixels the fit reads.
    """
    source = image.convert("RGB")
    font = _caption_font()
    probe = ImageDraw.Draw(source)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    band = int(bottom - top + 6)
    # The band sits ABOVE the picture, never over it: a caption drawn on the
    # pixels covered exactly the magnified dorsal tissue an agent was reading.
    labelled = Image.new("RGB", (source.width, source.height + band), (0, 0, 0))
    labelled.paste(source, (0, band))
    draw = ImageDraw.Draw(labelled)
    draw.text((3 - left, 3 - top), text, fill=(255, 255, 255), font=font)
    return labelled


def render_cache_key(
    ctx: EngineContext, record: SliceState, *, long_edge: int, frame: bool
) -> tuple[str, bool, int, int, str, bool]:
    """The key a render is cached under: the section plus everything it shows."""
    return (record.id, record.flip, record.rotation_deg, long_edge, ctx.spec.preprocess, frame)


def render_slice(
    ctx: EngineContext,
    record: SliceState,
    *,
    long_edge: int = VIEW_LONG_EDGE,
    frame: bool = False,
) -> Image.Image:
    """One section as the run sees it: normalized, framed, enhanced, corrected.

    With ``spec.preprocess == "auto"`` (the default) the section is run through
    :func:`~langslice.image_prep.adaptive_preprocess` — per-channel CLAHE plus a
    DAPI-weighted grayscale blend — so dim fluorescence reads like the atlas
    instead of like a black field. Display only: the user's file is never
    touched.

    *frame* crops to the tissue plus a small margin before the resize, so the
    section fills its frame about as much as a cropped atlas render does. It is
    off by default because it changes the image's coordinate frame: only the
    paths that SHOW a section to a model set it, never a fit, whose parameters
    are normalized against the render they were computed on.

    Renders are cached on *ctx*, so the returned image is shared: read it,
    never mutate it in place.
    """
    key = render_cache_key(ctx, record, long_edge=long_edge, frame=frame)
    cached = ctx.render_cache.get(key)
    if cached is not None:
        return cached

    with Image.open(ctx.image_path(record.id)) as handle:
        # Detach from the file handle: prepare_image_for_vlm can hand back the
        # very object it was given when no resize is needed.
        source = normalize_image(handle.copy())
    if frame:
        source = crop_to_tissue(source)
    prepped = prepare_image_for_vlm(source, max_long_edge=long_edge).image
    # How much the render shrank the pixels, before any quarter-turn: the
    # section's own micrometres per pixel times this is the canvas's.
    ctx.render_scale[key] = source.width / float(prepped.width)
    if ctx.spec.preprocess == "auto":
        prepped = adaptive_preprocess(prepped)
    rotate = _ROTATE_OPS.get(int(record.rotation_deg) % 360)
    if rotate is not None:
        prepped = prepped.transpose(rotate)
    if record.flip:
        prepped = prepped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ctx.render_cache[key] = prepped
    return prepped


def canvas_um_per_px(
    ctx: EngineContext,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    frame: bool = False,
) -> tuple[float | None, str]:
    """``(micrometres per pixel of the RENDER, source)`` for one section.

    The render is a downsample of the file, so the file's pixel size times
    the downsample factor is the working canvas's. ``(None, "")`` when
    neither the file nor the host supplies one.
    """
    from_file, source = ctx.calibration(record.id)
    if from_file is None:
        return None, source
    key = render_cache_key(ctx, record, long_edge=long_edge, frame=frame)
    if key not in ctx.render_scale:
        render_slice(ctx, record, long_edge=long_edge, frame=frame)
    return from_file * ctx.render_scale.get(key, 1.0), source


# --- the status table ----------------------------------------------------


def status_rows(state: StackState) -> list[dict[str, Any]]:
    """One row per section in corrected order. The ``ls`` of the environment.

    ``delta_to_next_mm`` is the SIGNED distance to the next section in
    corrected order that carries a position, and null when this section has
    none or no placed section follows it. ``transform_iou`` and
    ``transform_mirrored`` come off the recorded transform. Data only: no
    comparison against the nominal interval, no verdict.
    """
    ordered = state.in_order()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        here = record.position_mm
        delta: float | None = None
        if here is not None:
            following = next(
                (
                    value
                    for r in ordered[index + 1 :]
                    if (value := r.position_mm) is not None
                ),
                None,
            )
            if following is not None:
                delta = round(following - here, 3)
        transform = record.transform or {}
        rows.append(
            {
                "index": record.index_corrected,
                "id": record.id,
                "position_mm": round(here, 3) if here is not None else None,
                "delta_to_next_mm": delta,
                "flip": record.flip,
                "rotation_deg": record.rotation_deg,
                "damaged": record.damaged,
                "damage_note": record.damage_note,
                "transform": transform.get("kind"),
                "transform_iou": transform.get("iou"),
                "transform_mirrored": transform.get("mirrored"),
                "confidence": record.confidence,
                "caveats": list(record.caveats),
            }
        )
    return rows


def status_text(state: StackState) -> str:
    """The status rows as one line per section, for a text message."""
    lines = [
        "index  id  position_mm  delta_to_next_mm  flags  "
        "transform(kind, iou, mirrored)  confidence"
    ]
    for row in status_rows(state):
        flags: list[str] = []
        if row["flip"]:
            flags.append("flipped")
        if row["rotation_deg"]:
            flags.append(f"rotated {row['rotation_deg']}")
        if row["damaged"]:
            note = row["damage_note"]
            flags.append(f"damaged: {note}" if note else "damaged")
        flags.extend(row["caveats"])
        position = (
            "unplaced" if row["position_mm"] is None else f"{row['position_mm']:.3f} mm"
        )
        delta = (
            "-" if row["delta_to_next_mm"] is None else f"{row['delta_to_next_mm']:.3f}"
        )
        transform = ""
        if row["transform"]:
            transform = f"  transform={row['transform']}"
            if row["transform_iou"] is not None:
                transform += f" iou={float(row['transform_iou']):.3f}"
            if row["transform_mirrored"] is not None:
                transform += f" mirrored={bool(row['transform_mirrored'])}"
        lines.append(
            f"{row['index']:>3}  {row['id']}  {position}  {delta}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
            + transform
            + (f"  confidence={row['confidence']}" if row["confidence"] else "")
        )
    return "\n".join(lines)


def slice_flags(record: SliceState) -> list[str]:
    """The section's current corrections, as short human-readable flags."""
    flags: list[str] = []
    if record.rotation_deg:
        flags.append(f"rotated {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append(
            f"damaged: {record.damage_note}" if record.damage_note else "damaged"
        )
    return flags


def stack_image_parts(
    state: StackState, ctx: EngineContext, *, long_edge: int = SEED_IMAGE_LONG_EDGE
) -> list[types.Part]:
    """The whole stack as labelled text+image pairs, in corrected order.

    Each section gets a one-line label — ``"<corrected index>: <filename>"``
    plus any flags — immediately followed by its own image, rendered through
    :func:`render_slice` so preprocessing and corrections are already applied.

    One image per section rather than one thumbnail grid: a grid splits a fixed
    vision-encoder patch budget across every section at once and lets
    neighbouring sections share patch boundaries. A labelled sequence at a
    modest resolution reads better, and the label is what binds each set of
    pixels to a filename the model can quote back. The same label is burned
    into the image (:func:`caption`), so it survives any transport that drops
    the text part next to an attachment.
    """
    parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                f"The {len(state.slices)} sections of the stack follow, in "
                "their current corrected order, one image each. Every image is "
                "preceded by its label '<index>: <filename>' and is rendered "
                "with any rotation and flip already applied."
            )
        )
    ]
    for record in state.in_order():
        flags = slice_flags(record)
        label = f"{record.index_corrected}: {record.id}"
        if flags:
            label += f"  [{'; '.join(flags)}]"
        parts.append(types.Part.from_text(text=label))
        parts.append(
            image_to_part(
                caption(
                    render_slice(ctx, record, long_edge=long_edge, frame=True), label
                )
            )
        )
    return parts


# --- the physical overlay ------------------------------------------------



#: Black working space on each side of the larger of section and atlas, as a
#: fraction of that extent. Room for x/y moves, like ABBA's viewer.
WORKING_MARGIN = 0.12

@dataclass(frozen=True)
class CanvasGeometry:
    """Where the section and the atlas sit on one millimetre-true canvas.

    The canvas IS the section's frame, grown symmetrically when the atlas
    anatomy at true scale would not fit inside it. Both frames therefore share
    a centre, which is why a transform's rotation centre is the same in either
    (:func:`langslice.affine.normalized_physical_affine`).
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
    the canvas, which is not a calibration: measured on LSD_910 M01,
    fit-to-canvas put the atlas plate at 0.69x the tissue where true scale
    puts it at 1.05x. Its ANATOMY (the annotation's bounding box, not the
    atlas frame with its empty margins) is centred on the canvas centre, and
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


def scale_bar_px(um_per_px: float, mm: float = 1.0) -> int:
    """Length in pixels of a *mm*-millimetre bar on a *um_per_px* canvas."""
    return int(round(mm * 1000.0 / float(um_per_px)))


def _draw_scale_bar(canvas: np.ndarray, um_per_px: float, dark: bool) -> None:
    """A 1 mm bar in the bottom-left corner, labelled. Thin, like a viewer's."""
    height, width = canvas.shape[:2]
    length = scale_bar_px(um_per_px)
    if length < 4 or length > width:
        return
    color = (235, 235, 235) if dark else (40, 40, 40)
    thickness = max(1, round(min(height, width) / 400))
    x0, y0 = 12, height - 14
    cv2.rectangle(canvas, (x0, y0), (x0 + length, y0 + thickness), color, -1)
    cv2.putText(
        canvas, "1 mm", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
    )


def _line_color(dark: bool) -> tuple[int, int, int]:
    """The atlas hairline's neutral grey, for whichever background it lands on."""
    return (235, 235, 235) if dark else (40, 40, 40)


def _tissue_color(dark: bool) -> tuple[int, int, int]:
    """A SECOND grey, for the section's own silhouette in the outlines view."""
    return (150, 150, 150) if dark else (140, 140, 140)


def _draw_polys(
    canvas: np.ndarray,
    polys: list[np.ndarray],
    color: tuple[int, int, int],
    *,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
) -> None:
    """Closed x/y polylines as 1 px anti-aliased lines, at OUTPUT resolution.

    Each point runs through the same chain the pixels did: its own frame ->
    canvas (*scale*, *offset*), minus the zoom crop's *origin*, times *factor*,
    the canvas-px -> output-px ratio. Sub-pixel via OpenCV's 4-bit shift.
    """
    ox, oy = float(origin[0]), float(origin[1])
    for poly in polys:
        points = np.round(
            ((poly * scale + np.asarray(offset, dtype=np.float64)) - (ox, oy))
            * factor
            * 16.0
        ).astype(np.int32)
        cv2.polylines(canvas, [points], True, color, 1, cv2.LINE_AA, 4)


def _draw_outlines(
    canvas: np.ndarray,
    outlines: list[tuple[tuple[int, int, int], np.ndarray]],
    geometry: CanvasGeometry,
    *,
    dark: bool,
    factor: float = 1.0,
    origin: tuple[int, int] = (0, 0),
) -> None:
    """Atlas region borders the way ABBA draws them: one hairline, one neutral
    color, no rim. ABBA's border channel is the 1-voxel edge of the label
    volume shown in a single grey; the per-region colors belong to the filled
    map, not to lines over tissue. *factor* is canvas px -> output px, so the
    line is 1 px on the screen the model sees whatever the canvas size, and
    *origin* is the zoom crop's top-left corner in canvas pixels.
    """
    _draw_polys(
        canvas,
        [poly for _color, poly in outlines],
        _line_color(dark),
        scale=geometry.atlas_scale,
        offset=geometry.atlas_offset,
        origin=origin,
        factor=factor,
    )


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
) -> tuple[int, int, np.ndarray]:
    """The atlas template resized to the canvas's scale, ready to paste."""
    template = get_reference_slice(
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
) -> None:
    """Blend the atlas template under the outlines, at the same placement."""
    y0, x0, patch = _template_patch(
        atlas, position_mm, plane, pitch_deg, yaw_deg, geometry
    )
    if patch.size == 0:
        return
    window = canvas[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
    lit = patch.max(axis=2) > 0
    window[lit] = (
        window[lit] * (1.0 - opacity) + patch[lit] * opacity
    ).astype(np.uint8)


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
    template_opacity: float = 0.0,
    pad_to_fit_atlas: bool = True,
    pivot: tuple[float, float] | None = None,
    markers: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "",
    long_edge: int | None = None,
) -> tuple[list[Image.Image], float]:
    """The alignment screen in one of :data:`VIEW_MODES`, plus the overlap.

    One composition, several ways of showing it. Every view is built on the
    SAME millimetre-true canvas (:func:`canvas_geometry`) and the same crop, so
    a number read off one is the number on the others:

    * ``overlay`` — the warped section with the atlas family outlines on it,
      and the atlas template blended under them at *template_opacity*.
    * ``side_by_side`` — two images, the warped section and the atlas template,
      at the same micrometres per pixel and the same crop, outlines on both.
    * ``checkerboard`` — section and template in alternating tiles.
    * ``outlines`` — the atlas lines and the section's own silhouette contour
      in a second grey, on black. No pixels.

    *zoom* is ``[x0, y0, x1, y1]`` in fractions of the CANVAS; the crop happens
    before the resize to *long_edge*, so it is real magnification, and the
    scale bar is redrawn for the magnified micrometres per pixel.

    *pivot* is the rotation/scale centre in CANVAS pixels (``None`` is the
    canvas centre), and *markers* is ``(section points, atlas points)`` in
    canvas pixels, drawn on every panel as landmark pairs.

    Returns ``(images, silhouette_iou)`` — every image captioned, and the
    overlap between the warped section's tissue mask and the atlas anatomy at
    this placement.
    """
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

    ox, oy = (float(v) for v in geometry.section_offset)
    if isinstance(params, np.ndarray):
        section_matrix = np.asarray(params, dtype=np.float64)
    else:
        section_matrix = physical_affine_matrix(
            size=section.size,
            um_per_px=geometry.um_per_px,
            # The pivot arrives on the canvas; the matrix is built on the
            # section's frame and conjugated onto the canvas below.
            pivot=None if pivot is None else (pivot[0] - ox, pivot[1] - oy),
            **params,
        )
    matrix = (_shift((ox, oy)) @ _as_3x3(section_matrix) @ _shift((-ox, -oy)))[:2]
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
    outlines = family_outlines(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )

    def _template_canvas() -> np.ndarray:
        plate = np.zeros_like(warped)
        _blend_template(
            plate, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, opacity=1.0
        )
        return plate

    silhouette: list[np.ndarray] = []
    lines = True  # atlas outlines drawn on every panel...
    if mode == "section":
        panels = [(warped, label or "section")]
        lines = False  # ...except the clean views, which show one source alone
    elif mode == "template":
        panels = [(_template_canvas(), "atlas template")]
        lines = False
    elif mode == "side_by_side":
        panels = [(warped, label or "section"), (_template_canvas(), "atlas template")]
    elif mode == "checkerboard":
        panels = [(_checkerboard(warped, _template_canvas()), label or "section")]
    elif mode == "outlines":
        silhouette = _silhouette_polys(tissue)
        panels = [(np.zeros_like(warped), label or "section")]
        dark = True  # the canvas is black whatever the section is
    else:
        if template_opacity > 0.0:
            _blend_template(
                warped,
                atlas,
                position_mm,
                plane,
                pitch_deg,
                yaw_deg,
                geometry,
                opacity=float(np.clip(template_opacity, 0.0, 1.0)),
            )
        panels = [(warped, label or "section")]

    box = zoom_box(zoom, geometry.size)
    if isinstance(params, np.ndarray):
        knobs = "fitted matrix"
    else:
        knobs = (
            f"rot {params['rotation_deg']:.1f}  "
            f"scale {params['scale_x']:.3f}/{params['scale_y']:.3f}  "
            f"shift {params['translate_x_mm']:+.2f}/{params['translate_y_mm']:+.2f} mm"
        )

    images: list[Image.Image] = []
    for panel, head in panels:
        screen, factor = _to_screen(panel, box, long_edge)
        if lines:
            _draw_outlines(
                screen, outlines, geometry, dark=dark, factor=factor, origin=box[:2]
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
        # Two lines: one caption wide enough for all of it would run off a
        # 512px canvas, and `caption` clips rather than wraps.
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
        images.append(caption(Image.fromarray(screen, mode="RGB"), text))
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
    template_opacity: float = 0.0,
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
        template_opacity=template_opacity,
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
