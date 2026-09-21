"""Review-grade rendering of registration output over histology.

Everything here is presentation. It takes a label map (the warped atlas,
already classified to region ids), the histology working canvas, and a color
LUT, and returns images a person can judge a registration by. No atlas
loading, no registration math, no model calls — callers pass the LUT and the
acronyms in, so every function here is testable on a synthetic label map.

Why the old ``_overlay_borders`` looked bad, and what replaces it:

* a label-boundary mask is one aliased pixel wide and dilating it into a rim
  turns every diagonal into a 3px staircase → here boundaries are traced as
  polygons, low-pass filtered, and drawn anti-aliased at 1/16-pixel precision;
* one loud yellow for every border erases the region identity → here each
  boundary is drawn in a saturated version of its OWN region's color;
* no fills means nothing tells you which side of a line is which structure →
  here regions are filled translucently so the tissue reads through;
* no hierarchy means a thin cortical layer looks as important as the
  midbrain/cerebellum split → here fine borders are hairline and the coarse
  family borders are heavier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Contours, family colors and the shade rules moved to `atlas/render.py` when
# `linear` started drawing the same lines; re-exported here so this module
# stays the one import for review rendering.
from langslice.atlas.render import (
    BORDER_DARKEN,
    Rgb,
    border_color,
    darker,
    is_dark_background,
    region_contours,
)

__all__ = [
    "BORDER_DARKEN",
    "border_color",
    "checkerboard",
    "checkerboard_mask",
    "contact_sheet",
    "darker",
    "deformation_grid",
    "filled_regions",
    "is_dark_background",
    "label_anchor",
    "region_contours",
    "region_overlay",
    "split_view",
]

#: Everything sized in pixels is tuned at this canvas long edge and scaled.
_REFERENCE_LONG_EDGE = 2048.0

_FONT_CANDIDATES = (
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/google-carlito-fonts/Carlito-Bold.ttf",
    "/usr/share/fonts/adwaita-sans-fonts/AdwaitaSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
)


@lru_cache(maxsize=64)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """A bold sans face at *size*, whatever this machine happens to ship."""
    for path in _FONT_CANDIDATES:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1: scalable
    except TypeError:  # pragma: no cover - ancient Pillow
        return ImageFont.load_default()


def _turbo() -> np.ndarray:
    """256-entry RGB ramp; readable on both light and dark backgrounds."""
    ramp = np.arange(256, dtype=np.uint8).reshape(1, 256)
    return np.asarray(cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO))[0][:, ::-1].copy()


def _as_rgb(image: Image.Image, size: tuple[int, int] | None = None) -> np.ndarray:
    """``(H, W, 3)`` uint8 view of *image*, resized to ``(W, H)`` if asked."""
    rgb = image.convert("RGB")
    if size is not None and rgb.size != size:
        rgb = rgb.resize(size, resample=Image.Resampling.LANCZOS)
    return np.asarray(rgb, dtype=np.uint8).copy()


def _scale_for(shape: tuple[int, ...]) -> float:
    return max(shape[0], shape[1]) / _REFERENCE_LONG_EDGE


#: Family boundaries are drawn this much heavier than leaf boundaries — the
#: ARA hierarchy cue: a major division reads before its subdivisions do.
_FAMILY_BORDER_SCALE = 1.8


def filled_regions(
    labels: np.ndarray,
    *,
    lut: Mapping[int, Rgb],
    size: tuple[int, int] | None = None,
    smooth_window: int = 9,
    min_area_px: float = 2.0,
    border_px: float = 0.0,
    families: np.ndarray | None = None,
) -> np.ndarray:
    """A flat region map drawn from smoothed contours, at ``size`` (W, H).

    The alternative — NEAREST-upscaling a 25/39um annotation to a 2048px
    canvas — turns every boundary into a voxel staircase an image model then
    reproduces. Here each region is traced at atlas resolution, its outline
    low-pass filtered, scaled into the target canvas and filled as a polygon,
    so boundaries carry canvas-resolution detail.

    Fills are flat and un-antialiased on purpose: every pixel must be an
    EXACT palette color, because nearest-color classification is what turns
    the model's answer back into region ids. Even-odd filling keeps a
    region's holes empty, and regions are painted largest-first so a small
    one is never buried by a neighbour whose smoothed outline overshot it.

    ``border_px`` > 0 adds the ARA plate style on top: every boundary in
    *labels* delineated by a hairline in :func:`darker` of the region's own
    color, at that width measured on a 2048px canvas and scaled from there.
    Pass *families* — the same map merged to divisions — for the ARA
    hierarchy cue, division lines heavier than the subdivisions inside them.
    """
    height, width = labels.shape[:2]
    out_w, out_h = size if size is not None else (width, height)
    scale = np.array([out_w / width, out_h / height], dtype=np.float64)

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    contours = region_contours(labels, smooth_window=smooth_window, min_area_px=min_area_px)
    # A contour traces pixel CENTERS, so filling it alone leaves each region
    # half a source pixel short of its neighbours — at a 4x upscale that is a
    # black seam around everything. Stroking the same outline grows each
    # region back by half the stroke, so neighbours overlap instead of gapping
    # and the boundary lands wherever the later (smaller) region put it.
    grow = max(2, int(np.ceil(2.0 * float(scale.max()))))
    uids, counts = np.unique(labels, return_counts=True)
    areas = {int(uid): int(count) for uid, count in zip(uids, counts, strict=True)}
    for uid in sorted(contours, key=lambda u: -areas.get(u, 0)):
        color = lut.get(uid, (128, 128, 128))
        points = [
            np.round(poly * scale * 16.0).astype(np.int32).reshape(-1, 1, 2)
            for poly in contours[uid]
        ]
        cv2.fillPoly(canvas, points, color, cv2.LINE_8, 4)
        cv2.polylines(canvas, points, True, color, grow, cv2.LINE_8, 4)
    canvas = _close_seams(canvas, max_hole_px=(4 * grow) ** 2)
    if border_px <= 0.0:
        return canvas

    def _delineate(traced: dict[int, list[np.ndarray]], thickness: int) -> None:
        # LINE_8, like the fills: an anti-aliased line would blend the two
        # colors it runs between into pixels belonging to neither, and a
        # bordered render has to stay classifiable to exact palette colors.
        # The curve is still sub-pixel — it is the smoothed contour, drawn at
        # 1/16-pixel precision.
        for uid, polys in traced.items():
            points = [
                np.round(poly * scale * 16.0).astype(np.int32).reshape(-1, 1, 2) for poly in polys
            ]
            color = darker(lut.get(uid, (128, 128, 128)))
            cv2.polylines(canvas, points, True, color, thickness, cv2.LINE_8, 4)

    width = max(1, round(border_px * _scale_for((out_h, out_w))))
    _delineate(contours, width)
    if families is not None:
        _delineate(
            region_contours(families, smooth_window=smooth_window, min_area_px=min_area_px),
            max(width + 1, round(width * _FAMILY_BORDER_SCALE)),
        )
    return canvas


def _close_seams(canvas: np.ndarray, *, max_hole_px: int) -> np.ndarray:
    """Give every small enclosed unpainted speck its nearest region's color.

    Three regions meeting at a point leave a pinhole no stroke width closes
    (a mitred corner is cut off), and a pinhole of BACKGROUND inside the
    tissue is exactly the artifact this renderer exists to remove. Only
    enclosed specks are touched, so real unlabelled anatomy — which opens to
    the background or is far larger — keeps its color.
    """
    from scipy import ndimage

    painted = canvas.any(axis=2)
    holes = np.asarray(ndimage.binary_fill_holes(painted)) & ~painted
    if not holes.any():
        return canvas
    labelled, count = ndimage.label(holes)  # type: ignore[misc]
    sizes = np.bincount(np.asarray(labelled, dtype=np.intp).ravel())
    small = np.isin(labelled, np.nonzero(sizes[1:] <= max_hole_px)[0] + 1)
    if not small.any():
        return canvas
    _distance, (rows, cols) = ndimage.distance_transform_edt(  # type: ignore[misc]
        ~painted, return_indices=True
    )
    canvas[small] = canvas[rows[small], cols[small]]
    return canvas


def _draw_polys(
    base: np.ndarray,
    colored_polys: Iterable[tuple[Rgb, list[np.ndarray]]],
    *,
    thickness: int,
    opacity: float,
) -> np.ndarray:
    """Anti-aliased sub-pixel polylines at *opacity* over *base*.

    Drawing at full strength onto a copy and then blending the copy back
    reproduces exactly ``base*(1 - a*k) + color*a*k`` for OpenCV's per-pixel
    coverage ``a`` — i.e. a genuine sub-pixel-weight line, which is how you
    get a hairline border that does not shimmer.
    """
    if opacity <= 0.0:
        return base
    drawn = base.copy()
    for color, polys in colored_polys:
        points = [np.round(poly * 16.0).astype(np.int32) for poly in polys]
        if points:
            cv2.polylines(drawn, points, True, color, thickness, cv2.LINE_AA, 4)
    if opacity >= 1.0:
        return drawn
    return cv2.addWeighted(drawn, opacity, base, 1.0 - opacity, 0.0)


def label_anchor(mask: np.ndarray) -> tuple[float, float, float]:
    """``(x, y, radius)`` of the largest circle inscribed in *mask*.

    The pole of inaccessibility, not the centroid: a C-shaped or crescent
    region (dentate gyrus, cortical layer) has its centroid outside itself,
    and a label placed there points at the wrong structure.
    """
    padded = np.pad(mask.astype(np.uint8), 1)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)
    flat = int(np.argmax(dist))
    y, x = divmod(flat, dist.shape[1])
    return float(x - 1), float(y - 1), float(dist[y, x])


def _place_labels(
    labels: np.ndarray,
    *,
    names: Mapping[int, str],
    min_area_px: float,
    scale: float,
) -> list[tuple[str, float, float, int]]:
    """Acronym placements, biggest region first, skipping any that collide."""
    uids, counts = np.unique(labels, return_counts=True)
    order = np.argsort(counts)[::-1]
    # A label smaller than this is unreadable at review zoom, and an
    # unreadable label is worse than none: it is clutter over the tissue.
    # Regions too thin to hold one are dropped by the radius check below.
    min_size = max(9, int(round(21 * scale)))
    max_size = max(min_size, int(round(48 * scale)))

    placed: list[tuple[str, float, float, int]] = []
    boxes: list[tuple[float, float, float, float]] = []
    for index in order:
        uid = int(uids[index])
        if uid == 0 or counts[index] < min_area_px:
            continue
        text = str(names.get(uid, "")).strip()
        if not text:
            continue
        x, y, radius = label_anchor(labels == uid)
        size = int(round(min(max(radius * 0.85, min_size), max_size)))
        if radius < size * 0.55:
            continue  # too thin to hold its own name
        half_w = 0.30 * size * len(text) + 2
        half_h = 0.62 * size
        box = (x - half_w, y - half_h, x + half_w, y + half_h)
        if any(
            box[0] < other[2] and other[0] < box[2] and box[1] < other[3] and other[1] < box[3]
            for other in boxes
        ):
            continue
        boxes.append(box)
        placed.append((text, x, y, size))
    return placed


def _draw_text(
    image: Image.Image,
    placements: Sequence[tuple[str, float, float, int]],
    *,
    dark_background: bool,
) -> Image.Image:
    """Halo'd centered text; ink and halo flip with the canvas polarity."""
    if not placements:
        return image
    ink: Rgb = (255, 255, 255) if dark_background else (18, 18, 18)
    halo: Rgb = (0, 0, 0) if dark_background else (255, 255, 255)
    draw = ImageDraw.Draw(image)
    for text, x, y, size in placements:
        font = _font(size)
        stroke = max(1, size // 7)
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font, stroke_width=stroke)
        draw.text(
            (x - (right - left) / 2 - left, y - (bottom - top) / 2 - top),
            text,
            font=font,
            fill=ink,
            stroke_width=stroke,
            stroke_fill=halo,
        )
    return image


def region_overlay(
    histology: Image.Image,
    labels: np.ndarray,
    *,
    lut: Mapping[int, Rgb],
    names: Mapping[int, str] | None = None,
    families: np.ndarray | None = None,
    fill_alpha: float = 0.30,
    show_labels: bool = True,
    min_label_area_frac: float = 0.0035,
    smooth_window: int | None = None,
) -> Image.Image:
    """Translucent region fills plus smooth, per-region-colored boundaries.

    ``fill_alpha=0`` gives the borders-only review mode. Pass *families* (the
    same map merged to coarse divisions) to get the two-weight hierarchy:
    hairline sub-region borders under heavier division borders.
    """
    base = _as_rgb(histology, (labels.shape[1], labels.shape[0]))
    dark = is_dark_background(base)
    scale = _scale_for(labels.shape)
    window = smooth_window if smooth_window is not None else max(5, int(round(9 * scale)) | 1)

    out = base.astype(np.float32)
    if fill_alpha > 0.0:
        fill = np.zeros_like(base)
        for uid in np.unique(labels):
            if int(uid) == 0:
                continue
            fill[labels == uid] = lut.get(int(uid), (128, 128, 128))
        if dark:
            # Emissive imaging: a flat alpha-over lays a bright pastel haze on
            # black background and the section stops reading as fluorescence.
            # Modulating the fill by the tissue's own luminance tints the
            # signal and leaves true black alone.
            luma = cv2.cvtColor(base, cv2.COLOR_RGB2GRAY).astype(np.float32)[..., None] / 255.0
            fill = fill.astype(np.float32) * luma
        mask = (labels != 0)[..., None]
        out = np.where(mask, out * (1.0 - fill_alpha) + fill * fill_alpha, out)
    canvas = out.astype(np.uint8)

    fine = region_contours(labels, smooth_window=window)
    canvas = _draw_polys(
        canvas,
        (
            (border_color(lut.get(uid, (128, 128, 128)), dark_background=dark), polys)
            for uid, polys in fine.items()
        ),
        thickness=max(1, int(round(1.3 * scale))),
        opacity=0.55,
    )
    if families is not None:
        coarse = region_contours(families, smooth_window=window)
        canvas = _draw_polys(
            canvas,
            (
                (border_color(lut.get(uid, (128, 128, 128)), dark_background=dark), polys)
                for uid, polys in coarse.items()
            ),
            thickness=max(1, int(round(2.6 * scale))),
            opacity=0.92,
        )

    image = Image.fromarray(canvas, mode="RGB")
    if show_labels and names:
        placements = _place_labels(
            labels,
            names=names,
            min_area_px=min_label_area_frac * labels.size,
            scale=scale,
        )
        image = _draw_text(image, placements, dark_background=dark)
    return image


def checkerboard_mask(shape: tuple[int, int], tiles: int = 8) -> np.ndarray:
    """Boolean tile mask; True picks the first image. *tiles* spans the long edge."""
    height, width = shape
    tile = max(1, int(round(max(height, width) / max(1, tiles))))
    rows = (np.arange(height) // tile)[:, None]
    cols = (np.arange(width) // tile)[None, :]
    return (rows + cols) % 2 == 0


def checkerboard(
    first: Image.Image,
    second: Image.Image,
    *,
    tiles: int = 8,
    seam_opacity: float = 0.0,
) -> Image.Image:
    """Alternating tiles of two aligned images — the classic registration QC view.

    Anatomy that is registered runs straight through a tile seam; anatomy that
    is not steps sideways at every seam, which the eye catches instantly.
    """
    array_first = _as_rgb(first)
    array_second = _as_rgb(second, first.size)
    mask = checkerboard_mask(array_first.shape[:2], tiles)
    out = np.where(mask[..., None], array_first, array_second)
    if seam_opacity > 0.0:
        edges = np.zeros(mask.shape, dtype=bool)
        edges[:, 1:] |= mask[:, 1:] != mask[:, :-1]
        edges[1:, :] |= mask[1:, :] != mask[:-1, :]
        line = 235 if is_dark_background(array_first) else 40
        out[edges] = (out[edges] * (1 - seam_opacity) + line * seam_opacity).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")


def split_view(
    first: Image.Image,
    second: Image.Image,
    *,
    position: float = 0.5,
    axis: str = "x",
    seam_px: int | None = None,
) -> Image.Image:
    """*first* on one side of a movable seam, *second* on the other."""
    array_first = _as_rgb(first)
    array_second = _as_rgb(second, first.size)
    height, width = array_first.shape[:2]
    out = array_first.copy()
    seam = seam_px if seam_px is not None else max(1, int(round(3 * _scale_for((height, width)))))
    line = (245, 245, 245) if is_dark_background(array_first) else (25, 25, 25)
    if axis == "x":
        cut = int(round(min(max(position, 0.0), 1.0) * width))
        out[:, cut:] = array_second[:, cut:]
        out[:, max(0, cut - seam) : cut + seam] = line
    else:
        cut = int(round(min(max(position, 0.0), 1.0) * height))
        out[cut:, :] = array_second[cut:, :]
        out[max(0, cut - seam) : cut + seam, :] = line
    return Image.fromarray(out, mode="RGB")


def _marker_grid(markers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reshape flat ``[ox, oy, nx, ny]`` rows into ``(rows, cols, 2)`` grids.

    Missing nodes stay NaN so a partial grid still draws the segments it has.
    """
    xs = np.unique(markers[:, 0])
    ys = np.unique(markers[:, 1])
    source = np.full((len(ys), len(xs), 2), np.nan)
    target = np.full((len(ys), len(xs), 2), np.nan)
    col = np.searchsorted(xs, markers[:, 0])
    row = np.searchsorted(ys, markers[:, 1])
    source[row, col] = markers[:, 0:2]
    target[row, col] = markers[:, 2:4]
    return source, target


def deformation_grid(
    histology: Image.Image,
    markers: Sequence[Sequence[float]] | np.ndarray,
    *,
    dim: float = 0.55,
    show_reference: bool = True,
) -> Image.Image:
    """The warp drawn as a deformed grid, colored by local distortion.

    *markers* are VisuAlign-style ``[ox, oy, nx, ny]`` rows on a regular
    lattice, in the coordinates of *histology*. The straight faint lattice is
    where space started; the bright one is where it went.

    Color is displacement RELATIVE TO THE MEDIAN displacement, not absolute:
    the registration's global shift moves every node together and would
    otherwise saturate the whole ramp, hiding the local bending that is the
    only thing this view exists to show. The bulk shift is still perfectly
    visible — it is the offset between the two lattices.
    """
    base = _as_rgb(histology)
    height, width = base.shape[:2]
    scale = _scale_for((height, width))
    dark = is_dark_background(base)
    canvas = (base.astype(np.float32) * (1.0 - dim) + (0 if dark else 255) * dim).astype(np.uint8)

    rows = np.asarray(markers, dtype=np.float64).reshape(-1, 4)
    if not len(rows):
        return Image.fromarray(canvas, mode="RGB")
    source, target = _marker_grid(rows)
    displacement = target - source
    bulk = np.nanmedian(displacement.reshape(-1, 2), axis=0)
    magnitude = np.linalg.norm(displacement - bulk, axis=2)
    limit = float(np.nanpercentile(magnitude, 98)) or 1.0
    ramp = _turbo()

    if show_reference:
        faint = (200, 200, 200) if dark else (90, 90, 90)
        straight = [source[index] for index in range(source.shape[0])]
        straight += [source[:, index] for index in range(source.shape[1])]
        canvas = _draw_polys(
            canvas,
            [(faint, [line[~np.isnan(line[:, 0])] for line in straight])],
            thickness=max(1, int(round(1.0 * scale))),
            opacity=0.35,
        )

    thickness = max(1, int(round(2.4 * scale)))
    drawn = canvas.copy()
    for (row, col), _ in np.ndenumerate(magnitude):
        for dr, dc in ((0, 1), (1, 0)):
            nr, nc = row + dr, col + dc
            if nr >= magnitude.shape[0] or nc >= magnitude.shape[1]:
                continue
            start, end = target[row, col], target[nr, nc]
            if np.isnan(start).any() or np.isnan(end).any():
                continue
            level = np.nanmean([magnitude[row, col], magnitude[nr, nc]]) / limit
            color = ramp[int(min(max(level, 0.0), 1.0) * 255)]
            cv2.line(
                drawn,
                tuple(np.round(start * 16).astype(int)),
                tuple(np.round(end * 16).astype(int)),
                (int(color[0]), int(color[1]), int(color[2])),
                thickness,
                cv2.LINE_AA,
                4,
            )
    canvas = cv2.addWeighted(drawn, 0.95, canvas, 0.05, 0.0)

    image = Image.fromarray(canvas, mode="RGB")
    size = max(12, int(round(26 * scale)))
    readout = (
        f"bulk shift {np.linalg.norm(bulk):.0f} px   "
        f"max local distortion {float(np.nanmax(magnitude)):.0f} px"
    )
    return _draw_text(image, [(readout, width * 0.5, height - size, size)], dark_background=dark)


def contact_sheet(
    panels: Sequence[tuple[str, Image.Image]],
    *,
    columns: int = 2,
    cell_px: int = 1100,
    title: str | None = None,
) -> Image.Image:
    """Captioned grid of review panels — the thing a human actually opens."""
    if not panels:
        raise ValueError("contact_sheet needs at least one panel")
    cols = max(1, min(columns, len(panels)))
    rows = (len(panels) + cols - 1) // cols
    thumbs = [(caption, image.convert("RGB")) for caption, image in panels]
    scaled = [
        (
            caption,
            image.resize(
                (
                    max(1, round(image.width * min(cell_px / image.width, cell_px / image.height))),
                    max(
                        1, round(image.height * min(cell_px / image.width, cell_px / image.height))
                    ),
                ),
                resample=Image.Resampling.LANCZOS,
            ),
        )
        for caption, image in thumbs
    ]
    cell_w = max(image.width for _, image in scaled)
    cell_h = max(image.height for _, image in scaled)
    bar = max(20, cell_h // 26)
    top = bar * 2 if title else 0
    sheet = Image.new("RGB", (cols * cell_w, top + rows * (cell_h + bar)), (14, 14, 16))
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((bar // 2, bar // 2), title, font=_font(int(bar * 1.1)), fill=(245, 245, 245))
    for index, (caption, image) in enumerate(scaled):
        col, row = index % cols, index // cols
        x0, y0 = col * cell_w, top + row * (cell_h + bar)
        sheet.paste(image, (x0 + (cell_w - image.width) // 2, y0 + (cell_h - image.height) // 2))
        draw.text(
            (x0 + bar // 3, y0 + cell_h + bar // 6),
            caption,
            font=_font(int(bar * 0.8)),
            fill=(215, 215, 220),
        )
    return sheet
