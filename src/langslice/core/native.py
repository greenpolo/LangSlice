"""Sections at their image file's own resolution, a window at a time: what a zoom shows.

Every other picture is drawn from the section's working copy (at most
``image_prep.WORKING_MAX_EDGE`` long). A zoom on a section (``look``'s
``section`` and ``overlay`` pictures, and every change tool's picture) is
drawn again from the image file itself: :func:`section_pixels` takes, for
each pixel of the zoomed picture, the working-copy point it shows (the
caller's geometry: framing, quarter turn and flip, placement, deformation)
and returns the file's pixels there, read only around those points
(:func:`read_region`: a tiled or pyramidal TIFF through tifffile's zarr
store, never decoded whole; any other file decoded and cut).

The colours are the picture's own. A look (the raw channels with their
display settings, the default appearance, the preprocessed channel) is a
function of the whole section, CLAHE included, so it is not recomputed on a
window. Instead it is drawn on the whole working copy as every unzoomed
picture draws it, and carried to the file's pixels by a local linear fit
(a guided filter, He, Sun and Tang 2013): around each working-copy pixel
the look is fitted as an offset plus a gain on the raw channels, and the
fit is applied to the raw channels read from the file. A look that is a
plain map of the raw channels (the default raw display) is carried exactly;
CLAHE's local contrast is carried at the working copy's scale, and the
file's finer detail comes through at that contrast. Where the working copy
is too flat to fit, the gain falls back to the one fitted over the whole
window, never to zero.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.appearance import Look
from langslice.core.image_prep import IntensityRange, read_pages
from langslice.core.state import SliceState
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: Radius (working-copy pixels) of the windows the look is fitted over.
FIT_RADIUS = 4
#: Regularization of the local fit (channel values 0..1): where the working
#: copy varies less than about its square root, the gain tends to the one
#: fitted over the whole region.
FIT_EPS = 1e-3
#: Margin (working-copy pixels) read and fitted around the points asked for.
MARGIN = 3 * FIT_RADIUS
#: How many file pixels a region read may hold before it is read at a stride.
MAX_READ_PIXELS = 48_000_000

_ROTATIONS = {90: Image.Transpose.ROTATE_270, 180: Image.Transpose.ROTATE_180,
              270: Image.Transpose.ROTATE_90}


# --- orientation ------------------------------------------------------------------------


def unoriented_points(
    x: np.ndarray, y: np.ndarray, size: tuple[int, int], rotation_deg: int, flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Points of a render turned by *rotation_deg* (counter-clockwise, PIL's
    quarter turns) then flipped left-right, of *size* ``(width, height)``,
    as points of the same render before both (pixel centres at integers)."""
    width, height = size
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if flip:
        x = (width - 1) - x
    turn = int(rotation_deg) % 360
    if turn == 90:
        return (height - 1) - y, x
    if turn == 180:
        return (width - 1) - x, (height - 1) - y
    if turn == 270:
        return y, (width - 1) - x
    return x, y


def unoriented_fractions(
    fx: np.ndarray, fy: np.ndarray, rotation_deg: int, flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`unoriented_points` for fractions of the picture (0..1 edge to edge)."""
    fx = np.asarray(fx, dtype=np.float64)
    fy = np.asarray(fy, dtype=np.float64)
    if flip:
        fx = 1.0 - fx
    turn = int(rotation_deg) % 360
    if turn == 90:
        return 1.0 - fy, fx
    if turn == 180:
        return 1.0 - fx, 1.0 - fy
    if turn == 270:
        return fy, 1.0 - fx
    return fx, fy


def unorient(image: Image.Image, rotation_deg: int, flip: bool) -> Image.Image:
    """*image* (a render turned then flipped) as it was before both."""
    if flip:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    turn = _ROTATIONS.get(int(rotation_deg) % 360)
    return image.transpose(turn) if turn is not None else image


# --- reading a window of the file -------------------------------------------------------------


@dataclass(frozen=True)
class Region:
    """A window of an image file as read: its pages (each ``(rows, cols)`` or
    ``(rows, cols, samples)``, the file's own sample type), and where they
    sit: read pixel ``(j, i)`` is level pixel ``(origin + (j, i) * stride)``,
    a level pixel spanning ``downsample`` file pixels."""

    pages: list[np.ndarray]
    file_size: tuple[int, int]
    origin: tuple[int, int]
    stride: int
    downsample: float

    def to_read(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """File pixel coordinates (centres at integers) as read-array coordinates."""
        d = self.downsample
        rx = ((np.asarray(x) + 0.5) / d - 0.5 - self.origin[0]) / self.stride
        ry = ((np.asarray(y) + 0.5) / d - 0.5 - self.origin[1]) / self.stride
        return rx, ry


def read_region(path: str | Path, box: tuple[int, int, int, int], step: int = 1) -> Region:
    """The file pixels in *box* ``(x0, y0, x1, y1)``, about one read pixel
    per *step* file pixels: a TIFF (plain, tiled or pyramidal, one or several
    pages) through tifffile's zarr store, from the coarsest pyramid level
    still at least that fine, only the tiles or strips the box touches
    decoded; any other file (or a TIFF layout this does not read) decoded
    whole and cut. Pages as :func:`langslice.core.image_prep.read_pages`
    gives them."""
    step = max(1, int(step))
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        try:
            found = _read_tiff_region(path, box, step)
        except Exception as exc:  # noqa: BLE001 - an odd layout: decode it whole
            logger.debug("windowed read of %s failed (%s); decoding it whole", path, exc)
            found = None
        if found is not None:
            return found
    pages = read_pages(path)
    height, width = pages[0].shape[:2]
    x0, y0, x1, y1 = _clamped(box, (width, height))
    stride = _stride((x1 - x0) * (y1 - y0) * len(pages), step)
    return Region(pages=[page[y0:y1:stride, x0:x1:stride] for page in pages],
                  file_size=(width, height), origin=(x0, y0), stride=stride, downsample=1.0)


def _clamped(box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, x1 = max(0, min(x0, width - 1)), max(1, min(x1, width))
    y0, y1 = max(0, min(y0, height - 1)), max(1, min(y1, height))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def _stride(pixels: int, step: int) -> int:
    """A stride for reading *pixels* asked at *step*: half the step (so the
    caller can average the rest away), more when the read would be too big."""
    stride = max(1, step // 2)
    while pixels / float(stride * stride) > MAX_READ_PIXELS:
        stride += 1
    return stride


def _read_tiff_region(
    path: str | Path, box: tuple[int, int, int, int], step: int,
) -> Region | None:
    import tifffile
    import zarr

    with tifffile.TiffFile(path) as handle:
        if not handle.series:
            return None
        series = handle.series[0]
        axes = series.axes
        count = len(handle.pages)
        if axes in ("YX", "YXS"):
            many = False
        elif len(axes) == 3 and axes.endswith("YX") and count > 1 and series.shape[0] == count:
            many = True
        else:
            return None
        levels = list(series.levels)
        full = (int(series.shape[axes.index("X")]), int(series.shape[axes.index("Y")]))
        chosen, downsample = 0, 1.0
        for index, level in enumerate(levels):
            ratio = full[0] / float(level.shape[axes.index("X")])
            if ratio <= step + 1e-6 and ratio >= downsample:
                chosen, downsample = index, ratio
        x0, y0, x1, y1 = _clamped(box, full)
        lx0, ly0 = int(math.floor(x0 / downsample)), int(math.floor(y0 / downsample))
        lx1, ly1 = int(math.ceil(x1 / downsample)), int(math.ceil(y1 / downsample))
        rest = max(1, int(step / downsample))
        stride = _stride((lx1 - lx0) * (ly1 - ly0) * (count if many else 1), rest)
        store = series.aszarr(level=chosen)
        try:
            array: Any = zarr.open(store, mode="r")
            rows, cols = slice(ly0, ly1, stride), slice(lx0, lx1, stride)
            if many:
                data = np.asarray(array[:, rows, cols])
                pages = [np.asarray(plane) for plane in data]
            elif axes == "YXS":
                pages = [np.asarray(array[rows, cols, :])]
            else:
                pages = [np.asarray(array[rows, cols])]
        finally:
            store.close()
    return Region(pages=pages, file_size=full, origin=(lx0, ly0), stride=stride,
                  downsample=downsample)


# --- the raw channels of a window --------------------------------------------------------------


def _mapped(page: np.ndarray, held: IntensityRange) -> np.ndarray:
    """A page's samples as its working planes hold them (0..255, float)."""
    if page.dtype == np.bool_:
        return page.astype(np.float32) * 255.0
    if page.dtype == np.uint8:
        return page.astype(np.float32)
    values = (page.astype(np.float32) - held.low) / max(held.high - held.low, 1e-12) * 255.0
    return np.clip(values, 0.0, 255.0)


def window_planes(
    pages: list[np.ndarray], names: tuple[str, ...], ranges: dict[str, IntensityRange],
) -> list[np.ndarray]:
    """The raw channels *names* of a window, ``0..1`` float32, read as
    :func:`langslice.core.image_prep.channel_planes` reads the working copy
    (each page through the stretch its working plane was read with)."""
    if len(pages) == 1:
        page = np.asarray(pages[0])
        held = ranges[names[0]]
        if page.ndim == 2:
            return [_mapped(page, held) / 255.0]
        rgb = _mapped(page[..., :3], held) / 255.0
        if names == ("gray",):
            return [rgb[..., 0]]
        return [rgb[..., index] for index in range(3)]
    planes: list[np.ndarray] = []
    for page, name in zip(pages, names, strict=False):
        page = np.asarray(page)
        mapped = _mapped(page if page.ndim == 2 else page[..., :3], ranges[name]) / 255.0
        if mapped.ndim == 3:  # a colour page counts as its luminance
            mapped = mapped @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        planes.append(mapped.astype(np.float32))
    return planes


# --- carrying a look to the file's pixels ----------------------------------------------------


def _box(image: np.ndarray, radius: int) -> np.ndarray:
    size = 2 * radius + 1
    return cv2.boxFilter(np.ascontiguousarray(image, dtype=np.float32), -1, (size, size),
                         normalize=True, borderType=cv2.BORDER_REFLECT)


def fit_look(target: np.ndarray, guide: np.ndarray, *, radius: int = FIT_RADIUS,
             eps: float = FIT_EPS) -> tuple[np.ndarray, np.ndarray]:
    """``(offset (rows, cols, 3), gain (rows, cols, k, 3))``: *target* (the
    look, ``0..1``) as an offset plus a gain on *guide* (the k raw channels,
    ``0..1``) around every pixel, then averaged over the same windows (a
    colour guided filter). The gain is regularized towards the one fitted
    over the whole region, so a flat window keeps the region's contrast."""
    rows, cols, k = guide.shape
    g = guide.reshape(-1, k).astype(np.float64)
    t = target.reshape(-1, target.shape[2]).astype(np.float64)
    g_mean, t_mean = g.mean(axis=0), t.mean(axis=0)
    cov_all = np.cov(g, rowvar=False).reshape(k, k) + 1e-6 * np.eye(k)
    cross_all = ((g - g_mean).T @ (t - t_mean)) / max(len(g) - 1, 1)
    prior = np.linalg.solve(cov_all, cross_all)  # (k, 3)

    mean_g = np.stack([_box(guide[..., i], radius) for i in range(k)], axis=-1)
    mean_t = np.stack([_box(target[..., c], radius) for c in range(target.shape[2])], axis=-1)
    cov = np.empty((rows, cols, k, k), dtype=np.float32)
    for i in range(k):
        for j in range(i, k):
            value = _box(guide[..., i] * guide[..., j], radius) - mean_g[..., i] * mean_g[..., j]
            cov[..., i, j] = value
            cov[..., j, i] = value
    cross = np.empty((rows, cols, k, target.shape[2]), dtype=np.float32)
    for i in range(k):
        for c in range(target.shape[2]):
            cross[..., i, c] = (_box(guide[..., i] * target[..., c], radius)
                                - mean_g[..., i] * mean_t[..., c])
    eye = np.eye(k, dtype=np.float32)
    gain = np.linalg.solve(cov + eps * eye, cross + eps * prior.astype(np.float32))
    offset = mean_t - np.einsum("rck,rckd->rcd", mean_g, gain)
    gain = np.stack([np.stack([_box(gain[..., i, c], radius) for c in range(gain.shape[3])],
                              axis=-1) for i in range(k)], axis=-2)
    offset = np.stack([_box(offset[..., c], radius) for c in range(offset.shape[2])], axis=-1)
    return offset.astype(np.float32), gain.astype(np.float32)


def working_look(ws: Workspace, record: SliceState, look: Look) -> np.ndarray:
    """The section in *look* on its whole working copy, before its quarter
    turn and flip: ``(rows, cols, 3)`` ``0..1``, as an unzoomed picture of it
    shows it (the render is cached)."""
    from langslice.core.sections import render_slice

    source, _factor = ws.working_source(record.id)
    drawn = render_slice(ws, record, long_edge=max(source.size), frame=False, look=look)
    flat = unorient(drawn, record.rotation_deg, record.flip).convert("RGB")
    if flat.size != source.size:  # never expected: the render is the working copy
        flat = flat.resize(source.size, Image.Resampling.BILINEAR)
    return np.asarray(flat, dtype=np.float32) / 255.0


@dataclass(frozen=True)
class NativePixels:
    """:func:`section_pixels`' answer: the RGB pixels (uint8) asked for, which
    of them lie on the section's image, and how many picture pixels one file
    pixel spans there (above 1: the file has fewer pixels than the picture)."""

    rgb: np.ndarray
    inside: np.ndarray
    enlarged: float


def section_pixels(
    ws: Workspace, record: SliceState, look: Look, x: np.ndarray, y: np.ndarray,
) -> NativePixels:
    """The section's pixels at working-copy points ``(x, y)`` (arrays of the
    picture's shape; the working copy before its quarter turn and flip,
    pixel centres at integers), read from its image file at the file's own
    resolution and shown in *look* (see the module text). Raises when the
    file cannot be read."""
    from langslice.core.channels import intensity_ranges

    source, _factor = ws.working_source(record.id)
    width, height = source.size
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    inside = (x >= -0.5) & (x <= width - 0.5) & (y >= -0.5) & (y <= height - 0.5)
    if not inside.any():
        return NativePixels(np.zeros((*x.shape, 3), dtype=np.uint8), inside, 1.0)
    # The working-copy box the points need, with the fit's margin.
    bx0 = max(0, int(math.floor(x[inside].min())) - MARGIN)
    by0 = max(0, int(math.floor(y[inside].min())) - MARGIN)
    bx1 = min(width, int(math.ceil(x[inside].max())) + MARGIN + 1)
    by1 = min(height, int(math.ceil(y[inside].max())) + MARGIN + 1)

    # The look, fitted on the raw channels over that box.
    names, planes = ws.section_channels(record.id)
    guide = np.stack([_at_size(np.asarray(plane), (width, height))[by0:by1, bx0:bx1]
                      for plane in planes], axis=-1).astype(np.float32) / 255.0
    target = working_look(ws, record, look)[by0:by1, bx0:bx1]
    offset, gain = fit_look(target, guide)

    # The file's pixels around the points.
    path = ws.image_path(record.id)
    file_w, file_h = _file_size(path)
    fx, fy = file_w / float(width), file_h / float(height)
    px, py = (x + 0.5) * fx - 0.5, (y + 0.5) * fy - 0.5
    spacing = _spacing(px, py, inside)
    step = max(1, int(math.floor(spacing)))
    box = (int(math.floor(bx0 * fx)) - 2, int(math.floor(by0 * fy)) - 2,
           int(math.ceil(bx1 * fx)) + 2, int(math.ceil(by1 * fy)) + 2)
    region = read_region(path, box, step)
    raw = window_planes(region.pages, tuple(names), intensity_ranges(ws, record.id))
    rx, ry = region.to_read(px, py)
    # Average the read pixels down to about one per picture pixel first.
    shrink = spacing / (region.downsample * region.stride)
    if shrink > 1.5:
        rows, cols = raw[0].shape[:2]
        size = (max(1, round(cols / shrink)), max(1, round(rows / shrink)))
        raw = [cv2.resize(plane, size, interpolation=cv2.INTER_AREA) for plane in raw]
        rx = (rx + 0.5) * size[0] / float(cols) - 0.5
        ry = (ry + 0.5) * size[1] / float(rows) - 0.5
    mx, my = rx.astype(np.float32), ry.astype(np.float32)
    fine = np.stack([cv2.remap(np.ascontiguousarray(plane, dtype=np.float32), mx, my,
                               cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                     for plane in raw], axis=-1)

    # The fit, at the points, applied to the file's channels.
    wx, wy = (x - bx0).astype(np.float32), (y - by0).astype(np.float32)

    def at_points(values: np.ndarray) -> np.ndarray:
        return cv2.remap(np.ascontiguousarray(values, dtype=np.float32), wx, wy,
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    k = guide.shape[2]
    out = np.stack([at_points(offset[..., c]) for c in range(3)], axis=-1)
    for i in range(k):
        for c in range(3):
            out[..., c] += at_points(gain[..., i, c]) * fine[..., i]
    rgb = (np.clip(out, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return NativePixels(rgb=rgb, inside=inside, enlarged=1.0 / max(spacing, 1e-9))


def framed_window(
    ws: Workspace, record: SliceState, look: Look, window: tuple[float, ...], long_edge: int,
) -> tuple[Image.Image, float]:
    """``(picture, enlarged)``: *window* (``[x0, y0, x1, y1]`` fractions of
    the section's tissue-framed picture, turned and flipped as it is shown)
    read from the image file, its long side *long_edge* pixels, in *look*
    (:func:`section_pixels`); *enlarged*, how many picture pixels one file
    pixel spans (above 1: the file has fewer pixels than the picture)."""
    from langslice.core.image_prep import tissue_box

    source, _factor = ws.working_source(record.id)
    width, height = source.size
    bx0, by0, bx1, by1 = tissue_box(source) or (0, 0, width, height)
    box_w, box_h = bx1 - bx0, by1 - by0
    x0, y0, x1, y1 = (float(v) for v in window)
    corners_x, corners_y = unoriented_fractions(np.array([x0, x1]), np.array([y0, y1]),
                                                record.rotation_deg, record.flip)
    file_w, file_h = _file_size(ws.image_path(record.id))
    native_w = abs(corners_x[1] - corners_x[0]) * box_w * file_w / float(width)
    native_h = abs(corners_y[1] - corners_y[0]) * box_h * file_h / float(height)
    if int(record.rotation_deg) % 180:
        native_w, native_h = native_h, native_w
    scale = long_edge / max(native_w, native_h, 1e-9)
    cols, rows = max(1, round(native_w * scale)), max(1, round(native_h * scale))
    fx = x0 + (np.arange(cols, dtype=np.float64) + 0.5) / cols * (x1 - x0)
    fy = y0 + (np.arange(rows, dtype=np.float64) + 0.5) / rows * (y1 - y0)
    grid_x, grid_y = np.meshgrid(fx, fy)
    ux, uy = unoriented_fractions(grid_x, grid_y, record.rotation_deg, record.flip)
    found = section_pixels(ws, record, look, bx0 + ux * box_w - 0.5, by0 + uy * box_h - 0.5)
    return Image.fromarray(found.rgb), found.enlarged


def _at_size(plane: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (plane.shape[1], plane.shape[0]) == size:
        return plane
    return np.asarray(Image.fromarray(plane).resize(size, Image.Resampling.LANCZOS))


def _file_size(path: str) -> tuple[int, int]:
    """The image file's full-resolution ``(width, height)``, from its header."""
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            if handle.series:
                series = handle.series[0]
                return (int(series.shape[series.axes.index("X")]),
                        int(series.shape[series.axes.index("Y")]))
    with Image.open(path) as handle:
        return int(handle.size[0]), int(handle.size[1])


def _spacing(px: np.ndarray, py: np.ndarray, inside: np.ndarray) -> float:
    """File pixels between neighbouring picture pixels (the median, along
    both picture axes)."""
    steps: list[np.ndarray] = []
    if px.shape[1] > 1:
        keep = inside[:, 1:] & inside[:, :-1]
        steps.append(np.hypot(np.diff(px, axis=1), np.diff(py, axis=1))[keep])
    if px.shape[0] > 1:
        keep = inside[1:, :] & inside[:-1, :]
        steps.append(np.hypot(np.diff(px, axis=0), np.diff(py, axis=0))[keep])
    joined = np.concatenate(steps) if steps else np.array([1.0])
    return float(np.median(joined)) if joined.size else 1.0


__all__ = [
    "FIT_EPS", "FIT_RADIUS", "NativePixels", "Region", "fit_look", "framed_window", "read_region",
    "section_pixels", "unorient", "unoriented_fractions", "unoriented_points",
    "window_planes", "working_look",
]
