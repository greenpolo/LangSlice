"""Smooth drawing of a fitted atlas on the ORIGINAL section.

``DeformableRecord.labels`` samples atlas labels nearest-neighbour from the
25 um atlas grid, so borders traced from it are stair-stepped on finer
section pixels. Here the borders are drawn from the final map itself (linear
placement composed with the residual field), the way
:func:`langslice.core.atlas.render.placed_border_coverage` draws a placement:
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

from langslice.core.atlas.render import family_labels
from langslice.core.deformable.atlas_images import native_labels, placement_left, regions_mask
from langslice.core.deformable.record import DeformableRecord

#: Atlas-grid blur of each region indicator before bilinear sampling.
#: 1.0 leaves a ripple along shallow edges (the annotation's own 25 um stair
#: steps); 2.0 closes thin regions into dotted blobs.
BORDER_SMOOTHING_PX = 1.4
STRONG_COLOR = (255, 255, 0)
#: Second ink, for a marked set of regions (those excluded from a fit).
MARKED_COLOR = (255, 64, 160)
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


def warped_border_layers(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), marked: Iterable[str | int] = (),
    warped: bool = True,
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    native: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Border coverage in [0, 1] on the section grid, per layer, tissue-clipped.

    Layers: ``strong`` (edges touching a *highlight* region; every edge when
    nothing is highlighted), ``marked`` (edges touching a *marked* region,
    e.g. the regions excluded from a fit), ``faint`` (every other edge between
    color-family regions) and ``outer`` (edges against the background: the
    atlas's outer boundary). Highlighted and marked leaves keep their own
    identity, so their edges show even inside one color family. With
    ``warped=False`` the residual field is ignored (the linear placement alone).
    """
    s = int(supersample)
    width, height = record.section_size
    leaf = native if native is not None else native_labels(atlas, record.placement)
    highlight, marked = list(highlight), list(marked)
    # One-sided entries ("CTX:left") are sides of the section as drawn here.
    left = placement_left(atlas, record.placement, (*highlight, *marked))
    family = family_labels(leaf, atlas).astype(np.int64)
    flagged = (regions_mask(atlas, leaf, highlight, left) if highlight
               else np.zeros(leaf.shape, dtype=bool))
    flagged_marks = (regions_mask(atlas, leaf, marked, left) if marked
                     else np.zeros(leaf.shape, dtype=bool))
    # Codes: a highlighted leaf is leaf*4+1, a marked leaf leaf*4+3, anything
    # else its color family *4 (background stays 0), so the sets never merge.
    big = leaf.astype(np.int64)
    grouped = np.where(flagged_marks, big * 4 + 3, np.where(flagged, big * 4 + 1, family * 4))
    nx, ny = composed_native_grid(record if warped else _without_field(record), s)
    owner = smooth_owner_map(grouped, nx, ny, smoothing_px=smoothing_px)
    h_edge, v_edge = _owner_edges(owner)
    edges = np.zeros(owner.shape, dtype=bool)
    edges[:, 1:] |= h_edge
    edges[1:, :] |= v_edge

    def touching(flag: np.ndarray) -> np.ndarray:
        touch = np.zeros(owner.shape, dtype=bool)
        touch[:, 1:] |= h_edge & (flag[:, 1:] | flag[:, :-1])
        touch[1:, :] |= v_edge & (flag[1:, :] | flag[:-1, :])
        return touch

    nothing = np.zeros(owner.shape, dtype=bool)
    marked_edges = touching((owner & 3) == 3) if marked else nothing
    strong_edges = (touching((owner & 3) == 1) if highlight else edges) & ~marked_edges
    outer_edges = touching(owner == 0) & ~marked_edges
    fine_tissue = cv2.resize(record.tissue.astype(np.uint8), (width * s, height * s),
                             interpolation=cv2.INTER_NEAREST) > 0
    # One section pixel of slack so a border on the tissue edge itself shows.
    reach = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * s + 1, 2 * s + 1))
    fine_tissue = cv2.dilate(fine_tissue.astype(np.uint8), reach) > 0
    size = (width, height)
    return {
        "faint": _stroke(edges & ~strong_edges & ~marked_edges & fine_tissue, size, s, width_px),
        "strong": _stroke(strong_edges & fine_tissue, size, s, width_px),
        "marked": _stroke(marked_edges & fine_tissue, size, s, width_px),
        "outer": _stroke(outer_edges & fine_tissue, size, s, width_px),
    }


def _without_field(record: DeformableRecord) -> DeformableRecord:
    from dataclasses import replace

    return replace(record, field_mm=np.zeros_like(record.field_mm))


def _drawn_layers(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int], marked: Iterable[str | int], warped: bool, outlines: str,
    width_px: float, smoothing_px: float, supersample: int, native: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(faint, strong, marked)``: the coverage :func:`draw_warped_borders`
    draws, each layer as the *outlines* choice leaves it."""
    if outlines not in ("all", "outer", "none"):
        raise ValueError("outlines must be all, outer or none")
    highlight = list(highlight)
    layers = warped_border_layers(
        record, atlas, highlight=highlight, marked=marked, warped=warped, width_px=width_px,
        smoothing_px=smoothing_px, supersample=supersample, native=native,
    )
    blank = np.zeros_like(layers["faint"])
    if highlight:
        strong = layers["strong"]
        faint = {"all": layers["faint"], "outer": np.minimum(layers["faint"], layers["outer"]),
                 "none": blank}[outlines]
    else:
        faint = blank
        strong = {"all": layers["strong"], "outer": layers["outer"], "none": blank}[outlines]
    return faint, strong, layers["marked"]


def drawn_border_coverage(
    record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), marked: Iterable[str | int] = (),
    warped: bool = True, outlines: str = "all",
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    native: np.ndarray | None = None,
) -> np.ndarray:
    """Every line :func:`draw_warped_borders` draws with these arguments, as
    coverage in [0, 1] at full strength (the faint lines too): a picture's
    borders layer."""
    layers = _drawn_layers(record, atlas, highlight=highlight, marked=marked, warped=warped,
                           outlines=outlines, width_px=width_px, smoothing_px=smoothing_px,
                           supersample=supersample, native=native)
    return np.maximum.reduce(layers)


def draw_warped_borders(
    image: Image.Image, record: DeformableRecord, atlas: Any, *,
    highlight: Iterable[str | int] = (), marked: Iterable[str | int] = (),
    warped: bool = True, outlines: str = "all",
    width_px: float = 2.0, smoothing_px: float = BORDER_SMOOTHING_PX, supersample: int = 3,
    color: tuple[int, int, int] = STRONG_COLOR, marked_color: tuple[int, int, int] = MARKED_COLOR,
    native: np.ndarray | None = None,
) -> Image.Image:
    """The fitted atlas borders, smooth and antialiased, on the original section.

    Without *highlight* every border between color-family regions is drawn in
    *color*. With it, only the selected structures' edges are strong, over a
    faint outline of the rest. *marked* regions' edges are drawn in
    *marked_color* (a second set, e.g. regions excluded from the fit).
    *outlines* limits the other lines: ``all``, ``outer`` (only the atlas's
    outer boundary) or ``none``. Clipped to the tissue.
    """
    faint, strong, marked_lines = _drawn_layers(
        record, atlas, highlight=highlight, marked=marked, warped=warped, outlines=outlines,
        width_px=width_px, smoothing_px=smoothing_px, supersample=supersample, native=native)
    base = np.asarray(image.convert("RGB"), dtype=np.float32)
    if base.shape[:2] != strong.shape:
        raise ValueError("The image must be on the record's section grid")
    for coverage, ink, weight in (
        (faint, color, FAINT_ALPHA), (strong, color, 1.0), (marked_lines, marked_color, 1.0),
    ):
        alpha = (weight * coverage)[..., None]
        base = base * (1.0 - alpha) + np.array(ink, dtype=np.float32) * alpha
    return Image.fromarray(np.rint(base).astype(np.uint8))


def resampled_record(
    record: DeformableRecord, size: tuple[int, int],
    box: tuple[int, int, int, int] | None = None,
) -> DeformableRecord:
    """The same deformation on a resized (and optionally cropped) section grid.

    For drawing at picture size: the field is in millimetres, so it is only
    resampled; the placement is carried onto the new pixel centres and the
    tissue mask resized. *box* (x0, y0, x1, y1 on the resized grid) crops.
    Only what drawing reads is carried: no inverse, labels or parent.
    """
    from dataclasses import replace

    from langslice.core.affine import pixel_center_map

    width, height = record.section_size
    field = np.stack([
        cv2.resize(np.ascontiguousarray(record.field_mm[..., k], dtype=np.float32), size,
                   interpolation=cv2.INTER_LINEAR) for k in range(2)
    ], axis=-1)
    tissue = cv2.resize(record.tissue.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0
    matrix = pixel_center_map((width, height), size) @ record.placement.atlas_to_section
    mm = record.mm_per_px * width / float(size[0])
    if box is not None:
        x0, y0, x1, y1 = box
        field, tissue = field[y0:y1, x0:x1], tissue[y0:y1, x0:x1]
        matrix = np.array([[1.0, 0.0, -x0], [0.0, 1.0, -y0], [0.0, 0.0, 1.0]]) @ matrix
        size = (x1 - x0, y1 - y0)
    placement = replace(record.placement, atlas_to_section=matrix, section_mm_per_px=mm)
    return replace(
        record, placement=placement, section_size=size, field_mm=field,
        inverse_field_mm=None, labels=np.zeros(tissue.shape, dtype=record.labels.dtype),
        tissue=tissue, torn_band=np.zeros(tissue.shape, dtype=bool), parent=None,
    )


def warp_section_image(image: Image.Image, record: DeformableRecord) -> Image.Image:
    """The section resampled into its placed-atlas frame by the residual warp.

    *image* is the section on the record's frame at any size (the same
    oriented, unframed render the fit grid is a resize of). Pixel q of the
    result shows section point ``q + inverse_field(q)``, so drawing the result
    under the LINEAR placement shows the full registration: the placement's
    atlas lines then fall where the warp put them on the tissue. Without a
    stored inverse one is approximated (fixed point) at picture size.
    """
    from langslice.core.deformable.engines import invert_field

    width, height = image.size
    mm = record.mm_per_px * record.section_size[0] / float(width)

    def resized(field: np.ndarray) -> np.ndarray:
        return np.stack([
            cv2.resize(np.ascontiguousarray(field[..., k], dtype=np.float32), (width, height),
                       interpolation=cv2.INTER_LINEAR) for k in range(2)
        ], axis=-1)

    if record.inverse_field_mm is not None:
        inverse = resized(record.inverse_field_mm)
    else:
        inverse, _ = invert_field(resized(record.field_mm), (mm, mm))
    yy, xx = np.indices((height, width), dtype=np.float32)
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    out = cv2.remap(pixels, xx + inverse[..., 0] / mm, yy + inverse[..., 1] / mm,
                    cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return Image.fromarray(out)
