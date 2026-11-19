"""Where the tissue is in a picture of a section: the one segmentation.

A scanned slide is the section's tissue on a ground: the scanner's tile floor
and black margin in fluorescence, bright glass in brightfield. The tissue is
what the window cuts a section out to (its pictures' ground, the 3D view's
slabs), so the cut is made once, here, for every reader.

:func:`stain_of` turns a picture into the signal the cut reads: brightness for
a darkfield section, darkness under the slide's white for a brightfield one
(:func:`is_brightfield`, :class:`SlideTest`). :func:`tissue_mask` finds the
tissue in that signal in steps, each with its thresholds in a frozen
dataclass: :class:`Segmentation` (the tissue is textured and stands above the
scanner's floor, then its edge is closed, filled and feathered),
:class:`DebrisRules` (small or out-of-focus pieces apart from the tissue and
pieces at the picture's border are left out), :class:`ScanEdgeRules` (the
scanner's bright line and smears along the edge of what it scanned are left
out) and :class:`GlareFilter` (out-of-focus bright patches that stand out of
the tissue are left out). :func:`tissue_alpha` is the whole cut on a picture.

Nothing here changes a picture a fit reads: the picture a fit measures is the
core's own ``preprocessed`` image, and the thresholds that read the tissue for
it (:mod:`langslice.core.sections.reading`) mean something else.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import cv2
import numpy as np
from PIL import Image, ImageFilter
from SimpleITK import (
    GetArrayFromImage,
    GetImageFromArray,
    SmoothingRecursiveGaussian,
)

if TYPE_CHECKING:
    import numpy.typing as npt

GREY_MAX = 255.0


@dataclass(frozen=True)
class SlideTest:
    """Tells a brightfield slide (dark stain on bright glass) from darkfield.

    A picture is brightfield when the median of its scanned pixels, and of the
    rim of what was scanned (the glass around the tissue), is over
    *bright_median*. A pixel at or under *unscanned_level* (grey, 0 to 255)
    is the black ground the scanner left around its tiles, not scanned.
    *rim_px* is the width of the scanned region's rim. A brightfield section is
    read by its darkness under the slide's white, the *white_percentile* of
    its scanned pixels.
    """

    unscanned_level: float = 4.0
    bright_median: float = 140.0
    rim_px: int = 3
    white_percentile: float = 92.0


@dataclass(frozen=True)
class Segmentation:
    """The tissue as a textured region standing above the scanner's floor.

    The picture is cut on a copy at most *work_edge_px* long. The floor is the
    *floor_percentile* of the picture blurred (a Gaussian of *floor_blur_px*
    at full size, *blur_margin_px* more at any size) over its scanned pixels,
    or *floor_default* without any; a pixel is bright when it is over
    *floor_factor* times the floor and over *floor_minimum*. The tissue's grain
    is the picture's difference from itself blurred by *fine_blur_px*, itself
    blurred over *grain_blur_px*; a bright pixel is tissue where the grain is
    over *grain_fraction* of the bright pixels' median and over
    *grain_minimum*. A piece is kept when it covers *piece_fraction* of the
    copy and *piece_minimum_px* pixels. The kept tissue is closed by
    *close_px*, its holes filled, grown by *grow_px*, and its edge feathered
    by a Gaussian of *feather_px* at the picture's own size.
    """

    work_edge_px: int = 512
    floor_blur_px: float = 5.0
    blur_margin_px: float = 1.0
    floor_percentile: float = 30.0
    floor_default: float = 8.0
    floor_factor: float = 1.6
    floor_minimum: float = 12.0
    fine_blur_px: float = 1.0
    grain_blur_px: float = 4.0
    grain_fraction: float = 0.12
    grain_minimum: float = 0.6
    piece_fraction: float = 0.004
    piece_minimum_px: int = 4
    close_px: int = 3
    grow_px: int = 2
    feather_px: float = 2.0


@dataclass(frozen=True)
class DebrisRules:
    """What lies on the slide beside the tissue and is left out of it.

    A piece apart from the largest goes when it is under *part_fraction* of
    the largest and either blurred (its grain for its brightness under
    *blur_fraction* of the tissue's) or touching the scan's edge, or when it
    reaches within *border_band* (a fraction of the shorter side) of the
    picture's border.
    """

    part_fraction: float = 0.08
    blur_fraction: float = 0.5
    border_band: float = 0.02


@dataclass(frozen=True)
class ScanEdgeRules:
    """The scanner's bright line and smears along the edge of what it scanned.

    The edge band is *band_fraction* of the shorter side wide along the
    picture's border and along the black ground the scanner left. Blurred
    tissue reaching the band goes with *glow_rim_px* around it; tissue thinner
    than *thread_px* reaching it goes, *thread_fringe_px* of the thicker
    tissue's foot kept.
    """

    band_fraction: float = 0.03
    thread_px: int = 12
    thread_fringe_px: int = 2
    glow_rim_px: int = 2


@dataclass(frozen=True)
class GlareFilter:
    """Out-of-focus bright patches that stand out of the tissue.

    A patch is glare where its grain for its brightness is under
    *sharp_fraction* of the tissue's and it is over *bright_factor* times the
    tissue's brightness, and covers *area_fraction* of the picture. It goes,
    with *ring_px* around it, when over *apart_fraction* of that ring is not
    tissue, or when over *out_fraction* of it is not and over *halo_fraction*
    of that part still glows. A blurred patch inside the tissue stays.
    """

    sharp_fraction: float = 0.55
    bright_factor: float = 1.15
    area_fraction: float = 0.0004
    ring_px: int = 6
    out_fraction: float = 0.08
    halo_fraction: float = 0.3
    apart_fraction: float = 0.5


SLIDE = SlideTest()
SEGMENTATION = Segmentation()
DEBRIS = DebrisRules()
SCAN_EDGE = ScanEdgeRules()
GLARE = GlareFilter()


@dataclass(frozen=True)
class Stain:
    """The signal the tissue is cut from, and the slide it came from.

    *signal* is 0 to 255, float32, the picture's size: brightness for a
    darkfield section; for a brightfield one how far its darkest channel
    stands under the slide's white (0 where nothing was scanned). *brightfield*
    says which, and *balanced* is a brightfield picture divided by the white
    of its slide in each channel (rows by columns by three, 0 to 1; None for
    darkfield).
    """

    signal: npt.NDArray[np.float32]
    brightfield: bool
    balanced: npt.NDArray[np.float32] | None = None


def is_brightfield(picture: Image.Image, slide: SlideTest = SLIDE) -> bool:
    """Tell a brightfield slide (dark stain on bright glass) from darkfield.

    Parameters
    ----------
    picture
        A section's picture.
    slide
        The thresholds of the test.

    Returns
    -------
    bool
        Whether the median of its scanned pixels, and of the rim of what was
        scanned, is over ``slide.bright_median``.
    """
    grey = np.asarray(picture.convert("L")).astype(np.float32)
    scanned = (grey > slide.unscanned_level).astype(np.uint8)
    if not scanned.any():
        return False
    rim_px = slide.rim_px
    padded = np.pad(scanned, rim_px)
    core = np.asarray(cv2.erode(padded, disk(rim_px)), dtype=np.uint8)
    rim = (scanned > 0) & (core[rim_px:-rim_px, rim_px:-rim_px] < 1)
    return (
        float(np.median(grey[scanned > 0])) > slide.bright_median
        and float(np.median(grey[rim])) > slide.bright_median
    )


def stain_of(picture: Image.Image, slide: SlideTest = SLIDE) -> Stain:
    """Read a picture as the signal its tissue is cut from.

    Parameters
    ----------
    picture
        A section's picture, without atlas borders, caption or scale bar.
    slide
        The brightfield test's thresholds.

    Returns
    -------
    Stain
        Its grey, or for a brightfield slide how far its darkest channel
        stands under the white of the scanned glass in that channel
        (``slide.white_percentile``).
    """
    grey = np.asarray(picture.convert("L")).astype(np.float32)
    if not is_brightfield(picture, slide):
        return Stain(grey, False)
    rgb = np.asarray(picture.convert("RGB")).astype(np.float32)
    scanned = grey > slide.unscanned_level
    white = np.percentile(rgb[scanned], slide.white_percentile, axis=0)
    balanced = np.clip(rgb / np.maximum(white, 1.0), 0, 1).astype(np.float32)
    darkest = cast("npt.NDArray[np.float32]", np.min(balanced, axis=2))
    dark = np.where(scanned, (1.0 - darkest) * GREY_MAX, 0.0).astype(np.float32)
    return Stain(dark, True, balanced)


def tissue_alpha(picture: Image.Image) -> npt.NDArray[np.float32]:
    """Cut a section's tissue out of its picture.

    Parameters
    ----------
    picture
        A section's picture, without atlas borders, caption or scale bar.

    Returns
    -------
    np.ndarray
        The tissue's weight, 0 to 1, float32, the picture's size, its edge
        feathered (``Segmentation.feather_px``).
    """
    return tissue_mask(stain_of(picture).signal)


def tissue_mask(
    grey: npt.NDArray[np.float32], cut: Segmentation = SEGMENTATION
) -> npt.NDArray[np.float32]:
    """Find the tissue: textured and brighter than the scanner's tile floor.

    Parameters
    ----------
    grey
        The picture's one channel, 0 to 255.
    cut
        The segmentation's thresholds.

    Returns
    -------
    np.ndarray
        The tissue's weight, 0 to 1, its edge feathered, the picture's size.
    """
    height, width = cast("tuple[int, int]", grey.shape)
    scale = min(1.0, cut.work_edge_px / max(height, width))
    small = grey
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        small = np.asarray(
            Image.fromarray(grey.astype(np.uint8)).resize(size, Image.Resampling.BILINEAR)
        ).astype(np.float32)
    image = GetImageFromArray(small)
    wide = cut.floor_blur_px * scale + cut.blur_margin_px
    smooth = GetArrayFromImage(SmoothingRecursiveGaussian(image, [wide, wide]))
    values = smooth[smooth > SLIDE.unscanned_level]
    floor = float(np.percentile(values, cut.floor_percentile)) if values.size else cut.floor_default
    bright = smooth > max(cut.floor_minimum, floor * cut.floor_factor)
    fine = GetArrayFromImage(
        SmoothingRecursiveGaussian(image, [cut.fine_blur_px, cut.fine_blur_px])
    )
    spread = cut.grain_blur_px * scale + cut.blur_margin_px
    texture = GetArrayFromImage(
        SmoothingRecursiveGaussian(GetImageFromArray(np.abs(small - fine)), [spread, spread])
    )
    level = (
        max(cut.grain_minimum, cut.grain_fraction * float(np.median(texture[bright])))
        if bright.any()
        else cut.grain_minimum
    )
    binary = (bright & (texture > level)).astype(np.uint8)
    _, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = np.asarray(stats, dtype=np.int64)[:, cv2.CC_STAT_AREA]
    large = areas >= max(cut.piece_minimum_px, int(small.size * cut.piece_fraction))
    large[0] = False
    keep = without_debris(
        np.asarray(labels, dtype=np.int64),
        large,
        smooth,
        texture,
        bright,
        scan_edge(small),
    )
    keep = np.asarray(cv2.morphologyEx(keep, cv2.MORPH_CLOSE, disk(cut.close_px)), dtype=np.uint8)
    keep = fill_holes(keep)
    keep = np.asarray(cv2.dilate(keep, disk(cut.grow_px)), dtype=np.uint8)
    weight = Image.fromarray(keep * np.uint8(GREY_MAX))
    if scale < 1.0:
        weight = weight.resize((width, height), Image.Resampling.BILINEAR)
    weight = weight.filter(ImageFilter.GaussianBlur(cut.feather_px))
    return np.asarray(weight).astype(np.float32) / GREY_MAX


def without_debris(
    labels: npt.NDArray[np.int64],
    large: npt.NDArray[np.bool_],
    smooth: npt.NDArray[np.float32],
    texture: npt.NDArray[np.float32],
    bright: npt.NDArray[np.bool_],
    edge: npt.NDArray[np.bool_],
    debris: DebrisRules = DEBRIS,
    glare_rules: GlareFilter = GLARE,
) -> npt.NDArray[np.uint8]:
    """Keep the tissue's pieces and leave out what lies on the slide beside it.

    Dirt, bubbles and glare are out of focus: bright, with little of the
    tissue's grain. A piece apart from the largest goes when it is small and
    either blurred or at the scan's edge, or reaches the picture's border,
    before and after the scan edge's glow and threads are taken off; a
    blurred bright patch goes when it stands out of the tissue (most of its
    surround is not tissue, or part of it is and that part still glows). A
    blurred patch inside the tissue stays: it is over tissue.

    Parameters
    ----------
    labels
        Each pixel's piece, zero the ground.
    large
        Which pieces are large enough to be tissue.
    smooth
        The picture, smoothed.
    texture
        The picture's fine grain, smoothed.
    bright
        Where the picture stands above the scanner's floor.
    edge
        The band along the edge of the scan.
    debris
        The rules for pieces apart from the tissue.
    glare_rules
        The filter for glare patches.

    Returns
    -------
    np.ndarray
        One on the tissue, zero off it.
    """
    pieces = cast("list[int]", np.flatnonzero(large).tolist())
    if not pieces:
        return large[labels].astype(np.uint8)
    pixels = cast("list[int]", np.bincount(labels.ravel()).tolist())
    main = max(pieces, key=lambda piece: pixels[piece])
    sharp = (texture / (smooth + 1.0)).astype(np.float32)
    body = cast("npt.NDArray[np.bool_]", labels == main)
    focus = float(np.median(sharp[body]))
    glow = float(np.median(smooth[body]))
    height, width = cast("tuple[int, int]", labels.shape)
    keep = main_and_parts(large[labels].astype(np.uint8), sharp, focus, edge, debris)
    blurred = sharp < debris.blur_fraction * focus
    keep = without_threads(without_edge_glow(keep, edge, blurred), edge)
    keep = main_and_parts(keep, sharp, focus, edge, debris)
    glare = (
        bright
        & (sharp < glare_rules.sharp_fraction * focus)
        & (smooth > glare_rules.bright_factor * glow)
    ).astype(np.uint8)
    count, patches, stats, _ = cv2.connectedComponentsWithStats(glare, connectivity=8)
    sizes = cast(
        "list[int]",
        np.asarray(stats, dtype=np.int64)[:, cv2.CC_STAT_AREA].tolist(),
    )
    marks = np.asarray(patches, dtype=np.int64)
    tissue = keep > 0
    for patch in range(1, count):
        if sizes[patch] < height * width * glare_rules.area_fraction:
            continue
        core = cast("npt.NDArray[np.bool_]", marks == patch)
        spread = cv2.dilate(core.astype(np.uint8), disk(glare_rules.ring_px))
        reach = np.asarray(spread, dtype=np.uint8) > 0
        ring = reach & ~core
        beside = ring & ~tissue
        apart = np.count_nonzero(beside) / max(1, np.count_nonzero(ring))
        halo = np.count_nonzero(beside & bright) / max(1, np.count_nonzero(beside))
        if apart > glare_rules.apart_fraction or (
            apart > glare_rules.out_fraction and halo > glare_rules.halo_fraction
        ):
            keep[reach] = 0
    return keep


def main_and_parts(
    mask: npt.NDArray[np.uint8],
    sharp: npt.NDArray[np.float32],
    focus: float,
    edge: npt.NDArray[np.bool_],
    debris: DebrisRules = DEBRIS,
) -> npt.NDArray[np.uint8]:
    """Keep a mask's largest piece and the pieces that are tissue too.

    A piece apart from the largest goes when it is small and either blurred
    or at the edge of the scan, or when it reaches the picture's border.

    Parameters
    ----------
    mask
        One on the mask, zero off it.
    sharp
        The picture's grain for its brightness.
    focus
        The tissue's own grain for its brightness.
    edge
        The band along the edge of the scan.
    debris
        The rules for pieces apart from the tissue.

    Returns
    -------
    np.ndarray
        The mask without those pieces.
    """
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 2:
        return mask
    marks = np.asarray(labels, dtype=np.int64)
    pixels = cast(
        "list[int]",
        np.asarray(stats, dtype=np.int64)[:, cv2.CC_STAT_AREA].tolist(),
    )
    main = max(range(1, count), key=lambda piece: pixels[piece])
    height, width = cast("tuple[int, int]", mask.shape)
    band = max(1, round(debris.border_band * min(height, width)))
    border = np.ones(mask.shape, np.bool_)
    border[band:-band, band:-band] = False
    kept = np.ones(count, np.bool_)
    kept[0] = False
    for piece in range(1, count):
        if piece == main:
            continue
        where = cast("npt.NDArray[np.bool_]", marks == piece)
        blurred = float(np.median(sharp[where])) < debris.blur_fraction * focus
        small = pixels[piece] < debris.part_fraction * pixels[main]
        beside = blurred or np.count_nonzero(where & edge) > 0
        if (small and beside) or np.count_nonzero(where & border) > 0:
            kept[piece] = False
    return kept[marks].astype(np.uint8)


def without_edge_glow(
    mask: npt.NDArray[np.uint8],
    edge: npt.NDArray[np.bool_],
    blurred: npt.NDArray[np.bool_],
    rules: ScanEdgeRules = SCAN_EDGE,
) -> npt.NDArray[np.uint8]:
    """Take off a mask what is blurred and reaches the edge of the scan.

    A scanner leaves a bright line and smears along the edge of what it
    scanned, and they can touch the tissue. Blurred tissue away from that
    edge stays.

    Parameters
    ----------
    mask
        One on the mask, zero off it.
    edge
        The band along the edge of the scan.
    blurred
        Where the picture has little of the tissue's grain.
    rules
        The edge band's thresholds.

    Returns
    -------
    np.ndarray
        The mask without that glow.
    """
    soft = ((mask > 0) & blurred).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(soft, connectivity=8)
    marks = np.asarray(labels, dtype=np.int64)
    at_edge = cast("list[int]", np.unique(marks[edge]).tolist())
    dropped = np.zeros(count, np.bool_)
    dropped[at_edge] = True
    dropped[0] = False
    glow = cv2.dilate(dropped[marks].astype(np.uint8), disk(rules.glow_rim_px))
    return np.where(np.asarray(glow, dtype=np.uint8) > 0, 0, mask).astype(np.uint8)


def without_threads(
    mask: npt.NDArray[np.uint8],
    edge: npt.NDArray[np.bool_],
    rules: ScanEdgeRules = SCAN_EDGE,
) -> npt.NDArray[np.uint8]:
    """Take off a mask what is thin and reaches the edge of the scan.

    The scanner's bright line along that edge is sharp where it is not
    smeared. Thin tissue away from that edge stays.

    Parameters
    ----------
    mask
        One on the mask, zero off it.
    edge
        The band along the edge of the scan.
    rules
        The edge band's thresholds.

    Returns
    -------
    np.ndarray
        The mask without those threads.
    """
    thick = np.asarray(
        cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(rules.thread_px)), dtype=np.uint8
    )
    foot = np.asarray(cv2.dilate(thick, disk(rules.thread_fringe_px)), dtype=np.uint8)
    thin = ((mask > 0) & (foot < 1)).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(thin, connectivity=8)
    marks = np.asarray(labels, dtype=np.int64)
    at_edge = cast("list[int]", np.unique(marks[edge]).tolist())
    dropped = np.zeros(count, np.bool_)
    dropped[at_edge] = True
    dropped[0] = False
    return np.where(dropped[marks], 0, mask).astype(np.uint8)


def scan_edge(
    picture: npt.NDArray[np.float32],
    rules: ScanEdgeRules = SCAN_EDGE,
    slide: SlideTest = SLIDE,
) -> npt.NDArray[np.bool_]:
    """Find the band along the edge of what the scanner covered.

    Parameters
    ----------
    picture
        The picture's one channel, 0 to 255.
    rules
        The band's width.
    slide
        Which grey is the black ground the scanner left.

    Returns
    -------
    np.ndarray
        True within ``rules.band_fraction`` of the picture's border or of the
        black ground the scanner left around its tiles.
    """
    height, width = cast("tuple[int, int]", picture.shape)
    band = max(1, round(rules.band_fraction * min(height, width)))
    frame = np.ones(picture.shape, np.bool_)
    frame[1:-1, 1:-1] = False
    black = (picture <= slide.unscanned_level).astype(np.uint8)
    count, labels, _, _ = cv2.connectedComponentsWithStats(black, connectivity=8)
    marks = np.asarray(labels, dtype=np.int64)
    outside = cast("list[int]", np.unique(marks[frame]).tolist())
    around = np.zeros(count, np.bool_)
    around[outside] = True
    around[0] = False
    ground = (around[marks] | frame).astype(np.uint8)
    return np.asarray(cv2.dilate(ground, disk(band)), dtype=np.uint8) > 0


def disk(radius: int) -> npt.NDArray[np.uint8]:
    """Make a disk-shaped structuring element.

    Parameters
    ----------
    radius
        The disk's radius in pixels.

    Returns
    -------
    np.ndarray
        One inside the disk, a square of side ``2 * radius + 1``.
    """
    side = 2 * radius + 1
    return np.asarray(
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side)),
        dtype=np.uint8,
    )


def fill_holes(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Fill the holes of a binary mask: what the outside cannot reach.

    Parameters
    ----------
    mask
        One on the mask, zero off it.

    Returns
    -------
    np.ndarray
        The mask with its enclosed background set to one.
    """
    height, width = cast("tuple[int, int]", mask.shape)
    padded = np.zeros((height + 2, width + 2), np.uint8)
    padded[1:-1, 1:-1] = mask
    seeds = np.zeros((height + 4, width + 4), np.uint8)
    _ = cv2.floodFill(padded, seeds, (0, 0), 1)
    outside = padded[1:-1, 1:-1] - mask
    return (1 - outside).astype(np.uint8)
