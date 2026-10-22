"""Raw channels drawn together, in colours.

A section's raw channels (:meth:`langslice.core.workspace.Workspace.section_channels`)
are never edited. One raw channel alone is a section image and is drawn as the
file holds it, in gray (:mod:`langslice.core.section_images`). Several are drawn
together here: each in its overlay colour (:func:`channel_colors`), between
its automatic contrast limits (:func:`auto_limits`), per section. On a dark
background (fluorescence) the section's tissue is found on the mean of its
channels and each channel is drawn black at its background level and at full
colour at its :data:`TISSUE_WHITE_PERCENTILE` inside the tissue; on a light
background (brightfield) between its 1st and 99.5th percentiles
(:data:`OVERLAY_STRETCH`). Nothing a fit reads comes from here.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.image_prep import IntensityRange, working_intensity_ranges
from langslice.core.workspace import Workspace

#: Colours an overlay gives its channels, in order. A channel NAMED after a
#: colour (an RGB file's ``red``/``green``/``blue``) keeps that colour, so
#: the three planes of a colour image overlay back into it.
OVERLAY_PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("green", (0, 255, 0)),
    ("magenta", (255, 0, 255)),
    ("cyan", (0, 255, 255)),
    ("yellow", (255, 255, 0)),
    ("red", (255, 0, 0)),
    ("blue", (0, 96, 255)),
)
NAMED_COLORS: dict[str, tuple[int, int, int]] = {
    "red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255),
}
#: Percentiles an overlay maps to black and white, per channel, on a light
#: background (:func:`auto_limits`).
OVERLAY_STRETCH = (1.0, 99.5)
#: Percentile of a channel's intensities inside the tissue that its automatic
#: contrast draws at full colour on a dark background (:func:`auto_limits`):
#: the brightest twentieth of the tissue saturates, so faint fluorescence is
#: legible and a bright section is not blown out.
TISSUE_WHITE_PERCENTILE = 95.0
#: Pixels per plane :func:`auto_limits` samples.
AUTO_SAMPLES = 1_000_000


def channel_colors(names: Any) -> list[tuple[str, str, tuple[int, int, int]]]:
    """``(name, colour word, rgb)`` per channel of an overlay, in order."""
    out: list[tuple[str, str, tuple[int, int, int]]] = []
    taken = {str(name).lower() for name in names if str(name).lower() in NAMED_COLORS}
    palette = [entry for entry in OVERLAY_PALETTE if entry[0] not in taken]
    for name in names:
        key = str(name).lower()
        if key in NAMED_COLORS:
            out.append((str(name), key, NAMED_COLORS[key]))
        else:
            word, rgb = palette.pop(0) if palette else ("white", (255, 255, 255))
            out.append((str(name), word, rgb))
    return out


def colour_words(names: Sequence[str]) -> str:
    """The raw channels of an overlay in words: ``red + green + blue``, or
    ``DAPI cyan + GFP green`` for a channel not named after its colour."""
    return " + ".join(name if name.lower() == word else f"{name} {word}"
                      for name, word, _rgb in channel_colors(list(names)))


# --- file intensities ---------------------------------------------------------


def intensity_ranges(ctx: Workspace, section_id: str) -> dict[str, IntensityRange]:
    """Each raw channel's map from file intensities to its working plane (a
    zoom reads the file's own pixels through it, :mod:`langslice.core.native`).

    Cached on the workspace (a decode of the file the first time).
    """
    cached = ctx.intensity_cache.get(section_id)
    if cached is None:
        names, _planes = ctx.section_channels(section_id)
        pages = working_intensity_ranges(ctx.image_path(section_id))
        ranges = pages * len(names) if len(pages) == 1 else pages
        cached = dict(zip(names, ranges, strict=False))
        ctx.intensity_cache[section_id] = cached
    return cached


# --- drawing ------------------------------------------------------------------


def auto_limits(planes: Sequence[np.ndarray]) -> list[tuple[float, float]]:
    """Each plane's automatic contrast limits, as plane values.

    *planes* are one section's raw channels at working size (0..255
    values), sampled to about :data:`AUTO_SAMPLES` pixels. A dark
    background (the median of the planes' mean along the picture's border at
    most its median everywhere: fluorescence) separates tissue from
    background by Otsu's threshold on that mean (exact zeros, unscanned
    tiles or padding, left out of the threshold); each channel's low limit
    is its median over the background (zeros included, so a padded black
    background reads as 0), its high one its :data:`TISSUE_WHITE_PERCENTILE`
    over the tissue. A light background (brightfield), or no usable split, keeps the
    1st and 99.5th percentiles of each plane (:data:`OVERLAY_STRETCH`).
    """
    arrays = [np.asarray(plane, dtype=np.float32) for plane in planes]
    if not arrays:
        return []
    step = max(1, int(math.ceil(math.sqrt(arrays[0].size / AUTO_SAMPLES))))
    sampled = [array[::step, ::step] for array in arrays]

    def spread(values: np.ndarray) -> tuple[float, float]:
        low, high = (float(v) for v in np.percentile(values, OVERLAY_STRETCH))
        return (low, high if high > low else low + 1.0)

    fallback = [spread(values) for values in sampled]
    mean = np.mean(np.stack(sampled), axis=0)
    border = np.concatenate([mean[0], mean[-1], mean[:, 0], mean[:, -1]])
    if float(np.median(border)) > float(np.median(mean)):
        return fallback
    scanned = mean > 0
    if scanned.sum() < 100:
        return fallback
    values = np.clip(mean[scanned], 0, 255).astype(np.uint8).reshape(-1, 1)
    threshold, _ = cv2.threshold(values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue = mean > threshold
    background = ~tissue
    if tissue.sum() < 100 or background.sum() < 100:
        return fallback
    out: list[tuple[float, float]] = []
    for values, default in zip(sampled, fallback, strict=True):
        low = float(np.median(values[background]))
        high = float(np.percentile(values[tissue], TISSUE_WHITE_PERCENTILE))
        out.append((low, high) if high > low else default)
    return out


def composite(
    planes: Sequence[tuple[str, np.ndarray]],
    *,
    framed: Callable[[Image.Image], Image.Image],
    size: tuple[int, int],
) -> Image.Image:
    """Raw channels drawn together: an RGB picture of *size*.

    *planes* are ``(name, plane)`` at working size (float32, 0..255 values);
    *framed* sizes a working-size picture as the render does. Each channel is
    stretched on its whole working plane between its automatic limits
    (:func:`auto_limits`, measured on all of *planes*), then added in its
    colour (:func:`channel_colors`).
    """
    names = [name for name, _plane in planes]
    colours = channel_colors(names)
    total = np.zeros((size[1], size[0], 3), dtype=np.float32)
    automatic = auto_limits([plane for _name, plane in planes])
    for (_name, plane), (_n, _word, rgb), (low, high) in zip(
            planes, colours, automatic, strict=True):
        if high <= low:
            high = low + 1.0
        stretched = np.clip((plane - low) / (high - low), 0.0, 1.0)
        shown = np.asarray(framed(Image.fromarray((stretched * 255.0).astype(np.uint8))),
                           dtype=np.float32) / 255.0
        total += shown[..., None] * np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(np.clip(total, 0.0, 255.0).astype(np.uint8), mode="RGB")
