"""Raw channels and how they are displayed, as in napari.

A section's raw channels (:meth:`langslice.core.workspace.Workspace.section_channels`)
are never edited. Each channel NAME can carry display properties
(:class:`ChannelProperties`: contrast limits in the file's own intensities,
gamma, colormap), held stack-wide on ``StackState.appearance["channels"]``
so a change is undone and checkpointed with everything else. They change
how a raw channel is drawn for viewing (a picture's ``channels`` overlay and
the ``channels`` strip) and nothing else: no fit, no image model and no
preprocessed channel reads them (:mod:`langslice.core.appearance` holds the
preprocessed channel).

A channel without contrast limits is drawn with automatic ones: alone in
the strip as read; in an overlay by :func:`auto_limits`, per section, in its
overlay colour (:func:`~langslice.core.appearance.channel_colors`). On a dark
background (fluorescence) the section's tissue is found on the mean of its
channels and each channel is drawn black at its background level and at full
colour at its :data:`TISSUE_WHITE_PERCENTILE` inside the tissue; on a light
background (brightfield) between its 1st and 99.5th percentiles
(:data:`~langslice.core.appearance.OVERLAY_STRETCH`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from langslice.core.appearance import (
    NAMED_COLORS,
    OVERLAY_PALETTE,
    OVERLAY_STRETCH,
    Look,
    channel_colors,
)
from langslice.core.image_prep import IntensityRange, working_intensity_ranges
from langslice.core.state import StackState
from langslice.core.workspace import Workspace

#: Where the display properties live on ``StackState.appearance``.
CHANNELS_KEY = "channels"
#: The colormaps a channel can be drawn in: gray, or one colour.
COLORMAPS: dict[str, tuple[int, int, int]] = {
    "gray": (255, 255, 255),
    **NAMED_COLORS,
    **{word: rgb for word, rgb in OVERLAY_PALETTE if word not in NAMED_COLORS},
}
#: Accepted gamma range.
MIN_GAMMA = 0.1
MAX_GAMMA = 10.0
#: Pixels per section the stack's channel summary samples.
SUMMARY_SAMPLES = 200_000
#: Percentile of a channel's intensities inside the tissue that its automatic
#: contrast draws at full colour on a dark background (:func:`auto_limits`):
#: the brightest twentieth of the tissue saturates, so faint fluorescence is
#: legible without a setting and a bright section is not blown out.
TISSUE_WHITE_PERCENTILE = 95.0
#: Pixels per plane :func:`auto_limits` samples.
AUTO_SAMPLES = 1_000_000
#: The word for a channel drawn with automatic contrast limits (captions).
AUTO_WORD = "auto"


@dataclass(frozen=True)
class ChannelProperties:
    """How one raw channel is displayed.

    *contrast_limits* are file intensities (a 16-bit channel's are 0..65535
    values): the low one is drawn black, the high one at full colour; None
    keeps the default stretch. *gamma* 1 is linear. *colormap* None keeps
    the default colour (gray alone, the overlay colour among several).
    """

    contrast_limits: tuple[float, float] | None = None
    gamma: float = 1.0
    colormap: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "contrast_limits": (None if self.contrast_limits is None
                                else [float(v) for v in self.contrast_limits]),
            "gamma": float(self.gamma),
            "colormap": self.colormap,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ChannelProperties:
        limits = data.get("contrast_limits")
        return cls(
            contrast_limits=None if limits is None else (float(limits[0]), float(limits[1])),
            gamma=float(data.get("gamma", 1.0)),
            colormap=data.get("colormap"),
        )

    @property
    def is_default(self) -> bool:
        return self.contrast_limits is None and self.gamma == 1.0 and self.colormap is None


def validate_properties(
    *,
    contrast_limits: Sequence[float] | None = None,
    gamma: float | None = None,
    colormap: str | None = None,
) -> ChannelProperties:
    """One channel's display properties, or ``ValueError`` naming what is wrong."""
    limits: tuple[float, float] | None = None
    if contrast_limits is not None and len(contrast_limits):
        try:
            values = [float(value) for value in contrast_limits]
        except (TypeError, ValueError):
            raise ValueError("contrast_limits must be two numbers") from None
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError("contrast_limits must be two finite numbers, low then high")
        if not values[0] < values[1]:
            raise ValueError("contrast_limits: the low limit must be below the high one")
        limits = (values[0], values[1])
    value = 1.0
    if gamma is not None:
        try:
            value = float(gamma)
        except (TypeError, ValueError):
            raise ValueError("gamma must be a number") from None
        if not math.isfinite(value) or not MIN_GAMMA <= value <= MAX_GAMMA:
            raise ValueError(f"gamma must be from {MIN_GAMMA:g} to {MAX_GAMMA:g}")
    word: str | None = None
    if colormap is not None and str(colormap).strip():
        word = str(colormap).strip().lower()
        word = "gray" if word == "grey" else word
        if word not in COLORMAPS:
            raise ValueError("colormap must be one of " + ", ".join(COLORMAPS))
    return ChannelProperties(contrast_limits=limits, gamma=value, colormap=word)


# --- the stack's properties --------------------------------------------------


def all_properties(state: StackState) -> dict[str, ChannelProperties]:
    """Every channel's display properties, by channel name."""
    held = state.appearance.get(CHANNELS_KEY) or {}
    return {str(name): ChannelProperties.from_dict(value) for name, value in held.items()
            if isinstance(value, Mapping)}


def channel_properties(state: StackState, name: str) -> ChannelProperties | None:
    """*name*'s display properties, None when it has none."""
    return all_properties(state).get(str(name))


def set_properties(state: StackState, name: str, properties: ChannelProperties | None) -> None:
    """Write *name*'s display properties (None, or the defaults: none)."""
    held = dict(state.appearance.get(CHANNELS_KEY) or {})
    if properties is None or properties.is_default:
        held.pop(str(name), None)
    else:
        held[str(name)] = properties.to_dict()
    if held:
        state.appearance[CHANNELS_KEY] = held
    else:
        state.appearance.pop(CHANNELS_KEY, None)


def with_properties(state: StackState, look: Look) -> Look:
    """*look* with the display properties of the raw channels it draws.

    A ``{"channel": name}`` or ``{"overlay": [names]}`` look gains a
    ``properties`` entry for those of its channels that have any, so the
    render cache keys them; any other look, and a raw look whose channels
    have none, is returned as it is.
    """
    if look is None or not ("channel" in look or "overlay" in look):
        return look
    names = [look["channel"]] if "channel" in look else list(look["overlay"])
    held = all_properties(state)
    shown = {str(name): held[str(name)].to_dict() for name in names if str(name) in held}
    return {**look, "properties": shown} if shown else look


# --- file intensities ---------------------------------------------------------


def intensity_ranges(ctx: Workspace, section_id: str) -> dict[str, IntensityRange]:
    """Each raw channel's map from file intensities to its working plane.

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


def channel_summary(
    ctx: Workspace, section_ids: Sequence[str], name: str,
) -> dict[str, Any] | None:
    """*name*'s file intensities over the sections that have it: the sample
    type and its range, and the 1st and 99.5th percentiles (sampled at the
    working copies' resolution). None when no section has the channel."""
    samples: list[np.ndarray] = []
    first: IntensityRange | None = None
    for section_id in section_ids:
        names, planes = ctx.section_channels(section_id)
        if name not in names:
            continue
        held = intensity_ranges(ctx, section_id)[name]
        first = first or held
        plane = np.asarray(planes[names.index(name)]).ravel()
        step = max(1, plane.size // SUMMARY_SAMPLES)
        samples.append(np.asarray(held.to_file(plane[::step]), dtype=np.float64))
    if first is None:
        return None
    pooled = np.concatenate(samples)
    low, high = (float(v) for v in np.percentile(pooled, OVERLAY_STRETCH))
    return {
        "dtype": first.dtype,
        "dtype_range": [first.dtype_min, first.dtype_max],
        "percentiles": {f"{OVERLAY_STRETCH[0]:g}": round(low, 3),
                        f"{OVERLAY_STRETCH[1]:g}": round(high, 3)},
    }


# --- drawing ------------------------------------------------------------------


def fine_detail(stretched: np.ndarray) -> float:
    """Fine structure of one stretched channel (0..1): the spread of what a
    3 px blur removes, over the pixels brighter than the background. Nuclei,
    layers and fibre edges score high; flat autofluorescence scores low."""
    tissue = stretched > 0.05
    if not tissue.any():
        return 0.0
    fine = stretched - cv2.GaussianBlur(stretched, (0, 0), 3.0)
    return float(fine[tissue].std())


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
    1st and 99.5th percentiles of each plane
    (:data:`~langslice.core.appearance.OVERLAY_STRETCH`).
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


def _stretched(
    plane: np.ndarray, properties: ChannelProperties | None, limits: tuple[float, float] | None,
) -> np.ndarray:
    """One working plane (float32, 0..255 values) mapped to 0..1 between
    *limits* (plane values; None: the plane as read), then raised to its
    gamma."""
    low, high = limits if limits is not None else (0.0, 255.0)
    if high <= low:
        high = low + 1.0
    stretched = np.clip((plane - low) / (high - low), 0.0, 1.0)
    if properties is not None and properties.gamma != 1.0:
        stretched = np.power(stretched, properties.gamma).astype(np.float32)
    return stretched


def _plane_limits(
    properties: ChannelProperties | None, intensity: IntensityRange | None,
) -> tuple[float, float] | None:
    if properties is None or properties.contrast_limits is None or intensity is None:
        return None
    low, high = properties.contrast_limits
    return intensity.to_plane(low), intensity.to_plane(high)


def composite(
    planes: Sequence[tuple[str, np.ndarray]],
    *,
    framed: Callable[[Image.Image], Image.Image],
    size: tuple[int, int],
    properties: Mapping[str, Mapping[str, Any]] | None = None,
    ranges: Mapping[str, IntensityRange] | None = None,
    single: bool = False,
) -> Image.Image:
    """Raw channels drawn together: an RGB picture of *size*.

    *planes* are ``(name, plane)`` at working size (float32, 0..255 values);
    *framed* crops and sizes a working-size picture as the render does.
    Each channel is stretched on its WHOLE working plane (so a framed and an
    unframed picture share one stretch): between its contrast limits when
    it has them, else (an overlay) its automatic limits (:func:`auto_limits`,
    measured on all of *planes*) or (*single*, the ``channels`` strip) as
    read; then raised to its gamma and added in its
    colormap (gray alone, else :func:`channel_colors`). Among several
    channels, each one without contrast limits is dimmed by its fine detail
    relative to the most detailed of them, so a flat autofluorescence
    channel (stretched to full brightness on its own) cannot wash out the
    stain under it.
    """
    props = {str(name): ChannelProperties.from_dict(value)
             for name, value in (properties or {}).items()}
    names = [name for name, _plane in planes]
    defaults = ([(names[0], "gray", COLORMAPS["gray"])] if len(names) == 1
                else channel_colors(names))
    total = np.zeros((size[1], size[0], 3), dtype=np.float32)
    stretched_planes: list[np.ndarray] = []
    detail: list[float | None] = []
    colors: list[tuple[int, int, int]] = []
    automatic = None if single else auto_limits([plane for _name, plane in planes])
    for index, ((name, plane), (_name, _word, rgb)) in enumerate(
            zip(planes, defaults, strict=True)):
        held = props.get(name)
        limits = _plane_limits(held, (ranges or {}).get(name))
        stretched = _stretched(plane, held, limits if limits is not None else
                               None if automatic is None else automatic[index])
        stretched_planes.append(stretched)
        detail.append(None if limits is not None else fine_detail(stretched))
        colors.append(COLORMAPS[held.colormap] if held is not None and held.colormap
                      else rgb)
    measured = [amount for amount in detail if amount is not None]
    top = max(measured) if len(planes) > 1 and measured else 0.0
    for rgb, stretched, amount in zip(colors, stretched_planes, detail, strict=True):
        gain = amount / top if amount is not None and top > 0 else 1.0
        shown = np.asarray(framed(Image.fromarray(stretched_to_u8(stretched))),
                           dtype=np.float32) / 255.0
        total += gain * shown[..., None] * np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(np.clip(total, 0.0, 255.0).astype(np.uint8), mode="RGB")


def stretched_to_u8(stretched: np.ndarray) -> np.ndarray:
    """A 0..1 plane as 8-bit (truncated, as the overlay always has)."""
    return (stretched * 255.0).astype(np.uint8)


# --- words --------------------------------------------------------------------


def setting_words(properties: ChannelProperties | Mapping[str, Any] | None) -> str:
    """One channel's display setting for a caption: ``auto``, ``0-70 gamma
    0.8``, ``auto gamma 1.5 in cyan`` (the contrast limits in file
    intensities, or ``auto``; the gamma when not 1; the colormap when set)."""
    if properties is None:
        return AUTO_WORD
    held = (properties if isinstance(properties, ChannelProperties)
            else ChannelProperties.from_dict(properties))
    parts = [AUTO_WORD if held.contrast_limits is None
             else f"{held.contrast_limits[0]:g}-{held.contrast_limits[1]:g}"]
    if held.gamma != 1.0:
        parts.append(f"gamma {held.gamma:g}")
    if held.colormap:
        parts.append(f"in {held.colormap}")
    return " ".join(parts)


def describe(name: str, properties: ChannelProperties | Mapping[str, Any] | None) -> str:
    """A short caption form of one channel's display, e.g. ``DAPI 120-3400
    gamma 1.2 in gray``, or ``DAPI auto`` without properties
    (:func:`setting_words`)."""
    return f"{name} {setting_words(properties)}"


def display_words(
    names: Sequence[str], held: Mapping[str, ChannelProperties],
    colours: Mapping[str, str] | None = None,
) -> str:
    """Every channel of *names* with its own display setting, channels in a
    row with the same setting grouped: ``red, green auto; blue 0-70 gamma
    0.8``. *colours* gives the overlay colour of a channel not named after
    it (``DAPI in cyan``); a channel's own colormap is in its setting."""
    groups: list[tuple[str, list[str]]] = []
    for name in names:
        properties = held.get(name)
        setting = setting_words(properties)
        word = (colours or {}).get(name)
        own = properties is not None and bool(properties.colormap)
        shown = name if not word or own or word == name.lower() else f"{name} in {word}"
        if groups and groups[-1][0] == setting:
            groups[-1][1].append(shown)
        else:
            groups.append((setting, [shown]))
    return "; ".join(f"{', '.join(shown)} {setting}" for setting, shown in groups)


def describe_shown(state: StackState, names: Sequence[str],
                   colours: Mapping[str, str] | None = None) -> str:
    """Every channel in *names* with its display setting, for a caption
    (:func:`display_words`)."""
    return display_words(names, all_properties(state), colours)


def stack_display(ctx: Workspace, state: StackState) -> str:
    """Every raw channel's display setting, for the status reply: the first
    section's channels (in stack order) and any other channel that has
    properties (:func:`display_words`, e.g. ``red, green auto; blue 0-70
    gamma 0.8``); empty for a stack without sections."""
    held = all_properties(state)
    first = sorted(state.slices, key=lambda record: record.index_original)[:1]
    names = list(ctx.section_channels(first[0].id)[0]) if first else []
    names += [name for name in held if name not in names]
    return display_words(names, held)
