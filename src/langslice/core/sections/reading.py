"""Image ingest helpers for VLM preparation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Whole-slide microscopy exports routinely exceed PIL's ~179-megapixel
# decompression-bomb guard, which is meant for untrusted web content; our
# inputs are the user's local acquisitions.
Image.MAX_IMAGE_PIXELS = None

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS

DEFAULT_VLM_MAX_PIXELS = 12_000_000
DEFAULT_VLM_MAX_LONG_EDGE = 4096


#: Length units an image tag may name, in micrometres. Micron spellings are
#: matched separately: a file written by tifffile carries "µm" whose UTF-8
#: bytes routinely reach us as "Âµm", and a plain "m" lookup would read that
#: as METRES.
_LENGTH_UM: dict[str, float] = {"nm": 1e-3, "mm": 1e3, "cm": 1e4, "m": 1e6}


def _unit_to_um(unit: str) -> float | None:
    text = unit.strip().lower()
    if not text:
        return 1.0  # OME's default PhysicalSize unit is micrometres
    if "µ" in text or "μ" in text or "micro" in text or text.startswith("u"):
        return 1.0
    return _LENGTH_UM.get(text)


def read_pixel_size_um(path: str | Path) -> float | None:
    """Micrometres per pixel of an image file, or None when it does not say.

    Three sources, in order: the OME-XML in the TIFF ImageDescription
    (``PhysicalSizeX`` plus its unit), ImageJ's calibration (the
    ImageDescription's ``unit=`` with ``XResolution`` pixels per that unit),
    and ``XResolution`` with ``ResolutionUnit`` 3 (centimetre). Anything else —
    inch dpi or no unit (editors write 72/300 dpi placeholders), a PNG, a
    JPEG, a TIFF written without resolution — answers None, and the caller
    reports the section as uncalibrated.
    """
    try:
        with Image.open(path) as handle:
            tags = dict(getattr(handle, "tag_v2", None) or {})
    except Exception:
        return None

    description = tags.get(270)
    if isinstance(description, str) and "PhysicalSizeX" in description:
        value = re.search(r'PhysicalSizeX="([-+0-9.eE]+)"', description)
        unit = re.search(r'PhysicalSizeXUnit="([^"]*)"', description)
        factor = _unit_to_um(unit.group(1) if unit else "")
        if value is not None and factor is not None:
            size = float(value.group(1)) * factor
            if size > 0:
                return size

    resolution, unit_code = tags.get(282), int(tags.get(296, 0) or 0)
    per_unit_um = 10000.0 if unit_code == 3 else None
    if isinstance(description, str) and description.startswith("ImageJ="):
        unit = re.search(r"^unit=(.+)$", description, re.MULTILINE)
        if unit is not None and unit.group(1).strip():
            per_unit_um = _unit_to_um(unit.group(1))
    if resolution is None or per_unit_um is None:
        return None
    try:
        pixels_per_unit = float(resolution)
    except (TypeError, ValueError):
        return None
    return per_unit_um / pixels_per_unit if pixels_per_unit > 0 else None


def normalize_image(image: Image.Image) -> Image.Image:
    """Normalize an arbitrary PIL image to 8-bit RGB without mutating source."""
    mode = image.mode
    if mode == "RGB":
        return image

    if mode in ("I", "I;16", "I;16B", "I;32", "F"):
        arr = np.asarray(image, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi > lo:
            arr = (arr - lo) / (hi - lo) * 255.0
        else:
            arr = np.zeros_like(arr)
        gray8 = arr.astype(np.uint8)
        return Image.fromarray(gray8).convert("RGB")

    return image.convert("RGB")


def adaptive_preprocess(image: Image.Image) -> Image.Image:
    """Adaptive preprocessing for VLM input: CLAHE + weighted blend + brightness.

    The image's three planes through :func:`_blend_channels` at the
    automatic clip (``HOST_CLAHE_CLIP["medium"]``), weighted toward the
    STRUCTURAL channel by :func:`coverage_weights` (DAPI-blue, YFP-green,
    whatever lights the whole tissue; sparse tracer channels get little
    weight), brightness boost toward 90.
    """
    arr = np.asarray(normalize_image(image), dtype=np.uint8)
    planes = [arr[:, :, k] for k in range(3)]
    return _blend_channels(planes, coverage_weights(planes), clahe_clip=HOST_CLAHE_CLIP["medium"])


# --- host preprocessing: N exported channels -> the one image shown ---

#: CLAHE clip limit per user-chosen strength in custom host preprocessing.
#: "medium" is the automatic path's own clip (:func:`adaptive_preprocess`).
HOST_CLAHE_CLIP: dict[str, float] = {"low": 2.0, "medium": 4.0, "high": 8.0}
#: CLAHE tile grid for host preprocessing, the automatic path's own.
HOST_CLAHE_TILE = (8, 8)
HOST_PREPROCESS_MODES = ("auto", "custom")


#: Smallest long edge a section's working copy may have: the largest picture a
#: run shows, the 768 px overlay at "high". Every render is drawn from the
#: working copy, so a whole-slide scan is never decoded at full size.
WORKING_MIN_EDGE = 1536
#: Largest long edge of a working copy decoded from a file without a usable
#: pyramid level (it is downsampled once to this).
WORKING_MAX_EDGE = 3072


def read_working_image(
    path: str | Path, min_edge: int = WORKING_MIN_EDGE, max_edge: int = WORKING_MAX_EDGE,
) -> tuple[Image.Image, float]:
    """``(8-bit RGB image, file pixels per image pixel)`` for one section file.

    A whole-slide scan is read at its smallest pyramid level still at least
    *min_edge* long (tifffile), a JPEG through PIL's draft decode; whatever
    is still longer than *max_edge* is downsampled to it. The image is what
    :func:`normalize_image` gives for the file, only smaller: callers that
    measure in file pixels multiply by the returned factor.
    """
    image: Image.Image | None = None
    full_width = 0
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        image, full_width = _read_tiff_level(path, min_edge)
    if image is None:
        with Image.open(path) as handle:
            full_width = handle.size[0]
            if handle.format == "JPEG":
                handle.draft("RGB", (max_edge, max_edge))
            image = normalize_image(handle.copy())
    if max(image.size) > max_edge:
        scale = max_edge / float(max(image.size))
        size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
        image = image.resize(size, _RESAMPLE_LANCZOS, reducing_gap=3.0)
    return image, full_width / float(image.width)


def _read_tiff_level(path: str | Path, min_edge: int) -> tuple[Image.Image | None, int]:
    """The smallest plain YX / YXS pyramid level at least *min_edge* long.

    ``(None, 0)`` for anything else (multi-channel planes, odd layouts): the
    caller falls back to PIL, which reads the first page as before.
    """
    import tifffile

    with tifffile.TiffFile(path) as handle:
        if not handle.series:
            return None, 0
        series = handle.series[0]
        axes = series.axes
        if axes not in ("YX", "YXS") or series.dtype not in (np.uint8, np.uint16):
            return None, 0
        levels = list(series.levels)
        full_width = int(levels[0].shape[1])
        chosen = levels[0]
        for level in levels[1:]:
            if max(level.shape[0], level.shape[1]) >= min_edge:
                chosen = level
        array = np.asarray(chosen.asarray())
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        return None, 0
    return _page_image(array), full_width


def read_pages(path: str | Path) -> list[np.ndarray]:
    """Every page of an image file, first page first.

    A multi-page TIFF (one page per exported channel) gives one array per
    page; any other image, or a single-page TIFF, gives one.
    """
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            pages = [np.asarray(page.asarray()) for page in handle.pages]
        if pages:
            return pages
    with Image.open(path) as handle:
        return [np.asarray(handle.copy())]


def page_count(path: str | Path) -> int:
    """How many pages :func:`read_pages` would return, without decoding them."""
    if Path(path).suffix.lower() in (".tif", ".tiff"):
        import tifffile

        with tifffile.TiffFile(path) as handle:
            count = len(handle.pages)
        if count:
            return count
    return 1


def _page_image(page: np.ndarray) -> Image.Image:
    """One page as the 8-bit RGB image the single-file path reads."""
    array = np.asarray(page)
    if array.dtype == np.bool_:
        array = array.astype(np.uint8) * 255
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3:
        if array.shape[-1] not in (3, 4):
            raise ValueError(f"Unsupported page shape {array.shape}")
        if array.dtype != np.uint8:
            array = _stretch(array.astype(np.float32))
    elif array.ndim != 2:
        raise ValueError(f"Unsupported page shape {array.shape}")
    elif array.dtype.kind == "u" and array.dtype.itemsize <= 2:
        array = array.astype(np.uint8 if array.dtype.itemsize == 1 else np.uint16)
    else:
        array = array.astype(np.float32)
    return normalize_image(Image.fromarray(array))


def _stretch(array: np.ndarray) -> np.ndarray:
    lo, hi = float(array.min()), float(array.max())
    out = (array - lo) / (hi - lo) * 255.0 if hi > lo else np.zeros_like(array)
    return out.astype(np.uint8)


def _page_channel(page: np.ndarray) -> np.ndarray:
    """One page as one 8-bit channel (a colour page counts as its luminance)."""
    image = _page_image(page)
    if np.asarray(page).ndim == 3 and np.asarray(page).shape[-1] in (3, 4):
        return np.asarray(image.convert("L"), dtype=np.uint8)
    return np.asarray(image, dtype=np.uint8)[..., 0]


def host_preprocess_settings(
    settings: dict[str, object] | None, n_pages: int
) -> dict[str, object]:
    """Validated host preprocessing settings for an image of *n_pages* pages.

    ``{"mode": "auto"}`` (also for ``None``) or ``{"mode": "custom", "clahe":
    bool, "clahe_strength": "low"|"medium"|"high", "channel_weights": [w, ...]}``
    with one finite, non-negative weight per page, not all zero (absent means
    equal weights). Custom-only keys are ignored in auto mode.
    """
    values = dict(settings or {})
    mode = values.get("mode", "auto")
    if mode not in HOST_PREPROCESS_MODES:
        raise ValueError(f"Preprocessing mode must be one of {HOST_PREPROCESS_MODES}")
    if mode == "auto":
        return {"mode": "auto"}
    clahe = values.get("clahe", True)
    if not isinstance(clahe, bool):
        raise ValueError("Preprocessing clahe must be true or false")
    strength = values.get("clahe_strength", "medium")
    if strength not in HOST_CLAHE_CLIP:
        raise ValueError(f"CLAHE strength must be one of {tuple(HOST_CLAHE_CLIP)}")
    raw = values.get("channel_weights")
    if raw is None:
        weights = [1.0] * n_pages
    else:
        if not isinstance(raw, (list, tuple)) or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) for value in raw
        ):
            raise ValueError("Channel weights must be a list of numbers")
        weights = [float(value) for value in raw]
        if len(weights) != n_pages:
            raise ValueError(
                f"Expected {n_pages} channel weight(s), one per page; got {len(weights)}"
            )
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("Channel weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("At least one channel weight must be above zero")
    return {"mode": "custom", "clahe": clahe, "clahe_strength": strength,
            "channel_weights": weights}


def _blend_channels(
    channels: list[np.ndarray],
    weights: list[float],
    *,
    clahe_clip: float | None,
    tile: tuple[int, int] = HOST_CLAHE_TILE,
    target_brightness: float = 90.0,
    max_boost: float = 3.0,
) -> Image.Image:
    """Per-channel CLAHE (unless *clahe_clip* is None), weighted blend,
    brightness boost: :func:`adaptive_preprocess` for any number of
    channels."""
    import cv2

    total = float(sum(weights))
    normalized = [value / total for value in weights] if total > 0 else weights
    if clahe_clip is not None:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=tile)
        enhanced = [clahe.apply(channel).astype(np.float32) for channel in channels]
    else:
        enhanced = [channel.astype(np.float32) for channel in channels]
    blended = np.zeros(channels[0].shape, dtype=np.float32)
    for weight, channel in zip(normalized, enhanced, strict=True):
        blended = blended + weight * channel
    blended = np.clip(blended, 0, 255)
    brain_mask = blended > 10
    if brain_mask.sum() > 0:
        current_mean = float(blended[brain_mask].mean())
        boost = min(target_brightness / max(current_mean, 1.0), max_boost)
    else:
        boost = 1.0
    if boost > 1.05:
        blended = blended * boost
    out = np.clip(blended, 0, 255).astype(np.uint8)
    return Image.fromarray(np.stack([out, out, out], axis=-1))


def host_preprocess(
    pages: list[np.ndarray], settings: dict[str, object] | None = None
) -> Image.Image:
    """The one grayscale image the agent is shown, from a host's exported pages.

    *pages* are the pages of one snapshot (one per exported channel, see
    :func:`read_pages`); *settings* as :func:`host_preprocess_settings`.

    - auto, one page: exactly :func:`adaptive_preprocess` on that page read
      the way a single image file is read.
    - auto, several pages: the same method over N channels — per-channel
      CLAHE blended by squared tissue coverage (the channel lighting most of
      the tissue dominates) with the brightness boost.
    - custom: per-channel CLAHE at the chosen strength (or none), blended by
      the user's weights, with the same brightness boost.

    Returned as 8-bit RGB with three equal channels, the form
    :func:`adaptive_preprocess` returns.
    """
    if not pages:
        raise ValueError("No image pages to preprocess")
    shapes = {np.asarray(page).shape[:2] for page in pages}
    if len(shapes) != 1:
        raise ValueError(f"All pages must share one size; got {sorted(shapes)}")
    chosen = host_preprocess_settings(settings, len(pages))
    if chosen["mode"] == "auto" and len(pages) == 1:
        return adaptive_preprocess(_page_image(pages[0]))

    channels = [_page_channel(page) for page in pages]
    if chosen["mode"] == "custom":
        clip = HOST_CLAHE_CLIP[str(chosen["clahe_strength"])] if chosen["clahe"] else None
        return _blend_channels(
            channels, list(chosen["channel_weights"]),  # type: ignore[arg-type]
            clahe_clip=clip,
        )

    return _blend_channels(channels, coverage_weights(channels), clahe_clip=4.0)


def coverage_weights(channels: list[np.ndarray]) -> list[float]:
    """Automatic channel weights: squared tissue coverage, normalized.

    The structural stain (DAPI, Nissl, a fill) lights most of the tissue and
    dominates; a sparse tracer lighting small patches gets little weight. The
    automatic path of :func:`host_preprocess` and :func:`adaptive_preprocess`.
    """
    stack = np.stack(channels, axis=-1)
    tissue = stack.max(axis=-1) > 15
    weights = [1.0] * len(channels)
    if tissue.any():
        coverage = np.array(
            [float((channel[tissue] > 40).mean()) for channel in channels], dtype=np.float64
        )
        squared = coverage**2
        if squared.sum() > 0:
            weights = [float(value) for value in squared / squared.sum()]
    return weights


# --- raw channels and the agent-chosen appearance ---------------------------


def read_working_pages(
    path: str | Path, min_edge: int = WORKING_MIN_EDGE, max_edge: int = WORKING_MAX_EDGE,
) -> tuple[list[np.ndarray], float]:
    """``(pages, file pixels per page pixel)`` of one section file, working size.

    A single-page file gives one page: :func:`read_working_image`'s RGB array,
    so its pixels are exactly the working copy every render is drawn from. A
    multi-page file (one page per exported channel) gives one 8-bit 2D array
    per page: a pyramidal series is read at its smallest level at least
    *min_edge* long; otherwise each page is decoded on its own and downsampled
    to *max_edge* before the next one is read.
    """
    if page_count(path) <= 1:
        image, factor = read_working_image(path, min_edge, max_edge)
        return [np.asarray(image)], factor
    import tifffile

    pages: list[np.ndarray] = []
    full_width = 0
    with tifffile.TiffFile(path) as handle:
        series = handle.series[0] if handle.series else None
        levels = list(series.levels) if series is not None else []
        if (series is not None and len(levels) > 1 and len(series.axes) == 3
                and series.axes.endswith("YX")):
            full_width = int(levels[0].shape[-1])
            chosen = levels[0]
            for level in levels[1:]:
                if max(level.shape[-2], level.shape[-1]) >= min_edge:
                    chosen = level
            pages = [_page_channel(plane) for plane in np.asarray(chosen.asarray())]
        else:
            for page in handle.pages:
                channel = _page_channel(np.asarray(page.asarray()))
                full_width = full_width or int(channel.shape[1])
                pages.append(_shrink_channel(channel, max_edge))
    return pages, full_width / float(pages[0].shape[1])


def working_size(
    path: str | Path, *, pages: bool = False,
    min_edge: int = WORKING_MIN_EDGE, max_edge: int = WORKING_MAX_EDGE,
) -> tuple[tuple[int, int], float] | None:
    """``((width, height), file pixels per working pixel)`` of a file's working
    copy, read from its header: nothing is decoded.

    The size :func:`read_working_image` (or, with *pages*,
    :func:`read_working_pages`) would return, for the layouts they read
    directly: a plain or pyramidal TIFF, a multi-page TIFF, PNG, JPEG (its
    draft size). None for anything else; the caller decodes instead.
    """
    try:
        if pages and page_count(path) > 1:
            return _pages_working_size(path, min_edge, max_edge)
        size: tuple[int, int] | None = None
        full_width = 0
        if Path(path).suffix.lower() in (".tif", ".tiff"):
            size, full_width = _tiff_level_size(path, min_edge)
        if size is None:
            with Image.open(path) as handle:
                full_width = handle.size[0]
                if handle.format == "JPEG":
                    handle.draft("RGB", (max_edge, max_edge))
                size = (int(handle.size[0]), int(handle.size[1]))
        if max(size) > max_edge:
            scale = max_edge / float(max(size))
            size = (max(1, round(size[0] * scale)), max(1, round(size[1] * scale)))
        return size, full_width / float(size[0])
    except Exception:  # an odd layout: the caller decodes
        return None


def _tiff_level_size(path: str | Path, min_edge: int) -> tuple[tuple[int, int] | None, int]:
    """:func:`_read_tiff_level`'s choice, from the header: ``(size, full width)``."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        if not handle.series:
            return None, 0
        series = handle.series[0]
        if series.axes not in ("YX", "YXS") or series.dtype not in (np.uint8, np.uint16):
            return None, 0
        levels = list(series.levels)
        full_width = int(levels[0].shape[1])
        chosen = levels[0]
        for level in levels[1:]:
            if max(level.shape[0], level.shape[1]) >= min_edge:
                chosen = level
        shape = tuple(int(v) for v in chosen.shape)
    if len(shape) == 3 and shape[-1] not in (3, 4):
        return None, 0
    return (shape[1], shape[0]), full_width


def _pages_working_size(
    path: str | Path, min_edge: int, max_edge: int,
) -> tuple[tuple[int, int], float]:
    """:func:`read_working_pages`' page size for a multi-page file, from the header."""
    import tifffile

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0] if handle.series else None
        levels = list(series.levels) if series is not None else []
        if (series is not None and len(levels) > 1 and len(series.axes) == 3
                and series.axes.endswith("YX")):
            full_width = int(levels[0].shape[-1])
            chosen = levels[0]
            for level in levels[1:]:
                if max(level.shape[-2], level.shape[-1]) >= min_edge:
                    chosen = level
            height, width = (int(v) for v in chosen.shape[-2:])
            return (width, height), full_width / float(width)
        shape = tuple(int(v) for v in handle.pages[0].shape)  # type: ignore[union-attr]
    if len(shape) == 3 and shape[-1] not in (1, 3, 4):
        raise ValueError(f"Unsupported page shape {shape}")
    height, width = shape[0], shape[1]
    full_width = width
    if max(width, height) > max_edge:
        scale = max_edge / float(max(width, height))
        width, height = max(1, round(width * scale)), max(1, round(height * scale))
    return (width, height), full_width / float(width)


def _shrink_channel(channel: np.ndarray, max_edge: int) -> np.ndarray:
    if max(channel.shape) <= max_edge:
        return channel
    scale = max_edge / float(max(channel.shape))
    size = (max(1, round(channel.shape[1] * scale)), max(1, round(channel.shape[0] * scale)))
    resized = Image.fromarray(channel).resize(size, _RESAMPLE_LANCZOS, reducing_gap=3.0)
    return np.asarray(resized, dtype=np.uint8)


#: Names of a single-page colour file's channels, in order.
RGB_CHANNEL_NAMES = ("red", "green", "blue")


def channel_planes(
    pages: list[np.ndarray], names: list[str] | None = None,
) -> tuple[tuple[str, ...], list[np.ndarray]]:
    """``(names, 8-bit 2D planes)``: the raw channels of one section.

    One colour page is its red, green and blue planes (one ``gray`` plane
    when the three are identical); several pages are one plane each, named
    by *names* when the host supplied one per page, else ``ch1``, ``ch2``...
    """
    if len(pages) == 1:
        array = np.asarray(pages[0])
        if array.ndim == 2:
            return ("gray",), [_page_channel(array)]
        rgb = np.asarray(_page_image(array), dtype=np.uint8)
        planes = [np.ascontiguousarray(rgb[..., index]) for index in range(3)]
        if np.array_equal(planes[0], planes[1]) and np.array_equal(planes[1], planes[2]):
            return ("gray",), [planes[0]]
        return RGB_CHANNEL_NAMES, planes
    planes = [_page_channel(page) for page in pages]
    if names is not None and len(names) == len(planes):
        return tuple(str(name) for name in names), planes
    return tuple(f"ch{index + 1}" for index in range(len(planes))), planes


@dataclass(frozen=True)
class IntensityRange:
    """How one page's FILE intensities map onto its 8-bit working planes.

    Reading a file stretches it linearly: *low* becomes 0 and *high* 255
    (an 8-bit page is read as it is, ``0..255``; a 16-bit, integer or float
    one from its own minimum to its maximum). *dtype* and
    ``dtype_min``/``dtype_max`` are the file's sample type and its range
    (the observed range for a float file).
    """

    dtype: str
    dtype_min: float
    dtype_max: float
    low: float
    high: float

    def to_plane(self, value: float) -> float:
        """A file intensity as a working-plane value (not clipped to 0..255)."""
        return (float(value) - self.low) / (self.high - self.low) * 255.0

    def to_file(self, value: Any) -> Any:
        """Working-plane values as file intensities."""
        return self.low + np.asarray(value, dtype=np.float64) / 255.0 * (self.high - self.low)


def _intensity_range(dtype: np.dtype, low: float, high: float,
                     observed: tuple[float, float] | None = None) -> IntensityRange:
    if dtype == np.bool_:
        span = (0.0, 1.0)
    elif dtype.kind in "ui":
        info = np.iinfo(dtype)
        span = (float(info.min), float(info.max))
    else:
        span = observed if observed is not None else (low, high)
    if not high > low:  # a flat page reads as zeros
        high = low + 1.0
    return IntensityRange(str(dtype), span[0], span[1], float(low), float(high))


def _array_range(page: np.ndarray) -> IntensityRange:
    """The stretch :func:`_page_image` applies to *page* (and ``_page_channel``
    to a gray page): identity for 8-bit and boolean samples, else its own
    minimum to maximum."""
    array = np.asarray(page)
    if array.dtype == np.bool_:
        return _intensity_range(array.dtype, 0.0, 1.0)
    if array.dtype == np.uint8:
        return _intensity_range(array.dtype, 0.0, 255.0)
    low, high = float(array.min()), float(array.max())
    return _intensity_range(array.dtype, low, high, (low, high))


def _pil_range(image: Image.Image) -> IntensityRange:
    """The stretch :func:`normalize_image` applies to *image*."""
    if image.mode in ("I", "I;16", "I;16B", "I;32", "F"):
        array = np.asarray(image)
        low, high = float(array.min()), float(array.max())
        return _intensity_range(array.dtype, low, high, (low, high))
    if image.mode == "1":
        return _intensity_range(np.dtype(np.bool_), 0.0, 1.0)
    return _intensity_range(np.dtype(np.uint8), 0.0, 255.0)


def working_intensity_ranges(
    path: str | Path, min_edge: int = WORKING_MIN_EDGE,
) -> list[IntensityRange]:
    """One :class:`IntensityRange` per page :func:`read_working_pages` reads.

    It reads the same level or page the working copy is drawn from (a
    decode of the file; callers cache it), so a working-plane value maps
    back to the file intensity it was stretched from. A single-page file
    gives one range, shared by its colour planes.
    """
    if page_count(path) <= 1:
        if Path(path).suffix.lower() in (".tif", ".tiff"):
            import tifffile

            with tifffile.TiffFile(path) as handle:
                series = handle.series[0] if handle.series else None
                if (series is not None and series.axes in ("YX", "YXS")
                        and series.dtype in (np.uint8, np.uint16)):
                    levels = list(series.levels)
                    chosen = levels[0]
                    for level in levels[1:]:
                        if max(level.shape[0], level.shape[1]) >= min_edge:
                            chosen = level
                    array = np.asarray(chosen.asarray())
                    if array.ndim == 2 or array.shape[-1] in (3, 4):
                        return [_array_range(array)]
        with Image.open(path) as handle:
            return [_pil_range(handle)]
    import tifffile

    with tifffile.TiffFile(path) as handle:
        series = handle.series[0] if handle.series else None
        levels = list(series.levels) if series is not None else []
        if (series is not None and len(levels) > 1 and len(series.axes) == 3
                and series.axes.endswith("YX")):
            chosen = levels[0]
            for level in levels[1:]:
                if max(level.shape[-2], level.shape[-1]) >= min_edge:
                    chosen = level
            return [_array_range(plane) for plane in np.asarray(chosen.asarray())]
        return [_array_range(np.asarray(page.asarray())) for page in handle.pages]


#: Threads the ITK filters run N4 and denoising on: a multithreaded filter's
#: last bits depend on the split, so the count is fixed.
ITK_THREADS = 8
#: N4: the image and mask shrunk
#: by 4, four levels of 50 iterations, tolerance 1e-7, a B-spline with 4
#: control points per axis (mesh 1, order 3).
N4_SHRINK = 4
N4_ITERATIONS = (50, 50, 50, 50)
N4_TOLERANCE = 1e-7
N4_CONTROL_POINTS = 4
#: Denoising: ITK's patch-based filter, 3x3 patches, 50 sampled patches.
DENOISE_PATCH_RADIUS = 1
DENOISE_SAMPLE_PATCHES = 50


def _n4_corrected(array: np.ndarray, tissue: np.ndarray | None) -> np.ndarray:
    """N4 bias-field correction of a positive float32 plane, inside *tissue*."""
    import SimpleITK as sitk

    image = sitk.GetImageFromArray(array)
    mask = sitk.GetImageFromArray(
        (tissue if tissue is not None else np.ones(array.shape, bool)).astype(np.uint8)
    )
    shrunk = sitk.Shrink(image, [N4_SHRINK] * 2)
    shrunk_mask = sitk.Shrink(mask, [N4_SHRINK] * 2)
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetNumberOfThreads(ITK_THREADS)
    n4.SetMaximumNumberOfIterations(list(N4_ITERATIONS))
    n4.SetConvergenceThreshold(N4_TOLERANCE)
    n4.SetNumberOfControlPoints([N4_CONTROL_POINTS] * 2)
    n4.Execute(shrunk, shrunk_mask)
    # The field is fitted on the shrunk plane and evaluated on the whole one.
    field = n4.GetLogBiasFieldAsImage(image)
    return sitk.GetArrayFromImage(image / sitk.Exp(field)).astype(np.float32)


def _denoised(array: np.ndarray) -> np.ndarray:
    """ITK's patch-based (non-local means) denoising of a float32 plane."""
    import SimpleITK as sitk

    if float(array.max()) == float(array.min()):
        return array  # the filter refuses a constant image; there is no noise
    denoise = sitk.PatchBasedDenoisingImageFilter()
    denoise.SetNumberOfThreads(ITK_THREADS)
    denoise.SetPatchRadius(DENOISE_PATCH_RADIUS)
    denoise.SetNumberOfSamplePatches(DENOISE_SAMPLE_PATCHES)
    denoise.SetNoiseModel(denoise.NOMODEL)
    return sitk.GetArrayFromImage(denoise.Execute(sitk.GetImageFromArray(array))).astype(np.float32)


def enhance_plane(plane: np.ndarray, *, n4: bool, denoise: bool) -> np.ndarray:
    """N4 bias-field correction and/or denoising of one 8-bit plane (SimpleITK).

    Intensities are shifted positive first (N4 works on their logarithm) and
    the result is scaled back to the plane's own mean inside the tissue, so a
    dim channel stays dim; outside the tissue the plane is kept.
    """
    if not (n4 or denoise):
        return plane
    array = plane.astype(np.float32) + 1.0
    tissue = plane > 15
    out = array
    if n4:
        out = _n4_corrected(out, tissue if tissue.any() else None)
    if denoise:
        out = _denoised(out)
    region = tissue if tissue.any() else np.ones(plane.shape, dtype=bool)
    # Keep the channel's own brightness: a dim channel stays dim, so the
    # weights still mean what they say and its noise is not stretched up.
    level = float(out[region].mean())
    if not np.isfinite(level) or level <= 0:
        return plane
    scaled = out * (float(array[region].mean()) / level) - 1.0
    return np.where(region, np.clip(scaled, 0, 255), plane).astype(np.uint8)


def custom_appearance(
    planes: list[np.ndarray],
    *,
    channel_weights: list[float] | None = None,
    clahe_clip: float = HOST_CLAHE_CLIP["medium"],
    clahe_tiles: int = HOST_CLAHE_TILE[0],
    n4: bool = False,
    denoise: bool = False,
) -> Image.Image:
    """The grayscale image an explicit appearance draws from raw planes.

    Each plane with weight above zero is optionally N4-corrected and denoised
    (:func:`enhance_plane`), CLAHE-enhanced at *clahe_clip* over a
    *clahe_tiles* x *clahe_tiles* grid (0 = no CLAHE), then the planes are
    blended by *channel_weights* (None = :func:`coverage_weights`) and
    brightened like the automatic path. 8-bit RGB with three equal channels.
    """
    weights = list(channel_weights) if channel_weights is not None else coverage_weights(planes)
    if len(weights) != len(planes):
        raise ValueError(f"Expected {len(planes)} channel weight(s); got {len(weights)}")
    used = [
        enhance_plane(plane, n4=n4, denoise=denoise) if weight > 0 else plane
        for plane, weight in zip(planes, weights, strict=True)
    ]
    tiles = max(1, int(clahe_tiles))
    return _blend_channels(
        used, weights, clahe_clip=clahe_clip if clahe_clip > 0 else None, tile=(tiles, tiles),
    )


#: Margin left around the atlas anatomy when framing it, as a fraction of
#: the anatomy's long side.
FRAME_MARGIN = 0.06


def mask_box(
    size: tuple[int, int], mask: np.ndarray, *, margin: float = FRAME_MARGIN
) -> tuple[int, int, int, int] | None:
    """The bounding box of *mask* plus *margin*, on an image of *size*, or None.

    *mask* may be measured at any resolution; its coordinates are scaled onto
    *size*.

    None when the mask is empty or the box would be degenerate (the image is
    then kept whole).
    """
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return None
    height, width = mask.shape[:2]
    scale_x, scale_y = size[0] / float(width), size[1] / float(height)
    x0, x1 = float(xs.min()) * scale_x, (float(xs.max()) + 1.0) * scale_x
    y0, y1 = float(ys.min()) * scale_y, (float(ys.max()) + 1.0) * scale_y
    # One padding distance for both axes so framing cannot change the apparent
    # aspect ratio.
    pad = margin * max(x1 - x0, y1 - y0)
    box = (
        max(0, int(x0 - pad)),
        max(0, int(y0 - pad)),
        min(size[0], int(round(x1 + pad))),
        min(size[1], int(round(y1 + pad))),
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return None
    return box


def prepared_size(
    size: tuple[int, int],
    *,
    max_pixels: int = DEFAULT_VLM_MAX_PIXELS,
    max_long_edge: int = DEFAULT_VLM_MAX_LONG_EDGE,
) -> tuple[int, int]:
    """The ``(width, height)`` :func:`prepare_image_for_vlm` gives an image of
    *size*: aspect kept, never upsampled, each side floored."""
    if max_pixels <= 0 or max_long_edge <= 0:
        raise ValueError("max_pixels and max_long_edge must be positive")
    width, height = int(size[0]), int(size[1])
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    pixel_scale = math.sqrt(min(1.0, max_pixels / float(width * height)))
    edge_scale = min(1.0, max_long_edge / float(max(width, height)))
    scale_factor = min(1.0, pixel_scale, edge_scale)
    if scale_factor >= 0.999999:
        return width, height
    return (max(1, int(math.floor(width * scale_factor))),
            max(1, int(math.floor(height * scale_factor))))


def prepare_image_for_vlm(
    image: Image.Image,
    *,
    max_pixels: int = DEFAULT_VLM_MAX_PIXELS,
    max_long_edge: int = DEFAULT_VLM_MAX_LONG_EDGE,
) -> Image.Image:
    """Downsample an image for VLM use while preserving aspect ratio
    (:func:`prepared_size`)."""
    size = prepared_size(image.size, max_pixels=max_pixels, max_long_edge=max_long_edge)
    return image if size == image.size else image.resize(size, _RESAMPLE_LANCZOS)
