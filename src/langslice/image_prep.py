"""Image ingest helpers for VLM preparation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

# Whole-slide microscopy exports routinely exceed PIL's ~179-megapixel
# decompression-bomb guard, which is meant for untrusted web content; our
# inputs are the user's local acquisitions.
Image.MAX_IMAGE_PIXELS = None

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS

DEFAULT_VLM_MAX_PIXELS = 12_000_000
DEFAULT_VLM_MAX_LONG_EDGE = 4096


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    scale_factor: float
    effective_pixel_size_um: float | None = None

    @property
    def downsampled(self) -> bool:
        return self.scale_factor < 0.999999


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

    Two sources, in order: the OME-XML in the TIFF ImageDescription
    (``PhysicalSizeX`` plus its unit) and the TIFF resolution tags
    (``XResolution`` interpreted through ``ResolutionUnit``: 2 = inch,
    3 = centimetre; 1 means "no unit", i.e. no calibration). Anything else —
    a PNG export, a JPEG, a TIFF written without resolution — answers None,
    and the caller says so rather than inventing a scale.
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

    resolution, unit_code = tags.get(282), int(tags.get(296, 2) or 2)
    per_unit_um = {2: 25400.0, 3: 10000.0}.get(unit_code)
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


#: Brightfield CLAHE strength: stronger clip and finer tiles than the
#: fluorescence path, because absorbance stains (Nissl, ISH) carry their
#: structure in low-contrast density variations.
_BRIGHTFIELD_CLAHE_CLIP = 6.0
_BRIGHTFIELD_CLAHE_TILE = (16, 16)
#: Optical-density blend: violet/purple stains absorb green light most, so
#: the green channel carries the sharpest stain-density signal.
_BRIGHTFIELD_DENSITY_WEIGHTS = (0.2, 0.6, 0.2)
#: Densities at or below this are slide background; CLAHE output is not
#: applied there, so empty regions stay clean instead of amplifying slide
#: texture into gray blotches.
_BRIGHTFIELD_BG_DENSITY = 8


def _brightfield_preprocess(arr: np.ndarray) -> Image.Image:
    """Brightfield (absorbance) variant: CLAHE on optical density.

    Structure in brightfield is stain DENSITY, not brightness, so the image
    is inverted to per-channel optical density, blended green-heavy, contrast
    enhanced, and re-inverted — output keeps the native dark-tissue-on-light
    polarity.
    """
    import cv2

    od = 255.0 - arr.astype(np.float32)
    wr, wg, wb = _BRIGHTFIELD_DENSITY_WEIGHTS
    density = np.clip(wr * od[..., 0] + wg * od[..., 1] + wb * od[..., 2], 0, 255).astype(
        np.uint8
    )
    clahe = cv2.createCLAHE(
        clipLimit=_BRIGHTFIELD_CLAHE_CLIP, tileGridSize=_BRIGHTFIELD_CLAHE_TILE
    )
    enhanced = clahe.apply(density)
    enhanced = np.where(density > _BRIGHTFIELD_BG_DENSITY, enhanced, density)
    out = (255 - enhanced).astype(np.uint8)
    return Image.fromarray(np.stack([out, out, out], axis=-1))


def adaptive_preprocess(
    image: Image.Image,
    *,
    mode: str = "auto",
    clahe_clip: float = 4.0,
    clahe_tile: tuple[int, int] = (8, 8),
    target_brightness: float = 90.0,
    max_boost: float = 3.0,
    channel_weights: tuple[float, float, float] | None = None,
) -> Image.Image:
    """Adaptive preprocessing for VLM input: CLAHE + weighted blend + brightness.

    Two paths, selected by ``mode``:

    - ``"fluorescence"`` — per-channel CLAHE, a blend weighted toward the
      STRUCTURAL channel (auto-selected by tissue coverage when
      *channel_weights* is None: DAPI-blue, YFP-green, whatever lights the
      whole tissue; sparse tracer channels get little weight), brightness
      boost toward *target_brightness*.
    - ``"brightfield"`` — absorbance stains (Nissl, ISH): CLAHE runs on
      optical density with stronger clip and finer tiles
      (:func:`_brightfield_preprocess`), keeping dark-on-light polarity.
    - ``"auto"`` (default) — brightfield when the image border is bright
      (a light slide background), fluorescence otherwise.
    """
    import cv2

    arr = np.asarray(normalize_image(image), dtype=np.uint8)

    if mode not in ("auto", "fluorescence", "brightfield"):
        raise ValueError(f"Unknown adaptive_preprocess mode: {mode!r}")
    if mode == "auto":
        ring = max(2, int(round(0.02 * max(arr.shape[:2]))))
        border = np.concatenate(
            [
                arr[:ring].reshape(-1, 3),
                arr[-ring:].reshape(-1, 3),
                arr[:, :ring].reshape(-1, 3),
                arr[:, -ring:].reshape(-1, 3),
            ]
        )
        mode = "brightfield" if float(np.median(border)) > 140.0 else "fluorescence"
    if mode == "brightfield":
        return _brightfield_preprocess(arr)

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    # Structural-channel selection: the structural stain (DAPI, YFP fills,
    # autofluorescence) lights up MOST of the tissue; sparse tracer channels
    # light small patches brightly. Weight channels by squared tissue
    # coverage so broad channels dominate regardless of which color they
    # are; the DAPI-blue default only held for DAPI-stained images.
    tissue = arr.max(axis=2) > 15
    if channel_weights is None:
        if tissue.any():
            cov = np.array(
                [float((c[tissue] > 40).mean()) for c in (r, g, b)], dtype=np.float64
            )
            w = cov**2
            if w.sum() > 0:
                wn = w / w.sum()
                channel_weights = (float(wn[0]), float(wn[1]), float(wn[2]))
            else:
                channel_weights = (1 / 3, 1 / 3, 1 / 3)
        else:
            channel_weights = (1 / 3, 1 / 3, 1 / 3)

    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_tile)
    r_enh = clahe.apply(r).astype(np.float32)
    g_enh = clahe.apply(g).astype(np.float32)
    b_enh = clahe.apply(b).astype(np.float32)

    wr, wg, wb = channel_weights
    blended = wr * r_enh + wg * g_enh + wb * b_enh
    blended = np.clip(blended, 0, 255)

    brain_mask = blended > 10
    if brain_mask.sum() > 0:
        current_mean = float(blended[brain_mask].mean())
        boost = min(target_brightness / max(current_mean, 1.0), max_boost)
    else:
        boost = 1.0

    if boost > 1.05:
        blended = blended * boost

    blended = np.clip(blended, 0, 255).astype(np.uint8)
    gray_rgb = np.stack([blended, blended, blended], axis=-1)
    return Image.fromarray(gray_rgb)


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
    brightness boost: :func:`adaptive_preprocess`'s fluorescence path for any
    number of channels."""
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
    - auto, several pages: the same method over N channels — brightfield
      (optical-density CLAHE on the mean channel) when the image border is
      bright, else per-channel CLAHE blended by squared tissue coverage (the
      channel lighting most of the tissue dominates) with the brightness boost.
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

    stack = np.stack(channels, axis=-1)
    ring = max(2, int(round(0.02 * max(stack.shape[:2]))))
    border = np.concatenate([
        stack[:ring].reshape(-1), stack[-ring:].reshape(-1),
        stack[:, :ring].reshape(-1), stack[:, -ring:].reshape(-1),
    ])
    if float(np.median(border)) > 140.0:
        gray = np.round(stack.astype(np.float32).mean(axis=-1)).astype(np.uint8)
        return _brightfield_preprocess(np.stack([gray, gray, gray], axis=-1))
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


def host_preprocess_file(
    path: str | Path, settings: dict[str, object] | None = None
) -> Image.Image:
    """:func:`host_preprocess` over every page of the file at *path*."""
    return host_preprocess(read_pages(path), settings)


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


def ants_enhance(plane: np.ndarray, *, n4: bool, denoise: bool) -> np.ndarray:
    """ANTs N4 bias-field correction and/or denoising of one 8-bit plane.

    antspyx ships in LangSlice's optional ``registration`` extra and is
    imported only here; without it this raises with the install hint.
    Intensities are shifted positive first (N4 works on their logarithm) and
    the result is scaled back to the plane's own mean inside the tissue, so a
    dim channel stays dim; outside the tissue the plane is kept.
    """
    if not (n4 or denoise):
        return plane
    try:
        import ants
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "N4 and denoising need antspyx: install LangSlice's 'registration' "
            "extra (pip install 'langslice[registration]')"
        ) from exc
    array = plane.astype(np.float32) + 1.0
    image = ants.from_numpy(np.ascontiguousarray(array))
    tissue = plane > 15
    mask = (
        ants.from_numpy(np.ascontiguousarray(tissue.astype(np.float32)))
        if tissue.any() else None
    )
    if n4:
        image = ants.n4_bias_field_correction(image, mask=mask)
    if denoise:
        image = ants.denoise_image(image, mask=mask)
    out = np.asarray(image.numpy(), dtype=np.float32)
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
    clahe_clip: float = 4.0,
    clahe_tiles: int = HOST_CLAHE_TILE[0],
    n4: bool = False,
    denoise: bool = False,
) -> Image.Image:
    """The grayscale image an explicit appearance draws from raw planes.

    Each plane with weight above zero is optionally N4-corrected and denoised
    (:func:`ants_enhance`), CLAHE-enhanced at *clahe_clip* over a
    *clahe_tiles* x *clahe_tiles* grid (0 = no CLAHE), then the planes are
    blended by *channel_weights* (None = :func:`coverage_weights`) and
    brightened like the automatic path. 8-bit RGB with three equal channels.
    """
    weights = list(channel_weights) if channel_weights is not None else coverage_weights(planes)
    if len(weights) != len(planes):
        raise ValueError(f"Expected {len(planes)} channel weight(s); got {len(weights)}")
    used = [
        ants_enhance(plane, n4=n4, denoise=denoise) if weight > 0 else plane
        for plane, weight in zip(planes, weights, strict=True)
    ]
    tiles = max(1, int(clahe_tiles))
    return _blend_channels(
        used, weights, clahe_clip=clahe_clip if clahe_clip > 0 else None, tile=(tiles, tiles),
    )


#: Margin left around the foreground when framing, as a fraction of the
#: foreground's long side.
FRAME_MARGIN = 0.06

#: Long edge of the proxy the tissue silhouette is measured on. Otsu at full
#: whole-slide resolution costs seconds and buys nothing: a bounding box is a
#: bounding box.
_FRAME_PROXY_EDGE = 256


def crop_to_mask(
    image: Image.Image, mask: np.ndarray, *, margin: float = FRAME_MARGIN
) -> Image.Image:
    """Crop *image* to *mask*'s bounding box plus *margin*.

    *mask* may be measured at any resolution; its coordinates are scaled onto
    *image*. Returns *image* untouched when the mask is empty or the resulting
    box is degenerate.
    """
    box = mask_box(image.size, mask, margin=margin)
    return image if box is None else image.crop(box)


def mask_box(
    size: tuple[int, int], mask: np.ndarray, *, margin: float = FRAME_MARGIN
) -> tuple[int, int, int, int] | None:
    """The crop box :func:`crop_to_mask` uses on an image of *size*, or None.

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
    # aspect ratio of the tissue.
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


#: A blob at least this fraction of the biggest one's area is part of the
#: section (a second olfactory bulb, a cerebellum parted from the brainstem, a
#: torn-off piece); anything smaller is debris.
SECTION_PIECE_FRACTION = 0.2


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """The section's blobs in *mask*: the biggest plus any piece of comparable size.

    A slide carries more than the section: a dust speck, a pen mark, a sliver
    of a neighbouring section. The global bounding box stretches around all of
    it and the section comes back rendered at a fraction of the frame — the
    worst framing in a benchmark run was also its worst position. But a
    section is often in pieces (bulbs cut apart, the cerebellum parted from
    the brainstem, a tear): keeping only the biggest blob showed the model one
    bulb, or a cerebellum with no brainstem, and those sections were placed
    200-400 um off. So every blob of at least :data:`SECTION_PIECE_FRACTION`
    of the biggest one's area is kept.
    """
    import cv2

    try:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 2:  # background plus at most one blob: nothing to choose
            return mask
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = 1 + np.flatnonzero(areas >= SECTION_PIECE_FRACTION * areas.max())
        return np.isin(labels, keep)
    except Exception:  # a labeling failure must not cost us the framing
        return mask


def crop_to_tissue(image: Image.Image, *, margin: float = FRAME_MARGIN) -> Image.Image:
    """Crop a histology section to its tissue plus a margin.

    Framing, not enhancement: histology arrives filling most of its frame while
    a fixed-canvas atlas render leaves the brain small and centred, and that
    difference in apparent scale is itself a cue a model will read as anatomy.
    Cropping both to their foreground makes the two comparable.

    Foreground is "different from the background", with the background level
    read off the frame's own border — which works for dark-on-light brightfield
    and light-on-dark fluorescence alike, and (unlike an Otsu split) will not
    mistake a dim half of the tissue for background. The box is then taken
    around the section's own pieces (every blob of at least
    :data:`SECTION_PIECE_FRACTION` of the biggest one), so a speck of debris
    cannot drag the frame open while a section in two pieces stays whole.
    Falls back to the untouched image when the result would be degenerate.
    """
    box = tissue_box(image, margin=margin)
    return image if box is None else image.crop(box)


def tissue_box(
    image: Image.Image, *, margin: float = FRAME_MARGIN
) -> tuple[int, int, int, int] | None:
    """The crop box :func:`crop_to_tissue` uses, or None to keep the image whole."""
    mask = foreground_mask(image)
    if mask is None:
        return None
    return mask_box(image.size, mask, margin=margin)


def foreground_mask(
    image: Image.Image, *, proxy_edge: int = _FRAME_PROXY_EDGE
) -> np.ndarray | None:
    """Boolean tissue mask for *image*, or None when detection is degenerate.

    The mask is measured on a small proxy (long edge *proxy_edge*, by default
    ``_FRAME_PROXY_EDGE``) and returned at that proxy resolution — scale it
    onto whatever frame you need, as :func:`crop_to_mask` does. Foreground
    rule and section-piece selection are shared with :func:`crop_to_tissue`.
    The deformable fit asks for a finer proxy, because its masks bound a
    metric rather than a framing box.
    """
    proxy = image.convert("L")
    long_edge = max(proxy.size)
    if long_edge > proxy_edge:
        scale = proxy_edge / float(long_edge)
        proxy = proxy.resize(
            (max(1, round(proxy.width * scale)), max(1, round(proxy.height * scale))),
            Image.Resampling.BILINEAR,
        )
    arr = np.asarray(proxy, dtype=np.float32)
    if arr.size == 0:
        return None
    border = np.concatenate(
        [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]
    )
    background = float(np.median(border))
    spread = max(float(arr.max()) - float(arr.min()), 1.0)
    mask = np.abs(arr - background) > max(8.0, 0.12 * spread)
    covered = float(mask.mean())
    if covered < 0.005 or covered > 0.98:
        return None
    return _largest_component(mask)


def prepare_image_for_vlm(
    image: Image.Image,
    *,
    pixel_size_um: float | None = None,
    max_pixels: int = DEFAULT_VLM_MAX_PIXELS,
    max_long_edge: int = DEFAULT_VLM_MAX_LONG_EDGE,
) -> PreparedImage:
    """Downsample an image for VLM use while preserving aspect ratio."""
    if max_pixels <= 0 or max_long_edge <= 0:
        raise ValueError("max_pixels and max_long_edge must be positive")

    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    pixel_scale = math.sqrt(min(1.0, max_pixels / float(width * height)))
    edge_scale = min(1.0, max_long_edge / float(max(width, height)))
    scale_factor = min(1.0, pixel_scale, edge_scale)

    if scale_factor >= 0.999999:
        return PreparedImage(
            image=image,
            original_size=(width, height),
            output_size=(width, height),
            scale_factor=1.0,
            effective_pixel_size_um=float(pixel_size_um) if pixel_size_um is not None else None,
        )

    new_width = max(1, int(math.floor(width * scale_factor)))
    new_height = max(1, int(math.floor(height * scale_factor)))
    resized = image.resize((new_width, new_height), _RESAMPLE_LANCZOS)

    effective_pixel_size_um = None
    if pixel_size_um is not None:
        effective_pixel_size_um = float(pixel_size_um) * (float(width) / float(new_width))

    return PreparedImage(
        image=resized,
        original_size=(width, height),
        output_size=(new_width, new_height),
        scale_factor=float(new_width) / float(width),
        effective_pixel_size_um=effective_pixel_size_um,
    )
