"""Image ingest helpers for metadata detection and VLM preparation."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")

DEFAULT_VLM_MAX_PIXELS = 12_000_000
DEFAULT_VLM_MAX_LONG_EDGE = 4096


@dataclass(frozen=True)
class PixelSizeDetection:
    pixel_size_um: float
    source: str


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


@dataclass(frozen=True)
class LoadedImageState:
    canonical_image: Image.Image
    vlm_image: Image.Image
    pixel_size_um: float
    pixel_size_source: str
    metadata_pixel_size_um: float | None
    original_size: tuple[int, int]
    vlm_size: tuple[int, int]
    vlm_scale_factor: float
    vlm_effective_pixel_size_um: float

    def with_pixel_size(self, pixel_size_um: float, source: str) -> "LoadedImageState":
        effective = pixel_size_um / self.vlm_scale_factor
        return replace(
            self,
            pixel_size_um=float(pixel_size_um),
            pixel_size_source=source,
            vlm_effective_pixel_size_um=float(effective),
        )


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


def _rational_to_float(value: object) -> float | None:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        numerator = _rational_to_float(value[0])
        denominator = _rational_to_float(value[1])
        if numerator is None or denominator is None or abs(denominator) <= 1e-12:
            return None
        return numerator / denominator
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _tag_value(tag: object) -> object | None:
    return getattr(tag, "value", None)


def _page_tags(page: object) -> object | None:
    return getattr(page, "tags", None)


def _tag_lookup(tags: object, name: str) -> object | None:
    getter = getattr(tags, "get", None)
    if callable(getter):
        return getter(name)
    return None


def _ome_length_unit_to_um(unit: str | None) -> float | None:
    if unit is None or unit == "" or unit in {"µm", "um", "micrometer", "micrometre"}:
        return 1.0
    normalized = unit.strip().lower()
    if normalized in {"µm", "um", "micrometer", "micrometre"}:
        return 1.0
    if normalized in {"nm", "nanometer", "nanometre"}:
        return 0.001
    if normalized in {"mm", "millimeter", "millimetre"}:
        return 1000.0
    if normalized in {"cm", "centimeter", "centimetre"}:
        return 10000.0
    if normalized in {"m", "meter", "metre"}:
        return 1_000_000.0
    return None


def _parse_ome_pixel_size_um(metadata: str) -> PixelSizeDetection | None:
    try:
        root = ET.fromstring(metadata)
    except ET.ParseError:
        return None

    pixels_element = next((elem for elem in root.iter() if elem.tag.endswith("Pixels")), None)
    if pixels_element is None:
        return None

    size_x = pixels_element.attrib.get("PhysicalSizeX")
    size_y = pixels_element.attrib.get("PhysicalSizeY")
    if size_x is None or size_y is None:
        return None

    x_value = _rational_to_float(size_x)
    y_value = _rational_to_float(size_y)
    if x_value is None or y_value is None or x_value <= 0.0 or y_value <= 0.0:
        return None

    x_scale = _ome_length_unit_to_um(pixels_element.attrib.get("PhysicalSizeXUnit"))
    y_scale = _ome_length_unit_to_um(pixels_element.attrib.get("PhysicalSizeYUnit"))
    if x_scale is None or y_scale is None:
        return None

    x_um = x_value * x_scale
    y_um = y_value * y_scale
    if not math.isclose(x_um, y_um, rel_tol=1e-3, abs_tol=1e-6):
        return None

    return PixelSizeDetection(pixel_size_um=float((x_um + y_um) / 2.0), source="ome_metadata")


def _resolution_unit_scale_to_um(unit: object) -> float | None:
    if isinstance(unit, (int, float)):
        value = int(unit)
    elif isinstance(unit, str):
        try:
            value = int(unit)
        except ValueError:
            return None
    else:
        return None
    if value == 2:
        return 25400.0
    if value == 3:
        return 10000.0
    return None


def _pixel_size_from_tiff_tags(file_path: str) -> PixelSizeDetection | None:
    try:
        import tifffile
    except ImportError:
        return None

    with tifffile.TiffFile(file_path) as tif:
        ome_metadata = getattr(tif, "ome_metadata", None)
        if isinstance(ome_metadata, str):
            ome_detection = _parse_ome_pixel_size_um(ome_metadata)
            if ome_detection is not None:
                return ome_detection

        if not tif.pages:
            return None

        page = tif.pages[0]
        tags = _page_tags(page)
        if tags is None:
            return None

        x_tag = _tag_lookup(tags, "XResolution")
        y_tag = _tag_lookup(tags, "YResolution")
        unit_tag = _tag_lookup(tags, "ResolutionUnit")
        if x_tag is None or y_tag is None or unit_tag is None:
            return None

        x_resolution = _rational_to_float(_tag_value(x_tag))
        y_resolution = _rational_to_float(_tag_value(y_tag))
        unit_scale_um = _resolution_unit_scale_to_um(_tag_value(unit_tag))
        if (
            x_resolution is None
            or y_resolution is None
            or unit_scale_um is None
            or x_resolution <= 0.0
            or y_resolution <= 0.0
        ):
            return None

        x_um = unit_scale_um / x_resolution
        y_um = unit_scale_um / y_resolution
        if not math.isclose(x_um, y_um, rel_tol=1e-3, abs_tol=1e-6):
            return None

        return PixelSizeDetection(pixel_size_um=float((x_um + y_um) / 2.0), source="tiff_resolution")


def detect_pixel_size_um(file_path: str) -> PixelSizeDetection | None:
    """Best-effort pixel-size detection in microns per pixel."""
    lower_path = file_path.lower()
    if lower_path.endswith((".tif", ".tiff")):
        return _pixel_size_from_tiff_tags(file_path)
    return None


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


def load_image_state(
    file_path: str,
    *,
    fallback_pixel_size_um: float,
    max_pixels: int = DEFAULT_VLM_MAX_PIXELS,
    max_long_edge: int = DEFAULT_VLM_MAX_LONG_EDGE,
) -> LoadedImageState:
    """Load an image, detect pixel size, and derive a VLM-ready image."""
    with Image.open(file_path) as loaded:
        canonical_image = normalize_image(loaded.copy())

    detection = detect_pixel_size_um(file_path)
    if detection is not None:
        pixel_size_um = detection.pixel_size_um
        pixel_size_source = detection.source
        metadata_pixel_size_um = detection.pixel_size_um
    else:
        pixel_size_um = float(fallback_pixel_size_um)
        pixel_size_source = "manual_existing"
        metadata_pixel_size_um = None

    vlm_prep = prepare_image_for_vlm(
        canonical_image,
        pixel_size_um=pixel_size_um,
        max_pixels=max_pixels,
        max_long_edge=max_long_edge,
    )

    effective_pixel_size_um = vlm_prep.effective_pixel_size_um
    if effective_pixel_size_um is None:
        effective_pixel_size_um = pixel_size_um

    return LoadedImageState(
        canonical_image=canonical_image,
        vlm_image=vlm_prep.image,
        pixel_size_um=float(pixel_size_um),
        pixel_size_source=pixel_size_source,
        metadata_pixel_size_um=metadata_pixel_size_um,
        original_size=canonical_image.size,
        vlm_size=vlm_prep.output_size,
        vlm_scale_factor=vlm_prep.scale_factor,
        vlm_effective_pixel_size_um=float(effective_pixel_size_um),
    )
