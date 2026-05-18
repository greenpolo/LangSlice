"""VLM-oriented image preprocessing primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .constants import DEFAULT_VLM_MAX_LONG_EDGE, DEFAULT_VLM_MAX_PIXELS

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS
_SCALE_THRESHOLD = 0.999999


@dataclass(frozen=True)
class PreparedImage:
    image: Image.Image
    original_size: tuple[int, int]
    output_size: tuple[int, int]
    scale_factor: float
    effective_pixel_size_um: float | None = None

    @property
    def downsampled(self) -> bool:
        return self.scale_factor < _SCALE_THRESHOLD


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


def adaptive_preprocess(
    image: Image.Image,
    *,
    clahe_clip: float = 4.0,
    clahe_tile: tuple[int, int] = (8, 8),
    target_brightness: float = 90.0,
    max_boost: float = 3.0,
    channel_weights: tuple[float, float, float] = (0.15, 0.15, 0.70),
) -> Image.Image:
    """Adaptive preprocessing for VLM input: CLAHE + weighted blend + brightness.

    Designed for fluorescent histology with DAPI (blue) as the structural
    channel and viral tracers in red/green.  Produces a consistent grayscale
    output that matches atlas appearance regardless of stain intensity.

    Steps:
        1. CLAHE on each R, G, B channel independently (local contrast)
        2. Weighted blend to grayscale (default 70% blue + 15% red + 15% green)
        3. Adaptive brightness boost to reach *target_brightness* mean
    """
    import cv2

    arr = np.asarray(normalize_image(image), dtype=np.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

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

    if scale_factor >= _SCALE_THRESHOLD:
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
