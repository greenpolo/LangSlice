"""Image ingest helpers for VLM preparation."""

from __future__ import annotations

import math
from dataclasses import dataclass

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
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return image
    height, width = mask.shape[:2]
    scale_x, scale_y = image.width / float(width), image.height / float(height)
    x0, x1 = float(xs.min()) * scale_x, (float(xs.max()) + 1.0) * scale_x
    y0, y1 = float(ys.min()) * scale_y, (float(ys.max()) + 1.0) * scale_y
    # One padding distance for both axes so framing cannot change the apparent
    # aspect ratio of the tissue.
    pad = margin * max(x1 - x0, y1 - y0)
    box = (
        max(0, int(x0 - pad)),
        max(0, int(y0 - pad)),
        min(image.width, int(round(x1 + pad))),
        min(image.height, int(round(y1 + pad))),
    )
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        return image
    return image.crop(box)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """The biggest connected blob in *mask*, or *mask* itself if labeling fails.

    A slide carries more than the section: a neighbouring fragment, a dust
    speck, a pen mark. The global bounding box stretches around all of it and
    the section comes back rendered at a fraction of the frame — the worst
    framing in a benchmark run was also its worst position.
    """
    # ponytail: biggest blob wins, so a section torn clean in two keeps only
    # the larger half. Dilate before labeling if that ever costs more than
    # debris does.
    import cv2

    try:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8), connectivity=8
        )
        if count <= 2:  # background plus at most one blob: nothing to choose
            return mask
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == biggest
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
    around the LARGEST connected blob only, so a neighbouring fragment or a
    speck of debris cannot drag the frame open. Falls back to the untouched
    image when the result would be degenerate.
    """
    proxy = image.convert("L")
    long_edge = max(proxy.size)
    if long_edge > _FRAME_PROXY_EDGE:
        scale = _FRAME_PROXY_EDGE / float(long_edge)
        proxy = proxy.resize(
            (max(1, round(proxy.width * scale)), max(1, round(proxy.height * scale))),
            Image.Resampling.BILINEAR,
        )
    arr = np.asarray(proxy, dtype=np.float32)
    if arr.size == 0:
        return image
    border = np.concatenate(
        [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]
    )
    background = float(np.median(border))
    spread = max(float(arr.max()) - float(arr.min()), 1.0)
    mask = np.abs(arr - background) > max(8.0, 0.12 * spread)
    covered = float(mask.mean())
    if covered < 0.005 or covered > 0.98:
        return image
    return crop_to_mask(image, _largest_component(mask), margin=margin)


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
