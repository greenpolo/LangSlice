"""VLM-based brain slice estimation using Gemini."""

import base64
import io
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, cast

import numpy as np
from PIL import Image
from PIL import ImageOps

from langslice.vlm.config import MODEL_NAME, THINKING_LEVEL, get_client

logger = logging.getLogger(__name__)

# Retryable exception types (google-genai SDK error hierarchy)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _retry_generate(
    client: Any,
    *,
    model: str,
    contents: list[dict[str, object]],
    config: dict[str, object],
) -> Any:
    """Wrapper around client.models.generate_content with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(
                model=model, contents=contents, config=config
            )
        except Exception as exc:
            last_exc = exc
            # Check if the error has a retryable HTTP status code
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_BACKOFF_S * (2 ** attempt)
                    logger.warning(
                        "Gemini API error (status %s), retrying in %.1fs (attempt %d/%d)",
                        status, delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
            # Also retry on generic connection / timeout errors
            exc_name = type(exc).__name__.lower()
            if any(kw in exc_name for kw in ("timeout", "connection", "transport")):
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_BACKOFF_S * (2 ** attempt)
                    logger.warning(
                        "Transient error (%s), retrying in %.1fs (attempt %d/%d)",
                        type(exc).__name__, delay, attempt + 1, _MAX_RETRIES,
                    )
                    time.sleep(delay)
                    continue
            # Non-retryable error — raise immediately
            raise
    # Exhausted retries
    assert last_exc is not None
    raise last_exc

@dataclass
class APResult:
    position_mm: float
    reasoning: str


@dataclass
class AffineResult:
    rotation: float
    translateX: float
    translateY: float
    reasoning: str


@dataclass
class PreprocessOptions:
    enabled: bool = False
    crop_tissue: bool = True
    normalize_contrast: bool = True
    max_side_px: int = 1280
    tissue_percentile: float = 92.0
    min_foreground_fraction: float = 0.01
    margin_fraction: float = 0.06


def _image_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    """Convert PIL Image to base64 string."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=95, subsampling=0)
    else:
        img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _resize_max_side(img: Image.Image, max_side_px: int) -> Image.Image:
    if max_side_px <= 0:
        return img
    width, height = img.size
    largest = max(width, height)
    if largest <= max_side_px:
        return img

    scale = max_side_px / float(largest)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return img.resize((new_width, new_height), _RESAMPLE_LANCZOS)


def _border_fill_value(gray: np.ndarray) -> int:
    if gray.size == 0:
        return 255
    top = gray[0, :]
    bottom = gray[-1, :]
    left = gray[:, 0]
    right = gray[:, -1]
    border = np.concatenate([top, bottom, left, right])
    return int(np.clip(np.median(border), 0, 255))


def _crop_tissue_region(img: Image.Image, options: PreprocessOptions) -> Image.Image:
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    if gray.size == 0:
        return img

    threshold = float(np.percentile(gray, options.tissue_percentile))
    mask = gray < threshold
    if float(mask.mean()) < options.min_foreground_fraction:
        return img

    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return img

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1

    margin = int(round(max(x1 - x0, y1 - y0) * options.margin_fraction))
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(img.width, x1 + margin)
    y1 = min(img.height, y1 + margin)

    cropped = img.crop((x0, y0, x1, y1))

    width, height = cropped.size
    if width == height:
        return cropped

    fill = _border_fill_value(gray)
    if width > height:
        pad_total = width - height
        top = pad_total // 2
        bottom = pad_total - top
        return ImageOps.expand(cropped, border=(0, top, 0, bottom), fill=(fill, fill, fill))

    pad_total = height - width
    left = pad_total // 2
    right = pad_total - left
    return ImageOps.expand(cropped, border=(left, 0, right, 0), fill=(fill, fill, fill))


def _prepare_vlm_image(image: Image.Image, options: PreprocessOptions) -> Image.Image:
    prepared = image.convert("RGB")
    if not options.enabled:
        return prepared

    if options.crop_tissue:
        prepared = _crop_tissue_region(prepared, options)

    prepared = _resize_max_side(prepared, options.max_side_px)

    if options.normalize_contrast:
        prepared = ImageOps.autocontrast(prepared, cutoff=1)

    return prepared


def _part_text(text: str) -> dict[str, object]:
    return {"text": text}


def _part_image_base64(image_b64: str) -> dict[str, object]:
    inline_data: object = {
        "mime_type": "image/jpeg",
        "data": image_b64,
    }
    return {"inline_data": inline_data}


def _contents_from_parts(parts: list[dict[str, object]]) -> list[dict[str, object]]:
    return [{"role": "user", "parts": parts}]


def _generation_config(schema: dict[str, object]) -> dict[str, object]:
    return {
        "thinking_config": {"thinking_level": THINKING_LEVEL},
        "response_mime_type": "application/json",
        "response_schema": schema,
    }


def _extract_result(response: object) -> dict[str, object]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        parsed_dict = cast(dict[object, object], parsed)
        normalized: dict[str, object] = {}
        for k, v in parsed_dict.items():
            normalized[str(k)] = v
        return normalized

    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        logger.warning("Gemini response did not expose parsed JSON; returning empty result.")
    return {}


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _to_str(value: object, default: str = "N/A") -> str:
    if isinstance(value, str):
        return value
    return default


def estimate_position(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    preprocess_options: PreprocessOptions | None = None,
) -> APResult:
    """
    Two-pass physical position estimation (mm from anterior edge).

    Pass 1 (coarse): Compare against 8 evenly spaced reference slices.
    Pass 2 (fine): Compare against 5 reference slices at 0.2mm intervals around coarse estimate.
    """
    from langslice.atlas.core import get_position_range_mm, get_reference_slice, load_atlas

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    options = preprocess_options or PreprocessOptions()

    client = get_client()
    atlas = load_atlas(atlas_name)
    pos_lower, pos_upper = get_position_range_mm(atlas)

    _progress("Fetching coarse reference images (evenly spaced)...")
    coarse_positions = np.linspace(pos_lower, pos_upper, 8).tolist()

    coarse_refs: list[tuple[float, Image.Image]] = []
    for pos in coarse_positions:
        try:
            ref_img = get_reference_slice(atlas, pos)
            coarse_refs.append((pos, ref_img))
        except ValueError:
            continue

    if not coarse_refs:
        _progress("Warning: No coarse reference images available. Estimating without visual references.")
    else:
        _progress(f"Fetched {len(coarse_refs)} coarse reference images.")

    if options.enabled:
        _progress("Applying preprocessing to target and reference images...")
    target_prepared = _prepare_vlm_image(image, options)
    target_b64 = _image_to_base64(target_prepared)
    coarse_parts: list[dict[str, object]] = [
        _part_text("Image 1 is the target brain slice you need to analyze."),
        _part_image_base64(target_b64),
    ]

    for i, (pos, ref_img) in enumerate(coarse_refs):
        coarse_parts.append(_part_text(f"Reference Image {i + 2}: Atlas slice at {pos:.2f} mm from anterior edge."))
        coarse_prepared = _prepare_vlm_image(ref_img, options)
        coarse_parts.append(_part_image_base64(_image_to_base64(coarse_prepared)))

    coarse_parts.append(
        _part_text(
            "Compare the target brain slice (Image 1) to the reference atlas slices. "
            + "Estimate the rough Anterior-Posterior (AP) position of the target slice "
            + f"in millimeters from the anterior edge of the volume (range: 0.0 to {pos_upper:.1f}mm)."
        )
    )

    _progress("Analyzing coarse position...")
    coarse_response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts(coarse_parts),
        config=_generation_config(
            {
                "type": "OBJECT",
                "properties": {
                    "position_mm": {
                        "type": "NUMBER",
                        "description": "Estimated position in mm from anterior edge",
                    },
                    "reasoning": {
                        "type": "STRING",
                        "description": "Reasoning for the estimation",
                    },
                },
            }
        ),
    )

    coarse_result = _extract_result(coarse_response)
    coarse_pos = _to_float(coarse_result.get("position_mm"), 0.0)
    _progress(
        f"Coarse position estimated at {coarse_pos:.2f}mm. Fetching fine references..."
    )

    fine_positions = [
        round(coarse_pos + 0.4, 2),
        round(coarse_pos + 0.2, 2),
        round(coarse_pos, 2),
        round(coarse_pos - 0.2, 2),
        round(coarse_pos - 0.4, 2),
    ]

    fine_refs: list[tuple[float, Image.Image]] = []
    for pos in fine_positions:
        if not (pos_lower <= pos <= pos_upper):
            continue
        try:
            ref_img = get_reference_slice(atlas, pos)
            fine_refs.append((pos, ref_img))
        except ValueError:
            continue

    if not fine_refs:
        _progress("Warning: No fine reference images available.")
    else:
        _progress(f"Fetched {len(fine_refs)} fine reference images.")

    fine_parts: list[dict[str, object]] = [
        _part_text("Image 1 is the target brain slice you need to analyze."),
        _part_image_base64(target_b64),
    ]

    for i, (pos, ref_img) in enumerate(fine_refs):
        fine_parts.append(_part_text(f"Reference Image {i + 2}: Atlas slice at {pos:.2f} mm from anterior edge."))
        fine_prepared = _prepare_vlm_image(ref_img, options)
        fine_parts.append(_part_image_base64(_image_to_base64(fine_prepared)))

    fine_parts.append(
        _part_text(
            "Compare the target brain slice (Image 1) to these fine-grained reference atlas slices "
            + f"(centered around {coarse_pos:.2f}mm). Estimate the exact Anterior-Posterior (AP) position "
            + "of the target slice in millimeters from the anterior edge. Return a highly precise value."
        )
    )

    _progress("Analyzing fine position...")
    fine_response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts(fine_parts),
        config=_generation_config(
            {
                "type": "OBJECT",
                "properties": {
                    "position_mm": {
                        "type": "NUMBER",
                        "description": "Precise estimated position in mm from anterior edge",
                    },
                    "reasoning": {
                        "type": "STRING",
                        "description": "Reasoning for the precise estimation",
                    },
                },
            }
        ),
    )

    final_result = _extract_result(fine_response)
    final_pos = _to_float(final_result.get("position_mm"), coarse_pos)
    final_reasoning = _to_str(final_result.get("reasoning"), "N/A")

    return APResult(
        position_mm=final_pos,
        reasoning=f"Coarse estimate was {coarse_pos:.2f}mm. Fine reasoning: {final_reasoning}",
    )


def estimate_ap(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    preprocess_options: PreprocessOptions | None = None,
) -> APResult:
    return estimate_position(
        image=image,
        atlas_name=atlas_name,
        on_progress=on_progress,
        preprocess_options=preprocess_options,
    )


def _build_scaled_composite(
    target: Image.Image,
    atlas_name: str,
    position_mm: float,
    pixel_size_um: float,
    progress: Callable[[str], None],
) -> Image.Image | None:
    """Build a side-by-side composite with physically-correct atlas scaling.

    The atlas image is resized so that one atlas pixel maps to
    ``atlas_resolution_um / pixel_size_um`` target pixels, ensuring the
    two images share the same physical scale.
    """
    from langslice.atlas.core import get_composite_slice, load_atlas

    try:
        atlas = load_atlas(atlas_name)
        atlas_composite = get_composite_slice(atlas, position_mm, opacity=0.4)
        atlas_res_um = float(atlas.resolution[1])  # DV axis (coronal plane)
    except Exception as exc:
        progress(f"Warning: could not load atlas composite for affine prompt: {exc}")
        return None

    scale_factor = atlas_res_um / pixel_size_um
    new_w = max(1, int(round(atlas_composite.width * scale_factor)))
    new_h = max(1, int(round(atlas_composite.height * scale_factor)))
    atlas_scaled = atlas_composite.resize((new_w, new_h), _RESAMPLE_LANCZOS)

    progress(
        f"Atlas scaled for VLM: {atlas_composite.width}x{atlas_composite.height} -> "
        f"{new_w}x{new_h} (scale {scale_factor:.2f}x, "
        f"atlas {atlas_res_um:.0f}µm / slice {pixel_size_um:.1f}µm)"
    )

    # Build side-by-side composite: [target | atlas_scaled]
    # Pad the shorter image vertically to match heights.
    target_rgb = target.convert("RGB")
    atlas_rgb = atlas_scaled.convert("RGB")
    out_h = max(target_rgb.height, atlas_rgb.height)
    gap = 20  # pixel gap between images
    out_w = target_rgb.width + gap + atlas_rgb.width
    composite = Image.new("RGB", (out_w, out_h), (0, 0, 0))

    # Center each vertically
    target_y = (out_h - target_rgb.height) // 2
    atlas_y = (out_h - atlas_rgb.height) // 2
    composite.paste(target_rgb, (0, target_y))
    composite.paste(atlas_rgb, (target_rgb.width + gap, atlas_y))

    return composite


def estimate_affine(
    image: Image.Image,
    on_progress: Callable[[str], None] | None = None,
    preprocess_options: PreprocessOptions | None = None,
    atlas_name: str | None = None,
    position_mm: float | None = None,
    pixel_size_um: float | None = None,
) -> AffineResult:
    """
    Estimate 2D affine transformation to center and align a brain slice.

    When *atlas_name*, *position_mm*, and *pixel_size_um* are all provided the
    function constructs a physically-scaled composite (slice + atlas) and
    sends it to the VLM, giving the model accurate spatial context.

    Returns rotation (degrees), translateX and translateY (% of image size).
    """

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    options = preprocess_options or PreprocessOptions()

    client = get_client()
    target_prepared = _prepare_vlm_image(image, options)
    target_b64 = _image_to_base64(target_prepared)

    # Build physically-scaled composite when atlas context is available
    composite_b64: str | None = None
    has_atlas_context = (
        atlas_name is not None
        and position_mm is not None
        and pixel_size_um is not None
        and pixel_size_um > 0
    )
    if has_atlas_context:
        assert atlas_name is not None and position_mm is not None and pixel_size_um is not None
        composite = _build_scaled_composite(
            target_prepared, atlas_name, position_mm, pixel_size_um, _progress
        )
        if composite is not None:
            composite_b64 = _image_to_base64(composite)

    # Assemble prompt parts
    parts: list[dict[str, object]]
    if composite_b64 is not None:
        _progress("Estimating affine transformation (with atlas context)...")
        parts = [
            _part_text(
                "Image 1 is the target brain slice you need to register."
            ),
            _part_image_base64(target_b64),
            _part_text(
                "Image 2 is a side-by-side composite showing the target brain slice (left) "
                "and the atlas reference at the same physical scale (right). "
                "The atlas has been scaled so both images share the same physical "
                "coordinate space (µm per pixel)."
            ),
            _part_image_base64(composite_b64),
            _part_text(
                "Using both images, estimate the 2D affine transformation "
                "(rotation in degrees, translateX and translateY as percentages of image size) "
                "needed to align the target brain slice to match the atlas orientation. "
                "The atlas reference shows the expected upright coronal orientation. "
                "Compare the tissue outline and internal structures to determine rotation and offset."
            ),
        ]
    else:
        _progress("Estimating affine transformation...")
        parts = [
            _part_image_base64(target_b64),
            _part_text(
                "Analyze this brain slice image. Estimate the 2D affine transformation "
                "(rotation in degrees, translateX and translateY as percentages of image size) "
                "needed to center and align it upright. Assume the image might be slightly tilted "
                "or off-center."
            ),
        ]

    response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts(parts),
        config=_generation_config(
            {
                "type": "OBJECT",
                "properties": {
                    "rotation": {
                        "type": "NUMBER",
                        "description": "Rotation in degrees to make the slice upright",
                    },
                    "translateX": {
                        "type": "NUMBER",
                        "description": "X translation as percentage (-50 to 50)",
                    },
                    "translateY": {
                        "type": "NUMBER",
                        "description": "Y translation as percentage (-50 to 50)",
                    },
                    "reasoning": {
                        "type": "STRING",
                        "description": "Reasoning for the transformation",
                    },
                },
            }
        ),
    )

    result = _extract_result(response)
    rotation = _to_float(result.get("rotation"), 0.0)
    translate_x = _to_float(result.get("translateX"), 0.0)
    translate_y = _to_float(result.get("translateY"), 0.0)
    reasoning = _to_str(result.get("reasoning"), "N/A")

    _progress(f"Affine estimated: rot={rotation} deg, tx={translate_x}%, ty={translate_y}%")

    return AffineResult(
        rotation=rotation,
        translateX=translate_x,
        translateY=translate_y,
        reasoning=reasoning,
    )
