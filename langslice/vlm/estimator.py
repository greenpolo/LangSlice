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

from langslice.vlm.config import MODEL_NAME, THINKING_BUDGET, get_client

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
    ap_position: float
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
        "thinking_config": {"thinking_budget": THINKING_BUDGET},
        "response_mime_type": "application/json",
        "response_schema": schema,
        "tools": [{"code_execution": {}}],
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


def estimate_ap(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    preprocess_options: PreprocessOptions | None = None,
) -> APResult:
    """
    Two-pass AP position estimation.

    Pass 1 (coarse): Compare against reference slices at 1.0mm intervals.
    Pass 2 (fine): Compare against reference slices at 0.2mm intervals around coarse estimate.
    """
    from langslice.atlas.core import get_ap_range, get_reference_slice, load_atlas

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    options = preprocess_options or PreprocessOptions()

    client = get_client()
    atlas = load_atlas(atlas_name)
    most_anterior, most_posterior = get_ap_range(atlas)
    ap_lower = min(most_anterior, most_posterior)
    ap_upper = max(most_anterior, most_posterior)

    _progress("Fetching coarse reference images (1.0mm steps)...")
    coarse_aps = [3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0, -4.0]

    coarse_refs: list[tuple[float, Image.Image]] = []
    for ap in coarse_aps:
        if not (ap_lower <= ap <= ap_upper):
            continue
        try:
            ref_img = get_reference_slice(atlas, ap)
            coarse_refs.append((ap, ref_img))
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

    for i, (ap, ref_img) in enumerate(coarse_refs):
        sign = "+" if ap > 0 else ""
        coarse_parts.append(_part_text(f"Reference Image {i + 2}: Atlas slice at AP {sign}{ap} mm."))
        coarse_prepared = _prepare_vlm_image(ref_img, options)
        coarse_parts.append(_part_image_base64(_image_to_base64(coarse_prepared)))

    coarse_parts.append(
        _part_text(
            "Compare the target brain slice (Image 1) to the reference atlas slices. "
            + "Estimate the rough Anterior-Posterior (AP) position of the target slice relative "
            + "to Bregma in mm. Return a value between -5.0 and 5.0."
        )
    )

    _progress("Analyzing coarse AP position...")
    coarse_response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts(coarse_parts),
        config=_generation_config(
            {
                "type": "OBJECT",
                "properties": {
                    "ap_position": {
                        "type": "NUMBER",
                        "description": "Estimated AP position in mm",
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
    coarse_ap = _to_float(coarse_result.get("ap_position"), 0.0)
    _progress(
        f"Coarse AP estimated at {'+' if coarse_ap > 0 else ''}{coarse_ap}mm. Fetching fine references..."
    )

    fine_aps = [
        round(coarse_ap + 0.4, 2),
        round(coarse_ap + 0.2, 2),
        round(coarse_ap, 2),
        round(coarse_ap - 0.2, 2),
        round(coarse_ap - 0.4, 2),
    ]

    fine_refs: list[tuple[float, Image.Image]] = []
    for ap in fine_aps:
        if not (ap_lower <= ap <= ap_upper):
            continue
        try:
            ref_img = get_reference_slice(atlas, ap)
            fine_refs.append((ap, ref_img))
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

    for i, (ap, ref_img) in enumerate(fine_refs):
        sign = "+" if ap > 0 else ""
        fine_parts.append(_part_text(f"Reference Image {i + 2}: Atlas slice at AP {sign}{ap} mm."))
        fine_prepared = _prepare_vlm_image(ref_img, options)
        fine_parts.append(_part_image_base64(_image_to_base64(fine_prepared)))

    fine_parts.append(
        _part_text(
            "Compare the target brain slice (Image 1) to these fine-grained reference atlas slices "
            + f"(centered around {coarse_ap}mm). Estimate the exact Anterior-Posterior (AP) position "
            + "of the target slice relative to Bregma in mm. Return a highly precise value."
        )
    )

    _progress("Analyzing fine AP position...")
    fine_response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts(fine_parts),
        config=_generation_config(
            {
                "type": "OBJECT",
                "properties": {
                    "ap_position": {
                        "type": "NUMBER",
                        "description": "Precise estimated AP position in mm",
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
    final_ap = _to_float(final_result.get("ap_position"), coarse_ap)
    final_reasoning = _to_str(final_result.get("reasoning"), "N/A")

    return APResult(
        ap_position=final_ap,
        reasoning=f"Coarse estimate was {coarse_ap}mm. Fine reasoning: {final_reasoning}",
    )


def estimate_affine(
    image: Image.Image,
    on_progress: Callable[[str], None] | None = None,
    preprocess_options: PreprocessOptions | None = None,
) -> AffineResult:
    """
    Estimate 2D affine transformation to center and align a brain slice.

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

    _progress("Estimating affine transformation...")
    response = _retry_generate(
        client,
        model=MODEL_NAME,
        contents=_contents_from_parts([
            _part_image_base64(target_b64),
            _part_text(
                "Analyze this brain slice image. Estimate the 2D affine transformation "
                + "(rotation in degrees, translateX and translateY as percentages of image size) "
                + "needed to center and align it upright. Assume the image might be slightly tilted "
                + "or off-center."
            ),
        ]),
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
