"""Registration agent runtime for anatomical correspondences."""

from __future__ import annotations

import io
import importlib
import json
import logging
import time
from typing import Any, Callable, cast

from PIL import Image
from google.genai import types

from langslice.atlas import get_atlas_info, get_composite_slice, get_region_at_position, load_atlas
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.registration.types import RegistrationCorrespondence

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0


def _retry_generate(
    client: Any,
    *,
    model: str,
    contents: object,
    config: object,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            last_exc = exc
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if (
                isinstance(status, int)
                and status in _RETRYABLE_STATUS_CODES
                and attempt < _MAX_RETRIES
            ):
                delay = _INITIAL_BACKOFF_S * (2**attempt)
                if on_progress:
                    on_progress(f"Gemini registration retry in {delay:.1f}s after status {status}")
                time.sleep(delay)
                continue
            raise
    assert last_exc is not None
    raise last_exc


def _image_to_inline_data(img: Image.Image) -> dict[str, object]:
    buf = io.BytesIO()
    prepared = img.convert("RGB") if img.mode != "RGB" else img
    prepared.save(buf, format="JPEG", quality=95, subsampling=0)
    return {"inline_data": {"mime_type": "image/jpeg", "data": buf.getvalue()}}


def _extract_result(response: object) -> dict[str, object]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        return {str(k): v for k, v in cast(dict[object, object], parsed).items()}
    model_dump = getattr(parsed, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            return {str(k): v for k, v in cast(dict[object, object], dumped).items()}
    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        try:
            decoded = json.loads(text)
            if isinstance(decoded, dict):
                return {str(k): v for k, v in cast(dict[object, object], decoded).items()}
        except json.JSONDecodeError:
            logger.warning("Registration agent did not return parseable JSON")
    return {}


def _extract_count_tokens_metadata(response: object) -> dict[str, int | float | str | bool]:
    metadata: dict[str, int | float | str | bool] = {}
    for field in ("total_tokens", "total_billable_characters"):
        value = getattr(response, field, None)
        if isinstance(value, (int, float, str, bool)):
            metadata[field] = value
    return metadata


def _format_count_tokens(metadata: dict[str, int | float | str | bool]) -> str:
    parts: list[str] = []
    for field in ("total_tokens", "total_billable_characters"):
        value = metadata.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    error = metadata.get("error")
    if isinstance(error, str):
        parts.append(error)
    return ", ".join(parts) if parts else "count_tokens unavailable"


def _to_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"Expected numeric value, got {type(value).__name__}")


def estimate_registration_correspondences(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
) -> list[RegistrationCorrespondence]:
    """Run one Gemini pass to produce paired anatomical correspondences."""
    vlm_config = importlib.import_module("langslice.vlm.config")
    get_client = cast(Callable[[], Any], getattr(vlm_config, "get_client"))
    atlas = load_atlas(atlas_name)
    atlas_info = get_atlas_info(atlas)
    central_region = get_region_at_position(atlas, position_mm, include_hierarchy=True)
    atlas_image = get_composite_slice(atlas, position_mm)

    slice_prep = prepare_image_for_vlm(normalize_image(image))
    atlas_prep = prepare_image_for_vlm(normalize_image(atlas_image))

    if on_progress:
        on_progress(
            "Preparing registration agent inputs: "
            f"slice={slice_prep.output_size[0]}x{slice_prep.output_size[1]}px, "
            f"atlas={atlas_prep.output_size[0]}x{atlas_prep.output_size[1]}px"
        )

    schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "reasoning": {"type": "string"},
            "correspondences": {
                "type": "array",
                "minItems": 8,
                "maxItems": 15,
                "items": {
                    "type": "object",
                    "properties": {
                        "slice_x": {"type": "number"},
                        "slice_y": {"type": "number"},
                        "atlas_x": {"type": "number"},
                        "atlas_y": {"type": "number"},
                        "label": {"type": "string"},
                        "confidence": {"type": "string", "enum": ["high", "medium"]},
                        "rationale": {"type": "string"},
                    },
                    "required": [
                        "slice_x",
                        "slice_y",
                        "atlas_x",
                        "atlas_y",
                        "label",
                        "confidence",
                        "rationale",
                    ],
                },
            },
        },
        "required": ["reasoning", "correspondences"],
    }

    prompt = (
        "You are a neuroanatomy registration assistant. The AP position is already fixed. "
        "Return 8-15 paired anatomical correspondences between the histology slice and the atlas image. "
        "Choose spatially distributed, visually distinct structures that remain reliable on damaged tissue. "
        "Do not invent structures or fill missing tissue. Use only high or medium confidence. "
        "The atlas image includes region boundaries. "
        f"Atlas name: {atlas_name}. Position: {position_mm:.3f} mm. "
        f"Atlas shape: {atlas_info.get('shape')}. Resolution_um: {atlas_info.get('resolution_um')}. "
        f"Central structure hint: {central_region.get('structure')}"
    )

    contents = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"text": "First image: histology slice."},
                _image_to_inline_data(slice_prep.image),
                {"text": "Second image: atlas reference with boundaries."},
                _image_to_inline_data(atlas_prep.image),
            ],
        }
    ]

    config: dict[str, object] = {
        "thinking_config": {"thinking_level": vlm_config.THINKING_LEVEL},
        "response_mime_type": "application/json",
        "response_json_schema": schema,
    }
    client = get_client()
    if getattr(vlm_config, "count_tokens_enabled")():
        count_config = {"system_instruction": None}
        try:
            count_response = client.models.count_tokens(
                model=vlm_config.MODEL_NAME,
                contents=contents,
                config=count_config,
            )
            if on_progress:
                on_progress(
                    "Registration token preflight: "
                    f"{_format_count_tokens(_extract_count_tokens_metadata(count_response))}"
                )
        except Exception as exc:
            if on_progress:
                on_progress(f"Registration token preflight failed: {type(exc).__name__}: {exc}")

    response = _retry_generate(
        client,
        model=vlm_config.MODEL_NAME,
        contents=contents,
        config=config,
        on_progress=on_progress,
    )
    parsed = _extract_result(response)
    raw_corrs = parsed.get("correspondences", [])
    if not isinstance(raw_corrs, list) or not raw_corrs:
        raise RuntimeError("Registration agent returned no correspondences")

    correspondences: list[RegistrationCorrespondence] = []
    slice_scale_x = image.width / max(slice_prep.output_size[0], 1)
    slice_scale_y = image.height / max(slice_prep.output_size[1], 1)
    atlas_scale_x = atlas_image.width / max(atlas_prep.output_size[0], 1)
    atlas_scale_y = atlas_image.height / max(atlas_prep.output_size[1], 1)
    for item in raw_corrs[:15]:
        if not isinstance(item, dict):
            continue
        corr_dict = cast(dict[str, object], item)
        correspondences.append(
            RegistrationCorrespondence(
                slice_xy=(
                    _to_float(corr_dict["slice_x"]) * slice_scale_x,
                    _to_float(corr_dict["slice_y"]) * slice_scale_y,
                ),
                atlas_xy=(
                    _to_float(corr_dict["atlas_x"]) * atlas_scale_x,
                    _to_float(corr_dict["atlas_y"]) * atlas_scale_y,
                ),
                label=str(corr_dict["label"]),
                confidence=str(corr_dict["confidence"]),
                rationale=str(corr_dict["rationale"]),
            )
        )
    if len(correspondences) < 6:
        raise RuntimeError(
            f"Registration agent returned too few usable correspondences ({len(correspondences)})"
        )
    if on_progress:
        on_progress(f"Registration agent proposed {len(correspondences)} correspondences")
    return correspondences
