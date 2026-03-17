"""Registration agent runtime for anatomical correspondences."""

from __future__ import annotations

import importlib
import io
import json
import logging
import time
from typing import Any, Callable, cast

from PIL import Image

from langslice.atlas import (
    get_atlas_info,
    get_composite_slice,
    get_reference_slice,
    get_slice_region_metadata,
    load_atlas,
)
from langslice.atlas.core import _AtlasLike
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.registration.types import RegistrationCorrespondence

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0

_MAX_REGION_METADATA = 30

_SINGLE_PASS_SYSTEM_INSTRUCTION = """
You are a neuroanatomy registration assistant producing paired landmark correspondences.
RULES:
1. The atlas and histology depict the same coronal section, but their appearance, scale, rotation, and local distortion may differ.
2. Never copy, scale, rotate, or project coordinates from one image onto the other. Every point must come from direct visual inspection.
3. All coordinates use point_2d format [y, x] in the 0-1000 normalized range.
4. Work one correspondence at a time, visually confirming both atlas and histology before moving to the next pair.
5. Reason bidirectionally: sometimes start from atlas to slice, and sometimes start from slice to atlas.
6. Points must fall on real tissue, not background, padding, or space outside the section.
7. If a reliable match is not visible, set status to not_visible and both coordinates to [0, 0].
8. Prefer a spatially distributed mix of outer contour anchors, midline points, cavity or tract corners, and interior boundaries.
9. Include hemisphere or midline cues in labels whenever possible to reduce left-right swaps.
10. Do not waste effort naming anatomy if identity is uncertain. Use concise geometric labels instead of guessing structure names.
""".strip()


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


def _extract_normalized_point(point: object, *, field_name: str) -> tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) < 2:
        raise TypeError(f"{field_name} must be a [y, x] pair")
    norm_y = _to_float(point[0])
    norm_x = _to_float(point[1])
    return norm_y, norm_x


def _normalized_to_pixel_xy(
    norm_y: float,
    norm_x: float,
    *,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    width, height = image_size
    px_x = norm_x * max(width - 1, 1) / 1000.0
    px_y = norm_y * max(height - 1, 1) / 1000.0
    return px_x, px_y


def _build_region_metadata_text(
    atlas: _AtlasLike,
    position_mm: float,
    max_regions: int = _MAX_REGION_METADATA,
) -> str:
    regions = get_slice_region_metadata(atlas, position_mm)
    if not regions:
        return "No atlas regions were available for this slice."

    rows = [
        "Atlas-derived region hints (use as optional context, not as a checklist):",
        "acronym | name | centroid [y,x] | area% | location",
    ]
    for region in regions[:max_regions]:
        norm_y, norm_x = cast(tuple[int, int], region["centroid_normalized"])
        area_fraction = _to_float(region.get("area_fraction", 0.0))
        acronym = str(region.get("acronym", "")).strip() or "?"
        name = str(region.get("name", "")).strip() or "unknown"
        if abs(norm_x - 500) <= 60:
            lateral = "midline"
        elif norm_x < 500:
            lateral = "left"
        else:
            lateral = "right"
        if norm_y < 333:
            vertical = "dorsal"
        elif norm_y > 666:
            vertical = "ventral"
        else:
            vertical = "central"
        rows.append(
            f"- {acronym} | {name} | [{norm_y}, {norm_x}] | {area_fraction * 100.0:.1f}% | {vertical} {lateral}"
        )
    return "\n".join(rows)


def _build_single_pass_schema(target_count: int) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "correspondences": {
                "type": "array",
                "minItems": target_count,
                "maxItems": target_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "atlas_point_2d": {
                            "type": "array",
                            "description": "Atlas coordinate as [y, x] in the 0-1000 normalized range.",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "slice_point_2d": {
                            "type": "array",
                            "description": "Histology coordinate as [y, x] in the 0-1000 normalized range.",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "label": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["found", "uncertain", "not_visible"],
                        },
                    },
                    "required": [
                        "atlas_point_2d",
                        "slice_point_2d",
                        "label",
                        "status",
                    ],
                },
            }
        },
        "required": ["correspondences"],
    }


def _build_single_pass_request(
    *,
    atlas_prep: Any,
    slice_prep: Any,
    region_metadata_text: str,
    atlas_name: str,
    atlas_info: dict[str, object],
    position_mm: float,
    target_count: int,
    min_edge: int,
    thinking_budget: int,
    enable_code_execution: bool,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    prompt = (
        "TASK: Inspect both images and produce paired atlas-to-histology landmarks for registration.\n\n"
        f"Return exactly {target_count} correspondence objects.\n"
        f"Include at least {min_edge} outer contour anchors when visible.\n"
        "Also include midline points, cavity or tract corners, and interior boundaries when they are reliable.\n"
        "Distribute the accepted points across the whole section instead of clustering them.\n\n"
        f"Atlas: {atlas_name}\n"
        f"AP position: {position_mm:.3f} mm\n"
        f"Atlas shape: {atlas_info.get('shape')}\n"
        f"Atlas resolution (um): {atlas_info.get('resolution_um')}\n\n"
        f"{region_metadata_text}\n\n"
        "Procedure:\n"
        "1. Examine atlas and histology together.\n"
        "2. Select one reliable correspondence at a time.\n"
        "3. For each pair, verify the atlas point and the slice point independently before moving on.\n"
        "4. Use short labels with hemisphere or midline cues whenever possible.\n"
        "5. If a reliable match cannot be confirmed, mark it not_visible instead of forcing a guess.\n\n"
        "Output requirements:\n"
        "- atlas_point_2d and slice_point_2d must be [y, x] integers in the 0-1000 normalized range.\n"
        "- status must be found, uncertain, or not_visible.\n"
        "- Never copy or mechanically transform coordinates from one image to the other."
    )

    contents: list[dict[str, object]] = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"text": "Image 1: Atlas reference with boundary overlays."},
                _image_to_inline_data(atlas_prep.image),
                {"text": "Image 2: Histology slice."},
                _image_to_inline_data(slice_prep.image),
            ],
        }
    ]

    config: dict[str, object] = {
        "system_instruction": _SINGLE_PASS_SYSTEM_INSTRUCTION,
        "thinking_config": {"thinking_budget": thinking_budget},
        "temperature": 0.5,
        "response_mime_type": "application/json",
        "response_json_schema": _build_single_pass_schema(target_count),
    }
    if enable_code_execution:
        config["tools"] = [{"code_execution": {}}]
    return contents, config


def _estimate_correspondences_single_pass(
    client: Any,
    *,
    model: str,
    atlas_prep: Any,
    slice_prep: Any,
    region_metadata_text: str,
    atlas_name: str,
    atlas_info: dict[str, object],
    position_mm: float,
    target_count: int,
    min_edge: int,
    thinking_budget: int,
    enable_code_execution: bool,
    on_progress: Callable[[str], None] | None = None,
) -> list[dict[str, object]]:
    contents, config = _build_single_pass_request(
        atlas_prep=atlas_prep,
        slice_prep=slice_prep,
        region_metadata_text=region_metadata_text,
        atlas_name=atlas_name,
        atlas_info=atlas_info,
        position_mm=position_mm,
        target_count=target_count,
        min_edge=min_edge,
        thinking_budget=thinking_budget,
        enable_code_execution=enable_code_execution,
    )

    if on_progress:
        budget_label = (
            f"thinking_budget={thinking_budget}" if thinking_budget > 0 else "thinking=off"
        )
        on_progress(f"Registration: locating paired correspondences ({budget_label})...")

    response = _retry_generate(
        client,
        model=model,
        contents=contents,
        config=config,
        on_progress=on_progress,
    )
    parsed = _extract_result(response)
    correspondences = parsed.get("correspondences", [])
    if not isinstance(correspondences, list) or not correspondences:
        raise RuntimeError("Single-pass registration returned no correspondences")
    return list(correspondences)


def estimate_registration_correspondences(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    target_landmark_count: int = 8,
    min_edge_landmarks: int = 5,
    show_atlas_borders: bool = True,
    on_progress: Callable[[str], None] | None = None,
) -> list[RegistrationCorrespondence]:
    """Run a single-pass Gemini pipeline to produce paired landmark correspondences."""
    vlm_config = importlib.import_module("langslice.vlm.config")
    get_client = cast(Callable[[], Any], getattr(vlm_config, "get_client"))
    atlas = load_atlas(atlas_name)
    atlas_info = get_atlas_info(atlas)
    if show_atlas_borders:
        atlas_image = get_composite_slice(atlas, position_mm)
    else:
        atlas_image = get_reference_slice(atlas, position_mm)

    slice_prep = prepare_image_for_vlm(normalize_image(image))
    atlas_prep = prepare_image_for_vlm(normalize_image(atlas_image))
    region_metadata_text = _build_region_metadata_text(atlas, position_mm)

    target_count = max(1, min(int(target_landmark_count), 24))
    min_edge = max(0, min(int(min_edge_landmarks), target_count))

    if on_progress:
        on_progress(
            "Preparing registration agent inputs: "
            f"slice={slice_prep.output_size[0]}x{slice_prep.output_size[1]}px, "
            f"atlas={atlas_prep.output_size[0]}x{atlas_prep.output_size[1]}px"
        )

    client = get_client()
    model_name: str = vlm_config.MODEL_NAME
    enable_code_execution = (
        getattr(vlm_config, "CODE_EXECUTION_ENABLED", False)
        and model_name == "gemini-3-flash-preview"
    )
    thinking_budget: int = getattr(vlm_config, "REGISTRATION_THINKING_BUDGET", 8192)

    if getattr(vlm_config, "count_tokens_enabled")():
        try:
            count_contents, count_config = _build_single_pass_request(
                atlas_prep=atlas_prep,
                slice_prep=slice_prep,
                region_metadata_text=region_metadata_text,
                atlas_name=atlas_name,
                atlas_info=atlas_info,
                position_mm=position_mm,
                target_count=target_count,
                min_edge=min_edge,
                thinking_budget=thinking_budget,
                enable_code_execution=enable_code_execution,
            )
            count_response = client.models.count_tokens(
                model=model_name,
                contents=count_contents,
                config={"system_instruction": count_config.get("system_instruction")},
            )
            if on_progress:
                on_progress(
                    "Registration token preflight: "
                    f"{_format_count_tokens(_extract_count_tokens_metadata(count_response))}"
                )
        except Exception as exc:
            if on_progress:
                on_progress(f"Registration token preflight failed: {type(exc).__name__}: {exc}")

    raw_correspondences = _estimate_correspondences_single_pass(
        client,
        model=model_name,
        atlas_prep=atlas_prep,
        slice_prep=slice_prep,
        region_metadata_text=region_metadata_text,
        atlas_name=atlas_name,
        atlas_info=atlas_info,
        position_mm=position_mm,
        target_count=target_count,
        min_edge=min_edge,
        thinking_budget=thinking_budget,
        enable_code_execution=enable_code_execution,
        on_progress=on_progress,
    )

    visible_correspondences: list[dict[str, object]] = []
    for entry in raw_correspondences:
        if not isinstance(entry, dict):
            continue
        status = str(entry.get("status", "found"))
        if status == "not_visible":
            logger.info(
                "Correspondence %s marked not_visible; skipping",
                entry.get("label", "unknown"),
            )
            continue
        visible_correspondences.append(entry)

    candidate_correspondences = visible_correspondences

    # Convert normalized [y, x] values to original-image pixel (x, y).
    # We map the 0-1000 endpoints onto the valid pixel index range [0, size-1]
    # so model outputs at 1000 stay inside the image domain.

    correspondences: list[RegistrationCorrespondence] = []
    for entry in candidate_correspondences:
        try:
            a_norm_y, a_norm_x = _extract_normalized_point(
                entry.get("atlas_point_2d"),
                field_name="atlas_point_2d",
            )
            s_norm_y, s_norm_x = _extract_normalized_point(
                entry.get("slice_point_2d"),
                field_name="slice_point_2d",
            )
        except (TypeError, ValueError) as exc:
            logger.info(
                "Rejected correspondence %s due to invalid point: %s",
                entry.get("label", "unknown"),
                exc,
            )
            continue

        a_px_x, a_px_y = _normalized_to_pixel_xy(
            a_norm_y,
            a_norm_x,
            image_size=atlas_image.size,
        )
        s_px_x, s_px_y = _normalized_to_pixel_xy(
            s_norm_y,
            s_norm_x,
            image_size=image.size,
        )

        rationale_parts = [f"status={entry.get('status', 'found')}"]

        correspondences.append(
            RegistrationCorrespondence(
                slice_xy=(s_px_x, s_px_y),
                atlas_xy=(a_px_x, a_px_y),
                label=str(entry.get("label", f"landmark_{len(correspondences)}")),
                confidence="medium",
                rationale="; ".join(rationale_parts),
            )
        )

    if not correspondences:
        raise RuntimeError("Registration agent produced no usable correspondences")
    correspondences = correspondences[:target_count]

    if on_progress:
        on_progress(f"Registration agent proposed {len(correspondences)} correspondences")
    return correspondences
