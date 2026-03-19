"""Registration agent runtime for anatomical correspondences."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import io
import json
import logging
import os
import time
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
    tool_call_event,
    tool_result_event,
)
from langslice.atlas import (
    get_atlas_info,
    get_composite_slice,
    get_reference_slice,
    get_slice_region_metadata,
    load_atlas,
)
from langslice.atlas.core import _AtlasLike
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.registration.types import (
    LandmarkAnnotation,
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    render_landmark_annotations,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0

_MAX_REGION_METADATA = 30
_TOOL_LOOP_MAX_STEPS = 24

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


@dataclass
class _PreparedRegistrationInputs:
    atlas_image: Image.Image
    atlas_info: dict[str, object]
    atlas_prep: Any
    slice_prep: Any
    region_metadata_text: str
    target_count: int
    min_edge: int
    model_name: str
    thinking_budget: int
    temperature: float
    enable_code_execution: bool
    registration_dir: str | None
    atlas_path: str | None
    slice_path: str | None


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
    decoded = _extract_json_dict(response)
    if not decoded:
        logger.warning("Registration agent did not return parseable JSON")
    return decoded


def _emit_trace(
    on_trace: Callable[[dict[str, object]], None] | None,
    event: dict[str, object],
) -> None:
    if on_trace:
        on_trace(event)


def _extract_response_text_parts(response: object) -> tuple[list[str], list[str]]:
    text_parts: list[str] = []
    thought_parts: list[str] = []
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        return text_parts, thought_parts
    content = getattr(candidates[0], "content", None)
    for part in getattr(content, "parts", None) or []:
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        if bool(getattr(part, "thought", False)):
            thought_parts.append(text)
        else:
            text_parts.append(text)
    return text_parts, thought_parts


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


def _strip_code_fence(text: str) -> str:
    cleaned = text.strip()
    if not cleaned.startswith("```"):
        return cleaned
    lines = cleaned.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _decode_json_payload(text: str) -> object | None:
    candidates = [_strip_code_fence(text)]
    stripped = text.strip()
    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = stripped.find(open_char)
        end = stripped.rfind(close_char)
        if start >= 0 and end > start:
            candidates.append(stripped[start : end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def _extract_json_payload(response: object) -> object | None:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if isinstance(parsed, (dict, list)):
            return parsed
        model_dump = getattr(parsed, "model_dump", None)
        if callable(model_dump):
            dumped = model_dump()
            if isinstance(dumped, (dict, list)):
                return dumped

    text = getattr(response, "text", None)
    if isinstance(text, str) and text:
        payload = _decode_json_payload(text)
        if payload is not None:
            return payload

    text_parts, _thought_parts = _extract_response_text_parts(response)
    for part in text_parts:
        payload = _decode_json_payload(part)
        if payload is not None:
            return payload
    if text_parts:
        payload = _decode_json_payload("\n".join(text_parts))
        if payload is not None:
            return payload
    return None


def _extract_json_dict(response: object) -> dict[str, object]:
    payload = _extract_json_payload(response)
    if isinstance(payload, dict):
        return {str(k): v for k, v in cast(dict[object, object], payload).items()}
    return {}


def _prepare_registration_inputs(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    target_landmark_count: int,
    min_edge_landmarks: int,
    show_atlas_borders: bool,
    debug_dir: str | None,
    vlm_config: Any,
) -> _PreparedRegistrationInputs:
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
    model_name = str(getattr(vlm_config, "MODEL_NAME", "gemini-3-flash-preview"))
    thinking_budget = int(getattr(vlm_config, "REGISTRATION_THINKING_BUDGET", 8192))
    temperature = float(getattr(vlm_config, "TEMPERATURE", 0.5))
    enable_code_execution = bool(getattr(vlm_config, "CODE_EXECUTION_ENABLED", False)) and (
        model_name == "gemini-3-flash-preview"
    )

    registration_dir: str | None = None
    atlas_path: str | None = None
    slice_path: str | None = None
    if debug_dir:
        registration_dir = os.path.join(debug_dir, "registration")
        os.makedirs(registration_dir, exist_ok=True)
        atlas_path = os.path.join(registration_dir, "request_atlas.jpg")
        slice_path = os.path.join(registration_dir, "request_slice.jpg")
        atlas_prep.image.save(atlas_path, quality=95)
        slice_prep.image.save(slice_path, quality=95)
        with open(
            os.path.join(registration_dir, "request_region_metadata.txt"), "w", encoding="utf-8"
        ) as fh:
            fh.write(region_metadata_text)

    return _PreparedRegistrationInputs(
        atlas_image=atlas_image,
        atlas_info=atlas_info,
        atlas_prep=atlas_prep,
        slice_prep=slice_prep,
        region_metadata_text=region_metadata_text,
        target_count=target_count,
        min_edge=min_edge,
        model_name=model_name,
        thinking_budget=thinking_budget,
        temperature=temperature,
        enable_code_execution=enable_code_execution,
        registration_dir=registration_dir,
        atlas_path=atlas_path,
        slice_path=slice_path,
    )


def _emit_prepared_registration_trace(
    prepared: _PreparedRegistrationInputs,
    *,
    atlas_name: str,
    position_mm: float,
    workflow: str,
    show_atlas_borders: bool,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> None:
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Prepared registration request",
            summary=f"Atlas and histology inputs prepared at AP {position_mm:.3f} mm",
            parts=[
                image_part_from_pil(
                    prepared.atlas_prep.image,
                    label="Atlas reference with boundaries",
                    path=prepared.atlas_path,
                ),
                image_part_from_pil(
                    prepared.slice_prep.image,
                    label="Histology slice",
                    path=prepared.slice_path,
                ),
                json_part(
                    {
                        "atlas_name": atlas_name,
                        "position_mm": round(position_mm, 3),
                        "target_landmark_count": prepared.target_count,
                        "min_edge_landmarks": prepared.min_edge,
                        "thinking_budget": prepared.thinking_budget,
                        "temperature": prepared.temperature,
                        "workflow": workflow,
                    },
                    label="Request settings",
                ),
                json_part(
                    prepared.region_metadata_text,
                    label="Region metadata",
                    collapsible=True,
                ),
            ],
            metadata={
                "target_landmark_count": prepared.target_count,
                "thinking_budget": prepared.thinking_budget,
                "temperature": prepared.temperature,
                "workflow": workflow,
                "show_atlas_borders": show_atlas_borders,
            },
        ),
    )


def _filter_visible_entries(entries: list[dict[str, object]]) -> list[dict[str, object]]:
    visible_correspondences: list[dict[str, object]] = []
    for entry in entries:
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
    return visible_correspondences


def _entries_to_correspondences(
    entries: list[dict[str, object]],
    *,
    atlas_image_size: tuple[int, int],
    slice_image_size: tuple[int, int],
    target_count: int,
) -> list[RegistrationCorrespondence]:
    correspondences: list[RegistrationCorrespondence] = []
    for entry in entries:
        try:
            a_norm_y, a_norm_x = _extract_normalized_point(
                entry.get("atlas_point_2d"), field_name="atlas_point_2d"
            )
            s_norm_y, s_norm_x = _extract_normalized_point(
                entry.get("slice_point_2d"), field_name="slice_point_2d"
            )
        except (TypeError, ValueError) as exc:
            logger.info(
                "Rejected correspondence %s due to invalid point: %s",
                entry.get("label", "unknown"),
                exc,
            )
            continue

        a_px_x, a_px_y = _normalized_to_pixel_xy(a_norm_y, a_norm_x, image_size=atlas_image_size)
        s_px_x, s_px_y = _normalized_to_pixel_xy(s_norm_y, s_norm_x, image_size=slice_image_size)
        rationale_parts = [f"status={entry.get('status', 'found')}"]
        if entry.get("feature_description"):
            rationale_parts.append(str(entry.get("feature_description")))
        if entry.get("artifact_note"):
            rationale_parts.append(f"artifact={entry.get('artifact_note')}")

        correspondences.append(
            RegistrationCorrespondence(
                slice_xy=(s_px_x, s_px_y),
                atlas_xy=(a_px_x, a_px_y),
                label=str(entry.get("label", f"landmark_{len(correspondences) + 1}")),
                confidence="medium",
                rationale="; ".join(rationale_parts),
                slice_normalized_yx=(s_norm_y, s_norm_x),
                atlas_normalized_yx=(a_norm_y, a_norm_x),
            )
        )

    if not correspondences:
        raise RuntimeError("Registration agent produced no usable correspondences")
    return correspondences[:target_count]


def _crop_zoom_view(
    image: Image.Image, *, center_yx: tuple[float, float], zoom: float
) -> Image.Image:
    width, height = image.size
    zoom = max(1.0, float(zoom))
    crop_width = max(32, int(round(width / zoom)))
    crop_height = max(32, int(round(height / zoom)))
    center_x, center_y = _normalized_to_pixel_xy(center_yx[0], center_yx[1], image_size=image.size)
    left = max(0, int(round(center_x - (crop_width / 2.0))))
    top = max(0, int(round(center_y - (crop_height / 2.0))))
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    left = max(0, right - crop_width)
    top = max(0, bottom - crop_height)
    return image.crop((left, top, right, bottom)).resize(image.size, Image.Resampling.BICUBIC)


def _upsert_annotation(
    annotations: list[LandmarkAnnotation], annotation: LandmarkAnnotation
) -> None:
    for index, existing in enumerate(annotations):
        if existing.label == annotation.label:
            annotations[index] = annotation
            return
    annotations.append(annotation)


def _session_summary(session: RegistrationAnnotationSession) -> dict[str, object]:
    return {
        "workflow": session.workflow,
        "target_count": session.target_count,
        "placed_labels": [annotation.label for annotation in session.slice_annotations],
        "atlas_labels": [annotation.label for annotation in session.atlas_annotations],
        "slice_labels": [annotation.label for annotation in session.slice_annotations],
    }


def _confirmed_tool_loop_entries(session: RegistrationAnnotationSession) -> list[dict[str, object]]:
    atlas_by_label = {
        annotation.label: annotation
        for annotation in session.atlas_annotations
        if annotation.status != "not_visible"
    }
    slice_by_label = {
        annotation.label: annotation
        for annotation in session.slice_annotations
        if annotation.status != "not_visible"
    }

    def _label_sort_key(value: str) -> tuple[int, int | str]:
        stripped = str(value).strip()
        return (0, int(stripped)) if stripped.isdigit() else (1, stripped)

    shared_labels = sorted(set(atlas_by_label) & set(slice_by_label), key=_label_sort_key)
    entries: list[dict[str, object]] = []
    for label in shared_labels:
        atlas_annotation = atlas_by_label[label]
        slice_annotation = slice_by_label[label]
        entries.append(
            {
                "label": label,
                "status": "found",
                "atlas_point_2d": list(atlas_annotation.normalized_yx or (0.0, 0.0)),
                "slice_point_2d": list(slice_annotation.normalized_yx or (0.0, 0.0)),
                "feature_description": slice_annotation.feature_description,
                "artifact_note": slice_annotation.artifact_note,
            }
        )
    return entries


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
    temperature: float,
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
        "temperature": temperature,
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
    temperature: float,
    enable_code_execution: bool,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
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
        temperature=temperature,
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
    text_parts, thought_parts = _extract_response_text_parts(response)
    if text_parts or thought_parts:
        trace_parts: list[dict[str, object]] = []
        if thought_parts:
            trace_parts.append(
                json_part(thought_parts, label="Reasoning summary", collapsible=True)
            )
        if text_parts:
            trace_parts.append(json_part(text_parts, label="Model text", collapsible=True))
        _emit_trace(
            on_trace,
            model_event(
                stage="registration",
                title="Registration model response",
                summary="Model returned text alongside the structured correspondence output",
                parts=trace_parts,
                metadata={"thinking_budget": thinking_budget},
            ),
        )
    parsed = _extract_result(response)
    correspondences = parsed.get("correspondences", [])
    if not isinstance(correspondences, list) or not correspondences:
        raise RuntimeError("Single-pass registration returned no correspondences")
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Registration correspondences parsed",
            summary=f"Model proposed {len(correspondences)} correspondences",
            parts=[json_part(parsed, label="Structured response")],
            metadata={"correspondence_count": len(correspondences)},
        ),
    )
    return list(correspondences)


def _build_landmark_array_schema(*, point_field: str, target_count: int) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "landmarks": {
                "type": "array",
                "minItems": target_count,
                "maxItems": target_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string"},
                        point_field: {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "status": {"type": "string", "enum": ["found", "uncertain", "not_visible"]},
                        "feature_description": {"type": "string"},
                    },
                    "required": ["label", point_field, "status"],
                },
            }
        },
        "required": ["landmarks"],
    }


def _build_image_gen_config(
    *,
    model_name: str,
    vlm_config: Any,
    temperature: float,
    schema: dict[str, object] | None,
) -> dict[str, object]:
    config: dict[str, object] = {"temperature": temperature}
    supports_structured = bool(
        getattr(vlm_config, "supports_structured_image_output", lambda _model: False)(model_name)
    )
    if supports_structured and schema is not None:
        config["response_mime_type"] = "application/json"
        config["response_json_schema"] = schema
    return config


def _build_image_gen_atlas_request(
    *,
    atlas_prep: Any,
    atlas_name: str,
    position_mm: float,
    target_count: int,
) -> list[dict[str, object]]:
    border_count = max(1, target_count // 2)
    interior_count = max(1, target_count - border_count)
    prompt = (
        "You are annotating a coronal anatomical atlas for later landmark transfer.\n"
        f"Atlas: {atlas_name}\n"
        f"AP position: {position_mm:.3f} mm\n"
        f"Return exactly {target_count} numbered landmarks labeled 1 through {target_count}.\n"
        f"Place about {border_count} on the outer border and about {interior_count} inside the tissue.\n"
        "Choose landmarks that are visually distinct, evenly distributed, and likely matchable in real histology.\n"
        "Return JSON only. Use atlas_point_2d as [y, x] integers in the 0-1000 range.\n"
        "Do not skip numbers, do not repeat labels, and do not place points off tissue."
    )
    return [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"text": "Image 1: Atlas reference."},
                _image_to_inline_data(atlas_prep.image),
            ],
        }
    ]


def _build_image_gen_slice_request(
    *,
    annotated_atlas: Image.Image,
    slice_prep: Any,
    atlas_name: str,
    position_mm: float,
    target_count: int,
) -> list[dict[str, object]]:
    prompt = (
        "You are transferring numbered atlas landmarks onto a histology slice.\n"
        f"Atlas: {atlas_name}\n"
        f"AP position: {position_mm:.3f} mm\n"
        f"Return exactly {target_count} landmarks labeled 1 through {target_count}.\n"
        "The atlas image is already annotated with numbered landmarks. Place matching slice points in the same relative anatomical locations.\n"
        "Return JSON only. Use slice_point_2d as [y, x] integers in the 0-1000 range.\n"
        "Preserve the numbering exactly and do not invent new labels."
    )
    return [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"text": "Image 1: Annotated atlas reference."},
                _image_to_inline_data(annotated_atlas),
                {"text": "Image 2: Histology slice."},
                _image_to_inline_data(slice_prep.image),
            ],
        }
    ]


def _extract_landmarks_array(
    payload: dict[str, object], *, field_name: str
) -> list[dict[str, object]]:
    entries = payload.get("landmarks")
    if not isinstance(entries, list):
        raise RuntimeError(f"Expected JSON object with a 'landmarks' array for {field_name}")
    parsed_entries: list[dict[str, object]] = []
    for entry in entries:
        if isinstance(entry, dict):
            parsed_entries.append(entry)
    if not parsed_entries:
        raise RuntimeError(f"No usable landmarks returned for {field_name}")
    return parsed_entries


def _merge_two_shot_entries(
    atlas_entries: list[dict[str, object]],
    slice_entries: list[dict[str, object]],
    *,
    target_count: int,
) -> list[dict[str, object]]:
    atlas_by_label = {str(entry.get("label", "")).strip(): entry for entry in atlas_entries}
    slice_by_label = {str(entry.get("label", "")).strip(): entry for entry in slice_entries}
    merged: list[dict[str, object]] = []
    for index in range(1, target_count + 1):
        label = str(index)
        atlas_entry = atlas_by_label.get(label)
        slice_entry = slice_by_label.get(label)
        if atlas_entry is None or slice_entry is None:
            raise RuntimeError(f"Two-shot workflow did not return label {label} in both passes")
        merged.append(
            {
                "label": label,
                "status": str(slice_entry.get("status", atlas_entry.get("status", "found"))),
                "atlas_point_2d": atlas_entry.get("atlas_point_2d"),
                "slice_point_2d": slice_entry.get("slice_point_2d"),
                "feature_description": str(slice_entry.get("feature_description", "")).strip(),
            }
        )
    return merged


def _estimate_correspondences_image_gen_two_shot(
    client: Any,
    *,
    prepared: _PreparedRegistrationInputs,
    vlm_config: Any,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    if on_progress:
        on_progress("Registration: atlas landmark proposal pass (image-gen)...")
    atlas_response = _retry_generate(
        client,
        model=prepared.model_name,
        contents=_build_image_gen_atlas_request(
            atlas_prep=prepared.atlas_prep,
            atlas_name=atlas_name,
            position_mm=position_mm,
            target_count=prepared.target_count,
        ),
        config=_build_image_gen_config(
            model_name=prepared.model_name,
            vlm_config=vlm_config,
            temperature=prepared.temperature,
            schema=_build_landmark_array_schema(
                point_field="atlas_point_2d", target_count=prepared.target_count
            ),
        ),
        on_progress=on_progress,
    )
    atlas_payload = _extract_json_dict(atlas_response)
    atlas_entries = _extract_landmarks_array(atlas_payload, field_name="atlas")
    atlas_annotations = [
        LandmarkAnnotation(
            image_role="atlas",
            pixel_xy=_normalized_to_pixel_xy(
                *_extract_normalized_point(
                    entry.get("atlas_point_2d"), field_name="atlas_point_2d"
                ),
                image_size=prepared.atlas_prep.image.size,
            ),
            label=str(entry.get("label", "")).strip(),
            normalized_yx=_extract_normalized_point(
                entry.get("atlas_point_2d"), field_name="atlas_point_2d"
            ),
            status=str(entry.get("status", "found")),
            feature_description=str(entry.get("feature_description", "")).strip(),
        )
        for entry in atlas_entries
    ]
    annotated_atlas = render_landmark_annotations(prepared.atlas_prep.image, atlas_annotations)
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen atlas landmarks",
            summary=f"Model proposed {len(atlas_entries)} atlas landmarks",
            parts=[
                image_part_from_pil(annotated_atlas, label="Atlas with generated landmarks"),
                json_part(atlas_payload, label="Atlas landmark response", collapsible=True),
            ],
            metadata={"workflow": "image_gen_two_shot", "pass": 1},
        ),
    )

    if on_progress:
        on_progress("Registration: slice transfer pass (image-gen)...")
    slice_response = _retry_generate(
        client,
        model=prepared.model_name,
        contents=_build_image_gen_slice_request(
            annotated_atlas=annotated_atlas,
            slice_prep=prepared.slice_prep,
            atlas_name=atlas_name,
            position_mm=position_mm,
            target_count=prepared.target_count,
        ),
        config=_build_image_gen_config(
            model_name=prepared.model_name,
            vlm_config=vlm_config,
            temperature=prepared.temperature,
            schema=_build_landmark_array_schema(
                point_field="slice_point_2d", target_count=prepared.target_count
            ),
        ),
        on_progress=on_progress,
    )
    slice_payload = _extract_json_dict(slice_response)
    slice_entries = _extract_landmarks_array(slice_payload, field_name="slice")
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Image-gen slice transfer",
            summary=f"Model proposed {len(slice_entries)} slice landmarks",
            parts=[json_part(slice_payload, label="Slice landmark response", collapsible=True)],
            metadata={"workflow": "image_gen_two_shot", "pass": 2},
        ),
    )
    return _merge_two_shot_entries(
        atlas_entries,
        slice_entries,
        target_count=prepared.target_count,
    )


def _build_tool_loop_action_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "tool_name": {
                "type": "string",
                "enum": ["view_overview", "view_zoom_pair", "place_point_pair", "finish"],
            },
            "tool_args": {"type": "object"},
        },
        "required": ["tool_name", "tool_args"],
    }


def _build_tool_loop_prompt(
    *,
    atlas_name: str,
    position_mm: float,
    target_count: int,
) -> str:
    return (
        "You are placing matched atlas/slice landmarks using host tools.\n"
        f"Atlas: {atlas_name}\n"
        f"AP position: {position_mm:.3f} mm\n"
        f"Place exactly {target_count} paired landmarks labeled 1 through {target_count}.\n"
        "Work one point at a time or a very small batch.\n"
        "Suggested process: define the local feature, place the point pair, inspect locally at 3x, inspect more broadly at 1.5x, then lock it.\n"
        "You may choose which validation steps to repeat.\n"
        "Available tools: view_overview, view_zoom_pair, place_point_pair, finish.\n"
        "All coordinates must be [y, x] integers in the 0-1000 range of the full images.\n"
        "Persistent numbered annotations are shown in tool results.\n"
        "When enough pairs are placed, call finish."
    )


def _tool_loop_overview_parts(
    session: RegistrationAnnotationSession,
    *,
    atlas_image: Image.Image,
    slice_image: Image.Image,
    summary_label: str,
) -> list[dict[str, object]]:
    return [
        {"text": summary_label},
        _image_to_inline_data(render_landmark_annotations(atlas_image, session.atlas_annotations)),
        _image_to_inline_data(render_landmark_annotations(slice_image, session.slice_annotations)),
        {"text": json.dumps(_session_summary(session), indent=2)},
    ]


def _execute_tool_loop_action(
    action: dict[str, object],
    *,
    session: RegistrationAnnotationSession,
    atlas_image: Image.Image,
    slice_image: Image.Image,
    iteration: int,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[bool, list[dict[str, object]]]:
    tool_name = str(action.get("tool_name", "")).strip()
    tool_args = cast(dict[str, object], action.get("tool_args", {}))
    _emit_trace(
        on_trace,
        tool_call_event(
            stage="registration", tool_name=tool_name, args=tool_args, iteration=iteration
        ),
    )

    if tool_name == "view_overview":
        parts = _tool_loop_overview_parts(
            session,
            atlas_image=atlas_image,
            slice_image=slice_image,
            summary_label="Overview with current persistent annotations.",
        )
        _emit_trace(
            on_trace,
            tool_result_event(
                stage="registration",
                tool_name=tool_name,
                summary="Returned overview images with current annotations",
                parts=[
                    image_part_from_pil(
                        render_landmark_annotations(atlas_image, session.atlas_annotations),
                        label="Atlas overview",
                    ),
                    image_part_from_pil(
                        render_landmark_annotations(slice_image, session.slice_annotations),
                        label="Slice overview",
                    ),
                ],
                metadata={"iteration": iteration},
            ),
        )
        return False, parts

    if tool_name == "view_zoom_pair":
        zoom = _to_float(tool_args.get("zoom", 3.0))
        atlas_center = _extract_normalized_point(
            tool_args.get("atlas_center_2d"), field_name="atlas_center_2d"
        )
        slice_center = _extract_normalized_point(
            tool_args.get("slice_center_2d"), field_name="slice_center_2d"
        )
        atlas_zoom = _crop_zoom_view(
            render_landmark_annotations(atlas_image, session.atlas_annotations),
            center_yx=atlas_center,
            zoom=zoom,
        )
        slice_zoom = _crop_zoom_view(
            render_landmark_annotations(slice_image, session.slice_annotations),
            center_yx=slice_center,
            zoom=zoom,
        )
        parts = [
            {
                "text": f"Zoom pair generated at {zoom:.1f}x around the requested atlas and slice centers."
            },
            _image_to_inline_data(
                render_landmark_annotations(atlas_zoom, session.atlas_annotations)
            ),
            _image_to_inline_data(
                render_landmark_annotations(slice_zoom, session.slice_annotations)
            ),
            {
                "text": json.dumps(
                    {
                        "zoom": zoom,
                        "atlas_center_2d": atlas_center,
                        "slice_center_2d": slice_center,
                    },
                    indent=2,
                )
            },
        ]
        _emit_trace(
            on_trace,
            tool_result_event(
                stage="registration",
                tool_name=tool_name,
                summary=f"Returned zoomed atlas/slice views at {zoom:.1f}x",
                metadata={"iteration": iteration, "zoom": zoom},
            ),
        )
        return False, parts

    if tool_name == "place_point_pair":
        label = str(tool_args.get("label", "")).strip()
        atlas_norm = _extract_normalized_point(
            tool_args.get("atlas_point_2d"), field_name="atlas_point_2d"
        )
        slice_norm = _extract_normalized_point(
            tool_args.get("slice_point_2d"), field_name="slice_point_2d"
        )
        feature_description = str(tool_args.get("feature_description", "")).strip()
        artifact_note = str(tool_args.get("artifact_note", "")).strip()
        status = str(tool_args.get("status", "found")).strip() or "found"
        _upsert_annotation(
            session.atlas_annotations,
            LandmarkAnnotation(
                image_role="atlas",
                pixel_xy=_normalized_to_pixel_xy(
                    atlas_norm[0], atlas_norm[1], image_size=atlas_image.size
                ),
                label=label,
                normalized_yx=atlas_norm,
                status=status,
                feature_description=feature_description,
                artifact_note=artifact_note,
            ),
        )
        _upsert_annotation(
            session.slice_annotations,
            LandmarkAnnotation(
                image_role="slice",
                pixel_xy=_normalized_to_pixel_xy(
                    slice_norm[0], slice_norm[1], image_size=slice_image.size
                ),
                label=label,
                normalized_yx=slice_norm,
                status=status,
                feature_description=feature_description,
                artifact_note=artifact_note,
            ),
        )
        parts = _tool_loop_overview_parts(
            session,
            atlas_image=atlas_image,
            slice_image=slice_image,
            summary_label=f"Saved point pair {label}. Updated persistent annotations are shown below.",
        )
        _emit_trace(
            on_trace,
            tool_result_event(
                stage="registration",
                tool_name=tool_name,
                summary=f"Saved point pair {label}",
                metadata={"iteration": iteration, "label": label},
            ),
        )
        return False, parts

    if tool_name == "finish":
        current_count = len(_confirmed_tool_loop_entries(session))
        if current_count < int(session.target_count or 0):
            parts: list[dict[str, object]] = [
                {
                    "text": (
                        f"Cannot finish yet. You have {current_count} confirmed paired points but need "
                        f"{session.target_count}."
                    )
                },
                {"text": json.dumps(_session_summary(session), indent=2)},
            ]
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="registration",
                    tool_name=tool_name,
                    summary="Finish rejected because not enough points are placed",
                    metadata={"iteration": iteration, "confirmed_count": current_count},
                ),
            )
            return False, parts
        _emit_trace(
            on_trace,
            tool_result_event(
                stage="registration",
                tool_name=tool_name,
                summary=f"Finish accepted with {current_count} confirmed pairs",
                metadata={"iteration": iteration, "confirmed_count": current_count},
            ),
        )
        return True, [{"text": f"Finished with {current_count} confirmed point pairs."}]

    raise RuntimeError(f"Unknown tool-loop action: {tool_name}")


def _estimate_correspondences_tool_loop(
    client: Any,
    *,
    prepared: _PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    session = RegistrationAnnotationSession(
        workflow="multimodal_tool_loop",
        target_count=prepared.target_count,
        metadata={"atlas_name": atlas_name, "position_mm": round(position_mm, 3)},
    )
    contents: list[dict[str, object]] = [
        {
            "role": "user",
            "parts": [
                {
                    "text": _build_tool_loop_prompt(
                        atlas_name=atlas_name,
                        position_mm=position_mm,
                        target_count=prepared.target_count,
                    )
                },
                {"text": "Initial atlas overview with no points placed yet."},
                _image_to_inline_data(prepared.atlas_prep.image),
                {"text": "Initial histology overview with no points placed yet."},
                _image_to_inline_data(prepared.slice_prep.image),
                {"text": json.dumps(_session_summary(session), indent=2)},
            ],
        }
    ]
    config: dict[str, object] = {
        "temperature": prepared.temperature,
        "thinking_config": {"thinking_budget": prepared.thinking_budget},
        "response_mime_type": "application/json",
        "response_json_schema": _build_tool_loop_action_schema(),
    }

    for iteration in range(1, _TOOL_LOOP_MAX_STEPS + 1):
        if on_progress:
            on_progress(f"Registration tool loop: step {iteration}/{_TOOL_LOOP_MAX_STEPS}...")
        response = _retry_generate(
            client,
            model=prepared.model_name,
            contents=contents,
            config=config,
            on_progress=on_progress,
        )
        action = _extract_json_dict(response)
        if not action:
            raise RuntimeError("Tool-loop registration returned no action JSON")
        _emit_trace(
            on_trace,
            model_event(
                stage="registration",
                title="Tool-loop action",
                summary=f"Iteration {iteration}: {action.get('tool_name', 'unknown')}",
                parts=[json_part(action, label="Tool action")],
                metadata={"workflow": "multimodal_tool_loop", "iteration": iteration},
            ),
        )
        finished, tool_parts = _execute_tool_loop_action(
            action,
            session=session,
            atlas_image=prepared.atlas_prep.image,
            slice_image=prepared.slice_prep.image,
            iteration=iteration,
            on_trace=on_trace,
        )
        contents.append({"role": "model", "parts": [{"text": json.dumps(action)}]})
        contents.append({"role": "user", "parts": tool_parts})
        if finished:
            entries = _confirmed_tool_loop_entries(session)
            if len(entries) != prepared.target_count:
                raise RuntimeError(
                    f"Tool-loop finished with {len(entries)} point pairs; expected {prepared.target_count}"
                )
            return entries

    raise RuntimeError("Tool-loop registration exceeded the maximum number of steps")


def estimate_registration_correspondences(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    target_landmark_count: int = 8,
    min_edge_landmarks: int = 5,
    workflow: str = "single_pass",
    show_atlas_borders: bool = True,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
) -> list[RegistrationCorrespondence]:
    """Run the configured Gemini registration workflow to produce paired correspondences."""
    vlm_config = importlib.import_module("langslice.vlm.config")
    get_client = cast(Callable[[], Any], getattr(vlm_config, "get_client"))
    prepared = _prepare_registration_inputs(
        image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        target_landmark_count=target_landmark_count,
        min_edge_landmarks=min_edge_landmarks,
        show_atlas_borders=show_atlas_borders,
        debug_dir=debug_dir,
        vlm_config=vlm_config,
    )
    selected_workflow = str(workflow).strip() or str(
        getattr(vlm_config, "default_registration_workflow", lambda _name: "single_pass")(
            prepared.model_name
        )
    )

    if on_progress:
        on_progress(
            "Preparing registration agent inputs: "
            f"slice={prepared.slice_prep.output_size[0]}x{prepared.slice_prep.output_size[1]}px, "
            f"atlas={prepared.atlas_prep.output_size[0]}x{prepared.atlas_prep.output_size[1]}px, "
            f"workflow={selected_workflow}"
        )

    _emit_prepared_registration_trace(
        prepared,
        atlas_name=atlas_name,
        position_mm=position_mm,
        workflow=selected_workflow,
        show_atlas_borders=show_atlas_borders,
        on_trace=on_trace,
    )

    client = get_client()

    if selected_workflow == "single_pass" and getattr(vlm_config, "count_tokens_enabled")():
        try:
            count_contents, count_config = _build_single_pass_request(
                atlas_prep=prepared.atlas_prep,
                slice_prep=prepared.slice_prep,
                region_metadata_text=prepared.region_metadata_text,
                atlas_name=atlas_name,
                atlas_info=prepared.atlas_info,
                position_mm=position_mm,
                target_count=prepared.target_count,
                min_edge=prepared.min_edge,
                thinking_budget=prepared.thinking_budget,
                temperature=prepared.temperature,
                enable_code_execution=prepared.enable_code_execution,
            )
            count_response = client.models.count_tokens(
                model=prepared.model_name,
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

    if selected_workflow == "image_gen_two_shot":
        raw_correspondences = _estimate_correspondences_image_gen_two_shot(
            client,
            prepared=prepared,
            vlm_config=vlm_config,
            atlas_name=atlas_name,
            position_mm=position_mm,
            on_progress=on_progress,
            on_trace=on_trace,
        )
    elif selected_workflow == "multimodal_tool_loop":
        raw_correspondences = _estimate_correspondences_tool_loop(
            client,
            prepared=prepared,
            atlas_name=atlas_name,
            position_mm=position_mm,
            on_progress=on_progress,
            on_trace=on_trace,
        )
    else:
        raw_correspondences = _estimate_correspondences_single_pass(
            client,
            model=prepared.model_name,
            atlas_prep=prepared.atlas_prep,
            slice_prep=prepared.slice_prep,
            region_metadata_text=prepared.region_metadata_text,
            atlas_name=atlas_name,
            atlas_info=prepared.atlas_info,
            position_mm=position_mm,
            target_count=prepared.target_count,
            min_edge=prepared.min_edge,
            thinking_budget=prepared.thinking_budget,
            temperature=prepared.temperature,
            enable_code_execution=prepared.enable_code_execution,
            on_progress=on_progress,
            on_trace=on_trace,
        )

    correspondences = _entries_to_correspondences(
        _filter_visible_entries(raw_correspondences),
        atlas_image_size=prepared.atlas_image.size,
        slice_image_size=image.size,
        target_count=prepared.target_count,
    )

    if not correspondences:
        raise RuntimeError("Registration agent produced no usable correspondences")

    if on_progress:
        on_progress(f"Registration agent proposed {len(correspondences)} correspondences")
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Correspondence set ready",
            summary=f"Accepted {len(correspondences)} correspondences for solving",
            parts=[
                json_part(
                    [
                        {
                            "label": corr.label,
                            "slice_xy": corr.slice_xy,
                            "atlas_xy": corr.atlas_xy,
                            "confidence": corr.confidence,
                            "rationale": corr.rationale,
                        }
                        for corr in correspondences
                    ],
                    label="Accepted correspondences",
                    collapsible=True,
                )
            ],
            metadata={"accepted_count": len(correspondences), "workflow": selected_workflow},
        ),
    )
    return correspondences
