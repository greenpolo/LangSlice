"""Registration agent runtime — shared utilities and workflow router."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import importlib
import io
import json
import logging
import os
import threading
import time
from typing import Any, cast

from PIL import Image

from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    runtime_event,
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
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
)

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_HEARTBEAT_INTERVAL_S = 10.0

_MAX_REGION_METADATA = 30
_TOOL_LOOP_MAX_STEPS = 24


@dataclass
class _PreparedRegistrationInputs:
    atlas_image: Image.Image
    slice_image_size: tuple[int, int]
    atlas_info: dict[str, object]
    atlas_prep: Any
    slice_prep: Any
    region_metadata_text: str
    target_count: int
    min_edge: int
    model_name: str
    thinking_level: str
    temperature: float
    enable_code_execution: bool
    tool_loop_max_steps: int
    registration_dir: str | None
    atlas_path: str | None
    slice_path: str | None


# ---------------------------------------------------------------------------
# Retry / heartbeat helpers
# ---------------------------------------------------------------------------


def _retry_generate(
    client: Any,
    *,
    model: str,
    contents: object,
    config: object,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        attempt_label = f"{request_label} (attempt {attempt + 1}/{_MAX_RETRIES + 1})"
        try:
            return _run_with_progress_heartbeat(
                lambda: client.models.generate_content(
                    model=model, contents=contents, config=config
                ),
                request_label=attempt_label,
                on_progress=on_progress,
            )
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


def _format_elapsed_seconds(elapsed_s: float) -> str:
    if elapsed_s < 60.0:
        return f"{elapsed_s:.1f}s"
    minutes = int(elapsed_s // 60.0)
    seconds = int(round(elapsed_s - (minutes * 60)))
    return f"{minutes}m {seconds:02d}s"


def _run_with_progress_heartbeat(
    fn: Callable[[], Any],
    *,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
    heartbeat_interval_s: float = _HEARTBEAT_INTERVAL_S,
) -> Any:
    started_at = time.perf_counter()
    stop_event = threading.Event()
    heartbeat_thread: threading.Thread | None = None

    if on_progress:
        on_progress(f"{request_label}: request started")
        if heartbeat_interval_s > 0:

            def _heartbeat_loop() -> None:
                while not stop_event.wait(heartbeat_interval_s):
                    elapsed_s = time.perf_counter() - started_at
                    on_progress(
                        f"{request_label}: still waiting for Gemini after {_format_elapsed_seconds(elapsed_s)}"
                    )

            heartbeat_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
            heartbeat_thread.start()

    try:
        result = fn()
    except Exception as exc:
        elapsed_s = time.perf_counter() - started_at
        if on_progress:
            on_progress(
                f"{request_label}: request failed after {_format_elapsed_seconds(elapsed_s)} "
                f"({type(exc).__name__}: {exc})"
            )
        raise
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.05)

    elapsed_s = time.perf_counter() - started_at
    if on_progress:
        on_progress(f"{request_label}: response received in {_format_elapsed_seconds(elapsed_s)}")
    return result


# ---------------------------------------------------------------------------
# Image / JSON / coordinate helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Input preparation and region metadata
# ---------------------------------------------------------------------------


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
    enable_code_execution: bool | None = None,
    tool_loop_max_steps: int | None = None,
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
    thinking_level = str(getattr(vlm_config, "THINKING_LEVEL", "HIGH")).strip().upper()
    temperature = float(getattr(vlm_config, "TEMPERATURE", 0.5))
    supports_code_execution = cast(
        Callable[[str | None], bool],
        getattr(
            vlm_config,
            "supports_code_execution",
            lambda current_model: str(current_model).strip() == "gemini-3-flash-preview",
        ),
    )
    code_execution_requested = (
        bool(enable_code_execution)
        if enable_code_execution is not None
        else bool(getattr(vlm_config, "CODE_EXECUTION_ENABLED", False))
    )
    enable_code_execution = code_execution_requested and supports_code_execution(model_name)
    resolved_tool_loop_max_steps = max(
        1,
        int(
            tool_loop_max_steps
            if tool_loop_max_steps is not None
            else getattr(vlm_config, "REGISTRATION_TOOL_LOOP_MAX_STEPS", _TOOL_LOOP_MAX_STEPS)
        ),
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
        slice_image_size=image.size,
        atlas_info=atlas_info,
        atlas_prep=atlas_prep,
        slice_prep=slice_prep,
        region_metadata_text=region_metadata_text,
        target_count=target_count,
        min_edge=min_edge,
        model_name=model_name,
        thinking_level=thinking_level,
        temperature=temperature,
        enable_code_execution=enable_code_execution,
        tool_loop_max_steps=resolved_tool_loop_max_steps,
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
                    label="Atlas reference with boundaries"
                    if show_atlas_borders
                    else "Atlas reference",
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
                        "thinking_level": prepared.thinking_level,
                        "temperature": prepared.temperature,
                        "enable_code_execution": prepared.enable_code_execution,
                        "tool_loop_max_steps": prepared.tool_loop_max_steps,
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
                "thinking_level": prepared.thinking_level,
                "temperature": prepared.temperature,
                "enable_code_execution": prepared.enable_code_execution,
                "tool_loop_max_steps": prepared.tool_loop_max_steps,
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
    pixel_coordinates: bool = False,
) -> list[RegistrationCorrespondence]:
    correspondences: list[RegistrationCorrespondence] = []
    for entry in entries:
        try:
            a_y, a_x = _extract_normalized_point(
                entry.get("atlas_point_2d"), field_name="atlas_point_2d"
            )
            s_y, s_x = _extract_normalized_point(
                entry.get("slice_point_2d"), field_name="slice_point_2d"
            )
        except (TypeError, ValueError) as exc:
            logger.info(
                "Rejected correspondence %s due to invalid point: %s",
                entry.get("label", "unknown"),
                exc,
            )
            continue

        if pixel_coordinates:
            a_px_x, a_px_y = float(a_x), float(a_y)
            s_px_x, s_px_y = float(s_x), float(s_y)
        else:
            a_px_x, a_px_y = _normalized_to_pixel_xy(a_y, a_x, image_size=atlas_image_size)
            s_px_x, s_px_y = _normalized_to_pixel_xy(s_y, s_x, image_size=slice_image_size)
        rationale_parts = [f"status={entry.get('status', 'found')}"]
        if entry.get("feature_description"):
            rationale_parts.append(str(entry.get("feature_description")))
        if entry.get("artifact_note"):
            rationale_parts.append(f"artifact={entry.get('artifact_note')}")

        corr_kwargs: dict[str, object] = {
            "slice_xy": (s_px_x, s_px_y),
            "atlas_xy": (a_px_x, a_px_y),
            "label": str(entry.get("label", f"landmark_{len(correspondences) + 1}")),
            "confidence": "medium",
            "rationale": "; ".join(rationale_parts),
        }
        if not pixel_coordinates:
            corr_kwargs["slice_normalized_yx"] = (s_y, s_x)
            corr_kwargs["atlas_normalized_yx"] = (a_y, a_x)
        correspondences.append(RegistrationCorrespondence(**corr_kwargs))  # type: ignore[arg-type]

    if not correspondences:
        raise RuntimeError("Registration agent produced no usable correspondences")
    return correspondences[:target_count]


# ---------------------------------------------------------------------------
# Public entry point — workflow router
# ---------------------------------------------------------------------------


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
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
    debug_dir: str | None = None,
    enable_code_execution: bool | None = None,
    tool_loop_max_steps: int | None = None,
) -> list[RegistrationCorrespondence]:
    """Run the configured Gemini registration workflow to produce paired correspondences."""
    # Late imports to avoid circular dependencies and to keep workflow modules
    # patchable via monkeypatch on this module.
    from langslice.registration.agents_image_gen import (
        _estimate_correspondences_image_gen_two_shot,
    )
    from langslice.registration.agents_single_pass import (
        _build_single_pass_request,
        _estimate_correspondences_single_pass,
    )
    from langslice.registration.agents_tool_loop import (
        _estimate_correspondences_tool_loop,
    )

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
        enable_code_execution=enable_code_execution,
        tool_loop_max_steps=tool_loop_max_steps,
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
                thinking_level=prepared.thinking_level,
                temperature=prepared.temperature,
                enable_code_execution=prepared.enable_code_execution,
            )
            count_response = _run_with_progress_heartbeat(
                lambda: client.models.count_tokens(
                    model=prepared.model_name,
                    contents=count_contents,
                    config={"system_instruction": count_config.get("system_instruction")},
                ),
                request_label="Registration token preflight",
                on_progress=on_progress,
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
            atlas_name=atlas_name,
            position_mm=position_mm,
            on_progress=on_progress,
            on_trace=on_trace,
            on_annotation_session=on_annotation_session,
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
            thinking_level=prepared.thinking_level,
            temperature=prepared.temperature,
            enable_code_execution=prepared.enable_code_execution,
            show_atlas_borders=show_atlas_borders,
            on_progress=on_progress,
            on_trace=on_trace,
        )

    correspondences = _entries_to_correspondences(
        _filter_visible_entries(raw_correspondences),
        atlas_image_size=prepared.atlas_image.size,
        slice_image_size=image.size,
        target_count=prepared.target_count,
        pixel_coordinates=(selected_workflow == "image_gen_two_shot"),
    )

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
