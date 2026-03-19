"""VLM-based brain slice estimation using Gemini."""

import io
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, cast

import numpy as np
from PIL import Image
from google.genai import types
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
    tool_call_event,
    tool_result_event,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.vlm import config as vlm_config
from langslice.vlm.config import get_client

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
    contents: object,
    config: object,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    """Wrapper around client.models.generate_content with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as exc:
            last_exc = exc
            # Check if the error has a retryable HTTP status code
            status = getattr(exc, "status_code", None) or getattr(exc, "code", None)
            if isinstance(status, int) and status in _RETRYABLE_STATUS_CODES:
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_BACKOFF_S * (2**attempt)
                    msg = f"Gemini API error (status {status}), retrying in {delay:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})"
                    logger.warning(msg)
                    if on_progress:
                        on_progress(msg)
                    time.sleep(delay)
                    continue
            # Also retry on generic connection / timeout errors
            exc_name = type(exc).__name__.lower()
            if any(kw in exc_name for kw in ("timeout", "connection", "transport")):
                if attempt < _MAX_RETRIES:
                    delay = _INITIAL_BACKOFF_S * (2**attempt)
                    msg = f"Transient error ({type(exc).__name__}), retrying in {delay:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})"
                    logger.warning(msg)
                    if on_progress:
                        on_progress(msg)
                    time.sleep(delay)
                    continue
            # Non-retryable error - raise immediately
            raise
    # Exhausted retries
    assert last_exc is not None
    raise last_exc


@dataclass
class APResult:
    position_mm: float
    reasoning: str
    debug_dir: str | None = None


@dataclass
class _APLoopState:
    max_iterations: int
    estimate_result: dict[str, object] | None = None
    reasoning_log: list[dict[str, object]] = field(default_factory=list)
    turn_metrics: list[dict[str, object]] = field(default_factory=list)
    images_fetched: int = 0
    fetched_positions: list[float] = field(default_factory=list)
    saw_broad_sweep: bool = False
    saw_narrow_sweep: bool = False


@dataclass
class _ImagePayload:
    part: types.Part
    interaction_input: dict[str, object] | None
    transport: str
    file_name: str | None = None
    file_uri: str | None = None


def _first_model_content(response: object) -> object:
    candidates = getattr(response, "candidates", None)
    if not isinstance(candidates, list) or not candidates:
        prompt_feedback = getattr(response, "prompt_feedback", None)
        raise RuntimeError(f"Gemini returned no candidates. prompt_feedback={prompt_feedback}")

    first = candidates[0]
    finish_reason = getattr(first, "finish_reason", None)
    content = getattr(first, "content", None)
    if content is None:
        raise RuntimeError(f"Gemini candidate has no content. finish_reason={finish_reason}")
    return content


def _to_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _image_to_bytes(img: Image.Image, fmt: str = "JPEG") -> bytes:
    """Convert PIL Image to raw bytes."""
    buf = io.BytesIO()
    if img.mode != "RGB":
        img = img.convert("RGB")
    if fmt.upper() == "JPEG":
        img.save(buf, format=fmt, quality=95, subsampling=0)
    else:
        img.save(buf, format=fmt)
    return buf.getvalue()


def _file_state_name(file_obj: object) -> str | None:
    state = getattr(file_obj, "state", None)
    if isinstance(state, str):
        return state.upper()
    state_name = getattr(state, "name", None)
    if isinstance(state_name, str):
        return state_name.upper()
    return None


def _wait_for_uploaded_file(
    client: Any,
    *,
    file_name: str,
    timeout_s: float,
    on_progress: Callable[[str], None] | None = None,
) -> object:
    deadline = time.perf_counter() + timeout_s
    while True:
        uploaded = client.files.get(name=file_name)
        state_name = _file_state_name(uploaded)
        if state_name in {None, "ACTIVE"}:
            return uploaded
        if state_name in {"FAILED", "ERROR"}:
            raise RuntimeError(
                f"Gemini File API processing failed for {file_name}: state={state_name}"
            )
        if time.perf_counter() >= deadline:
            raise RuntimeError(
                f"Gemini File API processing timed out for {file_name}: last_state={state_name}"
            )
        if on_progress:
            on_progress(
                f"Waiting for Gemini file '{file_name}' to become ACTIVE (state={state_name})"
            )
        time.sleep(0.5)


def _build_image_payload(
    client: Any,
    *,
    image_bytes: bytes,
    display_name: str,
    use_file_api: bool,
    uploaded_file_names: list[str],
    on_progress: Callable[[str], None] | None = None,
) -> _ImagePayload:
    if not use_file_api:
        return _ImagePayload(
            part=types.Part(inline_data=types.Blob(mime_type="image/jpeg", data=image_bytes)),
            interaction_input=None,
            transport="inline_data",
        )

    payload = io.BytesIO(image_bytes)
    payload.name = f"{display_name}.jpg"
    uploaded = client.files.upload(
        file=payload,
        config=types.UploadFileConfig(mime_type="image/jpeg", display_name=display_name),
    )
    file_name = getattr(uploaded, "name", None)
    if not isinstance(file_name, str) or not file_name:
        raise RuntimeError(f"Gemini File API upload for '{display_name}' returned no file name")
    uploaded_file_names.append(file_name)
    active_file = _wait_for_uploaded_file(
        client,
        file_name=file_name,
        timeout_s=vlm_config.file_poll_timeout_s(),
        on_progress=on_progress,
    )
    file_uri = getattr(active_file, "uri", None) or getattr(uploaded, "uri", None)
    if not isinstance(file_uri, str) or not file_uri:
        raise RuntimeError(f"Gemini File API upload for '{display_name}' returned no URI")
    return _ImagePayload(
        part=types.Part.from_uri(file_uri=file_uri, mime_type="image/jpeg"),
        interaction_input={"type": "image", "uri": file_uri, "mime_type": "image/jpeg"},
        transport="file_api",
        file_name=file_name,
        file_uri=file_uri,
    )


def _inline_data_size(part: object) -> int:
    inline_data = getattr(part, "inline_data", None)
    if inline_data is None:
        return 0

    data = getattr(inline_data, "data", None)
    if isinstance(data, (bytes, bytearray)):
        return len(data)
    return 0


def _has_file_data(part: object) -> bool:
    file_data = getattr(part, "file_data", None)
    if file_data is None:
        return False
    file_uri = getattr(file_data, "file_uri", None)
    return isinstance(file_uri, str) and bool(file_uri)


def _history_metrics(contents: list[object]) -> dict[str, int]:
    metrics = {
        "content_count": len(contents),
        "part_count": 0,
        "text_parts": 0,
        "function_call_parts": 0,
        "function_response_parts": 0,
        "image_parts": 0,
        "image_bytes": 0,
    }

    for content in contents:
        content_parts = getattr(content, "parts", None) or []
        for part in content_parts:
            metrics["part_count"] += 1
            if getattr(part, "text", None):
                metrics["text_parts"] += 1
            if getattr(part, "function_call", None):
                metrics["function_call_parts"] += 1
            if getattr(part, "function_response", None):
                metrics["function_response_parts"] += 1

            blob_size = _inline_data_size(part)
            if blob_size > 0:
                metrics["image_parts"] += 1
                metrics["image_bytes"] += blob_size
                continue

            if _has_file_data(part):
                metrics["image_parts"] += 1

    return metrics


def _extract_usage_metadata(response: object) -> dict[str, int | float | str | bool]:
    usage = getattr(response, "usage_metadata", None)
    usage_fields = (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "thoughts_token_count",
        "cached_content_token_count",
        "tool_use_prompt_token_count",
    )

    metadata: dict[str, int | float | str | bool] = {}
    if usage is None:
        return metadata

    for field in usage_fields:
        value = getattr(usage, field, None)
        if isinstance(value, (int, float, str, bool)):
            metadata[field] = value

    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            dumped_dict = cast(dict[object, object], dumped)
            for field in usage_fields:
                value = dumped_dict.get(field)
                if isinstance(value, (int, float, str, bool)):
                    metadata[field] = value

    return metadata


def _extract_count_tokens_metadata(response: object) -> dict[str, int | float | str | bool]:
    fields = ("total_tokens", "total_billable_characters")
    metadata: dict[str, int | float | str | bool] = {}
    for field in fields:
        value = getattr(response, field, None)
        if isinstance(value, (int, float, str, bool)):
            metadata[field] = value
    return metadata


def _extract_interaction_usage_metadata(interaction: object) -> dict[str, int | float | str | bool]:
    usage = getattr(interaction, "usage", None)
    field_map = {
        "total_input_tokens": "prompt_token_count",
        "total_output_tokens": "candidates_token_count",
        "total_tokens": "total_token_count",
        "total_thought_tokens": "thoughts_token_count",
        "total_cached_tokens": "cached_content_token_count",
        "total_tool_use_tokens": "tool_use_prompt_token_count",
    }
    metadata: dict[str, int | float | str | bool] = {}
    if usage is None:
        return metadata
    for source_field, target_field in field_map.items():
        value = getattr(usage, source_field, None)
        if isinstance(value, (int, float, str, bool)):
            metadata[target_field] = value
    return metadata


def _count_tokens_if_enabled(
    client: Any,
    *,
    model: str,
    contents: object,
    system_instruction: str | None = None,
    tools: list[types.Tool] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> dict[str, int | float | str | bool]:
    if not vlm_config.count_tokens_enabled():
        return {}

    config: dict[str, object] = {}
    if system_instruction:
        config["system_instruction"] = system_instruction
    if tools:
        config["tools"] = tools
    try:
        response = client.models.count_tokens(model=model, contents=contents, config=config or None)
    except Exception as exc:
        message = f"Token preflight failed: {type(exc).__name__}: {exc}"
        logger.warning(message)
        if on_progress:
            on_progress(message)
        return {"error": message}
    return _extract_count_tokens_metadata(response)


def _format_usage_metadata(metadata: dict[str, int | float | str | bool]) -> str:
    preferred_fields = (
        "prompt_token_count",
        "tool_use_prompt_token_count",
        "thoughts_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
    )
    parts: list[str] = []
    for field in preferred_fields:
        value = metadata.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    return ", ".join(parts) if parts else "usage metadata unavailable"


def _format_count_tokens(metadata: dict[str, int | float | str | bool]) -> str:
    parts: list[str] = []
    for field in ("total_tokens", "total_billable_characters"):
        value = metadata.get(field)
        if value is not None:
            parts.append(f"{field}={value}")
    skipped = metadata.get("skipped")
    if isinstance(skipped, str):
        parts.append(f"skipped={skipped}")
    error = metadata.get("error")
    if isinstance(error, str):
        parts.append(error)
    return ", ".join(parts) if parts else "count_tokens unavailable"


def _emit_trace(
    on_trace: Callable[[dict[str, object]], None] | None,
    event: dict[str, object],
) -> None:
    if on_trace:
        on_trace(event)


def _extract_model_text_parts(model_content: types.Content) -> tuple[list[str], list[str]]:
    text_parts: list[str] = []
    thought_parts: list[str] = []
    for part in getattr(model_content, "parts", None) or []:
        text = getattr(part, "text", None)
        if not isinstance(text, str) or not text:
            continue
        if bool(getattr(part, "thought", False)):
            thought_parts.append(text)
        else:
            text_parts.append(text)
    return text_parts, thought_parts


def _extract_interaction_text_outputs(interaction: object) -> tuple[list[str], list[str]]:
    text_outputs: list[str] = []
    thought_outputs: list[str] = []
    for output in getattr(interaction, "outputs", None) or []:
        output_type = getattr(output, "type", None)
        text = getattr(output, "text", None)
        if output_type != "text" or not isinstance(text, str) or not text:
            continue
        if bool(getattr(output, "thought", False)):
            thought_outputs.append(text)
        else:
            text_outputs.append(text)
    return text_outputs, thought_outputs


def _build_nudge_text(state: _APLoopState) -> str:
    if not state.saw_broad_sweep:
        return (
            "Please continue with a broad coarse sweep now. Call `fetch_multiple_atlas_slices` "
            "with 4-5 widely spaced AP positions to find the correct neighborhood before reasoning further."
        )
    if not state.saw_narrow_sweep:
        return (
            "Please continue with a narrowed sweep now. Call `fetch_multiple_atlas_slices` "
            "around your best current neighborhood with tighter spacing before considering submission."
        )
    return (
        "Please continue. Before submitting, verify your leading candidate by checking at least one "
        "lower and one higher neighboring AP position around it using `fetch_multiple_atlas_slices` or `fetch_atlas_slice`."
    )


def _interaction_input_metrics(input_parts: Sequence[Mapping[str, object]]) -> dict[str, int]:
    metrics = {
        "content_count": 1,
        "part_count": len(input_parts),
        "text_parts": 0,
        "function_call_parts": 0,
        "function_response_parts": 0,
        "image_parts": 0,
        "image_bytes": 0,
    }
    for item in input_parts:
        item_type = item.get("type")
        if item_type == "text":
            metrics["text_parts"] += 1
        elif item_type == "image":
            metrics["image_parts"] += 1
        elif item_type == "function_call":
            metrics["function_call_parts"] += 1
        elif item_type == "function_result":
            metrics["function_response_parts"] += 1
    return metrics


def _sorted_unique_positions(
    positions: list[float],
    *,
    tolerance: float = 0.02,
) -> list[float]:
    unique_positions: list[float] = []
    for pos in sorted(positions):
        if not unique_positions or abs(pos - unique_positions[-1]) > tolerance:
            unique_positions.append(pos)
    return unique_positions


def _is_broad_multi_sweep(positions: list[float]) -> bool:
    if len(positions) < 4:
        return False
    return (max(positions) - min(positions)) >= 3.0


def _is_narrow_multi_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
    return (max(positions) - min(positions)) <= 1.5


def _has_neighbor_bracket(
    fetched_positions: list[float],
    center_mm: float,
    *,
    pos_lo: float,
    pos_hi: float,
    tolerance: float = 0.35,
    edge_margin: float = 0.25,
) -> bool:
    unique_positions = _sorted_unique_positions(fetched_positions)
    has_lower = any(center_mm - tolerance <= pos < center_mm for pos in unique_positions)
    has_upper = any(center_mm < pos <= center_mm + tolerance for pos in unique_positions)

    needs_lower = center_mm > pos_lo + edge_margin
    needs_upper = center_mm < pos_hi - edge_margin
    return (has_lower or not needs_lower) and (has_upper or not needs_upper)


def _get_regions_at_position(atlas: object, position_mm: float) -> list[str]:
    """Return brain region names visible at a given AP position."""
    from langslice.atlas.core import position_mm_to_index

    try:
        idx = position_mm_to_index(cast(Any, atlas), position_mm)
    except ValueError:
        return []

    atlas_obj = cast(Any, atlas)
    annotation_slice = np.asarray(atlas_obj.annotation[idx, :, :])
    unique_ids = np.unique(annotation_slice)
    unique_ids = unique_ids[unique_ids > 0]

    structures = atlas_obj.structures
    names: list[str] = []
    for uid in unique_ids[:30]:  # Cap at 30 to avoid huge lists
        uid_int = int(uid)
        if uid_int in structures:
            entry = structures[uid_int]
            names.append(f"{entry['acronym']} ({entry['name']})")
    return names


def _extract_generate_function_calls(
    model_content: types.Content,
) -> tuple[list[dict[str, object]], str | None]:
    model_parts = getattr(model_content, "parts", None) or []
    text_preview: str | None = None
    function_calls: list[dict[str, object]] = []
    for part in model_parts:
        text = getattr(part, "text", None)
        if text_preview is None and isinstance(text, str) and text:
            text_preview = text
        function_call = getattr(part, "function_call", None)
        if function_call is None:
            continue
        args = dict(function_call.args) if getattr(function_call, "args", None) else {}
        function_calls.append(
            {
                "call_id": None,
                "name": getattr(function_call, "name", ""),
                "args": args,
            }
        )
    return function_calls, text_preview


def _extract_interaction_function_calls(
    interaction: object,
) -> tuple[list[dict[str, object]], str | None]:
    outputs = getattr(interaction, "outputs", None) or []
    text_preview: str | None = None
    function_calls: list[dict[str, object]] = []
    for output in outputs:
        output_type = getattr(output, "type", None)
        if output_type == "text" and text_preview is None:
            text = getattr(output, "text", None)
            if isinstance(text, str) and text:
                text_preview = text
        if output_type != "function_call":
            continue
        arguments = getattr(output, "arguments", None)
        args = dict(arguments) if isinstance(arguments, dict) else {}
        function_calls.append(
            {
                "call_id": getattr(output, "id", None),
                "name": getattr(output, "name", ""),
                "args": args,
            }
        )
    return function_calls, text_preview


def _process_ap_function_calls(
    function_calls: list[dict[str, object]],
    *,
    iteration: int,
    atlas: object,
    pos_lo: float,
    pos_hi: float,
    target_h: int,
    run_dir: str | None,
    client: Any,
    use_file_api: bool,
    uploaded_file_names: list[str],
    state: _APLoopState,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> tuple[list[types.Part], list[dict[str, object]]]:
    import os

    atlas_obj = cast(Any, atlas)
    generate_parts: list[types.Part] = []
    interaction_inputs: list[dict[str, object]] = []

    def _append_response(
        *,
        call_id: object,
        name: str,
        response: dict[str, object],
        is_error: bool = False,
    ) -> None:
        generate_parts.append(types.Part.from_function_response(name=name, response=response))
        interaction_inputs.append(
            {
                "type": "function_result",
                "call_id": str(call_id)
                if isinstance(call_id, str) and call_id
                else f"{iteration + 1}:{name}",
                "name": name,
                "result": response,
                "is_error": is_error,
            }
        )

    for call in function_calls:
        name = str(call.get("name", ""))
        args_obj = call.get("args", {})
        args = args_obj if isinstance(args_obj, dict) else {}
        call_id = call.get("call_id")

        _emit_trace(
            on_trace,
            tool_call_event(
                stage="ap",
                tool_name=name,
                args=args,
                iteration=iteration + 1,
            ),
        )

        if on_progress:
            on_progress(f"Tool call [{iteration + 1}]: {name}({args})")

        if name == "fetch_atlas_slice":
            pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
            pos = max(pos_lo, min(pos_hi, pos))
            state.fetched_positions.append(pos)
            try:
                from langslice.atlas.core import get_reference_slice

                ref_img = get_reference_slice(atlas_obj, pos)
                ref_prepared = normalize_image(ref_img)
                scale = target_h / ref_prepared.height
                new_w = max(1, int(round(ref_prepared.width * scale)))
                new_h = max(1, int(round(ref_prepared.height * scale)))
                ref_scaled = ref_prepared.resize((new_w, new_h), _RESAMPLE_LANCZOS)
                ref_bytes = _image_to_bytes(ref_scaled)
                image_payload = _build_image_payload(
                    client,
                    image_bytes=ref_bytes,
                    display_name=f"ap_{iteration + 1:02d}_slice_{pos:.3f}mm",
                    use_file_api=use_file_api,
                    uploaded_file_names=uploaded_file_names,
                    on_progress=on_progress,
                )
                state.images_fetched += 1

                if run_dir:
                    ref_scaled.save(
                        os.path.join(run_dir, f"tool_{iteration + 1:02d}_slice_{pos:.2f}mm.jpg"),
                        quality=95,
                    )

                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "position_mm": pos,
                        "status": "ok",
                        "description": f"Atlas coronal section at {pos:.2f}mm from anterior edge",
                    },
                )
                generate_parts.append(image_payload.part)
                if image_payload.interaction_input is not None:
                    interaction_inputs.append(image_payload.interaction_input)
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Image at {pos:.2f}mm via {image_payload.transport}",
                    }
                )
                image_path = (
                    os.path.join(run_dir, f"tool_{iteration + 1:02d}_slice_{pos:.2f}mm.jpg")
                    if run_dir
                    else None
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary=f"Fetched atlas slice at {pos:.2f} mm",
                        parts=[
                            image_part_from_pil(
                                ref_scaled,
                                label=f"Atlas slice {pos:.2f} mm",
                                image_bytes=ref_bytes,
                                path=image_path,
                                metadata={"transport": image_payload.transport},
                            )
                        ],
                        metadata={
                            "iteration": iteration + 1,
                            "position_mm": round(pos, 3),
                            "transport": image_payload.transport,
                        },
                    ),
                )
            except ValueError as exc:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={"status": "error", "error": str(exc)},
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Error: {exc}",
                    }
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary=f"Error: {exc}",
                        parts=[json_part({"error": str(exc)}, label="Error")],
                        metadata={"iteration": iteration + 1, "status": "error"},
                    ),
                )

        elif name == "fetch_multiple_atlas_slices":
            positions_list = args.get("positions_mm", [])
            if not isinstance(positions_list, list):
                positions_list = []

            positions = [max(pos_lo, min(pos_hi, float(p))) for p in positions_list[:5]]
            state.fetched_positions.extend(positions)
            if positions and _is_broad_multi_sweep(positions):
                state.saw_broad_sweep = True
            if positions and _is_narrow_multi_sweep(positions):
                state.saw_narrow_sweep = True

            if not positions:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={"status": "error", "error": "No valid positions provided"},
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": "Error: empty input",
                    }
                )
                continue

            successes: list[str] = []
            image_parts: list[dict[str, object]] = []
            from langslice.atlas.core import get_reference_slice

            for pos in positions:
                try:
                    ref_img = get_reference_slice(atlas_obj, pos)
                    ref_prepared = normalize_image(ref_img)
                    scale = target_h / ref_prepared.height
                    new_w = max(1, int(round(ref_prepared.width * scale)))
                    new_h = max(1, int(round(ref_prepared.height * scale)))
                    ref_scaled = ref_prepared.resize((new_w, new_h), _RESAMPLE_LANCZOS)
                    ref_bytes = _image_to_bytes(ref_scaled)
                    image_payload = _build_image_payload(
                        client,
                        image_bytes=ref_bytes,
                        display_name=f"ap_{iteration + 1:02d}_multi_{pos:.3f}mm",
                        use_file_api=use_file_api,
                        uploaded_file_names=uploaded_file_names,
                        on_progress=on_progress,
                    )
                    state.images_fetched += 1

                    if run_dir:
                        ref_scaled.save(
                            os.path.join(
                                run_dir, f"tool_{iteration + 1:02d}_multi_{pos:.2f}mm.jpg"
                            ),
                            quality=95,
                        )

                    _append_response(
                        call_id=call_id,
                        name=name,
                        response={
                            "position_mm": pos,
                            "status": "ok",
                            "description": f"Atlas coronal section at {pos:.2f}mm",
                        },
                    )
                    generate_parts.append(image_payload.part)
                    if image_payload.interaction_input is not None:
                        interaction_inputs.append(image_payload.interaction_input)
                    successes.append(f"{pos:.2f}mm")
                    image_path = (
                        os.path.join(run_dir, f"tool_{iteration + 1:02d}_multi_{pos:.2f}mm.jpg")
                        if run_dir
                        else None
                    )
                    image_parts.append(
                        image_part_from_pil(
                            ref_scaled,
                            label=f"Atlas slice {pos:.2f} mm",
                            image_bytes=ref_bytes,
                            path=image_path,
                            metadata={"transport": image_payload.transport},
                        )
                    )
                except Exception as exc:
                    _append_response(
                        call_id=call_id,
                        name=name,
                        response={"position_mm": pos, "status": "error", "error": str(exc)},
                        is_error=True,
                    )

            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Fetched {len(successes)} slices: {', '.join(successes)}",
                }
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Fetched {len(successes)} atlas slices",
                    parts=image_parts
                    or [json_part({"positions_mm": positions}, label="Positions")],
                    metadata={"iteration": iteration + 1, "positions_mm": positions},
                ),
            )

        elif name == "get_atlas_info":
            from langslice.atlas.core import get_atlas_info as _get_atlas_info_core

            info = _get_atlas_info_core(atlas_obj)
            info["coordinate_note"] = "0.0mm is extreme Anterior; higher mm is more Posterior."
            _append_response(call_id=call_id, name=name, response=info)
            state.reasoning_log.append(
                {"iteration": iteration + 1, "tool": name, "args": {}, "result": str(info)}
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary="Atlas metadata returned",
                    parts=[json_part(info, label="Atlas info")],
                    metadata={"iteration": iteration + 1},
                ),
            )
            if on_progress:
                on_progress(f"  -> Atlas range: [{pos_lo:.2f}, {pos_hi:.2f}] mm")

        elif name == "get_region_names":
            pos = float(args.get("position_mm", (pos_lo + pos_hi) / 2))
            regions = _get_regions_at_position(atlas, pos)
            _append_response(
                call_id=call_id,
                name=name,
                response={"position_mm": pos, "regions": regions},
            )
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"{len(regions)} regions",
                }
            )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Returned {len(regions)} visible regions at {pos:.2f} mm",
                    parts=[json_part({"position_mm": pos, "regions": regions}, label="Regions")],
                    metadata={"iteration": iteration + 1, "position_mm": round(pos, 3)},
                ),
            )
            if on_progress:
                on_progress(f"  -> {len(regions)} regions at {pos:.2f}mm")

        elif name == "submit_estimate":
            est_pos = float(args.get("position_mm", 0.0))
            est_confidence = str(args.get("confidence", "unknown"))
            est_reasoning = str(args.get("reasoning", ""))
            has_neighbor_check = _has_neighbor_bracket(
                state.fetched_positions,
                est_pos,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
            )
            near_iteration_limit = iteration >= state.max_iterations - 2

            if not state.saw_broad_sweep and not near_iteration_limit:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": "Run a broad `fetch_multiple_atlas_slices` sweep before submitting.",
                    },
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected submit at {est_pos:.2f}mm: no broad sweep yet",
                    }
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary="Submit rejected: broad sweep required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            if not state.saw_narrow_sweep and not near_iteration_limit:
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": "Run a narrowed `fetch_multiple_atlas_slices` sweep around your best candidate before submitting.",
                    },
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected submit at {est_pos:.2f}mm: no narrow sweep yet",
                    }
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary="Submit rejected: narrow sweep required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            if not has_neighbor_check and not near_iteration_limit:
                lower = max(pos_lo, est_pos - 0.2)
                upper = min(pos_hi, est_pos + 0.2)
                _append_response(
                    call_id=call_id,
                    name=name,
                    response={
                        "status": "error",
                        "error": (
                            "Before submitting, verify at least one lower and one higher neighboring AP position "
                            f"around {est_pos:.2f} mm (for example {lower:.2f} mm and {upper:.2f} mm)."
                        ),
                    },
                    is_error=True,
                )
                state.reasoning_log.append(
                    {
                        "iteration": iteration + 1,
                        "tool": name,
                        "args": args,
                        "result": f"Rejected submit at {est_pos:.2f}mm: neighborhood not bracketed",
                    }
                )
                _emit_trace(
                    on_trace,
                    tool_result_event(
                        stage="ap",
                        tool_name=name,
                        summary="Submit rejected: neighboring AP checks required",
                        parts=[json_part(args, label="Rejected submit")],
                        metadata={"iteration": iteration + 1, "status": "rejected"},
                    ),
                )
                continue

            state.estimate_result = {
                "position_mm": est_pos,
                "confidence": est_confidence,
                "reasoning": est_reasoning,
            }
            state.reasoning_log.append(
                {
                    "iteration": iteration + 1,
                    "tool": name,
                    "args": args,
                    "result": f"Submitted {est_pos:.2f}mm ({est_confidence})",
                }
            )
            if on_progress:
                on_progress(
                    f"Agent submitted estimate: {est_pos:.2f}mm (confidence: {est_confidence})"
                )
            _emit_trace(
                on_trace,
                tool_result_event(
                    stage="ap",
                    tool_name=name,
                    summary=f"Submitted estimate {est_pos:.2f} mm ({est_confidence})",
                    parts=[json_part(state.estimate_result, label="Submitted estimate")],
                    metadata={"iteration": iteration + 1, "status": "accepted"},
                ),
            )

        else:
            _append_response(
                call_id=call_id,
                name=name,
                response={"status": "error", "error": f"Unknown tool: {name}"},
                is_error=True,
            )

    return generate_parts, interaction_inputs


def _run_interactions_ap_loop(
    client: Any,
    *,
    model_name: str,
    system_instruction: str,
    tool_declarations: types.Tool,
    initial_input: list[dict[str, object]],
    atlas: object,
    pos_lo: float,
    pos_hi: float,
    target_h: int,
    run_dir: str | None,
    uploaded_file_names: list[str],
    state: _APLoopState,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    interaction_trace: list[dict[str, object]] = []
    previous_interaction_id: str | None = None
    next_input = initial_input

    for iteration in range(state.max_iterations):
        request_metrics = _interaction_input_metrics(next_input)
        turn_metric: dict[str, object] = {
            "iteration": iteration + 1,
            "request": request_metrics,
            "mode": "interactions",
        }
        if iteration == 0:
            preflight_parts: list[types.Part] = []
            for item in next_input:
                item_type = item.get("type")
                if item_type == "text":
                    text = item.get("text")
                    if isinstance(text, str):
                        preflight_parts.append(types.Part(text=text))
                elif item_type == "image":
                    uri = item.get("uri")
                    if isinstance(uri, str):
                        preflight_parts.append(
                            types.Part.from_uri(file_uri=uri, mime_type="image/jpeg")
                        )
            preflight = _count_tokens_if_enabled(
                client,
                model=model_name,
                contents=preflight_parts,
                system_instruction=system_instruction,
                tools=[tool_declarations],
                on_progress=on_progress,
            )
            if preflight:
                turn_metric["preflight_count_tokens"] = preflight
                if on_progress:
                    on_progress(f"Turn 1 token preflight: {_format_count_tokens(preflight)}")

        if on_progress:
            on_progress(
                f"Interactions turn {iteration + 1}: sending {request_metrics['part_count']} parts, "
                f"{request_metrics['image_parts']} images ({request_metrics['image_bytes']} bytes)"
            )

        started_at = time.perf_counter()
        interaction = client.interactions.create(
            model=model_name,
            input=next_input,
            previous_interaction_id=previous_interaction_id,
            system_instruction=system_instruction,
            tools=[tool_declarations],
            generation_config={"temperature": vlm_config.TEMPERATURE},
        )
        turn_metric["wall_time_s"] = round(time.perf_counter() - started_at, 3)
        usage_metadata = _extract_interaction_usage_metadata(interaction)
        turn_metric["usage_metadata"] = usage_metadata
        state.turn_metrics.append(turn_metric)
        previous_interaction_id = cast(str | None, getattr(interaction, "id", None))

        outputs = getattr(interaction, "outputs", None) or []
        interaction_trace.append(
            {
                "iteration": iteration + 1,
                "input": next_input,
                "outputs": [
                    getattr(output, "model_dump", lambda: None)()
                    if callable(getattr(output, "model_dump", None))
                    else {
                        "type": getattr(output, "type", None),
                        "text": getattr(output, "text", None),
                        "name": getattr(output, "name", None),
                    }
                    for output in outputs
                ],
            }
        )

        text_outputs, thought_outputs = _extract_interaction_text_outputs(interaction)
        if text_outputs or thought_outputs:
            trace_parts: list[dict[str, object]] = []
            if thought_outputs:
                trace_parts.append(
                    json_part(thought_outputs, label="Reasoning summary", collapsible=True)
                )
            if text_outputs:
                trace_parts.append(json_part(text_outputs, label="Model text", collapsible=True))
            _emit_trace(
                on_trace,
                model_event(
                    stage="ap",
                    title=f"Model turn {iteration + 1}",
                    summary="Model returned text before the next tool step",
                    parts=trace_parts,
                    metadata={"iteration": iteration + 1, **usage_metadata},
                ),
            )

        function_calls, text_preview = _extract_interaction_function_calls(interaction)
        if not function_calls:
            if text_preview and on_progress:
                on_progress(f"Agent reasoning/text: {text_preview[:200]}...")
            next_input = [{"type": "text", "text": _build_nudge_text(state)}]
            continue

        _, interaction_inputs = _process_ap_function_calls(
            function_calls,
            iteration=iteration,
            atlas=atlas,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            target_h=target_h,
            run_dir=run_dir,
            client=client,
            use_file_api=True,
            uploaded_file_names=uploaded_file_names,
            state=state,
            on_progress=on_progress,
            on_trace=on_trace,
        )
        if state.estimate_result:
            break
        next_input = interaction_inputs

    return interaction_trace


def estimate_position(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
) -> APResult:
    """Agentic AP estimation using tool-use with self-correction.

    The model receives tools to explore the atlas freely:
    - fetch_atlas_slice: view any coronal section
    - get_atlas_info: get coordinate range and metadata
    - get_region_names: see what brain regions exist at a position
    - submit_estimate: declare the final answer

    Uses manual function calling so atlas images can be injected alongside
    tool responses. The model runs until it submits or hits max iterations.

    Set ``LANGSLICE_VLM_DEBUG_DIR`` to save all artifacts for review.
    """
    import os
    from datetime import datetime
    from google.genai import types
    from langslice.atlas.core import (
        get_position_range_mm,
        get_reference_slice,
        load_atlas,
        get_atlas_info as _get_atlas_info_core,
    )

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_client()
    atlas = load_atlas(atlas_name)
    pos_lo, pos_hi = get_position_range_mm(atlas)

    target_normalized = normalize_image(image)
    target_prep = prepare_image_for_vlm(target_normalized)
    target_prepared = target_prep.image
    target_bytes = _image_to_bytes(target_prepared)
    target_h = target_prepared.height
    target_info = {
        "original_width": target_prep.original_size[0],
        "original_height": target_prep.original_size[1],
        "width": target_prepared.width,
        "height": target_prepared.height,
        "mode": target_prepared.mode,
        "jpeg_bytes": len(target_bytes),
        "vlm_scale_factor": round(target_prep.scale_factor, 6),
    }
    if target_prep.downsampled:
        _progress(
            "Target image resized for Gemini budget: "
            f"{target_prep.original_size[0]}x{target_prep.original_size[1]}px -> "
            f"{target_prepared.width}x{target_prepared.height}px "
            f"(scale={target_prep.scale_factor:.4f})"
        )
    _progress(
        "Target image prepared for Gemini: "
        f"{target_prepared.width}x{target_prepared.height}px, "
        f"mode={target_prepared.mode}, jpeg_bytes={len(target_bytes)}"
    )

    # --- Debug artifact setup ---
    debug_root = debug_dir if debug_dir is not None else os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    run_dir: str | None = None
    if debug_root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_atlas = atlas_name.replace("/", "_").replace("\\", "_")
        run_dir = os.path.join(debug_root, f"{timestamp}_{safe_atlas}")
        os.makedirs(run_dir, exist_ok=True)
        target_prepared.save(os.path.join(run_dir, "target.jpg"), quality=95)
        _progress(f"Debug artifacts -> {run_dir}")

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Prepared AP estimation inputs",
            summary=(
                f"Target image {target_prepared.width}x{target_prepared.height}px prepared for Gemini"
            ),
            parts=[
                image_part_from_pil(
                    target_prepared,
                    label="Target slice",
                    image_bytes=target_bytes,
                    path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                    metadata={
                        "transport": "pending",
                        "vlm_scale_factor": target_info["vlm_scale_factor"],
                    },
                )
            ],
            metadata=target_info,
        ),
    )

    # --- Tool declarations ---
    tool_declarations = types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="fetch_atlas_slice",
                description=(
                    "Fetch a coronal brain atlas reference image at a specific "
                    "anterior-posterior position. The image will be shown to you. "
                    "Use this to visually compare against the target slice."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "position_mm": {
                            "type": "number",
                            "description": "AP position in mm from the anterior edge of the atlas",
                        },
                    },
                    "required": ["position_mm"],
                },
            ),
            types.FunctionDeclaration(
                name="get_atlas_info",
                description=(
                    "Get atlas metadata including the valid AP coordinate range, "
                    "resolution, species, and number of slices."
                ),
                parameters_json_schema={"type": "object", "properties": {}},
            ),
            types.FunctionDeclaration(
                name="get_region_names",
                description=(
                    "Get the names and acronyms of brain regions visible at a "
                    "specific AP position. Useful for confirming anatomical identity."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "position_mm": {
                            "type": "number",
                            "description": "AP position in mm from the anterior edge",
                        },
                    },
                    "required": ["position_mm"],
                },
            ),
            types.FunctionDeclaration(
                name="fetch_multiple_atlas_slices",
                description=(
                    "Fetch up to 5 coronal brain atlas reference images at multiple "
                    "anterior-posterior positions at once. The images will be shown to you "
                    "in order. Use this to perform a rapid coarse sweep (e.g., check every 2mm) "
                    "to quickly narrow down the general neighborhood."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "positions_mm": {
                            "type": "array",
                            "items": {"type": "number"},
                            "description": "List of up to 5 AP positions in mm to fetch",
                        },
                    },
                    "required": ["positions_mm"],
                },
            ),
            types.FunctionDeclaration(
                name="submit_estimate",
                description=(
                    "Submit your final AP position estimate. Only call this when "
                    "you are confident in your answer."
                ),
                parameters_json_schema={
                    "type": "object",
                    "properties": {
                        "position_mm": {
                            "type": "number",
                            "description": "Final estimated AP position in mm from the anterior edge",
                        },
                        "confidence": {
                            "type": "string",
                            "description": "Confidence level: low, medium, or high",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Detailed reasoning for the estimate",
                        },
                    },
                    "required": ["position_mm", "confidence", "reasoning"],
                },
            ),
        ]
    )

    # --- System prompt ---
    system_instruction = (
        "You are an expert neuroanatomist. You are given a histology brain slice image "
        "and must determine its Anterior-Posterior (AP) position within a reference atlas. "
        "The coordinate system is: 0.0 mm is the extreme anterior edge (e.g. olfactory bulb), "
        "while larger mm values move posterior toward the cerebellum and brainstem. "
        "You have tools to fetch atlas reference images at any AP coordinate, query which "
        "brain regions exist at a given position, and get atlas metadata.\n\n"
        "RECOMMENDED STRATEGY:\n"
        "1. Coarse Sweep: Call `fetch_multiple_atlas_slices` with 4-5 widely spaced coordinates "
        "   (e.g., 2.0, 4.0, 6.0, 8.0) as your first real image search step to instantly find the correct neighborhood.\n"
        "2. Finer Search: Identify the closest match, then call `fetch_multiple_atlas_slices` "
        "   again around that match with tighter spacing (e.g., +/-0.5 mm).\n"
        "3. Verification: Once narrowed down, check specific structural landmarks or use "
        "   `get_region_names` to confirm anatomical identity. Before submitting, compare at least one lower and one higher neighboring AP position around your leading candidate.\n"
        "4. Submit: Call `submit_estimate` only when you are highly confident.\n\n"
        "Do not guess blindly; use the tools to narrow down the answer methodically. Avoid long thought-only turns: either perform the next search step or submit once the neighborhood is bracketed."
    )

    feature_flags = vlm_config.feature_flags()
    requested_file_api = vlm_config.ap_use_file_api()
    requested_cache = vlm_config.ap_use_context_cache()
    requested_interactions = vlm_config.ap_use_interactions()

    use_file_api = requested_file_api and vlm_config.supports_file_api()
    use_context_cache = requested_cache
    use_interactions = requested_interactions and vlm_config.supports_interactions_api()

    if requested_file_api and not use_file_api:
        _progress(
            "AP File API requested but current backend does not support Gemini File API; using inline images."
        )
    if requested_interactions and not use_interactions:
        _progress(
            "Interactions API requested but current backend does not support it; using generate_content loop."
        )
    if use_interactions and not use_file_api:
        _progress(
            "Interactions pilot requires Gemini File API image references; enabling File API transport for this run."
        )
        use_file_api = True
    if use_interactions and use_context_cache:
        _progress("Interactions pilot bypasses AP context caching for this run.")
        use_context_cache = False

    feature_flags["effective_ap_use_file_api"] = use_file_api
    feature_flags["effective_ap_use_context_cache"] = use_context_cache
    feature_flags["effective_ap_use_interactions"] = use_interactions

    thinking_level = getattr(types.ThinkingLevel, vlm_config.THINKING_LEVEL, None)
    max_iterations = 20
    state = _APLoopState(max_iterations=max_iterations)
    history: list[types.Content] = []
    interaction_trace: list[dict[str, object]] = []
    uploaded_file_names: list[str] = []
    cache_name: str | None = None
    mode_used = "generate_content"

    try:
        target_payload = _build_image_payload(
            client,
            image_bytes=target_bytes,
            display_name="target_slice",
            use_file_api=use_file_api,
            uploaded_file_names=uploaded_file_names,
            on_progress=_progress,
        )
        target_info["input_transport"] = target_payload.transport
        _emit_trace(
            on_trace,
            runtime_event(
                stage="ap",
                title="Initial AP request queued",
                summary="Target slice attached to the first model turn",
                parts=[
                    json_part(
                        {
                            "transport": target_payload.transport,
                            "temperature": vlm_config.TEMPERATURE,
                            "prompt": "Here is the target brain slice. Determine its AP position in the atlas.",
                        },
                        label="Request context",
                    )
                ],
                metadata={
                    "transport": target_payload.transport,
                    "temperature": vlm_config.TEMPERATURE,
                },
            ),
        )

        initial_parts = [
            types.Part(
                text="Here is the target brain slice. Determine its AP position in the atlas."
            ),
            target_payload.part,
        ]
        history = [types.Content(role="user", parts=initial_parts)]

        if use_context_cache:
            try:
                cache = client.caches.create(
                    model=vlm_config.MODEL_NAME,
                    config=types.CreateCachedContentConfig(
                        contents=[
                            types.Content(
                                role="user",
                                parts=[
                                    types.Part(text="Target brain slice image for AP estimation."),
                                    target_payload.part,
                                ],
                            )
                        ],
                        system_instruction=system_instruction,
                        tools=[tool_declarations],
                        ttl=vlm_config.ap_cache_ttl(),
                    ),
                )
                cache_name_obj = getattr(cache, "name", None)
                if isinstance(cache_name_obj, str) and cache_name_obj:
                    cache_name = cache_name_obj
                    history = [
                        types.Content(
                            role="user",
                            parts=[
                                types.Part(
                                    text="Determine the AP position in the atlas for the cached target brain slice."
                                )
                            ],
                        )
                    ]
                    _progress(f"AP cache created: {cache_name}")
                else:
                    _progress(
                        "AP cache creation returned no cache name; continuing without cached content."
                    )
            except Exception as exc:
                _progress(
                    f"AP cache creation failed; continuing without cached content: {type(exc).__name__}: {exc}"
                )
                cache_name = None

        if cache_name:
            config = types.GenerateContentConfig(
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                cached_content=cache_name,
                temperature=vlm_config.TEMPERATURE,
            )
        else:
            config = types.GenerateContentConfig(
                tools=[tool_declarations],
                automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                thinking_config=types.ThinkingConfig(thinking_level=thinking_level),
                system_instruction=system_instruction,
                temperature=vlm_config.TEMPERATURE,
            )

        _progress(f"Starting agentic estimation (max {max_iterations} tool calls)...")

        if use_interactions:
            initial_interaction_input: list[dict[str, object]] = [
                {
                    "type": "text",
                    "text": "Here is the target brain slice. Determine its AP position in the atlas.",
                },
            ]
            if target_payload.interaction_input is not None:
                initial_interaction_input.append(target_payload.interaction_input)
            try:
                interaction_trace = _run_interactions_ap_loop(
                    client,
                    model_name=vlm_config.MODEL_NAME,
                    system_instruction=system_instruction,
                    tool_declarations=tool_declarations,
                    initial_input=initial_interaction_input,
                    atlas=atlas,
                    pos_lo=pos_lo,
                    pos_hi=pos_hi,
                    target_h=target_h,
                    run_dir=run_dir,
                    uploaded_file_names=uploaded_file_names,
                    state=state,
                    on_progress=_progress,
                    on_trace=on_trace,
                )
                mode_used = "interactions"
            except Exception as exc:
                _progress(
                    f"Interactions API pilot failed; falling back to generate_content loop: {type(exc).__name__}: {exc}"
                )
                state = _APLoopState(max_iterations=max_iterations)
                interaction_trace = []
                mode_used = "generate_content"

        if mode_used == "generate_content":
            for iteration in range(max_iterations):
                request_metrics = _history_metrics(cast(list[object], history))
                turn_metric: dict[str, object] = {
                    "iteration": iteration + 1,
                    "request": request_metrics,
                    "mode": "generate_content",
                }
                if iteration == 0 and not cache_name:
                    preflight = _count_tokens_if_enabled(
                        client,
                        model=vlm_config.MODEL_NAME,
                        contents=history,
                        system_instruction=None if cache_name else system_instruction,
                        tools=None if cache_name else [tool_declarations],
                        on_progress=_progress,
                    )
                    if preflight:
                        turn_metric["preflight_count_tokens"] = preflight
                        _progress(f"Turn 1 token preflight: {_format_count_tokens(preflight)}")
                elif iteration == 0 and cache_name:
                    turn_metric["preflight_count_tokens"] = {"skipped": "cached_content_active"}

                _progress(
                    f"Turn {iteration + 1}: sending {request_metrics['content_count']} messages, "
                    f"{request_metrics['part_count']} parts, {request_metrics['image_parts']} images "
                    f"({request_metrics['image_bytes']} bytes)"
                )
                turn_started_at = time.perf_counter()
                response = _retry_generate(
                    client,
                    model=vlm_config.MODEL_NAME,
                    contents=history,
                    config=config,
                    on_progress=_progress,
                )
                wall_time_s = time.perf_counter() - turn_started_at
                usage_metadata = _extract_usage_metadata(response)
                turn_metric["wall_time_s"] = round(wall_time_s, 3)
                turn_metric["usage_metadata"] = usage_metadata
                state.turn_metrics.append(turn_metric)
                _progress(
                    f"Turn {iteration + 1} completed in {wall_time_s:.2f}s; "
                    f"{_format_usage_metadata(usage_metadata)}"
                )

                model_content = cast(types.Content, _first_model_content(response))
                history.append(model_content)
                function_calls, text_preview = _extract_generate_function_calls(model_content)
                text_parts, thought_parts = _extract_model_text_parts(model_content)

                if text_parts or thought_parts:
                    trace_parts: list[dict[str, object]] = []
                    if thought_parts:
                        trace_parts.append(
                            json_part(thought_parts, label="Reasoning summary", collapsible=True)
                        )
                    if text_parts:
                        trace_parts.append(
                            json_part(text_parts, label="Model text", collapsible=True)
                        )
                    _emit_trace(
                        on_trace,
                        model_event(
                            stage="ap",
                            title=f"Model turn {iteration + 1}",
                            summary="Model responded before the next tool step",
                            parts=trace_parts,
                            metadata={"iteration": iteration + 1, **usage_metadata},
                        ),
                    )

                if not function_calls:
                    if text_preview:
                        _progress(f"Agent reasoning/text: {text_preview[:200]}...")
                    else:
                        _progress("Agent produced thought block but no tool calls.")
                    history.append(
                        types.Content(
                            role="user", parts=[types.Part(text=_build_nudge_text(state))]
                        )
                    )
                    continue

                tool_response_parts, _ = _process_ap_function_calls(
                    function_calls,
                    iteration=iteration,
                    atlas=atlas,
                    pos_lo=pos_lo,
                    pos_hi=pos_hi,
                    target_h=target_h,
                    run_dir=run_dir,
                    client=client,
                    use_file_api=use_file_api,
                    uploaded_file_names=uploaded_file_names,
                    state=state,
                    on_progress=_progress,
                    on_trace=on_trace,
                )
                if state.estimate_result:
                    break
                if tool_response_parts:
                    history.append(types.Content(role="user", parts=tool_response_parts))

        final_pos: float
        final_reasoning: str
        if state.estimate_result:
            final_pos = _to_float(state.estimate_result.get("position_mm"), (pos_lo + pos_hi) / 2)
            final_reasoning = str(state.estimate_result["reasoning"])
        else:
            final_pos = (pos_lo + pos_hi) / 2
            final_reasoning = "Agent did not submit an estimate within the iteration limit."
            _progress(f"Warning: Agent did not submit. Falling back to midpoint: {final_pos:.2f}mm")

        _progress(
            f"Final position estimated: {final_pos:.2f} mm ({state.images_fetched} atlas images fetched)"
        )
        _emit_trace(
            on_trace,
            runtime_event(
                stage="ap",
                title="AP estimation completed",
                summary=f"Final position {final_pos:.2f} mm",
                parts=[
                    json_part(
                        {
                            "position_mm": final_pos,
                            "reasoning": final_reasoning,
                            "images_fetched": state.images_fetched,
                        },
                        label="AP result",
                    )
                ],
                metadata={"images_fetched": state.images_fetched},
            ),
        )

        feature_flags["effective_ap_use_file_api"] = (
            target_info.get("input_transport") == "file_api"
        )
        feature_flags["effective_ap_use_context_cache"] = cache_name is not None
        feature_flags["effective_ap_use_interactions"] = mode_used == "interactions"

        if run_dir:
            reasoning_path = os.path.join(run_dir, "reasoning.txt")
            with open(reasoning_path, "w", encoding="utf-8") as f:
                f.write(f"AP Estimation - {atlas_name} - {datetime.now().isoformat()}\n")
                f.write(f"Model: {vlm_config.MODEL_NAME}\n")
                f.write(f"Temperature: {vlm_config.TEMPERATURE:.2f}\n")
                f.write(f"Mode: {mode_used}\n")
                f.write(
                    "Target Image: "
                    f"{target_info['width']}x{target_info['height']} px, "
                    f"mode={target_info['mode']}, jpeg_bytes={target_info['jpeg_bytes']}, "
                    f"transport={target_info['input_transport']}\n"
                )
                f.write(f"Feature Flags: {json.dumps(feature_flags, sort_keys=True)}\n")
                f.write("=" * 60 + "\n\n")
                f.write("TURN TELEMETRY\n")
                f.write("-" * 60 + "\n")
                for turn in state.turn_metrics:
                    request = cast(dict[str, object], turn["request"])
                    usage = cast(
                        dict[str, int | float | str | bool], turn.get("usage_metadata", {})
                    )
                    preflight = cast(
                        dict[str, int | float | str | bool], turn.get("preflight_count_tokens", {})
                    )
                    f.write(
                        f"Turn {turn['iteration']}: mode={turn.get('mode')}, wall_time_s={turn.get('wall_time_s')}, "
                        f"messages={request.get('content_count')}, parts={request.get('part_count')}, "
                        f"images={request.get('image_parts')}, image_bytes={request.get('image_bytes')}\n"
                    )
                    if preflight:
                        f.write(f"    preflight: {_format_count_tokens(preflight)}\n")
                    f.write(f"    usage: {_format_usage_metadata(usage)}\n")
                f.write("\n")
                for entry in state.reasoning_log:
                    f.write(f"[{entry['iteration']}] {entry['tool']}({entry.get('args', {})})\n")
                    f.write(f"    -> {entry['result']}\n\n")
                f.write("=" * 60 + "\n")
                f.write(f"FINAL ESTIMATE: {final_pos:.2f} mm\n")
                if state.estimate_result:
                    f.write(f"CONFIDENCE: {state.estimate_result.get('confidence', 'N/A')}\n")
                    f.write(f"REASONING: {state.estimate_result.get('reasoning', 'N/A')}\n")
                f.write("=" * 60 + "\n")

                if interaction_trace:
                    f.write("\n\nINTERACTIONS TRACE\n")
                    f.write("=" * 60 + "\n\n")
                    for turn in interaction_trace:
                        f.write(f"--- turn {turn['iteration']} input ---\n")
                        f.write(json.dumps(turn["input"], indent=2))
                        f.write("\n--- outputs ---\n")
                        f.write(json.dumps(turn["outputs"], indent=2))
                        f.write("\n\n")
                else:
                    f.write("\n\nFULL CONVERSATION HISTORY\n")
                    f.write("=" * 60 + "\n\n")
                    for content in history:
                        role = content.role if hasattr(content, "role") else "?"
                        f.write(f"--- {role} ---\n")
                        content_parts = content.parts or []
                        for part in content_parts:
                            if part.text:
                                f.write(f"  TEXT: {part.text[:500]}\n")
                            if part.function_call:
                                f.write(
                                    f"  CALL: {part.function_call.name}({dict(part.function_call.args) if part.function_call.args else {}})\n"
                                )
                            if part.function_response:
                                f.write(
                                    f"  RESPONSE: {part.function_response.name} -> {part.function_response.response}\n"
                                )
                            if part.inline_data:
                                blob_data = part.inline_data.data
                                data_len = (
                                    len(blob_data)
                                    if isinstance(blob_data, (bytes, bytearray))
                                    else 0
                                )
                                f.write(
                                    f"  IMAGE: {part.inline_data.mime_type} ({data_len} bytes)\n"
                                )
                            if _has_file_data(part):
                                file_data = getattr(part, "file_data", None)
                                file_uri = getattr(file_data, "file_uri", None)
                                f.write(f"  IMAGE: file_uri={file_uri}\n")
                        f.write("\n")

            telemetry_path = os.path.join(run_dir, "telemetry.json")
            with open(telemetry_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model": vlm_config.MODEL_NAME,
                        "mode": mode_used,
                        "atlas_name": atlas_name,
                        "target_image": target_info,
                        "feature_flags": feature_flags,
                        "temperature": vlm_config.TEMPERATURE,
                        "max_iterations": max_iterations,
                        "images_fetched": state.images_fetched,
                        "cache_name": cache_name,
                        "turns": state.turn_metrics,
                        "final_estimate_mm": final_pos,
                        "final_reasoning": final_reasoning,
                    },
                    f,
                    indent=2,
                )

            _progress(f"Reasoning log saved -> {reasoning_path}")
            _progress(f"Telemetry saved -> {telemetry_path}")

        return APResult(
            position_mm=final_pos,
            reasoning=final_reasoning,
            debug_dir=run_dir,
        )
    finally:
        if cache_name:
            try:
                client.caches.delete(name=cache_name)
            except Exception as exc:
                logger.warning("Failed to delete Gemini cache %s: %s", cache_name, exc)
        for file_name in reversed(uploaded_file_names):
            try:
                client.files.delete(name=file_name)
            except Exception as exc:
                logger.warning("Failed to delete Gemini file %s: %s", file_name, exc)


def estimate_ap(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
) -> APResult:
    return estimate_position(
        image=image,
        atlas_name=atlas_name,
        on_progress=on_progress,
    )
