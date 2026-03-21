"""VLM-based brain slice estimation using Gemini."""

import io
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence, cast

from PIL import Image
from google.genai import types
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.ai import config as vlm_config
from langslice.ai.config import get_client

logger = logging.getLogger(__name__)

# Retryable exception types (google-genai SDK error hierarchy)
_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_RETRIES = 3
_INITIAL_BACKOFF_S = 1.0
_RESAMPLE_LANCZOS = getattr(getattr(Image, "Resampling", Image), "LANCZOS")
_HEARTBEAT_INTERVAL_S = 10.0


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


def _retry_generate(
    client: Any,
    *,
    model: str,
    contents: object,
    config: object,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    """Wrapper around client.models.generate_content with exponential backoff."""
    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _run_with_progress_heartbeat(
                lambda: client.models.generate_content(
                    model=model, contents=contents, config=config
                ),
                request_label=f"{request_label} (attempt {attempt + 1}/{_MAX_RETRIES + 1})",
                on_progress=on_progress,
            )
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
        response = _run_with_progress_heartbeat(
            lambda: client.models.count_tokens(
                model=model, contents=contents, config=config or None
            ),
            request_label="AP token preflight",
            on_progress=on_progress,
        )
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


# ---------------------------------------------------------------------------
# Tool handler functions — extracted to estimator_tools.py
# ---------------------------------------------------------------------------
from langslice.ai.estimator_tools import (  # noqa: E402
    _build_nudge_text,
    _extract_generate_function_calls,
    _extract_interaction_function_calls,
    _get_regions_at_position,
    _has_neighbor_bracket,
    _is_broad_multi_sweep,
    _is_narrow_multi_sweep,
    _process_ap_function_calls,
    _sorted_unique_positions,
)

# Debug artifact writing — extracted to estimator_debug.py
from langslice.ai.estimator_debug import write_debug_artifacts  # noqa: E402


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
        interaction = _run_with_progress_heartbeat(
            lambda: client.interactions.create(
                model=model_name,
                input=next_input,
                previous_interaction_id=previous_interaction_id,
                system_instruction=system_instruction,
                tools=[tool_declarations],
                generation_config={"temperature": vlm_config.TEMPERATURE},
            ),
            request_label=f"AP interactions turn {iteration + 1}",
            on_progress=on_progress,
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
    max_iterations: int = 20,
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
    max_iterations = max(1, int(max_iterations))
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
                    image_part_from_pil(
                        target_prepared,
                        label="Target slice sent to Gemini",
                        image_bytes=target_bytes,
                        path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                        metadata={
                            "transport": target_payload.transport,
                            "vlm_scale_factor": target_info["vlm_scale_factor"],
                        },
                    ),
                    json_part(
                        {
                            "transport": target_payload.transport,
                            "temperature": vlm_config.TEMPERATURE,
                            "max_iterations": max_iterations,
                            "prompt": "Here is the target brain slice. Determine its AP position in the atlas.",
                        },
                        label="Request context",
                    ),
                ],
                metadata={
                    "transport": target_payload.transport,
                    "temperature": vlm_config.TEMPERATURE,
                    "max_iterations": max_iterations,
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
                    request_label=f"AP model turn {iteration + 1}",
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
            write_debug_artifacts(
                run_dir=run_dir,
                atlas_name=atlas_name,
                mode_used=mode_used,
                target_info=target_info,
                feature_flags=feature_flags,
                max_iterations=max_iterations,
                state=state,
                final_pos=final_pos,
                final_reasoning=final_reasoning,
                cache_name=cache_name,
                interaction_trace=interaction_trace,
                history=history,
                on_progress=_progress,
            )

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
