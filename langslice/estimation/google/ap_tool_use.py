"""VLM-based brain slice estimation using Gemini Interactions API."""

import io
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from google.genai import types
from PIL import Image

import langslice.vlm_config as vlm_config
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    runtime_event,
)
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.retry import (
    format_elapsed_seconds as _format_elapsed_seconds,  # noqa: F401 — re-exported
)
from langslice.retry import (
    retry_with_backoff,
)
from langslice.retry import (
    run_with_progress_heartbeat as _run_with_progress_heartbeat,  # noqa: F401 — re-exported
)
from langslice.vlm_config import get_client

logger = logging.getLogger(__name__)

_RESAMPLE_LANCZOS = Image.Resampling.LANCZOS


def _retry_interaction(
    client: Any,
    create_kwargs: dict[str, Any],
    *,
    request_label: str,
    on_progress: Callable[[str], None] | None = None,
) -> Any:
    """Call client.interactions.create with exponential backoff retries."""
    return retry_with_backoff(
        lambda: client.interactions.create(**create_kwargs),
        request_label=request_label,
        on_progress=on_progress,
    )


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
        img.save(buf, format=fmt, quality=85)
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

    for field_name in usage_fields:
        value = getattr(usage, field_name, None)
        if isinstance(value, (int, float, str, bool)):
            metadata[field_name] = value

    model_dump = getattr(usage, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        if isinstance(dumped, dict):
            dumped_dict = cast(dict[object, object], dumped)
            for field_name in usage_fields:
                value = dumped_dict.get(field_name)
                if isinstance(value, (int, float, str, bool)):
                    metadata[field_name] = value

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
    for field_name in preferred_fields:
        value = metadata.get(field_name)
        if value is not None:
            parts.append(f"{field_name}={value}")
    return ", ".join(parts) if parts else "usage metadata unavailable"


def _format_count_tokens(metadata: dict[str, int | float | str | bool]) -> str:
    parts: list[str] = []
    for field_name in ("total_tokens", "total_billable_characters"):
        value = metadata.get(field_name)
        if value is not None:
            parts.append(f"{field_name}={value}")
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
# Debug artifact writing — extracted to estimator_debug.py
from langslice.estimation.debug import write_debug_artifacts  # noqa: E402
from langslice.estimation.google.tool_definitions import (  # noqa: E402
    _build_nudge_text,
    _extract_function_calls,
    _process_ap_function_calls,
    _tool_dicts,
)


def estimate_position(
    image: Image.Image,
    atlas_name: str,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    max_iterations: int = 20,
    media_resolution: str = "ultra_high",
    show_borders: bool = False,
    anatomy_hints: str = "",
    model_name: str | None = None,
    send_individually: bool = True,
    atlas_resolution: int = 1024,
) -> APResult:
    """Agentic AP estimation using tool-use with self-correction.

    The model receives tools to explore the atlas freely:
    - fetch_atlas_slice: view any coronal section
    - get_atlas_info: get coordinate range and metadata
    - get_region_names: see what brain regions exist at a position
    - submit_estimate: declare the final answer

    Uses the Gemini Interactions API with server-side conversation state.
    The model runs until it submits or hits max iterations.

    Set ``LANGSLICE_VLM_DEBUG_DIR`` to save all artifacts for review.
    """
    import os
    from datetime import datetime

    def _progress(msg: str) -> None:
        if on_progress:
            on_progress(msg)
        logger.info(msg)

    client = get_client()
    atlas = _load_atlas_lazy(atlas_name)
    pos_lo, pos_hi = _get_position_range_lazy(atlas)

    target_normalized = normalize_image(image)
    target_prep = prepare_image_for_vlm(target_normalized)
    target_prepared = target_prep.image
    target_bytes = _image_to_bytes(target_prepared)
    target_h = target_prepared.height
    target_info: dict[str, Any] = {
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
        target_prepared.save(os.path.join(run_dir, "target.jpg"), quality=85)
        _progress(f"Debug artifacts -> {run_dir}")

    _emit_trace(
        on_trace,
        runtime_event(
            stage="ap",
            title="Prepared AP estimation inputs",
            summary=(
                f"Target image "
                f"{target_prepared.width}x{target_prepared.height}px "
                f"prepared for Gemini"
            ),
            parts=[
                image_part_from_pil(
                    target_prepared,
                    label="Target slice",
                    image_bytes=target_bytes,
                    path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                    metadata={
                        "transport": "file_api",
                        "vlm_scale_factor": target_info["vlm_scale_factor"],
                    },
                )
            ],
            metadata=target_info,
        ),
    )

    # --- System prompt ---
    system_instruction = (
        "You are an expert neuroanatomist. You are given a histology brain slice image "
        "and must determine its Anterior-Posterior (AP) position within a reference atlas. "
        "The coordinate system is: 0.0 mm is the extreme anterior edge (e.g. olfactory bulb), "
        "while larger mm values move posterior toward the cerebellum and brainstem. "
        "You have tools to fetch atlas reference images at any AP coordinate, query which "
        "brain regions exist at a given position, and get atlas metadata.\n\n"
        f"{anatomy_hints}"
        "RECOMMENDED STRATEGY:\n"
        "1. Call `fetch_atlas` with broadly spaced positions (e.g., [2, 4, 6, 8, 10]) "
        "   to find the general region.\n"
        "2. Call `fetch_atlas` with tighter positions around your best match.\n"
        "3. Call `fetch_atlas` with very fine positions (e.g., 0.1-0.2mm apart) to pinpoint.\n"
        "4. Verify neighbors, then submit.\n\n"
        "You can request up to 8 positions per call, spaced however you like — "
        "cluster them densely around a candidate, or spread them widely to explore.\n\n"
        "IMPORTANT: If at ANY point the atlas images don't look similar to the "
        "target, DO NOT continue narrowing in the same area. Go back and try a "
        "completely different region. It is better to restart your search than to "
        "commit to a wrong neighborhood.\n\n"
        "Think carefully before each tool call, but always follow up with an action."
    )

    feature_flags = vlm_config.feature_flags()
    temperature = vlm_config.TEMPERATURE
    thinking_level = vlm_config.THINKING_LEVEL

    max_iterations = max(1, int(max_iterations))
    state = _APLoopState(max_iterations=max_iterations)
    interaction_trace: list[dict[str, object]] = []
    uploaded_files: list[Any] = []

    try:
        # --- Upload target image via File API ---
        target_buf = io.BytesIO(target_bytes)
        target_file = client.files.upload(
            file=target_buf,
            config=types.UploadFileConfig(
                mime_type="image/jpeg",
                display_name="target_slice",
            ),
        )
        target_file_name = getattr(target_file, "name", None)
        if not isinstance(target_file_name, str) or not target_file_name:
            raise RuntimeError("Gemini File API upload for target slice returned no file name")
        uploaded_files.append(target_file)

        _wait_for_uploaded_file(
            client,
            file_name=target_file_name,
            timeout_s=30.0,
            on_progress=_progress,
        )
        target_uri = getattr(target_file, "uri", None)
        if not isinstance(target_uri, str) or not target_uri:
            raise RuntimeError("Gemini File API upload for target slice returned no URI")

        target_info["input_transport"] = "file_api"
        _emit_trace(
            on_trace,
            runtime_event(
                stage="ap",
                title="Initial AP request queued",
                summary="Target slice uploaded via File API",
                parts=[
                    image_part_from_pil(
                        target_prepared,
                        label="Target slice sent to Gemini",
                        image_bytes=target_bytes,
                        path=os.path.join(run_dir, "target.jpg") if run_dir else None,
                        metadata={
                            "transport": "file_api",
                            "vlm_scale_factor": target_info["vlm_scale_factor"],
                        },
                    ),
                    json_part(
                        {
                            "transport": "file_api",
                            "temperature": temperature,
                            "thinking_level": thinking_level,
                            "max_iterations": max_iterations,
                        },
                        label="Request context",
                    ),
                ],
                metadata={
                    "transport": "file_api",
                    "temperature": temperature,
                    "thinking_level": thinking_level,
                    "max_iterations": max_iterations,
                },
            ),
        )

        # --- Build initial input ---
        initial_input: list[dict[str, object]] = [
            {
                "type": "text",
                "text": "Here is the target brain slice. Determine its AP position in the atlas.",
            },
            {
                "type": "image",
                "uri": target_uri,
                "mime_type": "image/jpeg",
                "resolution": media_resolution,
            },
        ]

        # --- Tool declarations ---
        tools = _tool_dicts()

        # --- Main Interactions API loop ---
        _progress(f"Starting agentic estimation (max {max_iterations} tool calls)...")

        previous_interaction_id: str | None = None
        next_input: list[dict[str, object]] = initial_input

        for iteration in range(max_iterations):
            request_metrics = _interaction_input_metrics(next_input)
            turn_metric: dict[str, object] = {
                "iteration": iteration + 1,
                "request": request_metrics,
                "mode": "interactions",
            }

            if on_progress:
                part_count = request_metrics['part_count']
                img_count = request_metrics['image_parts']
                img_bytes = request_metrics['image_bytes']
                _progress(
                    f"Interactions turn {iteration + 1}: "
                    f"sending {part_count} parts, "
                    f"{img_count} images ({img_bytes} bytes)"
                )

            # Build interaction request kwargs
            create_kwargs: dict[str, Any] = {
                "model": model_name or vlm_config.MODEL_NAME,
                "input": next_input,
                "tools": tools,
                "system_instruction": system_instruction,
                "generation_config": {
                    "temperature": temperature,
                    "thinking_level": thinking_level,
                    "max_output_tokens": 4000,
                },
            }
            if previous_interaction_id is not None:
                create_kwargs["previous_interaction_id"] = previous_interaction_id

            started_at = time.perf_counter()
            interaction = _retry_interaction(
                client,
                create_kwargs,
                request_label=f"AP interactions turn {iteration + 1}",
                on_progress=_progress,
            )
            turn_metric["wall_time_s"] = round(time.perf_counter() - started_at, 3)
            usage_metadata = _extract_interaction_usage_metadata(interaction)
            turn_metric["usage_metadata"] = usage_metadata
            state.turn_metrics.append(turn_metric)
            previous_interaction_id = cast(str | None, getattr(interaction, "id", None))

            _progress(
                f"Turn {iteration + 1} completed in {turn_metric['wall_time_s']}s; "
                f"{_format_usage_metadata(usage_metadata)}"
            )

            # Record interaction trace for debug artifacts
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

            # Emit trace for model text/thought outputs
            text_outputs, thought_outputs = _extract_interaction_text_outputs(interaction)
            if text_outputs or thought_outputs:
                trace_parts: list[dict[str, object]] = []
                if thought_outputs:
                    trace_parts.append(
                        json_part(thought_outputs, label="Reasoning summary", collapsible=True)
                    )
                if text_outputs:
                    trace_parts.append(
                        json_part(
                            text_outputs,
                            label="Model text",
                            collapsible=True,
                        )
                    )
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

            # Extract function calls
            function_calls, text_preview = _extract_function_calls(interaction)

            if not function_calls:
                if text_preview and on_progress:
                    _progress(f"Agent reasoning/text: {text_preview[:200]}...")
                next_input = [{"type": "text", "text": _build_nudge_text(state)}]
                continue

            # Process tool calls
            interaction_inputs = _process_ap_function_calls(
                function_calls,
                iteration=iteration,
                atlas=atlas,
                pos_lo=pos_lo,
                pos_hi=pos_hi,
                target_h=target_h,
                run_dir=run_dir,
                state=state,
                target_image=target_prepared,
                media_resolution=media_resolution,
                show_borders=show_borders,
                send_individually=send_individually,
                atlas_resolution=atlas_resolution,
                on_progress=_progress,
                on_trace=on_trace,
            )
            if state.estimate_result:
                break
            next_input = interaction_inputs

        # --- Finalize result ---
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
            f"Final position estimated: {final_pos:.2f} mm "
            f"({state.images_fetched} atlas images fetched)"
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

        feature_flags["effective_ap_use_file_api"] = True
        feature_flags["effective_ap_use_context_cache"] = False
        feature_flags["effective_ap_use_interactions"] = True

        if run_dir:
            write_debug_artifacts(
                run_dir=run_dir,
                atlas_name=atlas_name,
                mode_used="interactions",
                target_info=target_info,
                feature_flags=feature_flags,
                max_iterations=max_iterations,
                state=state,
                final_pos=final_pos,
                final_reasoning=final_reasoning,
                cache_name=None,
                interaction_trace=interaction_trace,
                history=[],
                on_progress=_progress,
            )

        return APResult(
            position_mm=final_pos,
            reasoning=final_reasoning,
            debug_dir=run_dir,
        )
    finally:
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception as exc:
                logger.warning("Failed to delete Gemini file %s: %s", getattr(f, "name", "?"), exc)


# ---------------------------------------------------------------------------
# Lazy atlas imports to avoid circular dependencies
# ---------------------------------------------------------------------------


def _load_atlas_lazy(atlas_name: str) -> Any:
    from langslice.atlas.core import load_atlas
    return load_atlas(atlas_name)


def _get_position_range_lazy(atlas: Any) -> tuple[float, float]:
    from langslice.atlas.core import get_position_range_mm
    return get_position_range_mm(atlas)
