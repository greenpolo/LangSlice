"""Shared AP-estimation helpers used by the harness and image-generation adapter."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from typing import Any, cast

from langslice_harness.estimation._shared_common import _APLoopState as _APLoopState
from langslice_harness.estimation._shared_common import _emit_trace as _emit_trace
from langslice_harness.estimation._shared_common import (
    _fetch_atlas_slice_bytes as _fetch_atlas_slice_bytes,
)
from langslice_harness.estimation._shared_common import (
    _get_position_range_lazy as _get_position_range_lazy,
)
from langslice_harness.estimation._shared_common import _GroupLoopState as _GroupLoopState
from langslice_harness.estimation._shared_common import _image_to_bytes as _image_to_bytes
from langslice_harness.estimation._shared_common import _load_atlas_lazy as _load_atlas_lazy
from langslice_harness.estimation._shared_common import _to_float as _to_float

logger = logging.getLogger(__name__)


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


def _history_metrics(contents: Sequence[object]) -> dict[str, int]:
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


def _format_usage_metadata(metadata: dict[str, int | float | str | bool]) -> str:
    preferred_fields = (
        "prompt_token_count",
        "tool_use_prompt_token_count",
        "thoughts_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
    )
    parts = [
        f"{field_name}={value}"
        for field_name in preferred_fields
        if (value := metadata.get(field_name)) is not None
    ]
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
