"""Debug artifact writing for the AP estimation loop.

This module contains the function that writes reasoning logs, telemetry JSON,
and conversation history to disk when a debug run directory is available.
Extracted from ``estimator.py`` purely for readability — no behavioral changes.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast

from langslice.ai import config as vlm_config

if TYPE_CHECKING:
    from langslice.ai.estimator import _APLoopState


def write_debug_artifacts(
    *,
    run_dir: str,
    atlas_name: str,
    mode_used: str,
    target_info: dict[str, Any],
    feature_flags: dict[str, Any],
    max_iterations: int,
    state: _APLoopState,
    final_pos: float,
    final_reasoning: str,
    cache_name: str | None,
    interaction_trace: list[dict[str, object]],
    history: list[Any],
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """Write reasoning log and telemetry JSON to *run_dir*."""
    # Lazy imports to avoid circular dependency with estimator.py
    from langslice.ai.estimator import (
        _format_count_tokens,
        _format_usage_metadata,
        _has_file_data,
    )

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
                f"Turn {turn['iteration']}: "
                f"mode={turn.get('mode')}, "
                f"wall_time_s={turn.get('wall_time_s')}, "
                f"messages={request.get('content_count')}, "
                f"parts={request.get('part_count')}, "
                f"images={request.get('image_parts')}, "
                f"image_bytes={request.get('image_bytes')}\n"
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
                        fc = part.function_call
                        fc_args = dict(fc.args) if fc.args else {}
                        f.write(
                            f"  CALL: {fc.name}({fc_args})\n"
                        )
                    if part.function_response:
                        fr = part.function_response
                        f.write(
                            f"  RESPONSE: {fr.name} "
                            f"-> {fr.response}\n"
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

    if on_progress:
        on_progress(f"Reasoning log saved -> {reasoning_path}")
        on_progress(f"Telemetry saved -> {telemetry_path}")
