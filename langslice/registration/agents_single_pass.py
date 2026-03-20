"""Single-pass registration workflow for text-centric multimodal LLMs.

Targets models with structured output and thinking support such as
gemini-3-flash-preview and gemini-3.1-pro-preview.  The model receives
both the atlas and histology images in one turn and returns all paired
correspondences as a single JSON response.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

import langslice.registration.agents as _agents
from langslice.agent_trace import json_part, model_event, runtime_event

logger = logging.getLogger(__name__)

_SINGLE_PASS_SYSTEM_INSTRUCTION = """
You are a neuroanatomy registration assistant producing paired landmark correspondences.
RULES:
1. The atlas and histology depict the same coronal section, but their appearance, scale, rotation, and local distortion may differ.
2. Never copy, scale, rotate, or project coordinates from one image onto the other. Every point must come from direct visual inspection.
3. Output atlas_point_2d and slice_point_2d as [y, x] arrays. Use whatever coordinate system is most natural to you — pixel coordinates, normalized 0-1000, percentages, etc. State which system you chose in the coordinate_system field.
4. Work one correspondence at a time, visually confirming both atlas and histology before moving to the next pair.
5. Reason bidirectionally: sometimes start from atlas to slice, and sometimes start from slice to atlas.
6. Points must fall on real tissue, not background, padding, or space outside the section.
7. If a reliable match is not visible, set status to not_visible and both coordinates to [0, 0].
8. Prefer a spatially distributed mix of outer contour anchors, midline points, cavity or tract corners, and interior boundaries.
9. Include hemisphere or midline cues in labels whenever possible to reduce left-right swaps.
10. Do not waste effort naming anatomy if identity is uncertain. Use concise geometric labels instead of guessing structure names.
""".strip()


def _build_single_pass_schema(target_count: int) -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "coordinate_system": {
                "type": "string",
                "description": "The coordinate system used for all points (e.g. 'pixel', 'normalized_0_1000', 'percentage', etc.)",
            },
            "correspondences": {
                "type": "array",
                "minItems": target_count,
                "maxItems": target_count,
                "items": {
                    "type": "object",
                    "properties": {
                        "atlas_point_2d": {
                            "type": "array",
                            "description": "Atlas coordinate as [y, x].",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2,
                        },
                        "slice_point_2d": {
                            "type": "array",
                            "description": "Histology coordinate as [y, x].",
                            "items": {"type": "number"},
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
            },
        },
        "required": ["coordinate_system", "correspondences"],
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
    thinking_level: str,
    temperature: float,
    enable_code_execution: bool,
    show_atlas_borders: bool = True,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    atlas_w, atlas_h = atlas_prep.image.size
    slice_w, slice_h = slice_prep.image.size
    prompt = (
        "TASK: Inspect both images and produce paired atlas-to-histology landmarks for registration.\n\n"
        f"Return exactly {target_count} correspondence objects.\n"
        f"Include at least {min_edge} outer contour anchors when visible.\n"
        "Also include midline points, cavity or tract corners, and interior boundaries when they are reliable.\n"
        "Distribute the accepted points across the whole section instead of clustering them.\n\n"
        f"Atlas: {atlas_name}\n"
        f"AP position: {position_mm:.3f} mm\n"
        f"Atlas shape: {atlas_info.get('shape')}\n"
        f"Atlas resolution (um): {atlas_info.get('resolution_um')}\n"
        f"Atlas image dimensions: {atlas_w}x{atlas_h} pixels\n"
        f"Histology image dimensions: {slice_w}x{slice_h} pixels\n\n"
        f"{region_metadata_text}\n\n"
        "Procedure:\n"
        "1. Examine atlas and histology together.\n"
        "2. Select one reliable correspondence at a time.\n"
        "3. For each pair, verify the atlas point and the slice point independently before moving on.\n"
        "4. Use short labels with hemisphere or midline cues whenever possible.\n"
        "5. If a reliable match cannot be confirmed, mark it not_visible instead of forcing a guess.\n\n"
        "Output requirements:\n"
        "- atlas_point_2d and slice_point_2d as [y, x] in whatever coordinate system you find most natural.\n"
        "- Set coordinate_system to describe your choice (e.g. 'pixel', 'normalized_0_1000', etc.).\n"
        "- status must be found, uncertain, or not_visible.\n"
        "- Never copy or mechanically transform coordinates from one image to the other."
    )

    atlas_label = (
        "Image 1: Atlas reference with boundary overlays."
        if show_atlas_borders
        else "Image 1: Atlas reference."
    )
    contents: list[dict[str, object]] = [
        {
            "role": "user",
            "parts": [
                {"text": prompt},
                {"text": atlas_label},
                _agents._image_to_inline_data(atlas_prep.image),
                {"text": "Image 2: Histology slice."},
                _agents._image_to_inline_data(slice_prep.image),
            ],
        }
    ]

    config: dict[str, object] = {
        "system_instruction": _SINGLE_PASS_SYSTEM_INSTRUCTION,
        "thinking_config": {"thinking_level": thinking_level},
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
    thinking_level: str,
    temperature: float,
    enable_code_execution: bool,
    show_atlas_borders: bool = True,
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
        thinking_level=thinking_level,
        temperature=temperature,
        enable_code_execution=enable_code_execution,
        show_atlas_borders=show_atlas_borders,
    )

    if on_progress:
        on_progress(
            "Registration: locating paired correspondences "
            f"(thinking_level={thinking_level.lower()})..."
        )

    response = _agents._retry_generate(
        client,
        model=model,
        contents=contents,
        config=config,
        request_label="Registration single-pass model call",
        on_progress=on_progress,
    )
    text_parts, thought_parts = _agents._extract_response_text_parts(response)
    if text_parts or thought_parts:
        trace_parts: list[dict[str, object]] = []
        if thought_parts:
            trace_parts.append(
                json_part(thought_parts, label="Reasoning summary", collapsible=True)
            )
        if text_parts:
            trace_parts.append(json_part(text_parts, label="Model text", collapsible=True))
        _agents._emit_trace(
            on_trace,
            model_event(
                stage="registration",
                title="Registration model response",
                summary="Model returned text alongside the structured correspondence output",
                parts=trace_parts,
                metadata={"thinking_level": thinking_level},
            ),
        )
    parsed = _agents._extract_result(response)
    correspondences = parsed.get("correspondences", [])
    if not isinstance(correspondences, list) or not correspondences:
        raise RuntimeError("Single-pass registration returned no correspondences")
    _agents._emit_trace(
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
