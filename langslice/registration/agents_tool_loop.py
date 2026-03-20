"""Multimodal tool-loop registration workflow for text-centric LLMs.

Targets models with structured output and tool-use support such as
gemini-3-flash-preview, gemini-3.1-pro-preview, and gemini-3.1-flash-lite.
The model iteratively inspects images, places visible point annotations,
zooms in to verify placement, and finishes when enough pairs are confirmed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import Any, cast

from PIL import Image

import langslice.registration.agents as _agents
from langslice.agent_trace import (
    image_part_from_pil,
    json_part,
    model_event,
    tool_call_event,
    tool_result_event,
)
from langslice.registration.types import (
    LandmarkAnnotation,
    RegistrationAnnotationSession,
    render_landmark_annotations,
)

logger = logging.getLogger(__name__)


def _compute_zoom_window(
    image_size: tuple[int, int], *, center_yx: tuple[float, float], zoom: float
) -> tuple[int, int, int, int]:
    width, height = image_size
    zoom = max(1.0, float(zoom))
    crop_width = max(32, int(round(width / zoom)))
    crop_height = max(32, int(round(height / zoom)))
    center_x, center_y = _agents._normalized_to_pixel_xy(
        center_yx[0], center_yx[1], image_size=image_size
    )
    left = max(0, int(round(center_x - (crop_width / 2.0))))
    top = max(0, int(round(center_y - (crop_height / 2.0))))
    right = min(width, left + crop_width)
    bottom = min(height, top + crop_height)
    left = max(0, right - crop_width)
    top = max(0, bottom - crop_height)
    return left, top, right, bottom


def _crop_zoom_view(
    image: Image.Image, *, center_yx: tuple[float, float], zoom: float
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    width, height = image.size
    left, top, right, bottom = _compute_zoom_window((width, height), center_yx=center_yx, zoom=zoom)
    cropped = image.crop((left, top, right, bottom)).resize(image.size, Image.Resampling.BICUBIC)
    return cropped, (left, top, right, bottom)


def _pixel_xy_to_normalized_yx(
    px_x: float,
    px_y: float,
    *,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    width, height = image_size
    norm_x = (px_x * 1000.0) / max(width - 1, 1)
    norm_y = (px_y * 1000.0) / max(height - 1, 1)
    return norm_y, norm_x


def _window_to_normalized_bounds(
    window_px: tuple[int, int, int, int], *, image_size: tuple[int, int]
) -> dict[str, list[float]]:
    left, top, right, bottom = window_px
    top_left_yx = _pixel_xy_to_normalized_yx(left, top, image_size=image_size)
    bottom_right_yx = _pixel_xy_to_normalized_yx(
        max(left, right - 1), max(top, bottom - 1), image_size=image_size
    )
    return {
        "top_left": [round(top_left_yx[0], 2), round(top_left_yx[1], 2)],
        "bottom_right": [round(bottom_right_yx[0], 2), round(bottom_right_yx[1], 2)],
    }


def _resolve_action_point(
    tool_args: dict[str, object],
    *,
    image_role: str,
    image_size: tuple[int, int],
    session: RegistrationAnnotationSession,
) -> tuple[float, float]:
    global_field = f"{image_role}_point_2d"
    local_field = f"{image_role}_point_2d_local"
    global_value = tool_args.get(global_field)
    if global_value is not None:
        return _agents._extract_normalized_point(global_value, field_name=global_field)

    local_value = tool_args.get(local_field)
    if local_value is None:
        raise RuntimeError(
            f"Tool-loop point placement is missing '{global_field}' or '{local_field}'"
        )
    last_zoom_pair = session.metadata.get("last_zoom_pair")
    if not isinstance(last_zoom_pair, dict):
        raise RuntimeError(
            f"Received local coordinates for {image_role} without an active zoom window"
        )
    window_px = last_zoom_pair.get(f"{image_role}_window_px")
    if (
        not isinstance(window_px, tuple)
        or len(window_px) != 4
        or not all(isinstance(value, int) for value in window_px)
    ):
        raise RuntimeError(f"Missing zoom window metadata for {image_role}")
    local_norm_y, local_norm_x = _agents._extract_normalized_point(
        local_value, field_name=local_field
    )
    left, top, right, bottom = window_px
    crop_width = max(1, right - left)
    crop_height = max(1, bottom - top)
    global_px_x = left + (local_norm_x * max(crop_width - 1, 1) / 1000.0)
    global_px_y = top + (local_norm_y * max(crop_height - 1, 1) / 1000.0)
    return _pixel_xy_to_normalized_yx(global_px_x, global_px_y, image_size=image_size)


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


def _build_tool_loop_action_schema() -> dict[str, object]:
    point_schema: dict[str, object] = {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    }
    return {
        "oneOf": [
            {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool_name": {"const": "view_overview"},
                    "tool_args": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
                "required": ["tool_name", "tool_args"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool_name": {"const": "view_zoom_pair"},
                    "tool_args": {
                        "type": "object",
                        "properties": {
                            "zoom": {"type": "number"},
                            "atlas_center_2d": point_schema,
                            "slice_center_2d": point_schema,
                        },
                        "required": ["zoom", "atlas_center_2d", "slice_center_2d"],
                        "additionalProperties": False,
                    },
                },
                "required": ["tool_name", "tool_args"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool_name": {"const": "place_point_pair"},
                    "tool_args": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "atlas_point_2d": point_schema,
                            "slice_point_2d": point_schema,
                            "atlas_point_2d_local": point_schema,
                            "slice_point_2d_local": point_schema,
                            "feature_description": {"type": "string"},
                            "artifact_note": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["found", "uncertain", "not_visible"],
                            },
                        },
                        "required": ["label"],
                        "anyOf": [
                            {"required": ["atlas_point_2d", "slice_point_2d"]},
                            {
                                "required": [
                                    "atlas_point_2d_local",
                                    "slice_point_2d_local",
                                ]
                            },
                        ],
                        "additionalProperties": False,
                    },
                },
                "required": ["tool_name", "tool_args"],
                "additionalProperties": False,
            },
            {
                "type": "object",
                "properties": {
                    "thought": {"type": "string"},
                    "tool_name": {"const": "finish"},
                    "tool_args": {
                        "type": "object",
                        "properties": {},
                        "additionalProperties": False,
                    },
                },
                "required": ["tool_name", "tool_args"],
                "additionalProperties": False,
            },
        ]
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
        "Suggested process: define the local feature, place the point pair, "
        "inspect locally at 3x, inspect more broadly at 1.5x, then lock it.\n"
        "You may choose which validation steps to repeat.\n"
        "Available tools: view_overview, view_zoom_pair, place_point_pair, finish.\n"
        "Use [y, x] integers in the 0-1000 range.\n"
        "When you place points after a zoom view, prefer atlas_point_2d_local and "
        "slice_point_2d_local so coordinates are relative to the returned zoomed images. "
        "The host will map them back to the full image.\n"
        "If you are placing points from the full overview instead, use atlas_point_2d and "
        "slice_point_2d.\n"
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
    session.metadata.pop("last_zoom_pair", None)
    return [
        {"text": summary_label},
        _agents._image_to_inline_data(
            render_landmark_annotations(atlas_image, session.atlas_annotations)
        ),
        _agents._image_to_inline_data(
            render_landmark_annotations(slice_image, session.slice_annotations)
        ),
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
    _agents._emit_trace(
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
        _agents._emit_trace(
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
        zoom = _agents._to_float(tool_args.get("zoom", 3.0))
        atlas_center = _agents._extract_normalized_point(
            tool_args.get("atlas_center_2d"), field_name="atlas_center_2d"
        )
        slice_center = _agents._extract_normalized_point(
            tool_args.get("slice_center_2d"), field_name="slice_center_2d"
        )
        atlas_zoom, atlas_window_px = _crop_zoom_view(
            render_landmark_annotations(atlas_image, session.atlas_annotations),
            center_yx=atlas_center,
            zoom=zoom,
        )
        slice_zoom, slice_window_px = _crop_zoom_view(
            render_landmark_annotations(slice_image, session.slice_annotations),
            center_yx=slice_center,
            zoom=zoom,
        )
        session.metadata["last_zoom_pair"] = {
            "atlas_window_px": atlas_window_px,
            "slice_window_px": slice_window_px,
            "zoom": zoom,
        }
        parts = [
            {
                "text": (
                    f"Zoom pair generated at {zoom:.1f}x "
                    "around the requested atlas and slice centers. Use *_point_2d_local "
                    "to place points relative to these zoomed views."
                )
            },
            _agents._image_to_inline_data(atlas_zoom),
            _agents._image_to_inline_data(slice_zoom),
            {
                "text": json.dumps(
                    {
                        "zoom": zoom,
                        "atlas_center_2d": atlas_center,
                        "slice_center_2d": slice_center,
                        "atlas_window_2d": _window_to_normalized_bounds(
                            atlas_window_px, image_size=atlas_image.size
                        ),
                        "slice_window_2d": _window_to_normalized_bounds(
                            slice_window_px, image_size=slice_image.size
                        ),
                    },
                    indent=2,
                )
            },
        ]
        _agents._emit_trace(
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
        atlas_norm = _resolve_action_point(
            tool_args,
            image_role="atlas",
            image_size=atlas_image.size,
            session=session,
        )
        slice_norm = _resolve_action_point(
            tool_args,
            image_role="slice",
            image_size=slice_image.size,
            session=session,
        )
        feature_description = str(tool_args.get("feature_description", "")).strip()
        artifact_note = str(tool_args.get("artifact_note", "")).strip()
        status = str(tool_args.get("status", "found")).strip() or "found"
        _upsert_annotation(
            session.atlas_annotations,
            LandmarkAnnotation(
                image_role="atlas",
                pixel_xy=_agents._normalized_to_pixel_xy(
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
                pixel_xy=_agents._normalized_to_pixel_xy(
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
        _agents._emit_trace(
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
            _agents._emit_trace(
                on_trace,
                tool_result_event(
                    stage="registration",
                    tool_name=tool_name,
                    summary="Finish rejected because not enough points are placed",
                    metadata={"iteration": iteration, "confirmed_count": current_count},
                ),
            )
            return False, parts
        _agents._emit_trace(
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
    prepared: _agents._PreparedRegistrationInputs,
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
                _agents._image_to_inline_data(prepared.atlas_prep.image),
                {"text": "Initial histology overview with no points placed yet."},
                _agents._image_to_inline_data(prepared.slice_prep.image),
                {"text": json.dumps(_session_summary(session), indent=2)},
            ],
        }
    ]
    config: dict[str, object] = {
        "temperature": prepared.temperature,
        "thinking_config": {"thinking_level": prepared.thinking_level},
        "response_mime_type": "application/json",
        "response_json_schema": _build_tool_loop_action_schema(),
    }

    for iteration in range(1, prepared.tool_loop_max_steps + 1):
        if on_progress:
            on_progress(
                f"Registration tool loop: step {iteration}/{prepared.tool_loop_max_steps}..."
            )
        response = _agents._retry_generate(
            client,
            model=prepared.model_name,
            contents=contents,
            config=config,
            request_label=f"Registration tool-loop step {iteration}",
            on_progress=on_progress,
        )
        action = _agents._extract_json_dict(response)
        if not action:
            raise RuntimeError("Tool-loop registration returned no action JSON")
        _agents._emit_trace(
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
                    f"Tool-loop finished with {len(entries)} point pairs; "
                    f"expected {prepared.target_count}"
                )
            return entries

    raise RuntimeError("Tool-loop registration exceeded the maximum number of steps")
