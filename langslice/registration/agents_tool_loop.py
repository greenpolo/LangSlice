"""Multimodal tool-loop registration workflow using native Gemini function calling.

Uses ``types.FunctionDeclaration`` and ``types.Tool`` so the model invokes
host-side tools through the official function-calling protocol rather than
structured JSON output.  Border/interior quotas ensure balanced coverage.
"""

from __future__ import annotations

import importlib
import io
import json
import logging
import os
from collections.abc import Callable
from typing import Any

from google.genai import types
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

# ---------------------------------------------------------------------------
# Tool declarations
# ---------------------------------------------------------------------------

_VIEW_OVERVIEW_DECL = types.FunctionDeclaration(
    name="view_overview",
    description=(
        "Returns the full atlas and slice images with all current annotations rendered."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={},
    ),
)

_VIEW_ZOOM_PAIR_DECL = types.FunctionDeclaration(
    name="view_zoom_pair",
    description=(
        "Zooms into a region of both atlas and slice for detailed inspection. "
        "Existing annotations are visible in the zoomed view."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "zoom": types.Schema(
                type="NUMBER",
                description="Zoom factor (e.g., 3.0 for 3x, 1.5 for 1.5x)",
            ),
            "atlas_center_2d": types.Schema(
                type="ARRAY",
                description="[y, x] in 0-1000 normalized range",
                items=types.Schema(type="INTEGER"),
            ),
            "slice_center_2d": types.Schema(
                type="ARRAY",
                description="[y, x] in 0-1000 normalized range",
                items=types.Schema(type="INTEGER"),
            ),
        },
        required=["zoom", "atlas_center_2d", "slice_center_2d"],
    ),
)

_PLACE_POINT_PAIR_DECL = types.FunctionDeclaration(
    name="place_point_pair",
    description=(
        "Place a matched landmark pair on atlas and slice. "
        "Re-calling with the same label updates the position."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={
            "label": types.Schema(type="STRING", description='Point label (e.g., "1", "2")'),
            "category": types.Schema(
                type="STRING",
                description='"border" or "interior"',
                enum=["border", "interior"],
            ),
            "feature_description": types.Schema(
                type="STRING",
                description=(
                    "Rich description of the anatomical feature being matched "
                    '(e.g., "the deepest point of the dorsal midline notch")'
                ),
            ),
            "atlas_point_2d": types.Schema(
                type="ARRAY",
                description="[y, x] global coordinates in 0-1000 range",
                items=types.Schema(type="INTEGER"),
            ),
            "slice_point_2d": types.Schema(
                type="ARRAY",
                description="[y, x] global coordinates in 0-1000 range",
                items=types.Schema(type="INTEGER"),
            ),
            "atlas_point_2d_local": types.Schema(
                type="ARRAY",
                description="[y, x] local coordinates relative to last zoom view",
                items=types.Schema(type="INTEGER"),
            ),
            "slice_point_2d_local": types.Schema(
                type="ARRAY",
                description="[y, x] local coordinates relative to last zoom view",
                items=types.Schema(type="INTEGER"),
            ),
            "artifact_note": types.Schema(
                type="STRING",
                description="Note about damage/artifacts at this location",
            ),
        },
        required=["label", "category", "feature_description"],
    ),
)

_FINISH_DECL = types.FunctionDeclaration(
    name="finish",
    description=(
        "Complete landmark placement. Rejected if border and interior quotas are not met."
    ),
    parameters=types.Schema(
        type="OBJECT",
        properties={},
    ),
)

_ALL_TOOL_DECLARATIONS = [
    _VIEW_OVERVIEW_DECL,
    _VIEW_ZOOM_PAIR_DECL,
    _PLACE_POINT_PAIR_DECL,
    _FINISH_DECL,
]


# ---------------------------------------------------------------------------
# Zoom / coordinate helpers (preserved from previous version)
# ---------------------------------------------------------------------------


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
    left, top, right, bottom = _compute_zoom_window(
        (width, height), center_yx=center_yx, zoom=zoom
    )
    cropped = image.crop((left, top, right, bottom))
    # Upscale the crop to a fixed max size for readability, not to the
    # full image dimensions (which produces unnecessarily huge images).
    _ZOOM_MAX_EDGE = 512
    cw, ch = cropped.size
    scale = _ZOOM_MAX_EDGE / max(cw, ch)
    if scale > 1.0:
        cropped = cropped.resize(
            (int(cw * scale), int(ch * scale)), Image.Resampling.BICUBIC
        )
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


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------


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
    placed_border = 0
    placed_interior = 0
    for ann in session.atlas_annotations:
        if ann.category == "interior":
            placed_interior += 1
        else:
            placed_border += 1

    return {
        "workflow": session.workflow,
        "target_count": session.target_count,
        "border_count": session.border_count,
        "interior_count": session.interior_count,
        "placed_border": placed_border,
        "placed_interior": placed_interior,
        "placed_labels": [ann.label for ann in session.atlas_annotations],
        "atlas_labels": [ann.label for ann in session.atlas_annotations],
        "slice_labels": [ann.label for ann in session.slice_annotations],
    }


def _placed_tool_loop_entries(
    session: RegistrationAnnotationSession,
) -> list[dict[str, object]]:
    """Return entries for labels that have both atlas and slice annotations."""
    atlas_by_label = {ann.label: ann for ann in session.atlas_annotations}
    slice_by_label = {ann.label: ann for ann in session.slice_annotations}

    def _label_sort_key(value: str) -> tuple[int, int | str]:
        stripped = str(value).strip()
        return (0, int(stripped)) if stripped.isdigit() else (1, stripped)

    shared_labels = sorted(
        {label for label in atlas_by_label if label in slice_by_label},
        key=_label_sort_key,
    )
    entries: list[dict[str, object]] = []
    for label in shared_labels:
        atlas_ann = atlas_by_label[label]
        slice_ann = slice_by_label[label]
        entries.append(
            {
                "label": label,
                "status": "found",
                "atlas_point_2d": list(atlas_ann.normalized_yx or (0.0, 0.0)),
                "slice_point_2d": list(slice_ann.normalized_yx or (0.0, 0.0)),
                "feature_description": slice_ann.feature_description,
                "artifact_note": slice_ann.artifact_note,
            }
        )
    return entries


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


def _build_tool_loop_prompt(
    *,
    atlas_name: str,
    position_mm: float,
    border_count: int,
    interior_count: int,
) -> str:
    return (
        "You are an expert neuroanatomist. You are placing matched landmark points\n"
        "between a brain atlas image (left) and a histology slice image (right).\n"
        "\n"
        f"Atlas: {atlas_name}\n"
        f"AP position: {position_mm:.3f} mm\n"
        "\n"
        "COORDINATE SYSTEM: All point coordinates use [y, x] format normalized to\n"
        "0-1000, where [0, 0] is the top-left corner and [1000, 1000] is the\n"
        "bottom-right corner. These coordinates are relative to EACH IMAGE\n"
        "INDEPENDENTLY (atlas and slice have their own coordinate spaces).\n"
        "\n"
        f"TASK: Place {border_count} landmark pairs on the outermost edge/border\n"
        f"of the brain, then {interior_count} landmark pairs in the interior.\n"
        "\n"
        "STRATEGY:\n"
        "1. Start by calling view_overview to see both images side by side.\n"
        "2. Place your border points first -- choose anatomically distinct features\n"
        "   on the brain outline that are clearly identifiable in both images.\n"
        "   Good border landmarks: midline notches (dorsal/ventral), points of\n"
        "   maximum lateral curvature, hemisphere tips.\n"
        "3. Then place interior points -- use internal structures visible in both\n"
        "   images like ventricle tips, commissure boundaries, or distinct\n"
        "   tissue boundaries.\n"
        "4. Use view_zoom_pair to verify any placements you are uncertain about.\n"
        "   You can re-place a point by calling place_point_pair with the same\n"
        "   label to update its position.\n"
        "5. When all points are placed, call finish.\n"
        "\n"
        "IMPORTANT:\n"
        "- Prioritize local anatomical correspondence over global position.\n"
        '- Use rich feature descriptions (e.g., "deepest point of dorsal midline\n'
        '  notch" not "top of brain").\n'
        "- If a region is damaged/torn in the slice, note it in artifact_note\n"
        "  and choose a nearby intact feature instead.\n"
        "- Do NOT place points on the black background.\n"
        "- Do NOT assume left-right symmetry -- hemispheres may differ.\n"
        "- CRITICAL: Place border points only on the MAIN continuous brain\n"
        "  outline. Histology slices often have detached tissue fragments,\n"
        "  debris, or separate tissue pieces visible around the main section.\n"
        "  Ignore these — only use the largest connected brain section.\n"
        "- For border points, match the SAME anatomical curvature feature\n"
        "  in both atlas and slice. The atlas shows the idealized shape;\n"
        "  find where that same curve appears on the slice.\n"
        "- VENTRICLE WARNING: Lateral ventricles are often deformed, collapsed,\n"
        "  or expanded during histological slicing. Do NOT place points in\n"
        "  the open lumen/void. Instead, place points on the ventricle WALLS\n"
        "  or EDGES — match a specific wall segment or corner in both images\n"
        "  (e.g., 'the medial wall of the left ventricle where it meets the\n"
        "  septum'). Match wall-to-wall, edge-to-edge."
    )


# ---------------------------------------------------------------------------
# Image preparation (File API vs inline)
# ---------------------------------------------------------------------------


def _side_by_side(
    left: Image.Image,
    right: Image.Image,
    *,
    label_left: str = "Atlas",
    label_right: str = "Slice",
    gap: int = 8,
    max_height: int = 768,
) -> Image.Image:
    """Create a side-by-side composite of two images, scaled to match heights."""
    # Scale both to the same height
    lw, lh = left.size
    rw, rh = right.size
    target_h = min(max_height, max(lh, rh))
    left_scaled = left.resize(
        (int(lw * target_h / lh), target_h), Image.Resampling.LANCZOS
    )
    right_scaled = right.resize(
        (int(rw * target_h / rh), target_h), Image.Resampling.LANCZOS
    )
    sw, _ = left_scaled.size
    rw2, _ = right_scaled.size
    total_w = sw + gap + rw2
    # Black background with white divider
    composite = Image.new("RGB", (total_w, target_h), (0, 0, 0))
    composite.paste(left_scaled.convert("RGB"), (0, 0))
    composite.paste(right_scaled.convert("RGB"), (sw + gap, 0))
    return composite


def _image_to_bytes(img: Image.Image, *, fmt: str = "PNG", quality: int = 85) -> bytes:
    """Convert a PIL image to bytes in the given format."""
    buf = io.BytesIO()
    prepared = img.convert("RGB") if img.mode != "RGB" else img
    if fmt.upper() == "JPEG":
        prepared.save(buf, format="JPEG", quality=quality)
    else:
        prepared.save(buf, format="PNG")
    return buf.getvalue()


def _image_to_part(
    img: Image.Image, *, fmt: str = "PNG", quality: int = 85
) -> types.Part:
    """Convert a PIL image to a ``types.Part``."""
    mime = "image/jpeg" if fmt.upper() == "JPEG" else "image/png"
    return types.Part.from_bytes(data=_image_to_bytes(img, fmt=fmt, quality=quality), mime_type=mime)


def _prepare_base_images(
    client: Any,
    prepared: _agents._PreparedRegistrationInputs,
    *,
    atlas_override: Image.Image | None = None,
    slice_override: Image.Image | None = None,
) -> tuple[types.Part | Any, types.Part | Any, list[Any]]:
    """Upload or inline the base atlas and slice images.

    Returns (atlas_part, slice_part, uploaded_files).
    *uploaded_files* is non-empty only when File API was used.
    """
    _ai_config = importlib.import_module("langslice.ai.config")
    atlas_image = atlas_override or prepared.atlas_prep.image
    slice_image = slice_override or prepared.slice_prep.image

    uploaded_files: list[Any] = []
    if _ai_config.supports_file_api():
        atlas_bytes_io = io.BytesIO(_image_to_bytes(atlas_image))
        slice_bytes_io = io.BytesIO(_image_to_bytes(slice_image, fmt="JPEG"))
        atlas_file = client.files.upload(
            file=atlas_bytes_io,
            config=types.UploadFileConfig(mime_type="image/png"),
        )
        slice_file = client.files.upload(
            file=slice_bytes_io,
            config=types.UploadFileConfig(mime_type="image/jpeg"),
        )
        uploaded_files = [atlas_file, slice_file]
        atlas_part = types.Part.from_uri(file_uri=atlas_file.uri, mime_type="image/png")
        slice_part = types.Part.from_uri(file_uri=slice_file.uri, mime_type="image/jpeg")
        return atlas_part, slice_part, uploaded_files
    else:
        atlas_part = _image_to_part(atlas_image)
        slice_part = _image_to_part(slice_image, fmt="JPEG")
        return atlas_part, slice_part, uploaded_files


# ---------------------------------------------------------------------------
# Token tracking
# ---------------------------------------------------------------------------


def _accumulate_usage(accumulated: dict[str, int], usage_metadata: Any) -> None:
    if usage_metadata is None:
        return
    for field in (
        "prompt_token_count",
        "candidates_token_count",
        "total_token_count",
        "cached_content_token_count",
        "thoughts_token_count",
        "tool_use_prompt_token_count",
    ):
        value = getattr(usage_metadata, field, None)
        if value is not None:
            accumulated[field] = accumulated.get(field, 0) + int(value)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _execute_tool(
    fc: Any,
    *,
    session: RegistrationAnnotationSession,
    atlas_image: Image.Image,
    slice_image: Image.Image,
    iteration: int,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[dict[str, Any], list[types.Part], bool]:
    """Execute a single function call and return (result_dict, image_parts, is_finished)."""
    tool_name = fc.name
    tool_args: dict[str, Any] = dict(fc.args) if fc.args else {}
    _agents._emit_trace(
        on_trace,
        tool_call_event(
            stage="registration",
            tool_name=tool_name,
            args=tool_args,
            iteration=iteration,
        ),
    )

    if tool_name == "view_overview":
        return _handle_view_overview(
            session=session,
            atlas_image=atlas_image,
            slice_image=slice_image,
            iteration=iteration,
            on_trace=on_trace,
        )

    if tool_name == "view_zoom_pair":
        return _handle_view_zoom_pair(
            tool_args=tool_args,
            session=session,
            atlas_image=atlas_image,
            slice_image=slice_image,
            iteration=iteration,
            on_trace=on_trace,
        )

    if tool_name == "place_point_pair":
        return _handle_place_point_pair(
            tool_args=tool_args,
            session=session,
            atlas_image=atlas_image,
            slice_image=slice_image,
            iteration=iteration,
            on_trace=on_trace,
        )

    if tool_name == "finish":
        return _handle_finish(
            session=session,
            iteration=iteration,
            on_trace=on_trace,
        )

    raise RuntimeError(f"Unknown tool-loop action: {tool_name}")


def _handle_view_overview(
    *,
    session: RegistrationAnnotationSession,
    atlas_image: Image.Image,
    slice_image: Image.Image,
    iteration: int,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[dict[str, Any], list[types.Part], bool]:
    session.metadata.pop("last_zoom_pair", None)
    ref_size = max(slice_image.size)
    atlas_annotated = render_landmark_annotations(atlas_image, session.atlas_annotations, reference_size=ref_size)
    slice_annotated = render_landmark_annotations(slice_image, session.slice_annotations, reference_size=ref_size)
    composite = _side_by_side(atlas_annotated, slice_annotated)
    result_dict = {"status": "ok"}
    image_parts: list[types.Part] = [
        types.Part.from_text(
            text="Side-by-side overview: Atlas (left) | Slice (right). "
            "Current annotations shown."
        ),
        _image_to_part(composite, fmt="JPEG"),
        types.Part.from_text(text=json.dumps(_session_summary(session), indent=2)),
    ]
    _agents._emit_trace(
        on_trace,
        tool_result_event(
            stage="registration",
            tool_name="view_overview",
            summary="Returned overview images with current annotations",
            parts=[
                image_part_from_pil(atlas_annotated, label="Atlas overview"),
                image_part_from_pil(slice_annotated, label="Slice overview"),
            ],
            metadata={"iteration": iteration},
        ),
    )
    return result_dict, image_parts, False


def _handle_view_zoom_pair(
    *,
    tool_args: dict[str, Any],
    session: RegistrationAnnotationSession,
    atlas_image: Image.Image,
    slice_image: Image.Image,
    iteration: int,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[dict[str, Any], list[types.Part], bool]:
    zoom = _agents._to_float(tool_args.get("zoom", 3.0))
    atlas_center = _agents._extract_normalized_point(
        tool_args.get("atlas_center_2d"), field_name="atlas_center_2d"
    )
    slice_center = _agents._extract_normalized_point(
        tool_args.get("slice_center_2d"), field_name="slice_center_2d"
    )
    ref_size = max(slice_image.size)
    atlas_zoom, atlas_window_px = _crop_zoom_view(
        render_landmark_annotations(atlas_image, session.atlas_annotations, reference_size=ref_size),
        center_yx=atlas_center,
        zoom=zoom,
    )
    slice_zoom, slice_window_px = _crop_zoom_view(
        render_landmark_annotations(slice_image, session.slice_annotations, reference_size=ref_size),
        center_yx=slice_center,
        zoom=zoom,
    )
    session.metadata["last_zoom_pair"] = {
        "atlas_window_px": atlas_window_px,
        "slice_window_px": slice_window_px,
        "zoom": zoom,
    }

    result_dict: dict[str, Any] = {
        "status": "ok",
        "zoom": zoom,
        "atlas_window": _window_to_normalized_bounds(atlas_window_px, image_size=atlas_image.size),
        "slice_window": _window_to_normalized_bounds(slice_window_px, image_size=slice_image.size),
    }

    zoom_composite = _side_by_side(atlas_zoom, slice_zoom, max_height=512)
    image_parts: list[types.Part] = [
        types.Part.from_text(
            text=f"Side-by-side zoom at {zoom:.1f}x: Atlas (left) | Slice (right). "
            "Use *_point_2d_local to place points relative to these zoomed views."
        ),
        _image_to_part(zoom_composite, fmt="JPEG"),
    ]
    _agents._emit_trace(
        on_trace,
        tool_result_event(
            stage="registration",
            tool_name="view_zoom_pair",
            summary=f"Returned zoomed atlas/slice views at {zoom:.1f}x",
            metadata={"iteration": iteration, "zoom": zoom},
        ),
    )
    return result_dict, image_parts, False


def _handle_place_point_pair(
    *,
    tool_args: dict[str, Any],
    session: RegistrationAnnotationSession,
    atlas_image: Image.Image,
    slice_image: Image.Image,
    iteration: int,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[dict[str, Any], list[types.Part], bool]:
    label = str(tool_args.get("label", "")).strip()
    category = str(tool_args.get("category", "border")).strip() or "border"
    feature_description = str(tool_args.get("feature_description", "")).strip()
    artifact_note = str(tool_args.get("artifact_note", "")).strip()

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

    _upsert_annotation(
        session.atlas_annotations,
        LandmarkAnnotation(
            image_role="atlas",
            pixel_xy=_agents._normalized_to_pixel_xy(
                atlas_norm[0], atlas_norm[1], image_size=atlas_image.size
            ),
            label=label,
            normalized_yx=atlas_norm,
            status="found",
            feature_description=feature_description,
            artifact_note=artifact_note,
            category=category,
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
            status="found",
            feature_description=feature_description,
            artifact_note=artifact_note,
            category=category,
        ),
    )

    ref_size = max(slice_image.size)
    atlas_annotated = render_landmark_annotations(atlas_image, session.atlas_annotations, reference_size=ref_size)
    slice_annotated = render_landmark_annotations(slice_image, session.slice_annotations, reference_size=ref_size)
    composite = _side_by_side(atlas_annotated, slice_annotated)

    result_dict: dict[str, Any] = {
        "status": "ok",
        "label": label,
        "category": category,
        "message": f"Point {label} placed ({category}).",
    }
    image_parts: list[types.Part] = [
        types.Part.from_text(
            text=f"Saved point pair {label} ({category}). "
            "Side-by-side: Atlas (left) | Slice (right)."
        ),
        _image_to_part(composite, fmt="JPEG"),
        types.Part.from_text(text=json.dumps(_session_summary(session), indent=2)),
    ]
    _agents._emit_trace(
        on_trace,
        tool_result_event(
            stage="registration",
            tool_name="place_point_pair",
            summary=f"Saved point pair {label} ({category})",
            metadata={"iteration": iteration, "label": label, "category": category},
        ),
    )
    return result_dict, image_parts, False


def _handle_finish(
    *,
    session: RegistrationAnnotationSession,
    iteration: int,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> tuple[dict[str, Any], list[types.Part], bool]:
    border_count_target = session.border_count or 0
    interior_count_target = session.interior_count or 0

    # Count placed border/interior (points with both atlas and slice annotations)
    slice_labels = {ann.label for ann in session.slice_annotations}
    placed_border = 0
    placed_interior = 0
    for ann in session.atlas_annotations:
        if ann.label in slice_labels:
            if ann.category == "interior":
                placed_interior += 1
            else:
                placed_border += 1

    if placed_border < border_count_target or placed_interior < interior_count_target:
        message = (
            f"Cannot finish yet. "
            f"Border: {placed_border}/{border_count_target} placed. "
            f"Interior: {placed_interior}/{interior_count_target} placed."
        )
        result_dict: dict[str, Any] = {
            "status": "error",
            "message": message,
            "summary": json.loads(json.dumps(_session_summary(session))),
        }
        _agents._emit_trace(
            on_trace,
            tool_result_event(
                stage="registration",
                tool_name="finish",
                summary="Finish rejected because quotas not met",
                metadata={
                    "iteration": iteration,
                    "placed_border": placed_border,
                    "placed_interior": placed_interior,
                },
            ),
        )
        return result_dict, [], False

    total_placed = placed_border + placed_interior
    _agents._emit_trace(
        on_trace,
        tool_result_event(
            stage="registration",
            tool_name="finish",
            summary=f"Finish accepted with {total_placed} placed pairs",
            metadata={
                "iteration": iteration,
                "placed_border": placed_border,
                "placed_interior": placed_interior,
            },
        ),
    )
    result_dict = {
        "status": "ok",
        "message": f"Finished with {total_placed} point pairs.",
    }
    return result_dict, [], True


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def _estimate_correspondences_tool_loop(
    client: Any,
    *,
    prepared: _agents._PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    border_count: int | None = None,
    interior_count: int | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
    # Resolve border/interior counts from target_count if not specified
    if border_count is None and interior_count is None:
        total = prepared.target_count
        border_count = (total + 1) // 2  # Extra point goes to border
        interior_count = total - border_count
    elif border_count is None:
        border_count = max(0, prepared.target_count - (interior_count or 0))
    elif interior_count is None:
        interior_count = max(0, prepared.target_count - border_count)

    session = RegistrationAnnotationSession(
        workflow="multimodal_tool_loop",
        target_count=prepared.target_count,
        border_count=border_count,
        interior_count=interior_count,
        metadata={
            "atlas_name": atlas_name,
            "position_mm": round(position_mm, 3),
        },
    )

    # Boost slice exposure by 50% for better visibility of structures.
    from PIL import ImageEnhance

    slice_for_model = ImageEnhance.Brightness(prepared.slice_prep.image).enhance(1.5)

    # Upscale atlas for better visual comparison. The raw atlas is tiny
    # (e.g. 456x320) which makes landmark matching hard. Upscale to ~1K
    # on the long edge — large enough for detail, small enough for inline
    # image payloads in tool responses.
    _ATLAS_TARGET_LONG_EDGE = 1024
    atlas_orig = prepared.atlas_prep.image
    aw, ah = atlas_orig.size
    atlas_scale = _ATLAS_TARGET_LONG_EDGE / max(aw, ah)
    if atlas_scale > 1.0:
        atlas_for_model = atlas_orig.resize(
            (int(aw * atlas_scale), int(ah * atlas_scale)),
            Image.Resampling.LANCZOS,
        )
    else:
        atlas_for_model = atlas_orig

    # Prepare base image parts
    atlas_part, slice_part, uploaded_files = _prepare_base_images(
        client, prepared, atlas_override=atlas_for_model, slice_override=slice_for_model
    )

    # Build tools
    tools = types.Tool(function_declarations=_ALL_TOOL_DECLARATIONS)

    # Build config
    config = types.GenerateContentConfig(
        tools=[tools],
        thinking_config=types.ThinkingConfig(
            thinking_level=prepared.thinking_level,  # type: ignore[arg-type]
        ),
        temperature=prepared.temperature,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )

    # Build prompt
    prompt_text = _build_tool_loop_prompt(
        atlas_name=atlas_name,
        position_mm=position_mm,
        border_count=border_count,
        interior_count=interior_count,
    )

    # Initial history
    history: list[types.Content] = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=prompt_text),
                types.Part.from_text(text="Atlas reference image:"),
                atlas_part,
                types.Part.from_text(text="Histology slice image:"),
                slice_part,
                types.Part.from_text(text=json.dumps(_session_summary(session), indent=2)),
            ],
        )
    ]

    accumulated_usage: dict[str, int] = {}

    try:
        for iteration in range(1, prepared.tool_loop_max_steps + 1):
            if on_progress:
                on_progress(
                    f"Registration tool loop: step {iteration}/{prepared.tool_loop_max_steps}..."
                )

            response = _agents._retry_generate(
                client,
                model=prepared.model_name,
                contents=history,
                config=config,
                request_label=f"Registration tool-loop step {iteration}",
                on_progress=on_progress,
            )

            # Track tokens
            _accumulate_usage(accumulated_usage, getattr(response, "usage_metadata", None))

            # Emit model trace event
            model_content = response.candidates[0].content
            _agents._emit_trace(
                on_trace,
                model_event(
                    stage="registration",
                    title="Tool-loop model response",
                    summary=f"Iteration {iteration}: {len(model_content.parts)} parts",
                    parts=[json_part({"iteration": iteration}, label="Model response metadata")],
                    metadata={"workflow": "multimodal_tool_loop", "iteration": iteration},
                ),
            )

            # Add model response to history
            history.append(model_content)

            # Process function calls
            finished = False
            function_response_parts: list[types.Part] = []
            image_parts: list[types.Part] = []

            # Debug: log what model returned (including thoughts)
            fc_names = [p.function_call.name for p in model_content.parts if p.function_call]
            if on_progress:
                on_progress(f"  -> Model returned: {len(model_content.parts)} parts, functions={fc_names}")
                for idx, p in enumerate(model_content.parts):
                    p_type = "unknown"
                    if p.function_call:
                        p_type = f"function_call({p.function_call.name})"
                    elif getattr(p, "thought", False) and getattr(p, "text", None):
                        p_type = f"thought({len(p.text)} chars)"
                        on_progress(f"  -> Thought: {p.text[:200].replace(chr(10), ' ')}")
                    elif getattr(p, "text", None):
                        p_type = f"text({len(p.text)} chars)"
                        on_progress(f"  -> Text: {p.text[:120]}")
                    elif getattr(p, "thought_signature", None):
                        p_type = "thought_signature"
                    else:
                        # Dump all non-None attributes to understand unknown parts
                        attrs = {k: type(v).__name__ for k, v in vars(p).items() if v is not None and not k.startswith("_")}
                        p_type = f"unknown(attrs={attrs})"
                    on_progress(f"  -> Part[{idx}]: {p_type}")

            for part in model_content.parts:
                if not part.function_call:
                    continue

                fc = part.function_call
                if on_progress:
                    on_progress(f"  -> FC: {fc.name}({fc.args})")
                result_dict, result_images, is_finished = _execute_tool(
                    fc,
                    session=session,
                    atlas_image=atlas_for_model,
                    slice_image=slice_for_model,
                    iteration=iteration,
                    on_trace=on_trace,
                )

                # Save debug images per tool call if debug_dir is set
                if prepared.registration_dir and result_images:
                    step_dir = os.path.join(
                        prepared.registration_dir,
                        f"step_{iteration:02d}_{fc.name}",
                    )
                    os.makedirs(step_dir, exist_ok=True)
                    img_idx = 0
                    for rp in result_images:
                        inline = getattr(rp, "inline_data", None)
                        if inline and getattr(inline, "data", None):
                            img = Image.open(io.BytesIO(inline.data))
                            img.save(
                                os.path.join(step_dir, f"image_{img_idx}.png"),
                                format="PNG",
                            )
                            img_idx += 1
                        text_val = getattr(rp, "text", None)
                        if text_val:
                            with open(
                                os.path.join(step_dir, "context.txt"), "a", encoding="utf-8"
                            ) as fh:
                                fh.write(text_val + "\n")

                function_response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response=result_dict,
                            id=fc.id,
                        )
                    )
                )
                image_parts.extend(result_images)

                if is_finished:
                    finished = True

            # Send function responses + images in a single role="user" message.
            all_response_parts = function_response_parts + image_parts
            if all_response_parts:
                history.append(
                    types.Content(role="user", parts=all_response_parts)
                )
            else:
                # Model returned text/thought but no function calls. We must send
                # a user turn to avoid consecutive model turns, which stalls the loop.
                placed_count = len(session.atlas_annotations)
                total_needed = (border_count or 0) + (interior_count or 0)
                history.append(
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(
                            text=f"Continue placing landmarks. You have "
                            f"{placed_count}/{total_needed} placed points. "
                            f"Use your tools to place the next point."
                        )],
                    )
                )

            if finished:
                entries = _placed_tool_loop_entries(session)
                logger.info("Tool loop token usage: %s", accumulated_usage)
                session.metadata["token_usage"] = accumulated_usage
                return entries

        raise RuntimeError("Tool-loop registration exceeded the maximum number of steps")

    finally:
        # Clean up File API uploads
        for f in uploaded_files:
            try:
                client.files.delete(name=f.name)
            except Exception:
                logger.warning("Failed to delete uploaded file %s", getattr(f, "name", "unknown"))
        # Always log token usage
        logger.info("Tool loop token usage: %s", accumulated_usage)
        session.metadata["token_usage"] = accumulated_usage
