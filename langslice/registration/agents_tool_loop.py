"""Multimodal tool-loop registration workflow using the Gemini Interactions API.

STATUS: Experimental / on hold (2026-03-22).  Image-gen workflow is the
default for landmark placement.  This tool-loop approach is retained for
future model improvements but is NOT production-quality yet.

ARCHITECTURE DECISIONS (preserved for future iteration):
- Uses the Interactions API (``client.interactions.create``) with
  ``previous_interaction_id`` for server-side conversation state.  This
  eliminates the image-accumulation problem where 30+ images pile up in
  context — the server holds history, we only send the latest tool result.
- Atlas and slice are returned as SEPARATE images (not composites).
  Composites confuse coordinate spaces; separate images + rich text
  descriptions of features work better as verbal anchors.
- Review gate (two-pass finish) was removed — the model gets stuck in
  infinite verify loops when it can't judge its own placement quality.
- Zoom persistence: after placing a point, the zoomed view (if active) is
  returned so the model can verify without re-zooming.

KNOWN LIMITATIONS (as of gemini-3-flash-preview / gemini-3.1-pro-preview):
- Models use numerical/textual heuristics for coordinates rather than true
  visual grounding.  "Dorsal midline notch" → ~[130, 500] every time.
- Geometric triangulation: points form perfect symmetric patterns instead
  of matching actual anatomy.
- Coordinate scrubbing from history doesn't work: model leaks coords into
  free text, and scrubbing causes degenerate re-placement loops.
- Pro model (3.1-pro) is not better than Flash for spatial tasks — it's
  more methodical but equally inaccurate.
- The Interactions API occasionally returns ``status: incomplete`` after
  ~300s on complex turns (server-side timeout).

WHAT WORKS for future reference:
- Interactions API eliminates payload bloat (genuine infrastructure win).
- Separate images + rich text feature descriptions produce better feature
  targeting than composites.
- Place-then-verify prompt ordering gets the model to zoom after placing.
- Temperature 1.0, LOW thinking level, JPEG compression, atlas upscaled
  to 1K, slice exposure boost 1.5x.
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
        "between a brain atlas image and a histology slice image.\n"
        "These points will be used for REGISTRATION — aligning the slice to the\n"
        "atlas. Accurate, well-distributed points across the brain are essential\n"
        "for a good registration result.\n"
        "\n"
        "The images are shown SEPARATELY — atlas and slice are independent images\n"
        "with their own coordinate spaces.\n"
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
        "1. Start by calling view_overview to see both images.\n"
        "2. Place your border points first. For each point:\n"
        "   a. Identify a VISUALLY DISTINCT feature on the brain outline —\n"
        "      a sharp notch, a corner, a distinct curvature change.\n"
        "   b. Write a RICH TEXT DESCRIPTION in feature_description. Be very\n"
        "      specific (e.g., 'the sharp inward notch on the left lateral\n"
        "      cortical surface where the rhinal fissure creates a visible\n"
        "      indentation'). This is your anchor for matching.\n"
        "   c. Place the point pair (atlas and slice coordinates).\n"
        "   d. THEN zoom in (view_zoom_pair) to VERIFY your placement visually.\n"
        "      Check that the annotation marker sits exactly on the feature you\n"
        "      described. If it's off, re-place the point with corrected coords.\n"
        "   Good border landmarks: midline notches (dorsal/ventral), rhinal\n"
        "   fissure notches, points of maximum lateral curvature.\n"
        "3. Then place interior points the same way — place, then zoom to verify.\n"
        "   Good interior landmarks: ventricle wall corners, commissure\n"
        "   boundaries, distinct tissue boundary junctions.\n"
        "4. VERIFY every point by zooming AFTER placing it. If the marker is not\n"
        "   on the intended feature, call place_point_pair again to fix it.\n"
        "5. AVOID featureless regions. Do NOT place points in smooth, uniform\n"
        "   areas (e.g., middle of the caudate putamen, smooth cortical surface).\n"
        "   Only choose locations with a distinct visual landmark you can describe\n"
        "   in text and re-identify in the other image.\n"
        "6. When all points are placed and verified, call finish.\n"
        "\n"
        "IMPORTANT:\n"
        "- The atlas and slice are SEPARATE images. You will see them labeled\n"
        "  individually. Use your text description to bridge between them.\n"
        "- Prioritize local anatomical correspondence over global position.\n"
        "- If a region is damaged/torn in the slice, note it in artifact_note\n"
        "  and choose a nearby intact feature instead.\n"
        "- Do NOT place points on the black background.\n"
        "- Do NOT assume left-right symmetry -- hemispheres may differ.\n"
        "- CRITICAL: Place border points only on the MAIN continuous brain\n"
        "  outline. Histology slices often have detached tissue fragments,\n"
        "  debris, or separate tissue pieces visible around the main section.\n"
        "  Ignore these — only use the largest connected brain section.\n"
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
            atlas_image=atlas_image,
            slice_image=slice_image,
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
    result_dict = {"status": "ok"}
    image_parts: list[types.Part] = [
        types.Part.from_text(text="ATLAS overview (with current annotations):"),
        _image_to_part(atlas_annotated, fmt="JPEG"),
        types.Part.from_text(text="SLICE overview (with current annotations):"),
        _image_to_part(slice_annotated, fmt="JPEG"),
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

    image_parts: list[types.Part] = [
        types.Part.from_text(
            text=f"ATLAS zoomed at {zoom:.1f}x. "
            "Use atlas_point_2d_local to place points relative to this view."
        ),
        _image_to_part(atlas_zoom, fmt="JPEG"),
        types.Part.from_text(
            text=f"SLICE zoomed at {zoom:.1f}x. "
            "Use slice_point_2d_local to place points relative to this view."
        ),
        _image_to_part(slice_zoom, fmt="JPEG"),
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

    try:
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
    except RuntimeError as exc:
        # Return error to model so it can retry with correct coordinates
        error_dict: dict[str, Any] = {
            "status": "error",
            "message": f"Missing coordinates: {exc}. "
            "You must provide BOTH atlas_point_2d and slice_point_2d "
            "(or both local variants after a zoom).",
        }
        _agents._emit_trace(
            on_trace,
            tool_result_event(
                stage="registration",
                tool_name="place_point_pair",
                summary=f"Point placement rejected: {exc}",
                metadata={"iteration": iteration, "label": label},
            ),
        )
        return error_dict, [], False

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

    # Reset finish review flag since a point was re-placed
    session.metadata.pop("finish_reviewed", None)

    ref_size = max(slice_image.size)
    atlas_annotated = render_landmark_annotations(atlas_image, session.atlas_annotations, reference_size=ref_size)
    slice_annotated = render_landmark_annotations(slice_image, session.slice_annotations, reference_size=ref_size)

    # If there's an active zoom, return the zoomed view with the new point
    # rendered so the model can verify placement at high resolution and
    # continue placing nearby points without re-zooming.
    last_zoom = session.metadata.get("last_zoom_pair")
    if isinstance(last_zoom, dict) and "atlas_window_px" in last_zoom:
        zoom_factor = last_zoom.get("zoom", 3.0)
        atlas_window = last_zoom["atlas_window_px"]
        slice_window = last_zoom["slice_window_px"]

        atlas_crop = atlas_annotated.crop(atlas_window)
        slice_crop = slice_annotated.crop(slice_window)
        _ZOOM_MAX_EDGE = 512
        acw, ach = atlas_crop.size
        a_scale = _ZOOM_MAX_EDGE / max(acw, ach)
        if a_scale > 1.0:
            atlas_crop = atlas_crop.resize(
                (int(acw * a_scale), int(ach * a_scale)), Image.Resampling.BICUBIC
            )
        scw, sch = slice_crop.size
        s_scale = _ZOOM_MAX_EDGE / max(scw, sch)
        if s_scale > 1.0:
            slice_crop = slice_crop.resize(
                (int(scw * s_scale), int(sch * s_scale)), Image.Resampling.BICUBIC
            )

        result_dict: dict[str, Any] = {
            "status": "ok",
            "label": label,
            "category": category,
            "message": f"Point {label} placed ({category}). Showing current zoom view with annotation.",
        }
        image_parts: list[types.Part] = [
            types.Part.from_text(
                text=f"Saved point pair {label} ({category}). "
                f"ATLAS zoomed at {zoom_factor:.1f}x with annotation:"
            ),
            _image_to_part(atlas_crop, fmt="JPEG"),
            types.Part.from_text(
                text=f"SLICE zoomed at {zoom_factor:.1f}x with annotation:"
            ),
            _image_to_part(slice_crop, fmt="JPEG"),
            types.Part.from_text(text=json.dumps(_session_summary(session), indent=2)),
        ]
    else:
        result_dict = {
            "status": "ok",
            "label": label,
            "category": category,
            "message": f"Point {label} placed ({category}).",
        }
        image_parts = [
            types.Part.from_text(
                text=f"Saved point pair {label} ({category}). ATLAS overview:"
            ),
            _image_to_part(atlas_annotated, fmt="JPEG"),
            types.Part.from_text(text=f"SLICE overview:"),
            _image_to_part(slice_annotated, fmt="JPEG"),
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
    atlas_image: Image.Image,
    slice_image: Image.Image,
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

    # Accept immediately (no review gate)
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
# Interactions API helpers
# ---------------------------------------------------------------------------

import base64


def _pil_to_image_content(
    img: Image.Image, *, fmt: str = "JPEG", resolution: str = "high"
) -> dict[str, Any]:
    """Convert a PIL image to an Interactions API ImageContentParam dict."""
    data = _image_to_bytes(img, fmt=fmt)
    b64 = base64.b64encode(data).decode("utf-8")
    return {
        "type": "image",
        "data": b64,
        "mime_type": f"image/{fmt.lower()}",
        "resolution": resolution,
    }


def _upload_to_file_api(
    client: Any, img: Image.Image, *, fmt: str = "PNG"
) -> tuple[str, str, Any]:
    """Upload a PIL image via the Files API and return (uri, mime_type, file_obj)."""
    mime = f"image/{fmt.lower()}"
    data = _image_to_bytes(img, fmt=fmt)
    buf = io.BytesIO(data)
    uploaded = client.files.upload(
        file=buf,
        config=types.UploadFileConfig(mime_type=mime),
    )
    return uploaded.uri, mime, uploaded


def _cleanup_uploaded_files(client: Any, uploaded_files: list[Any]) -> None:
    """Delete files uploaded to the Files API."""
    for f in uploaded_files:
        try:
            client.files.delete(name=f.name)
        except Exception:
            logger.warning("Failed to delete uploaded file %s", getattr(f, "name", "?"))


def _tool_dicts() -> list[dict[str, Any]]:
    """Return tool declarations as FunctionParam dicts for the Interactions API."""
    return [
        {
            "type": "function",
            "name": "view_overview",
            "description": (
                "Returns the full atlas and slice images (separately) "
                "with all current annotations rendered."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "type": "function",
            "name": "view_zoom_pair",
            "description": (
                "Zooms into a region of both atlas and slice for detailed "
                "inspection. Returns separate atlas and slice zoomed images. "
                "Existing annotations are visible in the zoomed views."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "zoom": {
                        "type": "number",
                        "description": "Zoom factor (e.g., 3.0 for 3x)",
                    },
                    "atlas_center_2d": {
                        "type": "array",
                        "description": "[y, x] in 0-1000 normalized range",
                        "items": {"type": "integer"},
                    },
                    "slice_center_2d": {
                        "type": "array",
                        "description": "[y, x] in 0-1000 normalized range",
                        "items": {"type": "integer"},
                    },
                },
                "required": ["zoom", "atlas_center_2d", "slice_center_2d"],
            },
        },
        {
            "type": "function",
            "name": "place_point_pair",
            "description": (
                "Place a matched landmark pair on atlas and slice. "
                "Re-calling with the same label updates the position."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "description": 'Point label (e.g., "1", "2")'},
                    "category": {
                        "type": "string",
                        "description": '"border" or "interior"',
                        "enum": ["border", "interior"],
                    },
                    "feature_description": {
                        "type": "string",
                        "description": (
                            "Rich description of the anatomical feature being matched "
                            "(e.g., 'the deepest point of the dorsal midline notch')"
                        ),
                    },
                    "atlas_point_2d": {
                        "type": "array",
                        "description": "[y, x] global coordinates in 0-1000 range",
                        "items": {"type": "integer"},
                    },
                    "slice_point_2d": {
                        "type": "array",
                        "description": "[y, x] global coordinates in 0-1000 range",
                        "items": {"type": "integer"},
                    },
                    "atlas_point_2d_local": {
                        "type": "array",
                        "description": "[y, x] local coordinates relative to last zoom view",
                        "items": {"type": "integer"},
                    },
                    "slice_point_2d_local": {
                        "type": "array",
                        "description": "[y, x] local coordinates relative to last zoom view",
                        "items": {"type": "integer"},
                    },
                    "artifact_note": {
                        "type": "string",
                        "description": "Note about damage/artifacts at this location",
                    },
                },
                "required": ["label", "category", "feature_description"],
            },
        },
        {
            "type": "function",
            "name": "finish",
            "description": (
                "Complete landmark placement. Rejected if border and "
                "interior quotas are not met."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    ]


def _result_items_from_tool(
    result_dict: dict[str, Any],
    result_images: list[types.Part],
) -> list[dict[str, Any]]:
    """Convert tool execution output to Interactions API result items."""
    items: list[dict[str, Any]] = []
    for rp in result_images:
        text_val = getattr(rp, "text", None)
        if text_val:
            items.append({"type": "text", "text": text_val})
        inline = getattr(rp, "inline_data", None)
        if inline and getattr(inline, "data", None):
            mime = getattr(inline, "mime_type", "image/jpeg")
            b64 = base64.b64encode(inline.data).decode("utf-8")
            items.append({
                "type": "image", "data": b64, "mime_type": mime,
                "resolution": "high",
            })
    # Always include the result dict as text for structured info
    if not items:
        items.append({"type": "text", "text": json.dumps(result_dict, indent=2)})
    return items


# ---------------------------------------------------------------------------
# Main loop (Interactions API)
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

    # Upscale atlas for better visual comparison.
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

    # Build prompt
    prompt_text = _build_tool_loop_prompt(
        atlas_name=atlas_name,
        position_mm=position_mm,
        border_count=border_count,
        interior_count=interior_count,
    )

    # Upload base images via Files API for maximum resolution, falling back
    # to inline base64 if the Files API is unavailable.
    _ai_config = importlib.import_module("langslice.ai.config")
    uploaded_files: list[Any] = []
    if _ai_config.supports_file_api():
        atlas_uri, atlas_mime, atlas_file = _upload_to_file_api(
            client, atlas_for_model, fmt="PNG"
        )
        slice_uri, slice_mime, slice_file = _upload_to_file_api(
            client, slice_for_model, fmt="JPEG"
        )
        uploaded_files = [atlas_file, slice_file]
        atlas_content: dict[str, Any] = {
            "type": "image", "uri": atlas_uri,
            "mime_type": atlas_mime, "resolution": "high",
        }
        slice_content: dict[str, Any] = {
            "type": "image", "uri": slice_uri,
            "mime_type": slice_mime, "resolution": "high",
        }
        if on_progress:
            on_progress("  Uploaded base images via Files API (high resolution)")
    else:
        atlas_content = _pil_to_image_content(atlas_for_model, fmt="PNG")
        slice_content = _pil_to_image_content(slice_for_model, fmt="JPEG")
        if on_progress:
            on_progress("  Using inline base64 images (high resolution)")

    initial_input: list[dict[str, Any]] = [
        {"type": "text", "text": prompt_text},
        {"type": "text", "text": "Atlas reference image:"},
        atlas_content,
        {"type": "text", "text": "Histology slice image:"},
        slice_content,
        {"type": "text", "text": json.dumps(_session_summary(session), indent=2)},
    ]

    tools = _tool_dicts()
    prev_id: str | None = None

    for iteration in range(1, prepared.tool_loop_max_steps + 1):
        if on_progress:
            on_progress(
                f"Registration tool loop: step {iteration}/{prepared.tool_loop_max_steps}..."
            )

        # Build the interaction request
        create_kwargs: dict[str, Any] = {
            "model": prepared.model_name,
            "tools": tools,
            "system_instruction": (
                "You are an expert neuroanatomist placing matched landmark "
                "points. Follow the instructions in the user message exactly."
            ),
            "generation_config": {"temperature": prepared.temperature, "max_output_tokens": 4000},
        }
        if prev_id is None:
            create_kwargs["input"] = initial_input
        else:
            create_kwargs["input"] = current_input  # noqa: F821 — set below on prior iteration
            create_kwargs["previous_interaction_id"] = prev_id

        # Call the Interactions API with retries
        interaction = None
        last_err = None
        for attempt in range(1, 5):
            try:
                if on_progress:
                    on_progress(f"  step {iteration} (attempt {attempt}/4): request started")
                import time as _time
                _t0 = _time.monotonic()
                interaction = client.interactions.create(**create_kwargs)
                _elapsed = _time.monotonic() - _t0
                if on_progress:
                    on_progress(f"  step {iteration} (attempt {attempt}/4): response in {_elapsed:.1f}s")
                break
            except Exception as exc:
                last_err = exc
                if on_progress:
                    on_progress(f"  step {iteration} (attempt {attempt}/4): failed ({exc})")
                    # Log input shape for debugging 400 errors
                    inp = create_kwargs.get("input")
                    if isinstance(inp, list):
                        for ii, item in enumerate(inp):
                            if isinstance(item, dict):
                                itype = item.get("type", "?")
                                if itype == "function_result":
                                    result = item.get("result", {})
                                    ritems = result.get("items", []) if isinstance(result, dict) else []
                                    sizes = []
                                    for ri in ritems:
                                        if isinstance(ri, dict) and ri.get("type") == "image":
                                            sizes.append(f"img:{len(ri.get('data',''))//1024}KB")
                                        elif isinstance(ri, dict) and ri.get("type") == "text":
                                            sizes.append(f"txt:{len(ri.get('text',''))}ch")
                                    on_progress(f"    input[{ii}]: {itype} name={item.get('name')} items=[{', '.join(sizes)}]")
                if attempt < 4:
                    import time as _time
                    _time.sleep(min(2 ** attempt, 8))
        if interaction is None:
            raise RuntimeError(f"Interactions API failed after 4 attempts: {last_err}")

        prev_id = interaction.id

        # Log outputs
        fc_outputs = [o for o in interaction.outputs if getattr(o, "type", None) == "function_call"]
        if on_progress:
            on_progress(
                f"  -> Status: {interaction.status}, "
                f"outputs: {len(interaction.outputs)}, "
                f"function_calls: {[o.name for o in fc_outputs]}"
            )

        # If no function calls, nudge the model
        if not fc_outputs:
            placed_count = len(session.atlas_annotations)
            total_needed = (border_count or 0) + (interior_count or 0)
            current_input: Any = [
                {
                    "type": "text",
                    "text": (
                        f"Continue placing landmarks. You have "
                        f"{placed_count}/{total_needed} placed points. "
                        f"Use your tools to place the next point."
                    ),
                }
            ]
            continue

        # Process function calls
        finished = False
        result_contents: list[dict[str, Any]] = []

        for fc_out in fc_outputs:
            tool_name = fc_out.name
            tool_args = dict(fc_out.arguments) if fc_out.arguments else {}
            call_id = fc_out.id

            if on_progress:
                on_progress(f"  -> FC: {tool_name}({tool_args})")

            # Build a lightweight shim so existing _execute_tool works
            class _FCShim:
                def __init__(self, name: str, args: dict, id: str):
                    self.name = name
                    self.args = args
                    self.id = id

            fc_shim = _FCShim(tool_name, tool_args, call_id)
            result_dict, result_images, is_finished = _execute_tool(
                fc_shim,
                session=session,
                atlas_image=atlas_for_model,
                slice_image=slice_for_model,
                iteration=iteration,
                on_trace=on_trace,
            )

            # Save debug images
            if prepared.registration_dir and result_images:
                step_dir = os.path.join(
                    prepared.registration_dir,
                    f"step_{iteration:02d}_{tool_name}",
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

            # Build function result for next turn.
            # Send the structured result as a simple string in the function_result,
            # and images as separate content items alongside it.
            result_contents.append({
                "type": "function_result",
                "call_id": call_id,
                "name": tool_name,
                "result": json.dumps(result_dict),
            })
            # Add images and text labels as separate content items
            for rp in result_images:
                text_val = getattr(rp, "text", None)
                if text_val:
                    result_contents.append({"type": "text", "text": text_val})
                inline = getattr(rp, "inline_data", None)
                if inline and getattr(inline, "data", None):
                    mime = getattr(inline, "mime_type", "image/jpeg")
                    b64 = base64.b64encode(inline.data).decode("utf-8")
                    result_contents.append({
                        "type": "image", "data": b64, "mime_type": mime,
                        "resolution": "high",
                    })

            if is_finished:
                finished = True

        # Set the input for the next turn
        current_input = result_contents

        if finished:
            entries = _placed_tool_loop_entries(session)
            logger.info("Tool loop completed via Interactions API")
            _cleanup_uploaded_files(client, uploaded_files)
            return entries

    _cleanup_uploaded_files(client, uploaded_files)
    raise RuntimeError("Tool-loop registration exceeded the maximum number of steps")
