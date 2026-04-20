"""The four position-estimation tools, wired as plain Python functions for ADK auto-wrapping.

On success, image-returning tools (`fetch_atlas`, `zoom`, `side_by_side`)
return ``list[types.Part]`` with interleaved text labels and image payloads.
ADK's stock ``MultimodalToolResultsPlugin`` intercepts this shape, stashes the
parts, and injects them into the next model turn as real inline-data images.
Without the plugin (or without the list-of-Parts shape), ADK would serialize
the dict — including any embedded ``types.Part`` — into JSON-compatible
Struct text, and the model would never see the actual pixels. That path was
the root cause of the pre-fix 1.3-1.7 mm MAE regression.

Error paths still return dicts (``{"status": "error", "error": ...}``); the
plugin passes non-Part returns through unchanged, so dict errors reach the
model as ordinary function responses.

Tools also save images as artifacts via ``tool_context.save_artifact`` so
later tools (``zoom``, ``side_by_side``) and the retry loop can retrieve them
by key regardless of which model turn they were fetched on.
"""

from __future__ import annotations

import hashlib
import io
import math
from typing import Any

from google.genai import types
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from langslice.atlas.core import (
    get_in_plane_long_edge,
    get_reference_slice,
    load_atlas,
)
from langslice.harness.estimation.session import (
    ARTIFACT_ATLAS_PREFIX,
    ARTIFACT_SIDE_BY_SIDE_PREFIX,
    ARTIFACT_ZOOM_PREFIX,
    atlas_key,
)

# Composite layout constants for side_by_side.
_SBS_GAP_PX = 8
_SBS_LABEL_BAND_H = 28
_SBS_BG = (0, 0, 0)
_SBS_LABEL_FG = (255, 255, 255)

# ---- Pure helpers -------------------------------------------------------


def _parse_atlas_key(source: str) -> float:
    if not source.startswith(ARTIFACT_ATLAS_PREFIX):
        raise ValueError(f"Not an atlas source: {source!r}")
    tail = source[len(ARTIFACT_ATLAS_PREFIX):]
    try:
        return float(tail)
    except ValueError as exc:
        raise ValueError(f"Bad atlas position in {source!r}") from exc


def _is_broad_sweep(positions: list[float]) -> bool:
    return len(positions) >= 3


def _is_narrow_sweep(positions: list[float]) -> bool:
    if len(positions) < 3:
        return False
    return (max(positions) - min(positions)) <= 1.0


def _clamp_and_dedupe_positions(
    positions: list[float], *, pos_lo: float, pos_hi: float, dedupe_tol: float = 0.02
) -> list[float]:
    clamped = [max(pos_lo, min(pos_hi, float(p))) for p in positions]
    out: list[float] = []
    for p in clamped:
        if any(abs(p - q) <= dedupe_tol for q in out):
            continue
        out.append(p)
    return out


def _image_to_jpeg_bytes(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _image_to_part(img: Image.Image) -> types.Part:
    return types.Part.from_bytes(
        mime_type="image/jpeg", data=_image_to_jpeg_bytes(img)
    )


def _short_hash(bbox: list[int]) -> str:
    return hashlib.sha1(str(bbox).encode("utf-8")).hexdigest()[:12]


async def fetch_atlas(
    positions_mm: list[float], tool_context: Any
) -> list[types.Part] | dict[str, Any]:
    """Fetch 1-8 atlas sections along the session's slicing plane.

    Positions outside the valid range are clamped. Duplicate positions within
    0.02 mm of an already-requested one are coalesced. Each returned slice is
    saved as an artifact keyed 'atlas:<mm:.2f>' so later `zoom` /
    `side_by_side` calls can reference it.

    Returns:
        On success, ``list[types.Part]`` interleaving a header text, then
        ``[text("Atlas at {mm} mm (key='atlas:{mm}'):"), image_part]`` for each
        position. `MultimodalToolResultsPlugin` surfaces these as real inline
        images on the next model turn.
        On failure, ``{"status": "error", "error": "BAD_ARGS" | "EMPTY_RESULT"}``.
    """
    state = tool_context.state
    if not positions_mm:
        return {"status": "error", "error": "BAD_ARGS"}

    pos_lo = float(state["pos_lo"])
    pos_hi = float(state["pos_hi"])
    plane = state["plane"]
    atlas_name = state["atlas"]

    capped = list(positions_mm)[:8]
    positions = _clamp_and_dedupe_positions(capped, pos_lo=pos_lo, pos_hi=pos_hi)
    if not positions:
        return {"status": "error", "error": "EMPTY_RESULT"}

    atlas = load_atlas(atlas_name)
    image_parts: list[types.Part] = []
    descriptions: list[str] = []
    for pos in positions:
        img = get_reference_slice(atlas, pos, plane=plane)
        part = _image_to_part(img)
        image_parts.append(part)
        await tool_context.save_artifact(filename=atlas_key(pos), artifact=part)
        descriptions.append(f"{pos:.2f} mm")

    state.setdefault("fetched_positions", []).extend(positions)
    state["images_fetched"] = int(state.get("images_fetched", 0)) + len(positions)
    if _is_broad_sweep(positions):
        state["saw_broad_sweep"] = True
    if _is_narrow_sweep(positions):
        state["saw_narrow_sweep"] = True

    out_parts: list[types.Part] = [
        types.Part.from_text(
            text=(
                f"Fetched {len(positions)} atlas section"
                f"{'s' if len(positions) != 1 else ''} at: "
                + ", ".join(descriptions)
                + "."
            )
        )
    ]
    for pos, img_part in zip(positions, image_parts, strict=True):
        out_parts.append(
            types.Part.from_text(
                text=f"Atlas at {pos:.2f} mm (key='{atlas_key(pos)}'):"
            )
        )
        out_parts.append(img_part)
    return out_parts


def submit_estimate(
    position_mm: float, reasoning: str, tool_context: Any
) -> dict[str, Any]:
    """Submit the final position estimate for the target slice.

    Only call this when you have completed broad + narrow atlas sweeps and
    verified at least one neighbor on each side of your candidate position.
    """
    tool_context.state["result"] = {"position_mm": float(position_mm), "reasoning": str(reasoning)}
    tool_context.actions.escalate = True
    return {"status": "ok", "position_mm": float(position_mm)}


def submit_group_estimate(
    positions_mm: list[float], reasoning: str, tool_context: Any
) -> dict[str, Any]:
    """Submit the final position estimates for all slices in the group, in order."""
    tool_context.state["result"] = {
        "positions_mm": [float(p) for p in positions_mm],
        "reasoning": str(reasoning),
    }
    tool_context.actions.escalate = True
    return {"status": "ok", "positions_mm": [float(p) for p in positions_mm]}


async def zoom(
    source: str, bbox: list[int], tool_context: Any,
) -> list[types.Part] | dict[str, Any]:
    """Return a zoomed crop of a previously-fetched image.

    Args:
        source: Must match an existing artifact key. Accepts "target",
            "target:N", "atlas:<mm>", or a previous "zoom:...".
        bbox: [y1, x1, y2, x2] integers in [0, 1000] (Gemini/Gemma normalized coords).

    Returns:
        On success, a two-element ``list[types.Part]``: ``[text_label,
        image_part]`` where the label encodes the output key and source bbox.
        The crop is also saved as an artifact under that key.
        On failure, ``{"status": "error", "error": ...}`` with error in
        {"BAD_SOURCE", "BAD_BBOX", "EMPTY_CROP"}.
    """
    # 1. Validate bbox shape and ranges.
    if (
        not isinstance(bbox, list)
        or len(bbox) != 4
        or any(not isinstance(v, int) or isinstance(v, bool) for v in bbox)
        or any(v < 0 or v > 1000 for v in bbox)
    ):
        return {"status": "error", "error": "BAD_BBOX"}
    y1, x1, y2, x2 = bbox
    if y1 >= y2 or x1 >= x2:
        return {"status": "error", "error": "BAD_BBOX"}

    # 2. Load source artifact (any existing key is acceptable).
    try:
        part = await tool_context.load_artifact(filename=source)
    except Exception:
        return {"status": "error", "error": "BAD_SOURCE"}
    if part is None:
        return {"status": "error", "error": "BAD_SOURCE"}

    # 3. Decode Part bytes to a PIL image.
    inline = getattr(part, "inline_data", None)
    data = getattr(inline, "data", None) if inline is not None else None
    if not data:
        return {"status": "error", "error": "BAD_SOURCE"}
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode now so truncated files fail here
    except (UnidentifiedImageError, OSError, ValueError):
        return {"status": "error", "error": "BAD_SOURCE"}

    # 4. Convert normalized (0-1000) bbox to pixel coords.
    #    Use floor for lower bounds and ceil for upper bounds so any non-empty
    #    normalized bbox yields a non-empty pixel bbox. round()'s banker's
    #    rounding could non-deterministically collapse thin bboxes to zero.
    w, h = img.size
    px_x1 = max(0, math.floor(x1 * w / 1000))
    px_y1 = max(0, math.floor(y1 * h / 1000))
    px_x2 = min(w, math.ceil(x2 * w / 1000))
    px_y2 = min(h, math.ceil(y2 * h / 1000))
    if px_x2 <= px_x1 or px_y2 <= px_y1:
        return {"status": "error", "error": "EMPTY_CROP"}

    # 5. Crop.
    cropped = img.crop((px_x1, px_y1, px_x2, px_y2))

    # 6. Resize so the long edge matches the atlas in-plane long edge.
    #    Upscale is intentional — tiny crops need to be visible to the model.
    state = tool_context.state
    atlas = load_atlas(state["atlas"])
    long_edge = get_in_plane_long_edge(atlas, plane=state["plane"])
    cw, ch = cropped.size
    scale = long_edge / float(max(cw, ch))
    new_size = (max(1, round(cw * scale)), max(1, round(ch * scale)))
    resized = cropped.resize(new_size, Image.Resampling.LANCZOS)

    # 7. Save as artifact and return as Part for the next model turn.
    bbox_hash = _short_hash(bbox)
    key = f"{ARTIFACT_ZOOM_PREFIX}{source}:{bbox_hash}"
    zoom_part = _image_to_part(resized)
    await tool_context.save_artifact(filename=key, artifact=zoom_part)

    return [
        types.Part.from_text(
            text=(
                f"Zoom of '{source}' at bbox {bbox}; scaled to "
                f"{new_size[0]}x{new_size[1]} (key='{key}'):"
            )
        ),
        zoom_part,
    ]


def _load_image_from_part(part: Any) -> Image.Image | None:
    """Decode a google.genai.types.Part's inline_data into a PIL image.

    Returns None if the part has no usable inline payload or the bytes cannot
    be decoded.
    """
    inline = getattr(part, "inline_data", None)
    data = getattr(inline, "data", None) if inline is not None else None
    if not data:
        return None
    try:
        img = Image.open(io.BytesIO(data))
        img.load()  # force decode now so truncated files fail here
    except (UnidentifiedImageError, OSError, ValueError):
        return None
    return img


def _rescale_to_height(img: Image.Image, target_h: int) -> Image.Image:
    if img.height == target_h:
        return img
    scale = target_h / float(img.height)
    new_w = max(1, round(img.width * scale))
    return img.resize((new_w, target_h), Image.Resampling.LANCZOS)


async def side_by_side(
    left: str, right: str, tool_context: Any,
) -> list[types.Part] | dict[str, Any]:
    """Build an aspect-ratio-matched horizontal composite of two images.

    Both source images are rescaled to a common height (aspect-ratio preserved
    per panel), placed side-by-side with a thin gap, and stamped with short
    labels derived from the source keys.

    Args:
        left: Existing artifact key for the left panel. Accepts "target",
            "target:N", "atlas:<mm>", "zoom:...", or any previously-saved
            "side_by_side:..." key.
        right: Same as `left` but for the right panel.

    Returns:
        On success: a two-element ``list[types.Part]``: ``[text_label,
        image_part]`` where the label encodes the output key and both source
        keys. The composite is also saved as an artifact under that key.
        On failure: ``{"status": "error", "error": "BAD_SOURCE"}``.
    """
    # 1. Load both artifacts. A missing or unreadable artifact on either side
    #    is reported as BAD_SOURCE.
    try:
        left_part = await tool_context.load_artifact(filename=left)
        right_part = await tool_context.load_artifact(filename=right)
    except Exception:
        return {"status": "error", "error": "BAD_SOURCE"}
    if left_part is None or right_part is None:
        return {"status": "error", "error": "BAD_SOURCE"}

    # 2. Decode both payloads to PIL images.
    left_img = _load_image_from_part(left_part)
    right_img = _load_image_from_part(right_part)
    if left_img is None or right_img is None:
        return {"status": "error", "error": "BAD_SOURCE"}

    # 3. Rescale both panels to a common height. Use the max of the two so
    #    neither panel is upscaled unnecessarily — detail in the taller image
    #    is preserved.
    target_h = max(left_img.height, right_img.height)
    left_scaled = _rescale_to_height(left_img, target_h)
    right_scaled = _rescale_to_height(right_img, target_h)

    # 4. Build the RGB composite canvas: [left_panel][gap][right_panel] with a
    #    label band across the top.
    left_rgb = left_scaled if left_scaled.mode == "RGB" else left_scaled.convert("RGB")
    right_rgb = (
        right_scaled if right_scaled.mode == "RGB" else right_scaled.convert("RGB")
    )

    total_w = left_rgb.width + _SBS_GAP_PX + right_rgb.width
    total_h = target_h + _SBS_LABEL_BAND_H
    composite = Image.new("RGB", (total_w, total_h), color=_SBS_BG)
    composite.paste(left_rgb, (0, _SBS_LABEL_BAND_H))
    composite.paste(right_rgb, (left_rgb.width + _SBS_GAP_PX, _SBS_LABEL_BAND_H))

    # 5. Draw labels in the top band. load_default() is always available and
    #    returns a usable font on every supported Pillow version; guard anyway.
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    draw.text((4, 4), left, fill=_SBS_LABEL_FG, font=font)
    draw.text(
        (left_rgb.width + _SBS_GAP_PX + 4, 4),
        right,
        fill=_SBS_LABEL_FG,
        font=font,
    )

    # 6. Save and return. The key encodes both source keys so repeat calls
    #    with identical arguments deduplicate through the artifact store.
    key = f"{ARTIFACT_SIDE_BY_SIDE_PREFIX}{left}:{right}"
    sbs_part = _image_to_part(composite)
    await tool_context.save_artifact(filename=key, artifact=sbs_part)

    return [
        types.Part.from_text(
            text=(
                f"Side-by-side composite (key='{key}'). "
                f"Left: '{left}'. Right: '{right}'."
            )
        ),
        sbs_part,
    ]
