"""The four position-estimation tools, wired as plain Python functions for ADK auto-wrapping.

Tools return dicts. Image outputs are placed under the key "images" as a list
of google.genai.types.Part objects — ADK surfaces these to the next model turn.
Tools also save images as artifacts via tool_context.save_artifact so later
tools (zoom, side_by_side) can retrieve them by key.
"""

from __future__ import annotations

import hashlib
import io
from typing import Any

from google.genai import types
from PIL import Image

from langslice.atlas.core import (
    get_in_plane_long_edge,  # noqa: F401 -- used by later tasks (zoom)
    get_reference_slice,
    load_atlas,
)
from langslice.harness.estimation.session import (
    ARTIFACT_ATLAS_PREFIX,
    ARTIFACT_SIDE_BY_SIDE_PREFIX,  # noqa: F401 -- used by later tasks (side_by_side)
    ARTIFACT_TARGET,  # noqa: F401 -- used by later tasks (zoom/side_by_side)
    ARTIFACT_TARGET_PREFIX,  # noqa: F401 -- used by later tasks (side_by_side)
    ARTIFACT_ZOOM_PREFIX,  # noqa: F401 -- used by later tasks (zoom)
    atlas_key,
)

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
    return hashlib.sha1(str(bbox).encode("utf-8")).hexdigest()[:8]


def fetch_atlas(
    positions_mm: list[float], tool_context: Any
) -> dict[str, Any]:
    """Fetch 1-8 atlas sections along the session's slicing plane.

    Positions outside the valid range are clamped. Duplicate positions within
    0.02 mm of an already-requested one are coalesced. Each returned slice is
    saved as an artifact keyed 'atlas:<mm:.2f>' and returned as a types.Part
    image in the "images" field of the response so the next model turn can
    see it.

    Returns:
        {"status": "ok", "positions_mm": [...], "images": [Part, ...]} or
        {"status": "error", "error": "BAD_ARGS" | "EMPTY_RESULT"}.
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
    parts: list[types.Part] = []
    descriptions: list[str] = []
    for pos in positions:
        img = get_reference_slice(atlas, pos, plane=plane)
        part = _image_to_part(img)
        parts.append(part)
        key = atlas_key(pos)
        tool_context.save_artifact(key, part)
        descriptions.append(f"{pos:.2f} mm")

    # Update session state
    state.setdefault("fetched_positions", []).extend(positions)
    state["images_fetched"] = int(state.get("images_fetched", 0)) + len(positions)
    if _is_broad_sweep(positions):
        state["saw_broad_sweep"] = True
    if _is_narrow_sweep(positions):
        state["saw_narrow_sweep"] = True

    return {
        "status": "ok",
        "positions_mm": positions,
        "description": f"{len(positions)} atlas sections at: " + ", ".join(descriptions),
        "images": parts,
    }


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
