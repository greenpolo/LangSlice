"""Position-estimation tools, wired as plain Python functions for ADK auto-wrapping.

On success, `fetch_atlas` returns a normal dictionary function response plus
a private list of ``types.Part`` image/text payloads.
``PersistentMultimodalToolResultsPlugin`` removes that private payload before
ADK builds the function response, then injects the real image parts into the
next model request.

Error paths still return dicts (``{"status": "error", "error": ...}``); the
plugin passes them through unchanged, so dict errors reach the model as
ordinary function responses.

Fetched atlas images are also saved as artifacts via
``tool_context.save_artifact`` for debugging and request replay.
"""

from __future__ import annotations

import io
from typing import Any

from google.genai import types
from PIL import Image

from langslice.atlas.core import (
    get_reference_slice,
    load_atlas,
)
from langslice.harness.estimation.adk_plugins import MULTIMODAL_PARTS_RESULT_KEY
from langslice.harness.estimation.session import (
    ARTIFACT_ATLAS_PREFIX,
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


def _with_multimodal_parts(
    response: dict[str, Any], parts: list[types.Part]
) -> dict[str, Any]:
    out = dict(response)
    out[MULTIMODAL_PARTS_RESULT_KEY] = parts
    return out


async def fetch_atlas(
    positions_mm: list[float], tool_context: Any
) -> dict[str, Any]:
    """Fetch 1-8 atlas sections along the session's slicing plane.

    Positions outside the valid range are clamped. Duplicate positions within
    0.02 mm of an already-requested one are coalesced. Each returned slice is
    saved as an artifact keyed 'atlas:<mm:.2f>'.

    Returns:
        On success, a structured function response with status/positions plus
        private multimodal parts. The plugin surfaces those parts as real
        inline images on later model turns.
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
    return _with_multimodal_parts(
        {
            "status": "ok",
            "positions_mm": [round(float(pos), 2) for pos in positions],
            "description": (
                f"Fetched {len(positions)} atlas section"
                f"{'s' if len(positions) != 1 else ''}: "
                + ", ".join(descriptions)
            ),
        },
        out_parts,
    )


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
