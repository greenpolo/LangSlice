"""What the three in-tree engines share: the atlas image a call used, the section mask it was
given, and the ``inputs`` / ``estimated`` blocks every fit reply carries (``docs/plugins.md``).

The blocks are plain dicts of the shape ``langslice.core.engine_inputs`` gives the native
fits; this module imports only :mod:`langslice.plugin_api`, as every engine does.
A reply names what the fit used and nothing about how well it did.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import cv2
import numpy as np

from langslice.plugin_api import ToolContext, ToolError

#: The default atlas image, resolved by the job's atlas when the frame is made.
NISSL = "nissl"
BORROWED_NISSL = "ccfv3augmented_nissl"
#: The engines that take a section mask, for the refusal of the one that does not.
MASK_ENGINES = "fireants_syn, elastix_affine, emlddmm_stack and spomialign"
SORTING_SAVED = "saved with the deformation; set_mask source fit"


def frame_for(ctx: ToolContext, sid: str, atlas_image: str, section_image: str) -> Any:
    """The section's fit frame with *section_image* (a channel by name) and *atlas_image*."""
    frame = ctx.fit_frame(sid, atlas=(atlas_image, "annotation"), channel=section_image)
    atlas_plane(frame, atlas_image)
    return frame


def atlas_plane(frame: Any, asked: str) -> tuple[np.ndarray, str]:
    """The atlas image *asked* on *frame* and the name it resolved to ("nissl" is the atlas's
    Nissl, else its template)."""
    planes = frame.atlas
    key = asked if asked in planes else None
    if key is None and asked == NISSL:
        named = sorted(name for name in planes if "nissl" in name)
        key = next((name for name in (BORROWED_NISSL, *named, "template") if name in planes),
                   None)
    if key is None:
        raise ToolError("BAD_ARGS", f"the atlas has no image {asked!r}; the job statement lists "
                        "its images")
    resolved = dict(getattr(frame, "atlas_resolved", None) or {}).get(asked)
    if resolved is None:
        resolved = key if key != NISSL else getattr(frame, "atlas_image", NISSL)
    return planes[key], str(resolved)


def channel_image(frame: Any) -> np.ndarray:
    """The section channel the frame was asked for, on the fit grid (RGB, uint8)."""
    return np.asarray(frame.channel, np.uint8)


def section_mask(ctx: ToolContext, sid: str, mask: str) -> Any | None:
    """The named mask of the section (``UNKNOWN_MASK`` / ``STALE_MASK`` refuse), or None for
    ``"none"``."""
    if mask == "none":
        return None
    return ctx.mask(sid, mask)


def refuse_mask(tool: str, mask: str, why: str) -> None:
    """``MASK_UNSUPPORTED`` for an engine that takes no section mask."""
    if mask != "none":
        raise ToolError("MASK_UNSUPPORTED", f"{tool} {why}; mask is \"none\" for it. Engines "
                        f"that take a section mask: {MASK_ENGINES}.")


def _margin_mm(frame: Any) -> float | None:
    """The margin the damage zone adds around the marked regions (every engine here leaves the
    zone out, as the native fits do), as the frame states it."""
    margin = getattr(frame, "damage_margin_mm", None)
    return None if margin is None else float(margin)


def inputs_block(ctx: ToolContext, frame: Any, *, section_image: str, atlas_image: str,
                 resolved: str, mask: Any | None, brightness: str) -> dict[str, Any]:
    """The ``inputs`` block of one section: the channel, its recipe, the atlas image (as asked
    and as resolved), the section mask, the atlas-side damage and what the engine does to
    brightness."""
    recipe = getattr(frame, "channel_recipe", None)
    if isinstance(recipe, dict):
        recipe = hashlib.sha1(json.dumps(recipe, sort_keys=True).encode()).hexdigest()[:8]
    return {
        "section_image": section_image,
        "section_recipe": str(recipe or "default"),
        "atlas_image": atlas_image,
        "atlas_image_resolved": resolved,
        **({"atlas_image_note": note} if atlas_image == NISSL and (
            note := getattr(frame, "atlas_image_note", None)) else {}),
        "mask": {"name": "none", "kind": None, "stale": False, "earlier_registration": False}
        if mask is None else {
            "name": mask.name, "kind": mask.kind, "stale": bool(mask.stale),
            "earlier_registration": bool(getattr(mask, "earlier_registration", False))},
        "damage": {"regions": list(ctx.section(frame.id).damaged_regions),
                   "margin_mm": _margin_mm(frame), "side": "atlas"},
        "brightness": brightness,
    }


def left_out_fraction(weights: np.ndarray, inside: np.ndarray) -> float | None:
    """The share of the tissue (*inside*) whose weight is under one half."""
    inside = np.asarray(inside, bool)
    if not inside.any():
        return None
    return round(float((np.asarray(weights, np.float32) < 0.5)[inside].mean()), 3)


def estimated_block(weights: np.ndarray | None, inside: np.ndarray) -> dict[str, Any] | None:
    """The ``estimated`` block: the engine's own sorting, which is saved with the deformation,
    and the share of the tissue it left out; None when the engine estimated nothing."""
    if weights is None:
        return None
    return {"sorting": SORTING_SAVED, "left_out_fraction": left_out_fraction(weights, inside)}


def on_fit_grid(weights: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A weight map resized to the fit grid ``(width, height)`` as float16 in [0, 1]."""
    resized = cv2.resize(np.asarray(weights, np.float32), size, interpolation=cv2.INTER_LINEAR)
    return np.clip(resized, 0.0, 1.0).astype(np.float16)
