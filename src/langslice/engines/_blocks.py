"""What the in-tree engines share: the atlas image a call used, the section mask it was
given, and the ``inputs`` / ``estimated`` blocks every fit reply carries
(``docs/developers/plugins.md``).

The blocks are plain dicts of the shape ``langslice.core.fits.inputs`` gives
the native fits; this module reaches the job through
:mod:`langslice.plugin_api`. A reply names what the fit used and nothing about how well it did.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from langslice.core.sections.masks import MaskView
from langslice.plugin_api import FitFrame, RefusalCode, ToolContext, ToolError

NISSL = "nissl"
BORROWED_NISSL = "ccfv3augmented_nissl"
MASK_ENGINES = "fireants_syn, elastix_affine, emlddmm_stack and spomialign"
SORTING_SAVED = "saved with the deformation; make_mask source fit"


def call_folder(ctx: ToolContext, name: str) -> Path:
    """A new folder for one call's files, ``<tool's folder>/calls/NNNN_<name>``, made here so
    two calls running at once never share one."""
    calls = ctx.work_dir() / "calls"
    calls.mkdir(parents=True, exist_ok=True)
    n = len([path for path in calls.iterdir() if path.is_dir()])
    while True:
        folder = calls / f"{n:04d}_{name}"
        try:
            folder.mkdir()
        except FileExistsError:
            n += 1
            continue
        return folder


def job_path(context: ToolContext, path: Path) -> str:
    """Write a path as a reply or a record names it.

    Parameters
    ----------
    context
        The call's context, whose job folder the path is written against.
    path
        A file or folder the call made.

    Returns
    -------
    str
        The path relative to the job folder with ``/`` between its parts on
        every platform, or the whole path when it lies outside the folder.
    """
    if path.is_relative_to(context.job_dir):
        return path.relative_to(context.job_dir).as_posix()
    return str(path)


def frame_for(ctx: ToolContext, sid: str, atlas_image: str, section_image: str) -> FitFrame:
    """The section's fit frame with *section_image* (a section image by name) and
    *atlas_image*."""
    frame = ctx.frames.fit_frame(sid, atlas=(atlas_image, "annotation"), image=section_image)
    atlas_plane(frame, atlas_image)
    return frame


def atlas_plane(frame: FitFrame, asked: str) -> tuple[np.ndarray, str]:
    """The atlas image *asked* on *frame* and the name it resolved to ("nissl" is the atlas's
    Nissl, else its template)."""
    planes = frame.atlas
    key = asked if asked in planes else None
    if key is None and asked == NISSL:
        named = sorted(name for name in planes if "nissl" in name)
        key = next((name for name in (BORROWED_NISSL, *named, "template") if name in planes), None)
    if key is None:
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f"the atlas has no image {asked!r}; the job statement lists its images",
        )
    return planes[key], frame.atlas_resolved.get(asked) or key


def section_pixels(frame: FitFrame) -> np.ndarray:
    """The section image the frame was asked for, on the fit grid (RGB, uint8)."""
    return np.asarray(frame.section_image, np.uint8)


def section_mask(ctx: ToolContext, sid: str, mask: str) -> MaskView | None:
    """The named mask of the section (``NOT_FOUND`` / ``STALE_INPUT`` refuse), or None for
    ``"none"``."""
    if mask == "none":
        return None
    return ctx.reads.mask(sid, mask)


def refuse_mask(tool: str, mask: str, why: str) -> None:
    """``BAD_ARGUMENT`` for an engine that takes no section mask."""
    if mask != "none":
        raise ToolError(
            RefusalCode.BAD_ARGUMENT,
            f'{tool} {why}; mask is "none" for it. Engines '
            f"that take a section mask: {MASK_ENGINES}.",
        )


def inputs_block(
    ctx: ToolContext,
    frame: FitFrame,
    *,
    section_image: str,
    atlas_image: str,
    resolved: str,
    mask: MaskView | None,
    brightness: str,
) -> dict[str, Any]:
    """The ``inputs`` block of one section: the section image (name, kind, file, digest), the
    atlas image (as asked and as resolved), the section mask, the atlas-side damage and what
    the engine does to brightness."""
    reference = frame.section_image_ref
    return {
        "section_image": dict(reference) if reference else {"name": section_image},
        "atlas_image": atlas_image,
        "atlas_image_resolved": resolved,
        **(
            {"atlas_image_note": note}
            if atlas_image == NISSL and (note := frame.atlas_image_note)
            else {}
        ),
        "mask": {"name": "none", "kind": None, "stale": False, "earlier_registration": False}
        if mask is None
        else {
            "name": mask.name,
            "kind": mask.kind,
            "stale": bool(mask.stale),
            "earlier_registration": bool(mask.earlier_registration),
        },
        "damage": {
            "regions": list(ctx.reads.section(frame.id).damaged_regions),
            "margin_mm": frame.damage_margin_mm,
            "side": "atlas",
        },
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
