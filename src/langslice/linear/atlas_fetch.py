"""Atlas sections for the toolbox, rendered at the stack's cutting angles.

One entry point, :func:`atlas_section`, so the sections the agent looks at, the
sections a fit is measured against and the sections a preview overlays are the
same pixels. When ``cutting_angles_deg`` is 0/0 this is the flat voxel-grid
slice; otherwise the plane is resampled obliquely.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
from google.genai import types
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import resize_long_edge
from langslice.atlas.core import get_reference_slice, get_root_mask
from langslice.image_prep import crop_to_mask
from langslice.linear.render import MAX_IMAGES_PER_CALL, caption, image_to_part
from langslice.linear.state import StackState
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

#: Atlas sections one ``fetch_atlas`` call may return. Anything past this is
#: dropped — and reported back, never silently.
MAX_FETCH_POSITIONS = MAX_IMAGES_PER_CALL

#: Long edge of every atlas image ``fetch_atlas`` returns. A tissue-framed
#: atlas render is at native atlas resolution, so an anterior section would
#: arrive at ~180 px while the stack images are 512 px; resizing puts every
#: atlas image at the size of the sections it is compared with.
ATLAS_LONG_EDGE = 512


def _as_floats(values: list[Any]) -> list[float]:
    """Model output is a trust boundary: keep the numbers, skip the rest.

    One level of nesting is walked: a model occasionally emits
    ``positions_mm=[[1.5, 2.5]]``.
    """
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(_as_floats(list(value)))
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _clamp_and_dedupe(
    positions: list[float], *, pos_lo: float, pos_hi: float, dedupe_tol: float = 0.02
) -> list[float]:
    out: list[float] = []
    for value in positions:
        clamped = max(pos_lo, min(pos_hi, value))
        if any(abs(clamped - kept) <= dedupe_tol for kept in out):
            continue
        out.append(clamped)
    return out


def atlas_mask(
    ctx: EngineContext, state: StackState, position_mm: float, size: tuple[int, int]
) -> np.ndarray:
    """Binary tissue silhouette of the atlas section, at the stack's angles."""
    plane = cast(Plane, state.plane)
    if not state.is_oblique:
        return get_root_mask(ctx.atlas, position_mm, size, plane=plane)
    from langslice.oblique import sample_oblique_annotation

    labels = sample_oblique_annotation(
        ctx.atlas, position_mm, plane, state.pitch_deg, state.yaw_deg
    )
    mask = (labels != 0).astype(np.uint8) * 255
    resized = Image.fromarray(mask, mode="L").resize(
        size, resample=Image.Resampling.NEAREST
    )
    return np.asarray(resized, dtype=np.uint8)


def atlas_section(
    ctx: EngineContext,
    state: StackState,
    position_mm: float,
    *,
    frame: bool = False,
) -> Image.Image:
    """The atlas template section at *position_mm*, at the stack's angles.

    *frame* crops to the tissue silhouette plus a margin, so the section fills
    its frame about as much as a tissue-framed histology render does. The
    silhouette comes from the ANNOTATION, not from the render's non-zero
    pixels: reference volumes carry faint background noise that would put the
    bounding box back at the canvas edges.
    """
    image = get_reference_slice(
        ctx.atlas,
        position_mm,
        plane=cast(Plane, state.plane),
        pitch_deg=state.pitch_deg,
        yaw_deg=state.yaw_deg,
    )
    if not frame:
        return image
    try:
        mask = atlas_mask(ctx, state, position_mm, image.size)
    except Exception:
        return image
    return crop_to_mask(image, mask > 0)


def atlas_part(ctx: EngineContext, state: StackState, position_mm: float) -> types.Part:
    """One tissue-framed atlas section at *position_mm*, section-sized and captioned."""
    angles = (
        f" pitch {state.pitch_deg:.1f} yaw {state.yaw_deg:.1f}"
        if state.is_oblique
        else ""
    )
    return image_to_part(
        caption(
            resize_long_edge(
                atlas_section(ctx, state, position_mm, frame=True), ATLAS_LONG_EDGE
            ),
            f"atlas {position_mm:.2f} mm{angles}",
        )
    )


def make_fetch_atlas(state: StackState, ctx: EngineContext):
    """Build the ``fetch_atlas`` tool, closed over the run's atlas and plane."""
    pos_lo, pos_hi = ctx.position_range

    def fetch_atlas(positions_mm: list[float]) -> dict[str, Any]:
        """Fetch atlas sections at the positions you name, at most 4 per call.

        Sections are rendered at the stack's current cutting angles, each
        labelled with its position (and the angles, when the stack is oblique)
        in its top-left corner. Ask for
        more than 4 and only the first 4 are fetched; the rest come back under
        ``dropped_positions_mm`` with ``truncated: true``. Positions outside
        the atlas range are clamped, and positions within 0.02 mm of one
        already in the same call are coalesced.

        Args:
            positions_mm: Positions along the slicing axis, in millimetres.

        Returns:
            status/positions plus the atlas images, in the order requested.
        """
        requested = _as_floats(list(positions_mm or []))
        if not requested:
            return {"status": "error", "error": "BAD_ARGS"}
        dropped = [round(value, 2) for value in requested[MAX_FETCH_POSITIONS:]]
        positions = _clamp_and_dedupe(
            requested[:MAX_FETCH_POSITIONS], pos_lo=pos_lo, pos_hi=pos_hi
        )
        if not positions:
            return {"status": "error", "error": "EMPTY_RESULT"}

        parts: list[types.Part] = [
            atlas_part(ctx, state, position) for position in positions
        ]
        plural = "s" if len(positions) != 1 else ""
        result: dict[str, Any] = {
            "status": "ok",
            "positions_mm": [round(position, 2) for position in positions],
            "cutting_angles_deg": dict(state.cutting_angles_deg),
            # Each image carries its own burned-in label; the ordering note
            # says the same thing in the payload.
            "description": (
                f"Fetched {len(positions)} atlas section{plural}: "
                + ", ".join(f"{position:.2f} mm" for position in positions)
                + ". The attached atlas images appear in that same order, each "
                "labelled with its position in its top-left corner."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        if dropped:
            result["truncated"] = True
            result["dropped_positions_mm"] = dropped
            result["description"] += (
                f" You asked for {len(requested)} positions; only the first "
                f"{MAX_FETCH_POSITIONS} were fetched. NOT fetched, and not "
                "shown to you: "
                + ", ".join(f"{position:.2f} mm" for position in dropped)
                + "."
            )
        return result

    return fetch_atlas
