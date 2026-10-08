"""Atlas sections for the toolbox, rendered at a plane's cutting angles.

One entry point, :func:`atlas_section`, so the sections the agent looks at, the
sections a fit is measured against and the sections a preview overlays are the
same pixels. When the angles are 0/0 this is the flat voxel-grid slice;
otherwise the plane is resampled obliquely. *angles* is the
``(pitch, yaw)`` to draw at: a section's own (``SliceState.angles``) for a
picture of that section, ``StackState.view_angles`` for one without a
section; None reads ``StackState.stack_angles``, which refuses a stack whose
sections differ (:class:`langslice.core.state.MixedAngles`).
"""

from __future__ import annotations

import logging
import math
from typing import cast

import numpy as np
from PIL import Image

from langslice.core.affine import resize_long_edge
from langslice.core.atlas.core import get_reference_slice, get_root_mask
from langslice.core.image_prep import crop_to_mask
from langslice.core.sizes import opening_edge
from langslice.core.space import Plane
from langslice.core.state import Angles, StackState, plane_angles
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: Most atlas sections in the opening's atlas reference (laid out as strips by
#: :mod:`langslice.core.opening`, sent when a section has no position).
SEED_ATLAS_MAX_IMAGES = 48


def atlas_mask(
    ctx: Workspace, state: StackState, position_mm: float, size: tuple[int, int],
    *, angles: Angles | None = None,
) -> np.ndarray:
    """Binary tissue silhouette of the atlas section, at the plane's angles."""
    pitch, yaw = plane_angles(state, angles)
    return get_root_mask(ctx.atlas, position_mm, size, plane=cast(Plane, state.plane),
                         pitch_deg=pitch, yaw_deg=yaw)


def atlas_section(
    ctx: Workspace,
    state: StackState,
    position_mm: float,
    *,
    frame: bool = False,
    angles: Angles | None = None,
) -> Image.Image:
    """The atlas template section at *position_mm*, at the plane's angles.

    *frame* crops to the tissue silhouette plus a margin, so the section fills
    its frame about as much as a tissue-framed histology render does. The
    silhouette comes from the ANNOTATION, not from the render's non-zero
    pixels: reference volumes carry faint background noise that would put the
    bounding box back at the canvas edges.
    """
    pitch, yaw = plane_angles(state, angles)
    image = get_reference_slice(
        ctx.atlas,
        position_mm,
        plane=cast(Plane, state.plane),
        pitch_deg=pitch,
        yaw_deg=yaw,
    )
    if not frame:
        return image
    try:
        mask = atlas_mask(ctx, state, position_mm, image.size, angles=(pitch, yaw))
    except Exception as exc:
        logger.warning("atlas at %.3f mm: no root mask to frame it by (%s)", position_mm, exc)
        return image
    return crop_to_mask(image, mask > 0)


def atlas_sized(picture: Image.Image, long_edge: int) -> Image.Image:
    """*picture*, an atlas render at its own voxel size, at most *long_edge*.

    Shrunk to *long_edge* when larger; never upsampled, since the plane holds
    no detail finer than its voxels (a mouse section at 25 um is ~300-450 px).
    The same object when nothing changes.
    """
    if max(picture.size) <= int(long_edge):
        return picture
    return resize_long_edge(picture, int(long_edge))


def reference_atlas(
    ctx: Workspace, state: StackState, *, long_edge: int | None = None,
    max_images: int = SEED_ATLAS_MAX_IMAGES,
) -> tuple[float, list[tuple[float, Image.Image]]]:
    """The atlas at evenly spaced positions for the opening: ``(step, pictures)``.

    The reference spans the atlas's valid range at the
    nominal interval, or coarser when that would exceed *max_images*. Each
    picture is tissue-framed at the stack's cutting angles
    (``StackState.view_angles``: the median of the sections' when they
    differ) and at most *long_edge* (None: the run's opening size), never
    upsampled; a plane with nothing in it (an oblique plane through the
    volume's corner) is skipped.
    :mod:`langslice.core.opening` lays the pictures out as strips.
    """
    pos_lo, pos_hi = ctx.position_range
    span = pos_hi - pos_lo
    step = max(state.interval_mm or 0.0, span / max(1, max_images - 1))
    step = math.ceil(step / 0.05) * 0.05  # a round number of 50 um
    edge = int(long_edge or opening_edge(ctx))
    pictures: list[tuple[float, Image.Image]] = []
    for k in range(int(span / step) + 1):
        position = pos_lo + k * step
        picture = atlas_section(ctx, state, position, frame=True, angles=state.view_angles)
        if np.asarray(picture).max() < 8:
            continue  # an oblique plane through the volume's corner: nothing to show
        pictures.append((position, atlas_sized(picture, edge)))
    return step, pictures
