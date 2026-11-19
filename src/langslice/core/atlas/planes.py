"""Atlas sections for the toolbox, rendered at a plane's cutting angles.

One entry point, :func:`atlas_section`, so the sections the agent looks at, the
sections a fit is measured against and the sections a preview overlays are the
same pixels. When the angles are 0/0 this is the flat voxel-grid slice;
otherwise the plane is resampled obliquely. *angles* is the ``(pitch, yaw)`` to
draw at: a section's own (``Section.angles``) for a picture of that section,
``Stack.view_angles`` for one without a section; None reads
``Stack.stack_angles``, which refuses a stack whose sections differ
(:class:`langslice.core.stack.section.MixedAngles`).
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from PIL import Image

from langslice.core.atlas.load import get_reference_slice, get_root_mask
from langslice.core.failures import IMAGE_FAILURES
from langslice.core.geometry.affine import resize_long_edge
from langslice.core.geometry.pixel_frames import cropped_geometry
from langslice.core.sections.reading import mask_box
from langslice.core.stack.section import Angles
from langslice.core.stack.stack import Stack, plane_angles
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


PLANE_CACHE_SIZE = 128
_plane_lock = threading.Lock()


def atlas_mask(
    ctx: Workspace,
    state: Stack,
    position_mm: float,
    size: tuple[int, int],
    *,
    angles: Angles | None = None,
) -> np.ndarray:
    """Binary tissue silhouette of the atlas section, at the plane's angles."""
    pitch, yaw = plane_angles(state, angles)
    return get_root_mask(
        ctx.atlas, position_mm, size, plane=state.plane, pitch_deg=pitch, yaw_deg=yaw
    )


def atlas_section(
    ctx: Workspace,
    state: Stack,
    position_mm: float,
    *,
    frame: bool = False,
    angles: Angles | None = None,
) -> Image.Image:
    """The atlas template section at *position_mm*, at the plane's angles.

    *frame* crops to the brain silhouette plus a margin. The silhouette
    comes from the ANNOTATION, not from the render's non-zero
    pixels: reference volumes carry faint background noise that would put the
    bounding box back at the canvas edges. The image is the caller's own copy
    (``info`` included): the cached plane is never shared.
    """
    pitch, yaw = plane_angles(state, angles)
    key = (float(position_mm), str(state.plane), float(pitch), float(yaw), bool(frame))
    cache = ctx.plane_cache
    with _plane_lock:
        hit = cache.get(key)
        if hit is not None:
            cache.move_to_end(key)
    if hit is not None:
        return hit.copy()
    image = get_reference_slice(
        ctx.atlas,
        position_mm,
        plane=state.plane,
        pitch_deg=pitch,
        yaw_deg=yaw,
    )
    image.info["pixel_to_native"] = np.eye(3).tolist()
    if frame:
        try:
            mask = atlas_mask(ctx, state, position_mm, image.size, angles=(pitch, yaw))
        except IMAGE_FAILURES as exc:
            logger.warning("atlas at %.3f mm: no root mask to frame it by (%s)", position_mm, exc)
        else:
            box = mask_box(image.size, mask > 0)
            if box:
                image = cropped_geometry(image.crop(box), image, box)
    with _plane_lock:
        cache[key] = image
        while len(cache) > PLANE_CACHE_SIZE:
            cache.popitem(last=False)
    return image.copy()


def atlas_sized(picture: Image.Image, long_edge: int) -> Image.Image:
    """*picture*, an atlas render at its own voxel size, at most *long_edge*.

    Shrunk to *long_edge* when larger; never upsampled, since the plane holds
    no detail finer than its voxels (a mouse section at 25 um is ~300-450 px).
    The same object when nothing changes.
    """
    if max(picture.size) <= int(long_edge):
        return picture
    return cropped_geometry(resize_long_edge(picture, int(long_edge)), picture)
