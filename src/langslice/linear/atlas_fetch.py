"""Atlas sections for the toolbox, rendered at the stack's cutting angles.

One entry point, :func:`atlas_section`, so the sections the agent looks at, the
sections a fit is measured against and the sections a preview overlays are the
same pixels. When ``cutting_angles_deg`` is 0/0 this is the flat voxel-grid
slice; otherwise the plane is resampled obliquely.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from PIL import Image

from langslice.affine import resize_long_edge
from langslice.atlas.core import get_reference_slice, get_root_mask
from langslice.core.captions import caption
from langslice.core.sizes import opening_edge, picture_edge
from langslice.image_prep import crop_to_mask
from langslice.linear.state import StackState
from langslice.linear.workspace import Workspace
from langslice.space import Plane

#: Most atlas sections in the opening's atlas reference (laid out as strips by
#: :mod:`langslice.linear.opening`, sent when a section has no position).
SEED_ATLAS_MAX_IMAGES = 48


def atlas_mask(
    ctx: Workspace, state: StackState, position_mm: float, size: tuple[int, int]
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
    ctx: Workspace,
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


def atlas_sized(picture: Image.Image, long_edge: int) -> Image.Image:
    """*picture*, an atlas render at its own voxel size, at most *long_edge*.

    Shrunk to *long_edge* when larger; never upsampled, since the plane holds
    no detail finer than its voxels (a mouse section at 25 um is ~300-450 px).
    The same object when nothing changes.
    """
    if max(picture.size) <= int(long_edge):
        return picture
    return resize_long_edge(picture, int(long_edge))


def atlas_picture(
    ctx: Workspace, state: StackState, position_mm: float, *,
    long_edge: int | None = None, prepared: Image.Image | None = None,
) -> Image.Image:
    """One tissue-framed atlas section at *position_mm*, sized and captioned.

    *long_edge* None is the run's later-picture size
    (:func:`langslice.core.sizes.picture_edge`); *prepared* is a picture
    already drawn at *long_edge*.
    """
    long_edge = long_edge or picture_edge(ctx)
    angles = (
        f" pitch {state.pitch_deg:.1f} yaw {state.yaw_deg:.1f}"
        if state.is_oblique
        else ""
    )
    return caption(
        prepared if prepared is not None else atlas_sized(
            atlas_section(ctx, state, position_mm, frame=True), long_edge,
        ),
        f"atlas {position_mm:.2f} mm{angles}",
    )


def reference_atlas(
    ctx: Workspace, state: StackState, *, long_edge: int | None = None,
    max_images: int = SEED_ATLAS_MAX_IMAGES,
) -> tuple[float, list[tuple[float, Image.Image]]]:
    """The atlas at evenly spaced positions for the opening: ``(step, pictures)``.

    Until 2026-09-09 the model never saw the atlas as a set: four bare atlas
    sections from one ``view_atlas`` and then only ever half of a
    comparison pair. The reference spans the atlas's valid range at the
    nominal interval, or coarser when that would exceed *max_images*. Each
    picture is tissue-framed at the stack's cutting angles and at most
    *long_edge* (None: the run's opening size), never upsampled; a plane with
    nothing in it (an oblique plane through the volume's corner) is skipped.
    :mod:`langslice.linear.opening` lays the pictures out as strips.
    """
    pos_lo, pos_hi = ctx.position_range
    span = pos_hi - pos_lo
    step = max(state.interval_mm, span / max(1, max_images - 1))
    step = math.ceil(step / 0.05) * 0.05  # a round number of 50 um
    edge = int(long_edge or opening_edge(ctx))
    pictures: list[tuple[float, Image.Image]] = []
    for k in range(int(span / step) + 1):
        position = pos_lo + k * step
        picture = atlas_section(ctx, state, position, frame=True)
        if np.asarray(picture).max() < 8:
            continue  # an oblique plane through the volume's corner: nothing to show
        pictures.append((position, atlas_sized(picture, edge)))
    return step, pictures
