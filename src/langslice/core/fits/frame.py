"""The geometry a border trace or a deformable fit starts from.

The core half of the bridge between a written linear placement and the
nonlinear work on top of it:
:func:`prepare_linear_registration` renders the section and maps the native
atlas plane onto it (the trace's canvas and every deformable fit's grid start
here), and :func:`correction_fingerprint` identifies everything an image
correction's inputs depend on, so a saved reply is reused only at the exact
geometry that produced it. Neither generates an image nor changes the stack;
neither imports a provider. The image-model call itself is
``registration_tool.start_correction``, which takes the model as an argument.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core.geometry.affine import denormalized_affine
from langslice.core.pictures.canvas import CanvasGeometry, canvas_geometry
from langslice.core.sections.images import PREPROCESSED, entry_of, is_opening
from langslice.core.sections.renders import (
    PREVIEW_LONG_EDGE,
    canvas_um_per_px,
    render_cache_key,
    render_slice,
)
from langslice.core.space import Plane
from langslice.core.stack.codec import affine_fields, angles_to_json
from langslice.core.workspace import Workspace

if TYPE_CHECKING:
    from langslice.core.stack.stack import Stack


@dataclass(frozen=True)
class LinearRegistrationInput:
    """Native sampled atlas pixels mapped onto the returned section image.

    Pass ``atlas_to_slice`` as nonlinear's ``initial_atlas_to_slice``. The image
    already includes the section's orientation and display preprocessing; use
    native atlas image axes and no additional atlas mirror for this handoff.
    """

    image: Image.Image
    atlas_to_slice: np.ndarray
    atlas_name: str
    position_mm: float
    plane: Plane
    pitch_deg: float
    yaw_deg: float
    metadata: dict[str, Any]


def linear_placement_matrix(
    image_size: tuple[int, int],
    um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    params: Any,
) -> tuple[np.ndarray, CanvasGeometry]:
    """``(atlas_to_slice, geometry)``: the linear placement as one matrix.

    *atlas_to_slice* (3x3) maps native atlas-plane pixel centres ``[x, y, 1]``
    (the plane :func:`langslice.core.atlas.borders.annotation_slice` draws at
    *position_mm*, *plane* and the cutting angles) onto pixel centres of the
    oriented section render of *image_size* that the six stored numbers
    *params* are normalized against, at *um_per_px* (the render's). The one
    geometry path from a stored placement to the atlas: the image-correction
    handoff, the deformable fit grid and the job folder's maps and
    ``registration.json`` (:mod:`langslice.core.maps.section_maps`) all go
    through it. ``ValueError`` for a singular or non-finite placement.
    """
    matrix = np.vstack([denormalized_affine(params, image_size), [0.0, 0.0, 1.0]])
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The written affine is singular") from exc
    if not np.isfinite(inverse).all():
        raise ValueError("The written affine cannot be inverted to finite coordinates")
    geometry = canvas_geometry(image_size, um_per_px, atlas, position_mm, plane,
                               pitch_deg, yaw_deg)
    atlas_to_slice = inverse @ geometry.atlas_to_section_frame()
    if not np.isfinite(atlas_to_slice).all():
        raise ValueError("Atlas placement produced non-finite coordinates")
    return atlas_to_slice, geometry


def prepare_linear_registration(
    state: Stack,
    ctx: Workspace,
    section_id: str,
    *,
    long_edge: int = 2048,
    params: Sequence[float] | None = None,
) -> LinearRegistrationInput:
    """Prepare a supplied affine placement, preserving shear and physical scale.

    Requires a written position and an invertible affine. When the transform
    carries an ``orientation`` snapshot or a ``stale`` flag, it is checked
    before the fit is used. The scale is the one every placement picture and
    ``registration.json`` draw the section at
    (:func:`langslice.core.fits.affine.calibrate` on its working frame, carried
    to *long_edge*), so the six numbers place the atlas here exactly where the
    pictures show it, also for a section without a pixel size whose position
    moved after its transform was written.

    *params* stand in for the six numbers of the section's written transform,
    for a fit that starts from a placement it has not written:
    ``elastix_affine`` on a section with no transform yet
    starts from the identity.

    The image is the section's ``preprocessed`` image
    (:mod:`langslice.core.sections.images`, what the image model reads); the
    frame, size, scale and placement never depend on it.
    """
    if isinstance(long_edge, bool) or not isinstance(long_edge, int) or long_edge <= 0:
        raise ValueError("long_edge must be a positive integer")
    section = state.by_id(section_id)
    if section is None:
        raise ValueError(f"Unknown section: {section_id}")
    if section.position_mm is None or not np.isfinite(section.position_mm):
        raise ValueError("A finite written position is required: write the "
                         "section's position first")
    orientation = None
    if params is None:
        transform = section.transform
        if transform is None:
            raise ValueError(ctx.settings.missing_transform)
        if transform.stale:
            raise ValueError("The written affine is marked stale")
        orientation = transform.orientation
        params = transform.params
    if orientation is not None:
        expected = {"flip": section.flip, "rotation_deg": section.rotation_deg}
        if any(key in orientation and orientation[key] != value
               for key, value in expected.items()):
            raise ValueError("The written affine has a different section orientation")
    if state.atlas != ctx.settings.atlas or state.plane != ctx.settings.plane:
        raise ValueError("State and rendering context disagree about the atlas or plane")
    if state.plane not in ("coronal", "sagittal", "horizontal"):
        raise ValueError("Unsupported section plane")
    if not np.isfinite([section.pitch_deg, section.yaw_deg]).all():
        raise ValueError("Cutting angles must be finite")
    six = np.asarray(params, dtype=np.float64)
    if six.shape != (6,) or not np.isfinite(six).all():
        raise ValueError("The written affine must contain six finite parameters")

    image = render_slice(ctx, section, long_edge=long_edge)
    held = entry_of(section, PREPROCESSED)
    remade = None
    if held is not None and not is_opening(held):
        remade = {"name": PREPROCESSED, "digest": held.digest}
    with Image.open(ctx.image_path(section.id)) as source_image:
        original_size = list(source_image.size)

    um_per_px, source = canvas_um_per_px(ctx, section, long_edge=long_edge)
    if um_per_px is None:
        from langslice.core.fits.affine import calibrate

        working = render_slice(ctx, section, long_edge=PREVIEW_LONG_EDGE)
        preview_um_per_px, source = calibrate(state, ctx, section, working)
        preview_key = render_cache_key(ctx, section, long_edge=PREVIEW_LONG_EDGE)
        target_key = render_cache_key(ctx, section, long_edge=long_edge)
        um_per_px = preview_um_per_px * ctx.render_scale[target_key] / ctx.render_scale[preview_key]
    if not np.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError("Section calibration must be finite and positive")

    plane = cast(Plane, state.plane)
    atlas_to_slice, geometry = linear_placement_matrix(
        image.size, um_per_px, ctx.atlas, float(section.position_mm), plane,
        section.pitch_deg, section.yaw_deg, six)
    return LinearRegistrationInput(
        image=image.copy(), atlas_to_slice=atlas_to_slice, atlas_name=state.atlas,
        position_mm=float(section.position_mm), plane=plane,
        pitch_deg=section.pitch_deg, yaw_deg=section.yaw_deg,
        metadata={
            "source": "linear_state", "section_id": section.id,
            "orientation": {"rotation_deg": section.rotation_deg, "flip": section.flip},
            "orientation_snapshot_checked": orientation is not None,
            "preprocess": ctx.settings.preprocess,
            **({"section_image": remade} if remade is not None else {}),
            "image_size": list(image.size),
            "image_frame": "oriented rendered section; not acquisition image pixels",
            "source_image_size": original_size,
            "section_um_per_px": float(um_per_px), "calibration_source": source,
            "linear_params": six.tolist(),
            "atlas_to_slice": atlas_to_slice.tolist(),
            "atlas_grid_size": list(geometry.annotation.shape[::-1]),
            "image_axes": "native", "atlas_mirror_lr": False,
        },
    )


def digest(value: object) -> str:
    """SHA-256 of a JSON value (sorted keys; a value JSON cannot hold is
    hashed as its text): the fingerprints' and call keys' hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def correction_fingerprint(state: Stack, ctx: Workspace, section_id: str) -> str:
    """Identify the source image, placement and rendering settings for a correction."""
    section = state.by_id(section_id)
    if section is None:
        raise ValueError(f"Unknown section: {section_id}")
    source = Path(ctx.image_path(section.id)).resolve()
    stat = source.stat()
    nonlinear = dict(ctx.settings.trace_settings)
    return digest({
        "source": str(source), "source_size": stat.st_size, "source_mtime": stat.st_mtime_ns,
        "section_id": section.id, "atlas": state.atlas, "plane": state.plane,
        "position_mm": section.position_mm,
        "angles": angles_to_json(section.angles),
        "flip": section.flip, "rotation_deg": section.rotation_deg,
        "transform": affine_fields(section.transform, (
            "params", "calibration", "orientation", "stale",
        )),
        "preprocess": ctx.settings.preprocess,
        "pixel_size_um": ctx.settings.supplied_pixel_size,
        "nonlinear": nonlinear,
    })
