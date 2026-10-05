"""The geometry an image correction or a deformable fit starts from.

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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core.affine import denormalized_affine
from langslice.core.canvas import CanvasGeometry, canvas_geometry
from langslice.core.sections import (
    PREVIEW_LONG_EDGE,
    canvas_um_per_px,
    render_cache_key,
    render_slice,
)
from langslice.core.space import Plane
from langslice.core.workspace import Workspace

if TYPE_CHECKING:
    from langslice.core.state import StackState


#: Said when a section has no written in-plane transform and the job cannot
#: write one (Linear off): what the caller does instead. A missing transform
#: is the identity to the maps (``core.maps``), never to the nonlinear step.
NO_TRANSFORM_LINEAR_OFF = (
    "A written affine transform is required, and Linear is off for this job, so "
    "nothing here writes one. Supply the in-plane transforms with the job "
    "(inputs.transforms; --transforms on `langslice job FOLDER init`), or switch "
    "Linear on (task transform) and run fit_affine on the section first."
)


def missing_transform_message(spec: Any) -> str:
    """The refusal for a section with no written in-plane transform: with
    Linear (task ``transform``) off, :data:`NO_TRANSFORM_LINEAR_OFF`, what to
    do instead; with it on, the plain requirement (``fit_affine`` and
    ``adjust_transforms`` are the job's own tools)."""
    has = getattr(spec, "has", None)
    if callable(has) and not has("transform"):
        return NO_TRANSFORM_LINEAR_OFF
    return "A written affine transform is required"


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
    (the plane :func:`langslice.core.atlas.render.annotation_slice` draws at
    *position_mm*, *plane* and the cutting angles) onto pixel centres of the
    oriented, unframed section render of *image_size* that the six stored
    numbers *params* are normalized against, at *um_per_px* (the render's).
    The one geometry path from a stored placement to the atlas: the
    image-correction handoff, the deformable fit grid and the job folder's
    maps and ``registration.json`` (:mod:`langslice.core.maps`) all go
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
    sx, sy = geometry.section_offset
    ax, ay = geometry.atlas_offset
    atlas_to_section_frame = np.array(
        [[geometry.atlas_scale, 0.0, ax - sx],
         [0.0, geometry.atlas_scale, ay - sy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    atlas_to_slice = inverse @ atlas_to_section_frame
    if not np.isfinite(atlas_to_slice).all():
        raise ValueError("Atlas placement produced non-finite coordinates")
    return atlas_to_slice, geometry


def prepare_linear_registration(
    state: StackState,
    ctx: Workspace,
    section_id: str,
    *,
    long_edge: int = 2048,
    transform: dict[str, Any] | None = None,
) -> LinearRegistrationInput:
    """Prepare a supplied affine placement, preserving shear and physical scale.

    Requires a written position and an invertible affine. Legacy records do
    not retain an orientation snapshot; when supplied, an ``orientation``
    dictionary or ``stale`` flag is checked before using a fit. The scale
    is the one every placement picture and ``registration.json`` draw the
    section at (:func:`langslice.core.transform.calibrate` on its working
    frame, carried to *long_edge*), so the six numbers place the atlas here
    exactly where the pictures show it, also for a section without a pixel
    size whose position moved after its transform was written.

    *transform* stands in for the section's written transform (``params``),
    for a fit that starts from a placement it has not written:
    ``fit_affine``'s Elastix method on a section with no transform yet
    starts from the identity.
    """
    if isinstance(long_edge, bool) or not isinstance(long_edge, int) or long_edge <= 0:
        raise ValueError("long_edge must be a positive integer")
    record = state.by_id(section_id)
    if record is None:
        raise ValueError(f"Unknown section: {section_id}")
    if record.position_mm is None or not np.isfinite(record.position_mm):
        raise ValueError("A finite written position is required")
    if transform is None:
        transform = record.transform
    if not transform:
        raise ValueError(missing_transform_message(getattr(ctx, "spec", None)))
    if transform.get("spline"):
        raise ValueError("A linear placement is required; an existing spline cannot be discarded")
    if transform.get("stale"):
        raise ValueError("The written affine is marked stale")
    orientation = transform.get("orientation")
    if orientation is not None:
        expected = {"flip": record.flip, "rotation_deg": record.rotation_deg}
        if not isinstance(orientation, dict) or any(
            key in orientation and orientation[key] != value for key, value in expected.items()
        ):
            raise ValueError("The written affine has a different section orientation")
    if state.atlas != ctx.spec.atlas or state.plane != ctx.spec.plane:
        raise ValueError("State and rendering context disagree about the atlas or plane")
    if state.plane not in ("coronal", "sagittal", "horizontal"):
        raise ValueError("Unsupported section plane")
    if not np.isfinite([record.pitch_deg, record.yaw_deg]).all():
        raise ValueError("Cutting angles must be finite")
    try:
        params = np.asarray(transform["params"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("The written affine must contain six finite parameters") from exc
    if params.shape != (6,) or not np.isfinite(params).all():
        raise ValueError("The written affine must contain six finite parameters")

    image = render_slice(ctx, record, long_edge=long_edge, frame=False)
    with Image.open(ctx.image_path(record.id)) as source_image:
        original_size = list(source_image.size)

    um_per_px, source = canvas_um_per_px(ctx, record, long_edge=long_edge, frame=False)
    if um_per_px is None:
        from langslice.core.transform import calibrate

        working = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE, frame=False)
        preview_um_per_px, source = calibrate(state, ctx, record, working)
        preview_key = render_cache_key(ctx, record, long_edge=PREVIEW_LONG_EDGE, frame=False)
        target_key = render_cache_key(ctx, record, long_edge=long_edge, frame=False)
        um_per_px = preview_um_per_px * ctx.render_scale[target_key] / ctx.render_scale[preview_key]
    if not np.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError("Section calibration must be finite and positive")

    plane = cast(Plane, state.plane)
    atlas_to_slice, geometry = linear_placement_matrix(
        image.size, um_per_px, ctx.atlas, float(record.position_mm), plane,
        record.pitch_deg, record.yaw_deg, params)
    return LinearRegistrationInput(
        image=image.copy(), atlas_to_slice=atlas_to_slice, atlas_name=state.atlas,
        position_mm=float(record.position_mm), plane=plane,
        pitch_deg=record.pitch_deg, yaw_deg=record.yaw_deg,
        metadata={
            "source": "linear_state", "section_id": record.id,
            "orientation": {"rotation_deg": record.rotation_deg, "flip": record.flip},
            "orientation_snapshot_checked": orientation is not None,
            "preprocess": ctx.spec.preprocess, "image_size": list(image.size),
            "image_frame": "oriented rendered section; not acquisition image pixels",
            "source_image_size": original_size,
            "section_um_per_px": float(um_per_px), "calibration_source": source,
            "linear_params": params.tolist(),
            "atlas_to_slice": atlas_to_slice.tolist(),
            "atlas_grid_size": list(geometry.annotation.shape[::-1]),
            "image_axes": "native", "atlas_mirror_lr": False,
        },
    )


def digest(value: object) -> str:
    """SHA-256 of a JSON value (sorted keys, no NaN): the fingerprints' hash."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def correction_fingerprint(state: StackState, ctx: Workspace, section_id: str) -> str:
    """Identify the source image, placement and rendering settings for a correction."""
    record = state.by_id(section_id)
    if record is None:
        raise ValueError(f"Unknown section: {section_id}")
    source = Path(ctx.image_path(record.id)).resolve()
    stat = source.stat()
    transform = record.transform or {}
    nonlinear = dict(ctx.spec.to_dict().get("nonlinear") or {})
    # The deformable-fit engine never reaches the image call; leaving it in
    # made every trace saved before the field existed (or under another
    # engine choice) stale at an unchanged placement.
    nonlinear.pop("engine", None)
    return digest({
        "source": str(source), "source_size": stat.st_size, "source_mtime": stat.st_mtime_ns,
        "section_id": record.id, "atlas": state.atlas, "plane": state.plane,
        "position_mm": record.position_mm, "angles": record.cutting_angles_deg,
        "flip": record.flip, "rotation_deg": record.rotation_deg,
        "transform": {key: transform.get(key) for key in (
            "params", "calibration", "orientation", "stale", "spline",
        )},
        "preprocess": ctx.spec.preprocess,
        "pixel_size_um": ctx.spec.inputs.get("pixel_size_um"),
        "nonlinear": nonlinear,
    })
