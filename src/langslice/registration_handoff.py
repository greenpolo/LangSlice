"""Host-side bridge from a written linear placement to nonlinear inputs.

The sibling methods remain independent. This adapter only prepares an image and
geometry; it neither generates an image nor changes the linear checkpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.affine import denormalized_affine
from langslice.linear.render import (
    PREVIEW_LONG_EDGE,
    canvas_geometry,
    canvas_um_per_px,
    render_cache_key,
    render_slice,
)
from langslice.space import Plane

if TYPE_CHECKING:
    from langslice.linear.engine import EngineContext
    from langslice.linear.state import StackState
    from langslice.nonlinear.image_gen_registration import RegistrationCandidate


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


def prepare_linear_registration(
    state: StackState,
    ctx: EngineContext,
    section_id: str,
    *,
    long_edge: int = 2048,
) -> LinearRegistrationInput:
    """Prepare a supplied affine placement, preserving shear and physical scale.

    Requires a written position, invertible affine and recoverable calibration.
    Legacy records do not retain an orientation snapshot; when supplied, an
    ``orientation`` dictionary or ``stale`` flag is checked before using a fit.
    Missing calibration is never replaced with a new silhouette estimate.
    """
    if isinstance(long_edge, bool) or not isinstance(long_edge, int) or long_edge <= 0:
        raise ValueError("long_edge must be a positive integer")
    record = state.by_id(section_id)
    if record is None:
        raise ValueError(f"Unknown section: {section_id}")
    if record.position_mm is None or not np.isfinite(record.position_mm):
        raise ValueError("A finite written position is required")
    transform = record.transform
    if not transform:
        raise ValueError("A written affine transform is required")
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
    if not np.isfinite([state.pitch_deg, state.yaw_deg]).all():
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
    matrix = np.vstack([denormalized_affine(params, image.size), [0.0, 0.0, 1.0]])
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("The written affine is singular") from exc
    if not np.isfinite(inverse).all():
        raise ValueError("The written affine cannot be inverted to finite coordinates")

    um_per_px, source = canvas_um_per_px(ctx, record, long_edge=long_edge, frame=False)
    if um_per_px is None:
        calibration = transform.get("calibration") or {}
        try:
            preview_um_per_px = float(calibration["section_um_per_px"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("The written affine has no recoverable calibration") from exc
        render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE, frame=False)
        preview_key = render_cache_key(ctx, record, long_edge=PREVIEW_LONG_EDGE, frame=False)
        target_key = render_cache_key(ctx, record, long_edge=long_edge, frame=False)
        um_per_px = preview_um_per_px * ctx.render_scale[target_key] / ctx.render_scale[preview_key]
        source = str(calibration.get("source", "stored"))
    if not np.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError("Section calibration must be finite and positive")

    plane = cast(Plane, state.plane)
    geometry = canvas_geometry(
        image.size, um_per_px, ctx.atlas, record.position_mm, plane,
        state.pitch_deg, state.yaw_deg,
    )
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
    return LinearRegistrationInput(
        image=image.copy(), atlas_to_slice=atlas_to_slice, atlas_name=state.atlas,
        position_mm=float(record.position_mm), plane=plane,
        pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
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


def run_linear_registration(
    state: StackState,
    ctx: EngineContext,
    section_id: str,
    *,
    long_edge: int = 2048,
    **options: Any,
) -> RegistrationCandidate:
    """Run default nonlinear registration from this section's supplied placement.

    Provider, output and generation options may be passed through. Geometry is
    owned by the handoff, and attempts to override it are refused. Returned
    coordinates refer to the oriented rendered image, not the acquisition TIFF.
    The linear state is never changed by this operation.
    """
    owned = {
        "image", "atlas_name", "position_mm", "plane", "pitch_deg", "yaw_deg",
        "initial_atlas_to_slice", "initial_alignment_source", "image_axes", "atlas_mirror_lr",
    }
    conflicts = owned.intersection(options)
    if conflicts:
        raise ValueError("Handoff geometry cannot be overridden: " + ", ".join(sorted(conflicts)))
    prepared = prepare_linear_registration(state, ctx, section_id, long_edge=long_edge)
    from langslice.nonlinear.image_gen_registration import generate_registration_candidate

    candidate = generate_registration_candidate(
        prepared.image, atlas_name=prepared.atlas_name, position_mm=prepared.position_mm,
        plane=prepared.plane, pitch_deg=prepared.pitch_deg, yaw_deg=prepared.yaw_deg,
        initial_atlas_to_slice=prepared.atlas_to_slice,
        initial_alignment_source="linear_agent", **options,
    )
    candidate.metadata["linear_handoff"] = prepared.metadata
    return candidate
