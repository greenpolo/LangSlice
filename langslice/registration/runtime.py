"""Registration runtime orchestrating agent correspondences and deterministic solving."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import get_composite_slice, get_reference_slice, load_atlas
from langslice.registration.agents import estimate_registration_correspondences
from langslice.registration.solver import (
    fit_affine_from_correspondences,
    fit_tps_from_correspondences,
)
from langslice.registration.types import (
    RegistrationAnnotationSession,
    RegistrationCorrespondence,
    RegistrationResult,
    annotation_session_to_dict,
    build_annotation_session_from_correspondences,
    render_landmark_annotations,
)

logger = logging.getLogger(__name__)


class RegistrationFailure(RuntimeError):
    """Raised when registration runtime cannot produce a usable result."""


def _progress(on_progress: Callable[[str], None] | None, message: str) -> None:
    if on_progress:
        on_progress(message)
    logger.info(message)


def _emit_trace(
    on_trace: Callable[[dict[str, object]], None] | None,
    event: dict[str, object],
) -> None:
    if on_trace:
        on_trace(event)


def _save_image(path: Path, image: Image.Image) -> None:
    image.save(path)


def _write_debug_artifacts(
    run_dir: Path,
    *,
    slice_image: Image.Image,
    atlas_image: Image.Image,
    result: RegistrationResult,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_image(run_dir / "slice.png", slice_image)
    _save_image(run_dir / "atlas.png", atlas_image)
    session = result.annotation_session or build_annotation_session_from_correspondences(
        result.accepted_correspondences
    )
    _save_image(
        run_dir / "slice_landmarks.png",
        render_landmark_annotations(slice_image, session.slice_annotations),
    )
    _save_image(
        run_dir / "atlas_landmarks.png",
        render_landmark_annotations(atlas_image, session.atlas_annotations),
    )
    payload = {
        "affine": {
            "backend": result.affine_result.backend,
            "reasoning": result.affine_result.reasoning,
            "matrix": result.affine_result.matrix.tolist(),
            "provenance": result.affine_result.provenance,
        },
        "nonlinear": {
            "backend": result.nonlinear_result.backend,
            "reasoning": result.nonlinear_result.reasoning,
            "qc_metrics": result.nonlinear_result.qc_metrics,
            "provenance": result.nonlinear_result.provenance,
            "smoothing": result.nonlinear_result.smoothing,
        },
        "accepted_correspondences": [asdict(c) for c in result.accepted_correspondences],
        "annotation_session": annotation_session_to_dict(result.annotation_session),
    }
    (run_dir / "registration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def estimate_registration(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    target_landmark_count: int = 12,
    workflow: str = "multimodal_tool_loop",
    show_atlas_borders: bool = True,
    on_correspondences: Callable[[list[RegistrationCorrespondence]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    enable_code_execution: bool | None = None,
    tool_loop_max_steps: int | None = None,
    border_count: int | None = None,
    interior_count: int | None = None,
) -> RegistrationResult:
    """Run the new registration runtime and return affine + nonlinear results."""
    atlas = load_atlas(atlas_name)
    if show_atlas_borders:
        atlas_image = get_composite_slice(atlas, position_mm)
    else:
        atlas_image = get_reference_slice(atlas, position_mm)
    correspondences = estimate_registration_correspondences(
        image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        target_landmark_count=target_landmark_count,
        workflow=workflow,
        show_atlas_borders=show_atlas_borders,
        on_progress=on_progress,
        on_trace=on_trace,
        on_annotation_session=on_annotation_session,
        debug_dir=debug_dir,
        enable_code_execution=enable_code_execution,
        tool_loop_max_steps=tool_loop_max_steps,
        border_count=border_count,
        interior_count=interior_count,
    )
    if on_correspondences is not None:
        on_correspondences(correspondences)
    if len(correspondences) < 3:
        raise RegistrationFailure(
            f"Need at least 3 correspondence pairs to fit registration, got {len(correspondences)}"
        )
    accepted = list(correspondences)
    annotation_session = build_annotation_session_from_correspondences(
        accepted,
        workflow=workflow,
        target_count=target_landmark_count,
        metadata={
            "atlas_name": atlas_name,
            "position_mm": round(position_mm, 3),
            "show_atlas_borders": bool(show_atlas_borders),
            "workflow": workflow,
        },
    )
    affine_result, residuals = fit_affine_from_correspondences(
        accepted,
        source_size=atlas_image.size,
        output_size=image.size,
        backend="landmark_affine",
        reasoning="Affine derived from registration correspondences.",
    )
    affine_result.provenance["transform_direction"] = "atlas_to_slice"
    _ = residuals
    nonlinear_result = fit_tps_from_correspondences(
        accepted,
        output_size=image.size,
        reasoning="Regularized TPS derived from registration correspondences.",
    )

    output_debug_dir: str | None = None
    root = os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    if debug_dir:
        run_dir = Path(debug_dir) / "registration"
        output_debug_dir = str(run_dir)
        _write_debug_artifacts(
            run_dir,
            slice_image=image,
            atlas_image=atlas_image,
            result=RegistrationResult(
                correspondences=correspondences,
                accepted_correspondences=accepted,
                affine_result=affine_result,
                nonlinear_result=nonlinear_result,
                debug_dir=output_debug_dir,
                annotation_session=annotation_session,
            ),
        )
    elif root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(root) / f"{timestamp}_{atlas_name}_registration"
        output_debug_dir = str(run_dir)
        _write_debug_artifacts(
            run_dir,
            slice_image=image,
            atlas_image=atlas_image,
            result=RegistrationResult(
                correspondences=correspondences,
                accepted_correspondences=accepted,
                affine_result=affine_result,
                nonlinear_result=nonlinear_result,
                debug_dir=output_debug_dir,
                annotation_session=annotation_session,
            ),
        )

    atlas_landmarks = render_landmark_annotations(atlas_image, annotation_session.atlas_annotations)
    slice_landmarks = render_landmark_annotations(image, annotation_session.slice_annotations)
    registration_dir = str(Path(output_debug_dir)) if output_debug_dir else None
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Registration solve completed",
            summary=f"Affine accepted with {len(accepted)} landmark pairs",
            parts=[
                image_part_from_pil(
                    atlas_landmarks,
                    label="Atlas landmarks",
                    image_format="PNG",
                    path=str(Path(registration_dir) / "atlas_landmarks.png")
                    if registration_dir
                    else None,
                ),
                image_part_from_pil(
                    slice_landmarks,
                    label="Slice landmarks",
                    image_format="PNG",
                    path=str(Path(registration_dir) / "slice_landmarks.png")
                    if registration_dir
                    else None,
                ),
                json_part(
                    {
                        "accepted_pairs": len(accepted),
                        "affine_backend": affine_result.backend,
                        "rotation_deg": round(affine_result.rotation_deg, 4),
                        "translation_px": affine_result.translation_px,
                        "scale": affine_result.scale,
                        "shear": affine_result.shear,
                    },
                    label="Registration result",
                ),
            ],
            metadata={"accepted_pairs": len(accepted)},
        ),
    )

    _progress(on_progress, "Registration runtime completed")
    return RegistrationResult(
        correspondences=correspondences,
        accepted_correspondences=accepted,
        affine_result=affine_result,
        nonlinear_result=nonlinear_result,
        debug_dir=output_debug_dir,
        annotation_session=annotation_session,
    )
