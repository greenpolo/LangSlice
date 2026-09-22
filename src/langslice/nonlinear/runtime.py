"""Registration runtime orchestrating agent correspondences and deterministic solving."""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from langslice.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice.atlas import get_composite_slice, load_atlas
from langslice.nonlinear.image_gen_registration import (
    generate_registration_candidate,
)
from langslice.nonlinear.types import (
    Deformation,
    RegistrationResult,
    annotation_session_to_dict,
    candidate_to_registration_result,
    render_landmark_annotations,
)
from langslice.space import Plane

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
    # The dense image-gen registration path always populates
    # annotation_session (RegistrationCandidate.annotation_session is
    # required, non-Optional) — no fallback construction needed.
    assert result.annotation_session is not None
    session = result.annotation_session
    _save_image(
        run_dir / "slice_markers.png",
        render_landmark_annotations(slice_image, session.slice_annotations),
    )
    _save_image(
        run_dir / "atlas_markers.png",
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


def _dense_registration_debug_root(atlas_name: str, debug_dir: str | None) -> Path | None:
    root = os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    if debug_dir:
        return Path(debug_dir)
    if root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path(root) / f"{timestamp}_{atlas_name}_registration"
    return None


def _run_dense_registration(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane,
    atlas_image: Image.Image,
    debug_dir: str | None,
    on_progress: Callable[[str], None] | None,
    on_trace: Callable[[dict[str, object]], None] | None,
    provider: str,
    image_provider: str | None,
    image_model: str | None,
    openai_image_route: str,
    review_model: str | object | None,
    image_axes: str | None,
    canvas_pad: float,
    pitch_deg: float,
    yaw_deg: float,
    deformation: Deformation,
    passes: int,
    initial_atlas_to_slice: Sequence[Sequence[float]] | None,
    initial_alignment_source: str,
    atlas_mirror_lr: bool,
) -> RegistrationResult:
    dense_debug_root = _dense_registration_debug_root(atlas_name, debug_dir)
    runtime_debug_dir = str(dense_debug_root / "registration") if dense_debug_root else None
    effective_provider = image_provider or provider
    candidate_review_model = review_model if isinstance(review_model, str) else None

    _progress(on_progress, "Image-gen registration: generating registration candidate...")
    candidate = generate_registration_candidate(
        image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        plane=plane,
        provider=effective_provider,
        image_model=image_model,
        image_axes=image_axes,
        canvas_pad=canvas_pad,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        deformation=deformation,
        passes=passes,
        initial_atlas_to_slice=initial_atlas_to_slice,
        initial_alignment_source=initial_alignment_source,
        atlas_mirror_lr=atlas_mirror_lr,
        debug_dir=str(dense_debug_root) if dense_debug_root is not None else None,
        on_progress=on_progress,
        on_trace=on_trace,
        openai_image_route=openai_image_route,
        review_model=candidate_review_model,
    )

    result = candidate_to_registration_result(candidate, image.size, debug_dir=runtime_debug_dir)
    session = result.annotation_session
    if session is None:
        raise RegistrationFailure(
            "Dense registration candidate did not produce an annotation session."
        )

    if dense_debug_root is not None:
        assert runtime_debug_dir is not None
        _write_debug_artifacts(
            Path(runtime_debug_dir),
            slice_image=image,
            atlas_image=atlas_image,
            result=result,
        )

    markers = session.metadata.get("visualign_markers", [])
    registration_dir = (
        Path(runtime_debug_dir) / candidate.candidate_id if runtime_debug_dir else None
    )
    _emit_trace(
        on_trace,
        runtime_event(
            stage="registration",
            title="Registration solve completed",
            summary=f"Image-gen registration completed with {len(markers)} markers",
            parts=[
                image_part_from_pil(
                    candidate.generated_segmentation,
                    label="Raw border-correction reply",
                    image_format="PNG",
                    path=str(registration_dir / "generated_segmentation.png")
                    if registration_dir is not None
                    else None,
                ),
                image_part_from_pil(
                    candidate.warped_atlas,
                    label="Warped atlas",
                    image_format="PNG",
                    path=str(registration_dir / "warped_atlas.png")
                    if registration_dir is not None
                    else None,
                ),
                image_part_from_pil(
                    candidate.warped_border_overlay,
                    label="Warped border overlay",
                    image_format="PNG",
                    path=str(registration_dir / "warped_border_overlay.png")
                    if registration_dir is not None
                    else None,
                ),
                json_part(
                    {
                        "accepted_pairs": 0,
                        "n_markers": len(markers),
                        "candidate_id": candidate.candidate_id,
                        "visualign_markers": markers,
                        "candidate_metadata": session.metadata.get("candidate_metadata", {}),
                    },
                    label="Registration result",
                ),
            ],
            metadata={
                "accepted_pairs": 0,
                "n_markers": len(markers),
                "candidate_id": candidate.candidate_id,
                "candidate_metadata": session.metadata.get("candidate_metadata", {}),
                "method": "image_gen_registration",
            },
        ),
    )
    _progress(on_progress, "Registration runtime completed")
    return result


def estimate_registration(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    plane: Plane = "coronal",
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    provider: str = "google",
    image_provider: str | None = None,
    image_model: str | None = None,
    openai_image_route: str = "images",
    review_model: str | object | None = None,
    image_axes: str | None = None,
    canvas_pad: float = 0.0,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    deformation: Deformation = "bspline",
    passes: int = 1,
    initial_atlas_to_slice: Sequence[Sequence[float]] | None = None,
    initial_alignment_source: str = "supplied",
    atlas_mirror_lr: bool = False,
) -> RegistrationResult:
    """Run border-based registration and return affine + nonlinear results.

    A supplied ``initial_atlas_to_slice`` selects route "supplied" (one model
    call moves its drawn boundaries onto the tissue). Without it, route
    "atlas" fits a local silhouette placement and asks the model to draw
    boundaries from nothing against the outlined atlas template, with an
    optional second corrective call (``passes=2``).

    ``pitch_deg``/``yaw_deg`` are the block's cutting angles: every atlas
    render is resliced on that oblique plane instead of taken flat.
    ``deformation`` picks the Elastix stages, or ``"none"`` for no fit at all
    (identity residual; the CLI default). ``provider="none"`` calls no
    model: it retains a supplied placement, or fits a silhouette placement,
    without fitting a residual deformation.
    """
    atlas = load_atlas(atlas_name)
    atlas_image = get_composite_slice(atlas, position_mm, plane=plane)

    result = _run_dense_registration(
        image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        plane=plane,
        atlas_image=atlas_image,
        debug_dir=debug_dir,
        on_progress=on_progress,
        on_trace=on_trace,
        provider=provider,
        image_provider=image_provider,
        image_model=image_model,
        openai_image_route=openai_image_route,
        review_model=review_model,
        image_axes=image_axes,
        canvas_pad=canvas_pad,
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        deformation=deformation,
        passes=passes,
        initial_atlas_to_slice=initial_atlas_to_slice,
        initial_alignment_source=initial_alignment_source,
        atlas_mirror_lr=atlas_mirror_lr,
    )
    _progress(
        on_progress,
        "Registration outputs derived: "
        f"rot={result.affine_result.rotation_deg:.2f} deg, "
        f"scale=({result.affine_result.scale[0]:.3f}, {result.affine_result.scale[1]:.3f}), "
        f"shear={result.affine_result.shear:.3f}",
    )
    return result
