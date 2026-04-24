"""Registration runtime orchestrating agent correspondences and deterministic solving."""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import os
from collections.abc import Callable
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from langslice_harness.agent_trace import image_part_from_pil, json_part, runtime_event
from langslice_harness.atlas import get_composite_slice, load_atlas
from langslice_harness.harness.registration.image_gen_registration import (
    generate_registration_candidate,
)
from langslice_harness.harness.registration.runner import run_registration_review_session
from langslice_harness.harness.registration.types import (
    RegistrationCandidate,
    candidate_to_registration_result,
)
from langslice_harness.registration.types import (
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


def _run_registration_review_session_sync(
    *,
    image: Image.Image,
    atlas_name: str,
    position_mm: float,
    image_provider: str,
    image_model: str | None,
    review_model: str | object | None,
    pipeline_review_model: str | None,
    openai_image_route: str,
    max_candidates: int,
    debug_dir: str | None,
    on_progress: Callable[[str], None] | None,
    on_trace: Callable[[dict[str, object]], None] | None,
) -> RegistrationCandidate:
    candidate_result = run_registration_review_session(
        image=image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        image_provider=image_provider,
        image_model=image_model,
        model=review_model,
        pipeline_review_model=pipeline_review_model,
        openai_image_route=openai_image_route,
        max_candidates=max_candidates,
        debug_dir=debug_dir,
        on_progress=on_progress,
        on_trace=on_trace,
    )
    if inspect.isawaitable(candidate_result):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(candidate_result)
        candidate_result.close()
        raise RegistrationFailure(
            "Agentic registration cannot run inside an active event loop."
        )
    return candidate_result


def _run_dense_registration(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    selected_mode: str,
    atlas_image: Image.Image,
    debug_dir: str | None,
    on_correspondences: Callable[[list[RegistrationCorrespondence]], None] | None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None,
    on_progress: Callable[[str], None] | None,
    on_trace: Callable[[dict[str, object]], None] | None,
    provider: str,
    image_provider: str | None,
    image_model: str | None,
    openai_image_route: str,
    review_model: str | object | None,
    max_candidates: int,
) -> RegistrationResult:
    dense_debug_root = _dense_registration_debug_root(atlas_name, debug_dir)
    runtime_debug_dir = str(dense_debug_root / "registration") if dense_debug_root else None
    effective_provider = image_provider or provider
    candidate_review_model = review_model if isinstance(review_model, str) else None

    if selected_mode == "agentic":
        _progress(on_progress, "Image-gen registration: running review session...")
        candidate = _run_registration_review_session_sync(
            image=image,
            atlas_name=atlas_name,
            position_mm=position_mm,
            image_provider=effective_provider,
            image_model=image_model,
            review_model=review_model,
            pipeline_review_model=candidate_review_model,
            openai_image_route=openai_image_route,
            max_candidates=max_candidates,
            debug_dir=str(dense_debug_root) if dense_debug_root is not None else None,
            on_progress=on_progress,
            on_trace=on_trace,
        )
    else:
        _progress(on_progress, "Image-gen registration: generating registration candidate...")
        candidate = generate_registration_candidate(
            image,
            atlas_name=atlas_name,
            position_mm=position_mm,
            provider=effective_provider,
            image_model=image_model,
            debug_dir=str(dense_debug_root) if dense_debug_root is not None else None,
            on_progress=on_progress,
            on_trace=on_trace,
            openai_image_route=openai_image_route,
            review_model=candidate_review_model,
        )

    if on_correspondences is not None:
        on_correspondences([])
    result = candidate_to_registration_result(candidate, image.size, debug_dir=runtime_debug_dir)
    session = result.annotation_session
    if session is None:
        raise RegistrationFailure(
            "Dense registration candidate did not produce an annotation session."
        )
    if on_annotation_session is not None:
        on_annotation_session(session)

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
            summary=f"Image-gen registration accepted with {len(markers)} markers",
            parts=[
                image_part_from_pil(
                    candidate.generated_segmentation,
                    label="Generated segmentation",
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
    registration_mode: str = "direct",
    on_correspondences: Callable[[list[RegistrationCorrespondence]], None] | None = None,
    on_annotation_session: Callable[[RegistrationAnnotationSession], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
    debug_dir: str | None = None,
    provider: str = "google",
    image_provider: str | None = None,
    image_model: str | None = None,
    openai_image_route: str = "images",
    review_model: str | object | None = None,
    max_candidates: int = 3,
) -> RegistrationResult:
    """Run image-gen registration and return affine + nonlinear results."""
    atlas = load_atlas(atlas_name)
    atlas_image = get_composite_slice(atlas, position_mm)
    selected_mode = str(registration_mode).strip().lower() or "direct"

    if selected_mode not in {"direct", "agentic"}:
        raise RegistrationFailure(
            f"Unsupported registration_mode {registration_mode!r}; expected 'direct' or 'agentic'."
        )

    return _run_dense_registration(
        image,
        atlas_name=atlas_name,
        position_mm=position_mm,
        selected_mode=selected_mode,
        atlas_image=atlas_image,
        debug_dir=debug_dir,
        on_correspondences=on_correspondences,
        on_annotation_session=on_annotation_session,
        on_progress=on_progress,
        on_trace=on_trace,
        provider=provider,
        image_provider=image_provider,
        image_model=image_model,
        openai_image_route=openai_image_route,
        review_model=review_model,
        max_candidates=max_candidates,
    )
