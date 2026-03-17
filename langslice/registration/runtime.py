"""Registration runtime orchestrating agent correspondences and deterministic solving."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable

from PIL import Image, ImageDraw

from langslice.atlas import get_composite_slice, get_reference_slice, load_atlas
from langslice.registration.agents import estimate_registration_correspondences
from langslice.registration.solver import (
    fit_affine_from_correspondences,
    fit_tps_from_correspondences,
)
from langslice.registration.types import RegistrationCorrespondence, RegistrationResult

logger = logging.getLogger(__name__)


class RegistrationFailure(RuntimeError):
    """Raised when registration runtime cannot produce a usable result."""


def _progress(on_progress: Callable[[str], None] | None, message: str) -> None:
    if on_progress:
        on_progress(message)
    logger.info(message)


def _save_image(path: Path, image: Image.Image) -> None:
    image.save(path)


def _draw_landmarks(
    image: Image.Image, points: list[tuple[float, float]], labels: list[str]
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for (x, y), label in zip(points, labels):
        r = 6
        draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 80, 80), width=2)
        draw.text((x + 8, y + 4), label, fill=(255, 220, 120))
    return annotated


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
    _save_image(
        run_dir / "slice_landmarks.png",
        _draw_landmarks(
            slice_image,
            [c.slice_xy for c in result.accepted_correspondences],
            [c.label for c in result.accepted_correspondences],
        ),
    )
    _save_image(
        run_dir / "atlas_landmarks.png",
        _draw_landmarks(
            atlas_image,
            [c.atlas_xy for c in result.accepted_correspondences],
            [c.label for c in result.accepted_correspondences],
        ),
    )
    payload = {
        "qc_state": result.qc_state,
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
        "rejected_correspondences": [
            {**{k: (asdict(v) if hasattr(v, "slice_xy") else v) for k, v in item.items()}}
            for item in result.rejected_correspondences
        ],
    }
    (run_dir / "registration.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def estimate_registration(
    image: Image.Image,
    *,
    atlas_name: str,
    position_mm: float,
    pixel_size_um: float | None = None,
    target_landmark_count: int = 12,
    show_atlas_borders: bool = True,
    on_correspondences: Callable[[list[RegistrationCorrespondence]], None] | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> RegistrationResult:
    """Run the new registration runtime and return affine + nonlinear results."""
    _ = pixel_size_um
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
        show_atlas_borders=show_atlas_borders,
        on_progress=on_progress,
    )
    if len(correspondences) < 3:
        raise RegistrationFailure(
            f"Need at least 3 correspondence pairs to fit registration, got {len(correspondences)}"
        )
    if on_correspondences is not None:
        on_correspondences(correspondences)
    accepted = list(correspondences)
    rejected: list[dict[str, object]] = []
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
    qc_state = "accepted"

    debug_dir: str | None = None
    root = os.environ.get("LANGSLICE_VLM_DEBUG_DIR")
    if root:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = Path(root) / f"{timestamp}_{atlas_name}_registration"
        debug_dir = str(run_dir)
        _write_debug_artifacts(
            run_dir,
            slice_image=image,
            atlas_image=atlas_image,
            result=RegistrationResult(
                correspondences=correspondences,
                accepted_correspondences=accepted,
                rejected_correspondences=rejected,
                affine_result=affine_result,
                nonlinear_result=nonlinear_result,
                qc_state=qc_state,
                debug_dir=debug_dir,
            ),
        )

    _progress(on_progress, f"Registration runtime completed with state={qc_state}")
    return RegistrationResult(
        correspondences=correspondences,
        accepted_correspondences=accepted,
        rejected_correspondences=rejected,
        affine_result=affine_result,
        nonlinear_result=nonlinear_result,
        qc_state=qc_state,
        debug_dir=debug_dir,
    )
