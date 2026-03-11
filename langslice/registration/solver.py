"""Deterministic registration solvers for affine and TPS fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator

from langslice.registration.types import (
    AffineResult,
    NonlinearResult,
    RegistrationCorrespondence,
    is_valid_affine_matrix,
)


MIN_CORRESPONDENCES = 6
DEFAULT_TPS_SMOOTHING = 1.0


@dataclass
class VettingResult:
    accepted: list[RegistrationCorrespondence]
    rejected: list[dict[str, Any]]
    qc_metrics: dict[str, float]


def _confidence_weight(confidence: str) -> float:
    normalized = confidence.strip().lower()
    if normalized == "high":
        return 1.0
    if normalized == "medium":
        return 0.65
    if normalized == "low":
        return 0.35
    return 0.5


def _points_from_correspondences(
    correspondences: Sequence[RegistrationCorrespondence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atlas_points = np.asarray([c.atlas_xy for c in correspondences], dtype=np.float64)
    slice_points = np.asarray([c.slice_xy for c in correspondences], dtype=np.float64)
    weights = np.asarray([_confidence_weight(c.confidence) for c in correspondences], dtype=np.float64)
    return atlas_points, slice_points, weights


def vet_correspondences(
    correspondences: Sequence[RegistrationCorrespondence],
    *,
    slice_size: tuple[int, int],
    min_count: int = MIN_CORRESPONDENCES,
) -> VettingResult:
    rejected: list[dict[str, Any]] = []
    accepted: list[RegistrationCorrespondence] = []
    seen: set[tuple[int, int, int, int, str]] = set()
    for corr in correspondences:
        key = (
            int(round(corr.slice_xy[0])),
            int(round(corr.slice_xy[1])),
            int(round(corr.atlas_xy[0])),
            int(round(corr.atlas_xy[1])),
            corr.label.strip().lower(),
        )
        if key in seen:
            rejected.append({"correspondence": corr, "reason": "duplicate"})
            continue
        seen.add(key)
        accepted.append(corr)

    high_confidence = [c for c in accepted if c.confidence.strip().lower() == "high"]
    if len(high_confidence) >= min_count:
        dropped = [c for c in accepted if c not in high_confidence]
        for corr in dropped:
            rejected.append({"correspondence": corr, "reason": "pruned_low_confidence"})
        accepted = high_confidence

    if len(accepted) < min_count:
        raise ValueError(f"Need at least {min_count} correspondences, got {len(accepted)}")

    _, slice_points, _ = _points_from_correspondences(accepted)
    width, height = slice_size
    bbox_w = float(slice_points[:, 0].max() - slice_points[:, 0].min()) if len(slice_points) else 0.0
    bbox_h = float(slice_points[:, 1].max() - slice_points[:, 1].min()) if len(slice_points) else 0.0
    coverage = (bbox_w * bbox_h) / max(float(width * height), 1.0)
    if coverage < 0.08:
        raise ValueError(f"Landmark spread too small (coverage={coverage:.4f})")

    qc_metrics = {
        "accepted_count": float(len(accepted)),
        "rejected_count": float(len(rejected)),
        "slice_bbox_coverage": float(coverage),
    }
    return VettingResult(accepted=accepted, rejected=rejected, qc_metrics=qc_metrics)


def fit_affine_from_correspondences(
    correspondences: Sequence[RegistrationCorrespondence],
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
    backend: str,
    reasoning: str,
) -> tuple[AffineResult, np.ndarray]:
    atlas_points, slice_points, weights = _points_from_correspondences(correspondences)
    design = np.concatenate(
        [atlas_points, np.ones((atlas_points.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    weighted_design = design * weights[:, None]
    weighted_slice = slice_points * weights[:, None]
    params, _, _, _ = np.linalg.lstsq(weighted_design, weighted_slice, rcond=None)
    matrix = np.array(
        [
            [params[0, 0], params[1, 0], params[2, 0]],
            [params[0, 1], params[1, 1], params[2, 1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if not is_valid_affine_matrix(matrix):
        raise ValueError("Computed affine matrix failed sanity checks")

    predicted = (design @ params)
    residuals = np.linalg.norm(predicted - slice_points, axis=1)
    result = AffineResult(
        matrix=matrix,
        source_size=source_size,
        output_size=output_size,
        backend=backend,
        reasoning=reasoning,
        provenance={
            "landmark_count": len(correspondences),
            "mean_residual_px": float(residuals.mean()),
            "max_residual_px": float(residuals.max()),
        },
    )
    return result, residuals


def _jacobian_metrics(
    interpolator: RBFInterpolator,
    *,
    width: int,
    height: int,
    grid_size: int = 24,
    eps: float = 1.0,
) -> dict[str, float]:
    xs = np.linspace(0.0, max(width - 1, 1), grid_size)
    ys = np.linspace(0.0, max(height - 1, 1), grid_size)
    neg_count = 0
    min_det = math.inf
    for y in ys:
        for x in xs:
            p = np.array([[x, y]], dtype=np.float64)
            fx1 = interpolator(p + np.array([[eps, 0.0]], dtype=np.float64))[0]
            fx0 = interpolator(p - np.array([[eps, 0.0]], dtype=np.float64))[0]
            fy1 = interpolator(p + np.array([[0.0, eps]], dtype=np.float64))[0]
            fy0 = interpolator(p - np.array([[0.0, eps]], dtype=np.float64))[0]
            dfdx = (fx1 - fx0) / (2.0 * eps)
            dfdy = (fy1 - fy0) / (2.0 * eps)
            jac = np.array([[dfdx[0], dfdy[0]], [dfdx[1], dfdy[1]]], dtype=np.float64)
            det = float(np.linalg.det(jac))
            min_det = min(min_det, det)
            if det < 0.0:
                neg_count += 1

    total = float(grid_size * grid_size)
    return {
        "min_jacobian_det": float(min_det),
        "negative_jacobian_fraction": float(neg_count / total),
    }


def fit_tps_from_correspondences(
    correspondences: Sequence[RegistrationCorrespondence],
    *,
    output_size: tuple[int, int],
    smoothing: float = DEFAULT_TPS_SMOOTHING,
    reasoning: str,
) -> NonlinearResult:
    atlas_points, slice_points, _ = _points_from_correspondences(correspondences)
    interpolator = RBFInterpolator(
        atlas_points,
        slice_points,
        kernel="thin_plate_spline",
        smoothing=smoothing,
    )
    predicted = interpolator(atlas_points)
    residuals = np.linalg.norm(predicted - slice_points, axis=1)
    jacobian = _jacobian_metrics(interpolator, width=output_size[0], height=output_size[1])
    qc_metrics = {
        "mean_landmark_residual_px": float(residuals.mean()),
        "max_landmark_residual_px": float(residuals.max()),
        "smoothing": float(smoothing),
        **jacobian,
    }
    return NonlinearResult(
        atlas_points=atlas_points,
        slice_points=slice_points,
        smoothing=smoothing,
        backend="tps",
        reasoning=reasoning,
        output_size=output_size,
        qc_metrics=qc_metrics,
        provenance={"landmark_count": len(correspondences)},
    )
