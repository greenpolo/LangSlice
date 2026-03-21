"""Deterministic registration solvers for affine and TPS fitting."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from scipy.interpolate import RBFInterpolator

from langslice.registration.types import (
    AffineResult,
    NonlinearResult,
    RegistrationCorrespondence,
    is_valid_affine_matrix,
)


DEFAULT_TPS_SMOOTHING = 1.0


def _points_from_correspondences(
    correspondences: Sequence[RegistrationCorrespondence],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    atlas_points = np.asarray([c.atlas_xy for c in correspondences], dtype=np.float64)
    slice_points = np.asarray([c.slice_xy for c in correspondences], dtype=np.float64)
    weights = np.ones(len(correspondences), dtype=np.float64)
    return atlas_points, slice_points, weights


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

    predicted = design @ params
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
