"""The exact interpolating 2D thin-plate spline, in millimetres.

The kernel is r² log(r), as in BigWarp's
``jitk.spline.ThinPlateR2LogRSplineKernelTransform``; :mod:`langslice.core.abba_warp`
fits the deformation's pull-back with it to validate the landmark pairs ABBA
will build. No state or image mutations occur here.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _points(value: ArrayLike) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite N-by-2 [x, y] pairs")
    return points


def _kernel(delta: FloatArray) -> FloatArray:
    squared = np.sum(delta * delta, axis=-1)
    return 0.5 * squared * np.log(np.maximum(squared, 1e-300))


class ThinPlateKernel:
    """An exact interpolating 2D TPS. Arrays use rows of physical [x, y]."""

    def __init__(self, source: ArrayLike, target: ArrayLike, *, max_points: int = 64) -> None:
        source, target = _points(source), _points(target)
        if source.shape != target.shape or not 4 <= len(source) <= max_points:
            raise ValueError(f"Supply 4 to {max_points} corresponding source and target landmarks")
        self.origin = np.mean(source, axis=0)
        self.scale = float(np.max(np.ptp(source, axis=0)))
        if self.scale <= 0:
            raise ValueError("Source landmarks must span a two-dimensional region")
        self.source = (source - self.origin) / self.scale
        self.target = (target - self.origin) / self.scale
        for label, points in (("Source", self.source), ("Target", self.target)):
            distance = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
            np.fill_diagonal(distance, np.inf)
            if np.min(distance) < 1e-6:
                raise ValueError(f"{label} landmarks must be distinct and well separated")
            singular = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
            if singular[-1] < singular[0] * 1e-6:
                raise ValueError(f"{label} landmarks must not be collinear")
        polynomial = np.column_stack((np.ones(len(source)), self.source))
        system = np.block([
            [_kernel(self.source[:, None] - self.source[None, :]), polynomial],
            [polynomial.T, np.zeros((3, 3))],
        ])
        if np.linalg.cond(system) > 1e12:
            raise ValueError("Landmark arrangement is ill-conditioned")
        coefficients = np.linalg.solve(system, np.vstack((self.target, np.zeros((3, 2)))))
        self.weights = coefficients[:-3]
        self.affine = coefficients[-3:]

    def _forward(self, points: FloatArray) -> FloatArray:
        return (_kernel(points[:, None] - self.source[None, :]) @ self.weights
                + points @ self.affine[1:] + self.affine[0])

    def _jacobian(self, points: FloatArray) -> FloatArray:
        delta = points[:, None] - self.source[None, :]
        squared = np.sum(delta * delta, axis=-1)
        gradient = delta * (np.log(np.maximum(squared, 1e-300)) + 1)[..., None]
        return np.einsum("nki,kj->nji", gradient, self.weights) + self.affine[1:].T

    def forward(self, points_mm: ArrayLike) -> FloatArray:
        """Map physical source points to targets, in bounded memory chunks."""
        points = (_points(points_mm) - self.origin) / self.scale
        result = np.empty_like(points)
        for start in range(0, len(points), 4096):
            result[start:start + 4096] = self._forward(points[start:start + 4096])
        return result * self.scale + self.origin

    def jacobian(self, points_mm: ArrayLike) -> FloatArray:
        """Return N-by-2-by-2 derivatives (output coordinate, input coordinate)."""
        points = (_points(points_mm) - self.origin) / self.scale
        result = np.empty((len(points), 2, 2), dtype=np.float64)
        for start in range(0, len(points), 4096):
            result[start:start + 4096] = self._jacobian(points[start:start + 4096])
        return result

    def inverse(self, points_mm: ArrayLike, *, tolerance_mm: float = 1e-6,
                max_iterations: int = 40) -> FloatArray:
        """Damped Newton inversion; raise rather than silently render a failed fit."""
        if not np.isfinite(tolerance_mm) or tolerance_mm <= 0 or max_iterations < 1:
            raise ValueError("Inverse tolerance and iteration count must be positive")
        target = (_points(points_mm) - self.origin) / self.scale
        result = np.empty_like(target)
        try:
            initial_inverse = np.linalg.inv(self.affine[1:])
        except np.linalg.LinAlgError as exc:
            raise ValueError("The affine component is singular") from exc
        tolerance = tolerance_mm / self.scale
        for start in range(0, len(target), 4096):
            goal = target[start:start + 4096]
            current = (goal - self.affine[0]) @ initial_inverse
            for _ in range(max_iterations):
                residual = self._forward(current) - goal
                error = np.linalg.norm(residual, axis=1)
                active = error > tolerance
                if not active.any():
                    break
                jacobian = self._jacobian(current[active])
                determinant = np.linalg.det(jacobian)
                if np.any(determinant <= 1e-8):
                    raise ValueError("Spline inverse encountered a fold or singularity")
                step = np.linalg.solve(jacobian, residual[active, :, None])[..., 0]
                trial = current[active] - step
                for _ in range(16):
                    trial_error = np.linalg.norm(self._forward(trial) - goal[active], axis=1)
                    worse = trial_error >= error[active]
                    if not worse.any():
                        break
                    step[worse] *= 0.5
                    trial[worse] = current[active][worse] - step[worse]
                current[active] = trial
            if np.any(np.linalg.norm(self._forward(current) - goal, axis=1) > tolerance):
                raise ValueError("Spline inverse did not converge")
            result[start:start + len(goal)] = current
        return result * self.scale + self.origin
