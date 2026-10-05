"""Reader of a stored landmark Elastix B-spline (``backend="elastix"``).

Read only: nothing writes these payloads any more; checkpoints that carry
one still draw and export through it. The saved affine A maps original
section coordinates into atlas coordinates, and the saved residual pullback
B maps atlas landmarks to A(section landmarks). Resampling therefore uses
A^-1(B(x)); forward points invert that same mapping.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
MIN_RESIDUAL_STRETCH = 0.5
MAX_RESIDUAL_STRETCH = 2.0


def _points(value: ArrayLike | None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        raise ValueError("Coordinates must be finite N-by-2 [x, y] pairs")
    return result


def _extent(value: ArrayLike | None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2,) or not np.isfinite(result).all() or np.any(result <= 0):
        raise ValueError("extent_mm must contain positive finite width and height")
    return result


def _affine(value: ArrayLike | None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (2, 3) or not np.isfinite(result).all():
        raise ValueError("affine_mm must be a finite 2-by-3 matrix")
    if np.linalg.det(result[:, :2]) <= 1e-8:
        raise ValueError("The initial affine must preserve orientation and be nonsingular")
    return result


def _bounds(value: ArrayLike | None) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if (result.shape != (2, 2) or not np.isfinite(result).all()
            or np.any(result[1] <= result[0])):
        raise ValueError("domain_bounds_mm must contain [lower_xy, upper_xy]")
    return result


def _check_landmarks(source: FloatArray, target: FloatArray) -> None:
    if source.shape != target.shape or not 4 <= len(source) <= 64:
        raise ValueError("Supply 4 to 64 corresponding source and target landmarks")
    for name, pts in (("Source", source), ("Target", target)):
        scale = float(np.max(np.ptp(pts, axis=0)))
        singular = np.linalg.svd(pts - pts.mean(axis=0), compute_uv=False)
        if scale <= 0 or singular[-1] < singular[0] * 1e-6:
            raise ValueError(f"{name} landmarks must span a two-dimensional region")
        distances = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        if np.min(distances) < scale * 1e-6:
            raise ValueError(f"{name} landmarks must be distinct and well separated")


def _basis(t: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Cardinal cubic B-spline weights and derivatives for floor(x)-1..+2."""
    weights = np.stack(((1-t)**3, 3*t**3-6*t**2+4,
                        -3*t**3+3*t**2+3*t+1, t**3), axis=-1) / 6
    derivative = np.stack((-3*(1-t)**2, 9*t**2-12*t,
                           -9*t**2+6*t+3, 3*t**2), axis=-1) / 6
    return weights, derivative


class LandmarkElastix:
    """An immutable-in-use serialized residual spline with an explicit affine.

    The vectorized evaluator implements Elastix's cubic coefficient lattice;
    outside its valid support the residual is identity, as in native ITK.
    """

    def __init__(self, payload: Mapping[str, Any]) -> None:
        if payload.get("backend") != "elastix":
            raise ValueError("Expected an Elastix landmark transform")
        self.extent = _extent(payload.get("extent_mm"))
        self.affine = _affine(payload.get("affine_mm"))
        self._affine_inverse = np.linalg.inv(self.affine[:, :2])
        self.source = _points(payload.get("source")) * self.extent
        self.target = _points(payload.get("target")) * self.extent
        _check_landmarks(self.source, self.target)
        self.domain_bounds = _bounds(payload.get("domain_mm"))
        maps = payload.get("parameter_maps")
        if not isinstance(maps, list) or len(maps) != 1 or not isinstance(maps[0], dict):
            raise ValueError("Expected exactly one self-contained residual parameter map")
        pm = maps[0]
        if (pm.get("Transform") != ["BSplineTransform"]
                or pm.get("BSplineTransformSplineOrder", ["3"]) != ["3"]
                or pm.get("GridIndex", ["0", "0"]) != ["0", "0"]
                or pm.get("InitialTransformParameterFileName", ["NoInitialTransform"])
                != ["NoInitialTransform"]):
            raise ValueError("Expected a cubic B-spline without external transform references")
        try:
            size = np.asarray(pm["GridSize"], dtype=np.int64)
            self._origin = np.asarray(pm["GridOrigin"], dtype=np.float64)
            spacing = np.asarray(pm["GridSpacing"], dtype=np.float64)
            direction = np.asarray(pm["GridDirection"], dtype=np.float64).reshape(2, 2)
            coefficients = np.asarray(pm["TransformParameters"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed Elastix coefficient lattice") from exc
        if (size.shape != (2,) or np.any(size < 4) or np.any(size > 512)
                or self._origin.shape != (2,) or spacing.shape != (2,)
                or np.any(spacing <= 0) or not np.isfinite(spacing).all()
                or not np.isfinite(self._origin).all()
                or not np.isfinite(direction).all()
                or not np.allclose(direction.T @ direction, np.eye(2), atol=1e-8)
                or coefficients.shape != (2 * int(np.prod(size)),)
                or not np.isfinite(coefficients).all()):
            raise ValueError("Invalid Elastix coefficient lattice geometry or values")
        self._size = size
        self._physical_to_grid = np.diag(1 / spacing) @ direction.T
        self._coefficients = coefficients.reshape(2, size[1], size[0])

    def is_identity(self) -> bool:
        """True only when the complete mapping is identity, including its affine."""
        return bool(np.allclose(self.affine, np.eye(2, 3), rtol=0, atol=1e-12)
                    and np.max(np.abs(self._coefficients)) <= 1e-12)

    def _residual(self, points: FloatArray) -> tuple[FloatArray, FloatArray]:
        mapped = points.copy()
        jacobian = np.broadcast_to(np.eye(2), (len(points), 2, 2)).copy()
        for start in range(0, len(points), 4096):
            batch = points[start:start + 4096]
            grid = (batch - self._origin) @ self._physical_to_grid.T
            active = np.all((grid >= 1) & (grid < self._size - 2), axis=1)
            if not active.any():
                continue
            indices = np.flatnonzero(active) + start
            floor = np.floor(grid[active]).astype(np.int64)
            weights, derivatives = _basis(grid[active] - floor)
            displacement = np.zeros((len(floor), 2))
            derivative_grid = np.zeros((len(floor), 2, 2))
            for iy in range(4):
                for ix in range(4):
                    coefficient = self._coefficients[:, floor[:, 1] + iy - 1,
                                                     floor[:, 0] + ix - 1].T
                    displacement += coefficient * (weights[:, 0, ix] * weights[:, 1, iy])[:, None]
                    derivative_grid[:, :, 0] += coefficient * (
                        derivatives[:, 0, ix] * weights[:, 1, iy])[:, None]
                    derivative_grid[:, :, 1] += coefficient * (
                        weights[:, 0, ix] * derivatives[:, 1, iy])[:, None]
            mapped[indices] += displacement
            jacobian[indices] += derivative_grid @ self._physical_to_grid
        return mapped, jacobian

    def inverse(self, points_mm: ArrayLike) -> FloatArray:
        """Analytic atlas-to-original-section pullback A^-1(B(x))."""
        residual, _ = self._residual(_points(points_mm))
        return (residual - self.affine[:, 2]) @ self._affine_inverse.T

    def forward(self, points_mm: ArrayLike, *, tolerance_mm: float = 1e-6,
                max_iterations: int = 40) -> FloatArray:
        """Invert the residual pullback using damped Newton steps."""
        if not np.isfinite(tolerance_mm) or tolerance_mm <= 0 or max_iterations < 1:
            raise ValueError("Inverse tolerance and iteration count must be positive")
        source = _points(points_mm)
        goal = source @ self.affine[:, :2].T + self.affine[:, 2]
        current = goal.copy()
        for _ in range(max_iterations):
            mapped, derivative = self._residual(current)
            residual = mapped - goal
            error = np.linalg.norm(residual, axis=1)
            active = error > tolerance_mm
            if not active.any():
                return current
            jac = derivative[active]
            if np.any(np.linalg.det(jac) <= 1e-8):
                raise ValueError("Elastix landmark warp folds or collapses")
            step = np.linalg.solve(jac, residual[active, :, None])[..., 0]
            old = current[active]
            trial = old - step
            for _ in range(16):
                trial_error = np.linalg.norm(self._residual(trial)[0] - goal[active], axis=1)
                worse = trial_error >= error[active]
                if not worse.any():
                    break
                step[worse] *= 0.5
                trial[worse] = old[worse] - step[worse]
            current[active] = trial
        if np.any(np.linalg.norm(self._residual(current)[0] - goal, axis=1) > tolerance_mm):
            raise ValueError("Elastix landmark warp inverse did not converge")
        return current

    def jacobian(self, points_mm: ArrayLike) -> FloatArray:
        """Forward derivatives in original-section coordinates."""
        _, derivative = self._residual(self.forward(points_mm))
        return np.linalg.inv(derivative) @ self.affine[:, :2]

    def validate_domain(self, extent_mm: ArrayLike, *, grid_size: int = 129) -> dict[str, float]:
        """Screen dense samples for folds, extreme residual strain and failed inverses.

        Finite sampling is not a mathematical guarantee between sample locations.
        Stretch limits apply to the residual, leaving the accepted affine intact.
        """
        extent = _extent(extent_mm)
        if not np.allclose(extent, self.extent, rtol=1e-10, atol=1e-12):
            raise ValueError("Extent does not match the saved transform")
        if grid_size < 2:
            raise ValueError("grid_size must be at least two")
        lo, hi = self.domain_bounds
        xs, ys = np.meshgrid(np.linspace(lo[0], hi[0], max(grid_size, 129)),
                             np.linspace(lo[1], hi[1], max(grid_size, 129)))
        probes = np.vstack((np.column_stack((xs.ravel(), ys.ravel())), self.target))
        mapped, derivative = self._residual(probes)
        determinant = np.linalg.det(derivative)
        stretches = np.linalg.svd(derivative, compute_uv=False)
        if np.any(determinant <= 1e-6):
            raise ValueError("Elastix landmark warp folds or collapses; revise correspondences")
        if (np.min(stretches) < MIN_RESIDUAL_STRETCH
                or np.max(stretches) > MAX_RESIDUAL_STRETCH):
            raise ValueError("Elastix landmark warp stretches or compresses tissue excessively")
        source_probes = (mapped - self.affine[:, 2]) @ self._affine_inverse.T
        restored = self.forward(source_probes)
        roundtrip = float(np.max(np.linalg.norm(restored - probes, axis=1)))
        if roundtrip > 1e-4:
            raise ValueError("Elastix landmark warp failed inverse roundtrip validation")
        return {"minimum_residual_jacobian": float(np.min(determinant)),
                "minimum_residual_stretch": float(np.min(stretches)),
                "maximum_residual_stretch": float(np.max(stretches)),
                "maximum_roundtrip_error_mm": roundtrip}


@lru_cache(maxsize=16)
def _load_cached(serialized: str) -> LandmarkElastix:
    transform = LandmarkElastix(json.loads(serialized))
    transform.validate_domain(transform.extent)
    return transform


def load_elastix(payload: Mapping[str, Any]) -> LandmarkElastix:
    """Load and validate a self-contained JSON transform without refitting it."""
    try:
        saved = {key: value for key, value in payload.items() if key != "diagnostics"}
        serialized = json.dumps(saved, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("Elastix transform must contain finite JSON data") from exc
    return _load_cached(serialized)
