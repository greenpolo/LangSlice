"""Landmark deformation dispatch, rendering, and legacy TPS in physical millimeters.

New Elastix payloads dispatch to their saved coefficient evaluator.

The kernel is r² log(r), as in BigWarp's
``jitk.spline.ThinPlateR2LogRSplineKernelTransform``. BigWarp fits the
target-to-source pullback analytically. Forward mapping solves
that *same* map; no independent reverse TPS is fitted.
Persisted source/target coordinates share the oriented original section's
normalized frame, scaled by ``extent_mm=[width, height]``. Targets may lie
outside that frame. No state or image mutations occur here.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

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


class _ThinPlateKernel:
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
            raise ValueError("Spline affine component is singular") from exc
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

class ThinPlateSpline:
    """BigWarp transform: analytic target-to-source TPS, numerical forward map.

    Source and target are physical [x, y] rows. BigWarp's serializable native
    transform stores the pullback needed for resampling; its inverse supplies
    the source-to-target mapping. Both directions here use that single kernel.
    """

    def __init__(self, source: ArrayLike, target: ArrayLike, *, max_points: int = 64) -> None:
        self.source = _points(source).copy()
        self.target = _points(target).copy()
        self._pullback = _ThinPlateKernel(self.target, self.source, max_points=max_points)

    def forward(self, points_mm: ArrayLike, *, tolerance_mm: float = 1e-6,
                max_iterations: int = 40) -> FloatArray:
        """Map section source to target by solving the native pullback."""
        return self._pullback.inverse(points_mm, tolerance_mm=tolerance_mm,
                                      max_iterations=max_iterations)

    def inverse(self, points_mm: ArrayLike) -> FloatArray:
        """Map target to original section analytically, as native BigWarp renders."""
        return self._pullback.forward(points_mm)

    def jacobian(self, points_mm: ArrayLike) -> FloatArray:
        """Forward derivative, computed from the inverse function theorem."""
        return np.asarray(
            np.linalg.inv(self._pullback.jacobian(self.forward(points_mm))), dtype=np.float64)

    def validate_domain(self, extent_mm: ArrayLike, *, grid_size: int = 33) -> None:
        """Reject sampled folds and failed inverses across the section/landmark bounds.

        A finite grid is a conservative screening check, not a mathematical
        guarantee of global invertibility outside the sampled domain.
        """
        extent = np.asarray(extent_mm, dtype=np.float64)
        if extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0):
            raise ValueError("extent_mm must contain positive finite width and height")
        source = self.source
        target = self.target
        bounds = np.vstack((np.zeros(2), extent, source, target))
        lo, hi = bounds.min(axis=0), bounds.max(axis=0)
        x, y = np.meshgrid(np.linspace(lo[0], hi[0], grid_size),
                           np.linspace(lo[1], hi[1], grid_size))
        probes = np.vstack((np.column_stack((x.ravel(), y.ravel())), source))
        pullback_determinant = np.linalg.det(self._pullback.jacobian(probes))
        if np.any(pullback_determinant <= 1e-6):
            raise ValueError("Landmark warp folds or collapses the section; revise correspondences")
        determinant = np.linalg.det(self.jacobian(probes))
        if np.any(determinant <= 1e-6):
            raise ValueError("Landmark warp folds or collapses the section; revise correspondences")
        restored = self.inverse(self.forward(probes))
        if np.max(np.linalg.norm(restored - probes, axis=1)) > 1e-4:
            raise ValueError("Spline failed inverse roundtrip validation")
        self.inverse(probes)


def fit_spline(spline: Mapping[str, Any]) -> Any:
    """Load the authoritative deformation; old checkpoints retain their exact TPS.

    New landmark refinements carry serialized Elastix maps. Their landmark
    provenance must never be mistaken for an exact-interpolating TPS fit.
    """
    if not isinstance(spline, Mapping):
        raise ValueError("Landmark deformation must be a mapping")
    backend = spline.get("backend", "tps")
    if backend == "elastix":
        from langslice.landmark_elastix import load_elastix

        return load_elastix(spline)
    if backend != "tps":
        raise ValueError(f"Unknown landmark deformation backend: {backend}")
    extent = np.asarray(spline.get("extent_mm"), dtype=np.float64)
    if extent.shape != (2,) or not np.isfinite(extent).all() or np.any(extent <= 0):
        raise ValueError("extent_mm must contain positive finite width and height")
    transform = ThinPlateSpline(_points(spline["source"]) * extent,
                                _points(spline["target"]) * extent)
    transform.validate_domain(extent)
    return transform


def warp_section(section: NDArray[Any], spline: Mapping[str, Any],
                 output_size: tuple[int, int], section_offset: tuple[int, int],
                 um_per_px: float, fill: tuple[int, int, int] = (0, 0, 0)) -> NDArray[Any]:
    """Resample original tissue through its authoritative deformation pullback.

    ``section_offset`` is the section's unwarped top-left in canvas pixels.
    Pixel centers use OpenCV's integer coordinate convention. Millimeters are
    measured from that section origin using the persisted physical extent.
    Separate x/y pixel scales preserve geometry when thumbnail aspect rounding
    differs slightly from the original image. ``um_per_px`` describes the host
    canvas calibration; it does not overwrite that original physical extent.
    """
    import cv2

    if not np.isfinite(um_per_px) or um_per_px <= 0:
        raise ValueError("um_per_px must be positive and finite")
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("Output dimensions must be positive")
    transform = fit_spline(spline)
    extent_mm = np.asarray(spline["extent_mm"], dtype=np.float64)
    section_size = np.asarray([section.shape[1], section.shape[0]], dtype=np.float64)
    mm_per_pixel = extent_mm / section_size
    map_x = np.empty((height, width), dtype=np.float32)
    map_y = np.empty((height, width), dtype=np.float32)
    for row in range(0, height, 32):
        ys, xs = np.mgrid[row:min(row + 32, height), 0:width]
        canvas = np.column_stack((xs.ravel(), ys.ravel()))
        original = transform.inverse((canvas - np.asarray(section_offset)) * mm_per_pixel)
        pixels = original / mm_per_pixel
        map_x[row:row + len(xs)] = pixels[:, 0].reshape(xs.shape)
        map_y[row:row + len(xs)] = pixels[:, 1].reshape(xs.shape)
    return cv2.remap(section, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=fill)
