"""The exact interpolating 2D thin-plate spline, in millimetres.

The kernel is r² log(r), as in BigWarp's
``jitk.spline.ThinPlateR2LogRSplineKernelTransform``;
:mod:`langslice.core.maps.abba` fits the deformation's pull-back with it to
place and validate the landmark pairs ABBA will build. Squared distances
are expanded as ``|x|² + |s|² - 2 x·s`` so that evaluating the spline at
many points is a few matrix products; on the unit-scaled coordinates used
here that costs about 1e-15 of precision. No state or image mutations
occur here.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

#: Points evaluated per block: a block's distances to a thousand landmarks
#: (about 2 MB) stay in the processor's cache, several times faster than
#: blocks that spill to memory.
CHUNK = 256


def _points(value: ArrayLike) -> FloatArray:
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Coordinates must be finite N-by-2 [x, y] pairs")
    if not np.isfinite(points).all():
        raise ValueError("Coordinates must be finite N-by-2 [x, y] pairs")
    return points


def _squared(points: FloatArray, landmarks: FloatArray) -> FloatArray:
    """Squared distances from each point to each landmark."""
    squared = points @ (-2.0 * landmarks.T)
    squared += (points[:, 0] ** 2 + points[:, 1] ** 2)[:, None]
    squared += (landmarks[:, 0] ** 2 + landmarks[:, 1] ** 2)[None, :]
    return np.maximum(squared, 0.0, out=squared)


def _centre(points: FloatArray) -> FloatArray:
    """Average N points."""
    means = [np.mean(points[:, 0]), np.mean(points[:, 1])]
    return np.array(means, dtype=np.float64)


def _span(points: FloatArray) -> float:
    """Measure the longer side of the points' bounding box."""
    width = float(np.max(points[:, 0]) - np.min(points[:, 0]))
    height = float(np.max(points[:, 1]) - np.min(points[:, 1]))
    return max(width, height)


def _log(squared: FloatArray) -> FloatArray:
    """log(r²), finite at r = 0."""
    logged = np.maximum(squared, 1e-300)
    return np.log(logged, out=logged)


def _kernel(squared: FloatArray) -> FloatArray:
    kernel = _log(squared)
    kernel *= squared
    kernel *= 0.5
    return kernel


class ThinPlateKernel:
    """Fit an exact interpolating 2D TPS on rows of physical [x, y].

    Parameters
    ----------
    source
        N-by-2 landmarks the spline maps from.
    target
        N-by-2 points it maps them to.
    max_points
        The most landmarks accepted.

    Raises
    ------
    ValueError
        When the landmarks are too few or too many, repeated, collinear,
        or ill-conditioned.
    """

    def __init__(
        self, source: ArrayLike, target: ArrayLike, *, max_points: int
    ) -> None:
        source_mm, target_mm = _points(source), _points(target)
        count = len(source_mm)
        if source_mm.shape != target_mm.shape or not 4 <= count <= max_points:
            raise ValueError(
                f"Supply 4 to {max_points} corresponding source and "
                "target landmarks"
            )

        self.origin: FloatArray = _centre(source_mm)
        self.scale: float = _span(source_mm)
        if self.scale <= 0:
            raise ValueError(
                "Source landmarks must span a two-dimensional region"
            )
        self.source: FloatArray = (source_mm - self.origin) / self.scale
        self.target: FloatArray = (target_mm - self.origin) / self.scale
        _check_spread("Source", self.source)
        _check_spread("Target", self.target)

        polynomial = np.column_stack((np.ones(count), self.source))
        system = np.block(
            [
                [_kernel(_squared(self.source, self.source)), polynomial],
                [polynomial.T, np.zeros((3, 3))],
            ]
        )
        if np.linalg.cond(system) > 1e12:
            raise ValueError("Landmark arrangement is ill-conditioned")

        rhs = np.vstack((self.target, np.zeros((3, 2))))
        coefficients = np.linalg.solve(system, rhs).astype(np.float64)
        self.weights: FloatArray = coefficients[:-3]
        self.affine: FloatArray = coefficients[-3:]
        # the weights the derivative's three sums take, side by side
        self.gradient_weights: FloatArray = np.hstack(
            (
                self.weights,
                self.source[:, :1] * self.weights,
                self.source[:, 1:] * self.weights,
            )
        )

    def _forward(self, points: FloatArray) -> FloatArray:
        kernel = _kernel(_squared(points, self.source))
        return (
            kernel @ self.weights + points @ self.affine[1:] + self.affine[:1]
        )

    def _jacobian(self, points: FloatArray) -> FloatArray:
        radial = _log(_squared(points, self.source))
        radial += 1.0
        # d/dx_i of sum_k K_k w_k = x_i sum_k radial_k w_k
        #                           - sum_k radial_k s_ki w_k
        sums = radial @ self.gradient_weights
        columns = [
            points[:, axis, None] * sums[:, :2]
            - sums[:, 2 + 2 * axis : 4 + 2 * axis]
            for axis in (0, 1)
        ]
        return np.stack(columns, axis=-1) + self.affine[1:].T

    def forward(self, points_mm: ArrayLike) -> FloatArray:
        """Map physical source points to targets, in bounded memory blocks.

        Parameters
        ----------
        points_mm
            N-by-2 points on the source side.

        Returns
        -------
        FloatArray
            N-by-2 points on the target side.
        """
        points = (_points(points_mm) - self.origin) / self.scale
        result = np.empty_like(points)
        for start in range(0, len(points), CHUNK):
            stop = start + CHUNK
            result[start:stop] = self._forward(points[start:stop])
        return result * self.scale + self.origin

    def jacobian(self, points_mm: ArrayLike) -> FloatArray:
        """Differentiate the spline at physical source points.

        Parameters
        ----------
        points_mm
            N-by-2 points on the source side.

        Returns
        -------
        FloatArray
            N-by-2-by-2 derivatives (output coordinate, input coordinate).
        """
        points = (_points(points_mm) - self.origin) / self.scale
        result = np.empty((len(points), 2, 2), dtype=np.float64)
        for start in range(0, len(points), CHUNK):
            stop = start + CHUNK
            result[start:stop] = self._jacobian(points[start:stop])
        return result


def _check_spread(label: str, points: FloatArray) -> None:
    """Refuse landmarks that repeat or lie on one line."""
    squared = _squared(points, points)
    np.fill_diagonal(squared, np.inf)
    if np.min(squared) < 1e-12:
        raise ValueError(
            f"{label} landmarks must be distinct and well separated"
        )
    singular = np.linalg.svd(points - _centre(points), compute_uv=False)
    if singular[-1] < singular[0] * 1e-6:
        raise ValueError(f"{label} landmarks must not be collinear")
