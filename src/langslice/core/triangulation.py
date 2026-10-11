"""VisuAlign piecewise affine pull-backs, with one mesh in both directions."""

from __future__ import annotations

import numpy as np
from scipy.spatial import Delaunay


class PiecewiseAffine:
    """Map corresponding vertices through a Delaunay mesh of source pixels.

    The inverse uses the very same triangles, so a different destination
    Delaunay diagonal cannot change the registration. Outside the mesh the
    map is undefined (NaN). Folded or collapsed triangles are refused.
    """

    def __init__(
        self, source_xy: np.ndarray, target_xy: np.ndarray, simplices: np.ndarray | None = None
    ) -> None:
        self.source = np.asarray(source_xy, dtype=np.float64)
        self.target = np.asarray(target_xy, dtype=np.float64)
        if (
            self.source.ndim != 2
            or self.source.shape[1] != 2
            or self.target.shape != self.source.shape
            or not np.isfinite(self.source).all()
            or not np.isfinite(self.target).all()
        ):
            raise ValueError("Mesh vertices must be finite corresponding x,y pairs")
        if len(np.unique(self.source, axis=0)) != len(self.source):
            raise ValueError("Repeated section landmarks have no unique triangulation")
        self.delaunay = Delaunay(self.source) if simplices is None else None
        self.simplices = (
            self.delaunay.simplices
            if self.delaunay is not None
            else np.asarray(simplices, dtype=np.int32)
        )
        a, b = self.source[self.simplices], self.target[self.simplices]
        area_a, area_b = self._areas(a), self._areas(b)
        if np.any(np.abs(area_a) < 1e-8) or np.any(np.abs(area_b) < 1e-8):
            raise ValueError("Collapsed landmark triangle")
        if np.any(area_a * area_b <= 0):
            raise ValueError("Landmark correspondences fold a triangle")

    @staticmethod
    def _areas(triangles: np.ndarray) -> np.ndarray:
        a, b = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
        return a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]

    def _map(self, points: np.ndarray, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 2)
        out = np.full(flat.shape, np.nan)
        # Bounding boxes spare the barycentric calculation for most points.
        for indices in self.simplices:
            src, dst = source[indices], target[indices]
            eligible = (
                np.isnan(out[:, 0])
                & np.all(flat >= src.min(axis=0) - 1e-8, axis=1)
                & np.all(flat <= src.max(axis=0) + 1e-8, axis=1)
            )
            where = np.flatnonzero(eligible)
            if not where.size:
                continue
            weights = (flat[where] - src[0]) @ np.linalg.inv(
                np.stack([src[1] - src[0], src[2] - src[0]], axis=1)
            ).T
            keep = np.all(weights >= -1e-8, axis=1) & (weights.sum(axis=1) <= 1 + 1e-8)
            out[where[keep]] = dst[0] + weights[keep] @ np.stack([dst[1] - dst[0], dst[2] - dst[0]])
        return out.reshape(points.shape)

    def map(self, points: np.ndarray) -> np.ndarray:
        if self.delaunay is None:
            return self._map(points, self.source, self.target)
        points = np.asarray(points, dtype=np.float64)
        flat = points.reshape(-1, 2)
        simplex = self.delaunay.find_simplex(flat)
        out = np.full(flat.shape, np.nan)
        inside = simplex >= 0
        transforms = self.delaunay.transform[simplex[inside]]
        weights = np.einsum("nij,nj->ni", transforms[:, :2], flat[inside] - transforms[:, 2])
        weights = np.column_stack([weights, 1 - weights.sum(axis=1)])
        vertices = self.target[self.simplices[simplex[inside]]]
        out[inside] = np.einsum("ni,nij->nj", weights, vertices)
        return out.reshape(points.shape)

    def inverse(self, points: np.ndarray) -> np.ndarray:
        return self._map(points, self.target, self.source)


def visualign_mesh(
    source_xy: np.ndarray, target_xy: np.ndarray, *, size: tuple[int, int], margin: float = 0.1
) -> PiecewiseAffine:
    """Source-domain triangulation in pixels, four fixed 10% outside corners."""
    scale = np.asarray(size, dtype=np.float64)
    corners = (
        np.array(
            [
                [-margin, -margin],
                [1 + margin, -margin],
                [-margin, 1 + margin],
                [1 + margin, 1 + margin],
            ]
        )
        * scale
        - 0.5
    )
    source, target = np.asarray(source_xy), np.asarray(target_xy)
    if np.any(source <= corners.min(axis=0)) or np.any(source >= corners.max(axis=0)):
        raise ValueError("Section landmarks must lie inside the fixed mesh corners")
    return PiecewiseAffine(np.vstack([corners, source]), np.vstack([corners, target]))
