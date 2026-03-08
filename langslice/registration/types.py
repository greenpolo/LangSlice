"""Affine registration result types and matrix helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

AffineMatrixLike = Sequence[Sequence[float]] | np.ndarray


def identity_affine_matrix() -> np.ndarray:
    """Return a 3x3 identity affine matrix."""
    return np.eye(3, dtype=np.float64)


def coerce_affine_matrix(matrix: AffineMatrixLike) -> np.ndarray:
    """Normalize *matrix* to a finite 3x3 float64 array."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 affine matrix, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("Affine matrix contains non-finite values")
    return arr.copy()


def apply_affine_to_points(
    matrix: AffineMatrixLike,
    points: Sequence[Sequence[float]],
) -> np.ndarray:
    """Apply a homogeneous affine matrix to 2D points."""
    arr = coerce_affine_matrix(matrix)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"Expected points with shape (N, 2), got {pts.shape}")
    homogeneous = np.concatenate(
        [pts, np.ones((pts.shape[0], 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (arr @ homogeneous.T).T
    return transformed[:, :2]


def _translation_matrix(tx: float, ty: float) -> np.ndarray:
    return np.array(
        [[1.0, 0.0, tx], [0.0, 1.0, ty], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _rotation_matrix(rotation_deg: float) -> np.ndarray:
    theta = math.radians(rotation_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return np.array(
        [[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def affine_matrix_from_legacy_params(
    image_width: int,
    image_height: int,
    rotation_deg: float = 0.0,
    translate_x_pct: float = 0.0,
    translate_y_pct: float = 0.0,
) -> np.ndarray:
    """Build the old GUI transform as a full homogeneous matrix."""
    center_x = float(image_width) / 2.0
    center_y = float(image_height) / 2.0
    tx_px = float(image_width) * (translate_x_pct / 100.0)
    ty_px = float(image_height) * (translate_y_pct / 100.0)
    return (
        _translation_matrix(center_x + tx_px, center_y + ty_px)
        @ _rotation_matrix(rotation_deg)
        @ _translation_matrix(-center_x, -center_y)
    )


def decompose_affine_matrix(matrix: AffineMatrixLike) -> dict[str, float]:
    """Return translation, rotation, scale, and shear terms for display."""
    arr = coerce_affine_matrix(matrix)
    a, b, tx = arr[0, 0], arr[0, 1], arr[0, 2]
    c, d, ty = arr[1, 0], arr[1, 1], arr[1, 2]

    scale_x = math.hypot(a, c)
    if scale_x <= 1e-12:
        rotation_deg = math.degrees(math.atan2(-b, d)) if abs(d) > 1e-12 else 0.0
        scale_y = math.hypot(b, d)
        shear = 0.0
    else:
        a_n = a / scale_x
        c_n = c / scale_x
        shear = (a_n * b) + (c_n * d)
        b_ortho = b - (a_n * shear)
        d_ortho = d - (c_n * shear)
        scale_y = math.hypot(b_ortho, d_ortho)
        if scale_y > 1e-12:
            shear /= scale_y
            b_n = b_ortho / scale_y
            d_n = d_ortho / scale_y
            if (a_n * d_n) - (c_n * b_n) < 0.0:
                scale_y = -scale_y
                shear = -shear
        else:
            shear = 0.0
        rotation_deg = math.degrees(math.atan2(c_n, a_n))

    return {
        "rotation_deg": rotation_deg,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "shear": shear,
        "translate_x_px": tx,
        "translate_y_px": ty,
    }


def is_valid_affine_matrix(matrix: AffineMatrixLike) -> bool:
    """Return True if *matrix* looks like a usable affine transform."""
    try:
        arr = coerce_affine_matrix(matrix)
    except ValueError:
        return False

    determinant = float(np.linalg.det(arr[:2, :2]))
    if not math.isfinite(determinant) or abs(determinant) < 1e-8:
        return False

    parts = decompose_affine_matrix(arr)
    scale_x = abs(parts["scale_x"])
    scale_y = abs(parts["scale_y"])
    if scale_x < 0.02 or scale_y < 0.02:
        return False
    if scale_x > 25.0 or scale_y > 25.0:
        return False
    if abs(parts["shear"]) > 20.0:
        return False
    return True


@dataclass
class AffineResult:
    """Matrix-first affine registration result."""

    matrix: np.ndarray
    source_size: tuple[int, int]
    output_size: tuple[int, int]
    backend: str
    reasoning: str
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.matrix = coerce_affine_matrix(self.matrix)
        self.source_size = (int(self.source_size[0]), int(self.source_size[1]))
        self.output_size = (int(self.output_size[0]), int(self.output_size[1]))

    @classmethod
    def from_legacy_params(
        cls,
        *,
        image_width: int,
        image_height: int,
        rotation_deg: float,
        translate_x_pct: float,
        translate_y_pct: float,
        backend: str,
        reasoning: str,
        provenance: dict[str, Any] | None = None,
    ) -> "AffineResult":
        """Construct a matrix-first result from the old reduced parameters."""
        return cls(
            matrix=affine_matrix_from_legacy_params(
                image_width=image_width,
                image_height=image_height,
                rotation_deg=rotation_deg,
                translate_x_pct=translate_x_pct,
                translate_y_pct=translate_y_pct,
            ),
            source_size=(image_width, image_height),
            output_size=(image_width, image_height),
            backend=backend,
            reasoning=reasoning,
            provenance=provenance or {},
        )

    @property
    def rotation_deg(self) -> float:
        return decompose_affine_matrix(self.matrix)["rotation_deg"]

    @property
    def translation_px(self) -> tuple[float, float]:
        parts = decompose_affine_matrix(self.matrix)
        return parts["translate_x_px"], parts["translate_y_px"]

    @property
    def scale(self) -> tuple[float, float]:
        parts = decompose_affine_matrix(self.matrix)
        return parts["scale_x"], parts["scale_y"]

    @property
    def shear(self) -> float:
        return decompose_affine_matrix(self.matrix)["shear"]

    @property
    def output_width(self) -> int:
        return self.output_size[0]

    @property
    def output_height(self) -> int:
        return self.output_size[1]

    @property
    def rotation(self) -> float:
        """Compatibility alias for the old GUI display path."""
        return self.rotation_deg

    @property
    def translateX(self) -> float:
        """Compatibility alias expressed as percentage of the source width."""
        width = self.source_size[0]
        if width <= 0:
            return 0.0
        return (self.translation_px[0] / float(width)) * 100.0

    @property
    def translateY(self) -> float:
        """Compatibility alias expressed as percentage of the source height."""
        height = self.source_size[1]
        if height <= 0:
            return 0.0
        return (self.translation_px[1] / float(height)) * 100.0
