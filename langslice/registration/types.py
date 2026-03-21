"""Registration result types and matrix helpers."""

from __future__ import annotations

from collections.abc import Sequence
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

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
    ) -> AffineResult:
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


@dataclass(frozen=True)
class RegistrationCorrespondence:
    """One paired anatomical correspondence between atlas and slice."""

    slice_xy: tuple[float, float]
    atlas_xy: tuple[float, float]
    label: str
    confidence: str = "medium"
    rationale: str = ""
    slice_normalized_yx: tuple[float, float] | None = None
    atlas_normalized_yx: tuple[float, float] | None = None


@dataclass(frozen=True)
class LandmarkAnnotation:
    """One visible numbered landmark annotation on an atlas or slice image."""

    image_role: str
    pixel_xy: tuple[float, float]
    label: str
    normalized_yx: tuple[float, float] | None = None
    status: str = "confirmed"
    feature_description: str = ""
    artifact_note: str = ""


@dataclass
class RegistrationAnnotationSession:
    """Shared landmark annotation state used by registration workflows and the GUI."""

    workflow: str
    target_count: int | None = None
    atlas_annotations: list[LandmarkAnnotation] = field(default_factory=list)
    slice_annotations: list[LandmarkAnnotation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class NonlinearResult:
    """Regularized nonlinear registration result derived from landmarks."""

    atlas_points: np.ndarray
    slice_points: np.ndarray
    smoothing: float
    backend: str
    reasoning: str
    output_size: tuple[int, int]
    qc_metrics: dict[str, Any] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.atlas_points = np.asarray(self.atlas_points, dtype=np.float64)
        self.slice_points = np.asarray(self.slice_points, dtype=np.float64)
        if self.atlas_points.ndim != 2 or self.atlas_points.shape[1] != 2:
            raise ValueError("atlas_points must have shape (N, 2)")
        if self.slice_points.ndim != 2 or self.slice_points.shape[1] != 2:
            raise ValueError("slice_points must have shape (N, 2)")
        if self.atlas_points.shape != self.slice_points.shape:
            raise ValueError("atlas_points and slice_points must have the same shape")
        self.output_size = (int(self.output_size[0]), int(self.output_size[1]))
        self.smoothing = float(self.smoothing)


@dataclass
class RegistrationResult:
    """Complete registration runtime output."""

    correspondences: list[RegistrationCorrespondence]
    accepted_correspondences: list[RegistrationCorrespondence]
    rejected_correspondences: list[dict[str, Any]]
    affine_result: AffineResult
    nonlinear_result: NonlinearResult
    qc_state: str
    debug_dir: str | None = None
    annotation_session: RegistrationAnnotationSession | None = None


def build_annotation_session_from_correspondences(
    correspondences: Sequence[RegistrationCorrespondence],
    *,
    workflow: str = "multimodal_tool_loop",
    target_count: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> RegistrationAnnotationSession:
    """Project final correspondences into a workflow-neutral annotation session."""
    atlas_annotations: list[LandmarkAnnotation] = []
    slice_annotations: list[LandmarkAnnotation] = []

    for corr in correspondences:
        atlas_annotations.append(
            LandmarkAnnotation(
                image_role="atlas",
                pixel_xy=(float(corr.atlas_xy[0]), float(corr.atlas_xy[1])),
                label=str(corr.label),
                normalized_yx=corr.atlas_normalized_yx,
                feature_description=str(corr.rationale),
            )
        )
        slice_annotations.append(
            LandmarkAnnotation(
                image_role="slice",
                pixel_xy=(float(corr.slice_xy[0]), float(corr.slice_xy[1])),
                label=str(corr.label),
                normalized_yx=corr.slice_normalized_yx,
                feature_description=str(corr.rationale),
            )
        )

    return RegistrationAnnotationSession(
        workflow=str(workflow),
        target_count=target_count if target_count is not None else len(correspondences),
        atlas_annotations=atlas_annotations,
        slice_annotations=slice_annotations,
        metadata=dict(metadata or {}),
    )


def get_session_marker_points(
    session: RegistrationAnnotationSession | None,
    *,
    image_role: str,
) -> list[tuple[float, float, str]]:
    """Return `(x, y, label)` markers for one image role."""
    if session is None:
        return []
    annotations = session.slice_annotations if image_role == "slice" else session.atlas_annotations
    return [
        (float(annotation.pixel_xy[0]), float(annotation.pixel_xy[1]), str(annotation.label))
        for annotation in annotations
        if annotation.status != "not_visible"
    ]


def render_landmark_annotations(
    image: Image.Image,
    annotations: Sequence[LandmarkAnnotation],
    *,
    point_outline: tuple[int, int, int] = (255, 80, 80),
    label_fill: tuple[int, int, int] = (255, 220, 120),
) -> Image.Image:
    """Return an image with visible numbered landmark annotations baked in."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    max_dim = max(annotated.size) if annotated.size else 1
    radius = max(6, int(round(max_dim / 160.0)))
    x_offset = radius + 2
    y_offset = max(4, radius // 2)

    for annotation in annotations:
        if annotation.status == "not_visible":
            continue
        x, y = annotation.pixel_xy
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=point_outline,
            width=max(2, radius // 3),
        )
        draw.text((x + x_offset, y + y_offset), str(annotation.label), fill=label_fill)
    return annotated


def annotation_session_to_dict(
    session: RegistrationAnnotationSession | None,
) -> dict[str, Any] | None:
    """Serialize an annotation session for debug payloads."""
    if session is None:
        return None
    return asdict(session)
