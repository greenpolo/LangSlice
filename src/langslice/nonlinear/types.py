"""Registration result types, matrix helpers, and image-gen candidate types."""

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageDraw

AffineMatrixLike = Sequence[Sequence[float]] | np.ndarray

#: Elastix deformation model. ``"bspline"`` is the historical affine +
#: B-spline pair; ``"affine"`` stops after the affine stage, because the
#: B-spline stage driven by generated paintings measured BELOW the affine
#: stage alone on the hand-registered slices (family dice 0.65 vs 0.735 flat,
#: 0.74 vs 0.80 oblique).
Deformation = Literal["none", "bspline", "affine"]


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
    category: str = "border"


@dataclass
class RegistrationAnnotationSession:
    """Shared registration annotation state used by the GUI."""

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
    affine_result: AffineResult
    nonlinear_result: NonlinearResult
    debug_dir: str | None = None
    annotation_session: RegistrationAnnotationSession | None = None


def render_landmark_annotations(
    image: Image.Image,
    annotations: Sequence[LandmarkAnnotation],
    *,
    point_outline: tuple[int, int, int] = (0, 255, 200),
    label_fill: tuple[int, int, int] = (255, 255, 255),
    reference_size: int | None = None,
) -> Image.Image:
    """Return an image with visible numbered landmark annotations baked in.

    *reference_size* (optional) fixes the radius calculation to a specific
    image dimension so that annotations appear the same relative size
    regardless of actual image dimensions.  Pass the same value for atlas
    and slice to get consistent marker sizes in side-by-side views.
    """
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    ref = reference_size if reference_size is not None else max(annotated.size)
    ref = max(ref, 1)
    radius = max(6, int(round(ref / 100.0)))
    stroke = max(2, radius // 3)
    x_offset = radius + 3
    y_offset = max(4, radius // 2)

    for annotation in annotations:
        if annotation.status == "not_visible":
            continue
        x, y = annotation.pixel_xy
        # Filled dot with contrasting outline for visibility on any background
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=point_outline,
            outline=(0, 0, 0),
            width=stroke,
        )
        # White label text with dark shadow for readability
        draw.text((x + x_offset + 1, y + y_offset + 1), str(annotation.label), fill=(0, 0, 0))
        draw.text((x + x_offset, y + y_offset), str(annotation.label), fill=label_fill)
    return annotated


def annotation_session_to_dict(
    session: RegistrationAnnotationSession | None,
) -> dict[str, Any] | None:
    """Serialize an annotation session for debug payloads."""
    if session is None:
        return None
    return asdict(session)

@dataclass
class GeneratedSegmentation:
    """Image-gen output for a dense registration candidate."""

    image: Image.Image
    provider: str
    model: str
    route: str
    revised_prompt: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegistrationCandidate:
    """Dense image-gen registration candidate and its annotation session."""

    candidate_id: str
    generated_segmentation: Image.Image
    warped_atlas: Image.Image
    warped_border_overlay: Image.Image
    markers: list[list[float]]
    annotation_session: RegistrationAnnotationSession
    metadata: dict[str, Any] = field(default_factory=dict)
    # Integer atlas identities after registration, independent of palette colors.
    warped_labels: np.ndarray | None = None
    # Output canvas -> native oriented/mirrored atlas pixel centers, x then y.
    atlas_coordinate_map: np.ndarray | None = None


def candidate_to_registration_result(
    candidate: RegistrationCandidate,
    image_size: tuple[int, int],
    debug_dir: str | None = None,
) -> RegistrationResult:
    """Convert a dense registration candidate into a runtime result."""

    session = candidate.annotation_session
    session_metadata = deepcopy(session.metadata)
    session_metadata.update(
        {
            "visualign_markers": deepcopy(candidate.markers),
            "n_markers": len(candidate.markers),
            "candidate_id": candidate.candidate_id,
            "candidate_metadata": deepcopy(candidate.metadata),
        }
    )
    session.metadata = session_metadata

    affine_result = AffineResult(
        matrix=identity_affine_matrix(),
        source_size=image_size,
        output_size=image_size,
        backend="image_gen_registration_dense",
        reasoning="Dense VisuAlign markers are stored in annotation metadata.",
    )
    nonlinear_result = NonlinearResult(
        atlas_points=np.zeros((0, 2), dtype=np.float64),
        slice_points=np.zeros((0, 2), dtype=np.float64),
        smoothing=0.0,
        backend="elastix_bspline_visualign",
        reasoning=(
            "Dense VisuAlign markers are the transform representation stored "
            "in annotation metadata."
        ),
        output_size=image_size,
    )

    return RegistrationResult(
        correspondences=[],
        accepted_correspondences=[],
        affine_result=affine_result,
        nonlinear_result=nonlinear_result,
        debug_dir=debug_dir,
        annotation_session=session,
    )
