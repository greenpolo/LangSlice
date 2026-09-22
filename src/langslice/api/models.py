"""Pydantic models for the LangSlice engine stdio contract."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Plane = Literal["coronal", "sagittal", "horizontal"]
# Canonical names are access methods (see providers/registry.py); the first
# three are legacy aliases kept for old configs.
Provider = Literal[
    "google",
    "openai",
    "chatgpt",
    "gemini-api",
    "openai-api",
    "openai-oauth",
    # No model at all: registration registers the silhouette prior itself.
    "none",
]
# Adaptive CLAHE + DAPI-weighted grayscale on the section before it is sent.
PreprocessMode = Literal["none", "auto"]
# Elastix stages the fit runs (see nonlinear.types.Deformation).
Deformation = Literal["none", "bspline", "affine"]
# Route "atlas" (no supplied placement) draws once, or twice with a
# self-correction call (see nonlinear.image_gen_registration).
Passes = Literal[1, 2]
EngineMethod = Literal[
    "version",
    "setup.status",
    "setup.login",
    "setup.api_key",
    "linear.run",
    "nonlinear.abba",
    "register.run",
    "quick_affine.run",
    "export.run",
]
ENGINE_METHODS: tuple[EngineMethod, ...] = (
    "version",
    "setup.status",
    "setup.login",
    "setup.api_key",
    "linear.run",
    "nonlinear.abba",
    "register.run",
    "quick_affine.run",
    "export.run",
)


class EngineBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EngineError(EngineBaseModel):
    code: str
    message: str
    details: dict[str, object] = Field(default_factory=dict)


class EngineRequest(EngineBaseModel):
    id: str
    method: EngineMethod
    params: dict[str, object] = Field(default_factory=dict)


class EngineProgressEvent(EngineBaseModel):
    kind: Literal["progress"]
    message: str
    stage: str | None = None


class EngineLogEvent(EngineBaseModel):
    kind: Literal["log"]
    message: str


class EngineDataEvent(EngineBaseModel):
    """Host events: login URL, checkpoints and agent activity, never credentials."""

    kind: Literal["data"] = "data"
    payload: dict[str, object]


EngineEvent = Annotated[
    EngineProgressEvent | EngineLogEvent | EngineDataEvent, Field(discriminator="kind")
]


class SetupLoginRequest(EngineBaseModel):
    timeout_s: float = Field(default=300, ge=10, le=900, allow_inf_nan=False)


class SetupApiKeyRequest(EngineBaseModel):
    provider: Literal["openai-api", "gemini-api"]
    api_key: str = Field(min_length=1, max_length=8192, repr=False)


class EngineEventEnvelope(EngineBaseModel):
    id: str
    type: Literal["event"]
    event: EngineEvent


class EngineResultEnvelope(EngineBaseModel):
    id: str
    type: Literal["result"]
    result: dict[str, object] = Field(default_factory=dict)


class EngineErrorEnvelope(EngineBaseModel):
    id: str | None
    type: Literal["error"]
    error: EngineError


class VersionResult(EngineBaseModel):
    version: str


class RegisterRequest(EngineBaseModel):
    image_path: str
    atlas: str
    position_mm: float
    plane: Plane = "coronal"
    model: str | None = None
    image_model: str | None = None
    review_model: str | None = None
    thinking: str | None = None
    temperature: float | None = None
    preprocess: PreprocessMode = "auto"
    provider: Provider = "google"
    endpoint: str | None = None
    output_dir: str | None = None
    max_iterations: int = 20
    media_resolution: str | None = None
    openai_image_route: str = "images"
    canvas_pad: float = 0.0
    vlm_resolution: int | None = None
    image_axes: str | None = None
    # Block cutting angles; every atlas render is resliced on that plane.
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    deformation: Deformation = "bspline"
    # Route "atlas" only: one draw, or two with a self-correction call.
    passes: Passes = 1
    # Native sampled atlas pixels (after image_axes/mirror) -> acquisition pixels.
    initial_atlas_to_slice: list[list[float]] | None = None
    initial_alignment_source: str = "supplied"
    atlas_mirror_lr: bool = False

    @field_validator("initial_atlas_to_slice")
    @classmethod
    def validate_initial_alignment(cls, matrix: list[list[float]] | None):
        if matrix is None:
            return None
        if len(matrix) != 3 or any(len(row) != 3 for row in matrix):
            raise ValueError("initial_atlas_to_slice must be a 3x3 affine matrix")
        if not all(math.isfinite(value) for row in matrix for value in row):
            raise ValueError("initial_atlas_to_slice must contain finite numbers")
        if any(
            abs(value - target) > 1e-12
            for value, target in zip(matrix[2], (0, 0, 1), strict=True)
        ):
            raise ValueError("initial_atlas_to_slice must have final row [0, 0, 1]")
        determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
        if not math.isfinite(determinant) or determinant == 0:
            raise ValueError("initial_atlas_to_slice must be invertible")
        return matrix


class RegisterResult(EngineBaseModel):
    accepted_correspondence_count: int
    rotation_deg: float
    translation_px: tuple[float, float]
    scale: tuple[float, float]
    shear: float
    debug_dir: str | None = None
    annotation_session: dict[str, object] | None = None
    warped_atlas_path: str | None = None
    warped_border_overlay_path: str | None = None
    generated_segmentation_path: str | None = None
    generated_border_overlay_path: str | None = None
    slice_warped_to_atlas_path: str | None = None
    slice_atlas_border_overlay_path: str | None = None
    inverse_warp_status: str | None = None
    raw_correction_path: str | None = None
    rough_border_overlay_path: str | None = None
    corrected_border_overlay_path: str | None = None


class QuickAffineRequest(EngineBaseModel):
    image_path: str
    atlas: str
    position_mm: float
    plane: Plane = "coronal"
    output_dir: str | None = None
    output_path: str | None = None


class QuickAffineResult(EngineBaseModel):
    warped_slice_path: str
    elapsed_s: float
    silhouette_iou: float


class ExportRequest(EngineBaseModel):
    image_path: str
    atlas: str
    position_mm: float
    output_path: str
    affine_matrix: list[list[float]] | None = None
    output_width: int | None = None
    output_height: int | None = None
    rotation_deg: float = 0.0
    translate_x_pct: float = 0.0
    translate_y_pct: float = 0.0


class ExportResult(EngineBaseModel):
    output_path: str
    target: str
    aligner: str
    slices: int


def export_schema_bundle() -> dict[str, object]:
    models: dict[str, type[BaseModel]] = {
        "EngineError": EngineError,
        "EngineRequest": EngineRequest,
        "EngineProgressEvent": EngineProgressEvent,
        "EngineDataEvent": EngineDataEvent,
        "SetupLoginRequest": SetupLoginRequest,
        "SetupApiKeyRequest": SetupApiKeyRequest,
        "EngineLogEvent": EngineLogEvent,
        "EngineEventEnvelope": EngineEventEnvelope,
        "EngineResultEnvelope": EngineResultEnvelope,
        "EngineErrorEnvelope": EngineErrorEnvelope,
        "VersionResult": VersionResult,
        "RegisterRequest": RegisterRequest,
        "RegisterResult": RegisterResult,
        "QuickAffineRequest": QuickAffineRequest,
        "QuickAffineResult": QuickAffineResult,
        "ExportRequest": ExportRequest,
        "ExportResult": ExportResult,
    }
    return {
        "schema_version": "1",
        "schemas": {
            model_name: model_cls.model_json_schema() for model_name, model_cls in models.items()
        },
    }
