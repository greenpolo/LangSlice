"""Pydantic models for the LangSlice engine stdio contract."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

Plane = Literal["coronal", "sagittal", "horizontal"]
Provider = Literal["google", "openai"]
PreprocessMode = Literal["none", "auto"]
Workflow = Literal["tool_use", "image_gen"]
EngineMethod = Literal[
    "version",
    "estimate.run",
    "register.run",
    "quick_affine.run",
    "export.run",
]
ENGINE_METHODS: tuple[EngineMethod, ...] = (
    "version",
    "estimate.run",
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


EngineEvent = Annotated[EngineProgressEvent | EngineLogEvent, Field(discriminator="kind")]


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


class EstimateRequest(EngineBaseModel):
    image_path: str
    atlas: str
    plane: Plane = "coronal"
    model: str | None = None
    thinking: str | None = None
    temperature: float | None = None
    media_resolution: str | None = None
    max_iterations: int = 20
    preprocess: PreprocessMode = "none"
    provider: Provider = "google"
    endpoint: str | None = None
    output_dir: str | None = None
    workflow: Workflow | None = None


class EstimateResult(EngineBaseModel):
    position_mm: float
    reasoning: str
    debug_dir: str | None = None


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
    preprocess: PreprocessMode = "none"
    provider: Provider = "google"
    endpoint: str | None = None
    output_dir: str | None = None
    max_iterations: int = 20
    media_resolution: str | None = None


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
        "EngineLogEvent": EngineLogEvent,
        "EngineEventEnvelope": EngineEventEnvelope,
        "EngineResultEnvelope": EngineResultEnvelope,
        "EngineErrorEnvelope": EngineErrorEnvelope,
        "VersionResult": VersionResult,
        "EstimateRequest": EstimateRequest,
        "EstimateResult": EstimateResult,
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
            model_name: model_cls.model_json_schema()
            for model_name, model_cls in models.items()
        },
    }
