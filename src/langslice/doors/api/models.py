"""Pydantic models for the LangSlice engine stdio contract."""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Plane = Literal["coronal", "sagittal", "horizontal"]
EngineMethod = Literal[
    "version",
    "setup.status",
    "setup.login",
    "setup.api_key",
    "linear.run",
    "claude.prepare",
    "preprocess.preview",
    "linear.estimate",
    "export.run",
]
ENGINE_METHODS: tuple[EngineMethod, ...] = (
    "version",
    "setup.status",
    "setup.login",
    "setup.api_key",
    "linear.run",
    "claude.prepare",
    "preprocess.preview",
    "linear.estimate",
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


class PreprocessingSettings(EngineBaseModel):
    """How a host's exported channels become the one image the agent sees.

    "auto" is the automatic path; the other fields apply to "custom" only
    (see :func:`langslice.core.image_prep.host_preprocess`). The weights' count is
    checked against the snapshot's pages when an image is read.
    """

    mode: Literal["auto", "custom"] = "auto"
    clahe: bool = True
    clahe_strength: Literal["low", "medium", "high"] = "medium"
    channel_weights: list[float] | None = None

    @field_validator("channel_weights")
    @classmethod
    def validate_weights(cls, weights: list[float] | None):
        if weights is None:
            return None
        if not weights:
            raise ValueError("channel_weights must name at least one weight")
        if any(not math.isfinite(value) or value < 0 for value in weights):
            raise ValueError("channel_weights must be finite and non-negative")
        if sum(weights) <= 0:
            raise ValueError("at least one channel weight must be above zero")
        return weights


class PreprocessPreviewRequest(EngineBaseModel):
    image_path: str
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    output_path: str


class LinearEstimateRequest(EngineBaseModel):
    spec: dict[str, Any] = Field(default_factory=dict)
    n_slices: int = Field(ge=1)
    locked: int = Field(default=0, ge=0)


class LinearEstimateResult(EngineBaseModel):
    #: None when no estimate can be given (``available`` False; ``basis``
    #: says why in plain language).
    low: float | None
    high: float | None
    unit: Literal["percent_of_usage_window"]
    basis: str
    available: bool = True


class PreprocessPreviewResult(EngineBaseModel):
    output_path: str
    width: int
    height: int


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
        "PreprocessingSettings": PreprocessingSettings,
        "PreprocessPreviewRequest": PreprocessPreviewRequest,
        "PreprocessPreviewResult": PreprocessPreviewResult,
        "LinearEstimateRequest": LinearEstimateRequest,
        "LinearEstimateResult": LinearEstimateResult,
        "EngineLogEvent": EngineLogEvent,
        "EngineEventEnvelope": EngineEventEnvelope,
        "EngineResultEnvelope": EngineResultEnvelope,
        "EngineErrorEnvelope": EngineErrorEnvelope,
        "VersionResult": VersionResult,
        "ExportRequest": ExportRequest,
        "ExportResult": ExportResult,
    }
    return {
        "schema_version": "1",
        "schemas": {
            model_name: model_cls.model_json_schema() for model_name, model_cls in models.items()
        },
    }
