"""Shared training-data contracts for LangSlice training-core."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CONTRACT_SCHEMA_VERSION = "langslice_sft_single_slice_v1"

Plane = Literal["coronal", "sagittal", "horizontal"]


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    image_paths: list[str] = Field(min_length=1)
    text: str | None = None


class SubmitEstimateArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    position_mm: int | float
    reasoning: str | None = None


class SubmitEstimate(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    name: Literal["submit_estimate"]
    args: SubmitEstimateArgs


class TraceStep(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    submit: SubmitEstimate | None = None
    thinking: str | None = None

    @model_validator(mode="after")
    def _validate_step_shape(self) -> TraceStep:
        if self.submit is not None:
            if self.tool_call is not None or self.tool_result is not None:
                raise ValueError("submit step cannot include tool_call/tool_result")
            return self
        if self.tool_call is None and self.tool_result is None:
            raise ValueError("trace step must include submit or tool_call/tool_result")
        if self.tool_call is None or self.tool_result is None:
            raise ValueError("tool_call and tool_result must both be present")
        return self


class SFTExample(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    bucket: int
    atlas_name: str
    atlas_version: str
    plane: Plane
    subject_id: str
    system_prompt_kind: str
    query_image_paths: list[str] = Field(min_length=1)
    user_prompt_text: str
    trace: list[TraceStep] = Field(min_length=1)
    quality: dict[str, Any] = Field(default_factory=dict)
    gemini_reasoning: str | None = None

    @model_validator(mode="after")
    def _validate_trace_contract(self) -> SFTExample:
        if self.trace[-1].submit is None:
            raise ValueError("trace must end with submit")
        for idx, step in enumerate(self.trace[:-1]):
            if step.submit is not None:
                raise ValueError(f"trace[{idx}] cannot be submit")
            if step.tool_call is None or step.tool_result is None:
                raise ValueError(f"trace[{idx}] must include tool_call and tool_result")
        return self


class _ContractSchemaEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[CONTRACT_SCHEMA_VERSION]
    sft_example: SFTExample


def validate_sft_example(payload: dict[str, Any] | str) -> SFTExample:
    if isinstance(payload, str):
        return SFTExample.model_validate_json(payload)
    return SFTExample.model_validate(payload)


def export_contract_json_schema() -> dict[str, Any]:
    return _ContractSchemaEnvelope.model_json_schema()
