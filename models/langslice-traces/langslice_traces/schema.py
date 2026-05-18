"""Canonical dataclasses for trace inputs and rendered outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

Plane = Literal["coronal", "sagittal", "horizontal"]


@dataclass(frozen=True)
class ToolStep:
    """One non-terminal step: a fetch_atlas call and its tool result."""
    call_name: str                       # "fetch_atlas" in v1
    call_args: dict[str, Any]            # {"positions_mm": [floats]}
    result_image_paths: list[str]
    result_text: str


@dataclass(frozen=True)
class FinalAnswer:
    """The terminal submit step's payload."""
    name: str                            # "submit_estimate" in v1
    position_mm: float
    reasoning: str


@dataclass
class CanonicalTrace:
    """Parsed langslice-native trace ready for rendering in any mode.

    Mirrors the SFT JSONL corpus row shape with the multi-step ``trace`` list
    decomposed into typed ``tool_steps`` + a separate ``final_answer``.
    """
    atlas_name: str
    atlas_version: str
    plane: Plane
    subject_id: str
    system_prompt_kind: str               # "single_slice" in v1
    bucket: int
    query_image_paths: list[str]
    user_prompt_text: str
    tool_steps: list[ToolStep]
    final_answer: FinalAnswer | None
    quality: dict[str, Any] = field(default_factory=dict)
    gemini_reasoning: str | None = None
    dataset_root: Path | None = field(default=None, compare=False)


@dataclass(frozen=True)
class RenderMetadata:
    """Metadata fields carried alongside rendered messages for downstream consumers."""
    atlas_name: str
    atlas_version: str
    plane: str
    subject_id: str
    system_prompt_kind: str


@dataclass
class RenderedExample:
    """Output of a renderer — ready for processor.apply_chat_template(...).

    Mirrors models/langslice-gemma-4/training/sft/render.py::RenderedExample
    field-for-field, with two added fields for factory consumers:

    - ``target_mm`` is set by ``render_rl_prefix`` to the ground-truth position
      from manifest join; RL trainer uses this as the reward target.
    - ``label_mm`` is set by ``render_isft_prefix`` to the teacher's
      ``final_answer.position_mm``; iSFT trainer uses it as the supervised label.

    SFT renderers (full and answer-only) leave both as None.
    """
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    metadata: RenderMetadata
    image_paths: list[str] = field(default_factory=list)
    target_mm: float | None = None
    label_mm: float | None = None
