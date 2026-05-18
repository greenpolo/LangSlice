"""Render a CanonicalTrace as a minimal answer-only training example.

Used for ablation or for the answer-prediction head if it's ever decoupled
from the multi-turn reasoning. Emits:
  - system message
  - user message (image-first per Gemma 4 rule)
  - one assistant message containing a single submit_estimate tool_call,
    with reasoning="" (deliberately empty; the "no rationale" signal)
"""

from __future__ import annotations

from typing import Any

from ..schema import CanonicalTrace, RenderedExample, RenderMetadata
from ._common import (
    AtlasMetaCache,
    _assistant_tool_call,
    _parse_plane,
    _user_turn,
    build_system_prompt,
    build_tools_schema,
)


def render_sft_answer_only(
    canonical: CanonicalTrace,
    *,
    atlas_meta_cache: AtlasMetaCache,
    prefer_ground_truth: bool = False,
    ground_truth_mm: float | None = None,
) -> RenderedExample:
    """Render with no tool_steps and no rationale.

    If ``prefer_ground_truth`` is True and ``ground_truth_mm`` is provided,
    the submit args use GT instead of the teacher's ``final_answer.position_mm``.
    This gives a cleaner training signal when teaching pure pattern -> mm
    without the teacher's reasoning footprint.

    Raises
    ------
    ValueError
        If ``prefer_ground_truth=True`` and ``ground_truth_mm`` is ``None``.
        Silent fallback to the teacher value would defeat the purpose of the
        flag, so callers must commit to one path.
    """
    if prefer_ground_truth and ground_truth_mm is None:
        raise ValueError(
            "prefer_ground_truth=True requires ground_truth_mm to be provided"
        )

    if not prefer_ground_truth and canonical.final_answer is None:
        raise ValueError(
            "render_sft_answer_only without prefer_ground_truth requires "
            "final_answer to be set"
        )

    root = canonical.dataset_root
    if root is None:
        raise ValueError(
            "CanonicalTrace.dataset_root not set; load via iter_canonical_traces() "
            "or pass dataset_root= to parse_canonical_trace()"
        )

    system_prompt = build_system_prompt(
        kind=canonical.system_prompt_kind,
        atlas_name=canonical.atlas_name,
        plane=_parse_plane(canonical.plane),
        atlas_meta_cache=atlas_meta_cache,
    )
    tools = build_tools_schema(canonical.system_prompt_kind)

    use_gt = prefer_ground_truth and ground_truth_mm is not None
    if use_gt:
        position_mm = float(ground_truth_mm)
    else:
        assert canonical.final_answer is not None  # guarded above
        position_mm = canonical.final_answer.position_mm
    # final_answer.name is "submit_estimate" in v1; fall back to that literal
    # when the canonical has no teacher answer (procedurally generated trace).
    submit_name = (
        canonical.final_answer.name
        if canonical.final_answer is not None
        else "submit_estimate"
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        _user_turn(canonical.query_image_paths, canonical.user_prompt_text, root),
        _assistant_tool_call(
            "call_final_0",
            submit_name,
            {"position_mm": position_mm, "reasoning": ""},
        ),
    ]

    metadata = RenderMetadata(
        atlas_name=canonical.atlas_name,
        atlas_version=canonical.atlas_version,
        plane=canonical.plane,
        subject_id=canonical.subject_id,
        system_prompt_kind=canonical.system_prompt_kind,
    )
    return RenderedExample(
        messages=messages,
        tools=tools,
        metadata=metadata,
        # No atlas/tool images — only query images appear.
        image_paths=list(canonical.query_image_paths),
        target_mm=None,
        label_mm=None,
    )
