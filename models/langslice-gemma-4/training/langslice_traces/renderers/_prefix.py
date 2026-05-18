"""Prefix-walking helper shared by render_rl_prefix and render_isft_prefix."""

from __future__ import annotations

from typing import Any

from ..schema import CanonicalTrace, RenderedExample, RenderMetadata
from ._common import (
    AtlasMetaCache,
    _assistant_tool_call,
    _parse_plane,
    _tool_response,
    _user_turn,
    build_system_prompt,
    build_tools_schema,
)


def _render_prefix(
    canonical: CanonicalTrace,
    *,
    atlas_meta_cache: AtlasMetaCache,
) -> RenderedExample:
    """Build the prefix: system + user + each ToolStep's assistant tool_call + tool response.

    NO terminal submit step. The last message is the final tool response (or the
    user message if ``canonical.tool_steps`` is empty). Both ``target_mm`` and
    ``label_mm`` are ``None`` on the returned RenderedExample — caller sets the
    appropriate field.

    Mirrors the non-terminal walk in ``render_sft_full`` exactly, including the
    ``"call_{i}"`` id convention for tool_steps, so an SFT-full and a prefix
    rendered from the same canonical share their first ``2 + 2*N`` messages
    byte-for-byte (N = len(tool_steps)).
    """
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

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        _user_turn(canonical.query_image_paths, canonical.user_prompt_text, root),
    ]
    # Order matches render_sft_full: query images in the user turn, then
    # tool_result images in trace order. The terminal submit step is omitted
    # entirely — that's what the policy/iSFT step will produce.
    image_paths_in_order: list[str] = list(canonical.query_image_paths)

    seen_ids: set[str] = set()
    for i, step in enumerate(canonical.tool_steps):
        call_id = f"call_{i}"
        if call_id in seen_ids:
            raise RuntimeError(
                f"duplicate tool_call_id {call_id!r} at trace step {i}"
            )
        seen_ids.add(call_id)
        messages.append(_assistant_tool_call(call_id, step.call_name, step.call_args))
        messages.append(
            _tool_response(call_id, step.result_image_paths, step.result_text, root)
        )
        image_paths_in_order.extend(step.result_image_paths)

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
        image_paths=image_paths_in_order,
        target_mm=None,
        label_mm=None,
    )
