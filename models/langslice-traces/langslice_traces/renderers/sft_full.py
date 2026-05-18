"""Render a CanonicalTrace as a full SFT training example.

Byte-identical to ``models/langslice-gemma-4/training/sft/render.py::render_example``
for equivalent input data. The golden-equality test in
``tests/test_langslice_traces_factory.py`` pins this property.
"""

from __future__ import annotations

from ..schema import CanonicalTrace, RenderedExample
from ._common import AtlasMetaCache, _assistant_tool_call
from ._prefix import _render_prefix


def render_sft_full(
    canonical: CanonicalTrace,
    *,
    atlas_meta_cache: AtlasMetaCache,
) -> RenderedExample:
    """Translate a CanonicalTrace to HF chat-template messages + tools.

    Byte-identical to ``models/langslice-gemma-4/training/sft/render.py::render_example``
    when fed equivalent inputs. ``CanonicalTrace`` is the SFT trainer's
    ``Example`` shape with the raw ``trace`` list pre-decomposed into typed
    ``tool_steps`` + a separate ``final_answer``; the rendered output is
    indistinguishable.

    Delegates the non-terminal prefix walk to ``_render_prefix`` and then
    appends the terminal ``submit_estimate`` assistant tool_call. The terminal
    call_id uses the ``call_final_{n}`` convention where ``n`` is the index
    one past the last tool_step (matches the original ``example.trace[-1]``
    index).

    Raises
    ------
    ValueError
        If ``canonical.final_answer`` is ``None`` (e.g. a procedurally generated
        prefix). The full SFT format needs the terminal submit step.
    """
    if canonical.final_answer is None:
        raise ValueError(
            "render_sft_full requires final_answer to be set"
        )
    rendered = _render_prefix(canonical, atlas_meta_cache=atlas_meta_cache)

    # Terminal assistant tool_call for submit_estimate — no matching tool message.
    # ``_render_prefix`` used ``call_{i}`` ids for i in [0, len(tool_steps)), so
    # the terminal id ``call_final_{len(tool_steps)}`` is guaranteed unique.
    terminal_index = len(canonical.tool_steps)
    terminal_call_id = f"call_final_{terminal_index}"
    rendered.messages.append(
        _assistant_tool_call(
            terminal_call_id,
            canonical.final_answer.name,
            {
                "position_mm": canonical.final_answer.position_mm,
                "reasoning": canonical.final_answer.reasoning,
            },
        )
    )
    return rendered
