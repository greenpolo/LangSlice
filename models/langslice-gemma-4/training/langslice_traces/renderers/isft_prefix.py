"""Render a CanonicalTrace as an iSFT prefix with the teacher's final answer as label.

Shares the prefix-construction logic with ``rl_prefix``; differs only in what
goes into the supervision target.
"""

from __future__ import annotations

from ..schema import CanonicalTrace, RenderedExample
from ._common import AtlasMetaCache
from ._prefix import _render_prefix


def render_isft_prefix(
    canonical: CanonicalTrace,
    *,
    atlas_meta_cache: AtlasMetaCache,
) -> RenderedExample:
    """Emit prefix; supervised label is the teacher's submit position.

    Symmetric with ``render_rl_prefix`` but with ``label_mm`` (supervised target)
    set to the teacher's ``final_answer.position_mm``, since iSFT trains the
    answer step in supervised mode rather than via reward.

    For procedurally generated traces with ``final_answer=None``, ``label_mm``
    is left as ``None`` — the caller is responsible for supplying a label by
    other means (e.g. manifest GT join). The prefix itself still renders.
    """
    rendered = _render_prefix(canonical, atlas_meta_cache=atlas_meta_cache)
    if canonical.final_answer is not None:
        rendered.label_mm = canonical.final_answer.position_mm
    return rendered
