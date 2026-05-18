"""Render a CanonicalTrace as an RL rollout prefix.

The policy is trained to generate the submit step from this prefix; reward
is computed against ``target_mm`` (caller-provided GT from manifest join).
"""

from __future__ import annotations

from ..schema import CanonicalTrace, RenderedExample
from ._common import AtlasMetaCache
from ._prefix import _render_prefix


def render_rl_prefix(
    canonical: CanonicalTrace,
    *,
    atlas_meta_cache: AtlasMetaCache,
    ground_truth_mm: float,
) -> RenderedExample:
    """Emit everything up to but not including the terminal submit.

    The submit step is what GRPO will sample; ``target_mm`` is the reward target.
    Callers must pass ``ground_truth_mm`` from a manifest join — never use the
    teacher's ``final_answer.position_mm`` here (it's the teacher's guess, not GT).
    """
    rendered = _render_prefix(canonical, atlas_meta_cache=atlas_meta_cache)
    rendered.target_mm = float(ground_truth_mm)
    return rendered
