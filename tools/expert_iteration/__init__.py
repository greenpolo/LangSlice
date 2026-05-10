"""Expert Iteration SFT — single-round MVP.

See ``docs/`` and ``C:/Users/WalshLab/.claude/plans/agile-honking-cook.md`` for
the full plan. This package implements **Component 1** only: rollout → score →
filter → trace-format → append rollouts to a langslice-native SFT JSONL.

Multi-round chaining, vLLM lifecycle management, curriculum sampling, and
atlas-embedding precompute live in subsequent waves and are intentionally
absent from this module.
"""
