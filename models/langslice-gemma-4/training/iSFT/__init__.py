"""Expert Iteration SFT — single-round MVP.

See ``docs/training_overview.md`` for the public training map. This package
implements the expert-iteration loop: rollout, score, filter, trace-format, and
append rollouts to a langslice-native SFT JSONL.

Multi-round chaining, vLLM lifecycle management, curriculum sampling, and
atlas-embedding precompute live in subsequent waves and are intentionally
absent from this module.
"""
