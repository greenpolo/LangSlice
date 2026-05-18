"""Single-turn GRPO trainer for LangSlice (Lane A: terminal-state final-answer RL).

Sibling lane to :mod:`rlvr` (the parked multi-turn agent trainer). Where ``rlvr``
asks the policy to learn the whole tool-use loop, ``single_turn_rl`` trains only
the final coordinate decision: given a query image plus the atlas evidence the
production agent already gathered, output strict-JSON ``{"position_mm": <float>}``
and get scored by the same axis-normalized truncated-Gaussian reward.

See ``models/langslice-gemma-4/training/single_turn_rl/README.md``.
"""

from __future__ import annotations
