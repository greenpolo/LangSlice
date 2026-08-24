"""Whole-brain multi-slice estimation: a node graph over one stack state."""

from langslice.linear.whole_brain.engine import (
    CYCLE_LIMITS,
    EngineContext,
    build_context,
    run_brain,
    run_nodes,
)
from langslice.linear.whole_brain.state import BrainConfig, SliceState, StackState

__all__ = [
    "BrainConfig",
    "CYCLE_LIMITS",
    "EngineContext",
    "SliceState",
    "StackState",
    "build_context",
    "run_brain",
    "run_nodes",
]
