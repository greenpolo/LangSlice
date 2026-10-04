"""Deprecated: the former ``linear`` package's public names (layered folder move, 2026-10-04).

Kept for SliceBench (``from langslice.linear import JobSpec``) and hosts that
read specs; the modules live in their layers now (``langslice.core.spec``,
``langslice.core.state``, ``langslice.agent.engine``). The other modules here
(``engine``, ``spec``, ``state``, ``toolbox``, ``trace``, ``transform``,
``render``) are re-export shims for SliceBench too.
"""

from typing import TYPE_CHECKING, Any

from langslice.core.spec import (
    ALL_TASKS,
    DEFAULT_TASKS,
    JobSpec,
    NonlinearSpec,
    PositionSpec,
    ReorderSpec,
    TransformSpec,
)
from langslice.core.state import SliceState, StackState

if TYPE_CHECKING:
    from langslice.agent.engine import run


def __getattr__(name: str) -> Any:
    # The engine pulls in the agent framework; load it only when a run is
    # asked for, so hosts can read specs and price runs without that cost.
    if name == "run":
        from langslice.agent.engine import run

        return run
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ALL_TASKS",
    "DEFAULT_TASKS",
    "JobSpec",
    "NonlinearSpec",
    "PositionSpec",
    "ReorderSpec",
    "SliceState",
    "StackState",
    "TransformSpec",
    "run",
]
