"""Compatibility shim package for sibling repos; use :mod:`langslice.core.spec`,
:mod:`langslice.core.state` and :mod:`langslice.agent.engine`."""

from langslice.core.spec import (
    ALL_TASKS,
    DEFAULT_TASKS,
    JobSpec,
    NonlinearSpec,
    PositionSpec,
    TransformSpec,
)
from langslice.core.state import SliceState, StackState

__all__ = [
    "ALL_TASKS",
    "DEFAULT_TASKS",
    "JobSpec",
    "NonlinearSpec",
    "PositionSpec",
    "SliceState",
    "StackState",
    "TransformSpec",
]
