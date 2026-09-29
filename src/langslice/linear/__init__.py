"""Linear: order, position and one in-plane transform per section.

One agent environment over one stack of histology sections — one state, one
toolbox, one job statement — that a host switches on task by task
(``docs/linear_design.md``).

    from langslice.linear import JobSpec, run
    state = asyncio.run(run(JobSpec(image_folder="...", atlas="allen_mouse_25um")))
"""

from typing import TYPE_CHECKING, Any

from langslice.linear.spec import (
    ALL_TASKS,
    DEFAULT_TASKS,
    JobSpec,
    NonlinearSpec,
    PositionSpec,
    ReorderSpec,
    TransformSpec,
)
from langslice.linear.state import SliceState, StackState

if TYPE_CHECKING:
    from langslice.linear.engine import run


def __getattr__(name: str) -> Any:
    # The engine pulls in the agent framework; load it only when a run is
    # asked for, so hosts can read specs and price runs without that cost.
    if name == "run":
        from langslice.linear.engine import run

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
