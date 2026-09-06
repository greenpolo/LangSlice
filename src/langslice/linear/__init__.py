"""Linear: order, position and one in-plane transform per section.

One agent environment over one stack of histology sections — one state, one
toolbox, one job statement — that a host switches on task by task
(``docs/linear_design.md``).

    from langslice.linear import JobSpec, run
    state = asyncio.run(run(JobSpec(image_folder="...", atlas="allen_mouse_25um")))
"""

from langslice.linear.engine import run
from langslice.linear.spec import ALL_TASKS, JobSpec, PositionSpec, ReorderSpec, TransformSpec
from langslice.linear.state import SliceState, StackState

__all__ = [
    "ALL_TASKS",
    "JobSpec",
    "PositionSpec",
    "ReorderSpec",
    "SliceState",
    "StackState",
    "TransformSpec",
    "run",
]
