"""What every tool body of one run shares: the job, the workspace, the run's settings.

The tool bodies (:mod:`.looking`, :mod:`.changing`, :mod:`.fitting`) are
closures over one :class:`Door`: the job and the workspace they work on,
the toolbox's door-side record (:class:`langslice.doors.tools.toolbox.ToolBox`:
the look-before-commit gates), the run's picture size, the image model, and
the small helpers every body words its reply with (the status rows a write
changed, a section list resolved, a refusal as a payload).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.core.status import compact_rows
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.job.job import Job
from langslice.ops import views as ops_views
from langslice.ops.look import Looked

if TYPE_CHECKING:  # import cycle: the engine builds the toolbox
    from langslice.agent.engine import EngineContext
    from langslice.doors.tools.toolbox import ToolBox
    from langslice.providers.registry import ImageModel

#: One item of a tool's media list (under ``TOOL_MEDIA_PARTS_KEY``): a
#: picture (captioned PIL image) or a line of text, in reading order.
Media = Image.Image | str


@dataclass
class Door:
    """One run's tool door, as every tool body sees it."""

    job: Job
    ctx: EngineContext
    spec: JobSpec
    box: ToolBox
    #: The picture-size level the tools work at (``core.sizes``).
    level: str
    #: The look-before-commit gates of ``position.gated`` apply here.
    gated: bool
    #: The image model is in this run and reachable (``trace_borders``).
    traces_on: bool
    image_model: ImageModel | None
    #: Refuses a call naming more sections than the job lets one transform
    #: call take (``Job.over_cap``), or None.
    over_cap: Callable[[int], dict[str, Any] | None]

    @property
    def state(self) -> StackState:
        return self.job.state

    # --- the status rows -------------------------------------------------------

    def rows(self) -> dict[str, Any]:
        """The whole status table, the cutting angles and the interval breaks."""
        table = ops_views.status(self.job)
        return {"rows": compact_rows(table.rows),
                "cutting_angles_deg": table.cutting_angles_deg,
                "interval_breaks": table.interval_breaks}

    def changed(self, touched: Sequence[str]) -> dict[str, Any]:
        """The status rows of the sections a write touched, and nothing else
        (``status`` is the whole table, one call away)."""
        wanted = set(touched)
        table = ops_views.status(self.job)
        return {"changed": compact_rows([row for row in table.rows if row["id"] in wanted]),
                "n_sections": len(self.state.slices),
                "cutting_angles_deg": table.cutting_angles_deg,
                "interval_breaks": table.interval_breaks}

    def answered(self, *touched: str) -> dict[str, Any]:
        """A write's answer: ``ok`` and the rows it changed."""
        return {"status": "ok", **self.changed(list(touched))}

    # --- sections ---------------------------------------------------------------

    def resolve_many(self, refs: Sequence[Any]) -> tuple[list[SliceState], list[str]]:
        """The sections *refs* name (each once, in order) and the refs that name none."""
        known: list[SliceState] = []
        unknown: list[str] = []
        for ref in refs:
            record = self.state.resolve(ref)
            if record is None:
                unknown.append(str(ref))
            elif all(held.id != record.id for held in known):
                known.append(record)
        return known, unknown

    def forget_looks(self, moved: Sequence[str]) -> None:
        """A position moved (by a write, undo, redo or a reload) is a write to
        it: that section needs a new look, and the stack a new review (the
        gates)."""
        for name in moved:
            self.box.compared.pop(name, None)
        if moved:
            self.box.reviewed = False


def pictured(result: dict[str, Any], looked: Looked) -> dict[str, Any]:
    """*result* with the pictures :mod:`langslice.ops.look` drew and saved:
    ``pictures`` (``{"id", "caption"}`` each, in the order they are
    attached), ``not_shown`` and the pictures themselves. The tool door does
    not save them again (they carry ``pictures``)."""
    result["pictures"] = list(looked.entries)
    if looked.not_shown:
        result["not_shown"] = list(looked.not_shown)
    if looked.failed:
        result["render_failed"] = list(looked.failed)
    result[TOOL_MEDIA_PARTS_KEY] = list(looked.pictures)
    return result


def as_list(value: Any) -> list[Any]:
    """A model's list argument as a list (None is empty; a single item is a list of one)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


__all__ = ["Door", "Media", "as_list", "pictured"]
