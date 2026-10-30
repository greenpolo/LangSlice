"""What every tool body of one run shares: the job, the workspace, the run's settings.

Every tool body works over one :class:`Run`: the job and the workspace
they work on, the toolbox, the run's picture size, the image model, and the
small helpers every body words its reply with (the status rows a write
changed, a section list resolved). A run sits behind one door (the way in:
the built-in agent, MCP, the agent CLI, the library).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.stack.section import Section
from langslice.core.stack.stack import Stack
from langslice.core.stack.status import compact_rows
from langslice.job.job import Job
from langslice.job.settings.spec import JobSpec
from langslice.tools.status import stack_status

if TYPE_CHECKING:  # import cycle: the engine builds the toolbox
    from langslice.core.fits.trace.request import ImageModel
    from langslice.doors.agent.engine import EngineContext
    from langslice.tools.look import Looked
    from langslice.tools.pipeline import ToolBox

#: One item of a tool's media list (under ``TOOL_MEDIA_PARTS_KEY``): a
#: picture (captioned PIL image) or a line of text, in reading order.
Media = Image.Image | str


@dataclass
class Run:
    """One run, as every tool body sees it."""

    job: Job
    ctx: EngineContext
    spec: JobSpec
    box: ToolBox
    #: The picture-size level the tools work at (``core.pictures.sizes``).
    level: str
    #: The image model is in this run and reachable (``trace_borders``).
    traces_on: bool
    image_model: ImageModel | None

    @property
    def state(self) -> Stack:
        return self.job.state

    # --- the status rows -------------------------------------------------------

    def rows(self) -> dict[str, Any]:
        """The whole status table, the cutting angles and the interval breaks."""
        table = stack_status(self.job)
        return {"rows": compact_rows(table.rows),
                "cutting_angles_deg": table.cutting_angles_deg,
                "interval_breaks": table.interval_breaks}

    def changed(self, touched: Sequence[str]) -> dict[str, Any]:
        """The status rows of the sections a write touched, and nothing else
        (``status`` is the whole table, one call away)."""
        wanted = set(touched)
        table = stack_status(self.job)
        return {"changed": compact_rows([row for row in table.rows if row["id"] in wanted]),
                "n_sections": len(self.state.slices),
                "cutting_angles_deg": table.cutting_angles_deg,
                "interval_breaks": table.interval_breaks}

    def answered(self, *touched: str) -> dict[str, Any]:
        """A write's answer: ``ok`` and the rows it changed."""
        return {"status": "ok", **self.changed(list(touched))}

    # --- sections ---------------------------------------------------------------

    def resolve_many(self, refs: Sequence[Any]) -> tuple[list[Section], list[str]]:
        """The sections *refs* name (each once, in order) and the refs that name none."""
        known: list[Section] = []
        unknown: list[str] = []
        for ref in refs:
            record = self.state.resolve(ref)
            if record is None:
                unknown.append(str(ref))
            elif all(held.id != record.id for held in known):
                known.append(record)
        return known, unknown

def pictured(result: dict[str, Any], looked: Looked) -> dict[str, Any]:
    """*result* with the pictures :mod:`langslice.tools.look` drew and saved:
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


def spread(records: Sequence[Any], cap: int | None) -> list[Any]:
    """At most *cap* of *records* (None: all), spread evenly over them (the
    sections a change tool pictures when the call named none)."""
    if cap is None or len(records) <= cap:
        return list(records)
    step = (len(records) - 1) / max(cap - 1, 1)
    picks = sorted({int(round(index * step)) for index in range(cap)})
    return [records[index] for index in picks]


def drawn_image(section_image: Any) -> list[str]:
    """The ``images`` a fit's or a mask's picture is drawn with: the section
    image it read (none for ``preprocessed``, ``look``'s default)."""
    name = str(section_image or "preprocessed").strip()
    return [] if name == "preprocessed" else [name]


def as_list(value: Any) -> list[Any]:
    """A model's list argument as a list (None is empty; a single item is a list of one)."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


__all__ = ["Media", "Run", "as_list", "pictured"]
