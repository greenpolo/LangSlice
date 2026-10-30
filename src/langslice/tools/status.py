"""The ``status`` tool: the stack as it stands, as data.

Writes nothing and takes no undo step. :func:`stack_status` is the table
every write's reply quotes rows of; :func:`status` is the tool, which adds
what the run holds besides the rows: the raw channels, the made section
images, the background work still running and what this run lets the agent
change. The pictures are ``look``'s (:mod:`langslice.tools.look`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langslice.core.sections import images as section_images
from langslice.core.stack.stack import break_ids
from langslice.core.stack.status import stack_angles_entry, status_rows
from langslice.job.settings import frozen as freeze
from langslice.job.settings.tool_table import TOOLS, tools_for

if TYPE_CHECKING:
    from langslice.job.job import Job
    from langslice.tools.run import Run


@dataclass(frozen=True)
class StackStatus:
    """The stack as it stands: one row per section in corrected order
    (:func:`langslice.core.stack.status.status_rows`), the cutting angles and
    the
    interval breaks (the filenames of the sections after each gap).
    ``cutting_angles_deg`` is the stack's ``{"pitch", "yaw"}``, or ``"per
    section"`` when the sections' angles differ (each row then carries its
    own)."""

    rows: list[dict[str, Any]]
    cutting_angles_deg: dict[str, float] | str
    interval_breaks: list[str]


def status(run: Run) -> dict[str, object]:
    """Describe the stack and the run; return the ``status`` reply.

    Parameters
    ----------
    run
        The run the tool is called in.

    Returns
    -------
    dict
        The status rows, the cutting angles and interval breaks, the raw
        channels, the section images, the background work running (when
        any) and what the run lets the agent change.
    """
    job = run.job
    reply: dict[str, object] = {"status": "ok", **run.rows()}
    reply["raw_channels"] = raw_channels(run)
    reply["section_images"] = section_images.summary(
        job.state, run.ctx.settings
    )
    running = [work.summary() for work in job.background.running()]
    if running:
        reply["background_running"] = running
    reply["can_change"] = can_change(run)
    return reply


def stack_status(job: Job) -> StackStatus:
    """The status table (``status``). A section with properties the user froze
    (``JobSpec.frozen``) carries them as ``frozen: [...]``, one the user gave
    a damage note (``inputs.damaged``: a note, not damage)
    ``damage_by_user: true``; the others carry neither. A section holding a
    listed registration carries its number, ``registration``
    (:mod:`langslice.job.registrations`)."""
    state = job.state
    rows = status_rows(state)
    held = freeze.sections_of(job.spec.frozen, [row["id"] for row in rows])
    for row in rows:
        if row["id"] in held:
            row["frozen"] = held[row["id"]]
        if row["id"] in job.host_damaged:
            row["damage_by_user"] = True
        record = state.by_id(row["id"])
        number = job.registrations.number_of(state, record) if record is not None else None
        if number is not None:
            row["registration"] = number
    return StackStatus(rows=rows, cutting_angles_deg=stack_angles_entry(state),
                       interval_breaks=break_ids(state))


def raw_channels(run: Run) -> list[str] | dict[str, list[str]]:
    """The raw channels for ``status``: one list when every section shares
    them, else ``{filename: [names]}``."""
    named = {record.id: section_images.raw_names(run.ctx, record.id)
             for record in run.job.state.in_order()}
    distinct = {tuple(names) for names in named.values()}
    return list(next(iter(distinct))) if len(distinct) == 1 else named


def can_change(run: Run) -> list[str]:
    """What this run lets the agent change, in words (``status``)."""
    spec = run.spec

    def has(name: str) -> bool:
        return TOOLS[name].is_on(spec)

    frozen = spec.frozen_everywhere
    out: list[str] = []
    if spec.has("position") and "position" not in frozen:
        out.append("positions (the order follows them)")
    if has("position_sections") and "angles" not in frozen:
        out.append("cutting angles")
    if spec.has("transform"):
        if has("interactive_transform"):
            by_hand: list[str] = []
            if "orientation" not in frozen:
                by_hand.append("orientation" + ("" if spec.off("interactive_transform:flip")
                                                else " and mirroring"))
            if "affine" not in frozen:
                by_hand.append("in-plane transforms")
            out.append(" and ".join(by_hand) + " by hand")
        if has("elastix_affine"):
            out.append("in-plane transforms by elastix_affine")
    if has("mark_damage"):
        out.append("damaged regions")
    if spec.has("nonlinear"):
        fits = [name for name in ("fireants_syn", "trace_borders")
                if has(name) and (name != "trace_borders" or run.traces_on)]
        if has("set_landmarks") and has("landmark_triangulation"):
            fits.append("landmark pairs by set_landmarks, applied by landmark_triangulation")
        out.append("deformations" + (f" ({', '.join(fits)})" if fits else ""))
    if has("make_image"):
        out.append("the section images the pictures and fits read (make_image)")
    if has("make_mask"):
        out.append("masks for the fits")
    added = [name for name, tool in tools_for(spec).items()
             if name not in TOOLS and tool.kind == "write" and tool.is_on(spec)]
    if added:  # tools tool plugins added (a job without plugins has none)
        out.append(f"whatever the plugin tools change ({', '.join(added)})")
    return out
