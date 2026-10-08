"""The job statement every door gives a registration agent, in one place.

LangSlice's own agent (:mod:`langslice.agent.engine`), the MCP door
(:mod:`langslice.doors.mcp.server`, ``start_job``) and the agent CLI
(``langslice-job FOLDER brief``, :mod:`langslice.doors.cli.job`) give the
agent the same job: the statement :func:`langslice.agent.prompt.build_job_statement`
words (job, run facts, one line per tool, constraints, method), the opening
pictures (:func:`langslice.core.opening.opening_items`), the status table with
the recent run notes (:func:`status_and_notes`) and the user's notes
(:func:`read_notes`, kept in ``job.json`` so every door reads them).

What differs is only what each door must say differently, chosen by its
name (``agent``, ``mcp``, ``cli``: :data:`langslice.doors.declarations.DOORS`):
where the opening pictures are (the seed message; ``show_stack`` pages;
picture files the ``brief`` command saved) and how a long call answers. The
ADK agent gets the statement as its system instruction and the pictures and
:func:`status_and_notes` as its first message; a host-owned door
(:func:`job_statement`) gets everything as one text, the pictures delivered
its own way.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any

from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.core.status import status_text

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.layout import JobLayout

#: Said when the job's image model is not connected to LangSlice here.
IMAGE_MODEL_OFF = (
    "The image-model tool (trace_borders) is off: no image model is connected to "
    "LangSlice (no key or login for this job's image provider). Fit each "
    "section's deformation to its stain with ants_syn."
)

#: What the opening pictures show, said by every door that names them.
OPENING_SHOWS = (
    "They show every section in strips, in the stack's current order, each labelled, "
    "with the atlas at its current position beneath it at the same scale when the "
    "stack has positions; "
    "when a section has none, atlas reference strips at evenly spaced positions follow."
)

#: How many run notes the status text repeats (the newest).
RECENT_NOTES = 12

#: ``job.json``'s field holding the user's notes for the job (a saved ABBA
#: job's notes, ``langslice-job FOLDER init --notes``).
NOTES_KEY = "notes"


def status_and_notes(state: StackState) -> str:
    """The status table with its header, then the recent run notes: the text
    the ADK agent's first message ends with, and every other door's statement."""
    return (
        "Status table (filename, position, spacing to "
        "the next placed section, flags):\n"
        f"{status_text(state)}\n\n"
        "Recent run notes:\n"
        + ("\n".join(f"- {note}" for note in state.notes[-RECENT_NOTES:]) or "- (none)")
    )


def user_notes_lines(notes: str) -> list[str]:
    """The user's notes as statement lines (none when empty)."""
    return ["", "User notes:", notes] if notes.strip() else []


def read_notes(layout: JobLayout) -> str:
    """The user's notes for the job in ``job.json`` (``notes``); empty when none."""
    from langslice.job.layout import read_job_file

    try:
        record = read_job_file(layout) or {}
    except (OSError, ValueError):
        return ""
    return str(record.get(NOTES_KEY) or "")


def opening_for_mcp(pages: int) -> list[str]:
    """The MCP door's opening paragraph: the ``show_stack`` pages to read."""
    return [
        "Opening pictures: read every page with show_stack(page=1) through "
        f"show_stack(page={pages}) before any write. " + OPENING_SHOWS,
    ]


def opening_for_cli(folder: str, pictures: int | None) -> list[str]:
    """The agent CLI's opening paragraph and how its commands behave.

    *pictures* is how many opening pictures ``brief`` saved (None: not saved
    yet; ``init`` says how to)."""
    if pictures is None:
        opening = (f"Opening pictures: `langslice-job {folder} brief` saves them as "
                   "picture files; open every one before any write. ")
    else:
        opening = (f"Opening pictures: `brief` saved {pictures} picture files, listed in "
                   "reading order under `artifacts` (kind `opening`) and in BRIEF.md "
                   "with the text that goes with each; open every one before any write. ")
    return [
        opening + OPENING_SHOWS,
        "",
        "Commands:",
        f"- Every tool is a command: `langslice-job {folder} TOOL --args '{{...}}'` (or "
        "`--name value`); `langslice-job schema TOOL` gives its arguments and description. "
        "Pictures come back as files under `artifacts` (kind `view`, in the order the "
        "reply's `pictures` lists their numbers); open them to see them.",
        "- `--dry-run` runs a write without writing and reports what would change; "
        "the fits and trace_borders are only checked, not run.",
        f"- Long tools ({', '.join(long_verbs())}) can run with `--background`, which "
        "answers at once with a run id; `wait ID` collects the answer.",
        "- Commands may run in parallel: each write holds the job folder's lock, and a "
        "long tool re-checks each section before it applies (STALE_INPUT when the "
        "section changed meanwhile: run it again for that section).",
    ]


def long_verbs() -> list[str]:
    """The listed verbs that compute outside the job lock and may take long
    (:attr:`langslice.ops.registry.Verb.long`), in registry order."""
    from langslice.ops.registry import listed

    return [name for name, verb in listed().items() if verb.long]


def job_statement(
    spec: JobSpec, state: StackState, ctx: Workspace, *, door: str,
    tool_names: list[str], opening: list[str], notes: str = "",
    max_resolution: int | None = None, image_model_off: bool = False,
    auto: bool | None = None, gates: bool = True,
) -> str:
    """The whole statement a host-owned door (MCP, the agent CLI) gives:
    the job statement the ADK agent is given, worded for *door*, then
    *opening* (the door's own paragraph on its opening pictures), the
    image-model line when *image_model_off*, the user's *notes*, and the
    status table with the recent run notes.

    *image_model_off*: the job's image model is not connected here
    (:func:`langslice.doors.api.setup.image_model_connected`), so the
    statement is that of a run without one, plus :data:`IMAGE_MODEL_OFF`.
    """
    from langslice.agent.prompt import build_job_statement, display_facts

    low, high = ctx.position_range
    if image_model_off:
        spec = replace(spec, nonlinear=replace(spec.nonlinear, provider="none"))
    lines = [
        build_job_statement(
            spec, state, tool_names=tool_names, species=ctx.species,
            pos_lo=low, pos_hi=high, axis_ends=ctx.axis_ends,
            max_resolution=max_resolution, door=door, auto=auto, gates=gates,
            **display_facts(ctx, state),
        ),
    ]
    if opening:
        lines += ["", *opening]
    if image_model_off:
        lines += ["", IMAGE_MODEL_OFF]
    lines += user_notes_lines(notes)
    lines += ["", status_and_notes(state)]
    return "\n".join(lines)


def image_model_state(spec: JobSpec, *, connected: bool) -> dict[str, Any] | None:
    """The job's image model as the status and the brief report it: the
    provider, whether it is connected here, whether ``trace_borders`` is
    offered (None without the nonlinear task)."""
    if not spec.has("nonlinear"):
        return None
    uses = spec.nonlinear.uses_image_model
    return {"provider": spec.nonlinear.provider, "connected": bool(uses and connected),
            "trace_borders": bool(uses and connected)}


__all__ = [
    "IMAGE_MODEL_OFF", "NOTES_KEY", "OPENING_SHOWS", "image_model_state", "job_statement",
    "opening_for_cli", "opening_for_mcp", "read_notes", "status_and_notes", "user_notes_lines",
]
