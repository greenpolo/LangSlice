"""The job statement every door gives a registration agent, in one place.

LangSlice's own agent (:mod:`langslice.agent.engine`), the MCP door
(:mod:`langslice.doors.mcp.server`, ``start_job``) and the agent CLI
(``langslice-job FOLDER launch``, :mod:`langslice.doors.cli.launch`) give the
agent the same job: the statement :func:`langslice.agent.prompt.build_job_statement`
words (job, run facts, one line per tool, constraints, method), the opening
pictures (:func:`langslice.core.opening.opening_items`), the status table with
the recent run notes (:func:`status_and_notes`) and the user's notes
(:func:`read_notes`, kept in ``job.json`` so every door reads them).

What differs is only what each door must say differently, chosen by its
name (``agent``, ``mcp``, ``cli``: :data:`langslice.doors.declarations.DOORS`):
where the opening pictures are (the seed message; ``show_stack`` pages;
picture files the ``launch`` command saved) and how a long call answers. The
ADK agent's system prompt is :data:`SYSTEM_PROMPT`, the same for every job;
its first message is the statement, the user's notes, the pictures and
:func:`status_and_notes`. A host-owned door (:func:`job_statement`) gets
everything as one text, the pictures delivered its own way.
"""

from __future__ import annotations

import copy
from collections.abc import Collection
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from langslice.core.needs import render
from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.core.status import status_text

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.layout import JobLayout

#: Said when the job's image model is not connected to LangSlice here,
#: written with what it needs (:mod:`langslice.core.needs`).
#: The registration agent's system prompt: true of every job, naming no tool a
#: job can switch off; the job itself comes in the first message.
SYSTEM_PROMPT = (
    "You are an expert neuroanatomist registering one brain with LangSlice: a stack "
    "of histology sections from one brain, placed in a BrainGlobe atlas. The first "
    "message gives the job: what to do, the tools, and the facts about this stack and "
    "atlas. Work the job with the tools and finish with `submit`."
)

_IMAGE_MODEL_OFF = (
    "[[?trace_borders The image-model tool (trace_borders) is off: no image model is "
    "connected to LangSlice (no key or login for this job's image provider).[[?fireants_syn  "
    "Fit each section's deformation to its stain with fireants_syn.]]]]"
)
#: ... as a job that disables nothing reads it.
IMAGE_MODEL_OFF = render(_IMAGE_MODEL_OFF)


def image_model_off_text(disabled: Collection[str] = ()) -> str:
    """:data:`IMAGE_MODEL_OFF` for a job with *disabled* (empty when the job
    has no ``trace_borders`` to be off)."""
    return render(_IMAGE_MODEL_OFF, disabled)

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


def opening_for_cli(folder: str, listing: list[str],
                    disabled: Collection[str] = (), spec: Any = None) -> list[str]:
    """The agent CLI's opening paragraph and how its commands behave.

    *listing* is the opening in reading order: each strip's text line, then
    ``- <absolute path>`` of its picture file. *disabled* is the job's
    disabled list; *spec* the job's, whose plugin tools are listed with the
    long ones."""
    opening = [
        "Opening pictures, saved as files, in reading order, each after the text that "
        "goes with it. " + OPENING_SHOWS, "", *listing, "",
        "Open every one of these pictures before any write.",
    ]
    if spec is not None:  # a verb the run does not have is not mentioned either
        from langslice.ops.registry import enabled, verbs_for

        run = set(enabled(spec, scripting=True))
        disabled = [*disabled, *(name for name in verbs_for(spec) if name not in run)]
    return [
        *opening,
        "",
        "Commands:",
        f"- Every tool is a command: `langslice-job {folder} TOOL --args '{{...}}'` (or "
        f"`--name value`); `langslice-job {folder} schema TOOL [TOOL ...]` gives their "
        "arguments and descriptions, several at once. Pictures come back as files: "
        "`picture_files` lists them first, `artifacts` again with their kinds (`view`, in "
        "the order the reply's `pictures` lists their numbers); open them to see them.",
        render("- `--dry-run` runs a write without writing and reports what would change"
               "[[?elastix_affine|fireants_syn|trace_borders ; the "
               "[[#elastix_affine|fireants_syn=fits||trace_borders=trace_borders]] "
               "are only checked, not run]]" + (
                   "; a plugin's engine tools are only checked too" if getattr(
                       spec, "plugins", None) else "") + ".", disabled),
        f"- Long tools ({', '.join(long_verbs(disabled, spec))}): a call answers with the "
        "result or, still running after 30 s, with its run id and state `running`; the run "
        "goes on (also when your shell ends the command, where a process may outlive it) "
        "and `wait ID` answers within 30 s with its answer or its state (wait again). "
        "`--background` answers at once with the run id; `runs` lists the runs. Do not start "
        "a fit again while its run goes on.",
        "- Commands may run in parallel: each write holds the job folder's lock, and a "
        "long tool re-checks each section before it applies (STALE_INPUT when the "
        "section changed meanwhile: run it again for that section).",
    ]


def long_verbs(disabled: Collection[str] = (), spec: Any = None) -> list[str]:
    """The verbs that compute outside the job lock and may take long
    (:attr:`langslice.ops.registry.Verb.long`), in registry order, less the
    ones *disabled* names and, given *spec*, the ones its run does not have."""
    from langslice.core.toolset import is_off
    from langslice.ops.registry import verbs_for

    return [name for name, verb in verbs_for(spec).items()
            if verb.long and not is_off(disabled, name)
            and (spec is None or verb.tool.is_on(spec))]


def job_statement(
    spec: JobSpec, state: StackState, ctx: Workspace, *, door: str,
    tool_names: list[str], opening: list[str], notes: str = "",
    max_resolution: int | None = None, image_model_off: bool = False,
    auto: bool | None = None,
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
    if image_model_off:  # a copy that is not checked again: it is this door's view
        spec = copy.copy(spec)
        spec.nonlinear = replace(spec.nonlinear, provider="none")
    lines = [
        build_job_statement(
            spec, state, tool_names=tool_names, species=ctx.species,
            pos_lo=low, pos_hi=high, axis_ends=ctx.axis_ends,
            max_resolution=max_resolution, auto=auto,
            **display_facts(ctx, state),
        ),
    ]
    if opening:
        lines += ["", *opening]
    if image_model_off and (off_text := image_model_off_text(spec.switched_off)):
        lines += ["", off_text]
    lines += user_notes_lines(notes)
    lines += ["", status_and_notes(state)]
    return "\n".join(lines)


def image_model_state(spec: JobSpec, *, connected: bool) -> dict[str, Any] | None:
    """The job's image model as the status and ``init`` report it: the
    provider, whether it is connected here, whether ``trace_borders`` is
    offered (None without the nonlinear task)."""
    if not spec.has("nonlinear"):
        return None
    uses = spec.nonlinear.uses_image_model
    return {"provider": spec.nonlinear.provider, "connected": bool(uses and connected),
            "trace_borders": bool(uses and connected)}


__all__ = [
    "IMAGE_MODEL_OFF", "NOTES_KEY", "OPENING_SHOWS", "SYSTEM_PROMPT", "image_model_off_text",
    "image_model_state", "job_statement", "opening_for_cli", "opening_for_mcp", "read_notes",
    "status_and_notes", "user_notes_lines",
]
