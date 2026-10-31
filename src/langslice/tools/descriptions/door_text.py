"""What each door adds after the job statement: its opening, the notes.

The agent CLI and MCP give the statement as one text: the statement
(:mod:`~langslice.tools.descriptions.statement`), then the door's own
paragraph on its opening pictures (:func:`opening_for_mcp`,
:func:`opening_for_cli`), the image-model line when the job's image model is
not connected here, the user's notes (``job.json``'s ``notes``) and the
status table with the recent run notes (:func:`status_and_notes`). The ADK
agent's first message has the statement and the user's notes, then the
opening pictures, then the status table, as parts of their own.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence

from langslice.core.stack.stack import Stack
from langslice.core.stack.status import status_text
from langslice.job.layout import JobLayout, read_job_file
from langslice.job.settings.needs import render
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tool_table import is_off
from langslice.tools.catalog import enabled, verbs_for

#: Said when the job's image model is not connected to LangSlice here.
_IMAGE_MODEL_OFF = (
    "[[?trace_borders The image-model tool (trace_borders) is off: no image "
    "model is connected to LangSlice (no key or login for this job's image "
    "provider).[[?fireants_syn  Fit each section's deformation to its stain "
    "with fireants_syn.]]]]"
)
#: ... as a job that disables nothing reads it.
IMAGE_MODEL_OFF = render(_IMAGE_MODEL_OFF)

#: What the opening pictures show, said by every door that names them.
OPENING_SHOWS = (
    "They show every section in strips, in the stack's current order, each "
    "labelled, with the atlas at its current position beneath it at the same "
    "scale when the stack has positions; when a section has none, atlas "
    "reference strips at evenly spaced positions follow."
)
OPENING_PAGES = (
    "Opening pictures: read every page with show_stack(page=1) through "
    "show_stack(page={pages}) before any write. "
)
OPENING_FILES = (
    "Opening pictures, saved as files, in reading order, each after the "
    "text that goes with it. "
)
OPEN_EVERY_PICTURE = "Open every one of these pictures before any write."
COMMANDS = "Commands:"
EVERY_TOOL_A_COMMAND = (
    "- Every tool is a command: `langslice-job {folder} TOOL --args "
    "'{{...}}'` (or `--name value`); `langslice-job {folder} schema TOOL "
    "[TOOL ...]` gives their arguments and descriptions, several at once. "
    "Pictures come back as files: `picture_files` lists them first, "
    "`artifacts` again with their kinds (`view`, in the order the reply's "
    "`pictures` lists their numbers); open them to see them."
)
DRY_RUN = (
    "- `--dry-run` runs a write without writing and reports what would "
    "change[[?elastix_affine|fireants_syn|trace_borders ; the "
    "[[#elastix_affine|fireants_syn=fits||trace_borders=trace_borders]] are "
    "only checked, not run]]{plugins}."
)
DRY_RUN_PLUGINS = "; a plugin's engine tools are only checked too"
LONG_TOOLS = (
    "- Long tools ({tools}): a call answers with the result or, still "
    "running after 30 s, with its run id and state `running`; the run goes "
    "on (also when your shell ends the command, where a process may outlive "
    "it) and `wait ID` answers within 30 s with its answer or its state "
    "(wait again). `--background` answers at once with the run id; `runs` "
    "lists the runs. Do not start a fit again while its run goes on."
)
PARALLEL_COMMANDS = (
    "- Commands may run in parallel: each write holds the job folder's "
    "lock, and a long tool re-checks each section before it applies "
    "(STALE_INPUT when the section changed meanwhile: run it again for that "
    "section)."
)
USER_NOTES = "User notes:"
STATUS_TABLE = (
    "Status table (filename, position, spacing to the next placed section, "
    "flags):\n{table}\n\nRecent run notes:\n{notes}"
)
NO_RUN_NOTES = "- (none)"

#: How many run notes the status text repeats (the newest).
RECENT_NOTES = 12

#: ``job.json``'s field holding the user's notes for the job
#: (``langslice-job FOLDER init --notes``).
NOTES_KEY = "notes"


def status_and_notes(state: Stack) -> str:
    """Return the status table with its header, then the recent run notes.

    Parameters
    ----------
    state
        The stack whose status and notes are shown.

    Returns
    -------
    str
        What the ADK agent's first message ends with, and every other
        door's statement.
    """
    recent = [f"- {note}" for note in state.notes[-RECENT_NOTES:]]
    return STATUS_TABLE.format(
        table=status_text(state), notes="\n".join(recent) or NO_RUN_NOTES
    )


def user_notes_lines(notes: str) -> list[str]:
    """Return the user's notes as statement lines.

    Parameters
    ----------
    notes
        The user's notes for the job.

    Returns
    -------
    list of str
        A blank line, the heading and the notes; none when they are empty.
    """
    if not notes.strip():
        return []
    return ["", USER_NOTES, notes]


def read_notes(layout: JobLayout) -> str:
    """Return the user's notes for the job in ``job.json``.

    Parameters
    ----------
    layout
        The job folder.

    Returns
    -------
    str
        Its ``notes``; empty when it has none or the file does not read.
    """
    try:
        record = read_job_file(layout) or {}
    except (OSError, ValueError):
        return ""
    return str(record.get(NOTES_KEY) or "")


def image_model_off_text(disabled: Collection[str]) -> str:
    """Return :data:`IMAGE_MODEL_OFF` for a job.

    Parameters
    ----------
    disabled
        What the job switches off.

    Returns
    -------
    str
        The line; empty when the job has no ``trace_borders`` to be off.
    """
    return render(_IMAGE_MODEL_OFF, disabled)


def image_model_state(
    spec: JobSpec, *, connected: bool
) -> dict[str, str | bool] | None:
    """Return the job's image model as the status and ``init`` report it.

    Parameters
    ----------
    spec
        The job's settings.
    connected
        Whether the job's image model is connected to LangSlice here.

    Returns
    -------
    dict or None
        The provider, whether it is connected here, whether
        ``trace_borders`` is offered; None without the nonlinear task.
    """
    if not spec.has("nonlinear"):
        return None
    uses = spec.nonlinear.uses_image_model
    return {
        "provider": spec.nonlinear.provider,
        "connected": bool(uses and connected),
        "trace_borders": bool(uses and connected),
    }


def opening_for_mcp(pages: int) -> list[str]:
    """Return the MCP door's opening paragraph: the ``show_stack`` pages.

    Parameters
    ----------
    pages
        How many ``show_stack`` pages the opening has.

    Returns
    -------
    list of str
        The paragraph, as statement lines.
    """
    return [OPENING_PAGES.format(pages=pages) + OPENING_SHOWS]


def opening_for_cli(
    folder: str,
    listing: Sequence[str],
    disabled: Collection[str],
    *,
    spec: JobSpec | None = None,
) -> list[str]:
    """Return the agent CLI's opening paragraph and how its commands behave.

    Parameters
    ----------
    folder
        The job folder the commands name.
    listing
        The opening in reading order: each strip's text line, then
        ``- <absolute path>`` of its picture file.
    disabled
        What the job switches off.
    spec
        The job's settings: the tools its run lacks are not named either,
        and its plugin tools are listed with the long ones.

    Returns
    -------
    list of str
        The paragraph and the commands, as statement lines.
    """
    if spec is not None:  # a verb the run does not have is not named either
        run = set(enabled(spec, scripting=True))
        disabled = [
            *disabled,
            *(name for name in verbs_for(spec) if name not in run),
        ]
    plugins = DRY_RUN_PLUGINS if spec is not None and spec.plugins else ""
    return [
        OPENING_FILES + OPENING_SHOWS,
        "",
        *listing,
        "",
        OPEN_EVERY_PICTURE,
        "",
        COMMANDS,
        EVERY_TOOL_A_COMMAND.format(folder=folder),
        render(DRY_RUN.format(plugins=plugins), disabled),
        LONG_TOOLS.format(tools=", ".join(long_verbs(disabled, spec=spec))),
        PARALLEL_COMMANDS,
    ]


def long_verbs(
    disabled: Collection[str], *, spec: JobSpec | None = None
) -> list[str]:
    """Return the verbs that compute outside the job lock and may take long.

    Parameters
    ----------
    disabled
        What the job switches off.
    spec
        The job's settings; given, the verbs its run does not have are left
        out.

    Returns
    -------
    list of str
        The long verbs (``tools.catalog.Verb.long``), in the catalog's order.
    """
    return [
        name
        for name, verb in verbs_for(spec).items()
        if verb.long
        and not is_off(disabled, name)
        and (spec is None or verb.tool.is_on(spec))
    ]
