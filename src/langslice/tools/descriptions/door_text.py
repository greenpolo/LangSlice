"""What each door adds after the job statement: its opening, the notes.

MCP gives the statement as one text: the statement
(:mod:`~langslice.tools.descriptions.statement`), then the door's own
paragraph on its opening pictures (:func:`opening_for_mcp`), the
image-model line when the job's image model is
not connected here, the user's notes (``job.json``'s ``notes``) and the
status table with the recent run notes (:func:`status_and_notes`). The ADK
agent's first message has the statement and the user's notes, then the
opening pictures, then the status table, as parts of their own.
"""

from __future__ import annotations

from collections.abc import Collection

from langslice.core.stack.stack import Stack
from langslice.core.stack.status import status_text
from langslice.job.layout import JobLayout, read_job_file
from langslice.job.review import read_comments, revised
from langslice.job.settings.needs import render
from langslice.job.settings.spec import JobSpec
from langslice.job.settings.tool_table import is_off

_IMAGE_MODEL_OFF = (
    "[[?trace_borders The image-model tool (trace_borders) is off: no image "
    "model is connected to LangSlice (no key or login for this job's image "
    "provider).[[?fireants_syn  Fit each section's deformation to its stain "
    "with fireants_syn.]]]]"
)
IMAGE_MODEL_OFF = render(_IMAGE_MODEL_OFF)

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
SCRIPTS = "Scripts:"
SCRIPT_PYTHON = (
    "- You also have a shell and file tools, for what no tool does: write a "
    'script and run it as `"{python}" SCRIPT [ARGS ...]`. That is the job\'s '
    "own Python: it imports LangSlice and every package LangSlice is "
    "installed with; `{install} PACKAGE [PACKAGE ...]` adds another."
)
SCRIPT_LIBRARY = (
    '- In a script, `langslice.open_job(r"{folder}")` opens this job: every '
    "tool is a method of it with the same arguments, without the tools' "
    "limit on pictures or sections per call, and `job.context` holds the "
    "parts for a method of your own (`help(job.context)`: the section "
    "images and the atlas as arrays, a transform or a deformation of your "
    "own, a preview that writes nothing, a section image you computed)."
)
SCRIPT_SHARED = (
    "- The tools and your scripts work on the one job: each sees what the "
    "other wrote, and every write of either is one undo step."
)
SCRIPT_PICTURES = (
    "- A script's pictures are files (a method's reply lists them under "
    "`artifacts`): open them with your file tool to see them."
)
SCRIPT_FILES = (
    "- Keep your scripts in `{folder}/scripts/` and your other files in "
    "`{folder}/scratch/`; `{folder}/AGENTS.md` describes the job folder's "
    "files."
)
SCRIPTING = "\n".join(
    (
        SCRIPTS,
        SCRIPT_PYTHON,
        SCRIPT_LIBRARY,
        SCRIPT_SHARED,
        SCRIPT_PICTURES,
        SCRIPT_FILES,
    )
)
USER_NOTES = "User notes:"
REVIEW_OPEN = (
    "A person reviewed this registration in the LangSlice window and left "
    "{comments} on {sections}. review_comments shows each with the box it "
    "marks; work on them first."
)
STATUS_TABLE = (
    "Status table (filename, position, spacing to the next placed section, "
    "flags):\n{table}\n\nRecent run notes:\n{notes}"
)
NO_RUN_NOTES = "- (none)"

RECENT_NOTES = 12

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


def review_lines(
    layout: JobLayout, state: Stack, switched_off: Collection[str]
) -> list[str]:
    """Return the statement's line on the review comments still open.

    Parameters
    ----------
    layout
        The job folder (its ``review.json``).
    state
        The stack as it stands.
    switched_off
        The names the job switches off.

    Returns
    -------
    list of str
        A blank line and :data:`REVIEW_OPEN`; none when no comment is open
        or the job switches ``review_comments`` off.
    """
    if is_off(switched_off, "review_comments"):
        return []
    still_open = [c for c in read_comments(layout) if not revised(c, state)]
    if not still_open:
        return []
    sections = list(dict.fromkeys(c.section for c in still_open))
    if len(still_open) == 1:
        count = "a comment"
    else:
        count = f"{len(still_open)} comments"
    if len(sections) <= 3:
        named = ", ".join(sections)
    else:
        named = f"{len(sections)} sections"
    return ["", REVIEW_OPEN.format(comments=count, sections=named)]
