"""``langslice job FOLDER launch --role register|register-scripting``: the launch prompt.

A registering agent's job is the message that launched it. The main session
(the user's app) runs this command and passes ``result.prompt`` unchanged as
the registration subagent's whole prompt; the subagent's system prompt is
:data:`langslice.tools.descriptions.statement.SYSTEM_PROMPT`, the same
for every job.

* ``register`` (the MCP subagent): one line naming the job folder and
  ``start_job``, then the job statement worded for the MCP door, built by
  :func:`langslice.doors.mcp.server.mcp_statement` from the job opened as the
  MCP server opens it, so the server and this command cannot drift.
* ``register-scripting`` (the MCP subagent with a shell): the same message
  with a paragraph on scripts after its opening's
  (:func:`langslice.tools.descriptions.door_text.scripting_lines`): the
  interpreter of the job's own Python environment, made now when it is
  missing, the command that installs a package into it, the library and
  where the agent's files go. It registers with the MCP tools and with
  scripts of its own; ``langslice job`` runs no registration tool
  (:data:`langslice.doors.commands.job.catalog.CLI_VERBS`).

The exact text is saved in ``logs/launches/<six-digit number>_<role>.md``;
``langslice session-end`` compares the subagent's first message with those
files (:mod:`langslice.doors.commands.transcripts`). ``init`` writes
``TOOLS.md`` and reports the job's facts (:func:`facts`) without pictures.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langslice.doors.commands.job import environment
from langslice.job.folder.layout import JobLayout
from langslice.tools.descriptions.door_text import (
    SCRIPTING,
    image_model_state,
)

ROLES = ("register", "register-scripting")

_NUMBERED = re.compile(r"^(\d{6})_")


@dataclass
class Launch:
    """One launch prompt: its role and its text."""

    role: str
    prompt: str


def resolution_range(opened: Any) -> dict[str, int]:
    """``look``'s ``resolution`` range on this door: the smallest, the largest
    (the viewer's) and what 0 or nothing gives."""
    from langslice.core.pictures.sizes import (
        AUTO_RESOLUTION,
        MIN_RESOLUTION,
        PICTURE_EDGES,
    )

    return {
        "min": MIN_RESOLUTION,
        "max": int(opened.max_view_edge),
        "default": PICTURE_EDGES[AUTO_RESOLUTION][1],
    }


def facts(opened: Any) -> dict[str, Any]:
    """The CLI job's viewer, ``look`` resolution range and image model, as
    ``init`` reports them."""
    state = image_model_state(opened.spec, connected=opened.image_model_connected)
    return {
        "viewer": opened.viewer or "claude",
        "resolution": resolution_range(opened),
        **({"image_model": state} if state is not None else {}),
    }


def role_line(folder: str) -> str:
    """The launch prompt's first line."""
    return (
        f"Register the brain in job folder {folder}. Call "
        f'start_job(image_folder="{folder}") first: it opens the job and gives you '
        "its tools."
    )


def scripting_prompt(
    job_folder: Path,
    *,
    atlas_loader: Callable[[str], object] | None = None,
) -> str:
    """Return the ``register-scripting`` prompt for a job.

    It is the ``register`` prompt with the paragraph on scripts after its
    opening's. The job's own Python environment is made first when it is
    missing, so the interpreter the prompt names is there to run.

    Parameters
    ----------
    job_folder
        Where the job is: its ``langslice`` folder.
    atlas_loader
        What loads the job's atlas; None for LangSlice's own.

    Returns
    -------
    str
        The prompt.

    Raises
    ------
    ImportError
        When the MCP package is missing.
    OSError, subprocess.SubprocessError
        When the job's environment cannot be made.
    """
    layout = JobLayout(job_folder)
    environment.ensure(layout)
    python = environment.python_of(environment.env_dir(layout))
    paragraph = SCRIPTING.format(
        folder=layout.folder,
        python=python,
        install=environment.install_command(layout),
    )
    return mcp_prompt(job_folder, atlas_loader, scripting=paragraph)


def mcp_prompt(job_folder: Path, atlas_loader: Any = None, *, scripting: str = "") -> str:
    """The ``register`` prompt for the job in *job_folder*; the MCP package is
    needed (``ImportError`` when it is missing). *scripting*: the paragraph
    on scripts (:func:`scripting_prompt`)."""
    from langslice.doors.mcp.server import mcp_statement, open_folder

    def no_job(path: str) -> Any:
        raise ValueError(f"No LangSlice job in {path}")

    session = open_folder(str(job_folder), no_job, atlas_loader)
    try:
        image_folder = str(session.ctx.image_folder)
        folder = image_folder
        try:
            from langslice.doors.jobs import find

            if Path(find(image_folder)) != Path(session.job.folder):
                folder = str(session.job.folder)
        except (LookupError, OSError, ValueError):
            folder = str(session.job.folder)
        statement = mcp_statement(session, scripting=scripting)
        return role_line(folder) + "\n\n" + statement
    finally:
        session.close()


def save(layout: Any, role: str, prompt: str) -> Path:
    """Save *prompt* exactly in ``logs/launches/<next six-digit number>_<role>.md``."""
    folder: Path = layout.launches_dir
    folder.mkdir(parents=True, exist_ok=True)
    taken = [
        int(match.group(1)) for path in folder.iterdir() if (match := _NUMBERED.match(path.name))
    ]
    number = max(taken, default=0) + 1
    while True:
        path = folder / f"{number:06d}_{role}.md"
        try:
            with path.open("x", encoding="utf-8", newline="\n") as output:
                output.write(prompt)
        except FileExistsError:
            number += 1
            continue
        return path


def write_tools(folder: Path, box: Any) -> Path | None:
    """Write ``TOOLS.md``
    (:func:`langslice.doors.commands.job.catalog.tools_text`) for
    the toolbox *box* of the job in *folder*: its verbs as its run declares
    them. Its path, or None when it could not be written."""
    import logging

    from langslice.doors.commands.job.catalog import tools_text
    from langslice.doors.reference_card import TOOLS_FILE

    path = Path(folder) / TOOLS_FILE
    try:
        content = tools_text(str(folder), list(box.names), box.variant, box.job.spec)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except Exception:
        logging.getLogger(__name__).warning("Could not write %s", path, exc_info=True)
        return None
    return path
