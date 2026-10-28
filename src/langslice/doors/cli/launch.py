"""``langslice-job FOLDER launch --role register|register-scripting``: the launch prompt.

A registering agent's job is the message that launched it. The main session
(the user's app) runs this command and passes ``result.prompt`` unchanged as
the registration subagent's whole prompt; the subagent's system prompt is
:data:`langslice.doors.statement.SYSTEM_PROMPT`, the same for every job.

* ``register`` (the MCP subagent): one line naming the job folder and
  ``start_job``, then the job statement worded for the MCP door, built by
  :func:`langslice.doors.mcp.server.mcp_statement` from the job opened as the
  MCP server opens it, so the server and this command cannot drift.
* ``register-scripting`` (the CLI subagent): one line naming the job folder,
  then the job statement worded for the CLI door, whose opening paragraph lists
  the opening pictures, drawn and saved now as the job's opening views
  (:func:`langslice.core.opening.opening_items`: the same strips, captions and
  order as the ADK agent's seed message, at the job's viewer's limits,
  :func:`langslice.doors.jobs.job_viewer`), in reading order. ``TOOLS.md`` (every
  verb's description and arguments, :func:`langslice.doors.cli.catalog.tools_text`)
  is written beside the card.

The exact text is saved in ``logs/launches/<six-digit number>_<role>.md``;
``langslice session-end`` compares the subagent's first message with those
files (:mod:`langslice.doors.cli.transcripts`). ``init`` writes ``TOOLS.md`` and
reports the job's facts (:func:`facts`) without pictures.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

#: The roles, as the connectors name their agents.
ROLES = ("register", "register-scripting")

_NUMBERED = re.compile(r"^(\d{6})_")


@dataclass
class Launch:
    """One launch prompt: its text, the artifacts (the saved launch file, the
    opening pictures in reading order, ``TOOLS.md``) and what went unsaved."""

    role: str
    prompt: str
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    facts: dict[str, Any] = field(default_factory=dict)


def resolution_range(opened: Any) -> dict[str, int]:
    """``look``'s ``resolution`` range on this door: the smallest, the largest
    (the viewer's) and what 0 or nothing gives."""
    from langslice.core.sizes import AUTO_RESOLUTION, MIN_RESOLUTION, PICTURE_EDGES

    return {"min": MIN_RESOLUTION, "max": int(opened.max_view_edge),
            "default": PICTURE_EDGES[AUTO_RESOLUTION][1]}


def facts(opened: Any) -> dict[str, Any]:
    """The CLI job's viewer, ``look`` resolution range and image model, as
    ``init`` reports them."""
    from langslice.doors.statement import image_model_state

    state = image_model_state(opened.spec, connected=opened.image_model_connected)
    return {"viewer": opened.viewer or "claude", "resolution": resolution_range(opened),
            **({"image_model": state} if state is not None else {})}


def role_line(role: str, folder: str) -> str:
    """The launch prompt's first line."""
    if role == "register":
        return (f"Register the brain in job folder {folder}. Call "
                f'start_job(image_folder="{folder}") first: it opens the job and gives you '
                "its tools.")
    return f"Register the brain in job folder {folder} with the langslice-job command line."


def mcp_prompt(job_folder: Path, atlas_loader: Any = None) -> str:
    """The ``register`` prompt for the job in *job_folder*; the MCP package is
    needed (``ImportError`` when it is missing)."""
    from langslice.doors.mcp.server import mcp_statement, open_folder

    def no_job(path: str) -> Any:
        raise ValueError(f"No LangSlice job in {path}")

    session = open_folder(str(job_folder), no_job, atlas_loader)
    try:
        image_folder = str(session.ctx.image_folder)
        folder = image_folder
        try:  # the name start_job accepts: the images' folder when it leads to this job
            from langslice.doors.jobs import find

            if Path(find(image_folder)) != Path(session.job.folder):
                folder = str(session.job.folder)
        except (LookupError, OSError, ValueError):
            folder = str(session.job.folder)
        return role_line("register", folder) + "\n\n" + mcp_statement(session)
    finally:
        session.close()


def cli_prompt(opened: Any) -> Launch:
    """The ``register-scripting`` prompt of the open job *opened* (a
    :class:`langslice.doors.jobs.Opened` for the CLI: its viewer and door)."""
    from langslice.core.opening import VIEWER_LIMITS, opening_items
    from langslice.doors.jobs import image_model_off
    from langslice.doors.statement import (
        job_statement,
        opening_for_cli,
        read_notes,
    )
    from langslice.job.views import PICTURE_FILE, captured

    job = opened.job
    box = opened.tools()
    folder = str(job.folder)
    viewer = opened.viewer or "claude"
    items = opening_items(job.state, opened.ctx, limit=VIEWER_LIMITS[viewer][0])
    strips = [item for item in items if isinstance(item, Image.Image)]
    with captured() as saved:
        job.views.save(tool="opening", pictures=[(strip, None) for strip in strips])
    job.views.flush()
    launch = Launch(role="register-scripting", prompt="", facts=facts(opened))
    listing: list[str] = []
    label = ""
    number = 0
    for item in items:
        if isinstance(item, str):
            if listing:
                listing.append("")
            listing.append(item)
            label = item
            continue
        path = saved[number].folder / PICTURE_FILE if number < len(saved) else None
        if path is None:
            launch.warnings.append("An opening picture was not saved; "
                                   "`langslice.open_job` gives it to a script.")
        else:
            listing.append(f"- {path}")
            launch.artifacts.append({"path": str(path), "kind": "opening", "index": number,
                                     "label": label})
        number += 1
    spec = job.spec
    statement = job_statement(
        spec, job.state, opened.ctx, door="cli", tool_names=opened.listed_verbs(),
        opening=opening_for_cli(folder, listing, spec.switched_off, spec),
        notes=read_notes(job.layout), max_resolution=box.max_view_edge,
        image_model_off=image_model_off(spec, opened.image_model_connected), auto=True,
    )
    launch.prompt = role_line("register-scripting", folder) + "\n\n" + statement
    tools = write_tools(job.layout.folder, box)
    if tools is not None:
        launch.artifacts.append({"path": str(tools), "kind": "tools"})
    return launch


def save(layout: Any, role: str, prompt: str) -> Path:
    """Save *prompt* exactly in ``logs/launches/<next six-digit number>_<role>.md``."""
    folder: Path = layout.launches_dir
    folder.mkdir(parents=True, exist_ok=True)
    taken = [int(match.group(1)) for path in folder.iterdir()
             if (match := _NUMBERED.match(path.name))]
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
    """Write ``TOOLS.md`` (:func:`langslice.doors.cli.catalog.tools_text`) for
    the toolbox *box* of the job in *folder*: its verbs as its run declares
    them. Its path, or None when it could not be written."""
    import logging

    from langslice.doors.card import TOOLS_FILE
    from langslice.doors.cli.catalog import tools_text

    path = Path(folder) / TOOLS_FILE
    try:
        content = tools_text(str(folder), list(box.names), box.variant, box.job.spec)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except Exception:  # noqa: BLE001 - the launch stands without its tools file
        logging.getLogger(__name__).warning("Could not write %s", path, exc_info=True)
        return None
    return path
