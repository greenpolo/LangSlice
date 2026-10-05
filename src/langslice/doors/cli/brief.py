"""``langslice job FOLDER brief``: the job as LangSlice's own agent gets it.

A coding agent that registers a job through the agent CLI starts here. The
brief is the job statement every door gives
(:func:`langslice.doors.statement.job_statement`, worded for the CLI: where
its opening pictures are, how its commands answer), the user's notes, the
status table with the recent run notes, and the opening pictures
(:func:`langslice.core.opening.opening_items`: the same strips, captions and
order as the ADK agent's seed message), drawn at the job's viewer's limits
(:func:`langslice.doors.jobs.job_viewer`) and saved in the job folder as
the opening views (``views/<seq>_opening/view.jpg``, as the ADK run saves
its own). Everything is also written to ``BRIEF.md`` in the job folder,
which the job's reference card names first. ``init`` writes the statement
without pictures (:func:`build` with ``pictures=False``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

from langslice.doors.card import BRIEF_FILE


@dataclass
class Brief:
    """One brief: the statement, the opening in reading order (texts and
    saved pictures), the artifacts, the file it was written to."""

    statement: str
    #: ``{"text": ...}`` and ``{"picture": n, "path": ...}`` in reading order.
    opening: list[dict[str, Any]] = field(default_factory=list)
    #: Each opening picture (kind ``opening``, ``index`` its reading order,
    #: ``label`` the strip text before it), then the brief file.
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    path: Path | None = None
    facts: dict[str, Any] = field(default_factory=dict)


def resolution_range(opened: Any) -> dict[str, int]:
    """``view.resolution``'s range on this door: the smallest, the largest
    (the viewer's) and what 0 or nothing gives."""
    from langslice.core.sizes import AUTO_RESOLUTION, MIN_RESOLUTION, PICTURE_EDGES

    return {"min": MIN_RESOLUTION, "max": int(opened.max_view_edge),
            "default": PICTURE_EDGES[AUTO_RESOLUTION][1]}


def build(opened: Any, *, pictures: bool = True) -> Brief:
    """The brief of the open job *opened* (a :class:`langslice.doors.jobs.Opened`
    for the CLI: its viewer and door). With *pictures* the opening strips are
    drawn and saved as the job's opening views; ``BRIEF.md`` is written."""
    from langslice.core.opening import VIEWER_LIMITS, opening_items
    from langslice.doors.jobs import image_model_off
    from langslice.doors.statement import (
        image_model_state,
        job_statement,
        opening_for_cli,
        read_notes,
    )
    from langslice.job.views import PICTURE_FILE, captured

    job = opened.job
    box = opened.tools()
    folder = str(job.folder)
    viewer = opened.viewer or "claude"
    items: list[str | Image.Image] = []
    saved: list[Any] = []
    if pictures:
        items = opening_items(job.state, opened.ctx, limit=VIEWER_LIMITS[viewer][0])
        strips = [item for item in items if isinstance(item, Image.Image)]
        with captured() as saved:
            job.views.save(tool="opening", pictures=[(strip, None) for strip in strips])
        job.views.flush()
    connected = opened.image_model_connected
    spec = job.spec
    statement = job_statement(
        spec, job.state, opened.ctx, door="cli", tool_names=box.names,
        opening=opening_for_cli(folder, len(saved) if pictures else None),
        notes=read_notes(job.layout), max_resolution=box.max_view_edge,
        image_model_off=image_model_off(spec, connected), auto=True, gates=False,
    )
    brief = Brief(statement=statement, facts={
        "viewer": viewer, "resolution": resolution_range(opened),
        **({"image_model": state} if (state := image_model_state(
            spec, connected=connected)) is not None else {}),
    })
    label = ""
    number = 0
    for item in items:
        if isinstance(item, str):
            brief.opening.append({"text": item})
            label = item
            continue
        path = saved[number].folder / PICTURE_FILE if number < len(saved) else None
        brief.opening.append({"picture": number, "path": str(path) if path else None})
        if path is not None:
            brief.artifacts.append({"path": str(path), "kind": "opening", "index": number,
                                    "label": label})
        number += 1
    brief.path = write(job.layout.folder, brief, pictures=pictures)
    if brief.path is not None:
        brief.artifacts.append({"path": str(brief.path), "kind": "brief"})
    return brief


def text(folder: Path, brief: Brief, *, pictures: bool) -> str:
    """``BRIEF.md``'s text (see the module text)."""
    when = datetime.now().astimezone().isoformat(timespec="seconds")
    lines = [
        "# LangSlice job brief",
        "",
        f"Written by `langslice job {folder} brief` at {when}; run it again for the "
        "stack as it stands now (`status` gives the table alone). Read this first, then "
        "open every opening picture listed at the end before any write. `AGENTS.md` "
        "(the same as `CLAUDE.md`) is the reference card: files, coordinates, verbs, "
        "exit codes.",
        "",
        "## Job statement",
        "",
        brief.statement,
        "",
        "## Opening pictures",
        "",
    ]
    if not pictures:
        lines.append(f"Not saved yet: run `langslice job {folder} brief`.")
    for entry in brief.opening:
        if "text" in entry:
            lines += [entry["text"], ""]
        else:
            lines += [f"- `{entry['path']}`", ""]
    return "\n".join(lines).rstrip() + "\n"


def write(folder: Path, brief: Brief, *, pictures: bool) -> Path | None:
    """Write ``BRIEF.md`` into the job folder; its path (None when the folder
    cannot be written)."""
    import logging

    path = Path(folder) / BRIEF_FILE
    try:
        content = text(Path(folder), brief, pictures=pictures).replace("\r\n", "\n")
        with path.open("w", encoding="utf-8", newline="\n") as output:
            output.write(content)
    except OSError:
        logging.getLogger(__name__).warning("Could not write %s", path, exc_info=True)
        return None
    return path
