"""The agent's focus: the sections the viewer shows it working on.

The agent viewer follows the agent as ABBA's agent viewer did. Each call of
a followed tool (every write, and the two reads that look at sections,
``look`` and ``zoom``) puts its target sections in focus, read from the
``tool_start`` event every door appends to the job's ``logs/events.jsonl``
(``target_ids``: the filenames the toolbox resolved). A call with no
targets keeps the focus it found. The watcher keeps the latest focus and
puts it in every snapshot; before the first followed call the focus is
every section, in stack order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath

from langslice.core.jsonable import JsonObject
from langslice.job.settings.tool_table import TOOLS

FOLLOWED_READS = frozenset({"look", "zoom"})
_KIND_MOVES = {"read": False, "write": True}


@dataclass(frozen=True)
class Focus:
    """The sections the agent works on, and the call that named them.

    Before the first followed call: every section in stack order, and no
    call (the opening).
    """

    sections: tuple[str, ...]
    tool: str | None = None
    call: str | None = None
    at: str | None = None


def is_followed(name: str) -> bool:
    """Tell whether a tool's calls move the viewer's focus.

    Parameters
    ----------
    name
        The tool's name as its events give it.

    Returns
    -------
    bool
        True for a write (a tool of kind ``write``, a plugin's tool or one
        this version does not list, as the undo labels count them) and for
        :data:`FOLLOWED_READS`.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return True
    return name in FOLLOWED_READS or _KIND_MOVES[tool.kind]


def started_focus(record: JsonObject) -> Focus | None:
    """Return the focus a ``tool_start`` event sets.

    Parameters
    ----------
    record
        The event as ``logs/events.jsonl`` holds it.

    Returns
    -------
    Focus or None
        The call's target sections, in order and without repeats, with its
        tool, ``execution_id`` and start, when its tool is followed
        (:func:`is_followed`) and it names any; None otherwise (the focus
        stays as it was).
    """
    name = record.get("name")
    targets = record.get("target_ids")
    if not isinstance(name, str) or not is_followed(name):
        return None
    if not isinstance(targets, list) or not targets:
        return None
    sections = tuple(dict.fromkeys(str(target) for target in targets))
    call = record.get("execution_id")
    at = record.get("at")
    call_id = None if call is None else str(call)
    started = at if isinstance(at, str) else None
    return Focus(sections, name, call_id, started)


def shown_focus(record: JsonObject, section_ids: Sequence[str]) -> Focus | None:
    """Return the focus a followed read sets when it ends having named no section.

    A ``zoom`` names an earlier picture, not sections; the sections its new
    pictures show (``sections/<stem>/views/...`` in the ``tool_end`` event's
    ``views``) are the ones the agent is looking at.

    Parameters
    ----------
    record
        The ``tool_end`` event as ``logs/events.jsonl`` holds it.
    section_ids
        The job's sections, in stack order.

    Returns
    -------
    Focus or None
        Those sections, when the tool is followed, named no target and its
        pictures show any; None otherwise.
    """
    name = record.get("name")
    if not isinstance(name, str) or not is_followed(name) or record.get("target_ids"):
        return None
    views = record.get("views")
    if not isinstance(views, list):
        return None
    by_stem = {PurePosixPath(sid).stem: sid for sid in section_ids}
    shown: list[str] = []
    for view in views:
        parts = PurePosixPath(str(view).replace("\\", "/")).parts
        if len(parts) > 2 and parts[0] == "sections" and parts[1] in by_stem:
            shown.append(by_stem[parts[1]])
    if not shown:
        return None
    call = record.get("execution_id")
    at = record.get("at")
    return Focus(
        tuple(dict.fromkeys(shown)),
        name,
        None if call is None else str(call),
        at if isinstance(at, str) else None,
    )
