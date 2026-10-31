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

from dataclasses import dataclass

from langslice.core.jsonable import JsonObject
from langslice.job.settings.tool_table import TOOLS

#: The reads whose targets the viewer follows besides every write: the two
#: that look at sections.
FOLLOWED_READS = frozenset({"look", "zoom"})
#: Whether a call of each kind of tool (``job.settings.tool_table``) moves
#: the focus: a write does; a read only when it is one of FOLLOWED_READS.
_KIND_MOVES = {"read": False, "write": True}


@dataclass(frozen=True)
class Focus:
    """The sections the agent works on, and the call that named them.

    Before the first followed call: every section in stack order, and no
    call (the opening).
    """

    #: The target sections, in the call's order, without repeats.
    sections: tuple[str, ...]
    #: The tool, its call's ``execution_id`` and when it started (UTC).
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
    if tool is None:  # a plugin's tool, or one of another version: a write
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
