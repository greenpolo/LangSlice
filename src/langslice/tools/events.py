"""The events of a tool call: ``tool_start`` and ``tool_end``.

Every call through the tool pipeline sends two events to the toolbox's
observers, inside its lock: ``tool_start`` before the tool runs and
``tool_end`` after it, crash or not. An observer is a host's live display
(the LangSlice window, ABBA) or the job folder's ``logs/events.jsonl``
(:class:`EventLog`), so a viewer in another process follows any run on any
door. An event holds what the call was asked and answered, made plain
(:func:`plain`): never picture bytes, provider signatures or encrypted
reasoning, and it never enters the model's context.

================  ==========================================================
field             meaning
================  ==========================================================
``name``          the tool
``execution_id``  one id per call, shared by its two events
``id``            the framework's call id (ADK's), when the tool takes one
``args``          the arguments, defaults filled
``target_ids``    the sections the call names, by filename, resolved before
                  the tool can reorder the stack (:func:`target_ids`)
``failed``        ``tool_end``: the reply's status is a failure, or a crash
``response``      ``tool_end``: the reply, made plain
``views``         ``tool_end``: the paths of the pictures the call saved
================  ==========================================================
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.stack.stack import Stack
from langslice.job.job import Job

logger = logging.getLogger(__name__)

#: A host's observer of events, a tool call's or a session's.
LiveCallback = Callable[[dict[str, Any]], None]

#: Tools whose host display targets are the whole stack.
_STACK_TOOLS = frozenset({"status", "undo", "redo", "submit"})
#: The most bytes of a call's arguments a ``logs/events.jsonl`` line keeps.
EVENT_ARGS_BYTES = 2048
#: The keys an event never carries: picture bytes and provider secrets.
_HIDDEN_KEYS = frozenset(
    {TOOL_MEDIA_PARTS_KEY, "thought_signature", "encrypted_content"}
)


def plain(value: object) -> object:
    """Copy JSON content without object reprs, media bytes or secrets.

    Parameters
    ----------
    value
        A reply, an argument list or one of their values.

    Returns
    -------
    object
        The same content as plain JSON values; anything else becomes the
        text ``<non-text content>``.
    """
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, dict):
        items = cast("dict[object, object]", value).items()  # any keys
        return {
            key: plain(item)
            for key, item in items
            if isinstance(key, str) and key not in _HIDDEN_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in cast("Sequence[object]", value)]
    return "<non-text content>"


def call_id(context: object) -> str | None:
    """Return the framework's id of a call, from ADK's ``tool_context``.

    Parameters
    ----------
    context
        The ``tool_context`` the call was given; None at a door without one.

    Returns
    -------
    str or None
        The call's ``function_call_id``, None without one.
    """
    identifier: object = getattr(context, "function_call_id", None)
    return identifier if isinstance(identifier, str) else None


def reply_status(reply: object) -> object:
    """Return a reply's status word, None when it has none.

    Parameters
    ----------
    reply
        A tool's reply, made plain.

    Returns
    -------
    object
        The reply's ``status``; None for a reply that is no dict or has no
        status.
    """
    if not isinstance(reply, dict):
        return None
    return cast("dict[object, object]", reply).get("status")  # any keys


def target_ids(
    state: Stack, name: str, args: Mapping[str, object]
) -> list[str]:
    """Resolve host display targets before a tool can reorder the stack.

    Parameters
    ----------
    state
        The stack the call works on.
    name
        The tool called.
    args
        The call's arguments, defaults filled.

    Returns
    -------
    list of str
        The filenames of the sections the call names, each once; every
        section for a call on the whole stack.
    """
    every = [record.id for record in state.in_order()]
    sections = args.get("sections")
    if name in _STACK_TOOLS:
        return every
    if name == "position_sections" and not sections:
        return every  # the cutting angles alone: every section
    if name in ("make_image", "make_mask") and not sections:
        return every
    if name == "look" and not sections and args.get("mode") != "atlas":
        return every
    refs: list[object] = []
    if isinstance(sections, (list, tuple)):
        for entry in cast("Sequence[object]", sections):  # names or rows
            ref = entry
            if isinstance(entry, dict):
                ref = cast("dict[str, object]", entry).get("id", "")
            refs.append(ref)
    for key in ("section", "id"):
        if args.get(key) not in (None, ""):
            refs.append(args[key])
    refs.extend(cast("Sequence[object]", args.get("slices") or []))
    targets: list[str] = []
    for ref in refs:
        record = state.resolve(ref)
        if record is not None and record.id not in targets:
            targets.append(record.id)
    return targets


def event_args(arguments: object) -> dict[str, object]:
    """Cap a call's arguments for one ``logs/events.jsonl`` line.

    Parameters
    ----------
    arguments
        The call's arguments, made plain.

    Returns
    -------
    dict
        ``args``: the arguments as given when their JSON is within
        :data:`EVENT_ARGS_BYTES`, else that JSON cut to it, as a string,
        with ``args_truncated``.
    """
    data = json.dumps(arguments, default=str, ensure_ascii=False)
    encoded = data.encode("utf-8")
    if len(encoded) <= EVENT_ARGS_BYTES:
        return {"args": arguments}
    cut = encoded[:EVENT_ARGS_BYTES].decode("utf-8", "ignore")
    return {"args": cut, "args_truncated": True}


@dataclass(frozen=True)
class EventLog:
    """The job folder's observer: one ``logs/events.jsonl`` line per event.

    A line holds ``name``, ``execution_id``, ``door``, ``target_ids`` and
    ``args`` (:func:`event_args`); ``tool_end`` adds ``step`` (the undo
    history's depth after the call, as a saved picture's ``step``),
    ``views`` (relative to the job folder), ``failed`` and ``status`` (the
    reply's status word, ``ok`` when it has none). Never the reply or
    picture bytes. A job that writes nothing (``persist`` False, a dry run)
    logs nothing.
    """

    #: The job whose folder holds the log.
    job: Job
    #: The door each line names (``agent``, ``mcp``, ``cli``, ``library``).
    door: str

    def __call__(self, event: dict[str, Any]) -> None:
        """Write an event as one line of the job folder's event log.

        Parameters
        ----------
        event
            A ``tool_start`` or ``tool_end`` event.
        """
        job = self.job
        if not job.persist:
            return
        kind = str(event.get("kind"))
        fields: dict[str, Any] = {
            "name": event.get("name"),
            "execution_id": event.get("execution_id"),
            "door": self.door,
            "target_ids": list(event.get("target_ids") or []),
            **event_args(event.get("args") or {}),
        }
        if kind == "tool_end":
            status = reply_status(event.get("response"))
            views = cast("list[str]", event.get("views") or [])  # paths
            fields.update(
                step=len(job.undo_stack),
                views=[job.layout.relative(path) for path in views],
                failed=bool(event.get("failed")),
                status="ok" if status is None else str(status),
            )
        job.layout.log_event(kind, **fields)
