"""A coding agent's own words, read from Claude Code's transcript of the run.

A registration run by a coding agent leaves only its tool calls in the job
folder: the agent's messages and reasoning summaries stay in its app. Claude
Code writes every session as JSON lines under
``~/.claude/projects/<project>/`` (``<session>.jsonl``; a subagent's under
``<session>/subagents/``), appended while the session runs.
:class:`TranscriptWords` finds the transcript whose ``start_job`` call names
a job's folder and tails it, giving the viewer the records LangSlice's own
agent trace would hold: one ``session`` record, then a ``model`` record per
message (``text``) and per reasoning summary (``thought``), from that call
on.

Read-only, and only on this computer. ``LANGSLICE_CLAUDE_TRANSCRIPTS`` names
another folder of projects, or ``off``.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import cast

from langslice.core.jsonable import JsonObject, empty_object
from langslice.job.layout import (
    EVENTS_FILE,
    JOB_DIRNAME,
    LOGS_DIR,
    held_image_folder,
)

ENV = "LANGSLICE_CLAUDE_TRANSCRIPTS"
OFF = "off"
SEARCH_S = 3.0
EARLY_S = 120.0
LATE_S = 6 * 3600.0
MAX_CANDIDATES = 40
PATTERNS = ("*/*.jsonl", "*/*/subagents/*.jsonl")
ASSISTANT_LINE = b'"type":"assistant"'
START_JOB = "start_job"
AGENT = "Claude Code"
VERSION_DIGITS = 2
CLAUDE = "claude"


class Line(StrEnum):
    """The transcript's words this module reads."""

    TURN = "assistant"
    SUMMARY = "thinking"
    MESSAGE = "text"
    CALL = "tool_use"


@dataclass(frozen=True)
class Turn:
    """One assistant turn of a transcript.

    Parameters
    ----------
    at
        When it was written, as the trace writes times (:func:`local_time`).
    model
        The model's id.
    summaries
        The reasoning summaries that are not empty.
    messages
        The messages the agent wrote, in order.
    opened
        The folders its ``start_job`` calls name (:func:`folder_key`).
    """

    at: str
    model: str
    summaries: tuple[str, ...]
    messages: tuple[str, ...]
    opened: frozenset[str]


def projects_root() -> Path | None:
    """Find the folder of Claude Code's projects.

    Returns
    -------
    Path or None
        ``~/.claude/projects``, the folder ``LANGSLICE_CLAUDE_TRANSCRIPTS``
        names, or None when that is ``off`` or the folder is not there.
    """
    named = os.environ.get(ENV, "").strip()
    if named.lower() == OFF:
        return None
    root = Path(named) if named else Path.home() / ".claude" / "projects"
    return root if root.is_dir() else None


def folder_key(path: str) -> str:
    """Write a folder path the way two spellings of it compare equal.

    Parameters
    ----------
    path
        A folder path, with either slash and in any case.

    Returns
    -------
    str
        The path in lower case with forward slashes and no trailing slash.
    """
    return path.replace("\\", "/").rstrip("/").lower()


def job_folders(folder: Path, job_file: object) -> frozenset[str]:
    """Name the folders a ``start_job`` call for a job may give.

    Parameters
    ----------
    folder
        The job folder.
    job_file
        What its ``job.json`` holds, or None before it is read.

    Returns
    -------
    frozenset of str
        The job folder, the images folder it sits in and the one its
        ``job.json`` names, each as :func:`folder_key` writes it.
    """
    named = {folder}
    if folder.name == JOB_DIRNAME:
        named.add(folder.parent)
    for record in _objects([job_file]):
        images = held_image_folder(folder, dict(record))
        if images is not None:
            named.add(images)
    return frozenset(folder_key(str(one)) for one in named)


def model_name(model_id: str) -> str:
    """Write a Claude model id as the model's name.

    Parameters
    ----------
    model_id
        The id a transcript holds, such as ``claude-opus-4-1-20250805``.

    Returns
    -------
    str
        ``Claude Opus 4.1``; an id that is not Claude's as it is.
    """
    parts = model_id.split("-")
    if parts[0] != CLAUDE:
        return model_id
    words = [part.capitalize() for part in parts if not part.isdigit()]
    digits = [
        part
        for part in parts
        if part.isdigit() and len(part) <= VERSION_DIGITS
    ]
    return " ".join([*words, ".".join(digits)]).strip()


def local_time(stamp: str) -> str:
    """Write a transcript's UTC time as the trace writes times: local.

    Parameters
    ----------
    stamp
        An ISO time with its offset (``2026-01-02T03:04:05.678Z``).

    Returns
    -------
    str
        The same moment as a local ISO time without an offset, to the
        second; *stamp* as it is when it cannot be read.
    """
    try:
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except ValueError:
        return stamp
    local = moment.astimezone().replace(tzinfo=None)
    return local.isoformat(timespec="seconds")


def _objects(value: object) -> list[JsonObject]:
    """List the objects among a JSON array's items; none for another value."""
    items = cast("list[object]", value) if isinstance(value, list) else []
    return [cast("JsonObject", one) for one in items if isinstance(one, dict)]


def _said(content: object, kind: Line) -> tuple[str, ...]:
    """Collect the texts of one kind of block that are not empty."""
    blocks = (b for b in _objects(content) if b.get("type") == kind)
    texts = (str(block.get(kind) or "").strip() for block in blocks)
    return tuple(text for text in texts if text)


def _opened(content: object) -> frozenset[str]:
    """Collect the folders the turn's ``start_job`` calls name."""
    opened: set[str] = set()
    for block in _objects(content):
        name = str(block.get("name") or "")
        if block.get("type") != Line.CALL or not name.endswith(START_JOB):
            continue
        given = _objects([block.get("input")])
        values = list(given[0].values()) if given else []
        opened.update(folder_key(v) for v in values if isinstance(v, str))
    return frozenset(opened)


def _record(line: bytes) -> JsonObject | None:
    """Parse one line as a JSON object; None for anything else."""
    try:
        parsed = cast("object", json.loads(line))
    except ValueError:
        return None
    return next(iter(_objects([parsed])), None)


def _turn(line: bytes) -> Turn | None:
    """Parse one line; None unless it is an assistant turn."""
    record = _record(line) if ASSISTANT_LINE in line else None
    if record is None or record.get("type") != Line.TURN:
        return None
    message = next(iter(_objects([record.get("message")])), empty_object())
    content = message.get("content")
    return Turn(
        at=local_time(str(record.get("timestamp") or "")),
        model=str(message.get("model") or ""),
        summaries=_said(content, Line.SUMMARY),
        messages=_said(content, Line.MESSAGE),
        opened=_opened(content),
    )


@dataclass
class _Reader:
    """One transcript, read by byte offset; only complete lines are taken."""

    path: Path
    offset: int = 0
    partial: bytes = b""

    def read(self) -> list[Turn]:
        """Read the assistant turns appended since the last read."""
        try:
            size = self.path.stat().st_size
            if size <= self.offset:
                return []
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read(size - self.offset)
        except OSError:
            return []
        self.offset += len(data)
        whole, _, self.partial = (self.partial + data).rpartition(b"\n")
        turns = (_turn(line) for line in whole.split(b"\n"))
        return [turn for turn in turns if turn is not None]


def _candidates(root: Path, since: float, until: float) -> list[Path]:
    """List the transcripts that may be a job's, the likeliest first."""
    near: list[tuple[float, Path]] = []
    paths: list[Path] = []
    for pattern in PATTERNS:
        paths += root.glob(pattern)
    for path in paths:
        try:
            changed = path.stat().st_mtime
        except OSError:
            continue
        if since - EARLY_S <= changed <= until + LATE_S:
            near.append((abs(changed - until), path))
    near.sort()
    return [path for _, path in near[:MAX_CANDIDATES]]


def first_call_at(job_folder: Path) -> float | None:
    """Read when a job's first tool call was made.

    Parameters
    ----------
    job_folder
        The job folder, whose ``logs/events.jsonl`` holds the calls.

    Returns
    -------
    float or None
        The first event's time in seconds since the epoch; None while the
        file holds no event with a time.
    """
    try:
        with (job_folder / LOGS_DIR / EVENTS_FILE).open("rb") as handle:
            first = _record(handle.readline()) or {}
        stamp = str(first.get("at") or "")
        moment = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    except (OSError, ValueError):
        return None
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.timestamp()


class TranscriptWords:
    """The words of the Claude Code session that runs one job.

    Parameters
    ----------
    root
        The folder of Claude Code's projects; None reads nothing.
    clock
        The seconds that pace the search.
    """

    def __init__(
        self, root: Path | None, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        self.root = root
        self.clock = clock
        self._job: _Reader | None = None
        self._started = False
        self._since: float | None = None
        self._searched_at = float("-inf")
        self._readers: dict[Path, _Reader] = {}

    def read(
        self, job_folder: Path, job_file: object, until: float
    ) -> list[JsonObject]:
        """Read the records written since the last call.

        Parameters
        ----------
        job_folder
            The folder the job's files are in (``<images>/langslice``).
        job_file
            What its ``job.json`` holds, or None before it is read.
        until
            When the job's files last changed (seconds since the epoch).

        Returns
        -------
        list of JsonObject
            Trace records (``session``, ``model``) in the order written;
            empty until the job has made a tool call and its transcript is
            known.
        """
        if self.root is None:
            return []
        if self._job is not None:
            return self._records(self._job.read())
        now = self.clock()
        if now - self._searched_at < SEARCH_S:
            return []
        self._since = self._since or first_call_at(job_folder)
        if self._since is None:
            return []
        self._searched_at = now
        folders = job_folders(job_folder, job_file)
        for path in _candidates(self.root, self._since, until):
            reader = self._readers.setdefault(path, _Reader(path))
            turns = reader.read()
            opening = [
                n for n, turn in enumerate(turns) if turn.opened & folders
            ]
            if opening:
                self._job, self._readers = reader, {}
                return self._records(turns[opening[0] :])
        return []

    def _records(self, turns: list[Turn]) -> list[JsonObject]:
        records: list[JsonObject] = []
        for turn in turns:
            if not self._started:
                self._started = True
                name = model_name(turn.model) or None
                session = {"kind": "session", "agent": AGENT, "model": name}
                records.append({**session, "at": turn.at})
            records += [
                {"kind": "model", "thought": text, "at": turn.at}
                for text in turn.summaries
            ]
            records += [
                {"kind": "model", "text": text, "at": turn.at}
                for text in turn.messages
            ]
        return records
