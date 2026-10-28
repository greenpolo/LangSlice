"""Watching job folders by polling their files: what the agent viewer shows.

The GUI is a separate process from every run (the agent run, an MCP server, the
agent CLI's calls, a SliceBench subprocess), so it learns everything from the
job folder. :class:`JobWatcher` tails the append-only files by byte offset
(``logs/events.jsonl``, ``views.jsonl`` and the newest ``trace/*.jsonl``),
keeping an unterminated last line for the next round, and rereads the files
that are replaced whole (``state.json``, ``history/index.json``, ``job.json``,
``exports/spend.json``) when their size or modification time changes.

Everything a client needs comes back from :meth:`JobWatcher.snapshot` as plain
JSON: the activity items since a cursor, the sections table, undo and redo,
notes, spend and a status word. :class:`JobList` is the set of jobs a GUI
serves: the folders it was given and the jobs ``~/.langslice/recent_jobs.jsonl``
names.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from langslice.core.toolset import TOOLS
from langslice.job.history import UNDO_DEPTH
from langslice.job.layout import (
    EVENTS_FILE,
    EXPORTS_DIR,
    HISTORY_DIR,
    JOB_DIRNAME,
    JOB_FILE,
    LOGS_DIR,
    SPEND_FILE,
    STATE_FILE,
    TRACE_DIR,
    VIEWS_INDEX,
    JobLayout,
)

#: How often a watched job's files are looked at, at most.
POLL_S = 0.25
#: A job whose files changed this recently is Running.
RECENT_S = 30.0
#: The tools whose call is no undo step (``core.toolset`` kind ``read``); any
#: other (a write, a plugin tool, a tool of an older version) may be one.
_READS = frozenset(name for name, tool in TOOLS.items() if tool.kind == "read")
#: How many recent jobs the list holds.
MAX_RECENT = 40
#: The saved picture inside a view folder.
PICTURE = "view.jpg"
#: The longest text of one activity item the page receives.
MAX_TEXT = 60_000
#: The longest tool arguments the page receives, as JSON.
MAX_ARGS = 2048
#: The trace record kinds the viewer reads (LangSlice's own agent trace).
_TRACE_KINDS = {"session", "model", "summary", "error", "spend"}


class _Tail:
    """New complete lines of an append-only file, by byte offset."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self.partial = b""

    def read(self) -> tuple[list[dict[str, Any]], bool]:
        """``(records, truncated)``: the complete lines appended since the
        last read, parsed (a line that is not a JSON object is skipped), and
        whether the file became shorter than what was read (it was replaced
        or truncated; the caller starts over)."""
        try:
            size = self.path.stat().st_size
        except OSError:
            return [], False
        if size < self.offset:
            return [], True
        if size == self.offset:
            return [], False
        try:
            with self.path.open("rb") as handle:
                handle.seek(self.offset)
                data = handle.read(size - self.offset)
        except OSError:
            return [], False
        self.offset += len(data)
        data = self.partial + data
        *lines, self.partial = data.split(b"\n")
        records = []
        for line in lines:
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict):
                records.append(record)
        return records, False


class _Whole:
    """A file replaced whole (temp file and rename): reread when it changes."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.stamp: tuple[int, int] | None = None
        self.value: Any = None

    def read(self) -> bool:
        """Reread when the size or time changed; whether the value changed.
        A file caught mid-rename is tried once more."""
        for attempt in (0, 1):
            try:
                info = self.path.stat()
                stamp = (info.st_size, info.st_mtime_ns)
                if stamp == self.stamp:
                    return False
                value = json.loads(self.path.read_text(encoding="utf-8"))
            except FileNotFoundError:
                if attempt == 0:
                    continue
                if self.stamp is None:
                    return False
                self.stamp, self.value = None, None
                return True
            except (OSError, ValueError):
                if attempt == 0:
                    continue
                return False
            self.stamp, self.value = stamp, value
            return True
        return False


def _when(record: dict[str, Any]) -> float | None:
    """A record's ``at`` in seconds (an event's is UTC, the trace's local time)."""
    try:
        moment = datetime.fromisoformat(str(record.get("at")))
    except ValueError:
        return None
    return (moment if moment.tzinfo else moment.astimezone()).timestamp()


def _in_time_order(trace: list[dict[str, Any]], events: list[dict[str, Any]]
                   ) -> list[tuple[float, int, str, dict[str, Any]]]:
    """Trace records and events in one order: by time, and within a second a
    model turn before the calls it made and a session's end after them."""
    out: list[tuple[float, int, str, dict[str, Any]]] = []
    for source, records in (("trace", trace), ("event", events)):
        last = float("-inf")
        for record in records:
            when = _when(record)
            last = last if when is None else when
            if source == "event":
                rank = 1
            else:
                rank = 2 if record.get("kind") in ("summary", "error") else 0
            out.append((last, rank, source, record))
    out.sort(key=lambda entry: (entry[0], entry[1]))
    return out


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _short(value: Any, limit: int = MAX_TEXT) -> str:
    text = value if isinstance(value, str) else ""
    return text if len(text) <= limit else text[:limit] + " …"


def _capped_args(args: Any) -> Any:
    """A tool call's arguments as the events file holds them: an object, or
    the JSON text cut short (``args_truncated``), kept within ``MAX_ARGS``."""
    if args is None:
        return None
    text = args if isinstance(args, str) else json.dumps(args, default=str)
    if len(text) <= MAX_ARGS:
        return args
    return text[:MAX_ARGS] + " …"


def _view_folder(path: str) -> str:
    """A view's folder as ``views.jsonl`` names it, from a picture path
    (``views/000015_look/view.jpg`` -> ``views/000015_look``)."""
    path = path.replace("\\", "/")
    return path[: -len(PICTURE) - 1] if path.endswith("/" + PICTURE) else path.rstrip("/")


def humanized(name: str) -> str:
    """A tool's name as the page shows it: ``position_sections`` -> ``Position sections``."""
    words = str(name or "tool").replace("_", " ").strip()
    return words[:1].upper() + words[1:]


def _sections_words(ids: list[Any], limit: int = 3) -> str:
    names = [str(i) for i in ids if i not in (None, "")]
    if len(names) <= limit:
        return ", ".join(names)
    return ", ".join(names[:limit]) + f" and {len(names) - limit} more"


@dataclass
class Picture:
    """One saved picture: its number (``views.jsonl`` ``seq``) and folder."""

    number: int
    folder: Path


@dataclass
class JobWatcher:
    """One job folder, watched (see the module text)."""

    folder: Path
    clock: Any = time.time
    generation: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.folder = Path(self.folder)
        self._reset()

    def _reset(self) -> None:
        f = self.folder
        self.generation += 1
        self.events = _Tail(f / LOGS_DIR / EVENTS_FILE)
        self.views = _Tail(f / VIEWS_INDEX)
        self.trace: _Tail | None = None
        self.state = _Whole(f / STATE_FILE)
        self.history = _Whole(f / HISTORY_DIR / "index.json")
        self.job = _Whole(f / JOB_FILE)
        self.spend = _Whole(f / EXPORTS_DIR / SPEND_FILE)
        self.items: list[dict[str, Any]] = []
        self.pictures: dict[int, Picture] = {}
        self.open_tools: dict[str, dict[str, Any]] = {}
        #: What made each undo depth (1 is the oldest step), and the redo side.
        self.step_labels: dict[int, str] = {}
        self.redo_labels: list[str] = []
        self.depth: int | None = None
        self.session: dict[str, Any] = {}
        self.summary: dict[str, Any] | None = None
        self.trace_error = False
        self.door: str | None = None
        self.state_version = 0
        self.rows: list[dict[str, Any]] = []
        self.rows_note = ""
        self.rows_stale = False
        self.polled_at = 0.0
        self._registrations: Any = None

    # --- reading ----------------------------------------------------------------

    def poll(self, *, force: bool = False) -> None:
        """Read what changed since the last round (at most every ``POLL_S``)."""
        with self._lock:
            now = self.clock()
            if not force and now - self.polled_at < POLL_S:
                return
            self.polled_at = now
            self._poll()

    def _poll(self) -> None:
        events, cut_events = self.events.read()
        views, cut_views = self.views.read()
        trace_records, cut_trace = self._read_trace()
        if cut_events or cut_views or cut_trace:
            self._reset()
            self._poll()
            return
        # The trace and the events interleaved by time (a first read of a
        # finished job reads both whole); a call's pictures land after its end.
        for _, _, source, record in _in_time_order(trace_records, events):
            (self._add_trace if source == "trace" else self._add_event)(record)
        for record in views:
            self._add_view(record)
        changed = bool(events or views or trace_records)
        if self.job.read():
            changed = True
        if self.spend.read():
            changed = True
        state_changed = self.state.read() or self.rows_stale
        history_changed = self.history.read()
        if state_changed:
            self.rows_stale = False
            self._rows()
        if state_changed or history_changed or changed:
            self.state_version += 1

    def _read_trace(self) -> tuple[list[dict[str, Any]], bool]:
        folder = self.folder / TRACE_DIR
        try:
            files = [p for p in folder.glob("*.jsonl") if p.is_file()]
        except OSError:
            files = []
        if files:
            newest = max(files, key=lambda p: (p.stat().st_mtime_ns, p.name))
            if self.trace is None or self.trace.path != newest:
                if self.trace is not None:
                    # A new session's file: finish the old one first.
                    older, _ = self.trace.read()
                    self.trace = _Tail(newest)
                    records, cut = self.trace.read()
                    return older + records, cut
                self.trace = _Tail(newest)
        if self.trace is None:
            return [], False
        return self.trace.read()

    def _append(self, item: dict[str, Any]) -> None:
        item["seq"] = len(self.items) + 1
        self.items.append(item)

    def _add_event(self, record: dict[str, Any]) -> None:
        kind = record.get("kind")
        at = record.get("at")
        if record.get("door"):
            self.door = str(record["door"])
        if kind in ("tool_start", "tool_end"):
            key = str(record.get("execution_id") or record.get("id") or len(self.items))
            item = {
                "kind": "tool", "phase": "start" if kind == "tool_start" else "end",
                "key": key, "name": str(record.get("name") or "tool"),
                "label": humanized(str(record.get("name") or "tool")),
                "sections": [str(s) for s in record.get("target_ids") or []],
                "failed": bool(record.get("failed")), "status": record.get("status"),
                "step": record.get("step"), "door": record.get("door"), "at": at,
                "views": [_view_folder(str(v)) for v in record.get("views") or []],
                "args": _capped_args(record.get("args")),
            }
            if kind == "tool_start":
                self.open_tools[key] = item
            else:
                self.open_tools.pop(key, None)
                self._label_step(item, record.get("step"))
            self._append(item)
        elif kind == "open":
            self._append({"kind": "note", "text": "Job reopened" if record.get("resumed")
                          else "Job opened", "at": at})
        elif kind == "script_write":
            # library.ScriptContext: the properties written, on which sections.
            what = ", ".join(str(p) for p in _list(record.get("property")))
            words = _sections_words([str(s) for s in _list(record.get("sections"))])
            self._append({"kind": "note", "at": at,
                          "text": "A script wrote " + (what or "to the job")
                          + (f" ({words})" if words else "")})
        elif kind == "plugins":
            self._append({"kind": "note", "at": at, "text": "Tool plugins loaded"})

    def _label_step(self, item: dict[str, Any], step: Any) -> None:
        """Follow the undo depth a ``tool_end`` reports (``step``: the undo
        stack's length after the call, at most ``UNDO_DEPTH``): a write that
        deepens it labels the new depth; at the cap a write drops the oldest
        label; undo and redo move labels between the two sides."""
        if not isinstance(step, int) or isinstance(step, bool):
            return
        before, self.depth = self.depth, step
        if item["failed"]:
            return
        words = _sections_words(item["sections"])
        label = item["label"] + (f" ({words})" if words else "")
        name = item["name"]
        if name == "undo":
            if before is not None:
                for depth in range(before, step, -1):
                    self.redo_labels.append(self.step_labels.pop(depth, f"Step {depth}"))
            return
        if name == "redo":
            if before is not None:
                for depth in range(before + 1, step + 1):
                    if self.redo_labels:
                        self.step_labels[depth] = self.redo_labels.pop()
            return
        at_cap = step == before == UNDO_DEPTH and name not in _READS
        if at_cap:
            self.step_labels = {d - 1: text for d, text in self.step_labels.items() if d > 1}
        if at_cap or (before is None and name not in _READS) or (
                before is not None and step > before):
            self.step_labels[step] = label
            self.redo_labels.clear()

    def _add_trace(self, record: dict[str, Any]) -> None:
        kind = record.get("kind")
        if kind not in _TRACE_KINDS:
            return
        at = record.get("at")
        if kind == "session":
            self.session = {"model": record.get("model"), "agent": record.get("agent")}
            self._append({"kind": "note", "at": at, "text": "Agent session started"
                          + (f" ({record['model']})" if record.get("model") else "")})
        elif kind == "model":
            if record.get("thought"):
                self._append({"kind": "reasoning", "at": at,
                              "text": _short(record["thought"])})
            if record.get("text"):
                self._append({"kind": "agent", "at": at, "text": _short(record["text"])})
        elif kind == "summary":
            self.summary = record
            submitted = bool(record.get("submitted"))
            self._append({"kind": "note", "at": at, "text": "Agent session ended"
                          + (", submitted" if submitted else " without a submission")})
        elif kind == "error":
            self.trace_error = True
            self._append({"kind": "note", "at": at, "error": True,
                          "text": _short(str(record.get("message") or record.get("error")
                                             or "The agent session failed"), 2000)})
        elif kind == "spend" and record.get("message"):
            self._append({"kind": "note", "at": at, "text": _short(str(record["message"]),
                                                                   500)})

    def _add_view(self, record: dict[str, Any]) -> None:
        number, path = record.get("seq"), record.get("path")
        if not isinstance(number, int) or not isinstance(path, str):
            return
        folder = (self.folder / path).resolve()
        try:
            folder.relative_to(self.folder.resolve())
        except ValueError:
            return
        self.pictures[number] = Picture(number, folder)
        caption = ""
        try:
            view = json.loads((folder / "view.json").read_text(encoding="utf-8"))
            caption = str(view.get("caption") or "")
        except (OSError, ValueError, AttributeError):
            pass
        self._append({
            "kind": "picture", "number": number, "path": _view_folder(path),
            "tool": str(record.get("tool") or ""), "label": humanized(str(record.get("tool")
                                                                      or "picture")),
            "call": record.get("call"), "sections": [str(s) for s in
                                                     record.get("sections") or []],
            "mode": record.get("mode"), "caption": _short(caption, 4000),
        })

    def _rows(self) -> None:
        """The sections table from ``state.json``: the status rows when this
        LangSlice reads the state, else the raw section fields."""
        data = self.state.value
        self.rows, self.rows_note = [], ""
        if not isinstance(data, dict):
            return
        try:
            from langslice.core.state import StackState
            from langslice.core.status import status_rows
            from langslice.job.checkpoint import current_state
            from langslice.job.registrations import Registrations

            state = StackState.from_dict(current_state(data))
            if self._registrations is None:
                self._registrations = Registrations(JobLayout(self.folder))
            records = {r.id: r for r in state.slices}
            for order, row in enumerate(status_rows(state), 1):
                record = records.get(row["id"])
                number = None
                if record is not None:
                    try:
                        number = self._registrations.number_of(state, record)
                    except Exception:  # a listing that cannot be read is no number
                        number = None
                angles = row.get("cutting_angles_deg") or (
                    state.cutting_angles_deg if not state.mixed_angles else None)
                self.rows.append({
                    "order": order, "id": row["id"], "position_mm": row.get("position_mm"),
                    "angles": angles, "flip": row.get("flip"),
                    "rotation_deg": row.get("rotation_deg"),
                    "transform": (row.get("transform") or {}).get("kind"),
                    "deformation": (row.get("deformation") or {}).get("method"),
                    "damaged": bool(row.get("damaged")), "registration": number,
                })
        except Exception as exc:  # a state this version does not read
            self.rows_note = ("This job's state is not one this LangSlice reads (an older "
                              "or newer format); the table shows its raw fields.")
            if not isinstance(exc, ValueError):
                self.rows_note += f" ({type(exc).__name__})"
            slices = _list(data.get("slices"))
            ordered = sorted((s for s in slices if isinstance(s, dict)),
                             key=lambda s: (s.get("index_corrected") is None,
                                            s.get("index_corrected") or 0))
            for order, s in enumerate(ordered, 1):
                transform = _dict(s.get("transform"))
                deformation = _dict(s.get("deformation"))
                steps = _list(deformation.get("steps"))
                method = None
                if deformation:
                    last = steps[-1] if steps and isinstance(steps[-1], dict) else {}
                    method = last.get("method") or ("trace_borders" if last.get("trace")
                                                    else "fireants_syn")
                self.rows.append({
                    "order": order, "id": s.get("id"), "position_mm": s.get("position_mm"),
                    "angles": s.get("cutting_angles_deg") or data.get("cutting_angles_deg"),
                    "flip": s.get("flip"), "rotation_deg": s.get("rotation_deg"),
                    "transform": transform.get("kind"), "deformation": method,
                    "damaged": bool(s.get("damaged_regions") or s.get("damaged")),
                    "registration": None,
                })

    def peek(self) -> None:
        """Read only ``job.json`` and ``state.json`` (what the job list shows),
        without tailing the logs."""
        with self._lock:
            self.job.read()
            if self.state.read():
                self.rows_stale = True

    # --- what the page gets -------------------------------------------------------

    def picture(self, number: int) -> Path | None:
        """The saved ``view.jpg`` of picture *number*, or None."""
        with self._lock:
            found = self.pictures.get(number)
        if found is None:
            return None
        path = found.folder / PICTURE
        return path if path.is_file() else None

    def last_change(self) -> float:
        """When any of the job's files last changed (seconds since the epoch)."""
        latest = 0.0
        for path in (self.events.path, self.views.path, self.state.path, self.history.path,
                     self.job.path, self.trace.path if self.trace else None):
            if path is None:
                continue
            try:
                latest = max(latest, path.stat().st_mtime)
            except OSError:
                continue
        return latest

    def header(self) -> dict[str, Any]:
        job = _dict(self.job.value)
        spec = _dict(job.get("spec"))
        image_folder = job.get("image_folder") or spec.get("image_folder")
        if image_folder:
            # job.json stores the default image folder relative to the job folder.
            image_folder = str((self.folder / str(image_folder)).resolve())
        name = (Path(str(image_folder)).name if image_folder else
                (self.folder.parent.name if self.folder.name == JOB_DIRNAME
                 else self.folder.name))
        return {
            "name": name, "folder": str(self.folder), "image_folder": image_folder,
            "atlas": spec.get("atlas"), "plane": spec.get("plane"),
            "tasks": spec.get("tasks"),
            "model": self.session.get("model") or spec.get("model"),
            "door": self.door, "exists": self.folder.is_dir(),
        }

    def status(self) -> dict[str, str]:
        state = _dict(self.state.value)
        recent = self.clock() - self.last_change() < RECENT_S
        if state.get("submitted"):
            word = "Submitted"
        elif self.trace_error or (self.summary is not None
                                  and not self.summary.get("submitted")):
            word = "Failed"
        elif self.open_tools or recent:
            word = "Running"
        else:
            word = "Waiting"
        if self.open_tools:
            item = list(self.open_tools.values())[-1]
            words = _sections_words(item["sections"])
            sentence = f"{item['label']}" + (f" on {words}" if words else "") + " (running)"
        else:
            # The last tool call; without tool events (an older folder), the last picture.
            last = next((i for i in reversed(self.items) if i["kind"] == "tool"), None) or next(
                (i for i in reversed(self.items) if i["kind"] == "picture"), None)
            if last is None:
                sentence = ("No activity yet." if self.folder.is_dir()
                            else "Waiting for the job folder to appear.")
            else:
                words = _sections_words(last.get("sections") or [])
                sentence = "Last: " + last["label"] + (f" on {words}" if words else "")
        if word == "Submitted":
            sentence = "The registration was submitted. " + sentence
        return {"word": word, "sentence": sentence}

    def undo(self) -> dict[str, list[str]]:
        """The undo and redo steps, oldest first, each labelled by the call
        that made it (``Step n`` when no event names it: an older folder)."""
        index = _dict(self.history.value)
        undo = _list(index.get("undo"))
        redo = _list(index.get("redo"))
        held = list(reversed(self.redo_labels))
        return {"undo": [self.step_labels.get(d, f"Step {d}") for d in range(1, len(undo) + 1)],
                "redo": [held[i] if i < len(held) else f"Step {len(undo) + i + 1}"
                         for i in range(len(redo))]}

    def notes(self) -> dict[str, Any]:
        job = _dict(self.job.value)
        state = _dict(self.state.value)
        run_notes = _list(state.get("notes"))
        return {"user": str(job.get("notes") or ""),
                "run": [str(n) for n in run_notes][-50:]}

    def spend_facts(self) -> dict[str, Any] | None:
        spend = self.spend.value if isinstance(self.spend.value, dict) else None
        if spend is None and self.summary is not None:
            raw = self.summary.get("spend")
            spend = raw if isinstance(raw, dict) else None
        if spend is None:
            return None
        tokens = _dict(spend.get("tokens"))
        return {"model": spend.get("model"), "lane": spend.get("lane"),
                "tokens": {k: v for k, v in tokens.items() if isinstance(v, (int, float))},
                "usd_estimate": spend.get("usd_estimate"),
                "quota_used_percent": spend.get("quota_used_percent")}

    def snapshot(self, since: int = 0, generation: int = 0,
                 state_version: int = -1) -> dict[str, Any]:
        """The page's view of the job: the activity items after *since* (all of
        them when *generation* is another watch's), and the tables when
        *state_version* is not the current one."""
        self.poll()
        with self._lock:
            reset = generation != self.generation
            start = 0 if reset else max(0, since)
            out: dict[str, Any] = {
                "generation": self.generation, "reset": reset,
                "cursor": len(self.items), "items": self.items[start:],
                "header": self.header(), "status": self.status(),
                "state_version": self.state_version,
            }
            if reset or state_version != self.state_version:
                out.update(sections=self.rows, sections_note=self.rows_note,
                           history=self.undo(), notes=self.notes(),
                           spend=self.spend_facts())
            return out


def _folder_of(record: dict[str, Any]) -> str | None:
    for key in ("folder", "job_folder", "path"):
        if isinstance(record.get(key), str) and record[key]:
            return record[key]
    return None


class JobList:
    """The jobs one GUI serves: the folders it was given (first) and the
    recent jobs file's (*recent*, None for none), newest first; only these
    folders are ever read."""

    def __init__(self, folders: list[Path] | None = None, *,
                 recent: Path | None = None) -> None:
        self.given = [Path(os.path.abspath(f)) for f in folders or []]
        self.recent = recent
        self.lock = threading.Lock()
        self.watchers: list[JobWatcher] = []
        self._index: dict[Path, int] = {}
        self._recent_stamp: tuple[int, int] | None = None
        for folder in self.given:
            self._add(folder)

    def _add(self, folder: Path) -> JobWatcher:
        known = self._index.get(folder)
        if known is not None:
            return self.watchers[known]
        watcher = JobWatcher(folder)
        self._index[folder] = len(self.watchers)
        self.watchers.append(watcher)
        return watcher

    def refresh(self) -> None:
        """Add the jobs the recent-jobs file names that are not listed yet."""
        if self.recent is None:
            return
        try:
            info = self.recent.stat()
            stamp = (info.st_size, info.st_mtime_ns)
            if stamp == self._recent_stamp:
                return
            lines = self.recent.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        self._recent_stamp = stamp
        seen: list[Path] = []
        for line in reversed(lines):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            folder = _folder_of(record) if isinstance(record, dict) else None
            if folder is None:
                continue
            path = Path(os.path.abspath(folder))
            if path not in seen and path.is_dir():
                seen.append(path)
            if len(seen) >= MAX_RECENT:
                break
        with self.lock:
            for path in reversed(seen):
                self._add(path)

    def get(self, index: int) -> JobWatcher | None:
        with self.lock:
            return self.watchers[index] if 0 <= index < len(self.watchers) else None

    def listing(self) -> list[dict[str, Any]]:
        self.refresh()
        with self.lock:
            watchers = list(self.watchers)
        out = []
        for n, watcher in enumerate(watchers):
            watcher.peek()
            header = watcher.header()
            out.append({"index": n, "name": header["name"], "folder": header["folder"],
                        "atlas": header["atlas"], "changed": watcher.last_change(),
                        "status": watcher.status()["word"],
                        "given": watcher.folder in self.given})
        return out
