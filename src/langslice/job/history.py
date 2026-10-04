"""Undo/redo on disk: ``history/index.json`` plus one file per step.

Each step is a whole state, written once as its own file
(``step-000123.json``, a checkpoint: ``format_version`` + the state's
fields) when it enters the history; ``index.json`` lists the undo and redo
steps, oldest first, and is the only file rewritten per step. A step is
never rewritten, so one tool call costs one state file and a small index,
not the whole history: with depth 50 and a 40-section state of tens of
kilobytes, rewriting everything (the phase-2 ``linear_undo.json``) wrote
megabytes per call. Bounded: steps the index no longer names are deleted
after each index write.

An unreadable index or step starts the job without a history (logged): the
history is a convenience, the checkpoint is the record. A history from a
newer LangSlice is likewise ignored, never guessed at. Either way the
history on disk is left exactly as it is: :attr:`History.problem` is set and
:meth:`History.save` writes and deletes nothing until a later
:meth:`History.load` reads a valid one (undo works for the session, in
memory). Only a history that was read is pruned.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

from langslice.job.checkpoint import (
    FORMAT_KEY,
    STATE_FORMAT_VERSION,
    upgrade_state,
    write_json_atomic,
)

logger = logging.getLogger(__name__)

#: Undo steps kept (and saved). One tool call is one step, a batch included.
UNDO_DEPTH = 50
INDEX_FILE = "index.json"
#: The index's format. 1 (2026-10-03): ``undo`` and ``redo`` step file
#: names, oldest first, ``next`` the next step number.
HISTORY_FORMAT_VERSION = 1


def _step_name(number: int) -> str:
    return f"step-{number:06d}.json"


class History:
    """The history folder of one job, kept in line with the job's two stacks.

    The stacks themselves stay plain lists of state dicts on the job
    (:class:`langslice.job.job.Job`); :meth:`save` writes a file for each
    entry it has not written yet (matched by identity) and the index.
    """

    def __init__(self, folder: Path) -> None:
        self.folder = Path(folder)
        self.next = 1
        #: id(entry) -> (entry, step file name) for every entry on disk.
        self._known: dict[int, tuple[dict[str, Any], str]] = {}
        #: Why the history on disk could not be read (None: it was, or there
        #: is none). While set, :meth:`save` leaves the folder untouched.
        self.problem: str | None = None

    @property
    def index_path(self) -> Path:
        return self.folder / INDEX_FILE

    def _read_step(self, name: str) -> dict[str, Any]:
        data = json.loads((self.folder / name).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{name} does not hold a state")
        return upgrade_state(data, root=self.folder.parent)

    def load(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """``(undo, redo)`` from disk, each upgraded; empty without a history."""
        self._known = {}
        self.problem = None
        try:
            index = json.loads(self.index_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            if self.exists():  # step files without their index: not ours to delete
                return self._unreadable("its index is missing")
            return [], []
        except (OSError, ValueError) as exc:
            return self._unreadable(f"its index does not read ({exc})")
        version = index.get(FORMAT_KEY) if isinstance(index, dict) else None
        if not isinstance(version, int) or isinstance(version, bool) or version < 1:
            return self._unreadable(f"its index has an unreadable format {version!r}")
        if version > HISTORY_FORMAT_VERSION:
            return self._unreadable(f"it was written by a newer LangSlice (format {version})")
        try:
            undo_names = [str(name) for name in index.get("undo") or []]
            redo_names = [str(name) for name in index.get("redo") or []]
            undo = [self._read_step(name) for name in undo_names]
            redo = [self._read_step(name) for name in redo_names]
        except (OSError, ValueError) as exc:
            return self._unreadable(f"a step does not read ({exc})")
        self.next = max(int(index.get("next") or 1), 1)
        for entry, name in zip(undo + redo, undo_names + redo_names, strict=True):
            self._known[id(entry)] = (entry, name)
        return undo, redo

    def _unreadable(self, why: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.problem = (f"The undo history in {self.folder} cannot be used: {why}. It is left "
                        "untouched; undo covers this session's steps only.")
        logger.warning("%s", self.problem)
        return [], []

    def save(self, undo: list[dict[str, Any]], redo: list[dict[str, Any]]) -> None:
        """Write the steps not on disk yet, then the index; drop the rest.
        Nothing at all while the history on disk could not be read
        (:attr:`problem`)."""
        if self.problem is not None:
            return
        self.folder.mkdir(parents=True, exist_ok=True)
        known: dict[int, tuple[dict[str, Any], str]] = {}

        def name_of(entry: dict[str, Any]) -> str:
            held = self._known.get(id(entry))
            if held is not None and held[0] is entry:
                name = held[1]
            else:
                # Never over another writer's step (a second job on this folder).
                while (self.folder / _step_name(self.next)).exists():
                    self.next += 1
                name = _step_name(self.next)
                self.next += 1
                write_json_atomic(str(self.folder / name),
                                  {FORMAT_KEY: STATE_FORMAT_VERSION, **entry}, indent=None)
            known[id(entry)] = (entry, name)
            return name

        undo_names = [name_of(entry) for entry in undo]
        redo_names = [name_of(entry) for entry in redo]
        write_json_atomic(str(self.index_path), {
            FORMAT_KEY: HISTORY_FORMAT_VERSION,
            "state_format_version": STATE_FORMAT_VERSION,
            "depth": UNDO_DEPTH,
            "next": self.next,
            "undo": undo_names,
            "redo": redo_names,
        })
        self._known = known
        keep = set(undo_names) | set(redo_names) | {INDEX_FILE}
        for path in self.folder.iterdir():
            if path.name not in keep and path.name.startswith("step-"):
                try:
                    path.unlink()
                except OSError:
                    logger.warning("Could not remove old undo step %s", path, exc_info=True)

    def exists(self) -> bool:
        return self.index_path.exists() or (
            self.folder.is_dir() and any(name.startswith("step-")
                                         for name in os.listdir(self.folder)))
