"""The script door: ``langslice.open_job(folder)`` and the verbs as methods.

A script (or a coding agent's Python) opens a job folder and calls the same
verbs the agent tools and the CLI offer, by the same names and arguments
(:mod:`langslice.doors.declarations`)::

    import langslice

    job = langslice.open_job("/data/M04")          # the job folder or its images
    job.status()["rows"]
    reply = job.set_positions(entries=[{"id": "s01.tif", "position_mm": 5.2}])
    reply["images"]                                 # the pictures, as PIL images

Each method returns the tool's reply as a dict, its pictures (PIL images)
under ``"images"``; every picture is also saved in the job folder with its
layers, as every door saves them. The look-before-commit gates do not apply
(gates are tool-only) and ``view.resolution`` takes any size from 128 px to
the source's own pixels. Writes go through the job: one undo step each,
checkpointed, and picked up by an agent working on the same folder
(``Job.sync``), whose writes this handle picks up before each call.

Nothing here loads the agent framework or a model client.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from langslice.doors.jobs import Opened, open_folder


class JobHandle:
    """One open job folder; every verb its settings have is a method."""

    def __init__(self, opened: Opened) -> None:
        self._opened = opened
        self._tools = {tool.__name__: tool for tool in opened.tools().tools}

    # --- the verbs ------------------------------------------------------------------

    @property
    def verbs(self) -> list[str]:
        """The verbs this job has (its tasks decide), in registry order."""
        return list(self._tools)

    def __getattr__(self, name: str) -> Callable[..., dict[str, Any]]:
        tools = self.__dict__.get("_tools") or {}
        if name in tools:
            return tools[name]
        from langslice.ops.registry import VERBS

        if name in VERBS:
            raise AttributeError(f"This job's settings have no {name!r} (tasks "
                                 f"{self._opened.spec.tasks}); see .verbs")
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self._tools})

    # --- the job ----------------------------------------------------------------------

    @property
    def folder(self) -> str:
        """The job folder."""
        return str(self._opened.job.folder)

    @property
    def job(self) -> Any:
        """The job itself (:class:`langslice.job.job.Job`: state, undo, gates)."""
        return self._opened.job

    @property
    def state(self) -> Any:
        """The stack as it stands (:class:`langslice.core.state.StackState`);
        read it, write through the verbs."""
        return self._opened.job.state

    @property
    def workspace(self) -> Any:
        """The atlas and the section files (:class:`langslice.core.workspace.Workspace`)."""
        return self._opened.ctx

    def close(self) -> None:
        """Finish the job's background writes (pictures, image corrections)."""
        self._opened.close()

    def __enter__(self) -> JobHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<langslice job {self.folder} ({len(self.state.slices)} sections)>"


def open_job(
    folder: str | os.PathLike[str], *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> JobHandle:
    """Open the job in *folder* (the job folder, or the image folder beside
    it). Create one first with ``langslice job <images> init``.
    *atlas_loader* replaces BrainGlobe's (tests, offline hosts)."""
    return JobHandle(open_folder(folder, atlas_loader=atlas_loader, emit=emit))
