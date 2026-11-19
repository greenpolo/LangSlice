"""The context a plugin tool's handler gets, and a script's ``job.context``:
what it may read and write.

One :class:`ToolContext` per plugin call (a plugin must not keep it); a
script's (:class:`langslice.doors.library.script_context.ScriptContext`) lives
as long as its job handle. Who is calling is a :class:`Caller`: a plugin tool
(:meth:`Caller.of_tool`) or a script (:meth:`Caller.script`). Reads give plain
data in LangSlice's documented units and frames; writes go through the same
operations the native tools use (``tools.positions``, ``tools.transforms``,
``tools.deformable``, ``tools.damage``), so every write is one undo step that
checkpoints the state and rewrites ``registration.json``, frozen properties are
refused (``FROZEN``), and a section that changed since a digest was taken is
refused (``STALE_INPUT``). A tool declared ``read`` cannot write; a ``write``
tool writes the section properties named in its
:attr:`~langslice.core.plugins.ToolSpec.writes` only; a script writes any
property the job does not freeze. A plugin's files go under
:meth:`ToolContext.work_dir` and nowhere else.

A context holds its parts as named attributes, each a plain class that is
given what it needs: ``reads`` (:class:`~langslice.plugins.reads.Reads`: the
job's sections, atlas and images), ``frames``
(:class:`~langslice.plugins.frames.Frames`: atlas planes and points),
``proposals`` (:class:`~langslice.plugins.proposals.Proposals`: a method's
result as LangSlice stores it), ``fits`` (:class:`~langslice.plugins.fits.Fits`:
outside files and the native engines), ``writes``
(:class:`~langslice.plugins.writes.Writes`) and ``pictures``
(:class:`~langslice.plugins.pictures.Pictures`). The context itself has what
is about the call: ``tool``, ``spec``, ``job_dir``, ``call``, ``work_dir``,
``task_on`` and ``start_work``.
"""

from __future__ import annotations

import copy
import types
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.core.jsonable import to_jsonable
from langslice.core.plugins import ToolError, ToolResult
from langslice.core.refusal import RefusalCode, Refused
from langslice.plugins.caller import Call, Caller
from langslice.plugins.fits import Fits
from langslice.plugins.frames import Frames, Grids
from langslice.plugins.pictures import Pictures
from langslice.plugins.proposals import Proposals
from langslice.plugins.reads import Reads, Records
from langslice.plugins.writes import Writes

if TYPE_CHECKING:
    from langslice.tools.run import Run


def _not_for_scripts(what: str) -> ToolError:
    return ToolError(RefusalCode.TOOL_OFF, what)


class ToolContext:
    """What one plugin tool call, or a script, can read and write (see the
    module text).

    Parameters
    ----------
    run
        The run the call belongs to: its job, workspace and spec.
    caller
        Who the context works for, a plugin tool or a script.
    call
        The call's id, which the layers and records it makes keep.
    """

    def __init__(self, run: Run, caller: Caller, call: str) -> None:
        self._run = run
        self._caller = caller
        self._call = Call(caller, call)
        self._records = Records(run, caller)
        grids = Grids(run, self._records)
        self.reads = Reads(run, self._records)
        self.frames = Frames(run, self._records, grids, self.reads)
        self.proposals = Proposals(run, self._call, self._records, grids)
        self.fits = Fits(run, self._records, grids, self.frames, self.proposals)
        self.writes = Writes(run, self._call, self._records, self.proposals)
        self.pictures = Pictures(run, self._records, self.proposals)

    @property
    def call(self) -> str:
        """The call's id."""
        return self._call.id

    @property
    def tool(self) -> str:
        """The tool's name (``script`` for a script)."""
        return self._caller.name

    @property
    def spec(self) -> Mapping[str, Any]:
        """The job's spec (``JobSpec.to_dict``), read-only."""
        return types.MappingProxyType(copy.deepcopy(self._run.spec.to_dict()))

    @property
    def job_dir(self) -> Path:
        """The job folder."""
        return Path(self._run.job.folder)

    def work_dir(self) -> Path:
        """``<job>/plugins/<tool>/``: the one place this tool writes files
        (scratch, logs, an engine's inputs and outputs). Made on first use.
        A script has none (``TOOL_OFF``): it keeps its files in the
        job folder's ``scripts/`` and ``scratch/``."""
        if not self._caller.work_dir:
            raise _not_for_scripts(
                "a script keeps its files in the job folder's scripts/ and scratch/"
            )
        path = self._run.job.layout.plugin_dir(self._caller.name)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_on(self, task: str) -> bool:
        """Whether the job's *task* (``position``, ``transform``, ``nonlinear``) is on."""
        return bool(self._run.spec.has(task))

    def start_work(
        self,
        sections: Sequence[object],
        land: Callable[[ToolContext], ToolResult],
    ) -> str:
        """Start *land* as a piece of the job's background work on *sections*
        and return its id at once (``w3``); the tool returns meanwhile.

        On a work thread ``land(ctx)`` computes and writes through *ctx*, a
        context of its own for the same tool (``with ctx.writes.writing():`` makes
        its writes one undo step; pass ``expect=`` digests taken at the
        start, as a long tool does). Its :class:`ToolResult` is the work's
        notice: ``reply["notice"]`` (one paragraph) its text, the reply its
        result and the pictures saved and numbered with the job's. A
        :class:`ToolError` or :class:`Refused` it raises fails the work with
        its message. The next tool reply carries the notice
        (``background``); ``status`` lists running work; ``submit`` waits
        for it. While it runs, a write to any of *sections*, by any tool, is
        refused ``BUSY``; so is starting it while other work runs
        there."""
        from langslice.job.background import DONE, FAILED, Landed

        if not self._caller.background:
            raise _not_for_scripts(
                "a script runs its own computation: compute, then write with expect="
            )
        job = self._run.job
        ids = [self._records.get(section).id for section in sections]
        busy = job.busy_refusal(ids)
        if busy is not None:
            raise busy
        door, caller = self._run, self._caller

        def landing() -> Landed:
            context = ToolContext(door, caller, f"{self.call}-bg")
            try:
                result = land(context)
            except Refused as refusal:
                return Landed(status=FAILED, text=str(refusal))
            reply = dict(result.reply)
            return Landed(
                status=DONE,
                text=str(reply.pop("notice", "") or "done."),
                pictures=[
                    item if isinstance(item, Image.Image) else Image.fromarray(np.asarray(item))
                    for item in result.pictures
                ],
                result=to_jsonable(reply),
            )

        return job.background.start(caller.name, ids, landing, wait_for_images=False).id
