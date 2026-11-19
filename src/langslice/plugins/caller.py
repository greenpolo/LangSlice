"""Who a plugin context works for (a plugin tool, or a script) and which call it is."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from langslice.core.plugins import SCRIPT_CALLER, ToolSpec
from langslice.job.settings import frozen as freeze


@dataclass(frozen=True)
class Caller:
    """Who a :class:`ToolContext` works for: a plugin tool, or a script.

    ``name`` names it where a layer or background work records its maker
    (the tool's name, ``script``); ``label`` is what its writes are recorded
    as (a transform's ``kind``, a deformation's ``method`` and engine:
    ``plugin:<tool>`` or ``script``); ``kind`` and ``writes`` what it may
    write; ``version`` the plugin's own. ``work_dir``: it has a folder of
    its own (``plugins/<tool>/``); ``in_tool_call``: the tool door around it
    brings the job up to date before the call and drops the deformations
    its writes made stale after it; a script is in no tool call, so its
    reads sync first and its placement writes drop them themselves (in the
    same undo step); ``merge_calls``: its outermost writes join one undo
    step (one tool call) unless another write came between; ``background``:
    it may start background work."""

    name: str
    label: str
    kind: str
    writes: tuple[str, ...]
    version: str | None = None
    work_dir: bool = True
    in_tool_call: bool = True
    merge_calls: bool = True
    background: bool = True

    @classmethod
    def of_tool(cls, tool: ToolSpec) -> Caller:
        """A plugin tool's call."""
        return cls(
            name=tool.name,
            label=f"plugin:{tool.name}",
            kind=tool.kind,
            writes=tuple(tool.writes),
            version=tool.version,
        )

    @classmethod
    def script(cls) -> Caller:
        """A script's ``job.context``: writes every property the job leaves free,
        recorded as ``script``; no folder of its own (a script keeps its
        files in ``scripts/`` and ``scratch/``) and no background work (the
        script is the computation)."""
        return cls(
            name=SCRIPT_CALLER,
            label=SCRIPT_CALLER,
            kind="write",
            writes=tuple(freeze.PROPERTIES),
            work_dir=False,
            in_tool_call=False,
            merge_calls=False,
            background=False,
        )

    @property
    def is_script(self) -> bool:
        return self.name == SCRIPT_CALLER

    def provenance(self, call: str, note: str, meta: Any) -> dict[str, Any]:
        """What a deformation record keeps of the caller that wrote it:
        ``plugin`` (tool, call, version, note, meta) or ``script`` (call,
        note, meta, and ``argv0``, the running script's path as started)."""
        if self.is_script:
            import sys

            return {
                "script": {
                    "call": call,
                    "note": note,
                    "meta": meta,
                    "argv0": str(sys.argv[0]) if sys.argv else "",
                }
            }
        return {
            "plugin": {
                "tool": self.name,
                "call": call,
                "version": self.version,
                "note": note,
                "meta": meta,
            }
        }


class Call:
    """One call of a context: who it works for, its id and its write bookkeeping.

    The parts of a context that record the call's id or write share one
    :class:`Call`. ``depth`` counts the nested ``writing()`` blocks open and
    ``step`` is the last undo step the call made (start, pushed count, top
    step), which its next outermost write may join. A script's call is
    numbered: its id changes to ``script-<pid>-<n>`` when a write or a
    proposal begins and stays until the outermost write ends.

    Parameters
    ----------
    caller
        Who the call works for.
    call_id
        The call's id, which the layers and records it makes keep.
    """

    def __init__(self, caller: Caller, call_id: str) -> None:
        self.caller = caller
        self.id = call_id
        self.depth = 0
        self.step: tuple[int, int, int] | None = None
        self._count = 0
        self._pending = False

    def begin(self) -> None:
        """Number a script's call when it is not numbered yet (no effect for a
        plugin tool's call, whose id is the tool call's)."""
        if self.caller.is_script and self.depth == 0 and not self._pending:
            self._count += 1
            self.id = f"{SCRIPT_CALLER}-{os.getpid()}-{self._count}"
            self._pending = True

    def settle(self) -> None:
        """Let the next write of a script start a call of its own."""
        self._pending = False
