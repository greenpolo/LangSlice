"""Who a plugin context works for: a plugin tool, or a script."""

from __future__ import annotations

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
        return cls(name=tool.name, label=f"plugin:{tool.name}", kind=tool.kind,
                   writes=tuple(tool.writes), version=tool.version)

    @classmethod
    def script(cls) -> Caller:
        """A script's ``job.context``: writes every property the job leaves free,
        recorded as ``script``; no folder of its own (a script keeps its
        files in ``scripts/`` and ``scratch/``) and no background work (the
        script is the computation)."""
        return cls(name=SCRIPT_CALLER, label=SCRIPT_CALLER, kind="write",
                   writes=tuple(freeze.PROPERTIES), work_dir=False, in_tool_call=False,
                   merge_calls=False, background=False)

    @property
    def is_script(self) -> bool:
        return self.name == SCRIPT_CALLER

    def provenance(self, call: str, note: str, meta: Any) -> dict[str, Any]:
        """What a deformation record keeps of the caller that wrote it:
        ``plugin`` (tool, call, version, note, meta) or ``script`` (call,
        note, meta, and ``argv0``, the running script's path as started)."""
        if self.is_script:
            import sys

            return {"script": {"call": call, "note": note, "meta": meta,
                               "argv0": str(sys.argv[0]) if sys.argv else ""}}
        return {"plugin": {"tool": self.name, "call": call, "version": self.version,
                           "note": note, "meta": meta}}
