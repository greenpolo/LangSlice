"""Every tool and the operation it calls: the one list the doors are built from.

One :class:`Verb` per agent tool (:func:`langslice.doors.tools.toolbox.build_tools`
returns exactly these names, checked by ``tests/test_ops_registry.py``): the
operation the tool body calls, whether it writes, and the task group a host
shows it under. What of a tool a spec can switch (its task, its options, what
it needs, whether it can be disabled) is the core's list,
:data:`langslice.core.toolset.TOOLS`, which this one binds to the operations.
``Common`` tools are on in every run whatever the tasks (``mark_damage``
unless the job disables it); the others come with their task: Positioning
(``position``, or the stack's cutting angles left to the agent,
``transform.angles``), Linear (``transform``) or Nonlinear (``nonlinear``),
the user-facing task names. A task that is off removes its tools, and a
job's ``disabled`` list (:attr:`langslice.core.spec.JobSpec.disabled`)
removes single tools and tool options on top.

Every door is generated from this list: the agent tools and the
MCP tools (``build_tools`` builds the verbs :func:`enabled` names for the
run, in this order, each one declared by
:mod:`langslice.doors.declarations`), the agent CLI (``langslice-job ops``,
``langslice-job schema``, ``langslice-job FOLDER VERB``), the library's job
methods and the job folder's reference card.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langslice.core.toolset import (
    GROUPS,
    TOOLS,
    Group,
    Kind,
    ToolDef,
    positions_on,
)
from langslice.ops import (
    appearance,
    atlas,
    damage,
    deformable,
    exports,
    files,
    history,
    landmarks,
    look,
    notes,
    positions,
    submit,
    traces,
    transforms,
    views,
)

__all__ = ["GROUPS", "VERBS", "Group", "Kind", "Verb", "enabled", "positions_on"]


@dataclass(frozen=True)
class Verb:
    """One tool: its name, its operation, read or write, its task group, and
    the rest of its :class:`~langslice.core.toolset.ToolDef`."""

    name: str
    function: Callable[..., Any]
    kind: Kind
    group: Group
    #: The core's definition: ``when``, ``options``, ``essential``, ``needs``.
    tool: ToolDef
    #: The most one call takes, by what it counts (``pictures`` shown,
    #: ``sections`` named), as the tool door enforces them (past a picture
    #: limit the rest is named, not shown; past a section limit the call is
    #: refused). The reference card lists them.
    limits: Mapping[str, int] = field(default_factory=dict)

    @property
    def when(self) -> Callable[[Any], bool]:
        """Whether a run of a spec has this verb (a task's switch, a host
        switch); every verb exists in some run."""
        return self.tool.when

    @property
    def long(self) -> bool:
        """A long verb computes outside the job's write lock and takes it
        only to apply, re-checking each section's inputs (``ops.inputs``);
        every other verb runs whole under the lock (the doors hold it)."""
        return self.tool.long

    @property
    def scripting(self) -> bool:
        """A scripting verb is offered by the doors a script or a coding agent
        uses (the agent CLI, the library) and never to a model through the
        agent tools or MCP (:func:`enabled` with ``scripting``)."""
        return self.tool.scripting

    @property
    def image_model(self) -> bool:
        """The verb calls the run's image model (``trace_borders``): a door
        that cannot reach one leaves it out
        (:func:`enabled` with ``image_model`` False)."""
        return self.tool.image_model

#: Pictures one call of a picture verb shows; the rest are named, not shown.
_PICTURES: dict[str, int] = {"pictures": look.MAX_LOOK_PICTURES}

#: The operation each tool calls.
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "look": look.look,
    "zoom": look.zoom,
    "set_channel_properties": appearance.set_channel_properties,
    "set_preprocessed_channel_properties": appearance.set_preprocessed,
    "grep_atlas": atlas.grep_atlas,
    "grep_atlas_view": atlas.grep_atlas_view,
    "status": views.status,
    "list_files": files.list_files,
    "search_files": files.search_files,
    "read_file": files.read_file,
    "position_sections": positions.position_sections,
    "interactive_transform": transforms.interactive_transform,
    "mark_damage": damage.mark_damage,
    "elastix_affine": transforms.elastix_affine,
    "ants_syn": deformable.ants_syn,
    "set_landmarks": landmarks.set_landmarks,
    "landmark_triangulation": landmarks.landmark_triangulation,
    "trace_borders": traces.trace_borders,
    "note": notes.add_note,
    "undo": history.undo,
    "redo": history.redo,
    "submit": submit.submit,
    "export_maps": exports.export_maps,
}

#: The per-call limits of the tools that have them.
_LIMITS: dict[str, dict[str, int]] = {
    "look": _PICTURES,
    "set_preprocessed_channel_properties": _PICTURES,
    "grep_atlas_view": _PICTURES,
    "interactive_transform": {"sections": look.MAX_LOOK_PICTURES},
    "ants_syn": {"sections": deformable.MAX_ANTS_SYN_SECTIONS},
}

if set(_FUNCTIONS) != set(TOOLS):
    raise RuntimeError("the registry and langslice.core.toolset.TOOLS name different tools: "
                       f"{sorted(set(_FUNCTIONS) ^ set(TOOLS))}")

#: Every verb, in the order every door lists them.
VERBS: dict[str, Verb] = {
    name: Verb(name, _FUNCTIONS[name], tool.kind, tool.group, tool,
               limits=_LIMITS.get(name, {}))
    for name, tool in TOOLS.items()
}

def enabled(spec: Any, *, scripting: bool = False, image_model: bool = True) -> list[str]:
    """The verbs a run of *spec* (a :class:`~langslice.core.spec.JobSpec`)
    has, in :data:`VERBS` order: the tools every door builds for it: those
    its tasks give, less the ones its ``disabled`` list names. The scripting
    verbs (``Verb.scripting``) only with *scripting*: the agent
    CLI's and the library's toolbox, never the agent tools or MCP.
    *image_model* False: the door cannot reach the spec's image model (MCP
    with none connected), so the verbs that call it (``Verb.image_model``)
    are left out."""
    return [name for name, verb in VERBS.items()
            if verb.tool.is_on(spec) and (scripting or not verb.scripting)
            and (image_model or not verb.image_model)]
