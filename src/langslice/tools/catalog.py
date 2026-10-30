"""Every tool and the operation it calls: the one list the doors are built from.

One :class:`Verb` per agent tool (:func:`langslice.doors.tools.toolbox.build_tools`
returns exactly these names, checked by ``tests/test_ops_registry.py``): the
operation the tool body calls, whether it writes, and the task group a host
shows it under. What of a tool a spec can switch (its task, its options, what
it needs, whether it can be disabled) is the job's tool table,
:data:`langslice.job.settings.tool_table.TOOLS`, which this one binds to the
operations.
``Common`` tools are on in every run whatever the tasks (``mark_damage``
unless the job disables it); the others come with their task: Positioning
(``position``, or the stack's cutting angles left to the agent,
``transform.angles``), Linear (``transform``) or Nonlinear (``nonlinear``),
the user-facing task names. A task that is off removes its tools, and a
job's ``disabled`` list (:attr:`langslice.job.settings.spec.JobSpec.disabled`)
removes single tools and tool options on top.

Every door is generated from this list: the agent tools and the MCP tools
(``build_tools`` builds the verbs :func:`enabled` names for the run, in this
order, each one declared by :mod:`langslice.tools.descriptions.render`), the
agent CLI (``langslice-job ops``, ``langslice-job schema``, ``langslice-job
FOLDER VERB``), the library's job methods and the job folder's reference card.

A job that loads tool plugins (``JobSpec.plugins``,
``docs/developers/plugins.md``) has their tools after these: :func:`verbs_for` is the
per-spec view (:data:`VERBS` plus one :class:`Verb` per plugin tool, whose
function is the plugin's handler), and :func:`enabled` reads it. :data:`VERBS`
stays the native table.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from langslice.job.settings.tool_table import (
    GROUPS,
    TOOLS,
    Group,
    Kind,
    ToolDef,
    positions_on,
)
from langslice.tools import (
    affine_fit,
    atlas,
    damage,
    deformable,
    exports,
    files,
    history,
    images,
    landmark_warp,
    landmarks,
    look,
    masks,
    notes,
    positions,
    registrations,
    status,
    submit,
    traces,
    transforms,
    zoom,
)

__all__ = ["GROUPS", "VERBS", "Group", "Kind", "Verb", "enabled", "positions_on", "verbs_for"]


@dataclass(frozen=True)
class Verb:
    """One tool: its name, its operation, read or write, its task group, and
    the rest of its :class:`~langslice.job.settings.tool_table.ToolDef`."""

    name: str
    function: Callable[..., Any]
    kind: Kind
    group: Group
    #: The core's definition: ``when``, ``options``, ``essential``, ``needs``.
    tool: ToolDef
    #: The most one call takes at every door, by what it counts (``sections``
    #: named; a plugin's ``pictures`` shown), as the tool door enforces them
    #: (past a picture limit the rest is named, not shown; past a section limit
    #: the call is refused). The reference card lists them. The picture cap of
    #: LangSlice's own agent and MCP is the job's (``Job.picture_cap``).
    limits: Mapping[str, int] = field(default_factory=dict)

    @property
    def when(self) -> Callable[[Any], bool]:
        """Whether a run of a spec has this verb (a task's switch, a host
        switch); every verb exists in some run."""
        return self.tool.when

    @property
    def long(self) -> bool:
        """A long verb computes outside the job's write lock and takes it
        only to apply, re-checking each section's inputs (``tools.stale``);
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

#: The operation each tool calls.
_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "look": look.look_pictures,
    "zoom": zoom.zoom_picture,
    "make_image": images.make_section_image,
    "grep_atlas": atlas.search_regions,
    "grep_atlas_view": atlas.draw_regions,
    "status": status.stack_status,
    "list_files": files.list_folder,
    "search_files": files.search_folder,
    "read_file": files.read_folder_file,
    "position_sections": positions.position_sections,
    "interactive_transform": transforms.interactive_transform,
    "mark_damage": damage.mark_damage,
    "make_mask": masks.make_mask,
    "elastix_affine": affine_fit.elastix_affine,
    "fireants_syn": deformable.fireants_syn,
    "set_landmarks": landmarks.set_landmarks,
    "landmark_triangulation": landmark_warp.landmark_triangulation,
    "trace_borders": traces.trace_borders,
    "note": notes.add_note,
    "undo": history.undo,
    "redo": history.redo,
    "restore_registration": registrations.restore,
    "submit": submit.submit,
    "export_maps": exports.export_maps,
}

#: The per-call limits every door shares (compute, not pictures: the picture
#: cap of the agent tools and MCP is the job's, ``Job.picture_cap``).
_LIMITS: dict[str, dict[str, int]] = {
    "fireants_syn": {"sections": deformable.MAX_FIREANTS_SYN_SECTIONS},
}

if set(_FUNCTIONS) != set(TOOLS):
    raise RuntimeError("the registry and the tool table name different "
                       f"tools: {sorted(set(_FUNCTIONS) ^ set(TOOLS))}")

#: Every verb, in the order every door lists them.
VERBS: dict[str, Verb] = {
    name: Verb(name, _FUNCTIONS[name], tool.kind, tool.group, tool,
               limits=_LIMITS.get(name, {}))
    for name, tool in TOOLS.items()
}

def verbs_for(spec: Any = None) -> dict[str, Verb]:
    """Every verb a job of *spec* can have: :data:`VERBS`, then one verb per
    tool of the plugins its ``plugins`` name (loaded when not yet). Without
    plugins, :data:`VERBS` itself."""
    if not getattr(spec, "plugins", None):
        return VERBS
    from langslice.job import plugins

    return {**VERBS, **{
        name: Verb(name, tool.handler, tool.kind, tool.group, plugins.tool_def(tool),
                   limits=dict(tool.limits))
        for name, (_plugin, tool) in plugins.tool_specs(spec).items()}}


def enabled(spec: Any, *, scripting: bool = False, image_model: bool = True) -> list[str]:
    """The verbs a run of *spec*
    (a :class:`~langslice.job.settings.spec.JobSpec`) has, in :data:`VERBS`
    order: the tools every door builds for it: those
    its tasks give, less the ones its ``disabled`` list names. The scripting
    verbs (``Verb.scripting``) only with *scripting*: the agent
    CLI's and the library's toolbox, never the agent tools or MCP.
    *image_model* False: the door cannot reach the spec's image model (MCP
    with none connected), so the verbs that call it (``Verb.image_model``)
    are left out."""
    return [name for name, verb in verbs_for(spec).items()
            if verb.tool.is_on(spec) and (scripting or not verb.scripting)
            and (image_model or not verb.image_model)]
