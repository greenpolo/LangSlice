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

A job that loads tool plugins (``JobSpec.plugins``,
``docs/plugins.md``) has their tools after these: :func:`verbs_for` is the
per-spec view (:data:`VERBS` plus one :class:`Verb` per plugin tool, whose
function is the plugin's handler), and :func:`enabled` and :func:`listed` read
it. :data:`VERBS` stays the native table.

A tool that has been replaced keeps answering by its old name: :data:`RETIRED`
maps each retired name to its replacement and a hint, and every door answers
a call by a retired name with :func:`retired_payload` (``RETIRED_TOOL``:
which tool to use instead; nothing was done), every time it is called. A
retired name never becomes a verb again.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any

from langslice.core.needs import render
from langslice.core.toolset import (
    GROUPS,
    TOOLS,
    Group,
    Kind,
    ToolDef,
    is_off,
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

__all__ = ["GROUPS", "RETIRED", "VERBS", "Group", "Kind", "Verb", "enabled", "listed",
           "positions_on", "retired_payload", "verbs_for"]


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
        """The verb calls the run's image model (``trace_borders``,
        ``trace_from_atlas``): a door that cannot reach one leaves it out
        (:func:`enabled` with ``image_model`` False)."""
        return self.tool.image_model

    @property
    def hidden(self) -> bool:
        """A hidden verb (always a scripting verb) is in no listing: not in
        :func:`enabled`'s answer unless asked for (``hidden=True``), not in
        ``langslice-job ops``, the ``langslice-job schema`` of every verb, the job
        folder's card, the library's ``verbs`` or the public docs. The
        scripting doors still build it, so the agent CLI (``langslice-job
        FOLDER VERB``) and the library call it by name (``trace_from_atlas``:
        kept reachable for experiments, not advertised)."""
        return self.tool.hidden


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
    "trace_from_atlas": traces.trace_from_atlas,
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

def verbs_for(spec: Any = None) -> dict[str, Verb]:
    """Every verb a job of *spec* can have: :data:`VERBS`, then one verb per
    tool of the plugins its ``plugins`` name (loaded when not yet). Without
    plugins, :data:`VERBS` itself."""
    if not getattr(spec, "plugins", None):
        return VERBS
    from langslice.core import plugins

    return {**VERBS, **{
        name: Verb(name, tool.handler, tool.kind, tool.group, plugins.tool_def(tool),
                   limits=dict(tool.limits))
        for name, (_plugin, tool) in plugins.tool_specs(spec).items()}}


#: Every retired tool name: ``(replacement, hint)``. The replacement is the
#: verb that does the retired tool's job now ("" when none does); the hint
#: says how to call it. Every door answers a call by one of these names with
#: :func:`retired_payload`.
RETIRED: dict[str, tuple[str, str]] = {
    "view_slices": ("look", '[[?look:section mode "section" shows sections alone'
                    "[[?zoom ; zoom shows detail]]]]"),
    "view_atlas": ("look", '[[?look:atlas mode "atlas" with positions_mm shows the atlas '
                   "alone]]"),
    "view_placement": ("look", '[[?look:overlay mode "overlay" shows a section under its '
                       "registration[[?look:positioning ; ]]]]"
                       '[[?look:positioning mode "positioning" shows sections against '
                       "atlas positions]]"),
    "view_stack": ("look", '[[?look:positioning mode "positioning" without sections shows '
                   "the whole stack along the slicing axis]]"),
    "preprocess": ("set_preprocessed_channel_properties",
                   "it sets the channel that fits and the image model read"
                   "[[?set_channel_properties ; set_channel_properties sets how a raw "
                   "channel is displayed]]"),
    "set_positions": ("position_sections", "sections=[{id, position_mm}]"),
    "set_cutting_angles": ("position_sections", "cutting_angles={pitch_deg, yaw_deg}"),
    "reorder_slices": ("position_sections", "the stack's order follows the positions, so "
                       "writing the positions sets the order"),
    "orient_slices": ("interactive_transform", "sections=[{id[[?interactive_transform:flip , flip]]"
                      "[[?interactive_transform:rotate , rotate_quarter]]}]"),
    "adjust_transforms": ("interactive_transform",
                          "sections=[{id, rotation_deg, scale_x, scale_y, shear, "
                          "translate_x_mm, translate_y_mm}]; a value left out keeps the "
                          "section's current one"),
    "mark_damaged": ("mark_damage", "section and regions: the atlas regions the section "
                     "has lost"),
    "fit_affine": ("elastix_affine", "sections, restrict_to, atlas_image"),
    "fit_deformable": ("ants_syn", "sections, restrict_to, atlas_image, stiffness; "
                       "[[?trace_borders a fit of traced borders is part of trace_borders, "
                       "and ]]a section left without a deformation is named in submit's "
                       "left_linear"),
    "search_position": ("", "no tool searches for positions[[?look:positioning ; compare a "
                        'section with atlas positions in look mode "positioning"]]'),
}


def retired_payload(name: str, disabled: Collection[str] = ()) -> dict[str, Any] | None:
    """The answer to a call of a retired tool (:data:`RETIRED`), or None when
    *name* is not retired: ``RETIRED_TOOL``, the tool to use (``use``) and a
    message naming it and how to call it. Nothing was done. A job's
    *disabled* list takes out of the hint what it names; a replacement the
    job has disabled (or whose hint is left empty by it) is no replacement
    there."""
    held = RETIRED.get(str(name))
    if held is None:
        return None
    replacement, hint = held
    hint = render(hint, disabled)
    if replacement and (is_off(disabled, replacement) or not hint):
        replacement, hint = "", "this job does not have it"
    if replacement:
        message = f"{name} is no longer a tool. Use {replacement} instead: {hint}."
    else:
        message = f"{name} is no longer a tool, and nothing replaces it: {hint}."
    return {"status": "error", "error": "RETIRED_TOOL", "tool": str(name),
            **({"use": replacement} if replacement else {}),
            "message": message + " Nothing was done."}


def enabled(spec: Any, *, scripting: bool = False, image_model: bool = True,
            hidden: bool = False) -> list[str]:
    """The verbs a run of *spec* (a :class:`~langslice.core.spec.JobSpec`)
    has, in :data:`VERBS` order: the tools every door builds for it: those
    its tasks give, less the ones its ``disabled`` list names. The scripting
    verbs (``Verb.scripting``) only with *scripting*: the agent
    CLI's and the library's toolbox, never the agent tools or MCP.
    *image_model* False: the door cannot reach the spec's image model (MCP
    with none connected), so the verbs that call it (``Verb.image_model``)
    are left out. The hidden verbs (``Verb.hidden``) only with *hidden* as
    well as *scripting*: a door builds them, a listing never shows them."""
    return [name for name, verb in verbs_for(spec).items()
            if verb.tool.is_on(spec) and (scripting or not verb.scripting)
            and (hidden or not verb.hidden)
            and (image_model or not verb.image_model)]


def listed(spec: Any = None) -> dict[str, Verb]:
    """Every verb a listing shows (``langslice-job ops``, ``langslice-job schema``,
    the job folder's card): :data:`VERBS` without the hidden ones, and, for a
    *spec* that loads tool plugins, their tools after them."""
    return {name: verb for name, verb in verbs_for(spec).items() if not verb.hidden}
