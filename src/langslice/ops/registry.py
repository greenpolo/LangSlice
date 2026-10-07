"""Every tool and the operation it calls: the one list the doors are built from.

One :class:`Verb` per agent tool (:func:`langslice.doors.tools.toolbox.build_tools`
returns exactly these names, checked by ``tests/test_ops_registry.py``): the
operation the tool body calls, whether it writes, and the task group a host
shows it under. ``Common`` tools are on in every run whatever the tasks
(``mark_damage`` behind its own host switch, ``agent_damage``); the others
come with their task: Positioning (``position``, or the stack's cutting
angles left to the agent, ``transform.angles``), Linear (``transform``) or
Nonlinear (``nonlinear``), the user-facing task names. A task that is off
removes its tools.

Every door is generated from this list: the agent tools and the
MCP tools (``build_tools`` builds the verbs :func:`enabled` names for the
run, in this order, each one declared by
:mod:`langslice.doors.declarations`), the agent CLI (``langslice-job ops``,
``langslice-job schema``, ``langslice-job FOLDER VERB``), the library's job
methods and the job folder's reference card.

A tool that has been replaced keeps answering by its old name: :data:`RETIRED`
maps each retired name to its replacement and a hint, and every door answers
a call by a retired name with :func:`retired_payload` (``RETIRED_TOOL``:
which tool to use instead; nothing was done), every time it is called. A
retired name never becomes a verb again.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from langslice.ops import (
    appearance,
    atlas,
    damage,
    deformable,
    exports,
    files,
    history,
    look,
    notes,
    positions,
    submit,
    traces,
    transforms,
    views,
)

Kind = Literal["read", "write"]
Group = Literal["Common", "Positioning", "Linear", "Nonlinear"]
GROUPS: tuple[Group, ...] = ("Common", "Positioning", "Linear", "Nonlinear")


@dataclass(frozen=True)
class Verb:
    """One tool: its name, its operation, read or write, its task group."""

    name: str
    function: Callable[..., Any]
    kind: Kind
    group: Group
    #: Whether a run of a spec has this verb (a task's switch, a host
    #: switch); every verb exists in some run.
    when: Callable[[Any], bool] = field(default=lambda _spec: True)
    #: A long verb computes outside the job's write lock and takes it only
    #: to apply, re-checking each section's inputs (``ops.inputs``); every
    #: other verb runs whole under the lock (the doors hold it).
    long: bool = False
    #: A scripting verb is offered by the doors a script or a coding agent
    #: uses (the agent CLI, the library) and never to a model through the
    #: agent tools or MCP (:func:`enabled` with ``scripting``).
    scripting: bool = False
    #: The verb calls the run's image model (``trace_borders``,
    #: ``trace_from_atlas``): a door that cannot reach one leaves it out
    #: (:func:`enabled` with ``image_model`` False).
    image_model: bool = False
    #: A hidden verb (always a scripting verb) is in no listing: not in
    #: :func:`enabled`'s answer unless asked for (``hidden=True``), not in
    #: ``langslice-job ops``, the ``langslice-job schema`` of every verb, the job
    #: folder's card, the library's ``verbs`` or the public docs. The
    #: scripting doors still build it, so the agent CLI (``langslice-job
    #: FOLDER VERB``) and the library call it by name (``trace_from_atlas``:
    #: kept reachable for experiments, not advertised).
    hidden: bool = False
    #: The most one call takes, by what it counts (``pictures`` shown,
    #: ``sections`` named), as the tool door enforces them (past a picture
    #: limit the rest is named, not shown; past a section limit the call is
    #: refused). The reference card lists them.
    limits: Mapping[str, int] = field(default_factory=dict)


def _verbs(*verbs: Verb) -> dict[str, Verb]:
    return {verb.name: verb for verb in verbs}


def positions_on(spec: Any) -> bool:
    """Whether a run has ``position_sections``: the Positioning task, or the
    stack's cutting angles left to the agent (``transform.angles``)."""
    return bool(spec.has("position") or (spec.has("transform") and spec.transform.angles))


#: Pictures one call of a picture verb shows; the rest are named, not shown.
_PICTURES: dict[str, int] = {"pictures": look.MAX_LOOK_PICTURES}

#: Every verb, in the order every door lists them.
VERBS: dict[str, Verb] = _verbs(
    # --- looking (every run) -----------------------------------------------------
    Verb("look", look.look, "read", "Common", limits=_PICTURES),
    Verb("zoom", look.zoom, "read", "Common"),
    Verb("set_channel_properties", appearance.set_channel_properties, "write", "Common"),
    Verb("set_preprocessed_channel_properties", appearance.set_preprocessed, "write",
         "Common", limits=_PICTURES),
    Verb("grep_atlas", atlas.grep_atlas, "read", "Common"),
    Verb("grep_atlas_view", atlas.grep_atlas_view, "read", "Common", limits=_PICTURES),
    Verb("status", views.status, "read", "Common"),
    Verb("list_files", files.list_files, "read", "Common"),
    Verb("search_files", files.search_files, "read", "Common"),
    Verb("read_file", files.read_file, "read", "Common"),
    # --- changing -------------------------------------------------------------------
    Verb("position_sections", positions.position_sections, "write", "Positioning",
         when=positions_on),
    Verb("interactive_transform", transforms.interactive_transform, "write", "Linear",
         limits={"sections": look.MAX_LOOK_PICTURES},
         when=lambda spec: spec.has("transform") and spec.transform.interactive),
    Verb("mark_damage", damage.mark_damage, "write", "Common",
         when=lambda spec: spec.agent_damage),
    # --- fitting ----------------------------------------------------------------------
    Verb("elastix_affine", transforms.elastix_affine, "write", "Linear", long=True,
         when=lambda spec: spec.has("transform") and spec.transform.automatic),
    Verb("ants_syn", deformable.ants_syn, "write", "Nonlinear", long=True,
         limits={"sections": deformable.MAX_ANTS_SYN_SECTIONS},
         when=lambda spec: spec.has("nonlinear")),
    Verb("trace_borders", traces.trace_borders, "write", "Nonlinear", long=True,
         image_model=True,
         when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model),
    # The placement-free trace (route "atlas"): kept for experiments, called
    # by name from the agent CLI and the library, listed nowhere.
    Verb("trace_from_atlas", traces.trace_from_atlas, "write", "Nonlinear", long=True,
         image_model=True, scripting=True, hidden=True,
         when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model),
    # --- bookkeeping (every run) --------------------------------------------------------
    Verb("note", notes.add_note, "write", "Common"),
    Verb("undo", history.undo, "write", "Common"),
    Verb("redo", history.redo, "write", "Common"),
    Verb("submit", submit.submit, "write", "Common"),
    # The job folder's derived files on demand (submit writes them too):
    # changes no state, so a read; for scripts, not for a model. Long: the
    # maps are computed outside the job lock and written under it.
    Verb("export_maps", exports.export_maps, "read", "Common", scripting=True, long=True),
)

#: Every retired tool name: ``(replacement, hint)``. The replacement is the
#: verb that does the retired tool's job now ("" when none does); the hint
#: says how to call it. Every door answers a call by one of these names with
#: :func:`retired_payload`.
RETIRED: dict[str, tuple[str, str]] = {
    "view_slices": ("look", 'mode "section" shows sections alone; zoom shows detail'),
    "view_atlas": ("look", 'mode "atlas" with positions_mm shows the atlas alone'),
    "view_placement": ("look", 'mode "overlay" shows a section under its registration; '
                       'mode "positioning" shows sections against atlas positions'),
    "view_stack": ("look", 'mode "positioning" without sections shows the whole stack '
                   "along the slicing axis"),
    "preprocess": ("set_preprocessed_channel_properties",
                   "it sets the channel that fits and the image model read; "
                   "set_channel_properties sets how a raw channel is displayed"),
    "set_positions": ("position_sections", "sections=[{id, position_mm}]"),
    "set_cutting_angles": ("position_sections", "cutting_angles={pitch_deg, yaw_deg}"),
    "reorder_slices": ("position_sections", "the stack's order follows the positions, so "
                       "writing the positions sets the order"),
    "orient_slices": ("interactive_transform", "sections=[{id, flip, rotate_quarter}]"),
    "adjust_transforms": ("interactive_transform",
                          "sections=[{id, rotation_deg, scale_x, scale_y, shear, "
                          "translate_x_mm, translate_y_mm}]; a value left out keeps the "
                          "section's current one"),
    "mark_damaged": ("mark_damage", "section and regions: the atlas regions the section "
                     "has lost"),
    "fit_affine": ("elastix_affine", "sections, restrict_to, atlas_image"),
    "fit_deformable": ("ants_syn", "sections, restrict_to, atlas_image, stiffness; a fit "
                       "of traced borders is part of trace_borders, and a section left "
                       "without a deformation is named in submit's left_linear"),
    "search_position": ("", "no tool searches for positions; compare a section with "
                        'atlas positions in look mode "positioning"'),
}


def retired_payload(name: str) -> dict[str, Any] | None:
    """The answer to a call of a retired tool (:data:`RETIRED`), or None when
    *name* is not retired: ``RETIRED_TOOL``, the tool to use (``use``) and a
    message naming it and how to call it. Nothing was done."""
    held = RETIRED.get(str(name))
    if held is None:
        return None
    replacement, hint = held
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
    has, in :data:`VERBS` order: the tools every door builds for it. The
    scripting verbs (``Verb.scripting``) only with *scripting*: the agent
    CLI's and the library's toolbox, never the agent tools or MCP.
    *image_model* False: the door cannot reach the spec's image model (MCP
    with none connected), so the verbs that call it (``Verb.image_model``)
    are left out. The hidden verbs (``Verb.hidden``) only with *hidden* as
    well as *scripting*: a door builds them, a listing never shows them."""
    return [name for name, verb in VERBS.items()
            if verb.when(spec) and (scripting or not verb.scripting)
            and (hidden or not verb.hidden)
            and (image_model or not verb.image_model)]


def listed() -> dict[str, Verb]:
    """Every verb a listing shows (``langslice-job ops``, ``langslice-job schema``,
    the job folder's card): :data:`VERBS` without the hidden ones."""
    return {name: verb for name, verb in VERBS.items() if not verb.hidden}
