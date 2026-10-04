"""Every tool and the operation it calls: the one list the doors are built from.

One :class:`Verb` per agent tool (:func:`langslice.doors.tools.toolbox.build_tools`
returns exactly these names, checked by ``tests/test_ops_registry.py``): the
operation the tool body calls, whether it writes, and the task group a host
shows it under. ``Common`` tools are on in every run whatever the tasks
(``mark_damaged`` and ``preprocess`` behind their own host switches); the
others come with their task, Positioning (``reorder`` + ``position``),
Linear (``transform``) or Nonlinear (``nonlinear``), as
``docs/interface_design.md`` names them. A tool that also exists without its
task (``view_placement`` with Linear or Nonlinear alone) is listed under the
group that introduces it.

Every door is generated from this list (phase 5): the agent tools and the
MCP tools (``build_tools`` builds the verbs :func:`enabled` names for the
run, in this order, each one declared by
:mod:`langslice.doors.declarations`), the agent CLI (``langslice ops``,
``langslice schema``, ``langslice job FOLDER VERB``), the library's job
methods and the job folder's reference card. A verb is never renamed once
shipped: scripts and agents call it by name.
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
    history,
    notes,
    order,
    orientation,
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
    #: Other operations the tool calls for some arguments, by what selects
    #: them (``fit_deformable``'s ``keep_linear`` argument records the linear
    #: placement instead of fitting).
    alternates: Mapping[str, Callable[..., Any]] = field(default_factory=dict)
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
    #: The verb calls the run's image model (``trace_borders``): a door that
    #: cannot reach one leaves it out (:func:`enabled` with ``image_model``
    #: False).
    image_model: bool = False


def _verbs(*verbs: Verb) -> dict[str, Verb]:
    return {verb.name: verb for verb in verbs}


#: Every verb, in the order every door lists them.
VERBS: dict[str, Verb] = _verbs(
    Verb("status", views.status, "read", "Common"),
    Verb("view_slices", views.view_slices, "read", "Common"),
    Verb("view_atlas", views.view_atlas, "read", "Common"),
    Verb("note", notes.add_note, "write", "Common"),
    Verb("undo", history.undo, "write", "Common"),
    Verb("redo", history.redo, "write", "Common"),
    Verb("mark_damaged", damage.mark_damaged, "write", "Common",
         when=lambda spec: spec.agent_damage),
    Verb("preprocess", appearance.preprocess, "write", "Common",
         when=lambda spec: spec.agent_preprocessing),
    Verb("reorder_slices", order.reorder, "write", "Positioning",
         when=lambda spec: spec.has("reorder")),
    Verb("set_positions", positions.set_positions, "write", "Positioning",
         when=lambda spec: spec.has("position")),
    # The read-only view of the complete registration (placement and applied
    # deformation), wherever there is a placement to look at.
    Verb("view_placement", views.view_placement, "read", "Positioning",
         when=lambda spec: any(spec.has(task) for task in ("position", "transform",
                                                             "nonlinear"))),
    Verb("view_stack", views.view_stack, "read", "Positioning",
         when=lambda spec: spec.has("position")),
    Verb("run_deepslice", positions.run_deepslice, "write", "Positioning",
         when=lambda spec: spec.has("position") and spec.position.deepslice),
    Verb("search_position", positions.search_position, "read", "Positioning",
         when=lambda spec: spec.has("position") and spec.position.bayesian),
    # Orientation (flip + quarter-turn) is part of in-plane alignment.
    Verb("orient_slices", orientation.orient_sections, "write", "Linear",
         when=lambda spec: spec.has("transform")),
    Verb("fit_affine", transforms.fit_affine, "write", "Linear", long=True,
         when=lambda spec: spec.has("transform") and spec.transform.automatic),
    Verb("adjust_transforms", transforms.adjust_transforms, "write", "Linear",
         when=lambda spec: spec.has("transform") and spec.transform.interactive),
    Verb("set_cutting_angles", positions.set_cutting_angles, "write", "Linear",
         when=lambda spec: spec.has("transform") and spec.transform.angles),
    Verb("trace_borders", traces.trace_borders, "write", "Nonlinear", long=True,
         image_model=True,
         when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model),
    Verb("grep_atlas", atlas.grep_atlas, "read", "Nonlinear",
         when=lambda spec: spec.has("nonlinear")),
    Verb("fit_deformable", deformable.fit_deformable, "write", "Nonlinear", long=True,
         alternates={"keep_linear": deformable.keep_linear},
         when=lambda spec: spec.has("nonlinear")),
    Verb("submit", submit.submit, "write", "Common"),
    # The job folder's derived files on demand (submit writes them too):
    # changes no state, so a read; for scripts, not for a model. Long: the
    # maps are computed outside the job lock and written under it.
    Verb("export_maps", exports.export_maps, "read", "Common", scripting=True, long=True),
)


def enabled(spec: Any, *, scripting: bool = False, image_model: bool = True) -> list[str]:
    """The verbs a run of *spec* (a :class:`~langslice.core.spec.JobSpec`)
    has, in :data:`VERBS` order: the tools every door builds for it. The
    scripting verbs (``Verb.scripting``) only with *scripting*: the agent
    CLI's and the library's toolbox, never the agent tools or MCP.
    *image_model* False: the door cannot reach the spec's image model (MCP
    with none connected), so the verbs that call it (``Verb.image_model``)
    are left out."""
    return [name for name, verb in VERBS.items()
            if verb.when(spec) and (scripting or not verb.scripting)
            and (image_model or not verb.image_model)]


def table() -> list[dict[str, str]]:
    """The registry as plain rows: tool, operation, kind, group."""
    return [
        {"tool": verb.name, "operation": f"{verb.function.__module__}.{verb.function.__name__}",
         "kind": verb.kind, "group": verb.group}
        for verb in VERBS.values()
    ]
