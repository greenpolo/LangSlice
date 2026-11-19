"""The tools a job can have, and what of them can be switched off.

One :class:`ToolDef` per agent tool (the verbs of
:data:`langslice.tools.catalog.VERBS`, which binds each to its operation):
whether it writes, its task group, when a spec has it (``when``), whether it
can be switched off at all (``essential``) and what it needs while it is on
(``needs``). Every door is built from this list.

A job's ``disabled`` list
(:attr:`langslice.job.settings.spec.JobSpec.disabled`) takes tools off on top
of what its tasks and switches give: ``"zoom"`` is the tool, whole.

The valid names are derived from this table (:func:`valid_names`).
:func:`refusal` says, in one line, why a list cannot work; the spec refuses
it when it is made. The text a tool appears in is written with what it needs
and rendered for the job (:mod:`langslice.job.settings.needs`).

The in-tree engines' tools (DeepSlice, Brainways, STalign, SPOmiAlign,
emlddmm) are rows of this table like every other: ``engine`` names the
engine that holds the tool's declaration and handler, ``atlases`` and
``planes`` the jobs it is offered on. A tool with
``default_off`` is off unless the job's ``enabled`` list
(:attr:`langslice.job.settings.spec.JobSpec.enabled`) names it.

A job that loads tools of the user's own (``JobSpec.plugins``,
:mod:`langslice.job.plugins`) has them beside these: :func:`tools_for`
is the per-spec view (:data:`TOOLS` plus those tools), and the functions
below that take a *spec* or *tools* read it. Without any every answer is
the one :data:`TOOLS` gives.
"""

from __future__ import annotations

import fnmatch
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Protocol, cast

from langslice.core.tool_kinds import GROUPS as GROUPS
from langslice.core.tool_kinds import Group, Kind


def positions_on(spec: Any) -> bool:
    """Whether a run has ``position_sections``: the Positioning task, or the
    stack's cutting angles left to the agent (``transform.angles``)."""
    return bool(spec.has("position") or (spec.has("transform") and spec.transform.angles))


def _always(_spec: Any) -> bool:
    return True


class _Tasks(Protocol):
    """A job's settings, as far as a task switch reads them."""

    def has(self, task: str, /) -> bool: ...


def _has_any(tasks: tuple[str, ...], spec: _Tasks) -> bool:
    """Return whether a run of *spec* has any of *tasks* switched on."""
    return any(spec.has(task) for task in tasks)


def _task_on(*tasks: str) -> Callable[[_Tasks], bool]:
    """Return the switch of a tool that any of *tasks* turns on."""
    return partial(_has_any, tasks)


def _str_names(value: object) -> frozenset[str]:
    """Return the strings of a list, tuple or set setting, else none."""
    if not isinstance(value, (list, tuple, set, frozenset)):
        return frozenset()
    items = cast("Collection[object]", value)
    return frozenset(item for item in items if isinstance(item, str))


class _AtlasAndPlane(Protocol):
    """A job's settings, as far as an atlas and plane fit reads them."""

    atlas: str
    plane: str


@dataclass(frozen=True)
class ToolDef:
    """One tool, as far as a spec can switch it (see the module text)."""

    name: str
    kind: Kind
    group: Group
    when: Callable[[Any], bool] = _always
    long: bool = False
    scripting: bool = False
    image_model: bool = False
    essential: bool = False
    needs: tuple[tuple[str, ...], ...] = ()
    writes: tuple[str, ...] = ()
    engine: str = ""
    default_off: bool = False
    atlases: tuple[str, ...] = ()
    planes: tuple[str, ...] = ()

    def fits(self, spec: _AtlasAndPlane) -> bool:
        """Tell whether the tool is offered on a job's atlas and plane.

        Parameters
        ----------
        spec
            The job's settings.

        Returns
        -------
        bool
            True when ``atlases`` and ``planes`` each hold the job's, or
            are empty.
        """
        atlas, plane = str(spec.atlas), str(spec.plane)
        on_atlas = not self.atlases or any(
            fnmatch.fnmatchcase(atlas, pattern) for pattern in self.atlases
        )
        return on_atlas and (not self.planes or plane in self.planes)

    def is_on(self, spec: Any) -> bool:
        """Whether a run of *spec* has this tool: its ``when`` and not
        switched off (disabled, or with all it writes frozen)."""
        return bool(self.when(spec)) and not is_off(spec.switched_off, self.name)


TASK_WRITERS: dict[str, tuple[str, ...]] = {
    "position": ("position_sections",),
    "transform": ("interactive_transform", "elastix_affine"),
    "nonlinear": (
        "fireants_syn",
        "landmark_triangulation",
        "trace_borders",
        "stalign_lddmm",
        "spomialign",
        "emlddmm_stack",
    ),
}

_ENGINES = "langslice.engines."
DEEPSLICE = _ENGINES + "deepslice.plugin:tools"
BRAINWAYS = _ENGINES + "brainways.plugin:tools"
STALIGN = _ENGINES + "stalign.plugin:tools"
SPOMIALIGN = _ENGINES + "spomialign.plugin:tools"
EMLDDMM = _ENGINES + "emlddmm.plugin:tools"


TOOLS: dict[str, ToolDef] = {
    tool.name: tool
    for tool in (
        ToolDef("look", "read", "Common", essential=True),
        ToolDef("zoom", "read", "Common"),
        ToolDef("make_image", "write", "Common"),
        ToolDef("grep_atlas", "read", "Common"),
        ToolDef("grep_atlas_view", "read", "Common"),
        ToolDef("status", "read", "Common", essential=True),
        ToolDef("list_files", "read", "Common"),
        ToolDef("search_files", "read", "Common"),
        ToolDef("read_file", "read", "Common"),
        ToolDef("review_comments", "read", "Common"),
        ToolDef(
            "position_sections",
            "write",
            "Positioning",
            when=positions_on,
            writes=("position", "angles"),
        ),
        ToolDef(
            "interactive_transform",
            "write",
            "Linear",
            when=_task_on("transform"),
            writes=("orientation", "affine"),
        ),
        ToolDef("mark_damage", "write", "Common", writes=("damage",)),
        ToolDef(
            "make_mask",
            "write",
            "Common",
            when=_task_on("transform", "nonlinear"),
        ),
        ToolDef(
            "elastix_affine",
            "write",
            "Linear",
            long=True,
            when=_task_on("transform"),
            writes=("affine",),
        ),
        ToolDef(
            "fireants_syn",
            "write",
            "Nonlinear",
            long=True,
            when=_task_on("nonlinear"),
            writes=("deformation",),
        ),
        ToolDef(
            "set_landmarks",
            "write",
            "Nonlinear",
            when=_task_on("nonlinear"),
            needs=(("landmark_triangulation",),),
            writes=("deformation",),
        ),
        ToolDef(
            "landmark_triangulation",
            "write",
            "Nonlinear",
            when=_task_on("nonlinear"),
            needs=(("set_landmarks",),),
            writes=("deformation",),
        ),
        ToolDef(
            "trace_borders",
            "write",
            "Nonlinear",
            long=True,
            image_model=True,
            when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model,
            writes=("deformation",),
            default_off=True,
        ),
        ToolDef("note", "write", "Common"),
        ToolDef("undo", "write", "Common", essential=True),
        ToolDef("redo", "write", "Common"),
        ToolDef(
            "restore_registration",
            "write",
            "Common",
            when=_task_on("transform", "nonlinear"),
            writes=("orientation", "affine", "deformation"),
        ),
        ToolDef("submit", "write", "Common", essential=True),
        ToolDef("export_maps", "read", "Common", scripting=True, long=True),
        ToolDef(
            "deepslice_place",
            "write",
            "Positioning",
            when=positions_on,
            writes=("position", "angles", "affine"),
            engine=DEEPSLICE,
            default_off=True,
            atlases=("allen_mouse_*um", "ccfv3augmented_mouse_25um"),
            planes=("coronal",),
        ),
        ToolDef(
            "brainways_ap",
            "write",
            "Positioning",
            when=_task_on("position"),
            writes=("position",),
            engine=BRAINWAYS,
            default_off=True,
            atlases=("whs_sd_rat_39um",),
            planes=("coronal",),
        ),
        ToolDef(
            "stalign_lddmm",
            "write",
            "Nonlinear",
            long=True,
            when=_task_on("nonlinear"),
            writes=("affine", "deformation"),
            engine=STALIGN,
        ),
        ToolDef("stalign_plane_search", "read", "Positioning", when=positions_on, engine=STALIGN),
        ToolDef(
            "spomialign",
            "write",
            "Nonlinear",
            long=True,
            when=_task_on("transform", "nonlinear"),
            writes=("affine", "deformation"),
            engine=SPOMIALIGN,
        ),
        ToolDef(
            "emlddmm_stack",
            "write",
            "Nonlinear",
            long=True,
            when=_task_on("nonlinear"),
            writes=("affine", "deformation"),
            engine=EMLDDMM,
        ),
        ToolDef("emlddmm_plane", "read", "Positioning", when=positions_on, engine=EMLDDMM),
    )
}

ENGINES: tuple[str, ...] = tuple(
    dict.fromkeys(tool.engine for tool in TOOLS.values() if tool.engine)
)


def default_off_names() -> list[str]:
    """List the tools that are off unless a job's ``enabled`` names them.

    Returns
    -------
    list of str
        Their names, in table order.
    """
    return [tool.name for tool in TOOLS.values() if tool.default_off]


def engines_on(spec: object) -> list[str]:
    """List the engines a run has a tool of.

    Parameters
    ----------
    spec
        The job's settings.

    Returns
    -------
    list of str
        The engines' references, in table order.
    """
    on = [tool.engine for tool in TOOLS.values() if tool.engine and tool.is_on(spec)]
    return list(dict.fromkeys(on))


def check_enabled(names: Sequence[object]) -> str | None:
    """Say why a list is no ``enabled`` list.

    Parameters
    ----------
    names
        The list a job gives as ``enabled``.

    Returns
    -------
    str or None
        One line naming the first name that is no tool that starts off;
        None when every name is one.
    """
    known = default_off_names()
    for name in names:
        if name not in known:
            return (
                f"enabled: {name!r} is not a tool that starts off; the "
                f"tools that do: {', '.join(known)}. Every other tool "
                "is on unless disabled names it"
            )
    return None


def tools_for(spec: Any = None) -> Mapping[str, ToolDef]:
    """The tools a job of *spec* can have: :data:`TOOLS`, then the tools of the
    tools its ``plugins`` add (:mod:`langslice.job.plugins`; loaded when not
    yet, ``PluginError`` -- a ``ValueError`` -- when one cannot be). Without
    any, :data:`TOOLS` itself."""
    if not getattr(spec, "plugins", None):
        return TOOLS
    from langslice.job import plugins

    return {**TOOLS, **plugins.tool_defs(spec)}


def valid_names(spec: Any = None) -> list[str]:
    """Every name a ``disabled`` list takes, in table order: the tools that can
    be switched off, the tools of *spec*'s plugins after the native ones."""
    return [tool.name for tool in tools_for(spec).values() if not tool.essential]


def catalog(spec: Any = None) -> list[dict[str, Any]]:
    """The tools a host dialog can offer to switch off, in table order: name,
    group, kind, whether the tool can be disabled (``essential`` tools cannot)
    and what it needs (groups of names, one of each on). Plain data."""
    return [
        {
            "name": tool.name,
            "group": tool.group,
            "kind": tool.kind,
            "disableable": not tool.essential,
            "needs": [list(group) for group in tool.needs],
        }
        for tool in tools_for(spec).values()
    ]


def off_names(spec: Any) -> list[str]:
    """Every name that is off for *spec*: its ``disabled`` list, then the tools
    that start off and its ``enabled`` list does not name and the tools the
    job's atlas or plane does not fit (:meth:`ToolDef.fits`;
    :func:`langslice.job.plugins.not_offered` for a tool the job's ``plugins``
    add), so no text names them, then the tools the run has that can change
    nothing because everything they write (:attr:`ToolDef.writes`) is frozen
    on every section
    (:attr:`~langslice.job.settings.spec.JobSpec.frozen_everywhere`).
    A tool a task switched off is not listed: it is not in the run at all."""
    names = list(spec.disabled)
    enabled = _str_names(getattr(cast("object", spec), "enabled", ()))
    names.extend(
        tool.name
        for tool in TOOLS.values()
        if tool.when(spec)
        and not (
            tool.fits(cast("_AtlasAndPlane", spec))
            and (tool.name in enabled or not tool.default_off)
        )
    )
    if getattr(spec, "plugins", None):
        from langslice.job import plugins

        names.extend(plugins.not_offered(spec))
    if not spec.frozen:
        return list(dict.fromkeys(names))
    everywhere = spec.frozen_everywhere
    for tool in tools_for(spec).values():
        if tool.writes and set(tool.writes) <= everywhere and tool.when(spec):
            names.append(tool.name)
    return list(dict.fromkeys(names))


def is_off(disabled: Collection[str], name: str) -> bool:
    """Whether the tool *name* is switched off by *disabled*."""
    return name in disabled


def _either(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 else names[0]


def check_names(names: list[Any], spec: Any = None) -> str | None:
    """Why *names* hold a name no tool has, or a tool that cannot
    be switched off; None when they do not. *spec*: the job whose plugin
    tools are valid names too."""
    known = set(valid_names(spec))
    for name in names:
        if name in known:
            continue
        held = tools_for(spec).get(str(name))
        if held is not None and held.essential:
            return (
                f"disabled: {name} cannot be disabled (the run needs it); the tools that "
                f"can be: {', '.join(valid_names(spec))}"
            )
        return (
            f"disabled: unknown name {name!r}; the valid names are {', '.join(valid_names(spec))}"
        )
    return None


def refusal(spec: Any, *, image_model: bool = True) -> str | None:
    """One line saying why *spec*'s disabled list cannot work, or None when
    it can. The checks, in order: the names; every
    task that is on keeps a tool that writes its answer (without those that
    call the image model when *image_model* is False: a door that cannot
    reach it); a tool that is on keeps what it needs."""
    disabled = spec.disabled
    if not disabled:
        return None
    reason = check_names(disabled, spec)
    if reason is not None:
        return reason
    for task in spec.active_tasks:
        writers = TASK_WRITERS.get(task, ())
        if writers and not any(
            TOOLS[name].is_on(spec) and (image_model or not TOOLS[name].image_model)
            for name in writers
        ):
            return (
                f"disabled: the {task} task is on and every tool that writes its answer "
                f"({', '.join(writers)}) is off here; keep one of them on or leave the "
                "task out"
            )
    for tool in tools_for(spec).values():
        if not tool.is_on(spec):
            continue
        for group in tool.needs:
            if all(is_off(disabled, name) for name in group):
                return (
                    f"disabled: {tool.name} is on and needs {_either(list(group))}, "
                    f"which {'is' if len(group) == 1 else 'are'} off; turn "
                    f"{tool.name} off too or keep {_either(list(group))} on"
                )
    return None
