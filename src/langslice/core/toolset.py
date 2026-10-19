"""The tools a job can have, and what of them can be switched off.

One :class:`ToolDef` per agent tool (the verbs of
:data:`langslice.ops.registry.VERBS`, which binds each to its operation):
whether it writes, its task group, when a spec has it (``when``), the parts
of it that can be switched off on their own (``options``), whether it can be
switched off at all (``essential``) and what it needs while it is on
(``needs``). Every door is built from this list.

A job's ``disabled`` list (:attr:`langslice.core.spec.JobSpec.disabled`) takes
tools and options off on top of what its tasks and switches give:

- ``"zoom"``: the tool, whole;
- ``"look:positioning"``: one option of a tool (``tool:option``): here a
  ``look`` mode, taken out of the tool's schema, its description and every
  text that words it; ``"interactive_transform:flip"``: the mirror, and
  ``"interactive_transform:rotate"``: the quarter turn, parts of the tool that
  are no enumeration.

The valid names are derived from this table (:func:`valid_names`).
:func:`refusal` says, in one line, why a list cannot work; the spec refuses
it when it is made. The text a tool or option appears in is written with
what it needs and rendered for the job (:mod:`langslice.core.needs`).

A job that loads tool plugins (``JobSpec.plugins``,
:mod:`langslice.core.plugins`) has their tools beside these: :func:`tools_for`
is the per-spec view (:data:`TOOLS` plus the plugin tools), and the functions
below that take a *spec* or *tools* read it. With no plugins every answer is
the one :data:`TOOLS` gives.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

Kind = Literal["read", "write"]
Group = Literal["Common", "Positioning", "Linear", "Nonlinear"]
GROUPS: tuple[Group, ...] = ("Common", "Positioning", "Linear", "Nonlinear")

#: The ``look`` modes, in the order everything lists them.
LOOK_MODES: tuple[str, ...] = ("section", "atlas", "overlay", "positioning")

#: A tool-and-option name is ``tool:option``.
OPTION_SEPARATOR = ":"


def positions_on(spec: Any) -> bool:
    """Whether a run has ``position_sections``: the Positioning task, or the
    stack's cutting angles left to the agent (``transform.angles``)."""
    return bool(spec.has("position") or (spec.has("transform") and spec.transform.angles))


def _always(_spec: Any) -> bool:
    return True


@dataclass(frozen=True)
class ToolDef:
    """One tool, as far as a spec can switch it (see the module text)."""

    name: str
    kind: Kind
    group: Group
    #: Whether a run of a spec has this tool (a task's switch, a host
    #: switch); every tool exists in some run.
    when: Callable[[Any], bool] = _always
    #: A long verb computes outside the job's write lock and takes it only
    #: to apply, re-checking each section's inputs (``ops.inputs``); every
    #: other verb runs whole under the lock (the doors hold it).
    long: bool = False
    #: A scripting verb is offered by the doors a script or a coding agent
    #: uses (the agent CLI, the library) and never to a model through the
    #: agent tools or MCP.
    scripting: bool = False
    #: The verb calls the run's image model: a door that cannot reach one
    #: leaves it out.
    image_model: bool = False
    #: The parts of the tool that can be switched off on their own: option
    #: name -> the argument whose values the option is one of ("" when the
    #: option is no value of an argument). ``tool:option`` names them.
    options: Mapping[str, str] = field(default_factory=dict)
    #: The tool cannot be switched off (its options may be).
    essential: bool = False
    #: What the tool needs while it is on: groups of names, one of each group
    #: on (a tool, or a ``tool:option``). The reason is the tool's picture.
    needs: tuple[tuple[str, ...], ...] = ()
    #: The section properties the tool writes (:data:`langslice.core.frozen.PROPERTIES`):
    #: with every one of them frozen on every section the tool can change
    #: nothing and is off (:func:`off_names`).
    writes: tuple[str, ...] = ()

    def is_on(self, spec: Any) -> bool:
        """Whether a run of *spec* has this tool: its ``when`` and not
        switched off (disabled, or with all it writes frozen)."""
        return bool(self.when(spec)) and not is_off(spec.switched_off, self.name)


#: What a task needs written for its answer: it is refused, when it is on,
#: with every one of these off.
TASK_WRITERS: dict[str, tuple[str, ...]] = {
    "position": ("position_sections",),
    "transform": ("interactive_transform", "elastix_affine"),
    "nonlinear": ("fireants_syn", "landmark_triangulation", "trace_borders"),
}

_LOOK_OPTIONS = {mode: "mode" for mode in LOOK_MODES}
_OVERLAY = (("look:overlay",),)

#: Every tool, in the order every door lists them.
TOOLS: dict[str, ToolDef] = {tool.name: tool for tool in (
    # --- looking (every run) -----------------------------------------------------
    ToolDef("look", "read", "Common", options=_LOOK_OPTIONS, essential=True),
    ToolDef("zoom", "read", "Common"),
    ToolDef("set_channel_properties", "write", "Common"),
    ToolDef("set_preprocessed_channel_properties", "write", "Common"),
    ToolDef("grep_atlas", "read", "Common", options={"points": ""}),
    ToolDef("grep_atlas_view", "read", "Common"),
    ToolDef("status", "read", "Common", essential=True),
    ToolDef("list_files", "read", "Common"),
    ToolDef("search_files", "read", "Common"),
    ToolDef("read_file", "read", "Common"),
    # --- changing -------------------------------------------------------------------
    ToolDef("position_sections", "write", "Positioning", when=positions_on,
            needs=(("look:positioning", "look:overlay"),), writes=("position", "angles")),
    ToolDef("interactive_transform", "write", "Linear", options={"flip": "", "rotate": ""},
            when=lambda spec: spec.has("transform"), needs=_OVERLAY,
            writes=("orientation", "affine")),
    ToolDef("mark_damage", "write", "Common"),
    # --- fitting ----------------------------------------------------------------------
    ToolDef("elastix_affine", "write", "Linear", long=True,
            when=lambda spec: spec.has("transform"), needs=_OVERLAY, writes=("affine",)),
    ToolDef("fireants_syn", "write", "Nonlinear", long=True,
            when=lambda spec: spec.has("nonlinear"), needs=_OVERLAY,
            writes=("deformation",)),
    # The landmark pair: set_landmarks only places the pairs (it applies no
    # deformation); landmark_triangulation applies them. Each is useless
    # without the other, so each needs the other; set_landmarks lists
    # ``deformation`` as what it serves, so a deformation frozen on every
    # section takes both off.
    ToolDef("set_landmarks", "write", "Nonlinear",
            when=lambda spec: spec.has("nonlinear"),
            needs=(("look:atlas", "look:overlay"), ("look:section", "look:overlay"),
                   ("landmark_triangulation",)),
            writes=("deformation",)),
    ToolDef("landmark_triangulation", "write", "Nonlinear",
            when=lambda spec: spec.has("nonlinear"),
            needs=(("look:overlay",), ("set_landmarks",)), writes=("deformation",)),
    ToolDef("trace_borders", "write", "Nonlinear", long=True, image_model=True,
            when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model,
            needs=_OVERLAY, writes=("deformation",)),
    # --- bookkeeping (every run) --------------------------------------------------------
    ToolDef("note", "write", "Common"),
    ToolDef("undo", "write", "Common", essential=True),
    ToolDef("redo", "write", "Common"),
    ToolDef("submit", "write", "Common", essential=True),
    # The job folder's derived files on demand (submit writes them too):
    # changes no state, so a read; for scripts, not for a model.
    ToolDef("export_maps", "read", "Common", scripting=True, long=True),
)}


def tools_for(spec: Any = None) -> Mapping[str, ToolDef]:
    """The tools a job of *spec* can have: :data:`TOOLS`, then the tools of the
    plugins its ``plugins`` name (:mod:`langslice.core.plugins`; loaded when not
    yet, ``PluginError`` -- a ``ValueError`` -- when one cannot be). Without
    plugins, :data:`TOOLS` itself."""
    if not getattr(spec, "plugins", None):
        return TOOLS
    from langslice.core import plugins

    return {**TOOLS, **plugins.tool_defs(spec)}


def option_names(tool: str, tools: Mapping[str, ToolDef] = TOOLS) -> tuple[str, ...]:
    """The ``tool:option`` names of *tool*'s options."""
    return tuple(f"{tool}{OPTION_SEPARATOR}{option}" for option in tools[tool].options)


def valid_names(spec: Any = None) -> list[str]:
    """Every name a ``disabled`` list takes, in table order (a tool, then its
    options): the tools that can be switched off and every option, the tools
    of *spec*'s plugins after the native ones."""
    tools = tools_for(spec)
    names: list[str] = []
    for tool in tools.values():
        if not tool.essential:
            names.append(tool.name)
        names.extend(option_names(tool.name, tools))
    return names


def catalog(spec: Any = None) -> list[dict[str, Any]]:
    """The tools a host dialog can offer to switch off, in table order: name,
    group, kind, whether the tool can be disabled (``essential`` tools cannot,
    their options can), the names of its options and what it needs (groups
    of names, one of each on). Plain data."""
    return [{"name": tool.name, "group": tool.group, "kind": tool.kind,
             "disableable": not tool.essential, "options": list(tool.options),
             "needs": [list(group) for group in tool.needs]}
            for tool in tools_for(spec).values()]


def off_names(spec: Any) -> list[str]:
    """Every name that is off for *spec*: its ``disabled`` list, then the tools
    the run has that can change nothing because everything they write
    (:attr:`ToolDef.writes`) is frozen on every section
    (:attr:`~langslice.core.spec.JobSpec.frozen_everywhere`), and the mirror and
    the quarter turn (``interactive_transform:flip``, ``:rotate``) when the
    orientation is, and the plugin tools the job's atlas or plane does not
    fit (:func:`langslice.core.plugins.not_offered`), so no text names them.
    A tool a task switched off is not listed: it is not in the run at all."""
    names = list(spec.disabled)
    if getattr(spec, "plugins", None):
        from langslice.core import plugins

        names.extend(plugins.not_offered(spec))
    if not spec.frozen:
        return list(dict.fromkeys(names))
    everywhere = spec.frozen_everywhere
    for tool in tools_for(spec).values():
        if tool.writes and set(tool.writes) <= everywhere and tool.when(spec):
            names.append(tool.name)
    if "orientation" in everywhere:
        names.extend(f"interactive_transform{OPTION_SEPARATOR}{option}"
                     for option in ("flip", "rotate"))
    return list(dict.fromkeys(names))


def is_off(disabled: Collection[str], name: str) -> bool:
    """Whether *name* (a tool, or a ``tool:option``) is switched off by
    *disabled*: named there, or, for an option, its tool is."""
    return name in disabled or name.partition(OPTION_SEPARATOR)[0] in disabled


def options_on(disabled: Collection[str], tool: str) -> tuple[str, ...]:
    """The options of *tool* still on, in table order."""
    return tuple(option for option in TOOLS[tool].options
                 if not is_off(disabled, f"{tool}{OPTION_SEPARATOR}{option}"))


def _either(names: list[str]) -> str:
    return ", ".join(names[:-1]) + " or " + names[-1] if len(names) > 1 else names[0]


def check_names(names: list[Any], spec: Any = None) -> str | None:
    """Why *names* hold a name no tool or option has, or a tool that cannot
    be switched off; None when they do not. *spec*: the job whose plugin
    tools are valid names too."""
    known = set(valid_names(spec))
    for name in names:
        if name in known:
            continue
        held = tools_for(spec).get(str(name))
        if held is not None and held.essential:
            return (f"disabled: {name} cannot be disabled (the run needs it); the tools and "
                    f"options that can be: {', '.join(valid_names(spec))}")
        return (f"disabled: unknown name {name!r}; the valid names are "
                f"{', '.join(valid_names(spec))}")
    return None


def refusal(spec: Any, *, image_model: bool = True) -> str | None:
    """One line saying why *spec*'s disabled list cannot work, or None when
    it can. The checks, in order: the names; ``look`` keeps a mode; every
    task that is on keeps a tool that writes its answer (without those that
    call the image model when *image_model* is False: a door that cannot
    reach it); a tool that is on keeps what it needs."""
    disabled = spec.disabled
    if not disabled:
        return None
    reason = check_names(disabled, spec)
    if reason is not None:
        return reason
    if not options_on(disabled, "look"):
        return "disabled: every look mode is off; look keeps at least one of " \
               + ", ".join(option_names("look"))
    for task in spec.active_tasks:
        writers = TASK_WRITERS.get(task, ())
        if writers and not any(TOOLS[name].is_on(spec)
                               and (image_model or not TOOLS[name].image_model)
                               for name in writers):
            return (f"disabled: the {task} task is on and every tool that writes its answer "
                    f"({', '.join(writers)}) is off here; keep one of them on or leave the "
                    "task out")
    for tool in tools_for(spec).values():
        if not tool.is_on(spec):
            continue
        for group in tool.needs:
            if all(is_off(disabled, name) for name in group):
                return (f"disabled: {tool.name} is on and needs {_either(list(group))}, "
                        f"which {'is' if len(group) == 1 else 'are'} off; turn "
                        f"{tool.name} off too or keep {_either(list(group))} on")
    return None
