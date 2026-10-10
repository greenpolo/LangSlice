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
  text that words it; ``"interactive_transform:flip"``: the mirror, a part
  of the tool that is no enumeration.

The valid names are derived from this table (:func:`valid_names`).
:func:`refusal` says, in one line, why a list cannot work; the spec refuses
it when it is made. The text a tool or option appears in is written with
what it needs and rendered for the job (:mod:`langslice.core.needs`).
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
    #: A hidden verb (always a scripting verb) is in no listing.
    hidden: bool = False
    #: The parts of the tool that can be switched off on their own: option
    #: name -> the argument whose values the option is one of ("" when the
    #: option is no value of an argument). ``tool:option`` names them.
    options: Mapping[str, str] = field(default_factory=dict)
    #: The tool cannot be switched off (its options may be).
    essential: bool = False
    #: What the tool needs while it is on: groups of names, one of each group
    #: on (a tool, or a ``tool:option``). The reason is the tool's picture.
    needs: tuple[tuple[str, ...], ...] = ()

    def is_on(self, spec: Any) -> bool:
        """Whether a run of *spec* has this tool: its ``when`` and not
        switched off."""
        return bool(self.when(spec)) and not is_off(spec.disabled, self.name)


#: What a task needs written for its answer: it is refused, when it is on,
#: with every one of these off.
TASK_WRITERS: dict[str, tuple[str, ...]] = {
    "position": ("position_sections",),
    "transform": ("interactive_transform", "elastix_affine"),
    "nonlinear": ("ants_syn", "trace_borders"),
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
    ToolDef("grep_atlas", "read", "Common"),
    ToolDef("grep_atlas_view", "read", "Common"),
    ToolDef("status", "read", "Common", essential=True),
    ToolDef("list_files", "read", "Common"),
    ToolDef("search_files", "read", "Common"),
    ToolDef("read_file", "read", "Common"),
    # --- changing -------------------------------------------------------------------
    ToolDef("position_sections", "write", "Positioning", when=positions_on,
            needs=(("look:positioning", "look:overlay"),)),
    ToolDef("interactive_transform", "write", "Linear", options={"flip": ""},
            when=lambda spec: spec.has("transform"), needs=_OVERLAY),
    ToolDef("mark_damage", "write", "Common"),
    # --- fitting ----------------------------------------------------------------------
    ToolDef("elastix_affine", "write", "Linear", long=True,
            when=lambda spec: spec.has("transform"), needs=_OVERLAY),
    ToolDef("ants_syn", "write", "Nonlinear", long=True,
            when=lambda spec: spec.has("nonlinear"), needs=_OVERLAY),
    ToolDef("trace_borders", "write", "Nonlinear", long=True, image_model=True,
            when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model,
            needs=_OVERLAY),
    # The placement-free trace (route "atlas"): kept for experiments, called
    # by name from the agent CLI and the library, listed nowhere.
    ToolDef("trace_from_atlas", "write", "Nonlinear", long=True, image_model=True,
            scripting=True, hidden=True,
            when=lambda spec: spec.has("nonlinear") and spec.nonlinear.uses_image_model),
    # --- bookkeeping (every run) --------------------------------------------------------
    ToolDef("note", "write", "Common"),
    ToolDef("undo", "write", "Common", essential=True),
    ToolDef("redo", "write", "Common"),
    ToolDef("submit", "write", "Common", essential=True),
    # The job folder's derived files on demand (submit writes them too):
    # changes no state, so a read; for scripts, not for a model.
    ToolDef("export_maps", "read", "Common", scripting=True, long=True),
)}


def option_names(tool: str) -> tuple[str, ...]:
    """The ``tool:option`` names of *tool*'s options."""
    return tuple(f"{tool}{OPTION_SEPARATOR}{option}" for option in TOOLS[tool].options)


def valid_names() -> list[str]:
    """Every name a ``disabled`` list takes, in table order (a tool, then its
    options): the tools that can be switched off and every option. The hidden
    scripting verbs are accepted and not listed."""
    names: list[str] = []
    for tool in TOOLS.values():
        if not tool.essential and not tool.hidden:
            names.append(tool.name)
        names.extend(option_names(tool.name))
    return names


def catalog() -> list[dict[str, Any]]:
    """The tools a host dialog can offer to switch off, listed tools only, in
    table order: name, group, kind, whether the tool can be disabled
    (``essential`` tools cannot, their options can), the names of its options
    and what it needs (groups of names, one of each on). Plain data."""
    return [{"name": tool.name, "group": tool.group, "kind": tool.kind,
             "disableable": not tool.essential, "options": list(tool.options),
             "needs": [list(group) for group in tool.needs]}
            for tool in TOOLS.values() if not tool.hidden]


def accepted_names() -> set[str]:
    """Every name :func:`valid_names` lists, and the hidden verbs."""
    return {*valid_names(), *(tool.name for tool in TOOLS.values() if tool.hidden)}


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


def check_names(names: list[Any]) -> str | None:
    """Why *names* hold a name no tool or option has, or a tool that cannot
    be switched off; None when they do not."""
    known = accepted_names()
    for name in names:
        if name in known:
            continue
        held = TOOLS.get(str(name))
        if held is not None and held.essential:
            return (f"disabled: {name} cannot be disabled (the run needs it); the tools and "
                    f"options that can be: {', '.join(valid_names())}")
        return f"disabled: unknown name {name!r}; the valid names are {', '.join(valid_names())}"
    return None


def refusal(spec: Any, *, image_model: bool = True) -> str | None:
    """One line saying why *spec*'s disabled list cannot work, or None when
    it can. The checks, in order: the names; ``look`` keeps a mode; every
    task that is on keeps a tool that writes its answer (without those that
    call the image model when *image_model* is False: a door that cannot
    reach it); a tool that is on keeps what it needs; the viewing gates keep
    the positioning look."""
    disabled = spec.disabled
    if not disabled:
        return None
    reason = check_names(disabled)
    if reason is not None:
        return reason
    if not options_on(disabled, "look"):
        return "disabled: every look mode is off; look keeps at least one of " \
               + ", ".join(option_names("look"))
    for task in spec.tasks:
        writers = TASK_WRITERS.get(task, ())
        if writers and not any(TOOLS[name].is_on(spec)
                               and (image_model or not TOOLS[name].image_model)
                               for name in writers):
            return (f"disabled: the {task} task is on and every tool that writes its answer "
                    f"({', '.join(writers)}) is off here; keep one of them on or leave the "
                    "task out")
    for tool in TOOLS.values():
        if not tool.is_on(spec):
            continue
        for group in tool.needs:
            if all(is_off(disabled, name) for name in group):
                return (f"disabled: {tool.name} is on and needs {_either(list(group))}, "
                        f"which {'is' if len(group) == 1 else 'are'} off; turn "
                        f"{tool.name} off too or keep {_either(list(group))} on")
    if (spec.has("position") and spec.position.gated
            and is_off(disabled, "look:positioning")):
        return ("disabled: position.gated makes submit wait for a look in mode "
                "positioning, which is off; switch the gates off or keep look:positioning on")
    return None
