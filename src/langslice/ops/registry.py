"""Every tool and the operation it calls: the one list the doors are built from.

One :class:`Verb` per agent tool (:func:`langslice.linear.toolbox.build_tools`
returns exactly these names, checked by ``tests/test_ops_registry.py``): the
operation the tool body calls, whether it writes, and the task group a host
shows it under. ``Common`` tools are on in every run whatever the tasks
(``mark_damaged`` and ``preprocess`` behind their own host switches); the
others come with their task, Positioning (``reorder`` + ``position``),
Linear (``transform``) or Nonlinear (``nonlinear``), as
``docs/interface_design.md`` names them. A tool that also exists without its
task (``view_placement`` with Linear or Nonlinear alone) is listed under the
group that introduces it.

Phase 5 of the layered refactor generates the ADK, MCP and CLI doors from
this list; today the tool door is still written by hand and only checked
against it.
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


def _verbs(*verbs: Verb) -> dict[str, Verb]:
    return {verb.name: verb for verb in verbs}


#: Every tool, in the order :func:`~langslice.linear.toolbox.build_tools` adds them.
VERBS: dict[str, Verb] = _verbs(
    Verb("status", views.status, "read", "Common"),
    Verb("view_slices", views.view_slices, "read", "Common"),
    Verb("view_atlas", views.view_atlas, "read", "Common"),
    Verb("note", notes.add_note, "write", "Common"),
    Verb("undo", history.undo, "write", "Common"),
    Verb("redo", history.redo, "write", "Common"),
    Verb("mark_damaged", damage.mark_damaged, "write", "Common"),
    Verb("preprocess", appearance.preprocess, "write", "Common"),
    Verb("reorder_slices", order.reorder, "write", "Positioning"),
    Verb("set_positions", positions.set_positions, "write", "Positioning"),
    Verb("view_placement", views.view_placement, "read", "Positioning"),
    Verb("view_stack", views.view_stack, "read", "Positioning"),
    Verb("run_deepslice", positions.run_deepslice, "write", "Positioning"),
    Verb("search_position", positions.search_position, "read", "Positioning"),
    Verb("orient_slices", orientation.orient_sections, "write", "Linear"),
    Verb("fit_affine", transforms.fit_affine, "write", "Linear"),
    Verb("adjust_transforms", transforms.adjust_transforms, "write", "Linear"),
    Verb("set_cutting_angles", positions.set_cutting_angles, "write", "Linear"),
    Verb("trace_borders", traces.trace_borders, "write", "Nonlinear"),
    Verb("grep_atlas", atlas.grep_atlas, "read", "Nonlinear"),
    Verb("fit_deformable", deformable.fit_deformable, "write", "Nonlinear",
         alternates={"keep_linear": deformable.keep_linear}),
    Verb("submit", submit.submit, "write", "Common"),
)


def table() -> list[dict[str, str]]:
    """The registry as plain rows: tool, operation, kind, group."""
    return [
        {"tool": verb.name, "operation": f"{verb.function.__module__}.{verb.function.__name__}",
         "kind": verb.kind, "group": verb.group}
        for verb in VERBS.values()
    ]
