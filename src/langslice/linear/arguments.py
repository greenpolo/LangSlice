"""The shapes of the toolbox's arguments, and the one check that refuses strays.

Every picture tool takes its picture options in ONE argument, ``view``
(:class:`View`; :class:`ViewAuto` adds ``resolution`` where the user left
picture size to the agent). Per-entry arguments (``entries``, ``candidates``)
are typed dicts too. Typed dicts, not ``dict[str, Any]``: ADK and FastMCP
both turn them into a JSON schema that names every key and its type, so the
model sees the shape in the tool list, and the keys this module checks are
the keys that schema shows.

:func:`argument_refusal` is the one strictness rule, used at every door: the
toolbox's own wrapper (``toolbox.build_tools``: direct calls and nested
keys), the ADK plugin (``adk.plugins.StrictArgumentsPlugin``: ADK drops
unknown top-level arguments before a tool runs, so it is checked before
that) and the MCP server (``mcp_server.server``: FastMCP drops them too). An
unknown or misplaced key is refused with the key named and the accepted keys
listed; nothing is silently dropped.
"""

from __future__ import annotations

import inspect
import types
import typing
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ConfigDict, with_config
from typing_extensions import TypedDict, is_typeddict

# --- picture options --------------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
class View(TypedDict, total=False):
    """Picture options: what this call's pictures show (this call only)."""

    mode: str
    channels: list[str]
    atlas_channels: list[str]
    atlas_opacity: float
    regions: list[str]
    outlines: str
    border_color: str
    border_thickness: float
    zoom: list[float]
    deformation: str


@with_config(ConfigDict(extra="forbid"))
class ViewAuto(View, total=False):
    """Picture options, with the picture size (image resolution "auto")."""

    resolution: int


# --- per-entry arguments ----------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
class DamageEntry(TypedDict, total=False):
    """One ``mark_damaged`` entry."""

    id: str | int
    damaged: bool
    note: str


@with_config(ConfigDict(extra="forbid"))
class OrientEntry(TypedDict, total=False):
    """One ``orient_slices`` entry."""

    id: str | int
    flip: bool
    rotate_deg: int


@with_config(ConfigDict(extra="forbid"))
class PositionEntry(TypedDict, total=False):
    """One ``set_positions`` entry."""

    id: str | int
    position_mm: float


@with_config(ConfigDict(extra="forbid"))
class PlacementEntry(TypedDict, total=False):
    """One ``view_placement`` entry."""

    id: str | int
    positions_mm: list[float]


@with_config(ConfigDict(extra="forbid"))
class TransformEntry(TypedDict, total=False):
    """One ``adjust_transforms`` entry: the knobs (shear optional), the pivot and a note."""

    id: str | int
    rotation_deg: float
    scale_x: float
    scale_y: float
    translate_x_mm: float
    translate_y_mm: float
    shear: float
    pivot: str | list[float]
    note: str


@with_config(ConfigDict(extra="forbid"))
class Candidate(TypedDict, total=False):
    """One ``fit_deformable`` candidate: the settings it overrides."""

    stiffness: str
    fit_section: str
    fit_atlas: str
    engine: str


@with_config(ConfigDict(extra="forbid"))
class FixedCandidate(TypedDict, total=False):
    """A candidate where the user fixed the engine (no ``engine`` key)."""

    stiffness: str
    fit_section: str
    fit_atlas: str


#: Why a key a typed dict does not have is refused, where the plain "unknown"
#: would mislead.
KEY_NOTES: dict[Any, dict[str, str]] = {
    View: {
        "resolution": "The user fixed the picture size for this run, so resolution "
        "cannot be set.",
    },
    FixedCandidate: {
        "engine": "The user fixed the engine for this run.",
    },
}


# --- the check --------------------------------------------------------------


def _typed_dict(annotation: Any) -> tuple[Any, bool] | None:
    """``(typed dict, is a list of them)`` for an annotation, else None."""
    if is_typeddict(annotation):
        return annotation, False
    origin = typing.get_origin(annotation)
    if origin in (list, tuple):
        args = typing.get_args(annotation)
        if args and is_typeddict(args[0]):
            return args[0], True
    if origin is typing.Union or origin is types.UnionType:
        for arg in typing.get_args(annotation):
            found = _typed_dict(arg)
            if found is not None:
                return found
    return None


def issubclass_typed(typed: Any, base: Any) -> bool:
    """Whether typed dict *typed* is *base* or extends it."""
    return typed is base or base in getattr(typed, "__orig_bases__", ())


def _keys(typed: Any) -> list[str]:
    return list(typing.get_type_hints(typed))


def parameters(func: Callable[..., Any]) -> dict[str, Any]:
    """The tool's arguments as the model sees them: ``{name: annotation}``.

    ADK's ``tool_context`` is the framework's, never the model's.
    """
    signature = inspect.signature(func, eval_str=True)
    return {name: parameter.annotation for name, parameter in signature.parameters.items()
            if name != "tool_context"
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)}


def _note(key: str, typed: Any, params: dict[str, Any], *, top_level: bool) -> str:
    """Where a stray key belongs, when it belongs somewhere else."""
    explained = KEY_NOTES.get(typed, {}).get(key) if typed is not None else None
    if explained:
        return explained
    if not top_level and key in params:
        return f"`{key}` is a top-level argument of this tool, not a key here."
    for name, annotation in params.items():
        found = _typed_dict(annotation)
        if found is None or found[0] is typed:
            continue
        if key in _keys(found[0]):
            inside = f"each `{name}` entry" if found[1] else f"`{name}`"
            return f"`{key}` belongs inside {inside}."
        for kind, notes in KEY_NOTES.items():
            if key in notes and issubclass_typed(found[0], kind):
                return notes[key]
    return ""


def argument_refusal(func: Callable[..., Any], args: Mapping[str, Any]) -> dict[str, Any] | None:
    """The refusal for unknown or misplaced arguments, or None when all are known.

    Checks the top-level names against *func*'s signature, and every key of
    an argument typed as a typed dict (``view``) or a list of them
    (``entries``, ``candidates``) against that dict's keys. A value of the
    wrong kind (a string where an object belongs) is left to the tool.
    """
    params = parameters(func)
    problems: list[dict[str, Any]] = []
    stray = [key for key in args if key not in params and key != "tool_context"]
    if stray:
        notes = [text for key in stray if (text := _note(key, None, params, top_level=True))]
        problems.append({"argument": "(top level)", "unknown": stray,
                         "accepted": list(params), **({"notes": notes} if notes else {})})
    for name, annotation in params.items():
        if name not in args:
            continue
        found = _typed_dict(annotation)
        if found is None:
            continue
        typed, many = found
        accepted = _keys(typed)
        values = args[name]
        items = (list(enumerate(values)) if many and isinstance(values, (list, tuple))
                 else [(None, values)] if not many else [])
        for index, value in items:
            if not isinstance(value, Mapping):
                continue
            unknown = [key for key in value if key not in accepted]
            if not unknown:
                continue
            notes = [text for key in unknown
                     if (text := _note(key, typed, params, top_level=False))]
            problems.append({
                "argument": name if index is None else f"{name}[{index}]",
                "unknown": unknown, "accepted": accepted,
                **({"notes": notes} if notes else {}),
            })
    if not problems:
        return None
    name = getattr(func, "__name__", "this tool")
    message = " ".join(
        f"Unknown key(s) in {problem['argument']}: {', '.join(problem['unknown'])}; "
        f"accepted: {', '.join(problem['accepted'])}."
        + ("".join(f" {text}" for text in problem.get("notes", [])))
        for problem in problems
    )
    return {"status": "error", "error": "UNKNOWN_ARGUMENTS", "tool": name,
            "problems": problems, "message": f"{name}: {message} Nothing was done."}
