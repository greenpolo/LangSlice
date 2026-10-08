"""The shapes of the toolbox's arguments, and the one check that refuses strays.

An argument whose value is an object or a list of objects (``sections`` of
``position_sections`` and ``interactive_transform``, ``cutting_angles``,
``submit``'s ``left_linear``) is a typed dict. Typed dicts, not ``dict[str,
Any]``: ADK and FastMCP
both turn them into a JSON schema that names every key and its type, so the
model sees the shape in the tool list, and the keys this module checks are
the keys that schema shows.

:func:`normalize_arguments` is the other shared rule: what the native tools
accept as sent (a number where a filename is asked for, which the tools then
refuse by name, a null picture option) is made schema-valid before a door that
validates against the schema (FastMCP) sees it.

:func:`argument_refusal` is the one strictness rule, used at every door: the
toolbox's own wrapper (``toolbox.build_tools``: direct calls and nested
keys), the ADK plugin (``agent.plugins.StrictArgumentsPlugin``: ADK drops
unknown top-level arguments before a tool runs, so it is checked before
that) and the MCP server (``doors.mcp.server``: FastMCP drops them too). An
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

# --- per-entry arguments ----------------------------------------------------


@with_config(ConfigDict(extra="forbid"))
class SectionPosition(TypedDict, total=False):
    """One ``position_sections`` section: where it sits along the slicing axis."""

    id: str
    position_mm: float


@with_config(ConfigDict(extra="forbid"))
class CuttingAngles(TypedDict, total=False):
    """``position_sections``' cutting angles, for the whole stack."""

    pitch_deg: float
    yaw_deg: float


@with_config(ConfigDict(extra="forbid"))
class SectionTransform(TypedDict, total=False):
    """One ``interactive_transform`` section: absolute values, each optional."""

    id: str
    flip: bool
    rotate_quarter: int
    rotation_deg: float
    scale_x: float
    scale_y: float
    shear: float
    translate_x_mm: float
    translate_y_mm: float


@with_config(ConfigDict(extra="forbid"))
class LeftLinear(TypedDict, total=False):
    """One ``submit`` ``left_linear`` entry: a section left without a
    deformation, and why."""

    id: str
    reason: str


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
    if not top_level and key in params:
        return f"`{key}` is a top-level argument of this tool, not a key here."
    for name, annotation in params.items():
        found = _typed_dict(annotation)
        if found is None or found[0] is typed:
            continue
        if key in _keys(found[0]):
            inside = f"each `{name}` entry" if found[1] else f"`{name}`"
            return f"`{key}` belongs inside {inside}."
    return ""


def _is_index(value: Any) -> bool:
    """A whole number sent where a section reference (text) is typed."""
    return isinstance(value, int) and not isinstance(value, bool)


def _text_typed(annotation: Any) -> bool:
    return annotation is str


def _text_list_typed(annotation: Any) -> bool:
    return (typing.get_origin(annotation) in (list, tuple)
            and typing.get_args(annotation)[:1] == (str,))


def _without_nulls(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: item for key, item in value.items() if item is not None}
    return value


def normalize_arguments(func: Callable[..., Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """*args* as the native tools read them, in a shape *func*'s schema accepts.

    The tools take a section by filename, and ADK hands them a number as
    sent; a door that validates against the schema first (FastMCP's pydantic
    check) would refuse a number where the schema says text with no word on
    how sections are named. So a whole number in a text argument (``id``,
    ``section``) or in a list of text (``slices``) becomes its text, and the
    tool's own refusal (``UNKNOWN_SLICE_IDS``) says that sections are named
    by filename and lists them. A null inside a typed-dict
    argument (``cutting_angles``, each ``sections`` dict) means "not given",
    as the tools read it, and is dropped; a null object argument is its
    default. Nothing else changes: unknown keys are :func:`argument_refusal`'s.
    """
    params = parameters(func)
    out = dict(args)
    for name, annotation in params.items():
        if name not in out:
            continue
        value = out[name]
        found = _typed_dict(annotation)
        if found is not None:
            many = found[1]
            if value is None and not many:
                del out[name]
            elif many and isinstance(value, (list, tuple)):
                out[name] = [_without_nulls(item) for item in value]
            else:
                out[name] = _without_nulls(value)
        elif _text_typed(annotation) and _is_index(value):
            out[name] = str(value)
        elif _text_list_typed(annotation) and isinstance(value, (list, tuple)):
            out[name] = [str(item) if _is_index(item) else item for item in value]
    return out


def argument_refusal(func: Callable[..., Any], args: Mapping[str, Any]) -> dict[str, Any] | None:
    """The refusal for unknown or misplaced arguments, or None when all are known.

    Checks the top-level names against *func*'s signature, and every key of
    an argument typed as a typed dict (``cutting_angles``) or a list of them
    (``sections``, ``left_linear``) against that dict's keys. A value of the
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
