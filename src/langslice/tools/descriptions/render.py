"""Every tool's declaration for a run: the one place a description is laid out.

Each tool is declared once, from its description (a
:class:`~langslice.tools.descriptions.description.ToolDescription` in its
family's module) and the facts of the run
(:class:`~langslice.tools.descriptions.description.DescriptionFacts`). Every
door is built from the declaration:

==================  =================================================
door                reads
==================  =================================================
agent tools, MCP    ``doc`` and ``signature``, through :func:`declare`
``langslice job``   ``summary`` (``ops``), :func:`arguments_schema` and
                    ``doc`` (``schema``), ``text`` (``TOOLS.md``),
                    ``arguments`` (the fix of an unknown argument)
reference card      ``summary``
==================  =================================================

ADK and FastMCP send a function's ``__doc__`` as it is, so ``doc`` is laid
out as a docstring: the summary, then every other line indented eight
spaces; ``Args:`` entries four more, their continuation lines eight more.
A run's declaration words each text for what the job switches off (a
text that is a function of the run's
:class:`~langslice.job.settings.tools_on.ToolsOn` is called with it), fills
its slots, leaves out the arguments the run does not offer and narrows the
ones whose values the job switches off; ``trace_borders`` carries the run's
base image prompt last, as the image model receives it. A tool plugin's tool
is declared from its stub's signature and its description: the spec's
``description`` (a string, or a function of ``ToolsOn``), else the stub's
docstring, indented as above. Door-only parameters (ADK's
``tool_context``) are the tool body's, never declared.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from string import Template
from typing import Any

from langslice.core.fits.trace.prompts import border_correction_tool_prompt
from langslice.core.plugins import ToolSpec
from langslice.job import plugins
from langslice.job.settings.tool_table import TOOLS
from langslice.job.settings.tools_on import ToolsOn
from langslice.tools.descriptions import (
    bookkeeping,
    changing,
    fitting,
    landmarks,
    looking,
)
from langslice.tools.descriptions.description import (
    CLI_DOOR,
    Argument,
    DescriptionFacts,
    Distance,
    Offered,
    PerCap,
    PerDoor,
    Slot,
    Text,
    ToolDescription,
)

_FAMILIES: dict[str, ToolDescription] = {
    **looking.DESCRIPTIONS,
    **changing.DESCRIPTIONS,
    **fitting.DESCRIPTIONS,
    **landmarks.DESCRIPTIONS,
    **bookkeeping.DESCRIPTIONS,
}
DESCRIPTIONS: dict[str, ToolDescription] = {
    name: _FAMILIES[name] for name, tool in TOOLS.items() if not tool.engine
}

FULL = DescriptionFacts(prompt=border_correction_tool_prompt("coronal", provider="openai-oauth"))

RETURNS = dict[str, Any]

DOC_INDENT = 8

ENTRY_INDENT = 4

STUB_INDENT = 4


@dataclass(frozen=True)
class Declaration:
    """One tool as a door offers it."""

    name: str
    doc: str
    text: str
    arguments: Mapping[str, str]
    signature: inspect.Signature
    annotations: dict[str, object]

    @property
    def summary(self) -> str:
        """The description's first line."""
        return self.doc.split("\n", 1)[0].strip()


@functools.cache
def declaration(name: str, facts: DescriptionFacts = FULL) -> Declaration:
    """Return tool ``name`` as a run of ``facts`` declares it.

    Parameters
    ----------
    name
        A native tool, or a tool of a plugin ``facts`` holds.
    facts
        The run the tool is declared for.

    Returns
    -------
    Declaration
        Its description, argument texts and signature.
    """
    if name in DESCRIPTIONS:
        return _native(name, DESCRIPTIONS[name], facts)
    return _plugin(name, facts)


def _native(name: str, description: ToolDescription, facts: DescriptionFacts) -> Declaration:
    offered = {
        argument: entry
        for argument, entry in description.arguments.items()
        if is_offered(entry, facts)
    }
    slots = {key: _slot_text(slot, facts) for key, slot in description.slots.items()}
    on = facts.on

    blocks = [_worded(paragraph, on, slots) for paragraph in description.body]
    blocks = [block for block in blocks if block]
    texts = {argument: _worded(entry.text, on, slots) for argument, entry in offered.items()}
    if texts:
        blocks.append(_arguments_block(texts))
    returns = _worded(description.returns, on, slots)
    if returns:
        blocks.append(f"Returns:\n{_indented(returns, ENTRY_INDENT)}")
    if description.prompt_heading and facts.prompt:
        blocks += [_worded(description.prompt_heading, on, slots), facts.prompt]
    below = "\n\n".join(blocks)
    summary = _worded(description.summary, on, slots)
    doc = f"{summary}\n\n{_indented(below, DOC_INDENT)}\n{' ' * DOC_INDENT}"

    annotations: dict[str, object] = {
        argument: entry.annotation for argument, entry in offered.items()
    }
    parameters = [
        inspect.Parameter(
            argument,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=entry.default,
            annotation=annotations[argument],
        )
        for argument, entry in offered.items()
    ]
    annotations["return"] = RETURNS
    return Declaration(
        name,
        doc,
        below.strip(),
        {argument: _one_line(text) for argument, text in texts.items()},
        inspect.Signature(parameters, return_annotation=RETURNS),
        annotations,
    )


def _worded(text: Text, on: ToolsOn, slots: Mapping[str, str]) -> str:
    written = text(on) if callable(text) else text
    return Template(written).substitute(slots).rstrip(" ")


def is_offered(argument: Argument, facts: DescriptionFacts) -> bool:
    """Return whether a run of ``facts`` offers ``argument``.

    Parameters
    ----------
    argument
        One argument of a tool's description.
    facts
        The run the tool is declared for.

    Returns
    -------
    bool
        True when the run has the argument's fact and the tool it requires.
    """
    return _has(facts, argument.offered) and (
        not argument.requires or facts.on.has(argument.requires)
    )


def _has(facts: DescriptionFacts, offered: Offered) -> bool:
    match offered:
        case Offered.ALWAYS:
            return True
        case Offered.SIZE_CHOSEN:
            return facts.auto
        case Offered.VIEW_CHOSEN:
            return not facts.forced_view
        case Offered.POSITIONING:
            return facts.positions
        case Offered.UNCAPPED:
            return facts.uncapped


def _slot_text(slot: Slot, facts: DescriptionFacts) -> str:
    match slot:
        case PerCap():
            return slot.uncapped if facts.uncapped else slot.capped
        case PerDoor():
            return slot.cli if facts.door == CLI_DOOR else slot.agent
        case Distance():
            if slot.mouse_words and facts.brain_scale == 1.0:
                return slot.mouse_words
            return slot.words.format(f"{slot.um * facts.brain_scale:.0f}")


def _arguments_block(texts: Mapping[str, str]) -> str:
    entries: list[str] = []
    for argument, text in texts.items():
        first, _, rest = text.partition("\n")
        entry = f"{argument}: {first}"
        if rest:
            entry += "\n" + _indented(rest, ENTRY_INDENT)
        entries.append(entry)
    return "Args:\n" + _indented("\n".join(entries), ENTRY_INDENT)


def _indented(text: str, width: int) -> str:
    pad = " " * width
    lines: list[str] = []
    for line in text.split("\n"):
        if line:
            line = pad + line
        lines.append(line)
    return "\n".join(lines)


def _one_line(text: str) -> str:
    return " ".join(line.strip() for line in text.split("\n"))


def _plugin(name: str, facts: DescriptionFacts) -> Declaration:
    tool = plugin_tool(name, facts)
    stub = tool.stub
    if tool.description:
        written = tool.description(facts.on) if callable(tool.description) else tool.description
        first, newline, rest = written.strip().partition("\n")
        doc = first
        if newline:
            doc = f"{first}\n{_indented(rest, DOC_INDENT)}\n{' ' * DOC_INDENT}"
        below = rest
    else:
        doc = stub.__doc__ or ""
        first, newline, rest = doc.partition("\n")
        if newline:
            doc = f"{first}\n{_indented(rest, DOC_INDENT - STUB_INDENT)}"
        _, _, below = inspect.cleandoc(stub.__doc__ or "").partition("\n")
    annotations = dict(inspect.get_annotations(stub, eval_str=True))
    annotations["return"] = RETURNS
    signature = inspect.signature(stub, eval_str=True)
    return Declaration(
        name,
        doc,
        below.strip(),
        argument_texts(doc),
        signature.replace(return_annotation=RETURNS),
        annotations,
    )


def argument_texts(doc: str) -> dict[str, str]:
    """Return each ``Args`` entry of a plugin tool's declared ``doc``.

    A plugin's stub writes its arguments as a Google-style docstring does;
    declared, each entry starts twelve spaces in and continues sixteen.

    Parameters
    ----------
    doc
        The plugin tool's declared description.

    Returns
    -------
    dict of str to str
        Each argument's text on one line, by name.
    """
    texts: dict[str, str] = {}
    name = None
    for line in doc.split("\n"):
        if line.startswith(" " * 16) and name is not None:
            texts[name] += " " + line.strip()
            continue
        name = None
        head = line[12:] if line.startswith(" " * 12) and not line.startswith(" " * 13) else ""
        key, colon, text = head.partition(":")
        if colon and key.isidentifier():
            name = key
            texts[name] = text.strip()
    return texts


def plugin_tool(name: str, facts: DescriptionFacts) -> ToolSpec:
    """Return the spec of a tool plugin's tool.

    Parameters
    ----------
    name
        The plugin tool.
    facts
        The run, which holds the job's plugins.

    Returns
    -------
    ToolSpec
        The tool, whose ``description`` (else its stub's docstring) and
        stub's signature declare it.

    Raises
    ------
    KeyError
        When no plugin of the run has the tool.
    """
    for plugin in facts.plugins:
        for tool in plugin.tools:
            if tool.name == name:
                return tool
    if name in TOOLS and TOOLS[name].engine:
        return plugins.tool_specs(None)[name][1]
    raise KeyError(name)


def summary(name: str, facts: DescriptionFacts = FULL) -> str:
    """Return a tool's one-line description.

    Parameters
    ----------
    name
        A native tool, or a tool of a plugin ``facts`` holds.
    facts
        The run the tool is declared for.

    Returns
    -------
    str
        The first line of its description.
    """
    return declaration(name, facts).summary


def context_parameter(
    signature: inspect.Signature,
    *,
    annotation: object = inspect.Parameter.empty,
) -> inspect.Parameter:
    """Make ADK's ``tool_context`` parameter to end a declared signature.

    It is keyword-only after a keyword-only argument (a plugin stub's
    options after ``*``), and positional-or-keyword otherwise.

    Parameters
    ----------
    signature
        The declared signature it is appended to.
    annotation
        The parameter's annotation; none by default.

    Returns
    -------
    inspect.Parameter
        ``tool_context``, defaulting to None.
    """
    kinds = [parameter.kind for parameter in signature.parameters.values()]
    kind = inspect.Parameter.POSITIONAL_OR_KEYWORD
    if inspect.Parameter.KEYWORD_ONLY in kinds:
        kind = inspect.Parameter.KEYWORD_ONLY
    return inspect.Parameter("tool_context", kind, default=None, annotation=annotation)


def declare(
    name: str, body: Callable[..., Any], facts: DescriptionFacts = FULL
) -> Callable[..., Any]:
    """Return a tool body as the tool is declared in a run.

    The returned function carries the declaration's name, description and
    signature (what ADK and FastMCP read); a call binds its arguments to that
    signature, fills the declared defaults and hands every argument to
    ``body`` by name. A parameter of ``body`` that is not declared (a
    door-only one, or one the run does not offer) keeps ``body``'s default,
    except ADK's ``tool_context``, which is added to the signature so ADK
    passes it.

    Parameters
    ----------
    name
        The tool to declare.
    body
        The tool's body; it takes every declared argument by name.
    facts
        The run the tool is declared for.

    Returns
    -------
    Callable
        The declared tool.

    Raises
    ------
    TypeError
        When ``body`` lacks a declared argument.
    """
    declared = declaration(name, facts)
    taken = inspect.signature(body).parameters
    missing = [arg for arg in declared.signature.parameters if arg not in taken]
    if missing:
        raise TypeError(f"{name}: the tool body takes no {', '.join(missing)}")
    signature = declared.signature
    annotations = dict(declared.annotations)
    if "tool_context" in taken:
        signature = signature.replace(
            parameters=[
                *signature.parameters.values(),
                context_parameter(signature, annotation=Any),
            ]
        )
        annotations = {
            **{k: v for k, v in annotations.items() if k != "return"},
            "tool_context": Any,
            "return": annotations.get("return"),
        }

    def tool(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        return body(**bound.arguments)

    tool.__name__ = tool.__qualname__ = name
    tool.__doc__ = declared.doc
    tool.__annotations__ = annotations
    return signed(tool, signature)


def signed(function: Callable[..., Any], signature: inspect.Signature) -> Callable[..., Any]:
    """Give *function* the signature its callers and ``inspect`` are shown.

    ``inspect.signature`` reads a function's ``__signature__`` first, which
    is how a body taking ``*args, **kwargs`` is declared with named arguments.
    """
    function.__dict__["__signature__"] = signature
    return function


def arguments_schema(name: str, facts: DescriptionFacts = FULL) -> dict[str, Any]:
    """Return the JSON schema of a tool's arguments.

    It is built by pydantic from the declaration, the way FastMCP builds its
    input schemas: an object whose unknown keys are refused, as every door
    refuses them.

    Parameters
    ----------
    name
        A native tool, or a tool of a plugin ``facts`` holds.
    facts
        The run the tool is declared for.

    Returns
    -------
    dict
        The schema, titled with the tool's name and its one-line description.
    """
    from pydantic import ConfigDict, create_model

    declared = declaration(name, facts)
    fields: dict[str, Any] = {}
    for parameter in declared.signature.parameters.values():
        default = ... if parameter.default is inspect.Parameter.empty else parameter.default
        fields[parameter.name] = (parameter.annotation, default)
    model = create_model(f"{name}_arguments", __config__=ConfigDict(extra="forbid"), **fields)
    schema = model.model_json_schema()
    schema["title"] = name
    schema["description"] = declared.summary
    return schema
