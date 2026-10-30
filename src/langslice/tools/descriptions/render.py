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
``langslice-job``   ``summary`` (``ops``), :func:`arguments_schema` and
                    ``doc`` (``schema``), ``text`` (``TOOLS.md``),
                    ``arguments`` (the fix of an unknown argument)
reference card      ``summary``
==================  =================================================

ADK and FastMCP send a function's ``__doc__`` as it is, so ``doc`` is laid
out as a docstring: the summary, then every other line indented eight
spaces; ``Args:`` entries four more, their continuation lines eight more.
A run's declaration renders each text for what the job switches off
(:mod:`langslice.job.settings.needs`), fills its slots, leaves out the
arguments the run does not offer and narrows the ones whose values the job
switches off; ``trace_borders`` carries the run's base image prompt last,
as the image model receives it. A tool plugin's tool is declared from its
stub: the docstring indented as above, rendered for what the job switches
off and lacks, and the stub's signature. Door-only parameters (ADK's
``tool_context``) are the tool body's, never declared.
"""

from __future__ import annotations

import functools
import inspect
import typing
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from string import Template
from typing import Any, Literal

from langslice.core.fits.trace.prompts import border_correction_tool_prompt
from langslice.job.settings.needs import is_met, render
from langslice.job.settings.tool_table import OPTION_SEPARATOR, TOOLS, is_off
from langslice.tools.arguments import (
    SectionTransform,
    SectionTransformNoFlip,
    SectionTransformNoOrientation,
    SectionTransformNoRotate,
)
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
    ToolDescription,
)

#: Every tool's description, in the order every door lists the tools.
_FAMILIES: dict[str, ToolDescription] = {
    **looking.DESCRIPTIONS,
    **changing.DESCRIPTIONS,
    **fitting.DESCRIPTIONS,
    **landmarks.DESCRIPTIONS,
    **bookkeeping.DESCRIPTIONS,
}
DESCRIPTIONS: dict[str, ToolDescription] = {
    name: _FAMILIES[name] for name in TOOLS
}

#: The run a caller without a job declares: every argument, LangSlice's
#: base prompt for coronal sections and the OpenAI image models.
FULL = DescriptionFacts(
    prompt=border_correction_tool_prompt("coronal", provider="openai-oauth")
)

#: What every tool returns, as its declaration says: a JSON object.
RETURNS = dict[str, Any]

#: How far ``doc`` indents every line after the summary (see the module).
DOC_INDENT = 8

#: How far an ``Args:`` or ``Returns:`` entry sits under its heading.
ENTRY_INDENT = 4

#: How far a plugin stub's docstring indents its lines: a function body's.
STUB_INDENT = 4


@dataclass(frozen=True)
class Declaration:
    """One tool as a door offers it."""

    name: str
    #: The description as ADK and FastMCP read it (see the module text).
    doc: str
    #: The description below its summary, at no indentation (``TOOLS.md``).
    text: str
    #: Each argument's text on one line.
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


def _native(
    name: str, description: ToolDescription, facts: DescriptionFacts
) -> Declaration:
    offered = {
        argument: entry
        for argument, entry in description.arguments.items()
        if is_offered(entry, facts)
    }
    slots = {
        key: _slot_text(slot, facts) for key, slot in description.slots.items()
    }
    rendered = functools.partial(
        _rendered, slots=slots, switched_off=facts.switched_off
    )

    # the blocks under the summary: paragraphs, arguments, returns, prompt
    blocks = [rendered(paragraph) for paragraph in description.body]
    blocks = [block for block in blocks if block]
    texts = {
        argument: rendered(entry.text) for argument, entry in offered.items()
    }
    if texts:
        blocks.append(_arguments_block(texts))
    if description.returns:
        returns = _indented(rendered(description.returns), ENTRY_INDENT)
        blocks.append(f"Returns:\n{returns}")
    if description.prompt_heading and facts.prompt:
        blocks += [rendered(description.prompt_heading), facts.prompt]
    below = "\n\n".join(blocks)
    summary = rendered(description.summary)
    doc = f"{summary}\n\n{_indented(below, DOC_INDENT)}\n{' ' * DOC_INDENT}"

    annotations: dict[str, object] = {
        argument: narrowed(
            name, argument, entry.annotation, facts.switched_off
        )
        for argument, entry in offered.items()
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


def _rendered(
    text: str, *, slots: Mapping[str, str], switched_off: frozenset[str]
) -> str:
    filled = Template(text).substitute(slots)
    # a passage left out at a line's end leaves no space behind it
    return render(filled, switched_off).rstrip(" ")


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
        True when the run has the argument's fact and every name it needs.
    """
    return _has(facts, argument.offered) and is_met(
        argument.needs, facts.switched_off
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
    stub = plugin_stub(name, facts)
    written = stub.__doc__ or ""
    first, newline, rest = written.partition("\n")
    if newline:  # a stub's docstring sits one level in, the doc's two
        written = f"{first}\n{_indented(rest, DOC_INDENT - STUB_INDENT)}"
    doc = render(written, facts.switched_off | facts.absent)
    _, _, below = render(
        inspect.cleandoc(stub.__doc__ or ""), facts.switched_off | facts.absent
    ).partition("\n")
    return Declaration(
        name,
        doc,
        below.strip(),
        argument_texts(doc),
        inspect.signature(stub, eval_str=True),
        dict(inspect.get_annotations(stub, eval_str=True)),
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
        head = (
            line[12:]
            if line.startswith(" " * 12) and not line.startswith(" " * 13)
            else ""
        )
        key, colon, text = head.partition(":")
        if colon and key.isidentifier():
            name = key
            texts[name] = text.strip()
    return texts


def narrowed(
    name: str, argument: str, annotation: object, off: Collection[str]
) -> object:
    """Return an argument's type without the options the job switches off.

    The options of ``look``'s ``mode`` are values of its enumeration; the
    ``flip`` and ``rotate`` options of ``interactive_transform`` are keys of
    each section it takes.

    Parameters
    ----------
    name
        The tool the argument belongs to.
    argument
        One of the tool's arguments.
    annotation
        The argument's declared type.
    off
        What the job switches off.

    Returns
    -------
    object
        The type with the switched-off values or keys left out.
    """
    options = TOOLS[name].options if name in TOOLS else {}
    if argument in options.values():
        values: tuple[object, ...] = typing.get_args(annotation)
        kept = tuple(
            value
            for value in values
            if not is_off(off, f"{name}{OPTION_SEPARATOR}{value}")
        )
        return Literal[kept]
    if (name, argument) != ("interactive_transform", "sections"):
        return annotation
    no_flip = is_off(off, "interactive_transform:flip")
    no_rotate = is_off(off, "interactive_transform:rotate")
    return _SECTION_TRANSFORMS[no_flip, no_rotate]


#: ``interactive_transform``'s section shape by (flip off, rotate off).
_SECTION_TRANSFORMS: dict[tuple[bool, bool], object] = {
    (False, False): list[SectionTransform],
    (True, False): list[SectionTransformNoFlip],
    (False, True): list[SectionTransformNoRotate],
    (True, True): list[SectionTransformNoOrientation],
}


def plugin_stub(name: str, facts: DescriptionFacts) -> Callable[..., object]:
    """Return the stub of a tool plugin's tool.

    Parameters
    ----------
    name
        The plugin tool.
    facts
        The run, which holds the job's plugins.

    Returns
    -------
    Callable
        The function whose signature and docstring declare the tool.

    Raises
    ------
    KeyError
        When no plugin of the run has the tool.
    """
    for plugin in facts.plugins:
        for tool in plugin.tools:
            if tool.name == name:
                return tool.stub
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
    missing = [
        arg for arg in declared.signature.parameters if arg not in taken
    ]
    if missing:
        raise TypeError(f"{name}: the tool body takes no {', '.join(missing)}")
    signature = declared.signature
    annotations = dict(declared.annotations)
    if "tool_context" in taken:
        signature = signature.replace(
            parameters=[
                *signature.parameters.values(),
                inspect.Parameter(
                    "tool_context",
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=Any,
                ),
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
    tool.__signature__ = signature  # type: ignore[attr-defined]
    tool.__annotations__ = annotations
    return tool


def arguments_schema(
    name: str, facts: DescriptionFacts = FULL
) -> dict[str, Any]:
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
    model = create_model(f"{name}_arguments", __config__=ConfigDict(extra="forbid"),
                         **fields)
    schema = model.model_json_schema()
    schema["title"] = name
    schema["description"] = declared.summary
    return schema
