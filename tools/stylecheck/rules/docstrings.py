"""Checks on docstrings that the numpydoc validator does not make.

The numpydoc validator owns presence, the agreement of ``Parameters`` and
``Returns`` with the signature, and the form of the summary line. These
checks add what it cannot judge: how much a parameter's description says,
the summary's first word, how many other symbols a docstring points at,
and whether a lower layer names a higher one.

A docstring is split into numpydoc *sections* at each title underlined
with dashes. A *reference* is a Sphinx role such as ``:func:`name``` or a
dotted name written in double backticks, such as ````np.ndarray````; a
file name (``state.json``) is not a reference.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from dataclasses import dataclass

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    module_check,
)
from tools.stylecheck.rules.enclosing import is_property
from tools.stylecheck.rules.scopes import FUNCTION_NODES, parameters

#: A Sphinx cross-reference role and its target.
ROLE_PATTERN = re.compile(
    r":(?:py:)?(?:func|class|meth|mod|attr|data|obj|exc|const)"
    r":`~?([^`]+)`"
)

#: Text in double backticks.
LITERAL_PATTERN = re.compile(r"``([^`]+)``")

#: A dotted name, possibly called: ``np.ndarray``, ``Job.save()``.
DOTTED_PATTERN = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+(?:\(\))?")

#: File suffixes that make a dotted word a file name, not a reference.
FILE_SUFFIXES = frozenset(
    {
        "py",
        "json",
        "jsonl",
        "toml",
        "md",
        "txt",
        "yaml",
        "yml",
        "csv",
        "tsv",
        "png",
        "jpg",
        "jpeg",
        "tif",
        "tiff",
        "zarr",
        "nii",
        "gz",
        "html",
        "npz",
        "npy",
        "h5",
        "xml",
        "cfg",
        "ini",
        "log",
        "zip",
        "svg",
        "pdf",
        "ome",
        "nrrd",
        "mhd",
        "lock",
        "sh",
        "java",
        "jar",
    }
)

#: Words that carry no meaning of their own in a parameter description.
FILLER_WORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "be",
        "this",
        "that",
        "its",
        "it",
        "given",
        "value",
        "values",
        "parameter",
        "argument",
        "object",
        "instance",
        "input",
        "optional",
        "default",
        "none",
    }
)

#: The sections whose entries are parameters.
PARAMETER_SECTIONS = ("Parameters", "Other Parameters")

#: The one section where references are not counted.
SEE_ALSO = "See Also"

#: What a word is, for counting the words of a description.
WORD_PATTERN = re.compile(r"[A-Za-z0-9]+")

DocumentedNode = (
    ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
)


@dataclass(frozen=True)
class Docstring:
    """A docstring with the definition it belongs to.

    Parameters
    ----------
    owner
        The module, class or function the docstring documents.
    text
        The docstring, cleaned of its indentation.
    line
        The 1-based line where the docstring starts in the file.
    """

    owner: DocumentedNode
    text: str
    line: int


def docstrings(tree: ast.Module) -> Iterator[Docstring]:
    """Yield the docstring of the module and of every class and function.

    Parameters
    ----------
    tree
        The parsed module whose definitions are read.

    Yields
    ------
    Docstring
        Each docstring present, with its owner and first line.
    """
    owners = (ast.Module, ast.ClassDef, *FUNCTION_NODES)
    for node in ast.walk(tree):
        if not isinstance(node, owners):
            continue
        text = ast.get_docstring(node)
        if text is not None:
            yield Docstring(node, text, node.body[0].lineno)


def split_sections(text: str) -> dict[str, list[str]]:
    """Split a numpydoc docstring into its sections.

    Parameters
    ----------
    text
        A cleaned docstring.

    Returns
    -------
    dict of str to list of str
        Each section title with the lines under its underline; the lines
        before the first title are under ``""``.
    """
    lines = text.splitlines()
    sections: dict[str, list[str]] = {"": []}
    title = ""
    index = 0
    while index < len(lines):
        if _is_title(lines, index):
            title = lines[index].strip()
            sections[title] = []
            index += 2
            continue
        sections[title].append(lines[index])
        index += 1
    return sections


def _is_title(lines: list[str], index: int) -> bool:
    """Tell whether a line is a section title underlined by the next."""
    if index + 1 >= len(lines) or not lines[index].strip():
        return False
    underline = lines[index + 1].strip()
    return len(underline) >= 3 and set(underline) == {"-"}


def parameter_entries(lines: list[str]) -> list[tuple[list[str], str]]:
    """Read the entries of a ``Parameters`` section.

    Parameters
    ----------
    lines
        The section's lines, as :func:`split_sections` returns them.

    Returns
    -------
    list of tuple
        For each entry, the parameter names it documents (``x, y : float``
        documents two) and its description as one string.
    """
    entries: list[tuple[list[str], list[str]]] = []
    for line in lines:
        if line and not line[0].isspace():
            names = line.split(" : ")[0].rstrip(":")
            entries.append(([_bare(name) for name in names.split(",")], []))
        elif entries:
            entries[-1][1].append(line.strip())
    return [(names, " ".join(text).strip()) for names, text in entries]


def _bare(name: str) -> str:
    """Strip stars and spaces from a documented parameter name."""
    return name.strip().lstrip("*")


@module_check("parameter-description")
def check_parameter_description(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a parameter description that is too short or only its name.

    Guide rule 6.1. Each entry of ``Parameters`` and ``Other Parameters``
    needs at least ``min_parameter_words`` words, and at least one of
    them must be neither a filler word (``the``, ``value``, ...) nor a
    word of the parameter's own name, so ``mode : str. The mode value.``
    fails. An empty description is left to numpydoc.

    Parameters
    ----------
    module
        The parsed file whose docstrings are read.
    project
        The package index, unused by this check.
    config
        The settings that give the minimum word count.

    Yields
    ------
    Violation
        One finding per weak description.
    """
    for docstring in docstrings(module.tree):
        sections = split_sections(docstring.text)
        for title in PARAMETER_SECTIONS:
            entries = parameter_entries(sections.get(title, []))
            for names, description in entries:
                problem = _description_problem(names, description, config)
                if problem:
                    yield module.violation(
                        "parameter-description",
                        docstring.line,
                        f"Parameter {', '.join(names)!r} {problem} (6.1).",
                    )


def _description_problem(
    names: list[str], description: str, config: Config
) -> str:
    """Say what is wrong with a description, or return ``""``."""
    matches: list[str] = WORD_PATTERN.findall(description)
    words = [word.lower() for word in matches]
    if not words:
        return ""

    limit = config.min_parameter_words
    if len(words) < limit:
        return (
            f"has a description of {len(words)} words; write at least "
            f"{limit} that say more than its name"
        )

    own: set[str] = set()
    for name in names:
        own.update(_stem(part) for part in _parts(name))
    meaningful = {_stem(word) for word in words} - own - FILLER_WORDS
    if not meaningful:
        return "is described only by its own name; say what it is"
    return ""


def _parts(name: str) -> list[str]:
    """Split a parameter name into its lower-cased words."""
    return [part for part in name.lower().split("_") if part]


def _stem(word: str) -> str:
    """Drop a plural ``s`` so that ``sections`` matches ``section``."""
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


@module_check("summary-first-word")
def check_summary_first_word(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a function summary that opens with a banned word.

    Guide rule 6.2. A function's summary is an imperative verb phrase;
    one that opens with a word in ``banned_summary_words`` (``The``,
    ``A``, ``Whether``, ...) is a noun phrase or a question. The words
    are matched without regard to case. Class, module and property
    summaries are noun phrases and are not checked.

    Parameters
    ----------
    module
        The parsed file whose docstrings are read.
    project
        The package index, unused by this check.
    config
        The settings that list the banned first words.

    Yields
    ------
    Violation
        One finding per function whose summary opens with such a word.
    """
    banned = {word.casefold() for word in config.banned_summary_words}
    for docstring in docstrings(module.tree):
        owner = docstring.owner
        if not isinstance(owner, FUNCTION_NODES) or is_property(owner):
            continue
        first = _first_word(docstring.text)
        if first.casefold() in banned:
            yield module.violation(
                "summary-first-word",
                docstring.line,
                f"The summary opens with {first!r}; open it with an "
                "imperative verb saying what the function does (6.2).",
            )


def _first_word(text: str) -> str:
    """Return the first word of a docstring's first line."""
    match = WORD_PATTERN.search(text.strip().split("\n")[0])
    return match.group(0) if match else ""


@module_check("docstring-references")
def check_docstring_references(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a docstring that points at too many other symbols.

    Guide rule 6.4. Counts the distinct references (roles such as
    ``:func:`` and dotted names in double backticks) outside the
    ``See Also`` section. A dotted name that starts with one of the
    function's own parameters, or with ``self``, describes the function's
    own inputs and does not count; neither do file names.

    Parameters
    ----------
    module
        The parsed file whose docstrings are read.
    project
        The package index, unused by this check.
    config
        The settings that give the most references allowed.

    Yields
    ------
    Violation
        One finding per docstring over the limit.
    """
    limit = config.max_docstring_references
    for docstring in docstrings(module.tree):
        references = _references(docstring)
        if len(references) > limit:
            yield module.violation(
                "docstring-references",
                docstring.line,
                f"The docstring references {len(references)} other "
                f"symbols outside See Also; keep it to {limit} (6.4).",
            )


def _references(docstring: Docstring) -> set[str]:
    """Return the distinct symbols a docstring references outside See Also."""
    sections = split_sections(docstring.text)
    sections.pop(SEE_ALSO, None)
    body = "\n".join("\n".join(lines) for lines in sections.values())
    own = _own_names(docstring.owner)
    roles: list[str] = ROLE_PATTERN.findall(body)
    literals: list[str] = LITERAL_PATTERN.findall(body)
    references = set(roles)
    for literal in literals:
        if _is_reference(literal.strip(), own):
            references.add(literal.strip())
    return references


def _own_names(owner: DocumentedNode) -> set[str]:
    """Return the names a docstring may use freely: its parameters."""
    own = {"self", "cls"}
    if isinstance(owner, FUNCTION_NODES):
        own.update(parameter.arg for parameter in parameters(owner))
    return own


def _is_reference(literal: str, own: set[str]) -> bool:
    """Tell whether text in double backticks names another symbol."""
    if DOTTED_PATTERN.fullmatch(literal) is None:
        return False
    parts = literal.removesuffix("()").split(".")
    return parts[0] not in own and parts[-1] not in FILE_SUFFIXES


def backticked_names(text: str) -> list[str]:
    """Return the targets of roles and the texts in double backticks.

    Parameters
    ----------
    text
        A docstring's cleaned text.

    Returns
    -------
    list of str
        Every role target and double-backtick text, stripped, in order.
    """
    roles: list[str] = ROLE_PATTERN.findall(text)
    literals: list[str] = LITERAL_PATTERN.findall(text)
    names = [target.strip() for target in roles]
    names.extend(quoted.strip() for quoted in literals)
    return names
