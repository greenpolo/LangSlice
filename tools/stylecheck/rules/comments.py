"""Checks on comments: no history, and line-length waivers only for URLs.

*History* is text that tells how the code came to be rather than why it
is so now: a calendar date (in ISO form, or a month name followed by a
year) or one of the phrases in ``COMMENT_HISTORY``. In docstrings the
list is shorter (``DOCSTRING_HISTORY``): "previously", "legacy" and "no
longer" describe present behaviour there often enough ("the previously
fitted affine", "reads a legacy job folder", "a section no longer in
the stack") to be left to review. The phrase
"used to" counts only as a past habit, after a subject word or before
*be* or *have*, and not in the passive voice.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterator
from typing import TypeGuard

from tools.stylecheck.core import (
    Config,
    Module,
    Project,
    Violation,
    module_check,
)
from tools.stylecheck.rules.docstrings import docstrings

#: A calendar date: ISO form, or a month name and a year.
DATE_PATTERN = re.compile(
    r"\b(?:19|20)\d\d-[01]\d-[0-3]\d\b"
    r"|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)"
    r"[a-z]*\.?,? (?:19|20)\d\d\b"
)

#: "Used to" as a past habit: after a subject word, or before "be"/"have".
USED_TO_PATTERN = re.compile(
    r"\b(?:it|this|that|we|they|which|these|those|code|function|module)"
    r" used to\b|\bused to (?:be|have)\b",
    re.IGNORECASE,
)

#: Words before "used to" that make it the present passive.
PASSIVE_WORDS = frozenset(
    {"is", "are", "be", "been", "being", "was", "were", "get", "gets"}
)

#: Phrases that tell history in a ``#`` comment.
COMMENT_HISTORY = re.compile(
    r"\b(?:legacy|no longer|earlier version|previously|was changed"
    r"|formerly)\b",
    re.IGNORECASE,
)

#: Phrases that tell history in a docstring.
DOCSTRING_HISTORY = re.compile(
    r"\b(?:earlier version|was changed|formerly)\b",
    re.IGNORECASE,
)

#: A waiver of the line-length rule.
LINE_LENGTH_WAIVER = re.compile(r"noqa:[^#]*\bE501\b")

#: A web address, the one thing a long line may hold.
URL_PATTERN = re.compile(r"https?://")


@module_check("history-comment")
def check_history_comment(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a comment or docstring that tells history.

    Guide rule 7.2. See the module docstring for what counts as history
    in each kind of text. Each comment, and each docstring, is reported
    once, at its first history phrase.

    Parameters
    ----------
    module
        The parsed file whose comments and docstrings are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per comment or docstring that tells history.
    """
    for line, text in module.comments:
        phrase = history_phrase(text, COMMENT_HISTORY)
        if phrase:
            yield module.violation("history-comment", line, _message(phrase))
    for docstring in docstrings(module.tree):
        phrase = history_phrase(docstring.text, DOCSTRING_HISTORY)
        if phrase:
            yield module.violation(
                "history-comment", docstring.line, _message(phrase)
            )


def _message(phrase: str) -> str:
    """Write the finding for one history phrase."""
    return (
        f"The text tells history ({phrase!r}); state the reason as it "
        "holds now and leave the history to git (7.2)."
    )


def history_phrase(text: str, phrases: re.Pattern[str]) -> str:
    """Return the first history phrase in a text, or ``""``.

    Parameters
    ----------
    text
        A comment or docstring.
    phrases
        The pattern of history phrases for that kind of text.

    Returns
    -------
    str
        The matched date or phrase.
    """
    for pattern in (DATE_PATTERN, phrases):
        match = pattern.search(text)
        if match:
            return match.group(0)
    for match in USED_TO_PATTERN.finditer(text):
        before = text[: match.start()].split()
        if not before or before[-1].lower() not in PASSIVE_WORDS:
            return match.group(0)
    return ""


@module_check("noqa-line-length")
def check_noqa_line_length(
    module: Module, project: Project, config: Config
) -> Iterator[Violation]:
    """Flag a line-length waiver on a line that is not a URL or a table row.

    Guide rule 1.1. A ``noqa: E501`` is allowed on a line that holds a
    web address, and on a line inside a literal table: a list, tuple,
    set or dict over several lines whose items are all constants or rows
    of constants.

    Parameters
    ----------
    module
        The parsed file whose comments are read.
    project
        The package index, unused by this check.
    config
        The settings, unused by this check.

    Yields
    ------
    Violation
        One finding per waiver that is not allowed.
    """
    table_lines = _literal_table_lines(module.tree)
    for line, text in module.comments:
        if not LINE_LENGTH_WAIVER.search(text):
            continue
        source = module.lines[line - 1]
        if URL_PATTERN.search(source) or line in table_lines:
            continue
        yield module.violation(
            "noqa-line-length",
            line,
            "A noqa: E501 waiver is allowed only on a URL or a row of a "
            "literal table; let the formatter break this line (1.1).",
        )


def _literal_table_lines(tree: ast.Module) -> set[int]:
    """Return the lines that lie inside a multi-line literal table."""
    lines: set[int] = set()
    for node in ast.walk(tree):
        if not _is_literal_table(node):
            continue
        last = node.end_lineno or node.lineno
        lines.update(range(node.lineno, last + 1))
    return lines


def _is_literal_table(
    node: ast.AST,
) -> TypeGuard[ast.Dict | ast.List | ast.Tuple | ast.Set]:
    """Tell whether a node is a collection of constants over many lines."""
    if isinstance(node, ast.Dict):
        items = [*node.keys, *node.values]
    elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = list(node.elts)
    else:
        return False
    if node.lineno == node.end_lineno or not items:
        return False
    return all(_is_constant_row(item) for item in items)


def _is_constant_row(node: ast.AST | None) -> bool:
    """Tell whether a table item is a constant or a row of constants."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return isinstance(node.operand, ast.Constant)
    if isinstance(node, (ast.Tuple, ast.List)):
        return all(_is_constant_row(item) for item in node.elts)
    return False
