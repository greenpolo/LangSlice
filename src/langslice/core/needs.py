"""Text that names a tool or an option, written with what it needs.

Every sentence or passage a model reads that names a tool (``zoom``) or an
option (``look:positioning``) is written once, for the run that has them,
with what it needs marked in it, and rendered for the job: dropped, or
swapped for another wording, when a name it needs is in the job's disabled
list (:attr:`langslice.core.spec.JobSpec.disabled`). The marks:

- ``[[?NEEDS text]]``: *text* when NEEDS is met, else nothing. NEEDS is
  terms separated by commas, all of which must be met; a term is a name, or
  names separated by ``|`` of which one must be on; then one whitespace
  character (not part of *text*).
- ``[[?NEEDS text||other]]``: *other* in place of nothing.
- ``[[#NEEDS=item||NEEDS=item||...]]``: the items whose NEEDS are met, as a
  list ("a, b and c"; nothing when none is); ``[[%`` in place of ``[[#``
  makes it "a, b or c".

Marks nest. A line of marked text that would be too long in the source ends
in ``~``, which joins it to the next line (the newline and the next line's
indentation go). A name is off when it is in the list, or, for ``tool:option``,
when its tool is (:func:`langslice.core.toolset.is_off`). With nothing off a
text renders to itself without its marks, byte for byte; only a text that
lost something is tidied (a line left empty goes, trailing spaces go, a mark
left alone on a line joins the line before, blank lines do not pile up). A
text that is not written with marks (the user's own words, a filename) is
never rendered.
"""

from __future__ import annotations

import re
from collections.abc import Collection

from langslice.core.toolset import is_off

_TOKEN = re.compile(r"(\[\[[?#%]|\]\]|\|\|)")
_OPENERS = ("[[?", "[[#", "[[%")


def _needs_on(needs: str, off: Collection[str]) -> bool:
    return all(any(not is_off(off, name) for name in term.split("|"))
               for term in needs.split(",") if term)


def _group(tokens: list[str], index: int, kind: str, off: Collection[str],
           source: str) -> tuple[str, int]:
    """The group opened by *kind* at *tokens[index]* rendered, and the index
    after its closing mark."""
    branches: list[str] = [""]
    while index < len(tokens):
        token = tokens[index]
        if token == "]]":
            return _render_group(kind, branches, off, source), index + 1
        if token == "||":
            branches.append("")
            index += 1
        elif token in _OPENERS:
            inner, index = _group(tokens, index + 1, token[2], off, source)
            branches[-1] += inner
        else:
            branches[-1] += token
            index += 1
    raise ValueError(f"unclosed [[ in the text {source[:60]!r}")


def _render_group(kind: str, branches: list[str], off: Collection[str], source: str) -> str:
    if kind == "?":
        head = re.match(r"(\S+)\s(.*)\Z", branches[0], re.DOTALL)
        if head is None:
            raise ValueError(f"[[? needs names, then a space, in the text {source[:60]!r}")
        if _needs_on(head.group(1), off):
            return head.group(2)
        return branches[1] if len(branches) > 1 else ""
    items: list[str] = []
    for branch in branches:
        needs, equals, item = branch.partition("=")
        if not equals:
            raise ValueError(f"[[# item {branch!r} lacks NEEDS= in the text {source[:60]!r}")
        if _needs_on(needs.strip(), off):
            items.append(item.strip())
    if len(items) > 1:
        return ", ".join(items[:-1]) + (" and " if kind == "#" else " or ") + items[-1]
    return items[0] if items else ""


def _pass(text: str, off: Collection[str]) -> str:
    tokens = [piece for piece in _TOKEN.split(text) if piece]
    out = ""
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token in _OPENERS:
            inner, index = _group(tokens, index + 1, token[2], off, text)
            out += inner
        elif token in ("]]", "||"):
            out += token
            index += 1
        else:
            out += token
            index += 1
    return out


def _tidy(text: str) -> str:
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]*([.,;:])", r"\1", text)  # a mark left alone on a line
    return re.sub(r"\n{3,}", "\n\n", text)


def render(text: str, off: Collection[str] = ()) -> str:
    """*text* for a job whose disabled list is *off* (see the module text)."""
    if "[[" not in text:
        return text
    text = re.sub(r"~\n[ \t]*", "", text)
    whole = _pass(text, ())
    shown = _pass(text, off)
    return whole if shown == whole else _tidy(shown)


def marked(text: str) -> bool:
    """Whether *text* is written with marks."""
    return any(opener in text for opener in _OPENERS)
