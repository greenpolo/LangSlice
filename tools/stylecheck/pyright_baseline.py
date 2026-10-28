"""Basedpyright's baseline, held to "only shrinks" like the checker's own.

basedpyright lists the errors that predate strict mode in
``.basedpyright/baseline.json``: per file, one entry per error with its
rule (``code``) and its columns, without a line number. An ordinary run
removes the entries of fixed errors from the file itself; ``--writebaseline``
adds every current error, which is the hole this module closes. Under
``--against REF`` the count of entries per file and rule may not rise above
its count at the reference, with two exceptions: a rule whose report level
or the checking mode changed since the reference (a rule being adopted, which
the settings record shows), and the entries of a file that was moved (an
entry of the same rule and columns at the reference, in a file that no
longer exists).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from tools.stylecheck.jsondata import (
    as_int,
    as_list,
    as_table,
    as_text,
    load_json,
)

#: Where basedpyright keeps its baseline, relative to the repository root.
PYRIGHT_BASELINE = ".basedpyright/baseline.json"


class PyrightBaselineError(Exception):
    """Signal a basedpyright baseline that cannot be read."""


@dataclass(frozen=True, order=True)
class PyrightEntry:
    """Hold one error of basedpyright's baseline.

    Parameters
    ----------
    path
        The file as basedpyright writes it, such as ``./src/a.py``.
    code
        The rule, such as ``reportAny``.
    columns
        The start column, end column and line count of the error, which
        basedpyright matches errors by.
    """

    path: str
    code: str
    columns: tuple[int, int, int]


@dataclass(frozen=True)
class PyrightChange:
    """Record a file and rule whose entries grew since a reference.

    Parameters
    ----------
    path
        The file as basedpyright writes it.
    code
        The basedpyright rule the entries break.
    allowed
        The number of entries the reference allows, moves included.
    current
        The number of entries now.
    """

    path: str
    code: str
    allowed: int
    current: int


def parse_pyright_baseline(text: str, source: str) -> list[PyrightEntry]:
    """Read the entries of a basedpyright baseline.

    Parameters
    ----------
    text
        The file's contents.
    source
        Where the text came from, for the error message.

    Returns
    -------
    list of PyrightEntry
        Every entry, sorted.

    Raises
    ------
    PyrightBaselineError
        When the text is not basedpyright's ``{"files": {...}}`` object.
    """
    try:
        document = as_table(load_json(text))
    except json.JSONDecodeError as error:
        raise PyrightBaselineError(
            f"{source} is not valid JSON: {error}."
        ) from error
    files = None
    if document is not None:
        files = as_table(document.get("files"))
    if files is None:
        raise PyrightBaselineError(f"{source} has no files object.")

    entries: list[PyrightEntry] = []
    for path, items in files.items():
        for item in as_list(items) or []:
            entry = _entry(path, item)
            if entry is None:
                raise PyrightBaselineError(
                    f"{source} has an entry of {path} without a code and "
                    "a range."
                )
            entries.append(entry)
    return sorted(entries)


def pyright_growth(
    before: Iterable[PyrightEntry],
    after: Iterable[PyrightEntry],
    *,
    is_adopted: Callable[[PyrightEntry], bool],
    is_gone: Callable[[str], bool],
) -> list[PyrightChange]:
    """Find the files and rules with more entries than at a reference.

    Parameters
    ----------
    before
        The entries at the reference.
    after
        The entries now.
    is_adopted
        Tells whether an entry belongs to a rule adopted since the
        reference; such entries may be new.
    is_gone
        Tells whether a file of the reference no longer exists; its
        entries may reappear in another file, each one once.

    Returns
    -------
    list of PyrightChange
        Each file and rule whose new entries are neither adopted nor
        moved, with the counts.
    """
    old = Counter((entry.path, entry.code) for entry in before)
    moved = Counter(
        (entry.code, entry.columns) for entry in before if is_gone(entry.path)
    )
    groups: dict[tuple[str, str], list[PyrightEntry]] = {}
    for entry in after:
        if not is_adopted(entry):
            groups.setdefault((entry.path, entry.code), []).append(entry)

    previous = Counter(before)
    changes: list[PyrightChange] = []
    for (path, code), entries in sorted(groups.items()):
        allowed = old[(path, code)]
        excess = len(entries) - allowed
        new_entries = Counter(entries) - previous
        for entry in sorted(new_entries.elements()):
            signature = (entry.code, entry.columns)
            if excess > 0 and moved[signature] > 0:
                moved[signature] -= 1
                allowed += 1
                excess -= 1
        if excess > 0:
            changes.append(PyrightChange(path, code, allowed, len(entries)))
    return changes


def _entry(path: str, item: object) -> PyrightEntry | None:
    """Read one entry of a file's list, or None when it is malformed."""
    entry = as_table(item) or {}
    code = as_text(entry.get("code"))
    span = as_table(entry.get("range")) or {}
    numbers = [
        as_int(span.get(name))
        for name in ("startColumn", "endColumn", "lineCount")
    ]
    start, end, count = numbers
    if code is None or start is None or end is None or count is None:
        return None
    return PyrightEntry(path, code, (start, end, count))
