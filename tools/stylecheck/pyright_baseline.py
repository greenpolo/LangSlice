"""Basedpyright's baseline, held to "only shrinks" like the checker's own.

basedpyright lists the errors that predate strict mode in
``.basedpyright/baseline.json``: per file, one entry per error with its
rule (``code``) and its columns, without a line number. An ordinary run
removes the entries of fixed errors from the file itself; ``--writebaseline``
adds every current error, which is the hole this module closes. Under
``--against REF`` the count of entries per file and rule may not rise above
its count at the reference, with two exceptions: a rule whose report level
or the checking mode changed since the reference (a rule being adopted, which
the settings record shows), and the entries that left a file the move map
moves definitions out of, which may reappear in a file it moves them into.
A file that is gone without a declared move takes its entries with it.
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
    move_sources: Callable[[str], tuple[str, ...]],
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
    move_sources
        Gives the files the move map moves definitions out of into a file;
        an entry that left one of them may reappear in that file, each one
        once, matched by its span first and else by its rule alone.

    Returns
    -------
    list of PyrightChange
        Each file and rule whose new entries are neither adopted nor
        moved, with the counts.
    """
    before = list(before)
    after = list(after)
    old = Counter((entry.path, entry.code) for entry in before)
    groups: dict[tuple[str, str], list[PyrightEntry]] = {}
    for entry in after:
        if not is_adopted(entry):
            groups.setdefault((entry.path, entry.code), []).append(entry)
    pool = _Departed.from_baselines(before, after, move_sources)

    previous = Counter(before)
    changes: list[PyrightChange] = []
    for (path, code), entries in sorted(groups.items()):
        allowed = old[(path, code)]
        new_entries = sorted((Counter(entries) - previous).elements())
        allowed += pool.take(new_entries, len(entries) - allowed)
        if len(entries) > allowed:
            changes.append(PyrightChange(path, code, allowed, len(entries)))
    return changes


@dataclass
class _Departed:
    """Hold the reference's entries that left their file, once each.

    ``left`` holds, per file the move map moves definitions out of, its
    entries that are no longer there.
    """

    left: dict[str, Counter[PyrightEntry]]
    move_sources: Callable[[str], tuple[str, ...]]

    @classmethod
    def from_baselines(
        cls,
        before: list[PyrightEntry],
        after: list[PyrightEntry],
        move_sources: Callable[[str], tuple[str, ...]],
    ) -> _Departed:
        """Return the entries that left their files between two baselines."""
        sources: set[str] = set()
        for path in {entry.path for entry in after}:
            sources.update(move_sources(path))
        left: dict[str, Counter[PyrightEntry]] = {}
        for source in sources:
            then = Counter(entry for entry in before if entry.path == source)
            now = Counter(entry for entry in after if entry.path == source)
            left[source] = then - now
        return cls(left, move_sources)

    def take(self, entries: list[PyrightEntry], excess: int) -> int:
        """Count new entries as moved, up to the excess; return how many."""
        taken = 0
        for entry in entries:
            if taken >= excess:
                break
            if self._take_one(entry):
                taken += 1
        return taken

    def _take_one(self, entry: PyrightEntry) -> bool:
        """Use up one departed entry for a new one, by span, else by rule."""
        sources = self.move_sources(entry.path)
        for source in sources:
            same = PyrightEntry(source, entry.code, entry.columns)
            if self.left[source][same] > 0:
                self.left[source][same] -= 1
                return True
        for source in sources:
            for left in self.left[source]:
                if left.code == entry.code and self.left[source][left] > 0:
                    self.left[source][left] -= 1
                    return True
        return False


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
