"""The baseline: the errors that predate the checker, counted per place.

The file ``tools/stylecheck/baseline.json`` lists, for each rule, file and
enclosing definition, how many errors of that rule the definition may still
have. A run fails when a key's count rises above its entry, or when a key
without an entry has any error; a count below its entry means the entry can
shrink, which ``--prune`` writes. Entries carry no line numbers, so an edit
elsewhere in a file leaves them valid. Warnings never enter the file.

Beside the entries, the file records what they were counted with: the
checks that ran and the settings of the checker and its tools
(:mod:`tools.stylecheck.settings`). ``--write-baseline`` and ``--prune``
write the record of the run. The file is a JSON object with one item per
line, sorted, so that a change to it reads as a change to the lines of the
entries and settings it touches::

    {
    "checks": [
    "body-lines"
    ],
    "settings": {
    "stylecheck.max_body_lines": 100
    },
    "entries": [
    {"rule": "ruff:E501", "path": "a.py", "symbol": "main", "count": 2}
    ]
    }

``--against REF`` checks that the entries only shrank since a git commit
(:mod:`tools.stylecheck.against`).
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from tools.stylecheck.core import Severity, Violation
from tools.stylecheck.jsondata import as_list, as_table, load_json
from tools.stylecheck.scope import Key, Scope
from tools.stylecheck.settings import Record, parse_record

#: The fields of one baseline entry, in the order the file writes them.
ENTRY_FIELDS = ("rule", "path", "symbol", "count")

#: The fields of the file, in the order it writes them.
FILE_FIELDS = ("checks", "settings", "entries")


class BaselineError(Exception):
    """Signal a baseline file or a git reference that cannot be used."""


@dataclass(frozen=True)
class Baseline:
    """Hold a baseline file: its entries and what they were counted with.

    Parameters
    ----------
    counts
        The allowed count per rule, file and definition.
    record
        The checks and settings the entries were counted with.
    """

    counts: dict[Key, int]
    record: Record


@dataclass(frozen=True)
class CountChange:
    """Record a key whose error count differs from what a baseline allows.

    Parameters
    ----------
    key
        The rule, file and definition the count belongs to.
    allowed
        The count the baseline records, zero for an unlisted key.
    current
        The count the run or the newer baseline has for the key.
    """

    key: Key
    allowed: int
    current: int


@dataclass(frozen=True)
class Comparison:
    """Set the errors of one run against the baseline.

    Parameters
    ----------
    failing
        The errors of every key whose count exceeds the baseline's entry.
    excess
        Each key whose count exceeds the baseline's entry, with both counts.
    shrinkable
        Each baseline entry in the run's scope that the run came in under.
    n_covered
        How many of the run's errors the baseline allows.
    """

    failing: tuple[Violation, ...]
    excess: tuple[CountChange, ...]
    shrinkable: tuple[CountChange, ...]
    n_covered: int


def count_errors(violations: Iterable[Violation]) -> dict[Key, int]:
    """Count the errors of a run per baseline key, leaving out warnings.

    Parameters
    ----------
    violations
        Every finding of the run, errors and warnings.

    Returns
    -------
    dict
        The number of errors for each rule, file and definition.
    """
    counts = Counter(
        violation.key
        for violation in violations
        if violation.severity is Severity.ERROR
    )
    return dict(counts)


def compare(
    violations: Iterable[Violation],
    baseline: dict[Key, int],
    scope: Scope,
) -> Comparison:
    """Compare a run's errors with the baseline's entries.

    Parameters
    ----------
    violations
        Every finding of the run, already limited to ``scope``.
    baseline
        The allowed count per key, as read from the baseline file.
    scope
        The rules and files the run covered; entries outside it are
        neither compared nor reported as shrinkable.

    Returns
    -------
    Comparison
        The failing errors, the keys over and under their entries, and the
        number of errors the baseline covers.
    """
    errors = [
        violation
        for violation in violations
        if violation.severity is Severity.ERROR
    ]
    current = count_errors(errors)

    excess = [
        CountChange(key, baseline.get(key, 0), count)
        for key, count in sorted(current.items())
        if count > baseline.get(key, 0)
    ]
    over_keys = {change.key for change in excess}
    failing = [error for error in errors if error.key in over_keys]
    n_covered = len(errors) - len(failing)

    return Comparison(
        failing=tuple(failing),
        excess=tuple(excess),
        shrinkable=tuple(_shrinkable(baseline, current, scope)),
        n_covered=n_covered,
    )


def _shrinkable(
    baseline: dict[Key, int], current: dict[Key, int], scope: Scope
) -> list[CountChange]:
    """Return the in-scope entries whose run count is below the entry."""
    changes: list[CountChange] = []
    for key, allowed in sorted(baseline.items()):
        count = current.get(key, 0)
        if scope.contains(key) and count < allowed:
            changes.append(CountChange(key, allowed, count))
    return changes


def prune(
    baseline: dict[Key, int], current: dict[Key, int], scope: Scope
) -> dict[Key, int]:
    """Lower each entry to the run's count, never raising or adding one.

    Parameters
    ----------
    baseline
        The allowed count per key, as read from the baseline file.
    current
        The run's error count per key.
    scope
        The rules and files the run covered; entries outside it are kept
        as they are, since the run did not count them.

    Returns
    -------
    dict
        The pruned entries; an entry whose count falls to zero is removed.
    """
    pruned: dict[Key, int] = {}
    for key, allowed in baseline.items():
        count = allowed
        if scope.contains(key):
            count = min(allowed, current.get(key, 0))
        if count > 0:
            pruned[key] = count
    return pruned


def read_baseline(path: Path) -> Baseline:
    """Read a baseline file.

    Parameters
    ----------
    path
        The baseline file; it must exist.

    Returns
    -------
    Baseline
        The allowed count per key and the record of the run.

    Raises
    ------
    BaselineError
        When the file is not a baseline object of well-formed, distinct
        entries.
    """
    return parse_baseline(path.read_text(encoding="utf-8"), str(path))


def parse_baseline(text: str, source: str) -> Baseline:
    """Parse the text of a baseline file.

    Parameters
    ----------
    text
        The file's contents.
    source
        Where the text came from, for the error message.

    Returns
    -------
    Baseline
        The allowed count per key and the record of the run.

    Raises
    ------
    BaselineError
        When the text is not a baseline object of well-formed, distinct
        entries.
    """
    try:
        document = as_table(load_json(text))
    except json.JSONDecodeError as error:
        raise BaselineError(f"{source} is not valid JSON: {error}.") from error
    if document is None or sorted(document) != sorted(FILE_FIELDS):
        raise BaselineError(
            f"{source} must hold a JSON object with the fields "
            f"{list(FILE_FIELDS)}; --write-baseline writes one."
        )

    record = parse_record(document["checks"], document["settings"])
    if record is None:
        raise BaselineError(
            f"{source} needs a list of check names in checks and an object "
            "in settings."
        )
    entries = as_list(document["entries"])
    if entries is None:
        raise BaselineError(f"{source} must hold a list in entries.")

    counts: dict[Key, int] = {}
    for item in entries:
        entry = as_table(item)
        if entry is None:
            raise BaselineError(f"{source} has an entry that is not a dict.")
        key, count = _entry(entry, source)
        if key in counts:
            raise BaselineError(f"{source} lists {key} twice.")
        counts[key] = count
    return Baseline(counts, record)


def _entry(entry: dict[str, object], source: str) -> tuple[Key, int]:
    """Return the key and count of one entry, checking its fields."""
    if sorted(entry) != sorted(ENTRY_FIELDS):
        raise BaselineError(
            f"{source} has an entry with fields {sorted(entry)}; each entry "
            f"has exactly {list(ENTRY_FIELDS)}."
        )

    rule, path, symbol = entry["rule"], entry["path"], entry["symbol"]
    count = entry["count"]
    if not (
        isinstance(rule, str)
        and isinstance(path, str)
        and isinstance(symbol, str)
    ):
        raise BaselineError(f"{source} has an entry with a non-text key.")

    if not isinstance(count, int) or isinstance(count, bool) or count < 1:
        raise BaselineError(
            f"{source} gives {rule} {path} {symbol} the count {count!r}; a "
            "count is a whole number of at least 1."
        )
    return (rule, path, symbol), count


def format_baseline(baseline: Baseline) -> str:
    """Write a baseline as the text of a baseline file.

    Parameters
    ----------
    baseline
        The allowed count per key, and the record of the run; keys with a
        count below 1 are left out.

    Returns
    -------
    str
        A JSON object of the checks, the settings and the entries, each
        sorted, one item per line.
    """
    lines: list[str] = []
    for (rule, path, symbol), count in sorted(baseline.counts.items()):
        if count < 1:
            continue
        entry = {"rule": rule, "path": path, "symbol": symbol, "count": count}
        lines.append(_dumps(entry))

    checks = [_dumps(name) for name in sorted(baseline.record.checks)]
    settings_lines = [
        f"{_dumps(name)}: {_dumps(value)}"
        for name, value in sorted(baseline.record.settings.items())
    ]
    fields = (
        f'"checks": {_block("[", checks, "]")}',
        f'"settings": {_block("{", settings_lines, "}")}',
        f'"entries": {_block("[", lines, "]")}',
    )
    return "{\n" + ",\n".join(fields) + "\n}\n"


def write_baseline(path: Path, baseline: Baseline) -> None:
    """Save a baseline to a file, replacing what it held.

    Parameters
    ----------
    path
        The baseline file to write.
    baseline
        The allowed count per key and the record of the run.
    """
    path.write_text(format_baseline(baseline), encoding="utf-8")


def _dumps(value: object) -> str:
    """Return one value as compact JSON text on one line."""
    return json.dumps(value, ensure_ascii=False)


def _block(opening: str, lines: list[str], closing: str) -> str:
    """Return a JSON list or object with one item per line."""
    if not lines:
        return opening + closing
    return opening + "\n" + ",\n".join(lines) + "\n" + closing
