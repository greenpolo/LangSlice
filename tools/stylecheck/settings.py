"""What a baseline is written with: the checks, the tools' rules, the limits.

A baseline's entries mean something only next to the rules that produced
them, so the file records, beside its entries, the name of every check that
ran and every setting that decides what the checks and tools report: each
field of ``[tool.stylecheck]`` with the checker's defaults filled in
(``stylecheck.<field>``), the rule selection of the checker's own ruff run
(``checker-ruff.select`` and ``checker-ruff.ignore``, from
``external.py``), and every key of the ``[tool.ruff]``,
``[tool.numpydoc_validation]`` and ``[tool.basedpyright]`` tables
(``ruff.lint.select``, ``numpydoc.checks``, ``basedpyright.include`` and so
on).

``--against`` reads the record at the reference commit for two things.
A baseline key whose rule the reference did not run (a new check, a newly
selected ruff or numpydoc code, a code a file no longer ignores) may enter
the baseline: that is how a new rule is adopted. And every setting must be
at least as strict as at the reference, in the :class:`Direction` the
setting's kind gives; any other change loosens the checks.
"""

# numpydoc ships no type information; the one function used is typed below.
# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import dataclasses
import enum
import fnmatch
import functools
import json
from collections.abc import Callable, Container, Iterator
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import cast

from numpydoc.validate import get_validation_checks

from tools.stylecheck.core import Config
from tools.stylecheck.external import RUFF_IGNORE, RUFF_SELECT
from tools.stylecheck.jsondata import (
    as_int,
    as_list,
    as_table,
    as_text,
    canonical,
    load_toml,
)
from tools.stylecheck.scope import Key, Tool


class Direction(enum.StrEnum):
    """Say which change of a setting keeps the checks at least as strict."""

    #: A limit, which may only go down.
    LOWER = "lower"
    #: A minimum, which may only go up.
    HIGHER = "higher"
    #: A list of what is checked or banned; items may only be added.
    GROWS = "grows"
    #: A list of exemptions; items may only be removed.
    SHRINKS = "shrinks"
    #: File patterns to ignored codes; codes and patterns may only go.
    IGNORES = "ignores"
    #: Names to values; entries may only be added.
    TABLE = "table"
    #: A basedpyright report level, which may only rise.
    SEVERITY = "severity"
    #: basedpyright's ``typeCheckingMode``, which may only rise.
    MODE = "mode"
    #: Any change needs a reader to judge it.
    FIXED = "fixed"


#: The direction of every field of :class:`Config`; a new field is given
#: one here (tests/stylecheck/test_framework.py checks that each has one).
CONFIG_DIRECTIONS: dict[str, Direction] = {
    "paths": Direction.GROWS,
    "baseline": Direction.FIXED,
    "max_cognitive_complexity": Direction.LOWER,
    "max_nesting": Direction.LOWER,
    "max_body_lines": Direction.LOWER,
    "max_parameters": Direction.LOWER,
    "max_module_lines": Direction.LOWER,
    "max_class_methods": Direction.LOWER,
    "max_class_lines": Direction.LOWER,
    "max_paragraph_lines": Direction.LOWER,
    "max_returned_values": Direction.LOWER,
    "max_decorators": Direction.LOWER,
    "max_docstring_references": Direction.LOWER,
    "min_parameter_words": Direction.HIGHER,
    "banned_name_stems": Direction.GROWS,
    "banned_function_suffixes": Direction.GROWS,
    "banned_summary_words": Direction.GROWS,
    "type_names": Direction.TABLE,
    "heavy_modules": Direction.SHRINKS,
    "json_boundary_modules": Direction.SHRINKS,
    "cli_modules": Direction.SHRINKS,
    "decorators_module": Direction.FIXED,
    "checkers_module": Direction.FIXED,
    "codes_module": Direction.FIXED,
    "infra_package": Direction.FIXED,
    "docs_renderer_module": Direction.FIXED,
    "layers": Direction.FIXED,
}

#: The ``[tool.ruff]`` keys with a direction; every other key is fixed. The
#: keys whose value is a table are recorded whole, not key by key.
RUFF_DIRECTIONS: dict[str, Direction] = {
    "ruff.line-length": Direction.LOWER,
    "ruff.include": Direction.GROWS,
    "ruff.extend-include": Direction.GROWS,
    "ruff.exclude": Direction.SHRINKS,
    "ruff.extend-exclude": Direction.SHRINKS,
    "ruff.lint.select": Direction.GROWS,
    "ruff.lint.extend-select": Direction.GROWS,
    "ruff.lint.ignore": Direction.SHRINKS,
    "ruff.lint.per-file-ignores": Direction.IGNORES,
    "ruff.lint.extend-per-file-ignores": Direction.IGNORES,
    "ruff.lint.pycodestyle.max-line-length": Direction.LOWER,
    "ruff.lint.pep8-naming.extend-ignore-names": Direction.SHRINKS,
    "ruff.lint.flake8-import-conventions.extend-aliases": Direction.TABLE,
}

#: The other recorded keys with a direction; every other key is fixed.
OTHER_DIRECTIONS: dict[str, Direction] = {
    "checker-ruff.select": Direction.GROWS,
    "checker-ruff.ignore": Direction.SHRINKS,
    "numpydoc.checks": Direction.GROWS,
    "numpydoc.exclude": Direction.SHRINKS,
    "basedpyright.include": Direction.GROWS,
    "basedpyright.strict": Direction.GROWS,
    "basedpyright.exclude": Direction.SHRINKS,
    "basedpyright.ignore": Direction.SHRINKS,
    "basedpyright.typeCheckingMode": Direction.MODE,
}

#: basedpyright's report levels, least strict first; ``false`` and
#: ``true`` are its spellings of ``none`` and ``error``.
SEVERITIES = ("none", "hint", "information", "warning", "error")

#: basedpyright's checking modes, least strict first.
MODES = ("off", "basic", "standard", "strict", "recommended", "all")

#: The ruff table whose ignores the checker's own ruff run keeps; it
#: replaces ``per-file-ignores``, which holds the main run's alone.
CHECKER_RUFF_IGNORES = "ruff.lint.extend-per-file-ignores"


@dataclass(frozen=True)
class SettingChange:
    """Record a setting that is looser, or changed, since a reference.

    Parameters
    ----------
    name
        The setting, such as ``stylecheck.max_nesting``, or ``checks``
        for a check no longer run.
    before
        Its value at the reference, as JSON text; empty when it was unset.
    after
        Its value now, as JSON text; empty when it is unset.
    rule
        Which change the setting allows, in words.
    """

    name: str
    before: str
    after: str
    rule: str


@dataclass(frozen=True)
class Record:
    """Hold what a baseline was written with.

    Parameters
    ----------
    checks
        The checks that ran, by name, with ``format`` for the formatter.
    settings
        Every recorded setting by its dotted name, as JSON data.
    """

    checks: tuple[str, ...]
    settings: dict[str, object]

    def covers(self, key: Key) -> bool:
        """Return whether the rule of a baseline key was run.

        Parameters
        ----------
        key
            The rule, file and definition of a baseline entry.

        Returns
        -------
        bool
            For ``ruff:<CODE>``, whether the checker's ruff run selects the
            code and the file does not ignore it; for ``numpydoc:<CODE>``,
            whether the code is validated; for a check, whether it ran.
        """
        rule, path, _ = key
        tool, _, code = rule.partition(":")
        if tool == Tool.RUFF and code:
            return self._ruff_reports(code, path)
        if tool == Tool.NUMPYDOC and code:
            return code in self.numpydoc_codes()
        return rule in self.checks

    def numpydoc_codes(self) -> frozenset[str]:
        """Return the numpydoc codes the record validates.

        Returns
        -------
        frozenset of str
            The codes of ``numpydoc.checks``, with ``all`` and the codes
            listed beside it expanded as numpydoc does.
        """
        listed = recorded_texts(self.settings.get("numpydoc.checks"))
        expanded: set[str] = get_validation_checks(set(listed))
        return frozenset(expanded)

    def differs(self, prefix: str, other: Record) -> bool:
        """Return whether a group of settings differs from another record.

        Parameters
        ----------
        prefix
            The group's start, such as ``stylecheck.``.
        other
            The record to compare with.

        Returns
        -------
        bool
            True when a setting of the group is in one record only, or has
            another value there.
        """
        return _group(self, prefix) != _group(other, prefix)

    def config(self, current: Config) -> Config:
        """Return the checker's settings with this record's values.

        Parameters
        ----------
        current
            The settings now; a field the record lacks keeps its value.

        Returns
        -------
        Config
            The settings the record's baseline was written with.
        """
        values: dict[str, object] = {}
        for field in dataclasses.fields(Config):
            name = f"stylecheck.{field.name}"
            if name in self.settings:
                values[field.name] = _python_value(self.settings[name])
        return dataclasses.replace(current, **values)

    def _ruff_reports(self, code: str, path: str) -> bool:
        """Return whether the checker's ruff run reports a code in a file."""
        selected = recorded_texts(self.settings.get("checker-ruff.select"))
        ignored = recorded_texts(self.settings.get("checker-ruff.ignore"))
        if not _matches(code, selected) or _matches(code, ignored):
            return False
        patterns = as_table(self.settings.get(CHECKER_RUFF_IGNORES)) or {}
        for pattern, codes in patterns.items():
            if _path_matches(path, pattern) and _matches(
                code, recorded_texts(codes)
            ):
                return False
        return True


def current_record(root: Path, config: Config, checks: list[str]) -> Record:
    """Return the record of the checks and settings a run uses now.

    Parameters
    ----------
    root
        The repository root, whose ``pyproject.toml`` holds the tables.
    config
        The checker's settings, as read from that file.
    checks
        The names of the registered checks and of the formatter check.

    Returns
    -------
    Record
        The checks, sorted, and every recorded setting.
    """
    # cast: a dataclass's fields, as the plain data they hold
    fields = as_table(canonical(cast(object, dataclasses.asdict(config))))
    settings: dict[str, object] = {
        f"stylecheck.{name}": value for name, value in (fields or {}).items()
    }
    settings["checker-ruff.select"] = canonical(RUFF_SELECT)
    settings["checker-ruff.ignore"] = canonical(RUFF_IGNORE)

    document = load_toml((root / "pyproject.toml").read_text(encoding="utf-8"))
    tool = as_table(document.get("tool")) or {}
    tables: tuple[tuple[str, str, Container[str]], ...] = (
        ("ruff", "ruff", RUFF_DIRECTIONS),
        ("numpydoc_validation", "numpydoc", {}),
        ("basedpyright", "basedpyright", {}),
    )
    for table_name, prefix, leaves in tables:
        table = as_table(tool.get(table_name)) or {}
        settings.update(_flatten(table, prefix, leaves))
    return Record(tuple(sorted(checks)), dict(sorted(settings.items())))


def loosenings(before: Record, after: Record) -> list[SettingChange]:
    """Return each way a newer record runs looser checks than an older one.

    Parameters
    ----------
    before
        The record at the reference commit.
    after
        The record now.

    Returns
    -------
    list of SettingChange
        Each check no longer run, and each setting changed against its
        direction or, for a fixed setting, changed at all, sorted by name.
    """
    changes = [
        SettingChange("checks", json.dumps(name), "", "a check may not go")
        for name in before.checks
        if name not in after.checks
    ]

    for name in sorted({*before.settings, *after.settings}):
        old = before.settings.get(name)
        new = after.settings.get(name)
        if old == new:
            continue
        if old is None and name.startswith("stylecheck."):
            continue  # a field the checker gained, with the code that reads it

        direction = setting_direction(name)
        if not _ACCEPTS[direction](old, new):
            changes.append(
                SettingChange(
                    name, _json(old), _json(new), _RULE_WORDS[direction]
                )
            )
    return changes


def setting_direction(name: str) -> Direction:
    """Return the direction of a recorded setting.

    Parameters
    ----------
    name
        The setting's dotted name, such as ``ruff.lint.select``.

    Returns
    -------
    Direction
        The direction its kind gives; a basedpyright ``report`` key is a
        severity, and a key no table lists is fixed.
    """
    if name.startswith("stylecheck."):
        field = name.removeprefix("stylecheck.")
        return CONFIG_DIRECTIONS.get(field, Direction.FIXED)
    if name in RUFF_DIRECTIONS:
        return RUFF_DIRECTIONS[name]
    if name.startswith("basedpyright.report"):
        return Direction.SEVERITY
    return OTHER_DIRECTIONS.get(name, Direction.FIXED)


def parse_record(checks: object, settings: object) -> Record | None:
    """Read a record from the fields of a baseline file.

    Parameters
    ----------
    checks
        The file's ``checks`` field, which must be a list of names.
    settings
        The file's ``settings`` field, which must be an object.

    Returns
    -------
    Record or None
        The record, or None when either field has the wrong shape.
    """
    names = as_list(checks)
    table = as_table(settings)
    if names is None or table is None:
        return None
    texts = recorded_texts(names)
    if len(texts) != len(names):
        return None
    return Record(tuple(texts), table)


def differences(first: Record, second: Record) -> list[str]:
    """Name what two records hold differently.

    Parameters
    ----------
    first, second
        The records to compare, such as a baseline file's and the run's.

    Returns
    -------
    list of str
        ``check <name>`` for each check in one record only, then each
        setting whose value differs, sorted.
    """
    names = [
        f"check {name}"
        for name in sorted(set(first.checks) ^ set(second.checks))
    ]
    for name in sorted({*first.settings, *second.settings}):
        if first.settings.get(name) != second.settings.get(name):
            names.append(name)
    return names


def recorded_texts(value: object) -> list[str]:
    """Return the text items of a recorded list.

    Parameters
    ----------
    value
        A recorded setting's value.

    Returns
    -------
    list of str
        Its items that are text; none when the value is not a list.
    """
    items = as_list(value) or []
    return [text for text in map(as_text, items) if text is not None]


def _flatten(
    table: dict[str, object], prefix: str, leaves: Container[str]
) -> Iterator[tuple[str, object]]:
    """Yield a table's values by dotted name, descending into sub-tables."""
    for key, value in table.items():
        name = f"{prefix}.{key}"
        inner = as_table(value)
        if inner is not None and name not in leaves:
            yield from _flatten(inner, name, leaves)
        else:
            yield name, canonical(value)


def _group(record: Record, prefix: str) -> dict[str, object]:
    """Return the settings of a record whose names start with a prefix."""
    return {
        name: value
        for name, value in record.settings.items()
        if name.startswith(prefix)
    }


def _python_value(value: object) -> object:
    """Return recorded JSON data as Config holds it: tuples for lists."""
    items = as_list(value)
    if items is None:
        return value
    return tuple(_python_value(item) for item in items)


def _matches(code: str, selectors: list[str]) -> bool:
    """Return whether a ruff code falls under one of several selectors.

    A selector is a code or the start of one (``D2``, ``N``), so the
    match is by prefix, which may count an unrelated linter as selected
    (``N`` before ``NPY001``): that errs toward an older selection having
    run a rule, which never adopts an entry wrongly.
    """
    return any(code.startswith(selector) for selector in selectors)


def _path_matches(path: str, pattern: str) -> bool:
    """Return whether a file falls under a ruff per-file pattern."""
    if fnmatch.fnmatch(path, pattern):
        return True
    if path.startswith(f"{pattern.rstrip('/')}/"):
        return True
    return "/" not in pattern and fnmatch.fnmatch(
        PurePosixPath(path).name, pattern
    )


def _json(value: object) -> str:
    """Return a recorded value as JSON text; empty when it is unset."""
    if value is None:
        return ""
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def _items(value: object) -> set[str]:
    """Return a recorded list's items as JSON texts; none when unset."""
    return {_json(item) for item in as_list(value) or []}


def _lower(old: object, new: object) -> bool:
    """Accept a number that went down."""
    before, after = as_int(old), as_int(new)
    return before is not None and after is not None and after <= before


def _higher(old: object, new: object) -> bool:
    """Accept a number that went up."""
    before, after = as_int(old), as_int(new)
    return before is not None and after is not None and after >= before


def _grows(old: object, new: object) -> bool:
    """Accept a list that only gained items."""
    return _items(old) <= _items(new)


def _shrinks(old: object, new: object) -> bool:
    """Accept a list that only lost items."""
    return _items(new) <= _items(old)


def _ignores(old: object, new: object) -> bool:
    """Accept per-file ignores that only lost patterns or codes."""
    before = as_table(old) or {}
    after = as_table(new) or {}
    return all(
        _items(codes) <= _items(before.get(pattern))
        for pattern, codes in after.items()
    )


def _table(old: object, new: object) -> bool:
    """Accept a table that only gained entries."""
    before = as_table(old) or {}
    after = as_table(new) or {}
    return all(after.get(name) == value for name, value in before.items())


def _rank(value: object, order: tuple[str, ...]) -> int | None:
    """Return a level's place in its order; None for an unknown level."""
    if value is True:
        value = "error"
    elif value is False:
        value = "none"
    text = as_text(value)
    if text not in order:
        return None
    return order.index(text)


def _level_rose(order: tuple[str, ...], old: object, new: object) -> bool:
    """Accept a level that stayed or went up in its order."""
    before, after = _rank(old, order), _rank(new, order)
    return before is not None and after is not None and after >= before


def _fixed(old: object, new: object) -> bool:
    """Accept no change: a changed fixed setting needs a reader."""
    return old == new


#: The test of each direction: whether a change from the first value to
#: the second keeps the checks at least as strict.
_ACCEPTS: dict[Direction, Callable[[object, object], bool]] = {
    Direction.LOWER: _lower,
    Direction.HIGHER: _higher,
    Direction.GROWS: _grows,
    Direction.SHRINKS: _shrinks,
    Direction.IGNORES: _ignores,
    Direction.TABLE: _table,
    Direction.SEVERITY: functools.partial(_level_rose, SEVERITIES),
    Direction.MODE: functools.partial(_level_rose, MODES),
    Direction.FIXED: _fixed,
}

#: What each direction allows, for the report.
_RULE_WORDS: dict[Direction, str] = {
    Direction.LOWER: "a limit may only go down",
    Direction.HIGHER: "a minimum may only go up",
    Direction.GROWS: "items may only be added",
    Direction.SHRINKS: "exemptions may only be removed",
    Direction.IGNORES: "ignored codes may only be removed",
    Direction.TABLE: "entries may only be added",
    Direction.SEVERITY: "a report level may only rise",
    Direction.MODE: "the checking mode may only rise",
    Direction.FIXED: "a change needs a reader",
}
