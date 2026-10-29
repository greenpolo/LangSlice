"""The declared moves of definitions, which ``--against`` follows.

A commit that moves or renames modules changes the paths, and sometimes
the definitions, of baseline entries. ``tools/stylecheck/moves.json``
lists each move as an old file and definition and a new file and
definition::

    {"moves": [
      {"path": "src/a.py", "symbol": "f",
       "new_path": "src/b.py", "new_symbol": "f"}
    ]}

Under ``--against REF`` an entry of the new file and definition is
accepted up to what the old one lost since the reference: its whole
count there when it is no longer in the baseline. A definition is listed
once per new place: a module split into several files lists its module
level (``<module>``, which holds the imports each part copies) once per
part, and each part may have up to what the old one lost. The settings
that name modules (the boundary modules, the command modules, the
renderer, the layers) are read at the reference through the same map, so
a module that moved is not a loosened setting.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from tools.stylecheck.core import module_name_from_path
from tools.stylecheck.jsondata import as_list, as_table, as_text, load_json
from tools.stylecheck.scope import Key
from tools.stylecheck.settings import Record

#: Where the move map lives, relative to the repository root.
MOVES_FILE = "tools/stylecheck/moves.json"

#: The settings whose items are module names, each read as a list.
MODULE_LISTS = (
    "stylecheck.json_boundary_modules",
    "stylecheck.cli_modules",
)

#: The settings whose value is one module name.
MODULE_NAMES = (
    "stylecheck.docs_renderer_module",
    "stylecheck.decorators_module",
    "stylecheck.checkers_module",
    "stylecheck.codes_module",
    "stylecheck.infra_package",
)

#: The setting that holds import-linter's layers, highest first.
LAYERS = "stylecheck.layers"

#: The setting whose keys may be dotted ``module.Type`` names.
TYPE_NAMES = "stylecheck.type_names"

#: The fields of one move in the file.
FIELDS = ("path", "symbol", "new_path", "new_symbol")


class MovesError(Exception):
    """Signal a move map that cannot be read."""


#: A definition: its file relative to the root and its qualified name.
Place = tuple[str, str]


@dataclass(frozen=True)
class Moves:
    """Hold the declared moves of one commit.

    Parameters
    ----------
    places
        The new files and definitions of each old file and definition.
    """

    places: dict[Place, tuple[Place, ...]]

    def pool(
        self, allowed_counts: dict[Key, int], counts: dict[Key, int]
    ) -> Counter[Key]:
        """Count what the reference allows under each new key.

        Parameters
        ----------
        allowed_counts
            The baseline's counts at the reference.
        counts
            The baseline's counts now.

        Returns
        -------
        Counter of Key
            For each new key, what its old keys lost since the reference
            (their whole count when they are gone), summed over the old
            keys that moved there.
        """
        moved: Counter[Key] = Counter()
        for key, count in allowed_counts.items():
            rule, path, symbol = key
            lost = count - counts.get(key, 0)
            if lost <= 0:
                continue
            for place in self.places.get((path, symbol), ()):
                moved[(rule, *place)] += lost
        return moved

    def is_declared(self, key: Key) -> bool:
        """Return whether a baseline key's definition has a declared move.

        Parameters
        ----------
        key
            The rule, file and definition of a baseline entry.

        Returns
        -------
        bool
            True when the map lists the key's file and definition.
        """
        _, path, symbol = key
        return (path, symbol) in self.places

    def sources_by_target(self) -> dict[str, tuple[str, ...]]:
        """Return, for each new file, the files its definitions came from.

        Returns
        -------
        dict of str to tuple of str
            For each file that a move enters, relative to the root, the
            other files whose definitions moved into it, sorted.
        """
        sources: dict[str, set[str]] = {}
        for old, new in self._pairs():
            if old[0] != new[0]:
                sources.setdefault(new[0], set()).add(old[0])
        return {path: tuple(sorted(old)) for path, old in sources.items()}

    def modules(self) -> dict[str, set[str]]:
        """Return the modules each moved module's definitions went to.

        Returns
        -------
        dict of str to set of str
            For each old module, by dotted name, the new modules.
        """
        targets: dict[str, set[str]] = {}
        for old, new in self._pairs():
            module = module_name_from_path(old[0])
            targets.setdefault(module, set()).add(
                module_name_from_path(new[0])
            )
        return targets

    def translate(self, before: Record, after: Record) -> Record:
        """Return a reference's record with moved modules renamed.

        Parameters
        ----------
        before
            The record at the reference.
        after
            The record now; a one-module setting takes its new value when
            that value is one of the old module's targets.

        Returns
        -------
        Record
            The reference's checks, and its settings with each module name
            replaced by the modules its definitions moved to.
        """
        targets = self.modules()
        settings = dict(before.settings)
        for name in MODULE_LISTS:
            items = as_list(settings.get(name))
            if items is not None:
                renamed = _renamed(map(as_text, items), targets)
                settings[name] = sorted(renamed)
        for name in MODULE_NAMES:
            settings[name] = _one_renamed(
                as_text(settings.get(name)),
                as_text(after.settings.get(name)),
                targets,
            )
        layers = as_list(settings.get(LAYERS))
        if layers is not None:
            settings[LAYERS] = [_layer(layer, targets) for layer in layers]
        names = as_table(settings.get(TYPE_NAMES))
        if names is not None:
            settings[TYPE_NAMES] = {
                self._type_key(key): value for key, value in names.items()
            }
        return Record(before.checks, settings)

    def _pairs(self) -> Iterator[tuple[Place, Place]]:
        """Yield each old place with each of its new places."""
        for old, news in self.places.items():
            for new in news:
                yield old, new

    def _type_key(self, key: str) -> str:
        """Return a ``module.Type`` key with its type's new module."""
        module, _, type_name = key.rpartition(".")
        for (path, symbol), (new_path, _) in self._pairs():
            if symbol == type_name and module_name_from_path(path) == module:
                return f"{module_name_from_path(new_path)}.{type_name}"
        return key


def read_moves(root: Path) -> Moves:
    """Read the move map of a repository; empty when it has none.

    Parameters
    ----------
    root
        The repository root.

    Returns
    -------
    Moves
        The declared moves.

    Raises
    ------
    MovesError
        When the file is not a list of moves with four text fields.
    """
    path = root / MOVES_FILE
    if not path.exists():
        return Moves({})
    try:
        document = as_table(load_json(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise MovesError(
            f"{MOVES_FILE} is not valid JSON: {error}."
        ) from error

    places: dict[Place, tuple[Place, ...]] = {}
    items = None if document is None else as_list(document.get("moves"))
    for item in items or []:
        entry = as_table(item) or {}
        texts = [as_text(entry.get(field)) for field in FIELDS]
        old_path, symbol, new_path, new_symbol = texts
        if (
            old_path is None
            or symbol is None
            or new_path is None
            or new_symbol is None
        ):
            raise MovesError(
                f"{MOVES_FILE} has a move without its four text fields."
            )
        old = (old_path, symbol)
        places[old] = (*places.get(old, ()), (new_path, new_symbol))
    return Moves(places)


def _renamed(
    names: Iterable[str | None], targets: dict[str, set[str]]
) -> set[str]:
    """Return module names with each moved one replaced by its targets."""
    renamed: set[str] = set()
    for name in names:
        if name is not None:
            renamed |= targets.get(name, {name})
    return renamed


def _one_renamed(
    name: str | None, now: str | None, targets: dict[str, set[str]]
) -> str | None:
    """Return a one-module setting renamed to the value it has now."""
    if name is None or name not in targets:
        return name
    if now in targets[name]:
        return now
    return name


def _layer(layer: object, targets: dict[str, set[str]]) -> list[str]:
    """Return one layer's packages renamed, those inside another left out.

    A package moved inside another package of the same layer adds nothing
    the outer package does not already hold. The packages keep their
    order, each moved one's targets in its place.
    """
    renamed: list[str] = []
    for name in map(as_text, as_list(layer) or []):
        if name is None:
            continue
        for package in sorted(targets.get(name, {name})):
            if package not in renamed:
                renamed.append(package)
    outer: list[str] = []
    for package in renamed:
        inside = [other for other in renamed if package.startswith(other)]
        if inside == [package]:
            outer.append(package)
    return outer
