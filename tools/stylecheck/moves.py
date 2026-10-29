"""The declared moves of definitions, which ``--against`` follows.

A commit that moves or renames code changes the files, and sometimes the
names, of baseline entries. ``tools/stylecheck/moves.json`` lists each
move as an old file and definition and a new file and definition::

    {"moves": [
      {"path": "src/a.py", "symbol": "f",
       "new_path": "src/b.py", "new_symbol": "f"}
    ]}

A rename inside a file is a row with the same ``path`` and ``new_path``.
The map is the only way a baseline entry follows its code: an entry in a
file or under a name the map does not lead to is new.

``--against REF`` honours a row only when, at the reference, its ``path``
held its ``symbol`` (``<module>`` for module-level code) and, now, its
``new_path`` holds its ``new_symbol``. A row whose old definition is not
at the reference (its ``path`` absent there, or without the ``symbol``)
is inert, and ``--prune --against REF`` removes it, so the rows of a move
that has landed leave the map; a live row whose new definition is missing
is reported. An entry of the new file and
definition is accepted up to what the old one lost since the reference
(its whole count there when it is no longer in the baseline). A
definition is listed once per new place, and its places share what it
lost: a module split into several files lists its module level once per
part, and the parts together may have what the old module level had,
except for a rule that reports at most one error per file (such as
``format``), whose error each part may inherit once the old file lost it.
The
settings that name modules (the boundary modules, the command modules,
the renderer, the layers) are read at the reference through the same map,
so a module that moved is not a loosened setting.
"""

from __future__ import annotations

import ast
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from tools.stylecheck.core import Module, Project, module_name_from_path
from tools.stylecheck.history import show, tracked_files
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

#: The symbol of a module's code outside every function and class.
MODULE_LEVEL = "<module>"


class MovesError(Exception):
    """Signal a move map that cannot be read."""


#: A definition: its file relative to the root and its qualified name.
Place = tuple[str, str]


@dataclass(frozen=True)
class Move:
    """Hold one row of the move map.

    Parameters
    ----------
    path
        The old file, relative to the repository root.
    symbol
        The old definition's qualified name, or ``<module>``.
    new_path
        The new file, relative to the repository root.
    new_symbol
        The new definition's qualified name, or ``<module>``.
    """

    path: str
    symbol: str
    new_path: str
    new_symbol: str

    @property
    def old(self) -> Place:
        """Return the old file and definition."""
        return (self.path, self.symbol)

    @property
    def new(self) -> Place:
        """Return the new file and definition."""
        return (self.new_path, self.new_symbol)


@dataclass(frozen=True)
class Moves:
    """Hold the moves of one commit.

    Parameters
    ----------
    rows
        The moves, in the file's order.
    kept
        The modules, by dotted name, that exist both at the reference and
        now; a module named in a setting stays named when it is kept.
    """

    rows: tuple[Move, ...]
    kept: frozenset[str] = frozenset()

    def live(self, root: Path, ref: str) -> Moves:
        """Return the rows whose old definition is at a commit.

        Parameters
        ----------
        root
            The repository root, where git runs.
        ref
            The reference commit.

        Returns
        -------
        Moves
            The rows whose ``path`` held ``symbol`` at ``ref``; the others
            are inert there.
        """
        present = set(tracked_files(root, ref))
        then: dict[str, frozenset[str]] = {}
        for path in {row.path for row in self.rows if row.path in present}:
            then[path] = _symbols_at(root, ref, path)
        rows = tuple(
            row
            for row in self.rows
            if row.symbol in then.get(row.path, frozenset())
        )
        return Moves(rows, self.kept)

    def honoured(self, ref: str, project: Project) -> Moves:
        """Return the live rows whose new definition is in the code.

        Parameters
        ----------
        ref
            The reference commit, at which the rows are live.
        project
            The parsed package now, where each new definition must be.

        Returns
        -------
        Moves
            The rows whose ``new_path`` holds ``new_symbol`` now, with the
            modules kept since ``ref``.
        """
        before = set(tracked_files(project.root, ref))
        now = {module.relpath: module for module in project.modules.values()}
        rows = tuple(
            row
            for row in self.rows
            if row.new_path in now
            and row.new_symbol in _symbols(now[row.new_path])
        )
        kept = frozenset(
            module.dotted
            for module in project.modules.values()
            if module.relpath in before
        )
        return Moves(rows, kept)

    def lent(
        self,
        allowed_counts: dict[Key, int],
        counts: dict[Key, int],
        *,
        per_file: frozenset[str] = frozenset(),
    ) -> MovedErrors:
        """Return what the reference's moved entries may lend their places.

        Parameters
        ----------
        allowed_counts
            The baseline's counts at the reference.
        counts
            The baseline's counts now.
        per_file
            The rules that report at most one error per file; each place
            of a module level moved out of a file inherits its error.

        Returns
        -------
        MovedErrors
            For each old key, what it lost since the reference (its whole
            count when it is gone), shared by the keys its moves name.
        """
        lost: Counter[Key] = Counter()
        for key, count in allowed_counts.items():
            if count > counts.get(key, 0):
                lost[key] = count - counts.get(key, 0)
        origins: dict[Place, list[Place]] = {}
        for row in self.rows:
            origins.setdefault(row.new, []).append(row.old)
        return MovedErrors(lost, origins, per_file)

    def sources_by_target(self) -> dict[str, tuple[str, ...]]:
        """Return, for each new file, the files its definitions came from.

        Returns
        -------
        dict of str to tuple of str
            For each file that a move enters, relative to the root, the
            other files whose definitions moved into it, sorted.
        """
        sources: dict[str, set[str]] = {}
        for row in self.rows:
            if row.path != row.new_path:
                sources.setdefault(row.new_path, set()).add(row.path)
        return {path: tuple(sorted(old)) for path, old in sources.items()}

    def modules(self) -> dict[str, set[str]]:
        """Return the modules each moved module's definitions went to.

        Returns
        -------
        dict of str to set of str
            For each old module, by dotted name, the new modules, and the
            old module itself when it is kept.
        """
        targets: dict[str, set[str]] = {}
        for row in self.rows:
            module = module_name_from_path(row.path)
            new_modules = targets.setdefault(module, set())
            new_modules.add(module_name_from_path(row.new_path))
            if module in self.kept:
                new_modules.add(module)
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
            replaced by the modules its definitions moved to; a package's
            name by those of every module inside it, and itself while a
            module inside it is kept.
        """
        targets = self.modules()
        settings = dict(before.settings)
        for name in MODULE_LISTS:
            items = as_list(settings.get(name))
            if items is not None:
                renamed = self._renamed(map(as_text, items), targets)
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

    def _renamed(
        self, names: Iterable[str | None], targets: dict[str, set[str]]
    ) -> set[str]:
        """Return module names with each moved one replaced by its targets.

        A name covers its module and every module inside it, so a package
        stands for the targets of each moved module inside it, and for
        itself while a module inside it is kept.
        """
        renamed: set[str] = set()
        for name in names:
            if name is None:
                continue
            inside = [old for old in targets if _is_within(old, name)]
            if not inside:
                renamed.add(name)
                continue
            for old in inside:
                renamed |= targets[old]
            if any(_is_within(module, name) for module in self.kept):
                renamed.add(name)
        return renamed

    def _type_key(self, key: str) -> str:
        """Return a ``module.Type`` key with its type's new module."""
        module, _, type_name = key.rpartition(".")
        for row in self.rows:
            if (
                row.symbol == type_name
                and module_name_from_path(row.path) == module
            ):
                return f"{module_name_from_path(row.new_path)}.{type_name}"
        return key


@dataclass
class MovedErrors:
    """Hold what moved entries lost, lent once to the places they moved to.

    Parameters
    ----------
    lost
        For each key of the reference, how many errors it lost and has not
        lent yet.
    origins
        For each new file and definition, the old ones the map moves there.
    per_file
        The rules that report at most one error per file: a file's error
        is the state of all its code, so each place it moved to may have
        it, one each, without using it up.
    """

    lost: Counter[Key]
    origins: dict[Place, list[Place]] = field(default_factory=dict)
    per_file: frozenset[str] = frozenset()

    def take(self, key: Key, excess: int) -> int:
        """Lend up to ``excess`` errors to a new key from its old keys.

        Parameters
        ----------
        key
            The rule, file and definition of an entry above its allowance.
        excess
            How many errors the entry has above its allowance.

        Returns
        -------
        int
            How many the old keys lent; each old key's loss is spent once,
            whichever of its places takes it.
        """
        rule, path, symbol = key
        origins = self.origins.get((path, symbol), [])
        if rule in self.per_file:
            inherited = any(
                self.lost[(rule, old_path, old_symbol)] > 0
                for old_path, old_symbol in origins
            )
            return min(excess, 1) if inherited else 0
        taken = 0
        for old_path, old_symbol in origins:
            source = (rule, old_path, old_symbol)
            lent = min(excess - taken, self.lost[source])
            self.lost[source] -= lent
            taken += lent
        return taken


def read_moves(root: Path) -> Moves:
    """Read the move map of a repository; empty when it has none.

    Parameters
    ----------
    root
        The repository root.

    Returns
    -------
    Moves
        The rows of the file, none of them checked yet.

    Raises
    ------
    MovesError
        When the file is not a list of moves with four text fields.
    """
    path = root / MOVES_FILE
    if not path.exists():
        return Moves(())
    try:
        document = as_table(load_json(path.read_text(encoding="utf-8")))
    except json.JSONDecodeError as error:
        raise MovesError(
            f"{MOVES_FILE} is not valid JSON: {error}."
        ) from error

    rows: list[Move] = []
    items = None if document is None else as_list(document.get("moves"))
    for item in items or []:
        entry = as_table(item) or {}
        texts = [as_text(entry.get(name)) for name in FIELDS]
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
        rows.append(Move(old_path, symbol, new_path, new_symbol))
    return Moves(tuple(rows))


def write_moves(root: Path, moves: Moves) -> None:
    """Save the move map, one row per line, in the rows' order.

    Parameters
    ----------
    root
        The repository root.
    moves
        The rows to keep; an empty map is written as an empty list.
    """
    lines = [
        json.dumps(
            dict(zip(FIELDS, (*row.old, *row.new), strict=True)),
            ensure_ascii=False,
        )
        for row in moves.rows
    ]
    body = "\n" + ",\n".join(lines) + "\n" if lines else ""
    text = '{\n"moves": [' + body + "]\n}\n"
    (root / MOVES_FILE).write_text(text, encoding="utf-8")


def _symbols(module: Module) -> frozenset[str]:
    """Return the definitions of a module, its module level included."""
    names = {definition.qualname for definition in module.definitions}
    return frozenset({MODULE_LEVEL, *names})


def _symbols_at(root: Path, ref: str, path: str) -> frozenset[str]:
    """Return the definitions of a file at a commit; none when unparsed."""
    text = show(root, ref, path)
    if text is None:
        return frozenset()
    try:
        tree = ast.parse(text, filename=path)
    except SyntaxError:
        return frozenset()
    dotted = module_name_from_path(path)
    return _symbols(Module(root / path, path, dotted, text, tree))


def _is_within(module: str, name: str) -> bool:
    """Return whether a module is the named module or inside it."""
    return module == name or module.startswith(f"{name}.")


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
