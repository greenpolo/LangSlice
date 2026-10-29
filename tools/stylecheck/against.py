"""``--against REF``: the baselines only shrank and the checks only tightened.

A commit passes when, compared with the reference commit:

- every entry of the checker's baseline is at most its count there, except
  the entries the reference could not have had: a rule it did not run (a
  new check, a newly selected code), errors that a setting made stricter
  since then adds on today's code, and the entries of a definition the
  move map declares moved (:mod:`tools.stylecheck.moves`; the places of
  one old entry share at most what it lost since the reference, except
  that each place inherits a rule that reports once per file);
- the baseline holds no more than the code has: an entry above the run's
  count is pruned, so a fixed error cannot come back unseen;
- the file's record of checks and settings is the one the checker runs
  with now, and no setting is looser than at the reference
  (:mod:`tools.stylecheck.settings`), with the modules the reference's
  settings name read through the move map, unless ``--allow-loosening``
  is given;
- basedpyright's baseline only shrank, per file and rule
  (:mod:`tools.stylecheck.pyright_baseline`); an entry that left a file
  the move map moves definitions out of may reappear once in a file it
  moves them into, matched by its span first and else by its rule.

Only the rows of the move map that hold at the reference and now are
followed: the old definition was there, and the new one is here.

Which version of each baseline is the reference is
:func:`tools.stylecheck.history.reference`'s choice: the file at the
commit, its first committed version when it was created later, or nothing.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from tools.stylecheck.baseline import (
    Baseline,
    CountChange,
    count_errors,
    parse_baseline,
)
from tools.stylecheck.core import Config, Project
from tools.stylecheck.history import Reference, ReferenceKind, reference
from tools.stylecheck.moves import Move, Moves, read_moves
from tools.stylecheck.pyright_baseline import (
    PYRIGHT_BASELINE,
    PyrightChange,
    PyrightEntry,
    parse_pyright_baseline,
    pyright_growth,
)
from tools.stylecheck.run import internal_violations, per_file_rules
from tools.stylecheck.scope import Key
from tools.stylecheck.settings import (
    Record,
    SettingChange,
    differences,
    loosenings,
    recorded_texts,
)


@dataclass(frozen=True)
class Growth:
    """Hold the checker's baseline compared with its reference.

    Parameters
    ----------
    changes
        Each entry above what the reference allows.
    n_adopted
        How many errors entered as a new rule or a stricter setting.
    n_moved
        How many errors entered with a definition whose move the move map
        declares.
    """

    changes: tuple[CountChange, ...]
    n_adopted: int
    n_moved: int


@dataclass(frozen=True)
class AgainstResult:
    """Hold everything ``--against`` reports.

    Parameters
    ----------
    reference
        What the checker's baseline was compared with, in words.
    growth
        The checker's baseline set against its reference.
    unrecorded
        The checks and settings the baseline file records differently
        from the run.
    loosening
        Each setting looser than at the reference.
    loosening_allowed
        Whether ``--allow-loosening`` was given.
    pyright_reference
        What basedpyright's baseline was compared with, in words.
    pyright_changes
        Each file and rule of basedpyright's baseline that grew.
    slack
        Each entry of the checker's baseline above the run's count, which
        the caller fills in from its run.
    unhonoured
        The rows of the move map, live at the reference, that it does not
        follow: their new definition is not in the code.
    """

    reference: str
    growth: Growth
    unrecorded: tuple[str, ...]
    loosening: tuple[SettingChange, ...]
    loosening_allowed: bool
    pyright_reference: str
    pyright_changes: tuple[PyrightChange, ...]
    slack: tuple[CountChange, ...] = ()
    unhonoured: tuple[Move, ...] = ()

    @property
    def fails(self) -> bool:
        """Return whether the commit fails the comparison."""
        loosened = bool(self.loosening) and not self.loosening_allowed
        return bool(
            self.growth.changes
            or self.unrecorded
            or loosened
            or self.pyright_changes
            or self.slack
        )


def check_against(
    ref: str,
    project: Project,
    config: Config,
    baseline: Baseline,
    *,
    record: Record,
    allow_loosening: bool = False,
) -> AgainstResult:
    """Compare both baselines and the settings with a git commit.

    Parameters
    ----------
    ref
        The git commit, branch or tag to compare with.
    project
        The parsed package; its root is the repository, where git runs.
    config
        The checker's settings now.
    baseline
        The checker's baseline file as it is now.
    record
        The checks and settings the checker runs with now.
    allow_loosening
        Accept looser settings, reporting them without failing.

    Returns
    -------
    AgainstResult
        The growth of each baseline and the loosened settings.

    Raises
    ------
    tools.stylecheck.history.HistoryError
        When ``ref`` names no commit, or git fails.
    tools.stylecheck.baseline.BaselineError
        When the reference's baseline cannot be read.
    tools.stylecheck.moves.MovesError
        When the move map cannot be read.
    """
    root = project.root
    version = reference(root, ref, config.baseline)
    live = read_moves(root).live(root, ref)
    moves = live.honoured(ref, project)
    before: Baseline | None = None
    if version.text is not None:
        read = parse_baseline(version.text, f"{ref}:{config.baseline}")
        before = Baseline(read.counts, moves.translate(read.record, record))

    loosening: list[SettingChange] = []
    growth = Growth((), 0, 0)
    if before is not None:
        loosening = loosenings(before.record, record)
    if version.kind is not ReferenceKind.NEW:
        growth = _growth(
            project, config, before, baseline.counts, record, moves
        )

    earlier = None if before is None else before.record
    pyright = reference(root, ref, PYRIGHT_BASELINE)
    pyright_changes = _pyright_changes(root, pyright, earlier, record, moves)
    return AgainstResult(
        reference=version.description,
        growth=growth,
        unrecorded=tuple(differences(baseline.record, record)),
        loosening=tuple(loosening),
        loosening_allowed=allow_loosening,
        pyright_reference=pyright.description,
        pyright_changes=tuple(pyright_changes),
        unhonoured=tuple(row for row in live.rows if row not in moves.rows),
    )


def _growth(
    project: Project,
    config: Config,
    before: Baseline | None,
    counts: dict[Key, int],
    record: Record,
    moves: Moves,
) -> Growth:
    """Return the entries above what the reference baseline allows."""
    allowed_counts = {} if before is None else before.counts
    stricter: dict[Key, int] = {}
    if before is not None and record.differs("stylecheck.", before.record):
        stricter = _stricter_errors(project, config, before.record)
    lent = moves.lent(allowed_counts, counts, per_file=per_file_rules())

    changes: list[CountChange] = []
    n_adopted = n_moved = 0
    for key, count in sorted(counts.items()):
        allowed = allowed_counts.get(key, 0)
        if count <= allowed:
            continue
        if before is not None and not before.record.covers(key):
            n_adopted += count - allowed  # a rule the reference did not run
            continue

        extra = min(count - allowed, stricter.get(key, 0))
        n_adopted += extra
        allowed += extra

        moved = lent.take(key, count - allowed)
        n_moved += moved
        allowed += moved
        if count > allowed:
            changes.append(CountChange(key, allowed, count))
    return Growth(tuple(changes), n_adopted, n_moved)


def _stricter_errors(
    project: Project, config: Config, earlier: Record
) -> dict[Key, int]:
    """Count the errors today's settings add over the reference's settings.

    Both settings run the checker's own checks on today's code, so an
    error that the reference's settings also find is not counted: a
    stricter limit adopts only what the tightening itself reveals.
    """
    now = count_errors(internal_violations(project, config))
    then = count_errors(internal_violations(project, earlier.config(config)))
    return {
        key: count - then.get(key, 0)
        for key, count in now.items()
        if count > then.get(key, 0)
    }


def _pyright_changes(
    root: Path,
    version: Reference,
    earlier: Record | None,
    record: Record,
    moves: Moves,
) -> list[PyrightChange]:
    """Return the files and rules of basedpyright's baseline that grew."""
    path = root / PYRIGHT_BASELINE
    if version.kind is ReferenceKind.NEW or not path.exists():
        return []
    after = parse_pyright_baseline(
        path.read_text(encoding="utf-8"), PYRIGHT_BASELINE
    )
    before: list[PyrightEntry] = []
    if version.text is not None:
        source = f"the reference's {PYRIGHT_BASELINE}"
        before = parse_pyright_baseline(version.text, source)

    sources = moves.sources_by_target()
    return pyright_growth(
        before,
        after,
        is_adopted=functools.partial(_pyright_adopted, earlier, record),
        move_sources=functools.partial(_move_sources, sources),
    )


def _move_sources(
    sources: dict[str, tuple[str, ...]], name: str
) -> tuple[str, ...]:
    """Return the files the move map moves definitions out of into a file.

    The names are basedpyright's, which start with ``./``.
    """
    origins = sources.get(name.removeprefix("./"), ())
    return tuple(f"./{source}" for source in origins)


def _pyright_adopted(
    earlier: Record | None, record: Record, entry: PyrightEntry
) -> bool:
    """Return whether an error belongs to a basedpyright rule adopted since.

    A rule is adopted when its report level or the checking mode changed
    since the earlier record, or when its file was outside the trees the
    earlier record type-checked. Without an earlier record nothing is.
    """
    if earlier is None:
        return False
    for name in (
        f"basedpyright.{entry.code}",
        "basedpyright.typeCheckingMode",
    ):
        if earlier.settings.get(name) != record.settings.get(name):
            return True
    included = recorded_texts(earlier.settings.get("basedpyright.include"))
    if not included:
        return False
    path = PurePosixPath(entry.path.removeprefix("./"))
    return not any(path.is_relative_to(tree) for tree in included)
