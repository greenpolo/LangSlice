"""``--against REF``: the baselines only shrank and the checks only tightened.

A commit passes when, compared with the reference commit:

- every entry of the checker's baseline is at most its count there, except
  the entries the reference could not have had: a rule it did not run (a
  new check, a newly selected code), errors that a setting made stricter
  since then adds on today's code, and the entries of a definition whose
  file was moved (the same rule and definition, at the reference, in a file
  that no longer exists; at most its count there);
- the file's record of checks and settings is the one the checker runs
  with now, and no setting is looser than at the reference
  (:mod:`tools.stylecheck.settings`), unless ``--allow-loosening`` is given;
- basedpyright's baseline only shrank, per file and rule
  (:mod:`tools.stylecheck.pyright_baseline`).

Which version of each baseline is the reference is
:func:`tools.stylecheck.history.reference`'s choice: the file at the
commit, its first committed version when it was created later, or nothing.
"""

from __future__ import annotations

import functools
from collections import Counter
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
from tools.stylecheck.pyright_baseline import (
    PYRIGHT_BASELINE,
    PyrightChange,
    PyrightEntry,
    parse_pyright_baseline,
    pyright_growth,
)
from tools.stylecheck.run import internal_violations
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
        How many errors entered with a definition whose file moved.
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
    """

    reference: str
    growth: Growth
    unrecorded: tuple[str, ...]
    loosening: tuple[SettingChange, ...]
    loosening_allowed: bool
    pyright_reference: str
    pyright_changes: tuple[PyrightChange, ...]

    @property
    def fails(self) -> bool:
        """Return whether the commit fails the comparison."""
        loosened = bool(self.loosening) and not self.loosening_allowed
        return bool(
            self.growth.changes
            or self.unrecorded
            or loosened
            or self.pyright_changes
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
    """
    root = project.root
    version = reference(root, ref, config.baseline)
    before: Baseline | None = None
    if version.text is not None:
        before = parse_baseline(version.text, f"{ref}:{config.baseline}")

    loosening: list[SettingChange] = []
    growth = Growth((), 0, 0)
    if before is not None:
        loosening = loosenings(before.record, record)
    if version.kind is not ReferenceKind.NEW:
        growth = _growth(project, config, before, baseline.counts, record)

    earlier = None if before is None else before.record
    pyright = reference(root, ref, PYRIGHT_BASELINE)
    pyright_changes = _pyright_changes(root, pyright, earlier, record)
    return AgainstResult(
        reference=version.description,
        growth=growth,
        unrecorded=tuple(differences(baseline.record, record)),
        loosening=tuple(loosening),
        loosening_allowed=allow_loosening,
        pyright_reference=pyright.description,
        pyright_changes=tuple(pyright_changes),
    )


def _growth(
    project: Project,
    config: Config,
    before: Baseline | None,
    counts: dict[Key, int],
    record: Record,
) -> Growth:
    """Return the entries above what the reference baseline allows."""
    allowed_counts = {} if before is None else before.counts
    stricter: dict[Key, int] = {}
    if before is not None and record.differs("stylecheck.", before.record):
        stricter = _stricter_errors(project, config, before.record)
    pool = _moved_pool(project.root, allowed_counts, counts)

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

        rule, _, symbol = key
        moved = min(count - allowed, pool[(rule, symbol)])
        pool[(rule, symbol)] -= moved
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


def _moved_pool(
    root: Path, allowed_counts: dict[Key, int], counts: dict[Key, int]
) -> Counter[tuple[str, str]]:
    """Count the reference's errors per rule and definition in gone files.

    An entry counts when its file no longer exists and the baseline no
    longer lists its key, so each error moves once.
    """
    pool: Counter[tuple[str, str]] = Counter()
    for key, count in allowed_counts.items():
        rule, path, symbol = key
        if key not in counts and not (root / path).exists():
            pool[(rule, symbol)] += count
    return pool


def _pyright_changes(
    root: Path, version: Reference, earlier: Record | None, record: Record
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

    return pyright_growth(
        before,
        after,
        is_adopted=functools.partial(_pyright_adopted, earlier, record),
        is_gone=functools.partial(_is_gone, root),
    )


def _is_gone(root: Path, name: str) -> bool:
    """Return whether a file of a baseline no longer exists."""
    return not (root / name).exists()


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
