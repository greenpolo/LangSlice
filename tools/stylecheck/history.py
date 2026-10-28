"""A baseline file as it was at a git commit, for ``--against``.

``--against REF`` compares a baseline with its version at a commit. The
version to compare with is the file at that commit when it exists there.
A file absent at the commit was either deleted at or before it, and then
every entry of the file now is new, or created after it, and then its
first committed version is the reference, so a push that creates the file
and grows it is still caught. A file that no commit holds yet is the
baseline's creation and has nothing to compare with.
"""

from __future__ import annotations

import enum
import subprocess
from dataclasses import dataclass
from pathlib import Path


class HistoryError(Exception):
    """Signal a git reference that names no commit, or a failed git call."""


class ReferenceKind(enum.StrEnum):
    """Say which version of a file a comparison uses."""

    AT_REF = "at the reference"
    CREATED = "as created after the reference"
    DELETED = "deleted at the reference"
    NEW = "not committed yet"


@dataclass(frozen=True)
class Reference:
    """Hold the version of a file that a newer version is compared with.

    Parameters
    ----------
    kind
        Which version it is: at the commit, the first one after it, none
        because the file was deleted, or none because it is new.
    text
        The file's text at that version; None for a deleted or new file.
    description
        What the file is compared with, in words, for the report.
    """

    kind: ReferenceKind
    text: str | None
    description: str


def reference(root: Path, ref: str, relpath: str) -> Reference:
    """Find the version of a file to compare its current version with.

    Parameters
    ----------
    root
        The repository root, where git runs.
    ref
        A git commit, branch or tag.
    relpath
        The file relative to ``root``, with forward slashes.

    Returns
    -------
    Reference
        The file at ``ref``; or, when it is absent there, nothing (deleted
        at or before ``ref``), its first version committed after ``ref``,
        or nothing (never committed).

    Raises
    ------
    HistoryError
        When ``ref`` names no commit, or git fails.
    """
    commit = _commit(root, ref)
    text = show(root, commit, relpath)
    if text is not None:
        return Reference(ReferenceKind.AT_REF, text, f"{relpath} at {ref}")
    if _existed_before(root, commit, relpath):
        description = f"{ref}, where {relpath} is deleted"
        return Reference(ReferenceKind.DELETED, None, description)

    # absent at the reference and before it: created after the reference
    creation = _creating_commit(root, commit, relpath)
    if creation is None:
        description = f"nothing: {relpath} is new since {ref}"
        return Reference(ReferenceKind.NEW, None, description)
    description = f"{relpath} as created in {creation[:12]}, after {ref}"
    created = show(root, creation, relpath)
    return Reference(ReferenceKind.CREATED, created, description)


def show(root: Path, commit: str, relpath: str) -> str | None:
    """Return a file's text at a commit.

    Parameters
    ----------
    root
        The repository root, where git runs.
    commit
        A commit's hash or any name git resolves.
    relpath
        The file relative to ``root``, with forward slashes.

    Returns
    -------
    str or None
        The file's text, or None when the commit does not hold the file.
    """
    done = _git(root, ["show", f"{commit}:./{relpath}"])
    if done.returncode != 0:
        return None
    return done.stdout


def _git(root: Path, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one git command in ``root`` and return what it printed."""
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        capture_output=True,
        text=True,
        check=False,
    )


def _commit(root: Path, ref: str) -> str:
    """Return the full hash of the commit ``ref`` names."""
    done = _git(
        root, ["rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"]
    )
    if done.returncode != 0:
        raise HistoryError(f"{ref!r} names no commit in {root}.")
    return done.stdout.strip()


def _existed_before(root: Path, commit: str, relpath: str) -> bool:
    """Return whether any ancestor of ``commit`` had the file."""
    done = _git(root, ["log", "--format=%H", "-n", "1", commit, "--", relpath])
    if done.returncode != 0:
        raise HistoryError(f"git log failed: {done.stderr.strip()}")
    return bool(done.stdout.strip())


def _creating_commit(root: Path, commit: str, relpath: str) -> str | None:
    """Return the first commit after ``commit`` that adds the file."""
    arguments = ["log", "--format=%H", "--diff-filter=A", "--reverse"]
    done = _git(root, [*arguments, f"{commit}..HEAD", "--", relpath])
    if done.returncode != 0:
        raise HistoryError(f"git log failed: {done.stderr.strip()}")
    hashes = done.stdout.split()
    if not hashes:
        return None
    return hashes[0]
