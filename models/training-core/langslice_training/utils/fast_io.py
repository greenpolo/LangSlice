"""Detect when a training run's hot I/O paths sit on a slow host bind.

On this project's Docker-Desktop-on-WSL2 setup, Windows bind mounts surface
inside the container as ``9p`` (or ``drvfs``/``virtiofs``/gRPC-FUSE depending on
Docker's file-sharing backend), while the durable named volumes
(``out/cache_fast``, ``data/datasets``, ``/models``, ``~/.brainglobe``, the HF
cache) are native ``ext4``. Reading or writing a hot path over the 9p bind is an
order of magnitude slower and has repeatedly bottlenecked runs.

The policy is: heavy training I/O should live on the fast ext4 volume. This
module gives the trainers a cheap startup check that *flags* (or, in strict mode,
*refuses*) any hot path that resolves onto a slow filesystem — so a forgotten
``--output-dir`` or cache flag can't silently throttle a run.

The check is data-injectable (``mounts`` arg) so it is unit-testable off-box, and
degrades to a no-op when the mount table can't be read (e.g. a non-Linux host) —
better silent than crying wolf when we genuinely can't tell.
"""
from __future__ import annotations

import logging
import os
import posixpath
from collections.abc import Iterable, Mapping
from pathlib import Path

logger = logging.getLogger("langslice_training.fast_io")

# Filesystem types that mean "slow host/network bind" under Docker Desktop.
# Anything NOT in this set (ext4/xfs/btrfs/overlay/tmpfs/zfs/...) is treated as
# fast/local. We flag by slow-list rather than fast-list so an unrecognised
# local fs is never misreported as slow.
SLOW_FSTYPES = frozenset(
    {
        "9p",
        "9pfs",
        "drvfs",
        "drvfs2",
        "virtiofs",
        "fuse",
        "fuseblk",
        "fuse.grpcfs",
        "fuse.grpc-fuse",
        "cifs",
        "smbfs",
        "smb3",
        "ntfs",
        "ntfs3",
        "prjfs",
    }
)

# Set this env var truthy to upgrade the warning into a hard failure.
STRICT_ENV = "LANGSLICE_STRICT_FAST_IO"

_TRUTHY = frozenset({"1", "true", "yes", "on"})


def read_mounts(mounts_file: str | os.PathLike = "/proc/mounts") -> list[tuple[str, str]]:
    """Return ``[(mountpoint, fstype), ...]`` parsed from a mounts table.

    Returns an empty list if the file can't be read (non-Linux host, etc.).
    Each /proc/mounts line is ``spec mountpoint fstype options dump pass``; we
    take fields 2 and 3. (Our mountpoints contain no spaces, so the naive split
    is safe; the noisy options field — full of ``;`` and ``=`` for 9p — is
    ignored.)
    """
    try:
        text = Path(mounts_file).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    mounts: list[tuple[str, str]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3:
            mounts.append((parts[1], parts[2]))
    return mounts


def filesystem_for(
    path: str | os.PathLike,
    mounts: Iterable[tuple[str, str]] | None = None,
    *,
    cwd: str | os.PathLike | None = None,
) -> str | None:
    """Filesystem type of the nearest enclosing mountpoint for ``path``.

    Uses longest-prefix matching, so a fast volume nested inside a slow bind
    (e.g. ``out/cache_fast`` ext4 under ``out`` 9p) is resolved correctly.
    Returns ``None`` when the mount table is empty/unknown.
    """
    if mounts is None:
        mounts = read_mounts()
    mounts = list(mounts)
    if not mounts:
        return None
    # Normalise with posixpath (NOT os.path) so absolute container paths match
    # the POSIX mount table identically on any host the tests run on — os.path
    # on Windows would rewrite "/workspace/..." to "C:\workspace\...". Relative
    # paths resolve against `cwd` (default: process CWD); the trainer's CWD is on
    # the 9p bind, so a relative output-dir must resolve there to be caught.
    raw = os.fspath(path).replace("\\", "/")
    if posixpath.isabs(raw):
        target = posixpath.normpath(raw)
    else:
        base = (os.fspath(cwd) if cwd is not None else os.getcwd()).replace("\\", "/")
        target = posixpath.normpath(posixpath.join(base, raw))
    best_mp: str | None = None
    best_fs: str | None = None
    for mountpoint, fstype in mounts:
        mp = posixpath.normpath(mountpoint)
        prefix = mp.rstrip("/") + "/"
        if target == mp or target.startswith(prefix):
            if best_mp is None or len(mp) > len(best_mp):
                best_mp, best_fs = mp, fstype
    return best_fs


def is_slow_path(
    path: str | os.PathLike,
    mounts: Iterable[tuple[str, str]] | None = None,
    *,
    cwd: str | os.PathLike | None = None,
) -> bool:
    """True iff ``path`` resolves onto a known-slow filesystem (9p/virtiofs/...).

    Unknown/unreadable mount table -> False (we can't tell; don't warn).
    """
    fstype = filesystem_for(path, mounts, cwd=cwd)
    return fstype in SLOW_FSTYPES


def _strict_enabled(strict: bool | None) -> bool:
    if strict is not None:
        return strict
    return os.environ.get(STRICT_ENV, "").strip().lower() in _TRUTHY


def warn_if_slow_io(
    paths: Mapping[str, str | os.PathLike | None],
    *,
    mounts: Iterable[tuple[str, str]] | None = None,
    strict: bool | None = None,
    cwd: str | os.PathLike | None = None,
    fast_hint: str = "out/cache_fast (the WSL2 ext4 volume)",
) -> list[tuple[str, str, str]]:
    """Flag any hot ``role -> path`` that lives on a slow filesystem.

    Emits a single consolidated WARNING naming each offending path and its
    filesystem. If ``strict`` (or env ``LANGSLICE_STRICT_FAST_IO`` is truthy),
    raises ``RuntimeError`` instead. ``None`` paths are skipped. Returns the list
    of ``(role, path, fstype)`` that were flagged (empty == all clear).
    """
    if mounts is None:
        mounts = read_mounts()
    mounts = list(mounts)
    flagged: list[tuple[str, str, str]] = []
    for role, path in paths.items():
        if path is None:
            continue
        fstype = filesystem_for(path, mounts, cwd=cwd)
        if fstype in SLOW_FSTYPES:
            flagged.append((role, str(path), fstype))
    if not flagged:
        return flagged
    detail = "\n".join(
        f"  - {role}: {path}  (filesystem: {fstype})" for role, path, fstype in flagged
    )
    message = (
        "Slow filesystem detected for training I/O path(s) — these are on a "
        "Windows/host bind, not the fast WSL2 ext4 volume:\n"
        f"{detail}\n"
        f"This will bottleneck the run. Put heavy I/O under {fast_hint}. "
        f"(Set {STRICT_ENV}=1 to make this a hard error.)"
    )
    if _strict_enabled(strict):
        raise RuntimeError(message)
    logger.warning(message)
    return flagged
