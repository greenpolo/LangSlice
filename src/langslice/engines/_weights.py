"""Where the engines keep the weights they download, and the one fetch that fills it.

Every engine keeps its weights in a folder of its own under
:data:`CACHE_ROOT` (``~/.cache/langslice/<engine>``), or in the folder a
job's ``cache_folder`` option names. :func:`fetch` downloads one file,
checks its SHA-256 and moves it into place, so a file that is there is whole.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import urllib.request
from pathlib import Path
from typing import IO, cast

logger = logging.getLogger(__name__)

CACHE_ROOT = Path("~/.cache/langslice")


def weights_folder(engine: str, given: str = "") -> Path:
    """Name the folder that keeps an engine's weights.

    Parameters
    ----------
    engine
        The engine's folder name under :data:`CACHE_ROOT`.
    given
        The folder a job's ``cache_folder`` option names; empty is the
        engine's own folder.

    Returns
    -------
    Path
        The folder, ``~`` expanded.
    """
    return Path(given).expanduser() if given else CACHE_ROOT.expanduser() / engine


def sha256_of(path: Path) -> str:
    """Return a file's SHA-256 as hexadecimal text."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fetch(url: str, sha256: str, path: Path, *, timeout_s: float) -> Path:
    """Fetch a file to a path, once.

    The file is downloaded to a partial name beside *path*, checked against
    *sha256* and renamed into place; a failed download leaves nothing at
    *path*.

    Parameters
    ----------
    url
        Where the file is.
    sha256
        The file's SHA-256, hexadecimal.
    path
        Where the file is kept; a file already there with that digest is
        returned as it is.
    timeout_s
        The network timeout, seconds.

    Returns
    -------
    Path
        *path*.

    Raises
    ------
    ValueError
        When the file fetched has another digest.
    OSError
        When the file cannot be fetched or written.
    """
    if path.is_file() and sha256_of(path) == sha256:
        return path
    logger.info("Fetching %s", url)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    opened = cast("IO[bytes]", urllib.request.urlopen(url, timeout=timeout_s))
    with opened as response, partial.open("wb") as out:
        shutil.copyfileobj(response, out, 1 << 20)
    fetched = sha256_of(partial)
    if fetched != sha256:
        partial.unlink()
        raise ValueError(f"the file fetched from {url} has SHA-256 {fetched}, not {sha256}")
    partial.replace(path)
    return path
