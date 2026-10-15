"""Volumes derived from an atlas, kept on disk between processes.

BrainGlobe keeps what it downloads (``~/.brainglobe``) and nothing derived
from it, so every LangSlice process rebuilt the same volumes from the same
atlas: the template's cubic spline coefficients (``core.oblique``) and the
annotation as dense indices into its sorted ids. :func:`cached_volume`
keeps each one as a ``.npy`` file the first time it is built and opens it
memory-mapped afterwards: nothing is read until a plane touches it, and
processes reading the same file share the operating system's copy.

Where: ``LANGSLICE_ATLAS_CACHE`` (a folder, or ``off``), else
``~/.langslice/atlas-cache``; one folder per atlas name, BrainGlobe version
and :data:`FORMAT`, so a new atlas version or a change in what is stored is
never matched against an old file; the first write of a version
removes its folders of an older format. Atlases without a version (test and host
fakes) and atlases past :data:`MAX_VOXELS` are not cached on disk. The cache
only ever saves time: a file that cannot be written or read is built in
memory, as without a cache, and a warning is logged.
"""

from __future__ import annotations

import logging
import math
import os
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

#: The setting naming the cache folder; ``off`` (or ``0``, ``none``, empty)
#: turns the disk cache off.
ENV = "LANGSLICE_ATLAS_CACHE"
_OFF = {"", "0", "off", "none", "false", "no"}

#: Bumped whenever what a cached file holds changes, so old files are never read.
FORMAT = 1

#: Atlases with more voxels than this are not cached on disk: one float32
#: volume of the 10 um Allen atlas (1.2 billion voxels) is 4.8 GB.
MAX_VOXELS = 500_000_000


def cache_root() -> Path | None:
    """The cache folder, or None when the disk cache is off."""
    value = os.environ.get(ENV)
    if value is None:
        return Path.home() / ".langslice" / "atlas-cache"
    if value.strip().lower() in _OFF:
        return None
    return Path(value).expanduser()


def atlas_folder(atlas: Any) -> Path | None:
    """This atlas's folder in the cache; None when it is not cached on disk."""
    root = cache_root()
    name = getattr(atlas, "atlas_name", None)
    metadata = getattr(atlas, "metadata", None)
    version = metadata.get("version") if isinstance(metadata, dict) else None
    shape = getattr(atlas, "shape", None)
    if root is None or not name or not version or not shape:
        return None
    if math.prod(int(n) for n in shape) > MAX_VOXELS:
        return None
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{name}_v{version}")
    return root / safe / f"format{FORMAT}"


def _remove_older_formats(folder: Path) -> None:
    """Delete the ``format<N>`` folders beside *folder* (this atlas version's
    cache) other than *folder* itself: files of a :data:`FORMAT` nothing reads
    any more. Only folders named ``format<digits>`` are touched."""
    parent = folder.parent
    if not parent.is_dir():
        return
    for old in parent.iterdir():
        if old != folder and old.is_dir() and re.fullmatch(r"format\d+", old.name):
            shutil.rmtree(old, ignore_errors=True)
            logger.info("atlas cache: removed %s (an older format)", old)


def cached_volume(atlas: Any, name: str, build: Callable[[], np.ndarray]) -> np.ndarray:
    """The volume *name* of *atlas*: read memory-mapped from the cache, else
    *build* () and saved there. Read only either way.

    *name* says what the array is (``template_spline3``); its file is written
    under a temporary name and renamed into place, so a reader never sees a
    partial file and two processes building it at once both end well.
    """
    folder = atlas_folder(atlas)
    if folder is None:
        return build()
    path = folder / (re.sub(r"[^A-Za-z0-9._-]", "_", name) + ".npy")
    try:
        return np.load(path, mmap_mode="r")
    except FileNotFoundError:
        pass
    except (OSError, ValueError) as exc:  # unreadable or truncated: rebuilt below
        logger.warning("atlas cache: rebuilding %s (%s)", path, exc)
    array = build()
    temporary = path.with_name(f"{path.name}.{os.getpid()}.part")
    try:
        if not folder.exists():
            _remove_older_formats(folder)
        folder.mkdir(parents=True, exist_ok=True)
        with open(temporary, "wb") as handle:
            np.save(handle, np.ascontiguousarray(array))
        os.replace(temporary, path)
    except OSError as exc:
        logger.warning("atlas cache: could not save %s (%s)", path, exc)
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return array
    return np.load(path, mmap_mode="r")


__all__ = ["ENV", "FORMAT", "MAX_VOXELS", "atlas_folder", "cache_root", "cached_volume"]
