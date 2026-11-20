"""The section images the QUINT export names, and their section numbers.

QuickNII and VisuAlign open the images a QUINT file names from the folder
that holds the file, and read PNG and JPEG only; Nutil and PyNutil pair a
segmentation with a section by the ``_sNNN`` number in its filename (Nutil
finds the ``_sNNN`` text in the entry's filename, PyNutil compares the number
with the entry's ``nr``). So the export writes, beside ``quicknii.json`` and
``visualign.json``, one PNG per section named with its number, and the files
name those copies:

- the number (``nr``): the ``_sNNN`` number in each section's filename when
  every section of the stack has exactly one and no two share it (the
  QuickNII and DeepSlice convention), else the section's place in the
  stack, from 1 (:func:`section_numbers`); both are the stack's, so a
  section keeps its number whichever sections an export leaves out;
- the copy: ``<stem>_sNNN.png``, NNN zero-padded to three digits, or
  ``<stem>.png`` when the stem already carries that number
  (:func:`copy_filename`);
- the pixels: the section's ``preprocessed`` image (the 8-bit gray image
  every picture draws when it names none, so what the agent looked at) at
  the working copy's size, in the file's own orientation. The files keep
  the image file's ``width`` and ``height``: QuickNII and VisuAlign scale an
  image to them, and the anchoring and markers are fractions of them.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from langslice.core.shared_files import atomic_file

logger = logging.getLogger(__name__)

QUICKNII_FILE = "quicknii.json"
VISUALIGN_FILE = "visualign.json"
SECTION_NUMBER = re.compile(r"_s(\d+)")
NUMBER_DIGITS = 3
PNG_COMPRESSION = 1


def base_name(filename: str) -> str:
    """The last path component of *filename* (``/`` and ``\\`` both separate)."""
    return re.split(r"[\\/]", filename)[-1]


def section_number(filename: str) -> int | None:
    """The one ``_sNNN`` section number in *filename*'s base name, else None."""
    found = SECTION_NUMBER.findall(os.path.splitext(base_name(filename))[0])
    return int(found[0]) if len(found) == 1 else None


def section_numbers(filenames: Sequence[str]) -> list[int]:
    """Number the exported sections.

    Parameters
    ----------
    filenames
        The image filenames of every section of the stack, in its order.

    Returns
    -------
    list of int
        Each filename's ``_sNNN`` number when every one has exactly one and
        they are all different, else 1, 2, ... in the stack's order.
    """
    parsed = [section_number(name) for name in filenames]
    numbers = [number for number in parsed if number is not None]
    if len(numbers) == len(parsed) and len(set(numbers)) == len(numbers):
        return numbers
    return list(range(1, len(filenames) + 1))


def copy_filename(filename: str, number: int) -> str:
    """Name the PNG copy of a section image.

    Parameters
    ----------
    filename
        The section's image filename.
    number
        Its section number in the export.

    Returns
    -------
    str
        ``<stem>.png`` when the stem's one ``_sNNN`` is *number*, else
        ``<stem>_sNNN.png`` with NNN zero-padded to three digits.
    """
    stem = Path(filename).stem
    if section_number(filename) == number:
        return f"{stem}.png"
    return f"{stem}_s{number:0{NUMBER_DIGITS}d}.png"


def write_copies(
    folder: Path,
    copies: Sequence[tuple[str, np.ndarray]],
    previous: Iterable[str],
) -> list[Path]:
    """Write the PNG copies and remove the ones an earlier export left.

    Parameters
    ----------
    folder
        The exports folder, which holds the QUINT files.
    copies
        ``(copy filename, 8-bit image)`` per section.
    previous
        The copy filenames the folder's QUINT file named before this
        export; those not written now are removed.

    Returns
    -------
    list of Path
        The files written.
    """
    written: list[Path] = []
    for name, pixels in copies:
        path = folder / name
        with atomic_file(path) as handle:
            Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(
                handle, format="PNG", compress_level=PNG_COMPRESSION
            )
        written.append(path)
    keep = {name for name, _pixels in copies}
    for name in previous:
        base = Path(name).name
        stale = folder / base
        if base in keep or stale.suffix.lower() != ".png" or not stale.is_file():
            continue
        try:
            stale.unlink()
        except OSError:
            logger.warning("Could not remove the old copy %s", stale, exc_info=True)
    return written


def quint_copies(exports_dir: Path) -> list[str]:
    """Return the image filenames the folder's ``quicknii.json`` names.

    Parameters
    ----------
    exports_dir
        The job's exports folder.

    Returns
    -------
    list of str
        The ``filename`` of each slice (its last path component): the PNG
        copies an export wrote beside it. Empty when there is no readable
        file.
    """
    try:
        document = json.loads((exports_dir / QUICKNII_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    slices = document.get("slices") if isinstance(document, dict) else None
    if not isinstance(slices, list):
        return []
    names: list[str] = []
    for entry in slices:
        name = entry.get("filename") if isinstance(entry, dict) else None
        if isinstance(name, str) and Path(name).name:
            names.append(Path(name).name)
    return names
