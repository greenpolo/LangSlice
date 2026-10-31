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

import logging
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from langslice.job.imports.matching import section_number

logger = logging.getLogger(__name__)

#: Digits a written section number is padded to (``_s001``).
NUMBER_DIGITS = 3
#: PNG compression of the copies: fast, since a submit writes every one.
PNG_COMPRESSION = 1


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
        partial = path.with_name(f".{name}.partial")
        Image.fromarray(np.asarray(pixels, dtype=np.uint8)).save(
            partial, format="PNG", compress_level=PNG_COMPRESSION
        )
        partial.replace(path)
        written.append(path)
    keep = {name for name, _pixels in copies}
    for name in previous:
        base = Path(name).name
        stale = folder / base
        if (
            base in keep
            or stale.suffix.lower() != ".png"
            or not stale.is_file()
        ):
            continue
        try:
            stale.unlink()
        except OSError:
            logger.warning(
                "Could not remove the old copy %s", stale, exc_info=True
            )
    return written
