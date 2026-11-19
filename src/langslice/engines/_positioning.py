"""What the two positioning engines (DeepSlice and Brainways) share.

Their options, the images they write for a call, the lists a call is given,
the rounding of a reply's numbers, what a position write answers and the
sentences that point to ``look``'s positioning view.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar, cast

import numpy as np
from PIL import Image

from langslice.plugin_api import RefusalCode, WriteResult

LONG_EDGE = 3072
SEE_WRITTEN = "The positioning view of look shows the sections at these positions."

Kind = TypeVar("Kind", bool, int, str)


def option(config: Mapping[str, object], name: str, default: Kind) -> Kind:
    """Read one option as the kind of its default.

    Parameters
    ----------
    config
        The plugin's entry of the job's ``plugin_config``.
    name
        The option.
    default
        The value when the option is not given; its type is the option's.

    Returns
    -------
    bool, int or str
        The value given, or *default*.

    Raises
    ------
    ValueError
        When the value is of another type (a number is no ``bool``, a
        ``bool`` is no number).
    """
    value = config.get(name, default)
    if type(value) is not type(default):
        raise ValueError(f"{name} is a {type(default).__name__}; got {value!r}")
    return cast("Kind", value)


def long_edge(config: Mapping[str, object]) -> int:
    """Read the ``long_edge`` option: a positive number of pixels."""
    pixels = option(config, "long_edge", LONG_EDGE)
    if pixels < 1:
        raise ValueError(f"long_edge is a number of pixels, at least 1; got {pixels}")
    return pixels


def listed(names: Sequence[str] | None) -> list[str] | None:
    """Copy a list argument, keeping None as not given."""
    return None if names is None else list(names)


def rounded(value: float | None, digits: int) -> float | None:
    """Round a number for a reply (``+ 0.0`` leaves no ``-0.0``); None stays."""
    return None if value is None else round(float(value), digits) + 0.0


def write_png(array: np.ndarray, folder: Path, section: str) -> Path:
    """Write an image handed to an engine, as ``<folder>/images/<section>.png``.

    Parameters
    ----------
    array
        The pixels, uint8, grey or RGB.
    folder
        The call's folder.
    section
        The section's filename.

    Returns
    -------
    Path
        The file.
    """
    name = section if section.lower().endswith(".png") else f"{section}.png"
    path = folder / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, np.uint8)).save(path)
    return path


def row_errors(result: WriteResult | None) -> dict[str, str]:
    """Return each section a write left out, with its code.

    Parameters
    ----------
    result
        What the write answered; None when nothing was written.

    Returns
    -------
    dict
        The refusal code of each row that did not land; empty for a write
        refused whole.
    """
    if result is None or not result.ok:
        return {}
    rows = cast("list[dict[str, str]]", result.rows)
    landed = set(result.written)
    return {
        row["id"]: str(row.get("error", RefusalCode.PRECONDITION))
        for row in rows
        if row["id"] not in landed
    }


@dataclass(frozen=True)
class Clamped:
    """Hold a written position the atlas range clamped.

    Parameters
    ----------
    id
        The section's filename.
    asked_mm
        The position proposed, millimetres.
    used_mm
        The position written, millimetres.
    """

    id: str
    asked_mm: float
    used_mm: float


def clamped_rows(result: WriteResult | None) -> list[Clamped]:
    """Return the positions a write clamped into the atlas range.

    Parameters
    ----------
    result
        What the write answered; None when nothing was written.

    Returns
    -------
    list of Clamped
        One per row that says it was clamped; empty for a write refused whole.
    """
    if result is None or not result.ok:
        return []
    rows = cast("list[dict[str, str | float]]", result.rows)
    return [
        Clamped(str(row["id"]), float(row["clamped_from_mm"]), float(row["position_mm"]))
        for row in rows
        if "clamped_from_mm" in row
    ]


def clamped_reply(clamped: Sequence[Clamped]) -> list[dict[str, str | float]]:
    """Word clamped positions as a reply's ``clamped`` list."""
    return [{"id": row.id, "asked_mm": row.asked_mm, "used_mm": row.used_mm} for row in clamped]


def view_note(landed: bool, proposal: str) -> str:
    """Return the sentence a reply ends with when it carries no picture.

    Parameters
    ----------
    landed
        Whether the call wrote positions.
    proposal
        The tool's sentence for a call that wrote none.

    Returns
    -------
    str
        :data:`SEE_WRITTEN` after a write, else *proposal*.
    """
    return SEE_WRITTEN if landed else proposal
