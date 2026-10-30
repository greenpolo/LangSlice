"""The atlas regions a fit is restricted to, and its picture's zoom on them.

:func:`region_entries` checks a call's ``restrict_to`` (or ``exclude``)
against the atlas, the check every fit tool and a script's fit make.
:func:`region_box` is the regions' bounding box on a section's canvas, which
an ``elastix_affine`` row carries and a restricted fit's picture zooms to
when that magnifies it enough (:func:`zoom_window`).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from langslice.core.atlas.sides import has_sides, split_side
from langslice.core.fits.deformable.plane_images import (
    regions_mask,
    resolve_entries,
)
from langslice.core.pictures.placement import stage
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections.damage import normalized_entries
from langslice.tools.transforms import KNOB_DEFAULTS

if TYPE_CHECKING:
    from langslice.core.stack.section import Section
    from langslice.core.stack.stack import Stack
    from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)


def region_entries(
    workspace: Workspace, state: Stack, entries: Any
) -> tuple[str, ...]:
    """*entries* (region acronyms, names or ids, ``"CTX:left"`` for one side)
    checked against the atlas and normalized. Refused: ``BAD_ARGS`` (not a
    list of names, or a side that does not exist), ``UNKNOWN_REGIONS``,
    ``NO_SIDES`` (a side on a sagittal stack)."""
    if isinstance(entries, str) or not isinstance(entries, (list, tuple)):
        raise Refused(
            RefusalCode.BAD_ARGS,
            message="restrict_to is a list of atlas regions.",
        )
    try:
        items = cast("Sequence[object]", entries)  # a list, checked above
        names = normalized_entries(str(entry) for entry in items)
    except ValueError as exc:
        raise Refused(RefusalCode.BAD_ARGS, message=str(exc)) from exc
    if names:
        try:
            resolve_entries(workspace.atlas, names)
        except ValueError as exc:
            raise Refused(
                RefusalCode.UNKNOWN_REGIONS, message=str(exc)
            ) from exc
        if state.plane == "sagittal" and has_sides(names):
            raise Refused(
                RefusalCode.NO_SIDES,
                message="A sagittal section lies within one "
                "hemisphere, so a region cannot be limited to one side.",
            )
    return tuple(names)


#: Margin added round the regions' bounding box, as a fraction of the canvas.
BOX_MARGIN = 0.05
#: The least magnification a restricted fit's picture zooms to its regions'
#: box for (:func:`zoom_window`); a box covering more of the canvas is drawn
#: whole, its regions highlighted only.
MIN_ZOOM_GAIN = 1.5


def zoom_window(box: Sequence[float] | None) -> list[float] | None:
    """*box* (:func:`region_box`) as the zoom of a restricted fit's picture,
    or None when it would magnify less than :data:`MIN_ZOOM_GAIN` times
    (its larger side over 1 / ``MIN_ZOOM_GAIN`` of the canvas's)."""
    if not box:
        return None
    extent = max(float(box[2]) - float(box[0]), float(box[3]) - float(box[1]))
    return [float(v) for v in box] if extent * MIN_ZOOM_GAIN <= 1.0 else None


def region_box(
    workspace: Workspace,
    state: Stack,
    record: Section,
    regions: Sequence[str],
) -> list[float] | None:
    """The atlas regions' bounding box on the section's canvas.

    ``[x0, y0, x1, y1]`` as fractions of the canvas, with a small margin: the
    zoom :func:`langslice.core.pictures.canvas.zoom_box` takes. None when the
    regions are not in the plane there. A side (``CTX:left``) is read as the
    region on both sides.
    """
    if not regions:
        return None
    try:
        names = [split_side(name)[0] for name in regions]
        geometry = stage(
            workspace, state, record, dict(KNOB_DEFAULTS), "canvas"
        ).geometry
        mask = regions_mask(workspace.atlas, geometry.annotation, names)
    except Exception as exc:
        logger.warning("region box failed for %s: %s", record.id, exc)
        return None
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    width, height = geometry.size
    scale = geometry.atlas_scale
    left, top = geometry.atlas_offset
    x0, x1 = left + cols[0] * scale, left + (cols[-1] + 1) * scale
    y0, y1 = top + rows[0] * scale, top + (rows[-1] + 1) * scale
    return [
        round(max(0.0, x0 / width - BOX_MARGIN), 4),
        round(max(0.0, y0 / height - BOX_MARGIN), 4),
        round(min(1.0, x1 / width + BOX_MARGIN), 4),
        round(min(1.0, y1 / height + BOX_MARGIN), 4),
    ]
