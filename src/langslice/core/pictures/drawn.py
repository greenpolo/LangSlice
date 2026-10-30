"""What a call drew: the per-call picture notes."""

from __future__ import annotations

import contextlib
import contextvars
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.atlas.levels import OVERLAY_DEPTH

if TYPE_CHECKING:
    from langslice.core.pictures.canvas import PanelFrame
    from langslice.core.pictures.placement import CanvasFrame


# --- notes: which pictures a call returned ------------------------------------------


@dataclass
class PictureNote:
    """What one picture shows: its sections, its mode and, for a placement
    picture, the frame it was drawn in."""

    sections: tuple[str, ...] = ()
    mode: str | None = None
    frame: CanvasFrame | None = None
    panel: PanelFrame | None = None
    #: The section's applied deformation as the job stores it, when drawn.
    deformation: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    #: A deformable-fit picture's frame (:class:`WarpNote`): the section on
    #: its fit grid with the record's borders.
    warp: WarpNote | None = None
    #: What a later stage needs to redraw the picture: a JSON-able
    #: ``{"renderer": name, "args": {...}}`` (None: not redrawable).
    recipe: dict[str, Any] | None = None
    #: The one-line caption sent beside the picture (None: none).
    caption: str | None = None


@dataclass(frozen=True)
class WarpNote:
    """How a ``fit_deformable`` picture was drawn: the record resampled onto
    the picture's content (``record``, a
    :class:`~langslice.core.fits.deformable.record.DeformableRecord` on the
    picture's
    content, which starts ``band`` rows down: 0, the caption band being
    below it), whether the residual was
    drawn (``warped``; False is the linear placement alone), and the border
    style the lines were drawn with
    (``langslice.core.fits.deformable.draw_warped_borders``)."""

    record: Any
    band: int
    warped: bool
    highlight: tuple[str, ...] = ()
    marked: tuple[str, ...] = ()
    outlines: str = "all"
    depth: int = OVERLAY_DEPTH
    width_px: float = 2.0
    native: Any = None
    #: The size the record's grid was resized to and the box of it the
    #: picture shows (``resampled_record``'s ``size`` and ``box``), so the
    #: picture's residual is drawn again from the record on demand.
    grid_size: tuple[int, int] = (0, 0)
    box: tuple[int, int, int, int] = (0, 0, 0, 0)


_NOTES: contextvars.ContextVar[list[tuple[Image.Image, PictureNote]] | None] = (
    contextvars.ContextVar("langslice_picture_notes", default=None))


@contextlib.contextmanager
def collecting() -> Iterator[list[tuple[Image.Image, PictureNote]]]:
    """Collect the notes made while the block runs (one tool call)."""
    notes: list[tuple[Image.Image, PictureNote]] = []
    token = _NOTES.set(notes)
    try:
        yield notes
    finally:
        _NOTES.reset(token)


def note(image: Image.Image, **fields: Any) -> Image.Image:
    """Note what *image* shows (:class:`PictureNote` fields, among them
    ``recipe=`` and ``caption=``); returns it."""
    notes = _NOTES.get()
    if notes is not None:
        sections = fields.pop("sections", ())
        notes.append((image, PictureNote(sections=tuple(str(s) for s in sections), **fields)))
    return image


def annotate(image: Image.Image, **fields: Any) -> Image.Image:
    """Set :class:`PictureNote` fields (``recipe=``, ``caption=``) on the
    latest note made for this very image, keeping its frame and panel; a
    picture nothing noted yet gets a new note. Returns *image*."""
    notes = _NOTES.get()
    if notes is None:
        return image
    held = note_for(image, notes)
    if held is None:
        return note(image, **fields)
    for key, value in fields.items():
        if key == "sections":
            value = tuple(str(s) for s in value)
        setattr(held, key, value)
    return image


def note_of(image: Image.Image) -> PictureNote | None:
    """The latest note made for this very image in the block collecting now
    (:func:`collecting`), or None (none made, or nothing collecting)."""
    notes = _NOTES.get()
    return None if notes is None else note_for(image, notes)


def note_for(
    image: Image.Image, notes: list[tuple[Image.Image, PictureNote]],
) -> PictureNote | None:
    """The latest note made for this very image (identity, not equality)."""
    for noted, held in reversed(notes):
        if noted is image:
            return held
    return None
