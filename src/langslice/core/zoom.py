"""Zoom: a box of an earlier picture, drawn again at more detail.

:func:`redraw` takes the picture's recipe (the picture index keeps it,
:mod:`langslice.job.views`) and a box ``[x0, y0, x1, y1]`` in that
picture's pixels (top-left origin; the caption band lies below the content,
so a pixel of the content is the picture's own). It re-runs the recipe's
renderer on that box at the picture's own long edge
(:mod:`langslice.core.look`): a section, alone or under the atlas, read
again from its image file at the file's own resolution
(:mod:`langslice.core.native`), a positioning picture's sections from their
working copies, the atlas past its voxels by at most
:data:`langslice.core.positioning.ATLAS_UPSAMPLE`. A zoom's recipe holds
its window as fractions of the unzoomed picture, so a zoom of a zoom (a box
in the zoomed picture's pixels) maps back onto the first picture and is
redrawn from the source again, never enlarged from the zoom.

When the stack has changed since the picture was drawn, it is redrawn as it
was (the recipe's snapshot) and flagged ``stale``. A picture without a
recipe (the opening strips), or whose sections are no longer in the stack,
is cropped from its saved image and enlarged, ``redrawn`` False; its recipe
is a ``crop`` recipe (the box on the source picture), so a crop of a crop
maps back the same way, given the source picture again.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from langslice.core import look as looks
from langslice.core.captions import caption
from langslice.core.layers import annotate, note
from langslice.core.sizes import picture_edge
from langslice.core.state import StackState
from langslice.core.workspace import Workspace

#: The recipe renderer of a picture cropped from a saved image.
CROP_RENDERER = "crop"
#: How much a crop (no redraw) is enlarged at most: it holds no more detail.
MAX_CROP_ENLARGE = 4.0


class ZoomError(ValueError):
    """A zoom that cannot be drawn: ``code`` is ``BAD_BOX`` (not four
    numbers), ``EMPTY_BOX`` (no area inside the picture) or ``NO_PICTURE``
    (nothing to redraw and no saved picture to crop)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Zoomed:
    """A zoom: the captioned picture, its index caption and recipe, whether
    it was redrawn from the source (False: cropped from the saved picture)
    and whether the stack has changed since the source picture."""

    image: Image.Image
    caption: str
    recipe: dict[str, Any]
    redrawn: bool
    stale: bool = False
    sections: tuple[str, ...] = ()
    mode: str | None = None
    #: Micrometres per pixel of the zoom's content (None for a crop).
    um_per_px: float | None = None


def box_fractions(box: Sequence[float], shown: Sequence[float]) -> tuple[float, ...]:
    """*box* (``[x0, y0, x1, y1]`` pixels of a picture whose content is
    *shown* ``(width, height)``) as fractions of that content, ordered and
    clamped. :class:`ZoomError` for a box that is not four numbers or that
    leaves no area."""
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        raise ZoomError("BAD_BOX", "box must be four numbers [x0, y0, x1, y1]") from None
    if len(values) != 4:
        raise ZoomError("BAD_BOX", "box must be four numbers [x0, y0, x1, y1]")
    width, height = max(float(shown[0]), 1.0), max(float(shown[1]), 1.0)
    x0, x1 = sorted((values[0], values[2]))
    y0, y1 = sorted((values[1], values[3]))
    out = (min(max(x0 / width, 0.0), 1.0), min(max(y0 / height, 0.0), 1.0),
           min(max(x1 / width, 0.0), 1.0), min(max(y1 / height, 0.0), 1.0))
    if (out[2] - out[0]) * width < 2 or (out[3] - out[1]) * height < 2:
        raise ZoomError("EMPTY_BOX", f"box {values} has no area inside the picture "
                        f"({int(width)} x {int(height)} px above its caption)")
    return out


def compose(window: Sequence[float], inner: Sequence[float]) -> tuple[float, ...]:
    """*inner* (fractions of a picture drawn at *window*, itself fractions
    of the unzoomed picture; empty: whole) as fractions of the unzoomed
    picture."""
    x0, y0, x1, y1 = window if window else (0.0, 0.0, 1.0, 1.0)
    return (x0 + inner[0] * (x1 - x0), y0 + inner[1] * (y1 - y0),
            x0 + inner[2] * (x1 - x0), y0 + inner[3] * (y1 - y0))


def _shown(recipe: dict[str, Any] | None, picture: Image.Image | None) -> tuple[float, float]:
    """The content size a box is read on: the recipe's, else the picture's."""
    held = (recipe or {}).get("shown")
    if held:
        return float(held[0]), float(held[1])
    if picture is None:
        raise ZoomError("NO_PICTURE", "no picture to read the box on")
    return float(picture.width), float(picture.height)


def _image(picture: Image.Image | bytes | str | Path | None) -> Image.Image | None:
    if picture is None or isinstance(picture, Image.Image):
        return picture
    if isinstance(picture, bytes | bytearray):
        return Image.open(io.BytesIO(picture)).convert("RGB")
    return Image.open(picture).convert("RGB")


#: Redrawers by recipe renderer: ``(recipe, window, workspace, state,
#: store, long_edge) -> look.LookPicture``.
RENDERERS: dict[str, Callable[..., looks.LookPicture]] = {
    looks.RENDERER: lambda recipe, window, ws, state, store, edge: looks.redraw(
        recipe, window, ws, state, store=store, zoom_edge=edge),
}

STALE_NOTE = "drawn as the stack was then; it has changed since"


def redraw(
    recipe: dict[str, Any] | None,
    box: Sequence[float],
    workspace: Workspace,
    *,
    state: StackState | None = None,
    store: Any = None,
    picture: Image.Image | bytes | str | Path | None = None,
    long_edge: int | None = None,
) -> Zoomed:
    """*box* of the picture *recipe* describes, drawn again.

    With a recipe a renderer knows (:data:`RENDERERS`) and the stack
    *state*, the box is redrawn from the source (*store*: an overlay's
    applied deformations); *long_edge* is a positioning zoom's long edge
    (None: the picture's own). Otherwise *picture* (the saved picture:
    image, bytes or path; for a ``crop`` recipe, the picture it was cropped
    from) is cropped and enlarged towards *long_edge* (None: the run's
    picture size, :func:`langslice.core.sizes.picture_edge`). Notes the zoom with its recipe and
    caption (:func:`langslice.core.layers.note`).
    """
    renderer = (recipe or {}).get("renderer")
    source = _image(picture)
    if recipe is not None and renderer in RENDERERS and state is not None:
        inner = box_fractions(box, _shown(recipe, None))
        window = compose((recipe.get("args") or {}).get("zoom") or (), inner)
        try:
            drawn = RENDERERS[renderer](recipe, window, workspace, state, store, long_edge)
        except looks.LookError as exc:
            if exc.code != "UNKNOWN_SECTION" or source is None:
                raise
            return _crop(None, box, source, long_edge or picture_edge(workspace), stale=True,
                         note_text=str(exc))
        image, text = drawn.image, drawn.caption
        if drawn.stale:
            image = caption(image, STALE_NOTE)
            text = f"{text}; {STALE_NOTE}"
        kept = {key: value for key, value in recipe.items() if key not in drawn.recipe}
        out_recipe = {**kept, **drawn.recipe}
        if image is drawn.image:
            # The renderer's own note (an overlay's frame, for its layers) is kept.
            annotate(image, recipe=out_recipe, caption=text)
        else:
            note(image, sections=drawn.sections, mode=drawn.mode, recipe=out_recipe,
                 caption=text)
        return Zoomed(image=image, caption=text, recipe=out_recipe, redrawn=True,
                      stale=drawn.stale, sections=drawn.sections, mode=drawn.mode,
                      um_per_px=drawn.um_per_px)
    if source is None:
        raise ZoomError("NO_PICTURE", "this picture cannot be redrawn and no saved copy was "
                        "given to crop")
    return _crop(recipe if renderer == CROP_RENDERER else None, box, source,
                 long_edge or picture_edge(workspace))


def _crop(
    recipe: dict[str, Any] | None, box: Sequence[float], source: Image.Image,
    long_edge: int, *, stale: bool = False, note_text: str = "",
) -> Zoomed:
    """*box* cropped from *source* and enlarged towards *long_edge* (at
    most :data:`MAX_CROP_ENLARGE`, never shrunk); *recipe* a ``crop``
    recipe when *source* is the picture an earlier crop was cut from."""
    parent = (recipe or {}).get("args", {}).get("box")
    shown = _shown(recipe, source)
    inner = box_fractions(box, shown)
    if parent:
        px0, py0, px1, py1 = (float(v) for v in parent)
        on_source = (px0 + inner[0] * (px1 - px0), py0 + inner[1] * (py1 - py0),
                     px0 + inner[2] * (px1 - px0), py0 + inner[3] * (py1 - py0))
    else:
        on_source = (inner[0] * source.width, inner[1] * source.height,
                     inner[2] * source.width, inner[3] * source.height)
    left, top, right, bottom = (round(v) for v in on_source)
    cut = source.crop((left, top, right, bottom))
    target = float(long_edge)
    factor = max(1.0, min(target / max(cut.size), MAX_CROP_ENLARGE))
    if factor > 1.0:
        cut = cut.resize((max(1, round(cut.width * factor)), max(1, round(cut.height * factor))),
                         Image.Resampling.LANCZOS)
    where = ", ".join(str(round(v)) for v in on_source)
    label = f"crop of an earlier picture, not redrawn: box [{where}]"
    if factor > 1.0:
        label += f", enlarged x{factor:.1f}"
    if stale:
        label += f" ({note_text})" if note_text else ""
    image = caption(cut, label)
    out_recipe = {**{k: v for k, v in (recipe or {}).items() if k not in ("args", "shown")},
                  "renderer": CROP_RENDERER,
                  "args": {"box": [round(v, 2) for v in on_source]},
                  "shown": [cut.width, cut.height]}
    note(image, mode="zoom", recipe=out_recipe, caption=label)
    return Zoomed(image=image, caption=label, recipe=out_recipe, redrawn=False, stale=stale,
                  mode="zoom")


__all__ = [
    "CROP_RENDERER", "RENDERERS", "Zoomed", "ZoomError", "box_fractions", "compose", "redraw",
]
