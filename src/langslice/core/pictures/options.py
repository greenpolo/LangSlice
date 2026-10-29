"""Picture options and the section and atlas renderers.

A :class:`DisplayOptions` is one picture's options, built by the core's own
callers (``look``'s pictures, :mod:`langslice.core.pictures.look`; the fit,
trace and damage pictures, :func:`default_options`) and drawn by the shared
renderers (the physical canvas in
:func:`langslice.core.pictures.canvas.physical_views`, the section and atlas
pictures below):

- ``mode`` — :data:`MODE_RULES` says what each mode draws.
- ``image`` and ``channels`` — what of the SECTION is shown: one section image
  alone, in gray (:mod:`langslice.core.sections.images`; empty is
  ``preprocessed``, what every fit reads by default), or several raw channels
  added in distinct colours with automatic contrast (ABBA's multichannel
  display, :mod:`langslice.core.sections.channels`).
- ``atlas_channels`` — what of the ATLAS is shown: any of the atlas's images
  (``template`` and the additional references the BrainGlobe atlas ships,
  by their names, :mod:`langslice.core.atlas.images`) and ``borders`` (the
  region lines).
  Images are drawn under the section at ``atlas_opacity`` (overlay modes) or
  as the atlas picture; two images are added in two colours. No ``borders``
  means no lines.
- ``atlas_opacity``, ``regions`` (one side allowed, ``"CTX:left"``),
  ``outlines`` (which lines ``borders`` draws: all or none), ``border_color``,
  ``border_thickness``, ``zoom`` (``[x0, y0, x1, y1]`` pixels of the picture
  the same call returns unzoomed, top-left origin: :func:`zoom_fractions`),
  ``deformation`` (placement pictures: draw the applied warp, or ``none`` for
  the linear placement alone).

Options belong to one call: nothing here writes state.
"""

from __future__ import annotations

from dataclasses import dataclass

from langslice.core.atlas.images import atlas_images
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.pictures.canvas import (
    AUTO_BORDER_COLOR,
)
from langslice.core.pictures.sizes import PICTURE_EDGES
from langslice.core.sections.channels import channel_colors, colour_words
from langslice.core.sections.images import PREPROCESSED, Look
from langslice.core.stack.section import SliceState
from langslice.core.stack.stack import StackState
from langslice.core.workspace import Workspace

#: The atlas channel that is lines; every other atlas channel is an image.
BORDERS = "borders"


#: Raw channels one overlay may add up.
MAX_OVERLAY_CHANNELS = 6
#: Atlas image opacity in an overlay when the call lists an image but no opacity.
DEFAULT_ATLAS_OPACITY = 0.5
#: Region line width in output pixels, everywhere.
DEFAULT_BORDER_THICKNESS = 1.0


def zoom_fractions(zoom: tuple[float, ...] | list[float],
                   shown: tuple[int, int]) -> tuple[float, ...]:
    """A pixel zoom as fractions of the content it was read on.

    *zoom* is ``[x0, y0, x1, y1]`` in pixels of the picture the call returns
    unzoomed (top-left origin, x right, y down; the caption band is below
    the content, so a picture pixel is the content's own) and *shown* that
    picture's content size ``(width, height)``. The corners are ordered and
    clamped to the content; empty is the whole picture, ``()``.
    """
    if not zoom:
        return ()
    width, height = max(int(shown[0]), 1), max(int(shown[1]), 1)
    x0, x1 = sorted((float(zoom[0]), float(zoom[2])))
    y0, y1 = sorted((float(zoom[1]), float(zoom[3])))
    return (min(max(x0 / width, 0.0), 1.0), min(max(y0 / height, 0.0), 1.0),
            min(max(x1 / width, 0.0), 1.0), min(max(y1 / height, 0.0), 1.0))


@dataclass(frozen=True)
class ModeRule:
    """What one mode draws."""

    #: A section is drawn (``channels`` applies).
    section: bool
    #: Something of the atlas is drawn (the atlas keys apply).
    atlas: bool
    #: ``atlas_channels`` when the caller gives none.
    atlas_default: tuple[str, ...] = ()
    #: The atlas images are blended under the section (``atlas_opacity`` applies).
    opacity: bool = False


#: Every picture mode, by name: ``look``'s (``section``, ``template`` for
#: ``atlas``, ``overlay``, ``stacked`` for ``positioning``) and ``borders``
#: (a deformable fit or trace picture).
MODE_RULES: dict[str, ModeRule] = {
    "section": ModeRule(section=True, atlas=False),
    "template": ModeRule(section=False, atlas=True, atlas_default=("template",)),
    "stacked": ModeRule(section=True, atlas=True, atlas_default=("template",)),
    "overlay": ModeRule(section=True, atlas=True, atlas_default=("borders",), opacity=True),
    "borders": ModeRule(section=True, atlas=True, atlas_default=("borders",), opacity=True),
}


@dataclass(frozen=True)
class DisplayOptions:
    """One call's validated picture options."""

    mode: str
    #: ``[x0, y0, x1, y1]`` pixels of the unzoomed picture (:func:`zoom_fractions`);
    #: empty is the whole picture.
    zoom: tuple[float, ...]
    #: Raw channel names drawn together in their colours (empty: one image).
    channels: tuple[str, ...]
    #: The section image drawn alone (empty: ``preprocessed``).
    image: str
    atlas_channels: tuple[str, ...]
    atlas_opacity: float
    #: ``(name as asked, ids including descendants)`` per highlighted region;
    #: a name may carry a side (``"CTX:left"``, :mod:`langslice.core.atlas.sides`),
    #: which the renderers resolve against the picture's placement.
    regions: tuple[tuple[str, frozenset[int]], ...]
    #: Which lines ``borders`` draws: ``all`` or ``none``.
    outlines: str
    border_color: str
    border_thickness: float
    #: ``applied`` or ``none`` (placement pictures).
    deformation: str = "applied"
    #: Long edge of each picture this call returns
    #: (:func:`langslice.core.pictures.sizes.picture_edge`).
    long_edge: int = PICTURE_EDGES["low"][1]
    #: Ontology depth of the regions whose boundaries ``all`` draws
    #: (:func:`langslice.core.atlas.levels.labels_at`).
    depth: int = OVERLAY_DEPTH

    @property
    def full_view(self) -> bool:
        return not self.zoom

    def window(self, shown: tuple[int, int]) -> list[float]:
        """The zoom as fractions of the unzoomed content of size *shown*, as
        the renderers take it (empty = whole)."""
        return list(zoom_fractions(self.zoom, shown))

    @property
    def borders(self) -> bool:
        """Whether the atlas region lines are drawn."""
        return BORDERS in self.atlas_channels

    @property
    def atlas_images(self) -> tuple[str, ...]:
        """The atlas images shown, in the order the atlas lists them."""
        return tuple(kind for kind in self.atlas_channels if kind != BORDERS)

    @property
    def layer(self) -> str:
        """The lines layer the renderers draw: ``all``, ``outer`` or ``none``."""
        return self.outlines if self.borders else "none"

    @property
    def lines(self) -> bool:
        """Whether any atlas line is drawn (the border style applies)."""
        return self.borders or bool(self.regions)

    def look(self, state: StackState, record: SliceState) -> Look:
        """The look a picture of *record* is drawn in: ``preprocessed`` (None),
        another section image alone, or the raw channels together."""
        if self.channels:
            return {"overlay": list(self.channels)}
        if self.image and self.image != PREPROCESSED:
            return {"image": self.image}
        return None

    def section_tag(self, state: StackState | None = None) -> str:
        """Caption fragment naming what of the section is shown: the section
        image, or the raw channels in their colours."""
        if self.channels:
            return "  [raw " + colour_words(self.channels) + "]"
        return f"  [{self.image or PREPROCESSED}]"

    def atlas_name(self) -> str:
        """How captions name the atlas picture."""
        images = self.atlas_images
        if not images:
            return "lines only" if self.borders else "none"
        if len(images) == 1:
            return images[0]
        return " + ".join(f"{kind} {word}" for kind, word, _rgb in channel_colors(images))

def available_atlas_channels(ctx: Workspace) -> tuple[str, ...]:
    """The atlas channels of this run's atlas: its images
    (:func:`langslice.core.atlas.images.atlas_images`), then ``borders``."""
    return (*atlas_images(ctx.atlas), BORDERS)


def default_options(
    mode: str, *, atlas_channels: tuple[str, ...] | None = None,
    long_edge: int = PICTURE_EDGES["low"][1],
) -> DisplayOptions:
    """The options a picture uses when called with none (internal callers)."""
    rule = MODE_RULES[mode]
    return DisplayOptions(
        mode=mode, zoom=(), channels=(), image="",
        atlas_channels=rule.atlas_default if atlas_channels is None else atlas_channels,
        atlas_opacity=0.0, regions=(), outlines="all", border_color=AUTO_BORDER_COLOR,
        border_thickness=DEFAULT_BORDER_THICKNESS, long_edge=long_edge,
    )


__all__ = [
    "BORDERS",
    "DisplayOptions",
    "MODE_RULES",
    "available_atlas_channels",
    "default_options",
]
