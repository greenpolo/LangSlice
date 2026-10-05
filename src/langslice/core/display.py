"""Picture options: ONE argument, ``view``, the same on every tool that returns a picture.

``view`` is a dict (:class:`langslice.doors.tools.arguments.View`) with these keys,
validated once by the tool door (:func:`langslice.doors.tools.view_options.parse_view`)
into a :class:`DisplayOptions` and drawn by the shared renderers (the physical
canvas in :func:`langslice.core.canvas.physical_views`, the tissue-framed
pictures below):

- ``mode`` — per tool (its :class:`langslice.doors.tools.view_options.Profile`),
  default per tool; :data:`MODE_RULES` says what each mode draws.
- ``channels`` — what of the SECTION is shown: one or more raw channel names
  (each stretched by percentile; one is gray, several are added in distinct
  colours, ABBA's multichannel display), or ONE version:
  ``view`` (the agent's own appearance, the default) or ``fit`` (what
  registration reads). Raw channels and a version cannot be mixed.
- ``atlas_channels`` — what of the ATLAS is shown: any of ``ara`` (the
  reference template), ``nissl`` (ABBA's cached Allen Nissl, only where that
  cache is installed and matches the run's atlas,
  :mod:`langslice.core.deformable.abba_atlas`) and ``borders`` (the region lines).
  Images are drawn under the section at ``atlas_opacity`` (overlay modes) or
  as the atlas picture; two images are added in two colours. No ``borders``
  means no lines. The defaults per mode reproduce the pictures each mode drew
  before ``view`` existed.
- ``atlas_opacity``, ``regions`` (one side allowed, ``"CTX:left"``),
  ``outlines`` (which lines ``borders`` draws: all or outer), ``border_color``,
  ``border_thickness``, ``zoom``, ``deformation`` (placement pictures: draw the
  applied warp, or ``none`` for the linear placement alone), ``resolution``
  (only where the host's image resolution is "auto").

This module is the core half: the options record, the mode rules and the
renderers. Options belong to one call: nothing here writes state (a
section's appearance changes only through ``preprocess``).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from PIL import Image

from langslice.core.affine import resize_long_edge
from langslice.core.appearance import (
    Look,
    channel_colors,
    section_settings,
)
from langslice.core.atlas.core import get_reference_slice
from langslice.core.atlas.render import (
    annotation_slice,
    family_outlines,
    outer_outline,
)
from langslice.core.canvas import (
    REGION_CONTEXT_ALPHA,
    _draw_polys,
    normalize_border_style,
    region_polys,
    regions_left,
)
from langslice.core.image_prep import mask_box
from langslice.core.sections import render_slice
from langslice.core.sizes import PICTURE_EDGES
from langslice.core.space import Plane
from langslice.core.state import Angles, SliceState, StackState, plane_angles
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: Every atlas channel, in the order pictures and docs list them.
ATLAS_CHANNELS: tuple[str, ...] = ("ara", "nissl", "borders")
#: The atlas channels that are images (``borders`` is lines).
ATLAS_IMAGES: tuple[str, ...] = ("ara", "nissl")
#: The preprocessed versions of a section a picture may show.
VERSIONS: tuple[str, ...] = ("view", "fit")
#: ``outlines``: every family boundary, or the outer contour alone.
OUTLINE_CHOICES: tuple[str, ...] = ("all", "outer")
#: ``deformation``: draw the applied warp, or the linear placement alone.
DEFORMATION_CHOICES: tuple[str, ...] = ("applied", "none")
#: Raw channels one overlay may add up.
MAX_OVERLAY_CHANNELS = 6
#: Atlas image opacity in an overlay when the call lists an image but no opacity.
DEFAULT_ATLAS_OPACITY = 0.5
#: Region line width in output pixels, everywhere.
DEFAULT_BORDER_THICKNESS = 1.0
DEFAULT_BORDER_COLOR = "yellow"
#: Intensity percentile mapped to white when a Nissl plane is shown.
NISSL_PERCENTILE = 99.5
FULL_VIEW = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class ModeRule:
    """What one mode draws, and so which ``view`` keys mean something in it."""

    #: A section is drawn (``channels`` applies).
    section: bool
    #: Something of the atlas is drawn (the atlas keys apply).
    atlas: bool
    #: ``atlas_channels`` when the call gives none.
    atlas_default: tuple[str, ...] = ()
    #: The atlas images are blended under the section (``atlas_opacity`` applies).
    opacity: bool = False
    #: The mode cannot draw without an atlas image (ara or nissl).
    needs_image: bool = False
    #: The mode cannot draw without some atlas channel.
    needs_atlas: bool = False


#: Every mode a picture tool has, by name.
MODE_RULES: dict[str, ModeRule] = {
    "section": ModeRule(section=True, atlas=False),
    "channels": ModeRule(section=True, atlas=False),
    "template": ModeRule(section=False, atlas=True, atlas_default=("ara",), needs_atlas=True),
    "stacked": ModeRule(section=True, atlas=True, atlas_default=("ara",), needs_atlas=True),
    "side_by_side": ModeRule(section=True, atlas=True, atlas_default=("ara", "borders"),
                             needs_atlas=True),
    "overlay": ModeRule(section=True, atlas=True, atlas_default=("borders",), opacity=True),
    "ab": ModeRule(section=True, atlas=True, atlas_default=("borders",), opacity=True),
    "checkerboard": ModeRule(section=True, atlas=True, atlas_default=("ara", "borders"),
                             needs_image=True),
    "outlines": ModeRule(section=True, atlas=True, atlas_default=("borders",), opacity=True),
    "borders": ModeRule(section=True, atlas=True, atlas_default=("borders",), opacity=True),
}


@dataclass(frozen=True)
class DisplayOptions:
    """One call's validated picture options."""

    mode: str
    zoom: tuple[float, ...]
    #: Raw channel names shown (empty when a version is shown).
    channels: tuple[str, ...]
    #: ``view`` or ``fit`` (empty when raw channels are shown).
    version: str
    atlas_channels: tuple[str, ...]
    atlas_opacity: float
    #: ``(name as asked, ids including descendants)`` per highlighted region;
    #: a name may carry a side (``"CTX:left"``, :mod:`langslice.core.atlas.sides`),
    #: which the renderers resolve against the picture's placement.
    regions: tuple[tuple[str, frozenset[int]], ...]
    #: Which lines ``borders`` draws: ``all`` or ``outer``.
    outlines: str
    border_color: str
    border_thickness: float
    #: ``applied`` or ``none`` (placement pictures).
    deformation: str = "applied"
    #: Long edge of each picture this call returns
    #: (:func:`langslice.core.sizes.picture_edge`).
    long_edge: int = PICTURE_EDGES["low"][1]
    #: The clamped ``resolution`` the agent asked for ("auto" only; None when
    #: it asked for none). A contact sheet sizes its tiles by it.
    resolution: int | None = None
    #: Whether the run's level is "auto" (the payload then echoes the size).
    auto: bool = False
    #: Why the asked resolution was changed ("" when it was not).
    resolution_note: str = ""
    #: The keys the call gave (the echo reports what applied, not defaults
    #: that mean nothing in this mode).
    given: frozenset[str] = frozenset()
    #: Whether this tool takes ``deformation``.
    has_deformation: bool = False
    #: Whether ``channels`` picks what this tool shows of the section. Not for
    #: ``preprocess`` (a target's appearance) or ``fit_deformable`` (the image
    #: the fit read): their echo names no channels or version.
    channels_apply: bool = True

    @property
    def full_view(self) -> bool:
        return not self.zoom or self.zoom == FULL_VIEW

    @property
    def window(self) -> list[float]:
        """The zoom as the renderers take it (empty = whole)."""
        return [] if self.full_view else list(self.zoom)

    @property
    def borders(self) -> bool:
        """Whether the atlas region lines are drawn."""
        return "borders" in self.atlas_channels

    @property
    def atlas_images(self) -> tuple[str, ...]:
        """The atlas images shown (``ara``/``nissl``), in canonical order."""
        return tuple(kind for kind in self.atlas_channels if kind in ATLAS_IMAGES)

    @property
    def layer(self) -> str:
        """The lines layer the renderers draw: ``all``, ``outer`` or ``none``."""
        return self.outlines if self.borders else "none"

    @property
    def lines(self) -> bool:
        """Whether any atlas line is drawn (the border style applies)."""
        return self.borders or bool(self.regions)

    def look(self, state: StackState, record: SliceState) -> Look:
        """The look a picture of *record* is drawn in."""
        if self.version == "fit":
            return section_settings(state, "fit", record.id)
        if self.version or not self.channels:
            return section_settings(state, "view", record.id)
        # One channel is stretched too, in gray (the `channels` strip is the
        # unmodified picture).
        return {"overlay": list(self.channels)}

    def channel_colors(self) -> dict[str, str]:
        """Colour of each raw channel in an overlay (empty for one channel or a version)."""
        if len(self.channels) < 2:
            return {}
        return {name: word for name, word, _rgb in channel_colors(self.channels)}

    def section_tag(self) -> str:
        """Caption fragment naming what of the section is shown ("" for the view version)."""
        if self.version == "fit":
            return "  [fit appearance]"
        if self.version or not self.channels:
            return ""
        if len(self.channels) == 1:
            return f"  [raw {self.channels[0]}, stretched]"
        return "  [raw " + " + ".join(
            name if name.lower() == word else f"{name} {word}"
            for name, word in self.channel_colors().items()) + "]"

    def atlas_name(self) -> str:
        """How captions name the atlas picture."""
        images = self.atlas_images
        if not images:
            return "lines only" if self.borders else "none"
        if images == ("ara",):
            return "template"
        if len(images) == 1:
            return images[0]
        return " + ".join(f"{kind} {word}" for kind, word, _rgb in channel_colors(images))

    def echo(self) -> dict[str, Any]:
        """The options as a payload's ``view`` field: what this call drew."""
        rule = MODE_RULES[self.mode]
        out: dict[str, Any] = {"mode": self.mode}
        if rule.section and self.channels_apply:
            out["channels"] = [self.version] if self.version else list(self.channels)
            colors = self.channel_colors()
            if colors:
                out["channel_colors"] = colors
        if rule.atlas:
            out["atlas_channels"] = list(self.atlas_channels)
            if len(self.atlas_images) > 1:
                out["atlas_colors"] = {kind: word for kind, word, _rgb
                                       in channel_colors(self.atlas_images)}
            if rule.opacity and self.atlas_images:
                out["atlas_opacity"] = self.atlas_opacity
            if self.regions:
                out["regions"] = [name for name, _ids in self.regions]
            if self.borders:
                out["outlines"] = self.outlines
            if self.lines:
                out["border_color"] = self.border_color
                out["border_thickness"] = self.border_thickness
        out["zoom"] = list(self.zoom or FULL_VIEW)
        if self.has_deformation:
            out["deformation"] = self.deformation
        # The size is the agent's to choose only at "auto"; any other level
        # keeps it out of model-facing text.
        if self.auto:
            out["resolution"] = self.resolution or self.long_edge
        if self.resolution_note:
            out["resolution_note"] = self.resolution_note
        return out


def available_atlas_channels(ctx: Workspace) -> tuple[str, ...]:
    """The atlas channels this host can draw."""
    return ATLAS_CHANNELS if ctx.abba_atlas is not None else ("ara", "borders")


def default_options(
    mode: str, *, atlas_channels: tuple[str, ...] | None = None,
    long_edge: int = PICTURE_EDGES["low"][1],
) -> DisplayOptions:
    """The options a picture uses when called with none (internal callers)."""
    rule = MODE_RULES[mode]
    return DisplayOptions(
        mode=mode, zoom=(), channels=(), version="view",
        atlas_channels=rule.atlas_default if atlas_channels is None else atlas_channels,
        atlas_opacity=0.0, regions=(), outlines="all", border_color="#ffff00",
        border_thickness=DEFAULT_BORDER_THICKNESS, long_edge=long_edge,
    )


# --- atlas images --------------------------------------------------------


def _nissl_plane(
    ctx: Workspace, state: StackState, position_mm: float, labels: np.ndarray,
    angles: Angles,
) -> np.ndarray:
    """ABBA's Nissl on the native plane grid, percentile-stretched to 0..1."""
    source = ctx.abba_atlas
    if source is None:
        raise ValueError("The nissl atlas channel needs ABBA's cached Allen atlas")
    values = np.asarray(source.sample_plane(
        "NISSL", ctx.atlas, position_mm, cast(Plane, state.plane), *angles,
    ), dtype=np.float32)
    inside = values[(labels > 0) & (values > 0)]
    top = float(np.percentile(inside, NISSL_PERCENTILE)) if inside.size else 0.0
    if top <= 0:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip(values / top, 0.0, 1.0)


def atlas_image_picture(
    ctx: Workspace, state: StackState, kinds: tuple[str, ...], position_mm: float,
    *, angles: Angles | None = None,
) -> Image.Image | None:
    """The atlas images *kinds* on the native plane grid; None for ``ara`` alone.

    At *angles* (:func:`langslice.core.state.plane_angles`: a section's own,
    or the stack's when None).

    ``ara`` alone is left to the renderers' own reference path (the pixels
    every earlier picture showed). ``nissl`` is ABBA's Nissl, stretched
    inside the atlas anatomy. Two images are added in their two colours
    (:func:`langslice.core.appearance.channel_colors`); no image is a black
    plane (the lines alone).
    """
    images = tuple(kind for kind in kinds if kind in ATLAS_IMAGES)
    if images == ("ara",):
        return None
    plane = cast(Plane, state.plane)
    pitch, yaw = plane_angles(state, angles)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=pitch, yaw_deg=yaw,
    ))
    shape = labels.shape
    if not images:
        return Image.new("L", (shape[1], shape[0]), 0)

    def gray(kind: str) -> np.ndarray:
        if kind == "nissl":
            return _nissl_plane(ctx, state, position_mm, labels, (pitch, yaw))
        reference = get_reference_slice(ctx.atlas, position_mm, plane=plane,
                                        pitch_deg=pitch, yaw_deg=yaw)
        if reference.size != (shape[1], shape[0]):
            reference = reference.resize((shape[1], shape[0]), Image.Resampling.BILINEAR)
        return np.asarray(reference.convert("L"), dtype=np.float32) / 255.0

    if len(images) == 1:
        return Image.fromarray((gray(images[0]) * 255.0).astype(np.uint8), mode="L")
    total = np.zeros((*shape, 3), dtype=np.float32)
    for kind, _word, rgb in channel_colors(images):
        total += gray(kind)[..., None] * np.asarray(rgb, dtype=np.float32)
    return Image.fromarray(np.clip(total, 0, 255).astype(np.uint8), mode="RGB")


def regions_in_plane(
    ctx: Workspace, state: StackState, position_mm: float, options: DisplayOptions,
    *, angles: Angles | None = None,
) -> list[str]:
    """The highlighted regions with at least one pixel in the plane at
    *position_mm* and *angles* (the stack's when None)."""
    if not options.regions:
        return []
    pitch, yaw = plane_angles(state, angles)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=cast(Plane, state.plane),
        pitch_deg=pitch, yaw_deg=yaw,
    ))
    return [name for name, ids in options.regions if np.isin(labels, list(ids)).any()]


def _crop_fraction(image: Image.Image, zoom: tuple[float, ...]) -> Image.Image:
    from langslice.core.canvas import zoom_box

    return image.crop(zoom_box(list(zoom), image.size))


# --- tissue-framed pictures ----------------------------------------------


def framed_section(
    ctx: Workspace, state: StackState, record: SliceState, options: DisplayOptions,
    *, long_edge: int | None = None, look: Look | str = "options",
) -> Image.Image:
    """One section as corrected, tissue-framed, in the call's look and zoom.

    *long_edge* None is the call's picture size (``options.long_edge``);
    *look* overrides the options' (``preprocess`` draws a target's
    appearance before and after its write). A zoom renders the framed section
    larger (never past its working copy) and crops the requested fraction,
    so it magnifies.
    """
    long_edge = long_edge or options.long_edge
    drawn: Look = options.look(state, record) if look == "options" else cast(Look, look)
    if options.full_view:
        return render_slice(ctx, record, long_edge=long_edge, frame=True, look=drawn)
    x0, y0, x1, y1 = options.zoom
    extent = max(abs(x1 - x0), abs(y1 - y0), 1e-3)
    larger = render_slice(
        ctx, record, long_edge=int(math.ceil(long_edge / min(1.0, extent))), frame=True,
        look=drawn,
    )
    return _crop_fraction(larger, options.zoom)


def channel_strip(
    ctx: Workspace, state: StackState, record: SliceState, options: DisplayOptions,
    *, tile_edge: int,
) -> tuple[Image.Image, list[str]]:
    """One section's raw channels side by side, each unmodified and labelled.

    Every tile is the same tissue frame as the section's other pictures, one
    raw plane in grayscale exactly as read (no stretch, no enhancement), at
    *tile_edge* at most. Returns the strip and the channel names in order.
    """
    from langslice.core.captions import caption
    from langslice.core.sheets import beside

    names, _planes = ctx.section_channels(record.id)
    strip: Image.Image | None = None
    for name in names:
        tile = caption(
            framed_section(ctx, state, record, options, long_edge=tile_edge,
                           look={"channel": name}),
            name,
        )
        strip = tile if strip is None else beside(strip, tile)
    assert strip is not None
    return strip, list(names)


def framed_atlas(
    ctx: Workspace, state: StackState, position_mm: float, options: DisplayOptions,
    *, long_edge: int | None = None, fill: bool = False, angles: Angles | None = None,
) -> Image.Image:
    """The atlas at *position_mm*, framed to its anatomy, with the call's lines.

    At *angles*: a section's own for a picture beside that section, the
    stack's view angles for ``view_atlas``; None, the stack's one angle.

    At most *long_edge* (None: ``options.long_edge``) and never upsampled
    past the plane's own voxels, unless *fill*: then drawn at exactly
    *long_edge*, lines included, for a picture that puts the atlas beside a
    section of that size. With ``ara`` alone (no lines, no regions, no zoom)
    and no *fill* this is the picture ``view_atlas`` always sent.
    """
    from langslice.core.atlas_fetch import atlas_mask, atlas_section, atlas_sized

    edge = int(long_edge or options.long_edge)

    def sized(picture: Image.Image) -> Image.Image:
        return resize_long_edge(picture, edge) if fill else atlas_sized(picture, edge)

    pitch, yaw = plane_angles(state, angles)
    if options.atlas_images == ("ara",) and not options.lines and options.full_view:
        return sized(atlas_section(ctx, state, position_mm, frame=True, angles=(pitch, yaw)))
    plane = cast(Plane, state.plane)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=pitch, yaw_deg=yaw,
    ))
    picture = atlas_image_picture(ctx, state, options.atlas_channels, position_mm,
                                  angles=(pitch, yaw))
    if picture is None:
        picture = get_reference_slice(
            ctx.atlas, position_mm, plane=plane,
            pitch_deg=pitch, yaw_deg=yaw,
        )
    native = (labels.shape[1], labels.shape[0])
    if picture.size != native:
        picture = picture.resize(native, Image.Resampling.BILINEAR)
    try:
        mask = atlas_mask(ctx, state, position_mm, picture.size, angles=(pitch, yaw)) > 0
    except Exception as exc:
        logger.warning("atlas at %.3f mm: framed by its labels, no root mask (%s)",
                       position_mm, exc)
        mask = labels > 0
    box = mask_box(picture.size, mask) or (0, 0, picture.width, picture.height)
    if not options.full_view:
        from langslice.core.canvas import zoom_box

        inner = zoom_box(list(options.zoom), (box[2] - box[0], box[3] - box[1]))
        box = (box[0] + inner[0], box[1] + inner[1], box[0] + inner[2], box[1] + inner[3])
    cropped = picture.crop(box)
    shown = sized(cropped).convert("RGB")
    if not options.lines:
        return shown
    factor = shown.width / float(cropped.width)
    canvas = np.asarray(shown, dtype=np.uint8).copy()
    rgb, _ = normalize_border_style(options.border_color, options.border_thickness)
    context: list[np.ndarray] = []
    if options.borders:
        draw = outer_outline if options.outlines == "outer" else family_outlines
        context = [poly for _color, poly in draw(
            ctx.atlas, position_mm, plane=plane, pitch_deg=pitch, yaw_deg=yaw,
        )]
    origin = (box[0], box[1])
    if context:
        _draw_polys(canvas, context, rgb, thickness=options.border_thickness, origin=origin,
                    factor=factor, alpha=REGION_CONTEXT_ALPHA if options.regions else 1.0)
    if options.regions:
        # An atlas-only picture has no section: a side is the picture's own.
        left = regions_left(ctx.atlas, options.regions, position_mm, plane,
                            pitch, yaw, np.eye(2))
        _draw_polys(canvas, region_polys(labels, options.regions, left), rgb,
                    thickness=options.border_thickness, origin=origin, factor=factor)
    return Image.fromarray(canvas, mode="RGB")


def atlas_caption(state: StackState, position_mm: float, options: DisplayOptions,
                  *, angles: Angles | None = None) -> str:
    """The label burned into an atlas picture drawn at *angles* (the stack's
    when None)."""
    from langslice.core.captions import angles_label

    shown = angles_label(plane_angles(state, angles))
    name = options.atlas_name()
    extra = "" if name == "template" else f" {name}"
    if options.borders and options.outlines == "outer":
        extra += " outer outline"
    if options.regions:
        extra += " regions " + ",".join(name for name, _ids in options.regions)
    return f"atlas {position_mm:.2f} mm{shown}{extra}"


__all__ = [
    "ATLAS_CHANNELS", "DisplayOptions", "MODE_RULES",
    "atlas_caption", "atlas_image_picture", "available_atlas_channels", "channel_strip",
    "default_options", "framed_atlas", "framed_section", "regions_in_plane",
]
