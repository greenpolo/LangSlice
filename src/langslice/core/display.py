"""Picture options and the section and atlas renderers.

A :class:`DisplayOptions` is one picture's options, built by the core's own
callers (``look``'s pictures, :mod:`langslice.core.look`; the fit, trace and
damage pictures, :func:`default_options`) and drawn by the shared renderers
(the physical canvas in :func:`langslice.core.canvas.physical_views`, the
section and atlas pictures below):

- ``mode`` — :data:`MODE_RULES` says what each mode draws.
- ``channels`` — what of the SECTION is shown: one or more raw channel names
  (each with its display setting or automatic contrast limits,
  :mod:`langslice.core.channels`; one is gray, several are added in distinct
  colours, ABBA's multichannel display), or the version ``preprocessed``
  (what registration reads); neither is the section's default render.
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

import logging
import math
from dataclasses import dataclass
from typing import cast

import numpy as np
from PIL import Image

from langslice.core.appearance import (
    PREPROCESSED,
    Look,
    channel_colors,
    preprocessed_settings,
)
from langslice.core.atlas.core import get_reference_slice
from langslice.core.atlas.images import TEMPLATE, atlas_images
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.atlas.render import (
    HIGHLIGHT_WIDTH,
    annotation_slice,
    region_outlines,
)
from langslice.core.canvas import (
    AUTO_BORDER_COLOR,
    REGION_CONTEXT_ALPHA,
    _draw_polys,
    line_coverage,
    region_polys,
    regions_left,
    resolve_border_color,
)
from langslice.core.channels import all_properties, display_words, with_properties
from langslice.core.image_prep import mask_box
from langslice.core.sections import render_slice
from langslice.core.sizes import PICTURE_EDGES
from langslice.core.space import Plane
from langslice.core.state import Angles, SliceState, StackState, plane_angles
from langslice.core.workspace import Workspace

logger = logging.getLogger(__name__)

#: The atlas channel that is lines; every other atlas channel is an image.
BORDERS = "borders"


#: Raw channels one overlay may add up.
MAX_OVERLAY_CHANNELS = 6
#: Atlas image opacity in an overlay when the call lists an image but no opacity.
DEFAULT_ATLAS_OPACITY = 0.5
#: Region line width in output pixels, everywhere.
DEFAULT_BORDER_THICKNESS = 1.0
#: Intensity percentile (inside the atlas anatomy) mapped to white when an
#: atlas image other than the template is shown.
ATLAS_IMAGE_PERCENTILE = 99.5


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
    #: Raw channel names shown (empty: the default render, or the version).
    channels: tuple[str, ...]
    #: ``preprocessed``, or empty.
    version: str
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
    #: (:func:`langslice.core.sizes.picture_edge`).
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
        """The look a picture of *record* is drawn in: the version
        ``preprocessed`` is the preprocessed channel, no channels
        the default render (None); raw channels carry their display
        properties."""
        if self.version == PREPROCESSED:
            return preprocessed_settings(state, record.id)
        if not self.channels:
            return None
        # One channel is stretched too, in gray (the `channels` strip is the
        # unmodified picture).
        return with_properties(state, {"overlay": list(self.channels)})

    def channel_colors(self, state: StackState | None = None) -> dict[str, str]:
        """Colour of each raw channel in an overlay (empty for one channel or a
        version); with *state*, a channel's own colormap wins."""
        if len(self.channels) < 2:
            return {}
        held = all_properties(state) if state is not None else {}
        out: dict[str, str] = {}
        for name, word, _rgb in channel_colors(self.channels):
            own = held[name].colormap if name in held else None
            out[name] = own or word
        return out

    def section_tag(self, state: StackState | None = None) -> str:
        """Caption fragment naming what of the section is shown ("" for the
        default render): each raw channel with its display setting, ``auto``
        without *state* (:func:`langslice.core.channels.display_words`)."""
        if self.version == PREPROCESSED:
            return "  [preprocessed channel]"
        if not self.channels:
            return ""
        held = all_properties(state) if state is not None else {}
        colours = self.channel_colors(state) or {self.channels[0]: "gray"}
        return "  [raw " + display_words(self.channels, held, colours) + "]"

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
        mode=mode, zoom=(), channels=(), version="",
        atlas_channels=rule.atlas_default if atlas_channels is None else atlas_channels,
        atlas_opacity=0.0, regions=(), outlines="all", border_color=AUTO_BORDER_COLOR,
        border_thickness=DEFAULT_BORDER_THICKNESS, long_edge=long_edge,
    )


# --- atlas images --------------------------------------------------------


def _image_plane(
    ctx: Workspace, state: StackState, kind: str, position_mm: float, labels: np.ndarray,
    angles: Angles,
) -> np.ndarray:
    """The atlas image *kind* on the native plane grid, percentile-stretched to 0..1."""
    from langslice.core.oblique import sample_oblique_plane

    values = np.asarray(sample_oblique_plane(
        ctx.atlas, position_mm, cast(Plane, state.plane), *angles, volume=kind, order=1,
    ), dtype=np.float32)
    if values.shape != labels.shape:
        values = np.asarray(Image.fromarray(values, mode="F").resize(
            (labels.shape[1], labels.shape[0]), Image.Resampling.BILINEAR), dtype=np.float32)
    inside = values[(labels > 0) & (values > 0)]
    top = float(np.percentile(inside, ATLAS_IMAGE_PERCENTILE)) if inside.size else 0.0
    if top <= 0:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip(values / top, 0.0, 1.0)


def atlas_image_picture(
    ctx: Workspace, state: StackState, kinds: tuple[str, ...], position_mm: float,
    *, angles: Angles | None = None,
) -> Image.Image | None:
    """The atlas images *kinds* on the native plane grid; None for the template alone.

    At *angles* (:func:`langslice.core.state.plane_angles`: a section's own,
    or the stack's when None).

    ``template`` alone is left to the renderers' own reference path (the pixels
    every earlier picture showed). Any other atlas image is stretched inside
    the atlas anatomy. Two images are added in their two colours
    (:func:`langslice.core.appearance.channel_colors`); no image is a black
    plane (the lines alone).
    """
    images = tuple(kind for kind in kinds if kind != BORDERS)
    if images == (TEMPLATE,):
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
        if kind != TEMPLATE:
            return _image_plane(ctx, state, kind, position_mm, labels, (pitch, yaw))
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
    *position_mm* and *angles* (the stack's when None); a region with a side
    (``"CTX:left"``) counts only that side of the atlas picture, as it is
    drawn."""
    from langslice.core.atlas.sides import restrict, split_side

    if not options.regions:
        return []
    pitch, yaw = plane_angles(state, angles)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=cast(Plane, state.plane),
        pitch_deg=pitch, yaw_deg=yaw,
    ))
    # An atlas-only picture has no section: a side is the picture's own.
    left = regions_left(ctx.atlas, options.regions, position_mm, state.plane, pitch, yaw,
                        np.eye(2))
    return [name for name, ids in options.regions
            if restrict(np.isin(labels, list(ids)), split_side(name)[1], left).any()]


def _crop_fraction(image: Image.Image, zoom: tuple[float, ...]) -> Image.Image:
    from langslice.core.canvas import zoom_box
    from langslice.core.picture_geometry import cropped_geometry

    box = zoom_box(list(zoom), image.size)
    return cropped_geometry(image.crop(box), image, box)


# --- section and atlas pictures -----------------------------------------


def framed_section(
    ctx: Workspace, state: StackState, record: SliceState, options: DisplayOptions,
    *, long_edge: int | None = None, look: Look | str = "options",
) -> Image.Image:
    """One section as corrected, the whole image, in the call's look and zoom.

    *long_edge* None is the call's picture size (``options.long_edge``);
    *look* overrides the options' (``preprocess`` draws a target's
    appearance before and after its write). A zoom renders the section
    larger (never past its working copy) and crops the requested fraction,
    so it magnifies.
    """
    long_edge = long_edge or options.long_edge
    drawn: Look = options.look(state, record) if look == "options" else cast(Look, look)
    whole = render_slice(ctx, record, long_edge=long_edge, look=drawn)
    if options.full_view:
        return whole
    window = options.window(whole.size)
    x0, y0, x1, y1 = window
    extent = max(abs(x1 - x0), abs(y1 - y0), 1e-3)
    larger = render_slice(
        ctx, record, long_edge=int(math.ceil(long_edge / min(1.0, extent))), look=drawn,
    )
    return _crop_fraction(larger, tuple(window))


def framed_atlas(
    ctx: Workspace, state: StackState, position_mm: float, options: DisplayOptions,
    *, long_edge: int | None = None, um_per_px: float | None = None,
    angles: Angles | None = None,
) -> Image.Image:
    """The atlas at *position_mm*, framed to its anatomy, with the call's lines.

    At *angles*: a section's own for a picture beside that section, the
    stack's view angles for a picture without a section; None, the stack's
    one angle.

    At most *long_edge* (None: ``options.long_edge``) and never upsampled
    past the plane's own voxels, unless *um_per_px*: then drawn at exactly
    that many micrometres per pixel, lines included, for a picture that puts
    the atlas beside a section drawn at the same scale
    (:mod:`langslice.core.scale`).
    """
    from langslice.core.atlas.render import atlas_um_per_px
    from langslice.core.atlas_fetch import atlas_mask, atlas_section, atlas_sized
    from langslice.core.scale import resized

    edge = int(long_edge or options.long_edge)

    def sized(picture: Image.Image) -> Image.Image:
        if um_per_px is not None:
            return resized(picture, atlas_um_per_px(ctx.atlas) / float(um_per_px))
        return atlas_sized(picture, edge)

    pitch, yaw = plane_angles(state, angles)
    if options.atlas_images == ("template",) and not options.lines and options.full_view:
        from langslice.core.picture_geometry import cropped_geometry

        whole = atlas_section(ctx, state, position_mm, frame=True, angles=(pitch, yaw))
        return cropped_geometry(sized(whole), whole)
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

        whole = sized(picture.crop(box)).size
        inner = zoom_box(options.window(whole), (box[2] - box[0], box[3] - box[1]))
        box = (box[0] + inner[0], box[1] + inner[1], box[0] + inner[2], box[1] + inner[3])
    cropped = picture.crop(box)
    shown = sized(cropped).convert("RGB")
    from langslice.core.picture_geometry import cropped_geometry

    picture.info["pixel_to_native"] = np.eye(3).tolist()
    cropped_geometry(shown, picture, box)
    if not options.lines:
        return shown
    factor = shown.width / float(cropped.width)
    canvas = np.asarray(shown, dtype=np.uint8).copy()
    context: list[np.ndarray] = []
    if options.borders:
        context = region_outlines(ctx.atlas, position_mm, depth=options.depth, plane=plane,
                                  pitch_deg=pitch, yaw_deg=yaw)
    origin = (box[0], box[1])
    region_lines: list[np.ndarray] = []
    if options.regions:
        # An atlas-only picture has no section: a side is the picture's own.
        left = regions_left(ctx.atlas, options.regions, position_mm, plane,
                            pitch, yaw, np.eye(2))
        region_lines = region_polys(labels, options.regions, left)
    # One colour for the whole picture, resolved against this atlas plate's
    # own pixels (there is no section here to judge contrast against).
    drawn_polys = context + region_lines
    coverage = (
        line_coverage(canvas.shape[:2], drawn_polys, thickness=options.border_thickness,
                     origin=origin, factor=factor)
        if drawn_polys else None
    )
    rgb, _, border_word = resolve_border_color(
        options.border_color, options.border_thickness, rgb=canvas, coverage=coverage,
    )
    if context:
        _draw_polys(canvas, context, rgb, thickness=options.border_thickness, origin=origin,
                    factor=factor, alpha=REGION_CONTEXT_ALPHA if options.regions else 1.0)
    if region_lines:
        _draw_polys(canvas, region_lines, rgb,
                    thickness=options.border_thickness * HIGHLIGHT_WIDTH, origin=origin,
                    factor=factor)
    result = Image.fromarray(canvas, mode="RGB")
    result.info["pixel_to_native"] = shown.info["pixel_to_native"]
    if drawn_polys:
        result.info["border_word"] = border_word
    return result


def atlas_caption(state: StackState, position_mm: float, options: DisplayOptions,
                  *, angles: Angles | None = None, median: bool = False) -> str:
    """The label burned into an atlas picture drawn at *angles* (the stack's
    when None); *median*: drawn at the median of differing sections' angles,
    which the label says (:func:`langslice.core.captions.view_angles_label`)."""
    from langslice.core.captions import view_angles_label

    shown = view_angles_label(plane_angles(state, angles), median=median)
    name = options.atlas_name()
    extra = "" if name == TEMPLATE else f" {name}"
    if options.regions:
        extra += " regions " + ",".join(name for name, _ids in options.regions)
    return f"atlas {position_mm:.2f} mm{shown}{extra}"


__all__ = [
    "BORDERS", "DisplayOptions", "MODE_RULES",
    "atlas_caption", "atlas_image_picture", "available_atlas_channels",
    "default_options", "framed_atlas", "framed_section", "regions_in_plane",
]
