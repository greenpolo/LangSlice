"""Display options: one set of arguments for every tool that returns a picture.

Every picture tool takes the same nine arguments — ``mode``, ``zoom``,
``section_image``, ``atlas_image``, ``atlas_opacity``, ``regions``,
``outlines``, ``border_color``, ``border_thickness`` — validated here once into
a :class:`DisplayOptions` and drawn by the shared renderers (the physical
canvas in :func:`langslice.linear.render.physical_views`, the tissue-framed
pictures below). Options belong to one call: nothing here writes state, so a
call's options never change a stored default (the section's appearance lives
on the state and changes only through ``preprocess``).

Atlas images are host-dependent: every host has ``ara`` (the BrainGlobe
reference) and ``borders`` (the atlas regions drawn as lines); ``nissl``
(ABBA's cached Allen Nissl, :mod:`langslice.deformable.abba_atlas`) exists
only where that cache is installed and matches the run's atlas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.affine import resize_long_edge
from langslice.atlas.core import get_reference_slice
from langslice.atlas.render import (
    annotation_slice,
    family_labels,
    family_outlines,
    outer_outline,
)
from langslice.image_prep import mask_box
from langslice.linear.appearance import Look, section_settings, view_look
from langslice.linear.render import (
    AUTO_RESOLUTION,
    OUTLINE_LAYERS,
    PICTURE_EDGES,
    REGION_CONTEXT_ALPHA,
    RESOLUTION_RANGE,
    _draw_polys,
    normalize_border_style,
    picture_edge,
    region_polys,
    render_slice,
    resolution_level,
)
from langslice.linear.state import SliceState, StackState
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

#: Atlas images a picture may show, in the order the docs list them.
ATLAS_IMAGES: tuple[str, ...] = ("ara", "borders", "nissl")
#: The section's current appearance (``section_image``'s default).
CURRENT = "current"
#: Intensity percentile mapped to white when a Nissl plane is shown.
NISSL_PERCENTILE = 99.5
FULL_VIEW = (0.0, 0.0, 1.0, 1.0)

#: The shared argument text, appended to every picture tool's docstring
#: (``{modes}`` is filled per tool).
DISPLAY_DOC = """
        Display options (the same on every picture tool; this call only,
        never a stored setting):
            mode: {modes}
            zoom: [x0, y0, x1, y1] fractions, cropped before sizing so it
                magnifies; empty is all.
            section_image: "current" appearance or one raw channel by name.
            atlas_image: "ara" (reference), "borders" (region lines) or
                "nissl" (only on hosts with ABBA's atlas).
            atlas_opacity: 0..1, the atlas image under the lines in overlay.
            regions: acronyms or ids, descendants included: only their
                borders at full strength, the outlines layer faint behind.
            outlines: "all", "outer" or "none"; empty is the mode's default
                (all on the physical canvas, none on framed pictures).
            border_color: named or #RRGGBB. border_thickness: 0.25..8 px.
"""

#: The ``resolution`` argument's line, present only when the host level is
#: "auto" (:func:`resolution_argument`).
RESOLUTION_DOC = (
    "            resolution: long edge in pixels of each picture this call "
    "returns (of each section\n"
    "                tile in view_stack), {low}..{high}; 0 is {default}. "
    "Never larger than the\n"
    "                image it is drawn from.\n"
)


def with_display_doc(modes: str):
    """Append :data:`DISPLAY_DOC` (this tool's *modes*) to a tool's docstring."""

    def apply(tool: Any) -> Any:
        tool.__doc__ = (tool.__doc__ or "").rstrip() + "\n" + DISPLAY_DOC.format(modes=modes)
        return tool

    return apply


def resolution_doc() -> str:
    """:data:`RESOLUTION_DOC` with its numbers filled in."""
    low, high = RESOLUTION_RANGE
    return RESOLUTION_DOC.format(low=low, high=high, default=PICTURE_EDGES[AUTO_RESOLUTION][1])


def resolution_argument(tool: Any, level: str) -> Any:
    """*tool* with its ``resolution`` argument only where the host level is "auto".

    Every picture tool is written with ``resolution: int = 0``. At "auto" the
    tool is returned with the argument's line added to its docstring (after
    the display options, or before ``Returns:`` when it lists its own); at
    every other level a wrapper is returned whose signature, annotations and
    docstring have no ``resolution`` at all, so the model never sees it, and
    the body always receives 0. A tool without the argument is returned as
    it is, its docstring gaining the line at "auto" when it documents the
    display options as per-entry keys (``adjust_transforms``).
    """
    import functools
    import inspect

    signature = inspect.signature(tool, eval_str=True)
    if "resolution" not in signature.parameters:
        # A tool taking the display options per entry (`adjust_transforms`)
        # documents the entry key the same way.
        if level == AUTO_RESOLUTION and "Display options" in (tool.__doc__ or ""):
            tool.__doc__ = (tool.__doc__ or "").rstrip("\n ") + "\n" + resolution_doc()
        return tool
    if level == AUTO_RESOLUTION:
        doc = (tool.__doc__ or "").rstrip("\n ") + "\n"
        if "Display options" not in doc and "\n        Returns:" in doc:
            head, tail = doc.split("\n        Returns:", 1)
            doc = head.rstrip("\n") + "\n" + resolution_doc() + "\n        Returns:" + tail
        else:
            doc += resolution_doc()
        tool.__doc__ = doc
        return tool

    @functools.wraps(tool)
    def without(*args: Any, **kwargs: Any) -> Any:
        kwargs.pop("resolution", None)
        return tool(*args, **kwargs)

    without.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[p for name, p in signature.parameters.items() if name != "resolution"],
    )
    without.__annotations__ = {
        name: value for name, value in inspect.get_annotations(tool, eval_str=True).items()
        if name != "resolution"
    }
    del without.__wrapped__  # the signature above is the whole truth
    return without


def clamp_resolution(value: Any) -> tuple[int | None, str]:
    """``(long edge, note)`` for a requested ``resolution``; None for "default".

    0, None and "" are the default. A number outside :data:`RESOLUTION_RANGE`
    is clamped into it and *note* says so; a value that is not a number
    raises ``ValueError``.
    """
    if value in (None, "", 0):
        return None, ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("resolution must be a number of pixels")
    low, high = RESOLUTION_RANGE
    edge = int(round(number))
    if edge < low or edge > high:
        clamped = min(high, max(low, edge))
        return clamped, f"resolution {edge} is outside {low}..{high}; used {clamped}"
    return edge, ""


@dataclass(frozen=True)
class DisplayOptions:
    """One call's validated display options."""

    mode: str
    zoom: tuple[float, ...]
    section_image: str
    atlas_image: str
    atlas_opacity: float
    #: ``(name as asked, ids including descendants)`` per highlighted region.
    regions: tuple[tuple[str, frozenset[int]], ...]
    outlines: str
    border_color: str
    border_thickness: float
    #: Long edge of each picture this call returns
    #: (:func:`langslice.linear.render.picture_edge`).
    long_edge: int = PICTURE_EDGES["low"][1]
    #: The clamped ``resolution`` the agent asked for ("auto" only; None when
    #: it asked for none). A contact sheet sizes its tiles by it.
    resolution: int | None = None
    #: Whether the run's level is "auto" (the payload then echoes the size).
    auto: bool = False
    #: Why the asked resolution was changed ("" when it was not).
    resolution_note: str = ""

    @property
    def full_view(self) -> bool:
        return not self.zoom or self.zoom == FULL_VIEW

    @property
    def window(self) -> list[float]:
        """The zoom as the renderers take it (empty = whole)."""
        return [] if self.full_view else list(self.zoom)

    def echo(self) -> dict[str, Any]:
        """The options as a payload's ``view`` field."""
        return {
            "mode": self.mode,
            "zoom": list(self.zoom or FULL_VIEW),
            "section_image": self.section_image,
            "atlas_image": self.atlas_image,
            "atlas_opacity": self.atlas_opacity,
            **({"regions": [name for name, _ids in self.regions]} if self.regions else {}),
            "outlines": self.outlines,
            "border_color": self.border_color,
            "border_thickness": self.border_thickness,
            # The size is the agent's to choose only at "auto"; any other
            # level keeps it out of model-facing text.
            **({"resolution": self.resolution or self.long_edge} if self.auto else {}),
            **({"resolution_note": self.resolution_note} if self.resolution_note else {}),
        }

    def look(self, state: StackState, record: SliceState) -> Look:
        """The look a picture of *record* is drawn in."""
        return view_look(state, record, self.section_image)


def available_atlas_images(ctx: EngineContext) -> tuple[str, ...]:
    """The atlas images this host can draw."""
    return ATLAS_IMAGES if ctx.abba_atlas is not None else ("ara", "borders")


def display_facts(
    ctx: EngineContext, state: StackState,
) -> dict[str, Any]:
    """The job statement's display facts: raw channels and atlas images here.

    ``channels`` is one list when every section shares it, else a mapping
    filename -> names. Unreadable files contribute nothing (the seed will
    report them).
    """
    named: dict[str, list[str]] = {}
    for record in state.in_order():
        try:
            named[record.id] = list(ctx.section_channels(record.id)[0])
        except Exception:
            continue
    distinct = {tuple(names) for names in named.values()}
    channels: Any = list(next(iter(distinct))) if len(distinct) == 1 else named
    return {"channels": channels or None, "atlas_images": available_atlas_images(ctx)}


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "error": code, "message": message, **extra}


def parse_display(
    ctx: EngineContext,
    state: StackState,
    *,
    modes: tuple[str, ...],
    mode: str,
    zoom: list[float] | None,
    section_image: str,
    atlas_image: str,
    atlas_opacity: float,
    regions: list[str] | None,
    outlines: str,
    border_color: str,
    border_thickness: float,
    sections: list[SliceState] = (),  # type: ignore[assignment]
    zoom_ok: bool = True,
    framed_modes: tuple[str, ...] = (),
    resolution: Any = 0,
) -> DisplayOptions | dict[str, Any]:
    """Validate one call's options; a refusal dict names every bad argument's fix.

    *sections* are the sections the call shows; a named ``section_image``
    channel must exist on each. *zoom_ok* False refuses any zoom but the full
    view (pictures whose pieces are framed independently). An empty
    *outlines* is the mode's own default: ``none`` for the tissue-framed
    pictures in *framed_modes*, ``all`` on the physical canvas.

    *resolution* is read only when the run's level is "auto"
    (:func:`clamp_resolution`); every other level sizes the pictures itself.
    """
    auto = resolution_level(ctx) == AUTO_RESOLUTION
    asked: int | None = None
    note = ""
    if auto:
        try:
            asked, note = clamp_resolution(resolution)
        except (TypeError, ValueError):
            low, high = RESOLUTION_RANGE
            return _error("BAD_RESOLUTION",
                          f"resolution must be a whole number of pixels, {low}..{high}")
    view = str(mode or modes[0]).strip().lower()
    if view not in modes:
        return {"status": "error", "error": "BAD_MODE", "modes": list(modes)}
    try:
        window = tuple(float(value) for value in (zoom or []))
    except (TypeError, ValueError):
        return _error("BAD_ZOOM", "zoom must be [x0, y0, x1, y1] fractions")
    if window and (len(window) != 4 or not all(math.isfinite(v) for v in window)):
        return {"status": "error", "error": "BAD_ZOOM",
                "expected": "[x0, y0, x1, y1] as fractions of the picture"}
    if window and window != FULL_VIEW and not zoom_ok:
        return {"status": "error", "error": "ZOOM_UNSUPPORTED", "mode": view,
                "supported_zoom": list(FULL_VIEW)}
    layer = str(outlines or "").strip().lower() or (
        "none" if view in framed_modes else "all"
    )
    if layer not in OUTLINE_LAYERS:
        return {"status": "error", "error": "BAD_OUTLINES", "layers": list(OUTLINE_LAYERS)}
    try:
        rgb, thickness = normalize_border_style(border_color, border_thickness)
    except ValueError as exc:
        return _error("INVALID_BORDER_STYLE", str(exc))
    try:
        opacity = float(atlas_opacity)
    except (TypeError, ValueError):
        return _error("BAD_ARGS", "atlas_opacity must be a number from 0 to 1")
    if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
        return _error("BAD_ARGS", "atlas_opacity must be a number from 0 to 1")
    kind = str(atlas_image or "ara").strip().lower()
    if kind not in ATLAS_IMAGES:
        return {"status": "error", "error": "BAD_ATLAS_IMAGE",
                "atlas_images": list(available_atlas_images(ctx))}
    if kind not in available_atlas_images(ctx):
        return _error(
            "ATLAS_IMAGE_UNAVAILABLE",
            f"The {kind} atlas image needs ABBA's cached Allen atlas matching "
            f"{state.atlas}; this host has none.",
            atlas_image=kind, available=list(available_atlas_images(ctx)),
        )
    picked = str(section_image or CURRENT).strip()
    if picked != CURRENT:
        missing = {}
        for record in sections:
            names, _planes = ctx.section_channels(record.id)
            if picked not in names:
                missing[record.id] = list(names)
        if missing:
            return _error(
                "UNKNOWN_CHANNEL", f"No channel named {picked!r}.",
                channels=missing,
            )
    resolved: list[tuple[str, frozenset[int]]] = []
    if regions:
        if not isinstance(regions, (list, tuple)):
            return _error("BAD_ARGS", "regions must be a list of acronyms or ids")
        from langslice.deformable.atlas_images import resolve_structures, with_descendants

        if not getattr(ctx.atlas, "structures", None):
            return _error("NO_STRUCTURES", "This atlas has no region hierarchy.")
        try:
            for name in regions:
                ids = with_descendants(ctx.atlas, resolve_structures(ctx.atlas, [name]))
                resolved.append((str(name).strip(), ids))
        except ValueError as exc:
            return _error("UNKNOWN_REGIONS", str(exc))
    return DisplayOptions(
        mode=view,
        zoom=window,
        section_image=picked,
        atlas_image=kind,
        atlas_opacity=opacity,
        regions=tuple(resolved),
        outlines=layer,
        border_color="#" + "".join(f"{channel:02x}" for channel in rgb),
        border_thickness=thickness,
        long_edge=picture_edge(ctx, asked),
        resolution=asked,
        auto=auto,
        resolution_note=note,
    )


# --- atlas images --------------------------------------------------------


def atlas_plane_picture(
    ctx: EngineContext, state: StackState, kind: str, position_mm: float,
) -> Image.Image | None:
    """The atlas image *kind* on the native plane grid; None for ``ara``.

    ``ara`` is left to the renderers' own reference path (the pixels every
    earlier picture showed). ``nissl`` is ABBA's Nissl, percentile-stretched
    inside the atlas anatomy; ``borders`` is the family boundaries as white
    lines on black.
    """
    if kind == "ara":
        return None
    plane = cast(Plane, state.plane)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
    ))
    if kind == "nissl":
        source = ctx.abba_atlas
        if source is None:
            raise ValueError("The nissl atlas image needs ABBA's cached Allen atlas")
        values = np.asarray(source.sample_plane(
            "NISSL", ctx.atlas, position_mm, plane, state.pitch_deg, state.yaw_deg,
        ), dtype=np.float32)
        inside = values[(labels > 0) & (values > 0)]
        top = float(np.percentile(inside, NISSL_PERCENTILE)) if inside.size else 0.0
        gray = np.zeros(values.shape, dtype=np.uint8) if top <= 0 else (
            np.clip(values / top, 0.0, 1.0) * 255.0
        ).astype(np.uint8)
        return Image.fromarray(gray, mode="L")
    if kind == "borders":
        families = family_labels(labels, ctx.atlas)
        edges = np.zeros(families.shape, dtype=bool)
        edges[:-1, :] |= families[:-1, :] != families[1:, :]
        edges[1:, :] |= families[1:, :] != families[:-1, :]
        edges[:, :-1] |= families[:, :-1] != families[:, 1:]
        edges[:, 1:] |= families[:, 1:] != families[:, :-1]
        return Image.fromarray(edges.astype(np.uint8) * 255, mode="L")
    raise ValueError(f"Unknown atlas image {kind!r}")


def regions_in_plane(
    ctx: EngineContext, state: StackState, position_mm: float, options: DisplayOptions,
) -> list[str]:
    """The highlighted regions with at least one pixel in the plane at *position_mm*."""
    if not options.regions:
        return []
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=cast(Plane, state.plane),
        pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
    ))
    return [name for name, ids in options.regions if np.isin(labels, list(ids)).any()]


def _crop_fraction(image: Image.Image, zoom: tuple[float, ...]) -> Image.Image:
    from langslice.linear.render import zoom_box

    return image.crop(zoom_box(list(zoom), image.size))


# --- tissue-framed pictures ----------------------------------------------


def framed_section(
    ctx: EngineContext, state: StackState, record: SliceState, options: DisplayOptions,
    *, long_edge: int | None = None, target: str = "view",
) -> Image.Image:
    """One section as corrected, tissue-framed, in the call's look and zoom.

    *long_edge* None is the call's picture size (``options.long_edge``).
    *target* ``"fit"`` shows the fit appearance instead of the view one (a
    named raw channel wins either way). A zoom renders the framed section
    larger (never past its working copy) and crops the requested fraction,
    so it magnifies.
    """
    long_edge = long_edge or options.long_edge
    look = options.look(state, record)
    if target == "fit" and options.section_image == CURRENT:
        look = section_settings(state, "fit", record.id)
    if options.full_view:
        return render_slice(ctx, record, long_edge=long_edge, frame=True, look=look)
    x0, y0, x1, y1 = options.zoom
    extent = max(abs(x1 - x0), abs(y1 - y0), 1e-3)
    larger = render_slice(
        ctx, record, long_edge=int(math.ceil(long_edge / min(1.0, extent))), frame=True, look=look,
    )
    return _crop_fraction(larger, options.zoom)


def framed_atlas(
    ctx: EngineContext, state: StackState, position_mm: float, options: DisplayOptions,
    *, long_edge: int | None = None, fill: bool = False,
) -> Image.Image:
    """The atlas at *position_mm*, framed to its anatomy, with the call's lines.

    At most *long_edge* (None: ``options.long_edge``) and never upsampled
    past the plane's own voxels, unless *fill*: then drawn at exactly
    *long_edge*, lines included, for a picture that puts the atlas beside a
    section of that size. With the default options (``ara``, no lines, no
    regions, no zoom) and no *fill* this is the picture ``view_atlas`` sends.
    """
    from langslice.linear.atlas_fetch import atlas_mask, atlas_section, atlas_sized

    edge = int(long_edge or options.long_edge)

    def sized(picture: Image.Image) -> Image.Image:
        return resize_long_edge(picture, edge) if fill else atlas_sized(picture, edge)

    drawn = options.outlines != "none" or bool(options.regions)
    if options.atlas_image == "ara" and not drawn and options.full_view:
        return sized(atlas_section(ctx, state, position_mm, frame=True))
    plane = cast(Plane, state.plane)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
    ))
    picture = atlas_plane_picture(ctx, state, options.atlas_image, position_mm)
    if picture is None:
        picture = get_reference_slice(
            ctx.atlas, position_mm, plane=plane,
            pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
        )
    native = (labels.shape[1], labels.shape[0])
    if picture.size != native:
        picture = picture.resize(native, Image.Resampling.BILINEAR)
    try:
        mask = atlas_mask(ctx, state, position_mm, picture.size) > 0
    except Exception:
        mask = labels > 0
    box = mask_box(picture.size, mask) or (0, 0, picture.width, picture.height)
    if not options.full_view:
        from langslice.linear.render import zoom_box

        inner = zoom_box(list(options.zoom), (box[2] - box[0], box[3] - box[1]))
        box = (box[0] + inner[0], box[1] + inner[1], box[0] + inner[2], box[1] + inner[3])
    cropped = picture.crop(box)
    shown = sized(cropped).convert("RGB")
    if not drawn:
        return shown
    factor = shown.width / float(cropped.width)
    canvas = np.asarray(shown, dtype=np.uint8).copy()
    rgb, _ = normalize_border_style(options.border_color, options.border_thickness)
    context: list[np.ndarray] = []
    if options.outlines != "none":
        draw = outer_outline if options.outlines == "outer" else family_outlines
        context = [poly for _color, poly in draw(
            ctx.atlas, position_mm, plane=plane, pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
        )]
    origin = (box[0], box[1])
    if context:
        _draw_polys(canvas, context, rgb, thickness=options.border_thickness, origin=origin,
                    factor=factor, alpha=REGION_CONTEXT_ALPHA if options.regions else 1.0)
    if options.regions:
        _draw_polys(canvas, region_polys(labels, options.regions), rgb,
                    thickness=options.border_thickness, origin=origin, factor=factor)
    return Image.fromarray(canvas, mode="RGB")


def atlas_caption(state: StackState, position_mm: float, options: DisplayOptions) -> str:
    """The label burned into an atlas picture."""
    angles = (
        f" pitch {state.pitch_deg:.1f} yaw {state.yaw_deg:.1f}" if state.is_oblique else ""
    )
    extra = "" if options.atlas_image == "ara" else f" {options.atlas_image}"
    if options.regions:
        extra += " regions " + ",".join(name for name, _ids in options.regions)
    return f"atlas {position_mm:.2f} mm{angles}{extra}"


def default_options(
    mode: str, *, outlines: str = "all", long_edge: int = PICTURE_EDGES["low"][1],
) -> DisplayOptions:
    """The options a tool uses when it is called with none (internal callers)."""
    return DisplayOptions(
        mode=mode, zoom=(), section_image=CURRENT, atlas_image="ara", atlas_opacity=0.0,
        regions=(), outlines=outlines, border_color="#ffff00", border_thickness=0.5,
        long_edge=long_edge,
    )
