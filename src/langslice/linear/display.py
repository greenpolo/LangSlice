"""Picture options: ONE argument, ``view``, the same on every tool that returns a picture.

``view`` is a dict (:class:`langslice.linear.arguments.View`) with these keys,
validated here once into a :class:`DisplayOptions` and drawn by the shared
renderers (the physical canvas in
:func:`langslice.linear.render.physical_views`, the tissue-framed pictures
below):

- ``mode`` — per tool (its :class:`Profile`), default per tool.
- ``channels`` — what of the SECTION is shown: one or more raw channel names
  (one is grayscale, unmodified; several are each stretched by percentile and
  added in distinct colours, ABBA's multichannel display), or ONE version:
  ``view`` (the agent's own appearance, the default) or ``fit`` (what
  registration reads). Raw channels and a version cannot be mixed.
- ``atlas_channels`` — what of the ATLAS is shown: any of ``ara`` (the
  reference template), ``nissl`` (ABBA's cached Allen Nissl, only where that
  cache is installed and matches the run's atlas,
  :mod:`langslice.deformable.abba_atlas`) and ``borders`` (the region lines).
  Images are drawn under the section at ``atlas_opacity`` (overlay modes) or
  as the atlas picture; two images are added in two colours. No ``borders``
  means no lines. The defaults per mode reproduce the pictures each mode drew
  before ``view`` existed.
- ``atlas_opacity``, ``regions`` (one side allowed, ``"CTX:left"``),
  ``outlines`` (which lines ``borders`` draws: all or outer), ``border_color``,
  ``border_thickness``, ``zoom``, ``deformation`` (placement pictures: draw the
  applied warp, or ``none`` for the linear placement alone), ``resolution``
  (only where the host's image resolution is "auto").

Strict: a key that means nothing for the tool or the mode (atlas keys on a
picture with no atlas, ``outlines`` without ``borders``...) is refused with
the reason; unknown keys are refused before a tool runs
(:func:`langslice.linear.arguments.argument_refusal`). Options belong to one
call: nothing here writes state (a section's appearance changes only through
``preprocess``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.affine import resize_long_edge
from langslice.atlas.core import get_reference_slice
from langslice.atlas.render import (
    annotation_slice,
    family_outlines,
    outer_outline,
)
from langslice.image_prep import mask_box
from langslice.linear.appearance import (
    Look,
    channel_colors,
    section_settings,
)
from langslice.linear.arguments import View, ViewAuto
from langslice.linear.render import (
    AUTO_RESOLUTION,
    PICTURE_EDGES,
    REGION_CONTEXT_ALPHA,
    RESOLUTION_RANGE,
    _draw_polys,
    normalize_border_style,
    picture_edge,
    region_polys,
    regions_left,
    render_slice,
    resolution_level,
)
from langslice.linear.state import SliceState, StackState
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

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
#: Every key ``view`` may carry (the strict wrapper refuses anything else).
VIEW_KEYS: tuple[str, ...] = tuple(ViewAuto.__annotations__)


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
class Profile:
    """One tool's picture options: its modes and which keys it can use at all."""

    #: The tool's modes; the first is its default.
    modes: tuple[str, ...]
    #: ``channels`` applies (the tool draws the section from its channels).
    channels: bool = True
    #: The atlas keys apply in some mode of this tool.
    atlas: bool = True
    #: ``deformation`` applies (the tool draws a stored placement).
    deformation: bool = False
    #: Modes in which ``zoom`` applies (None: every mode).
    zoom_modes: tuple[str, ...] | None = None
    #: Per-mode ``atlas_channels`` defaults that differ from :data:`MODE_RULES`.
    atlas_defaults: dict[str, tuple[str, ...]] = field(default_factory=dict)
    #: Why ``channels`` means nothing here (when ``channels`` is False).
    channels_reason: str = "this tool's pictures are not drawn from the section's channels"
    #: Why ``deformation`` means nothing here.
    deformation_reason: str = "this tool does not draw a stored placement"
    #: Modes that draw the section under its stored placement (None: every
    #: mode); ``deformation`` means nothing in the others.
    deformation_modes: tuple[str, ...] | None = None

    def atlas_default(self, mode: str) -> tuple[str, ...]:
        return self.atlas_defaults.get(mode, MODE_RULES[mode].atlas_default)


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
    #: a name may carry a side (``"CTX:left"``, :mod:`langslice.atlas.sides`),
    #: which the renderers resolve against the picture's placement.
    regions: tuple[tuple[str, frozenset[int]], ...]
    #: Which lines ``borders`` draws: ``all`` or ``outer``.
    outlines: str
    border_color: str
    border_thickness: float
    #: ``applied`` or ``none`` (placement pictures).
    deformation: str = "applied"
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
    #: The keys the call gave (the echo reports what applied, not defaults
    #: that mean nothing in this mode).
    given: frozenset[str] = frozenset()
    #: Whether this tool takes ``deformation``.
    has_deformation: bool = False

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
        if len(self.channels) == 1:
            return {"channel": self.channels[0]}
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
            return f"  [raw {self.channels[0]}]"
        return "  [" + " + ".join(f"{name} {word}" for name, word in
                                   self.channel_colors().items()) + "]"

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
        if rule.section:
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


def available_atlas_channels(ctx: EngineContext) -> tuple[str, ...]:
    """The atlas channels this host can draw."""
    return ATLAS_CHANNELS if ctx.abba_atlas is not None else ("ara", "borders")


def display_facts(
    ctx: EngineContext, state: StackState,
) -> dict[str, Any]:
    """The job statement's display facts: raw channels and atlas channels here.

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
    return {"channels": channels or None, "atlas_channels": available_atlas_channels(ctx)}


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


def view_schema(tool: Any, level: str) -> Any:
    """*tool* with ``view`` typed for this run: :class:`ViewAuto` only at "auto".

    Every picture tool is written with ``view: View``; at "auto" a wrapper is
    returned whose signature and annotations carry :class:`ViewAuto` (the
    ``resolution`` key), so the model sees that key only where it may use it.
    At every other level a ``resolution`` key is refused by the strict
    argument check (:data:`langslice.linear.arguments.KEY_NOTES`).
    """
    import functools
    import inspect

    signature = inspect.signature(tool, eval_str=True)
    if "view" not in signature.parameters or level != AUTO_RESOLUTION:
        return tool

    @functools.wraps(tool)
    def with_size(*args: Any, **kwargs: Any) -> Any:
        return tool(*args, **kwargs)

    with_size.__signature__ = signature.replace(  # type: ignore[attr-defined]
        parameters=[p.replace(annotation=ViewAuto) if name == "view" else p
                    for name, p in signature.parameters.items()],
    )
    with_size.__annotations__ = {
        **inspect.get_annotations(tool, eval_str=True), "view": ViewAuto,
    }
    del with_size.__wrapped__  # the signature above is the whole truth
    return with_size


def _error(code: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"status": "error", "error": code, "message": message, **extra}


def _names(value: Any, key: str) -> list[str] | dict[str, Any]:
    """A list of names (a bare string is one), or the refusal."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not all(isinstance(v, str) for v in value):
        return _error("BAD_VIEW", f"view.{key} must be a list of names")
    out: list[str] = []
    for name in value:
        cleaned = name.strip()
        if cleaned and cleaned not in out:
            out.append(cleaned)
    return out


def parse_view(
    ctx: EngineContext,
    state: StackState,
    view: Any,
    profile: Profile,
    *,
    sections: list[SliceState] = (),  # type: ignore[assignment]
) -> DisplayOptions | dict[str, Any]:
    """Validate one call's ``view``; a refusal dict names every problem's fix.

    *sections* are the sections the call shows; named raw channels must
    exist on each. A key that means nothing for *profile* or for the mode is
    refused (``VIEW_KEY_UNUSED``) rather than dropped.
    """
    if view is None:
        view = {}
    if not isinstance(view, Mapping):
        return _error("BAD_VIEW", "view must be an object of picture options")
    given = {key for key, value in view.items() if value is not None}
    unknown = sorted(key for key in given if key not in VIEW_KEYS)
    if unknown:
        return _error("UNKNOWN_ARGUMENTS", f"Unknown view key(s): {', '.join(unknown)}.",
                      accepted=list(VIEW_KEYS))
    auto = resolution_level(ctx) == AUTO_RESOLUTION
    if "resolution" in given and not auto:
        return _error("RESOLUTION_FIXED", "The user fixed the picture size for this run, so "
                      "view.resolution cannot be set.")
    asked: int | None = None
    note = ""
    if auto:
        try:
            asked, note = clamp_resolution(view.get("resolution", 0))
        except (TypeError, ValueError):
            low, high = RESOLUTION_RANGE
            return _error("BAD_RESOLUTION",
                          f"view.resolution must be a whole number of pixels, {low}..{high}")

    mode = str(view.get("mode") or profile.modes[0]).strip().lower()
    if mode not in profile.modes:
        return _error("BAD_MODE", f"view.mode must be one of: {', '.join(profile.modes)}.",
                      modes=list(profile.modes))
    rule = MODE_RULES[mode]

    # --- which keys mean something here -------------------------------------
    unused: list[dict[str, str]] = []

    def refuse(key: str, reason: str) -> None:
        unused.append({"key": key, "reason": reason})

    atlas_keys = ("atlas_channels", "atlas_opacity", "regions", "outlines",
                  "border_color", "border_thickness")
    if "channels" in given:
        if not profile.channels:
            refuse("channels", profile.channels_reason)
        elif mode == "channels":
            refuse("channels", "channels mode shows every raw channel, each on its own")
        elif not rule.section:
            refuse("channels", f"mode {mode} draws no section")
    for key in atlas_keys:
        if key not in given:
            continue
        if not profile.atlas:
            refuse(key, "this tool's pictures show no atlas")
        elif not rule.atlas:
            refuse(key, f"mode {mode} draws no atlas")
    if "deformation" in given:
        if not profile.deformation:
            refuse("deformation", profile.deformation_reason)
        elif profile.deformation_modes is not None and mode not in profile.deformation_modes:
            refuse("deformation", f"mode {mode} does not draw the section under its "
                   "placement; " + ", ".join(profile.deformation_modes) + " do")

    # --- values -----------------------------------------------------------
    channels: tuple[str, ...] = ()
    version = "view"
    if "channels" in given and not any(item["key"] == "channels" for item in unused):
        picked = _names(view.get("channels"), "channels")
        if isinstance(picked, dict):
            return picked
        versions = [name for name in picked if name.lower() in VERSIONS]
        raw = [name for name in picked if name.lower() not in VERSIONS]
        if versions and raw:
            return _error("BAD_CHANNELS", "view.channels is either raw channel names or one "
                          "version (view or fit), not both.")
        if len(versions) > 1:
            return _error("BAD_CHANNELS", "view.channels takes one version: view or fit.")
        if raw:
            if len(raw) > MAX_OVERLAY_CHANNELS:
                return _error("BAD_CHANNELS", f"At most {MAX_OVERLAY_CHANNELS} raw channels "
                              "can be overlaid.")
            missing = {}
            for record in sections:
                names, _planes = ctx.section_channels(record.id)
                lacking = [name for name in raw if name not in names]
                if lacking:
                    missing[record.id] = {"unknown": lacking, "channels": list(names)}
            if missing:
                return _error("UNKNOWN_CHANNEL", "No such raw channel on these sections.",
                              sections=missing)
            channels, version = tuple(raw), ""
        elif versions:
            version = versions[0].lower()

    atlas_channels: tuple[str, ...] = profile.atlas_default(mode) if rule.atlas else ()
    available = available_atlas_channels(ctx)
    if "atlas_channels" in given and rule.atlas and profile.atlas:
        picked = _names(view.get("atlas_channels"), "atlas_channels")
        if isinstance(picked, dict):
            return picked
        wrong = [name for name in picked if name.lower() not in ATLAS_CHANNELS]
        if wrong:
            return _error("BAD_ATLAS_CHANNELS", f"Unknown atlas channel(s): {', '.join(wrong)}.",
                          atlas_channels=list(available))
        lowered = {name.lower() for name in picked}
        absent = [name for name in ATLAS_CHANNELS if name in lowered and name not in available]
        if absent:
            return _error("ATLAS_CHANNEL_UNAVAILABLE",
                          f"The {', '.join(absent)} atlas channel needs ABBA's cached Allen atlas "
                          f"matching {state.atlas}; this host has none.",
                          available=list(available))
        atlas_channels = tuple(name for name in ATLAS_CHANNELS if name in lowered)
    images = [kind for kind in atlas_channels if kind in ATLAS_IMAGES]
    if rule.atlas and profile.atlas:
        if rule.needs_atlas and not atlas_channels:
            return _error("BAD_ATLAS_CHANNELS", f"Mode {mode} needs at least one atlas channel.",
                          atlas_channels=list(available))
        if rule.needs_image and not images:
            return _error("BAD_ATLAS_CHANNELS", f"Mode {mode} needs an atlas image "
                          "(ara or nissl) in atlas_channels.", atlas_channels=list(available))

    opacity = DEFAULT_ATLAS_OPACITY if (rule.opacity and images) else 0.0
    if "atlas_opacity" in given and rule.atlas and profile.atlas:
        if not rule.opacity:
            refuse("atlas_opacity", f"mode {mode} draws the atlas image beside or instead of "
                   "the section, not under it")
        elif not images:
            refuse("atlas_opacity", "no atlas image (ara or nissl) is in atlas_channels")
        else:
            try:
                opacity = float(cast(Any, view.get("atlas_opacity")))
            except (TypeError, ValueError):
                return _error("BAD_VIEW", "view.atlas_opacity must be a number from 0 to 1")
            if not math.isfinite(opacity) or not 0.0 <= opacity <= 1.0:
                return _error("BAD_VIEW", "view.atlas_opacity must be a number from 0 to 1")

    layer = "all"
    if "outlines" in given and rule.atlas and profile.atlas:
        if "borders" not in atlas_channels:
            refuse("outlines", "borders is not in atlas_channels, so no outline lines are drawn")
        else:
            layer = str(view.get("outlines") or "").strip().lower()
            if layer == "none":
                return _error("BAD_VIEW", "For no lines, leave borders out of "
                              "view.atlas_channels; outlines is all or outer.")
            if layer not in OUTLINE_CHOICES:
                return _error("BAD_VIEW", "view.outlines must be all or outer.",
                              outlines=list(OUTLINE_CHOICES))

    resolved: list[tuple[str, frozenset[int]]] = []
    if "regions" in given and rule.atlas and profile.atlas:
        regions = view.get("regions")
        if not isinstance(regions, (list, tuple)):
            return _error("BAD_VIEW", "view.regions must be a list of acronyms or ids")
        from langslice.atlas.sides import has_sides
        from langslice.deformable.atlas_images import resolve_entries

        if regions and not getattr(ctx.atlas, "structures", None):
            return _error("NO_STRUCTURES", "This atlas has no region hierarchy.")
        try:
            for name in regions:
                ((_region, _side, ids),) = resolve_entries(ctx.atlas, [name])
                resolved.append((str(name).strip(), ids))
        except ValueError as exc:
            return _error("UNKNOWN_REGIONS", str(exc))
        if state.plane == "sagittal" and has_sides(regions):
            return _error("NO_SIDES", "A sagittal section lies within one hemisphere, so a "
                          "region cannot be limited to one side.")

    color = DEFAULT_BORDER_COLOR
    thickness: Any = DEFAULT_BORDER_THICKNESS
    lines = "borders" in atlas_channels or bool(resolved)
    for key in ("border_color", "border_thickness"):
        if key in given and rule.atlas and profile.atlas and not lines:
            refuse(key, "no lines are drawn: borders is not in atlas_channels and no regions "
                   "are named")
    if rule.atlas and profile.atlas:
        color = view.get("border_color", color) if "border_color" in given else color
        thickness = (view.get("border_thickness", thickness) if "border_thickness" in given
                     else thickness)
    try:
        rgb, width = normalize_border_style(color, thickness)
    except ValueError as exc:
        return _error("INVALID_BORDER_STYLE", str(exc))

    deform = "applied"
    if "deformation" in given and profile.deformation:
        deform = str(view.get("deformation") or "").strip().lower()
        if deform not in DEFORMATION_CHOICES:
            return _error("BAD_VIEW", "view.deformation must be applied or none.",
                          deformation=list(DEFORMATION_CHOICES))

    try:
        window = tuple(float(value) for value in (view.get("zoom") or []))
    except (TypeError, ValueError):
        return _error("BAD_ZOOM", "view.zoom must be [x0, y0, x1, y1] fractions")
    if window and (len(window) != 4 or not all(math.isfinite(v) for v in window)):
        return _error("BAD_ZOOM", "view.zoom must be [x0, y0, x1, y1] fractions of the picture")
    if window and window != FULL_VIEW and profile.zoom_modes is not None \
            and mode not in profile.zoom_modes:
        return _error("ZOOM_UNSUPPORTED", f"Mode {mode} of this tool takes no zoom.",
                      mode=mode, supported_zoom=list(FULL_VIEW))

    if unused:
        return {
            "status": "error", "error": "VIEW_KEY_UNUSED", "unused": unused,
            "message": "; ".join(f"view.{item['key']}: {item['reason']}" for item in unused)
            + ". Leave these keys out.",
        }
    return DisplayOptions(
        mode=mode,
        zoom=window,
        channels=channels,
        version=version,
        atlas_channels=atlas_channels,
        atlas_opacity=opacity,
        regions=tuple(resolved),
        outlines=layer,
        border_color="#" + "".join(f"{channel:02x}" for channel in rgb),
        border_thickness=width,
        deformation=deform,
        long_edge=picture_edge(ctx, asked),
        resolution=asked,
        auto=auto,
        resolution_note=note,
        given=frozenset(given),
        has_deformation=profile.deformation,
    )


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
    ctx: EngineContext, state: StackState, position_mm: float, labels: np.ndarray,
) -> np.ndarray:
    """ABBA's Nissl on the native plane grid, percentile-stretched to 0..1."""
    source = ctx.abba_atlas
    if source is None:
        raise ValueError("The nissl atlas channel needs ABBA's cached Allen atlas")
    values = np.asarray(source.sample_plane(
        "NISSL", ctx.atlas, position_mm, cast(Plane, state.plane),
        state.pitch_deg, state.yaw_deg,
    ), dtype=np.float32)
    inside = values[(labels > 0) & (values > 0)]
    top = float(np.percentile(inside, NISSL_PERCENTILE)) if inside.size else 0.0
    if top <= 0:
        return np.zeros(values.shape, dtype=np.float32)
    return np.clip(values / top, 0.0, 1.0)


def atlas_image_picture(
    ctx: EngineContext, state: StackState, kinds: tuple[str, ...], position_mm: float,
) -> Image.Image | None:
    """The atlas images *kinds* on the native plane grid; None for ``ara`` alone.

    ``ara`` alone is left to the renderers' own reference path (the pixels
    every earlier picture showed). ``nissl`` is ABBA's Nissl, stretched
    inside the atlas anatomy. Two images are added in their two colours
    (:func:`langslice.linear.appearance.channel_colors`); no image is a black
    plane (the lines alone).
    """
    images = tuple(kind for kind in kinds if kind in ATLAS_IMAGES)
    if images == ("ara",):
        return None
    plane = cast(Plane, state.plane)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
    ))
    shape = labels.shape
    if not images:
        return Image.new("L", (shape[1], shape[0]), 0)

    def gray(kind: str) -> np.ndarray:
        if kind == "nissl":
            return _nissl_plane(ctx, state, position_mm, labels)
        reference = get_reference_slice(ctx.atlas, position_mm, plane=plane,
                                        pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg)
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
    ctx: EngineContext, state: StackState, record: SliceState, options: DisplayOptions,
    *, tile_edge: int,
) -> tuple[Image.Image, list[str]]:
    """One section's raw channels side by side, each unmodified and labelled.

    Every tile is the same tissue frame as the section's other pictures, one
    raw plane in grayscale exactly as read (no stretch, no enhancement), at
    *tile_edge* at most. Returns the strip and the channel names in order.
    """
    from langslice.linear.render import beside, caption

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
    ctx: EngineContext, state: StackState, position_mm: float, options: DisplayOptions,
    *, long_edge: int | None = None, fill: bool = False,
) -> Image.Image:
    """The atlas at *position_mm*, framed to its anatomy, with the call's lines.

    At most *long_edge* (None: ``options.long_edge``) and never upsampled
    past the plane's own voxels, unless *fill*: then drawn at exactly
    *long_edge*, lines included, for a picture that puts the atlas beside a
    section of that size. With ``ara`` alone (no lines, no regions, no zoom)
    and no *fill* this is the picture ``view_atlas`` always sent.
    """
    from langslice.linear.atlas_fetch import atlas_mask, atlas_section, atlas_sized

    edge = int(long_edge or options.long_edge)

    def sized(picture: Image.Image) -> Image.Image:
        return resize_long_edge(picture, edge) if fill else atlas_sized(picture, edge)

    if options.atlas_images == ("ara",) and not options.lines and options.full_view:
        return sized(atlas_section(ctx, state, position_mm, frame=True))
    plane = cast(Plane, state.plane)
    labels = np.asarray(annotation_slice(
        ctx.atlas, position_mm, plane=plane, pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
    ))
    picture = atlas_image_picture(ctx, state, options.atlas_channels, position_mm)
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
    if not options.lines:
        return shown
    factor = shown.width / float(cropped.width)
    canvas = np.asarray(shown, dtype=np.uint8).copy()
    rgb, _ = normalize_border_style(options.border_color, options.border_thickness)
    context: list[np.ndarray] = []
    if options.borders:
        draw = outer_outline if options.outlines == "outer" else family_outlines
        context = [poly for _color, poly in draw(
            ctx.atlas, position_mm, plane=plane, pitch_deg=state.pitch_deg, yaw_deg=state.yaw_deg,
        )]
    origin = (box[0], box[1])
    if context:
        _draw_polys(canvas, context, rgb, thickness=options.border_thickness, origin=origin,
                    factor=factor, alpha=REGION_CONTEXT_ALPHA if options.regions else 1.0)
    if options.regions:
        # An atlas-only picture has no section: a side is the picture's own.
        left = regions_left(ctx.atlas, options.regions, position_mm, plane,
                            state.pitch_deg, state.yaw_deg, np.eye(2))
        _draw_polys(canvas, region_polys(labels, options.regions, left), rgb,
                    thickness=options.border_thickness, origin=origin, factor=factor)
    return Image.fromarray(canvas, mode="RGB")


def atlas_caption(state: StackState, position_mm: float, options: DisplayOptions) -> str:
    """The label burned into an atlas picture."""
    angles = (
        f" pitch {state.pitch_deg:.1f} yaw {state.yaw_deg:.1f}" if state.is_oblique else ""
    )
    name = options.atlas_name()
    extra = "" if name == "template" else f" {name}"
    if options.borders and options.outlines == "outer":
        extra += " outer outline"
    if options.regions:
        extra += " regions " + ",".join(name for name, _ids in options.regions)
    return f"atlas {position_mm:.2f} mm{angles}{extra}"


#: ``view`` keys a tool takes; re-exported so tools annotate one name.
__all__ = [
    "ATLAS_CHANNELS", "DisplayOptions", "MODE_RULES", "Profile", "View", "ViewAuto",
    "atlas_caption", "atlas_image_picture", "available_atlas_channels", "channel_strip",
    "clamp_resolution", "default_options", "display_facts", "framed_atlas", "framed_section",
    "parse_view", "regions_in_plane", "view_schema",
]
