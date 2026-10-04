"""The ``view`` argument: one call's picture options, validated for one tool.

Every picture tool takes its picture options in ONE argument, ``view`` (a
dict, :class:`langslice.linear.arguments.View`). This door-side module turns
it into the core's :class:`langslice.linear.display.DisplayOptions` or a
refusal that names every problem's fix: :func:`parse_view` checks it against
the tool's :class:`Profile` (its modes, first = default, and which keys mean
something for it) and :data:`langslice.linear.display.MODE_RULES` (what each
mode draws). Strict: a key that means nothing for the tool or the mode
(atlas keys on a picture with no atlas, ``outlines`` without
``borders``...) is refused with the reason; unknown keys are refused before
a tool runs (:func:`langslice.linear.arguments.argument_refusal`).
:func:`view_schema` types ``view`` per run (``resolution`` only at image
resolution "auto"). The keys themselves are described in
:mod:`langslice.linear.display`, which draws the pictures.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from langslice.linear.arguments import ViewAuto
from langslice.linear.display import (
    ATLAS_CHANNELS,
    ATLAS_IMAGES,
    DEFAULT_ATLAS_OPACITY,
    DEFAULT_BORDER_COLOR,
    DEFAULT_BORDER_THICKNESS,
    DEFORMATION_CHOICES,
    FULL_VIEW,
    MAX_OVERLAY_CHANNELS,
    MODE_RULES,
    OUTLINE_CHOICES,
    VERSIONS,
    DisplayOptions,
    available_atlas_channels,
)
from langslice.linear.opening import DEFAULT_IMAGE_LIMIT, IMAGE_LIMITS
from langslice.linear.render import (
    AUTO_RESOLUTION,
    MIN_RESOLUTION,
    normalize_border_style,
    picture_edge,
    resolution_level,
)
from langslice.linear.state import SliceState, StackState
from langslice.linear.workspace import Workspace
from langslice.providers.registry import canonical_provider

#: Every key ``view`` may carry (the strict wrapper refuses anything else).
VIEW_KEYS: tuple[str, ...] = tuple(ViewAuto.__annotations__)


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


def image_limit(ctx: Any) -> tuple[int, int]:
    """``(long edge, patch budget)`` of one image for this run's model lane
    (:data:`langslice.linear.opening.IMAGE_LIMITS`), read off the driver
    context's ``model`` (a ``provider/name`` string)."""
    model = str(getattr(ctx, "model", "") or "")
    provider = canonical_provider(model.split("/", 1)[0]) if "/" in model else ""
    return IMAGE_LIMITS.get(provider, DEFAULT_IMAGE_LIMIT)


def view_edge_limit(ctx: Any) -> int:
    """Largest picture the agent may ask for per call (``view.resolution``):
    the model lane's largest image edge (:func:`image_limit`). On the OpenAI
    lanes a near-square picture past ~1600 px still meets the patch budget,
    which shrinks it."""
    return image_limit(ctx)[0]


def clamp_resolution(value: Any, max_edge: int) -> tuple[int | None, str]:
    """``(long edge, note)`` for a requested ``resolution``; None for "default".

    0, None and "" are the default. A number outside :data:`MIN_RESOLUTION`
    .. *max_edge* (the driver model's largest image) is clamped into it and
    *note* says so; a value that is not a number raises ``ValueError``.
    """
    if value in (None, "", 0):
        return None, ""
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("resolution must be a number of pixels")
    low, high = MIN_RESOLUTION, int(max_edge)
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
    ctx: Workspace,
    state: StackState,
    view: Any,
    profile: Profile,
    *,
    max_edge: int,
    sections: list[SliceState] = (),  # type: ignore[assignment]
) -> DisplayOptions | dict[str, Any]:
    """Validate one call's ``view``; a refusal dict names every problem's fix.

    *max_edge* is the largest picture the driver model takes (the cap of
    ``resolution`` at "auto"). *sections* are the sections the call shows;
    named raw channels must exist on each. A key that means nothing for
    *profile* or for the mode is refused (``VIEW_KEY_UNUSED``) rather than
    dropped.
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
            asked, note = clamp_resolution(view.get("resolution", 0), max_edge)
        except (TypeError, ValueError):
            return _error("BAD_RESOLUTION", "view.resolution must be a whole number of "
                          f"pixels, {MIN_RESOLUTION}..{max_edge}")

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
        channels_apply=profile.channels,
    )
