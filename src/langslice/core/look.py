"""``look``'s four pictures, drawn by the core's renderers.

One entry point, :func:`look`, takes a :class:`LookRequest` and returns one
:class:`LookPicture` per picture:

- ``section`` — each section alone, oriented, tissue-framed
  (:func:`langslice.core.display.framed_section`): its raw channels with
  their display properties (:mod:`langslice.core.channels`), or its
  ``preprocessed`` channel;
- ``atlas`` — the atlas plane alone at each of ``positions_mm``, at the
  stack's view angles (``StackState.view_angles``), in the chosen atlas
  layers (``template``, ``nissl``, ``borders``;
  :func:`langslice.core.display.framed_atlas`);
- ``overlay`` — each section under its current registration with the atlas
  borders and any chosen atlas layers on it
  (:func:`langslice.core.placement.placement_pictures`, mode ``overlay``),
  the applied deformation drawn unless ``warp`` is ``none``;
- ``positioning`` — the stack along its slicing axis, ABBA's layout
  (:mod:`langslice.core.positioning`): one picture, or for a stack longer
  than :data:`langslice.core.positioning.PER_PICTURE` several, each a
  consecutive run of positions with its own ruler segment.

Defaults never read the job's state: no ``sections`` is every section; no
``channels`` is every raw channel, each with its display properties (as
napari shows a file); the atlas layers default per mode
(:data:`ATLAS_LAYER_DEFAULTS`). The same request draws the same kind of
picture at every stage of a run.

Each picture carries a caption for the picture index (position, angles,
scale, the channel settings in force) and a recipe, ``{"renderer": "look",
"args": ..., "state": ..., "base": [w, h], "shown": [w, h], "um_per_px": x}``:
the request (one picture's sections or position; a positioning picture's
the whole call's sections and positions and its ``part``), the facts of the stack
the picture was drawn from (:func:`snapshot`), the unzoomed and shown
content sizes and the micrometres per pixel shown. :func:`redraw` draws a
recipe again at a zoom window (:mod:`langslice.core.zoom`), from the stack
as it was when the stack has changed since. Picture numbers are not burned
into the pixels; the doors send them beside each picture.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from PIL import Image

from langslice.core.appearance import PREPROCESSED, channel_colors, preprocessed_settings
from langslice.core.appearance import describe as describe_look
from langslice.core.atlas.render import atlas_um_per_px
from langslice.core.atlas_fetch import atlas_section
from langslice.core.captions import angles_label, caption, view_angles_label
from langslice.core.channels import CHANNELS_KEY, all_properties, describe
from langslice.core.display import (
    DEFAULT_ATLAS_OPACITY,
    DEFAULT_BORDER_THICKNESS,
    MAX_OVERLAY_CHANNELS,
    DisplayOptions,
    atlas_caption,
    available_atlas_channels,
    canonical_atlas_name,
    framed_atlas,
    framed_section,
)
from langslice.core.layers import annotate, note
from langslice.core.placement import placement_pictures
from langslice.core.positioning import (
    ATLAS_UPSAMPLE,
    POSITIONING_MAX_WIDTH,
    positioning_pictures,
)
from langslice.core.scale import framed_um_per_px
from langslice.core.sections import PREVIEW_LONG_EDGE, render_slice
from langslice.core.sizes import picture_edge
from langslice.core.state import Angles, SliceState, StackState
from langslice.core.transform import calibrate
from langslice.core.workspace import Workspace

#: look's modes, in the order the docs list them.
MODES: tuple[str, ...] = ("section", "atlas", "overlay", "positioning")
#: The recipe's renderer name (:mod:`langslice.core.zoom` dispatches on it).
RENDERER = "look"
#: The ``channels`` entry that picks the section's preprocessed channel.
PREPROCESSED_CHANNEL = PREPROCESSED
#: ``atlas_layers`` when the request gives none, per mode.
ATLAS_LAYER_DEFAULTS: dict[str, tuple[str, ...]] = {
    "section": (),
    "atlas": ("template",),
    "overlay": ("borders",),
    "positioning": ("template",),
}
#: ``warp``: the applied deformation drawn, or the linear placement alone.
WARP_CHOICES: tuple[str, ...] = ("applied", "none")
#: The :data:`langslice.core.display.MODE_RULES` entry each mode is drawn under.
_DISPLAY_MODE = {"section": "section", "atlas": "template", "overlay": "overlay",
                 "positioning": "stacked"}
#: The section fields a picture of it depends on (:func:`snapshot`).
SECTION_FIELDS: tuple[str, ...] = (
    "index_original", "index_corrected", "position_mm", "cutting_angles_deg", "flip",
    "rotation_deg", "transform", "deformation", "damaged_regions",
)
#: The appearance entries a picture depends on (display properties and the
#: preprocessed channel's recipe).
APPEARANCE_KEYS: tuple[str, ...] = (CHANNELS_KEY, PREPROCESSED)
_BORDER_COLOR = "#ffff00"


class LookError(ValueError):
    """A request look cannot draw: ``code`` is ``UNKNOWN_MODE``,
    ``UNKNOWN_SECTION``, ``UNKNOWN_CHANNEL``, ``MIXED_CHANNELS``,
    ``TOO_MANY_CHANNELS``, ``UNKNOWN_LAYER``, ``NO_POSITIONS`` (atlas mode
    without ``positions_mm``), ``NO_POSITION`` (a section without one, in
    overlay mode), ``BAD_WARP`` or ``UNKNOWN_PART`` (a positioning part past
    the call's last picture)."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class LookRequest:
    """One look call, sections already resolved to filenames."""

    mode: str
    #: Section ids (empty: every section, in stack order).
    sections: tuple[str, ...] = ()
    #: Atlas positions (``atlas`` and ``positioning``).
    positions_mm: tuple[float, ...] = ()
    #: Raw channel names, or ``("preprocessed",)``; empty: every raw channel.
    channels: tuple[str, ...] = ()
    #: Any of ``template``, ``nissl``, ``borders``; None: the mode's default.
    atlas_layers: tuple[str, ...] | None = None
    #: Opacity of atlas images under an overlay; None: 0.5 when one is shown.
    atlas_opacity: float | None = None
    warp: str = "applied"
    #: Each picture's long edge (positioning: its largest tile); None:
    #: the run's picture size (:func:`langslice.core.sizes.picture_edge`).
    long_edge: int | None = None
    #: A zoom window, ``[x0, y0, x1, y1]`` fractions of the unzoomed picture's
    #: content (:mod:`langslice.core.zoom`); empty: the whole picture.
    zoom: tuple[float, ...] = ()
    #: Positioning: which picture of a split call (0-based); None: every one
    #: (with ``zoom``: the first).
    part: int | None = None

    def args(self) -> dict[str, Any]:
        """The request as a recipe's ``args`` (JSON)."""
        return {
            "mode": self.mode, "sections": list(self.sections),
            "positions_mm": [float(v) for v in self.positions_mm],
            "channels": list(self.channels),
            "atlas_layers": None if self.atlas_layers is None else list(self.atlas_layers),
            "atlas_opacity": self.atlas_opacity, "warp": self.warp,
            "long_edge": self.long_edge, "zoom": [float(v) for v in self.zoom],
            "part": self.part,
        }

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> LookRequest:
        layers = args.get("atlas_layers")
        return cls(
            mode=str(args["mode"]), sections=tuple(str(s) for s in args.get("sections") or ()),
            positions_mm=tuple(float(v) for v in args.get("positions_mm") or ()),
            channels=tuple(str(c) for c in args.get("channels") or ()),
            atlas_layers=None if layers is None else tuple(str(v) for v in layers),
            atlas_opacity=args.get("atlas_opacity"), warp=str(args.get("warp") or "applied"),
            long_edge=args.get("long_edge"),
            zoom=tuple(float(v) for v in args.get("zoom") or ()),
            part=None if args.get("part") is None else int(args["part"]),
        )


@dataclass(frozen=True)
class LookPicture:
    """One picture: the captioned image, its index caption, its recipe, the
    sections it shows and its mode."""

    image: Image.Image
    caption: str
    recipe: dict[str, Any]
    sections: tuple[str, ...]
    mode: str
    #: Micrometres per pixel of the picture's content.
    um_per_px: float
    #: Drawn from the stack as it was when the recipe was made, which has
    #: changed since (:func:`redraw`).
    stale: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


# --- the stack a picture depends on ------------------------------------------------


def _plain(value: Any) -> Any:
    """*value* as JSON reads it back (tuples to lists, numpy to numbers)."""

    def default(item: Any) -> Any:
        if isinstance(item, np.generic):
            return item.item()
        if isinstance(item, np.ndarray):
            return item.tolist()
        return str(item)

    return json.loads(json.dumps(value, default=default))


def snapshot(state: StackState, ids: Sequence[str], *, view_angles: bool = False) -> dict[str, Any]:
    """The facts of *state* a picture of *ids* depends on: each section's
    :data:`SECTION_FIELDS`, the display properties and the preprocessed
    recipe, and with *view_angles* the angles an atlas picture without a
    section is drawn at."""
    sections: dict[str, Any] = {}
    for section_id in ids:
        record = state.by_id(section_id)
        if record is not None:
            sections[section_id] = {name: getattr(record, name) for name in SECTION_FIELDS}
    data: dict[str, Any] = {
        "sections": sections,
        "appearance": {key: state.appearance[key] for key in APPEARANCE_KEYS
                       if key in state.appearance},
    }
    if view_angles:
        data["view_angles"] = list(state.view_angles)
    return _plain(data)


def changed_since(state: StackState, held: dict[str, Any]) -> bool:
    """Whether *state* differs from the snapshot *held* in anything the
    picture depends on (a section gone included)."""
    ids = list((held.get("sections") or {}).keys())
    now = snapshot(state, ids, view_angles="view_angles" in held)
    return now != _plain(held)


def restored(state: StackState, held: dict[str, Any]) -> StackState:
    """A copy of *state* with the snapshot *held* put back; *state* is
    untouched. :class:`LookError` ``UNKNOWN_SECTION`` when a section of the
    snapshot is no longer in the stack."""
    out = StackState.from_dict(copy.deepcopy(state.to_dict()))
    for section_id, fields in (held.get("sections") or {}).items():
        record = out.by_id(section_id)
        if record is None:
            raise LookError("UNKNOWN_SECTION", f"{section_id} is no longer in the stack")
        for name, value in fields.items():
            if name == "cutting_angles_deg":
                value = {"pitch": float(value["pitch"]), "yaw": float(value["yaw"])}
            setattr(record, name, copy.deepcopy(value))
    appearance = held.get("appearance") or {}
    for key in APPEARANCE_KEYS:
        if key in appearance:
            out.appearance[key] = copy.deepcopy(appearance[key])
        else:
            out.appearance.pop(key, None)
    return out


# --- options and words ----------------------------------------------------------------


def _channels_for(ws: Workspace, record: SliceState, channels: Sequence[str]
                  ) -> tuple[tuple[str, ...], str]:
    """``(raw names, version)`` *record* is drawn in: every raw channel by
    default, or the named ones, or (version) its preprocessed channel."""
    wanted = tuple(str(name) for name in channels)
    if PREPROCESSED_CHANNEL in wanted:
        if len(wanted) > 1:
            raise LookError("MIXED_CHANNELS", "'preprocessed' is shown alone, not with raw "
                            "channels")
        return (), PREPROCESSED
    names = tuple(ws.section_channels(record.id)[0])
    if not wanted:
        return names[:MAX_OVERLAY_CHANNELS], ""
    unknown = [name for name in wanted if name not in names]
    if unknown:
        raise LookError("UNKNOWN_CHANNEL", f"{record.id} has no channel "
                        f"{', '.join(map(repr, unknown))}; its channels: {', '.join(names)}, "
                        "or 'preprocessed'")
    if len(wanted) > MAX_OVERLAY_CHANNELS:
        raise LookError("TOO_MANY_CHANNELS",
                        f"at most {MAX_OVERLAY_CHANNELS} raw channels are drawn together")
    return wanted, ""


def _atlas_layers(ws: Workspace, request: LookRequest) -> tuple[str, ...]:
    if request.atlas_layers is None:
        return ATLAS_LAYER_DEFAULTS[request.mode]
    layers = tuple(dict.fromkeys(canonical_atlas_name(str(v)) for v in request.atlas_layers))
    allowed = available_atlas_channels(ws)
    unknown = [name for name in layers if name not in allowed]
    if unknown:
        raise LookError("UNKNOWN_LAYER", f"atlas layer {', '.join(map(repr, unknown))} is not "
                        f"drawn here; layers: {', '.join(allowed)}")
    return tuple(name for name in allowed if name in layers)


def _options(
    request: LookRequest, *, names: tuple[str, ...], version: str, layers: tuple[str, ...],
    long_edge: int, zoom_px: tuple[float, ...] = (),
) -> DisplayOptions:
    images = [name for name in layers if name != "borders"]
    opacity = (request.atlas_opacity if request.atlas_opacity is not None
               else DEFAULT_ATLAS_OPACITY if images else 0.0)
    return DisplayOptions(
        mode=_DISPLAY_MODE[request.mode], zoom=zoom_px, channels=names, version=version,
        atlas_channels=layers, atlas_opacity=float(opacity), regions=(), outlines="all",
        border_color=_BORDER_COLOR, border_thickness=DEFAULT_BORDER_THICKNESS,
        deformation=request.warp, long_edge=int(long_edge),
    )


def channel_words(state: StackState, record: SliceState, names: Sequence[str], version: str,
                  ) -> tuple[str, str]:
    """``(short, full)``: the channels shown, for a burned label and for the
    index caption (with the display settings in force)."""
    if version == PREPROCESSED:
        return ("preprocessed",
                f"preprocessed channel ({describe_look(preprocessed_settings(state, record.id))})")
    held = all_properties(state)
    colours = {name: word for name, word, _rgb in channel_colors(names)} if len(names) > 1 else {}
    short: list[str] = []
    full: list[str] = []
    for name in names:
        if name in held:
            short.append(describe(name, held[name]))
            full.append(describe(name, held[name]))
            continue
        word = colours.get(name, "gray")
        # One channel is drawn in gray; among several, a channel named for
        # its colour needs no colour word.
        short.append(name if len(names) == 1 or word == name.lower() else f"{name} {word}")
        full.append(f"{name} in {word}" if word != name.lower() else name)
    plain = [name for name in names if name not in held]
    contrast = ("" if not plain else
                "; percentile contrast" + ("" if len(plain) == len(names) else
                                           f" for {', '.join(plain)}"))
    return "raw " + " + ".join(short), "raw " + " + ".join(full) + contrast


def _orientation(record: SliceState) -> str:
    turns = []
    if record.rotation_deg:
        turns.append(f"rotated {record.rotation_deg}")
    if record.flip:
        turns.append("flipped")
    return f" ({', '.join(turns)})" if turns else ""


def _where(record: SliceState) -> str:
    if record.position_mm is None:
        return "no position"
    return f"at {float(record.position_mm):.2f} mm{angles_label(record.angles)}"


# --- the four modes -------------------------------------------------------------------


def _records(state: StackState, ids: Sequence[str]) -> list[SliceState]:
    if not ids:
        return sorted(state.slices, key=lambda record: record.index_original)
    out: list[SliceState] = []
    for section_id in ids:
        record = state.by_id(str(section_id))
        if record is None:
            raise LookError("UNKNOWN_SECTION", f"no section {section_id!r} in the stack")
        out.append(record)
    return out


def _zoom_pixels(window: Sequence[float], base: tuple[float, float]) -> tuple[float, ...]:
    if not window:
        return ()
    return (window[0] * base[0], window[1] * base[1], window[2] * base[0], window[3] * base[1])


def _zoom_tag(window: Sequence[float], base: Sequence[float], shown: Sequence[float]) -> str:
    if not window:
        return ""
    extent = max((window[2] - window[0]) * base[0], (window[3] - window[1]) * base[1], 1e-9)
    return f"zoom x{max(shown) / extent:.1f}  "


def _recipe(request: LookRequest, held: dict[str, Any], *, base: tuple[int, int],
            shown: tuple[int, int], um_per_px: float) -> dict[str, Any]:
    return {"renderer": RENDERER, "args": request.args(), "state": held,
            "base": [int(base[0]), int(base[1])], "shown": [int(shown[0]), int(shown[1])],
            "um_per_px": round(float(um_per_px), 4)}


def _section_picture(
    ws: Workspace, state: StackState, record: SliceState, request: LookRequest, long_edge: int,
    base: tuple[int, int] | None,
) -> LookPicture:
    names, version = _channels_for(ws, record, request.channels)
    options = _options(request, names=names, version=version, layers=(), long_edge=long_edge)
    look_ = options.look(state, record)
    whole = render_slice(ws, record, long_edge=long_edge, frame=True, look=look_)
    base = base or whole.size
    if request.zoom:
        options = replace(options, zoom=_zoom_pixels(request.zoom, base))
    picture = framed_section(ws, state, record, options)
    working = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
    working_um, _source = calibrate(state, ws, record, working)
    base_um = framed_um_per_px(ws, record, working_um, long_edge=long_edge, look=look_)
    enlarged = ""
    if request.zoom and max(picture.size) < long_edge:
        # The box holds fewer source pixels than the picture size: shown
        # larger, and said so (no detail is added).
        factor = min(long_edge / max(picture.size), ATLAS_UPSAMPLE)
        if factor > 1.05:
            picture = picture.resize((round(picture.width * factor),
                                      round(picture.height * factor)), Image.Resampling.LANCZOS)
            enlarged = f" (source pixels enlarged x{factor:.1f})"
    um = base_um * (max((request.zoom[2] - request.zoom[0]) * base[0],
                        (request.zoom[3] - request.zoom[1]) * base[1]) / max(picture.size)
                    if request.zoom else 1.0)
    short, full = channel_words(state, record, names, version)
    label = (f"{_zoom_tag(request.zoom, base, picture.size)}{record.id}"
             f"{_orientation(record)}  [{short}]{enlarged}")
    text = (f"{record.id} section{_orientation(record)}, "
            f"{_where(record)}; {um:.1f} um/px; {full}")
    image = caption(picture, label)
    held = snapshot(state, [record.id])
    one = replace(request, sections=(record.id,))
    recipe = _recipe(one, held, base=base, shown=picture.size, um_per_px=um)
    note(image, sections=(record.id,), mode="section", recipe=recipe, caption=text)
    return LookPicture(image=image, caption=text, recipe=recipe, sections=(record.id,),
                       mode="section", um_per_px=um)


def _overlay_picture(
    ws: Workspace, state: StackState, record: SliceState, request: LookRequest, long_edge: int,
    base: tuple[int, int] | None, store: Any,
) -> LookPicture:
    if record.position_mm is None:
        raise LookError("NO_POSITION", f"{record.id} has no position to draw the atlas at")
    names, version = _channels_for(ws, record, request.channels)
    layers = _atlas_layers(ws, request)
    options = _options(request, names=names, version=version, layers=layers,
                       long_edge=long_edge)
    position = float(record.position_mm)
    if request.zoom and base is None:
        whole = placement_pictures(ws, state, record, position, options, {}, store=store)
        x0, y0, x1, y1 = whole.canvas.panels[0].content_box if whole.canvas else (0, 0, 1, 1)
        base = (x1 - x0, y1 - y0)
    if request.zoom and base is not None:
        options = replace(options, zoom=_zoom_pixels(request.zoom, base))
    placed = placement_pictures(ws, state, record, position, options, {}, store=store)
    image = placed.images[0]
    shown: tuple[int, int] = image.size
    um = 0.0
    if placed.canvas is not None and placed.canvas.panels:
        panel = placed.canvas.panels[0]
        x0, y0, x1, y1 = panel.content_box
        shown = (x1 - x0, y1 - y0)
        um = placed.canvas.frame.um_per_px / panel.factor
    base = base or shown
    short, full = channel_words(state, record, names, version)
    drawn = placed.row.get("deformation_drawn")
    warp_words = ("deformation drawn" if drawn else
                  "deformation held but not drawable" if drawn is False else
                  "linear placement only" if request.warp == "none" and record.deformation
                  else "no deformation")
    atlas_words = " + ".join(layers) or "none"
    if options.atlas_images and options.atlas_opacity > 0:
        atlas_words += f" (images at {options.atlas_opacity:g})"
    text = (f"{record.id} overlay{_orientation(record)}, "
            f"{_where(record)}; {placed.row.get('transform', 'identity')} transform, "
            f"{warp_words}; {um:.1f} um/px; {full}; atlas {atlas_words}")
    held = snapshot(state, [record.id])
    one = replace(request, sections=(record.id,))
    recipe = _recipe(one, held, base=base, shown=shown, um_per_px=um)
    annotate(image, recipe=recipe, caption=text)
    return LookPicture(image=image, caption=text, recipe=recipe, sections=(record.id,),
                       mode="overlay", um_per_px=um, extra=dict(placed.row))


def _atlas_picture(
    ws: Workspace, state: StackState, position_mm: float, request: LookRequest, long_edge: int,
    base: tuple[int, int] | None, base_um: float | None, angles: Angles,
) -> LookPicture:
    layers = _atlas_layers(ws, request)
    options = _options(request, names=(), version="", layers=layers, long_edge=long_edge)
    voxel = atlas_um_per_px(ws.atlas)
    if base is None or base_um is None:
        whole = framed_atlas(ws, state, position_mm, options, angles=angles)
        native = atlas_section(ws, state, position_mm, frame=True, angles=angles)
        base, base_um = whole.size, voxel * max(native.size) / max(whole.size)
    if request.zoom:
        extent = max((request.zoom[2] - request.zoom[0]) * base[0],
                     (request.zoom[3] - request.zoom[1]) * base[1])
        um = max(base_um * extent / long_edge, voxel / ATLAS_UPSAMPLE)
        at_um = (base[0] * base_um / um, base[1] * base_um / um)
        options = replace(options, zoom=_zoom_pixels(request.zoom, at_um))
        picture = framed_atlas(ws, state, position_mm, options, angles=angles, um_per_px=um)
    else:
        picture = framed_atlas(ws, state, position_mm, options, angles=angles)
        um = base_um
    median = state.drawn_at_median(angles)
    label = _zoom_tag(request.zoom, base, picture.size) + atlas_caption(
        state, position_mm, options, angles=angles, median=median)
    layers_words = " + ".join(layers) or "none"
    plane = (view_angles_label(angles, median=True) if median
             else f"{angles_label(angles)} (the stack's cutting angles)")
    text = f"atlas at {position_mm:.2f} mm{plane}; {um:.1f} um/px; layers {layers_words}"
    image = caption(picture, label)
    held = snapshot(state, [], view_angles=True)
    one = replace(request, sections=(), positions_mm=(float(position_mm),))
    recipe = _recipe(one, held, base=base, shown=picture.size, um_per_px=um)
    note(image, mode="atlas", recipe=recipe, caption=text,
         extra={"position_mm": float(position_mm)})
    return LookPicture(image=image, caption=text, recipe=recipe, sections=(), mode="atlas",
                       um_per_px=um, extra={"position_mm": float(position_mm)})


def _positioning(
    ws: Workspace, state: StackState, request: LookRequest, long_edge: int, angles: Angles,
    zoom_edge: int | None,
) -> list[LookPicture]:
    records = _records(state, request.sections)
    layers = _atlas_layers(ws, request)
    atlas_options = _options(request, names=(), version="", layers=layers, long_edge=long_edge)
    shown_words: dict[str, str] = {}

    def look_of(record: SliceState) -> Any:
        names, version = _channels_for(ws, record, request.channels)
        shown_words.setdefault(record.id, channel_words(state, record, names, version)[0])
        return _options(request, names=names, version=version, layers=(),
                        long_edge=long_edge).look(state, record)

    for record in records:  # refuse a bad channel before drawing anything
        look_of(record)
    words = sorted(set(shown_words.values()))
    try:
        drawn = positioning_pictures(
            ws, state, records, [float(v) for v in request.positions_mm], look_of=look_of,
            atlas_options=atlas_options, angles=angles, tile_edge=long_edge,
            max_width=POSITIONING_MAX_WIDTH, part=request.part, window=request.zoom,
            zoom_edge=zoom_edge, shown=words[0] if len(words) == 1 else "",
        )
    except IndexError as exc:
        raise LookError("UNKNOWN_PART", str(exc)) from None
    held = snapshot(state, [record.id for record in records], view_angles=True)
    whole = replace(request, sections=tuple(r.id for r in records))
    out: list[LookPicture] = []
    for one in drawn:
        layout = one.layout
        ids = tuple(slot.key for slot in layout.sections)
        rows = ", ".join(f"{slot.label[0]} {slot.position_mm:.2f} mm"
                         if slot.position_mm is not None else f"{slot.label[0]} no position"
                         for slot in layout.sections)
        text = one.caption + (f". Sections: {rows}" if rows else "")
        recipe = _recipe(replace(whole, part=layout.part), held,
                         base=(layout.width, layout.height), shown=one.content,
                         um_per_px=one.um_per_px)
        extra = {"part": layout.part, "parts": layout.parts,
                 "positions_mm": [float(slot.position_mm or 0.0) for slot in layout.atlas],
                 "range_mm": [float(v) for v in layout.range_mm]}
        note(one.image, sections=ids, mode="positioning", recipe=recipe, caption=text,
             extra=extra)
        out.append(LookPicture(image=one.image, caption=text, recipe=recipe, sections=ids,
                               mode="positioning", um_per_px=one.um_per_px, extra=extra))
    return out


def look(
    ws: Workspace, state: StackState, request: LookRequest, *, store: Any = None,
    angles: Angles | None = None, base: tuple[int, int] | None = None,
    base_um: float | None = None, zoom_edge: int | None = None,
) -> list[LookPicture]:
    """The pictures of *request*: one per section (``section``,
    ``overlay``), one per position (``atlas``), one per run of positions
    (``positioning``: one for a short stack; with ``part``, that one).

    *store* (a :class:`~langslice.core.deformation.RecordStore`) gives an
    overlay its applied deformation. *angles* is the plane an atlas picture
    without a section is drawn at (None: ``StackState.view_angles``).
    *base* / *base_um* are the unzoomed picture's content size and scale
    when the request zooms (:func:`redraw` passes the recipe's); without
    them the unzoomed picture is measured first. *zoom_edge* is a
    positioning zoom's long edge (None: the unzoomed picture's).
    """
    if request.mode not in MODES:
        raise LookError("UNKNOWN_MODE", f"mode must be one of {', '.join(MODES)}")
    if request.warp not in WARP_CHOICES:
        raise LookError("BAD_WARP", f"warp must be one of {', '.join(WARP_CHOICES)}")
    long_edge = int(request.long_edge or picture_edge(ws))
    plane = angles if angles is not None else state.view_angles
    if request.mode == "positioning":
        return _positioning(ws, state, request, long_edge, plane, zoom_edge)
    if request.mode == "atlas":
        if not request.positions_mm:
            raise LookError("NO_POSITIONS", "atlas mode needs positions_mm")
        return [_atlas_picture(ws, state, float(mm), request, long_edge, base, base_um, plane)
                for mm in request.positions_mm]
    records = _records(state, request.sections)
    if request.mode == "section":
        return [_section_picture(ws, state, record, request, long_edge, base)
                for record in records]
    return [_overlay_picture(ws, state, record, request, long_edge, base, store)
            for record in records]


def redraw(
    recipe: dict[str, Any], window: Sequence[float], ws: Workspace, state: StackState,
    *, store: Any = None, zoom_edge: int | None = None,
) -> LookPicture:
    """The picture of *recipe* drawn again at *window* (fractions of its
    unzoomed content; empty: whole), from the stack as it was when the
    recipe was made (``stale`` when it has changed since).
    :class:`LookError` ``UNKNOWN_SECTION`` when a section it showed is gone.
    """
    request = replace(LookRequest.from_args(recipe["args"]),
                      zoom=tuple(float(v) for v in window))
    held = recipe.get("state") or {}
    stale = bool(held) and changed_since(state, held)
    drawn = restored(state, held) if stale else state
    angles = held.get("view_angles")
    base = recipe.get("base")
    pictures = look(
        ws, drawn, request, store=store,
        angles=(float(angles[0]), float(angles[1])) if angles else None,
        base=(int(base[0]), int(base[1])) if base else None,
        base_um=_base_um(recipe), zoom_edge=zoom_edge,
    )
    return replace(pictures[0], stale=stale)


def _base_um(recipe: dict[str, Any]) -> float | None:
    """The unzoomed picture's micrometres per pixel, from a recipe drawn at
    any window."""
    um = recipe.get("um_per_px")
    window = (recipe.get("args") or {}).get("zoom") or []
    base, shown = recipe.get("base"), recipe.get("shown")
    if um is None or not base or not shown:
        return None
    if not window:
        return float(um)
    extent = max((window[2] - window[0]) * base[0], (window[3] - window[1]) * base[1])
    return float(um) * max(shown) / max(extent, 1e-9)


__all__ = [
    "ATLAS_LAYER_DEFAULTS", "LookError", "LookPicture", "LookRequest", "MODES",
    "PREPROCESSED_CHANNEL", "RENDERER", "changed_since", "channel_words", "look", "redraw",
    "restored", "snapshot",
]
