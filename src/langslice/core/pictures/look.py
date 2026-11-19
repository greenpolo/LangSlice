"""``look``'s four pictures, drawn by the core's renderers.

One entry point, :func:`look`, takes a :class:`LookRequest` and returns one
:class:`LookPicture` per picture:

- ``section`` — each section alone, oriented, the whole image
  (:func:`langslice.core.pictures.panels.framed_section`): one of its section
  images (:mod:`langslice.core.sections.images`; ``preprocessed`` by default),
  or several raw channels together (:mod:`langslice.core.sections.channels`);
- ``atlas`` — the atlas plane alone at each of ``positions_mm``, at the stack's
  view angles (``Stack.view_angles``), in the chosen atlas layers (the
  atlas's images and ``borders``;
  :func:`langslice.core.pictures.panels.framed_atlas`);
- ``overlay`` — each section under its current registration with the atlas
  borders and any chosen atlas layers on it
  (:func:`langslice.core.pictures.placement.placement_pictures`, mode
  ``overlay``), the applied deformation drawn unless ``warp`` is ``none``;
- ``positioning`` — the stack along its slicing axis, ABBA's layout
  (:mod:`langslice.core.pictures.positioning`): one picture, or for a stack
  longer than :data:`langslice.core.pictures.positioning.PER_PICTURE` several,
  each a consecutive run of positions with its own ruler segment.

Defaults never read the job's state: no ``sections`` is every section; no
``images`` is the ``preprocessed`` image, what every fit reads by default;
the atlas layers default per mode
(:data:`ATLAS_LAYER_DEFAULTS`). The same request draws the same kind of
picture at every stage of a run.

Each picture carries a caption for the picture index (position, angles, scale,
the section image drawn) and a recipe, ``{"renderer": "look", "args": ...,
"state": ..., "base": [w, h], "shown": [w, h], "um_per_px": x}``: the request
(one picture's sections or position; a positioning picture's the whole call's
sections and positions and its ``part``), the facts of the stack the picture
was drawn from (:func:`snapshot`), the unzoomed and shown content sizes and the
micrometres per pixel shown. :func:`redraw` draws a recipe again at a zoom
window (:mod:`langslice.core.pictures.zoom`), from the stack as it was when the
stack has changed since. Picture numbers are not burned into the pixels; the
doors send them beside each picture.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Collection, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.core.atlas.borders import atlas_um_per_px
from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.atlas.planes import atlas_section
from langslice.core.fits.affine import calibrate
from langslice.core.jsonable import JsonObject, to_jsonable
from langslice.core.pictures.canvas import AUTO_BORDER_COLOR
from langslice.core.pictures.captions import (
    angles_label,
    caption,
    view_angles_label,
)
from langslice.core.pictures.drawn import annotate, note
from langslice.core.pictures.modes import LOOK_MODES
from langslice.core.pictures.options import (
    DEFAULT_ATLAS_OPACITY,
    DEFAULT_BORDER_THICKNESS,
    MAX_OVERLAY_CHANNELS,
    DisplayOptions,
    available_atlas_channels,
)
from langslice.core.pictures.panels import (
    atlas_caption,
    framed_atlas,
    framed_section,
)
from langslice.core.pictures.placement import placement_pictures
from langslice.core.pictures.positioning import (
    ATLAS_UPSAMPLE,
    POSITIONING_MAX_WIDTH,
    positioning_pictures,
)
from langslice.core.pictures.scale import framed_um_per_px
from langslice.core.pictures.sizes import picture_edge
from langslice.core.refusal import RefusalCode, Refused
from langslice.core.sections import images as images_
from langslice.core.sections.channels import colour_words
from langslice.core.sections.images import PREPROCESSED
from langslice.core.sections.native import framed_window
from langslice.core.sections.renders import PREVIEW_LONG_EDGE, render_slice
from langslice.core.stack.rows import fields_to_json, with_fields
from langslice.core.stack.section import Angles, Section
from langslice.core.stack.stack import Stack, unknown_sections
from langslice.core.workspace import Workspace

if TYPE_CHECKING:
    from langslice.core.fits.deformable.record import DeformableRecord
    from langslice.core.fits.deformation import RecordStore

MODES: tuple[str, ...] = LOOK_MODES
RENDERER = "look"
ATLAS_LAYER_DEFAULTS: dict[str, tuple[str, ...]] = {
    "section": (),
    "atlas": ("template",),
    "overlay": ("borders",),
    "positioning": ("template",),
}
WARP_CHOICES: tuple[str, ...] = ("applied", "none")
_DISPLAY_MODE = {
    "section": "section",
    "atlas": "template",
    "overlay": "overlay",
    "positioning": "stacked",
}
FROM_FILE = " from the image file"
SECTION_FIELDS: tuple[str, ...] = (
    "index_original",
    "index_corrected",
    "position_mm",
    "cutting_angles_deg",
    "flip",
    "rotation_deg",
    "transform",
    "deformation",
    "damaged_regions",
    "landmarks",
    "images",
)
_BORDER_COLOR = AUTO_BORDER_COLOR

logger = logging.getLogger(__name__)

UNCALIBRATED = " (uncalibrated)"


@dataclass(frozen=True)
class LookRequest:
    """One look call, sections already resolved to filenames."""

    mode: str
    sections: tuple[str, ...] = ()
    positions_mm: tuple[float, ...] = ()
    images: tuple[str, ...] = ()
    atlas_layers: tuple[str, ...] | None = None
    atlas_opacity: float | None = None
    warp: str = "applied"
    long_edge: int | None = None
    zoom: tuple[float, ...] = ()
    part: int | None = None
    highlight: tuple[str, ...] = ()
    region_only: bool = False
    layers: tuple[str, ...] = ()
    masks: tuple[str, ...] = ()
    depth: int = OVERLAY_DEPTH
    registrations: tuple[tuple[str, int, bool], ...] = ()

    def registration_of(self, section: str) -> tuple[int, bool] | None:
        """``(number, held)`` of the registration *section* is drawn under,
        when the request names one."""
        for name, number, held in self.registrations:
            if name == section:
                return number, held
        return None

    def args(self) -> dict[str, Any]:
        """The request as a recipe's ``args`` (JSON)."""
        return {
            "mode": self.mode,
            "sections": list(self.sections),
            "positions_mm": [float(v) for v in self.positions_mm],
            "images": list(self.images),
            "atlas_layers": None if self.atlas_layers is None else list(self.atlas_layers),
            "atlas_opacity": self.atlas_opacity,
            "warp": self.warp,
            "long_edge": self.long_edge,
            "zoom": [float(v) for v in self.zoom],
            "part": self.part,
            "highlight": list(self.highlight),
            "depth": self.depth,
            **({"region_only": True} if self.region_only else {}),
            **({"layers": list(self.layers)} if self.layers else {}),
            **({"masks": list(self.masks)} if self.masks else {}),
            **(
                {
                    "registrations": [
                        [name, number, held] for name, number, held in self.registrations
                    ]
                }
                if self.registrations
                else {}
            ),
        }

    @classmethod
    def from_args(cls, args: dict[str, Any]) -> LookRequest:
        unknown = sorted(set(args) - set(cls.__dataclass_fields__))
        if unknown:
            raise Refused(
                RefusalCode.STALE_INPUT,
                f"Unknown picture recipe key(s) {unknown} (views.jsonl)",
            )
        layers = args["atlas_layers"]
        return cls(
            mode=str(args["mode"]),
            sections=tuple(str(s) for s in args["sections"]),
            positions_mm=tuple(float(v) for v in args["positions_mm"]),
            images=tuple(str(c) for c in args["images"]),
            atlas_layers=None if layers is None else tuple(str(v) for v in layers),
            atlas_opacity=args["atlas_opacity"],
            warp=str(args["warp"]),
            long_edge=args["long_edge"],
            zoom=tuple(float(v) for v in args["zoom"]),
            part=None if args["part"] is None else int(args["part"]),
            highlight=tuple(str(v) for v in args["highlight"]),
            region_only=bool(args.get("region_only", False)),
            layers=tuple(str(v) for v in args.get("layers") or ()),
            masks=tuple(str(v) for v in args.get("masks") or ()),
            depth=int(args["depth"]),
            registrations=tuple(
                (str(name), int(number), bool(held))
                for name, number, held in args.get("registrations") or ()
            ),
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
    um_per_px: float
    stale: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def snapshot(state: Stack, ids: Sequence[str], *, view_angles: bool = False) -> dict[str, Any]:
    """The facts of *state* a picture of *ids* depends on: each section's
    :data:`SECTION_FIELDS` (their section images included), and with
    *view_angles* the angles an atlas picture without a section is drawn at."""
    sections: dict[str, Any] = {}
    for section_id in ids:
        section = state.by_id(section_id)
        if section is not None:
            names = (*SECTION_FIELDS, "layers") if section.layers else SECTION_FIELDS
            sections[section_id] = fields_to_json(section, names)
    data: dict[str, Any] = {"sections": sections}
    if view_angles:
        data["view_angles"] = list(state.view_angles)
    return to_jsonable(data)


ANNOTATION_FIELDS: tuple[str, ...] = ("landmarks", "damaged_regions", "layers")


def _without_annotations(data: dict[str, Any]) -> dict[str, Any]:
    sections = {
        section_id: {k: v for k, v in fields.items() if k not in ANNOTATION_FIELDS}
        for section_id, fields in (data.get("sections") or {}).items()
    }
    return {**data, "sections": sections}


def differs_since(state: Stack, held: dict[str, Any]) -> bool:
    """Whether *state* differs from the snapshot *held* at all, landmark
    markers included: then an old picture is drawn from the restored snapshot."""
    ids = list((held.get("sections") or {}).keys())
    return snapshot(state, ids, view_angles="view_angles" in held) != to_jsonable(held)


REGISTRATION_FIELDS: tuple[str, ...] = ("flip", "rotation_deg", "transform", "deformation")


def changed_since(state: Stack, held: dict[str, Any], *, besides: Collection[str] = ()) -> bool:
    """Whether *state* differs from the snapshot *held* in anything the
    picture depends on (a section gone included). Landmark markers and
    damage marks are not part of it (:data:`ANNOTATION_FIELDS`): editing pairs
    or marking damage changes nothing a picture shows of the stack. The
    sections of *besides* were drawn under a registration they did not hold:
    their :data:`REGISTRATION_FIELDS` are not compared."""
    ids = list((held.get("sections") or {}).keys())
    now = _without_annotations(snapshot(state, ids, view_angles="view_angles" in held))
    then = _without_annotations(to_jsonable(held))
    for section_id in besides:
        for data in (now, then):
            fields = (data.get("sections") or {}).get(section_id)
            if isinstance(fields, dict):
                for name in REGISTRATION_FIELDS:
                    fields.pop(name, None)
    return now != then


def restored(state: Stack, held: dict[str, Any]) -> Stack:
    """A copy of *state* with the snapshot *held* put back; *state* is
    untouched. :class:`Refused` ``NOT_FOUND`` when a section of the
    snapshot is no longer in the stack."""
    out = copy.deepcopy(state)
    snapshots = cast("dict[str, JsonObject]", held.get("sections") or {})
    for section_id in snapshots:
        if out.by_id(section_id) is None:
            raise Refused(
                RefusalCode.NOT_FOUND,
                f"{section_id} is no longer in the stack, so this "
                "picture cannot be drawn again; draw the stack as "
                "it is now.",
            )
    for index, section in enumerate(out.slices):
        fields = snapshots.get(section.id)
        if fields is not None:
            out.slices[index] = with_fields(section, {"layers": [], **fields})
    return out


def _images_for(
    ws: Workspace, record: Section, images: Sequence[str]
) -> tuple[tuple[str, ...], str]:
    """``(raw names drawn together, image drawn alone)`` for *record*: the
    ``preprocessed`` image by default, one named image alone, or several raw
    channels together."""
    wanted = tuple(dict.fromkeys(str(name) for name in images))
    if not wanted:
        return (), PREPROCESSED
    known = images_.image_names(ws, record)
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise Refused(
            RefusalCode.NOT_FOUND,
            f"{record.id} has no section image "
            f"{', '.join(map(repr, unknown))}; its images: {', '.join(known)}",
        )
    if len(wanted) == 1:
        return (), wanted[0]
    raw = images_.raw_names(ws, record.id)
    made = [name for name in wanted if name not in raw]
    if made:
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            f"{made[0]!r} is a made image and is drawn alone; "
            "several images are drawn together only when all are raw channels",
        )
    if len(wanted) > MAX_OVERLAY_CHANNELS:
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            f"at most {MAX_OVERLAY_CHANNELS} raw channels are drawn together",
        )
    return wanted, ""


def image_text(ws: Workspace, record: Section, name: str) -> str:
    """What section image *name* of *record* is, for a caption."""
    if images_.kind_of(ws, record, name) == images_.RAW:
        return "raw channel, as the file holds it"
    entry = images_.entry_of(record, name)
    source = None if entry is None else entry.source
    held: dict[str, Any] = dict(source or images_.opening_source(ws.settings))
    if "recipe" in held:
        return images_.recipe_words(held["recipe"])
    if "script" in held:
        return f"made by a script: {held['script']}"
    return f"{images_.opening_words(held)}, made when the job opened"


def _atlas_layers(ws: Workspace, request: LookRequest) -> tuple[str, ...]:
    if request.atlas_layers is None:
        return ATLAS_LAYER_DEFAULTS[request.mode]
    layers = tuple(dict.fromkeys(str(v) for v in request.atlas_layers))
    allowed = available_atlas_channels(ws)
    unknown = [name for name in layers if name not in allowed]
    if unknown:
        raise Refused(
            RefusalCode.NOT_FOUND,
            f"atlas layer {', '.join(map(repr, unknown))} is not "
            f"drawn here; layers: {', '.join(allowed)}",
        )
    return tuple(name for name in allowed if name in layers)


def _highlighted(ws: Workspace, request: LookRequest) -> tuple[tuple[str, frozenset[int]], ...]:
    """The request's highlighted regions as the renderers take them,
    ``(name, ids with descendants)``; :class:`Refused` ``NOT_FOUND``."""
    if not request.highlight:
        return ()
    from langslice.core.fits.deformable.plane_images import resolve_entries

    names = list(request.highlight)
    try:
        found = resolve_entries(ws.atlas, names)
    except ValueError as exc:
        raise Refused(RefusalCode.NOT_FOUND, message=str(exc)) from None
    return tuple(
        (name, frozenset(ids)) for name, (_region, _side, ids) in zip(names, found, strict=True)
    )


def _options(
    request: LookRequest,
    *,
    names: tuple[str, ...],
    image: str,
    layers: tuple[str, ...],
    long_edge: int,
    zoom_px: tuple[float, ...] = (),
    regions: tuple[tuple[str, frozenset[int]], ...] = (),
) -> DisplayOptions:
    images = [name for name in layers if name != "borders"]
    opacity = (
        request.atlas_opacity
        if request.atlas_opacity is not None
        else DEFAULT_ATLAS_OPACITY
        if images
        else 0.0
    )
    return DisplayOptions(
        mode=_DISPLAY_MODE[request.mode],
        zoom=zoom_px,
        channels=names,
        image=image,
        atlas_channels=layers,
        atlas_opacity=float(opacity),
        regions=regions,
        outlines="none" if request.region_only else "all",
        border_color=_BORDER_COLOR,
        border_thickness=DEFAULT_BORDER_THICKNESS,
        deformation=request.warp,
        long_edge=int(long_edge),
        depth=request.depth,
    )


def image_words(
    ws: Workspace,
    record: Section,
    names: Sequence[str],
    image: str,
) -> tuple[str, str]:
    """``(short, full)``: what of the section is drawn, for a burned label and
    for the index caption: the section image and what it is, or the raw
    channels in their colours."""
    if names:
        words = "raw " + colour_words(names)
        return words, f"{words}, each with automatic contrast"
    return image, f"section image {image} ({image_text(ws, record, image)})"


def _orientation(record: Section) -> str:
    turns = []
    if record.rotation_deg:
        turns.append(f"rotated {record.rotation_deg}")
    if record.flip:
        turns.append("flipped")
    return f" ({', '.join(turns)})" if turns else ""


def _registration_words(request: LookRequest, record: Section) -> str:
    """``, registration 3 (not held)``: the registration a picture draws the
    section under, when the request names one; empty otherwise."""
    named = request.registration_of(record.id)
    if named is None:
        return ""
    number, held = named
    return f", registration {number} ({'held' if held else 'not held'})"


def _where(record: Section) -> str:
    return f"at {record.position_mm:.2f} mm{angles_label(record.angles)}"


def _records(state: Stack, ids: Sequence[str]) -> list[Section]:
    if not ids:
        return sorted(state.slices, key=lambda record: record.index_original)
    records = [state.by_id(str(name)) for name in ids]
    unknown = [str(name) for name, record in zip(ids, records, strict=True) if record is None]
    if unknown:
        raise Refused(RefusalCode.NOT_FOUND, **unknown_sections(state, unknown))
    return [record for record in records if record is not None]


def _zoom_pixels(window: Sequence[float], base: tuple[float, float]) -> tuple[float, ...]:
    if not window:
        return ()
    return (window[0] * base[0], window[1] * base[1], window[2] * base[0], window[3] * base[1])


def _zoom_tag(window: Sequence[float], base: Sequence[float], shown: Sequence[float]) -> str:
    if not window:
        return ""
    extent = max((window[2] - window[0]) * base[0], (window[3] - window[1]) * base[1], 1e-9)
    return f"zoom x{max(shown) / extent:.1f}  "


def _recipe(
    request: LookRequest,
    held: dict[str, Any],
    *,
    base: tuple[int, int],
    shown: tuple[int, int],
    um_per_px: float,
) -> dict[str, Any]:
    return {
        "renderer": RENDERER,
        "args": request.args(),
        "state": held,
        "base": [int(base[0]), int(base[1])],
        "shown": [int(shown[0]), int(shown[1])],
        "um_per_px": round(float(um_per_px), 4),
    }


def _section_picture(
    ws: Workspace,
    state: Stack,
    record: Section,
    request: LookRequest,
    long_edge: int,
    base: tuple[int, int] | None,
) -> LookPicture:
    names, image = _images_for(ws, record, request.images)
    options = _options(request, names=names, image=image, layers=(), long_edge=long_edge)
    look_ = options.look(state, record)
    whole = render_slice(ws, record, long_edge=long_edge, look=look_)
    base = base or whole.size
    picture: Image.Image | None = None
    enlarged = ""
    if request.zoom:
        try:
            picture, scale = framed_window(ws, record, look_, tuple(request.zoom), long_edge)
            enlarged = FROM_FILE + (f" (file pixels enlarged x{scale:.1f})" if scale > 1.05 else "")
        except Exception:
            logger.warning(
                "zoom of %s: the image file could not be read; drawn from its working copy",
                record.id,
                exc_info=True,
            )
    if picture is None:
        if request.zoom:
            options = replace(options, zoom=_zoom_pixels(request.zoom, base))
        picture = framed_section(ws, state, record, options)
    working = render_slice(ws, record, long_edge=PREVIEW_LONG_EDGE)
    working_um, source = calibrate(state, ws, record, working)
    base_um = framed_um_per_px(ws, record, working_um, long_edge=long_edge, look=look_)
    if request.zoom and not enlarged and max(picture.size) < long_edge:
        factor = min(long_edge / max(picture.size), ATLAS_UPSAMPLE)
        if factor > 1.05:
            from langslice.core.geometry.pixel_frames import cropped_geometry

            picture = cropped_geometry(
                picture.resize(
                    (round(picture.width * factor), round(picture.height * factor)),
                    Image.Resampling.LANCZOS,
                ),
                picture,
            )
            enlarged = f" (source pixels enlarged x{factor:.1f})"
    um = base_um * (
        max(
            (request.zoom[2] - request.zoom[0]) * base[0],
            (request.zoom[3] - request.zoom[1]) * base[1],
        )
        / max(picture.size)
        if request.zoom
        else 1.0
    )
    short, full = image_words(ws, record, names, image)
    registration = _registration_words(request, record)
    label = (
        f"{_zoom_tag(request.zoom, base, picture.size)}{record.id}"
        f"{_orientation(record)}{registration}  [{short}]{enlarged}"
    )
    text = (
        f"{record.id} section{_orientation(record)}{registration}, "
        f"{_where(record)}; {um:.1f} um/px{UNCALIBRATED if source == 'uncalibrated' else ''}; "
        f"{full}" + (f";{FROM_FILE}" if enlarged.startswith(FROM_FILE) else "")
    )
    from langslice.core.fits.landmarks import draw_landmarks

    panel = {
        "role": "section",
        "section": record.id,
        "bounds": [0, 0, *picture.size],
        "pixel_to_file_fraction": picture.info["pixel_to_file_fraction"],
    }
    panels = [panel]
    picture = picture.copy()
    draw_landmarks(picture, panels, state, record)
    drawn_layers = _draw_layers(picture, panels, state, record, request, ws)
    drawn_masks = _draw_masks(picture, panels, state, record, request, ws)
    text += _mask_words(drawn_masks, record)
    image = caption(picture, label, min_width=long_edge if request.zoom else 0)
    held = snapshot(state, [record.id])
    one = replace(request, sections=(record.id,))
    recipe = _recipe(one, held, base=base, shown=picture.size, um_per_px=um)
    recipe["endpoint_panels"] = panels
    recipe["registration_context"] = _landmark_context(state, record)
    note(image, sections=(record.id,), mode="section", recipe=recipe, caption=text)
    return LookPicture(
        image=image,
        caption=text,
        recipe=recipe,
        sections=(record.id,),
        mode="section",
        um_per_px=um,
        extra={
            **({"layers": drawn_layers} if drawn_layers else {}),
            **({"masks": drawn_masks} if drawn_masks else {}),
        },
    )


def _overlay_picture(
    ws: Workspace,
    state: Stack,
    record: Section,
    request: LookRequest,
    long_edge: int,
    base: tuple[int, int] | None,
    store: RecordStore | None,
) -> LookPicture:
    names, picked = _images_for(ws, record, request.images)
    layers = _atlas_layers(ws, request)
    options = _options(
        request,
        names=names,
        image=picked,
        layers=layers,
        long_edge=long_edge,
        regions=_highlighted(ws, request),
    )
    position = record.position_mm
    if request.zoom and base is None:
        whole = placement_pictures(ws, state, record, position, options, {}, store=store)
        x0, y0, x1, y1 = whole.canvas.panels[0].content_box if whole.canvas else (0, 0, 1, 1)
        base = (x1 - x0, y1 - y0)
    if request.zoom and base is not None:
        options = replace(options, zoom=_zoom_pixels(request.zoom, base))
    placed = placement_pictures(
        ws,
        state,
        record,
        position,
        options,
        {},
        store=store,
        label_note=_registration_words(request, record),
    )
    image = placed.images[0]
    shown: tuple[int, int] = image.size
    um = 0.0
    if placed.canvas is not None and placed.canvas.panels:
        panel = placed.canvas.panels[0]
        x0, y0, x1, y1 = panel.content_box
        shown = (x1 - x0, y1 - y0)
        um = placed.canvas.frame.um_per_px / panel.factor
    base = base or shown
    short, full = image_words(ws, record, names, picked)
    drawn = placed.row.get("deformation_drawn")
    warp_words = (
        "deformation drawn: the section is warped onto the atlas, which keeps its place"
        if drawn
        else "deformation held but not drawable"
        if drawn is False
        else "linear placement only"
        if request.warp == "none" and record.deformation
        else "no deformation"
    )
    atlas_words = " + ".join(layers) or "none"
    if options.atlas_images and options.atlas_opacity > 0:
        atlas_words += f" (images at {options.atlas_opacity:g})"
    border_word = image.info.get("border_word")
    if border_word:
        atlas_words += f"; borders {border_word}"
    if options.regions:
        emphasis = "only" if request.region_only else "drawn thick, the other borders faint"
        atlas_words += f"; regions {', '.join(name for name, _ids in options.regions)} {emphasis}"
    read = (
        placed.canvas.panels[0].from_file
        if placed.canvas is not None and placed.canvas.panels
        else None
    )
    calibration = placed.row.get("calibration") or {}
    uncalibrated = UNCALIBRATED if calibration.get("source") == "uncalibrated" else ""
    text = (
        f"{record.id} overlay{_orientation(record)}{_registration_words(request, record)}, "
        f"{_where(record)}; "
        f"{placed.row.get('transform_words') or 'identity transform'}, "
        f"{warp_words}; {um:.1f} um/px{uncalibrated}; {full}; atlas {atlas_words}"
        + (f";{FROM_FILE}" if read is not None else "")
    )
    held = snapshot(state, [record.id])
    one = replace(request, sections=(record.id,))
    recipe = _recipe(one, held, base=base, shown=shown, um_per_px=um)
    from langslice.core.fits.landmarks import draw_landmarks, section_panel
    from langslice.core.pictures.layers import (
        picture_to_native,
        section_to_picture,
    )

    panels = []
    if placed.canvas is not None and placed.canvas.panels:
        panel = placed.canvas.panels[0]
        swap = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
        to_section = np.linalg.inv(swap @ section_to_picture(panel) @ swap)
        warp_key = None
        if drawn:
            applied = record.deformation
            warp_key = "" if applied is None else applied.key or ""
        panels = [
            section_panel(
                record,
                placed.canvas.frame.section.size,
                pixel_to_render=to_section,
                bounds=panel.content_box,
                warp_key=warp_key,
            ),
            {
                "role": "atlas",
                "section": record.id,
                "bounds": list(panel.content_box),
                "pixel_to_native": (swap @ picture_to_native(panel) @ swap).tolist(),
                "native_size": list(panel.geometry.annotation.shape[::-1]),
            },
        ]
        draw_landmarks(image, panels, state, record, warp=placed.canvas.frame.warp)
    drawn_layers = _draw_layers(
        image,
        panels,
        state,
        record,
        request,
        ws,
        warp_record=placed.canvas.frame.warp if placed.canvas else None,
    )
    drawn_masks = _draw_masks(
        image,
        panels,
        state,
        record,
        request,
        ws,
        warp=placed.canvas.frame.warp if placed.canvas else None,
    )
    text += _mask_words(drawn_masks, record)
    recipe["endpoint_panels"] = panels
    recipe["registration_context"] = _landmark_context(state, record)
    annotate(image, recipe=recipe, caption=text)
    return LookPicture(
        image=image,
        caption=text,
        recipe=recipe,
        sections=(record.id,),
        mode="overlay",
        um_per_px=um,
        extra={
            **placed.row,
            **({"layers": drawn_layers} if drawn_layers else {}),
            **({"masks": drawn_masks} if drawn_masks else {}),
        },
    )


def overlay_region_window(
    ws: Workspace,
    state: Stack,
    recipe: dict[str, Any],
    region: str,
    *,
    store: RecordStore | None = None,
) -> tuple[dict[str, Any], tuple[float, ...]]:
    """A single region's bounds on the saved overlay's full canvas.

    Use its snapshot and the renderer's side-aware contours, including the
    registration's mirror, rather than a box measured on today's placement.
    The margin is 5% of the region's extent on each axis.
    """
    request = LookRequest.from_args(recipe["args"])
    request = replace(request, zoom=(), highlight=(region,), region_only=True)
    held = recipe.get("state") or {}
    drawn = restored(state, held) if held else state
    records = _records(drawn, request.sections)
    if len(records) != 1:
        raise Refused(
            RefusalCode.BAD_ARGUMENT,
            "region needs an overlay picture of one section; "
            "draw that section's overlay alone and give its "
            "picture.",
        )
    record = records[0]
    names, image = _images_for(ws, record, request.images)
    options = _options(
        request,
        names=names,
        image=image,
        layers=_atlas_layers(ws, request),
        long_edge=int(request.long_edge or picture_edge(ws)),
        regions=_highlighted(ws, request),
    )
    placed = placement_pictures(ws, drawn, record, record.position_mm, options, {}, store=store)
    assert placed.canvas is not None
    panel = placed.canvas.panels[0]
    if not panel.highlighted:
        raise Refused(RefusalCode.NOT_FOUND, f"Region {region!r} is absent from this overlay.")
    points = np.concatenate(panel.highlighted)
    geometry = panel.geometry
    points = points * geometry.atlas_scale + np.asarray(geometry.atlas_offset)
    lower, upper = points.min(axis=0), points.max(axis=0)
    margin = np.maximum((upper - lower) * 0.05, 2 * geometry.atlas_scale)
    lower = np.maximum(0, (lower - margin) / np.asarray(geometry.size))
    upper = np.minimum(1, (upper + margin) / np.asarray(geometry.size))
    window = (float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1]))
    return {**recipe, "args": request.args()}, window


def _draw_layers(
    picture: Image.Image,
    panels: list[dict[str, Any]],
    state: Stack,
    record: Section,
    request: LookRequest,
    ws: Workspace,
    *,
    warp_record: DeformableRecord | None = None,
) -> list[dict[str, Any]]:
    """The plugin layers *request* names, drawn on *picture* over its *panels*
    (:mod:`langslice.core.pictures.plugin_layers`); one note per layer, empty
    when the
    request names none."""
    if not request.layers or not panels:
        return []
    from langslice.core.pictures.plugin_layers import draw_layers

    return draw_layers(
        picture, panels, state, record, request.layers, workspace=ws, warp_record=warp_record
    )


def _draw_masks(
    picture: Image.Image,
    panels: list[dict[str, Any]],
    state: Stack,
    record: Section,
    request: LookRequest,
    ws: Workspace,
    *,
    warp: Any = None,
) -> list[dict[str, Any]]:
    """The masks *request* names, outlined on *picture* over its *panels* where their
    weight reaches 0.5, drawn as a scalar layer by
    :mod:`langslice.core.pictures.plugin_layers` (never a fill). One note per
    mask:
    ``{"name", "kind", "stale", "drawn"}`` and, for one that is not drawn,
    ``"reason"``. A mask this section does not hold is a note, not a refusal."""
    if not request.masks or not panels:
        return []
    from langslice.core.pictures.plugin_layers import (
        _colour,
        _draw_scalar,
        _Mapper,
    )
    from langslice.core.plugins import Layer
    from langslice.core.sections import masks as section_masks
    from langslice.core.sections.native import unorient

    held = record.masks
    mapper = _Mapper(state, record, panels, ws, warp)
    notes: list[dict[str, Any]] = []
    for index, name in enumerate(request.masks):
        entry: dict[str, Any] = {"name": name}
        notes.append(entry)
        if name not in held:
            entry.update(
                drawn=False, reason=f"{record.id} has no mask {name!r}; make it with make_mask"
            )
            continue
        try:
            if ws.job_folder is None:
                raise ValueError("this workspace has no job folder to read a mask from")
            view = section_masks.read_mask(ws.job_folder, state, record, name)
            raster = np.asarray(
                unorient(
                    Image.fromarray(np.asarray(view.weights, dtype=np.float32), mode="F"),
                    record.rotation_deg,
                    record.flip,
                ),
                dtype=np.float32,
            )
            shown = _draw_scalar(
                picture,
                mapper,
                Layer.scalar(raster, frame="section_px", level=0.5, name=name),
                _colour(index),
            )
            entry.update(
                kind=view.kind,
                stale=bool(view.stale),
                drawn=True,
                shown=shown,
                earlier_registration=bool(view.earlier_registration),
            )
        except KeyError:
            entry.update(
                drawn=False, reason=f"{record.id} has no mask {name!r}; make it with make_mask"
            )
        except (OSError, ValueError) as exc:
            logger.warning("mask %s of %s could not be drawn", name, record.id, exc_info=True)
            entry.update(drawn=False, reason=f"the mask could not be read or placed ({exc})")
    return notes


def _mask_words(notes: list[dict[str, Any]], record: Section) -> str:
    """The caption sentence saying what a mask outline means (the ringed patches are the
    section pixels the mask leaves out of a fit) and the mask's name and source."""
    parts = []
    for note_ in notes:
        if not note_.get("drawn"):
            continue
        entry = record.masks.get(cast("str", note_["name"]))
        if note_.get("kind") == "fit":
            source = "copied from the sorting the section's deformation saved"
        else:
            section_image, atlas_image = "?", "?"
            if entry is not None:
                section_image = entry.section_image
                atlas_image = entry.atlas_image
            source = f"sorted from section image {section_image} against atlas image {atlas_image}"
        parts.append(f"{note_['name']!r} ({source})")
    if not parts:
        return ""
    return (
        "; coloured outlines trace the edge of the mask (its weight crossing 0.5): the "
        "section pixels in the ringed patches are the ones it leaves out of a fit that "
        "uses it, the rest is compared; mask " + ", ".join(parts)
    )


def _landmark_context(state: Stack, record: Section) -> dict[str, Any]:
    from langslice.core.fits.landmarks import context

    return context(state, record)


def _atlas_picture(
    ws: Workspace,
    state: Stack,
    position_mm: float,
    request: LookRequest,
    long_edge: int,
    base: tuple[int, int] | None,
    base_um: float | None,
    angles: Angles,
) -> LookPicture:
    layers = _atlas_layers(ws, request)
    options = _options(request, names=(), image="", layers=layers, long_edge=long_edge)
    voxel = atlas_um_per_px(ws.atlas)
    measured = base is None or base_um is None
    whole = None
    if base is None or base_um is None:
        whole = framed_atlas(ws, state, position_mm, options, angles=angles)
        native = atlas_section(ws, state, position_mm, frame=True, angles=angles)
        base, base_um = whole.size, voxel * max(native.size) / max(whole.size)
    if request.zoom:
        extent = max(
            (request.zoom[2] - request.zoom[0]) * base[0],
            (request.zoom[3] - request.zoom[1]) * base[1],
        )
        um = max(base_um * extent / long_edge, voxel / ATLAS_UPSAMPLE)
        at_um = (base[0] * base_um / um, base[1] * base_um / um)
        options = replace(options, zoom=_zoom_pixels(request.zoom, at_um))
        picture = framed_atlas(ws, state, position_mm, options, angles=angles, um_per_px=um)
    else:
        picture = (
            whole
            if whole is not None
            else framed_atlas(ws, state, position_mm, options, angles=angles)
        )
        um = base_um
        if request.long_edge:
            wanted = (
                max(base_um * max(picture.size) / int(request.long_edge), voxel / ATLAS_UPSAMPLE)
                if measured
                else base_um
            )
            if wanted < voxel * 0.99:
                um = wanted
                picture = framed_atlas(ws, state, position_mm, options, angles=angles, um_per_px=um)
                base = picture.size
    section = state.by_id(request.sections[0]) if len(request.sections) == 1 else None
    median = state.drawn_at_median(angles) if section is None else False
    label = _zoom_tag(request.zoom, base, picture.size) + atlas_caption(
        state, position_mm, options, angles=angles, median=median
    )
    layers_words = " + ".join(layers) or "none"
    plane = (
        view_angles_label(angles, median=True)
        if median
        else f"{angles_label(angles)} (the stack's cutting angles)"
    )
    if section:
        plane = f"{angles_label(angles)} ({section.id}'s cutting angles)"
    border_word = picture.info.get("border_word")
    text = f"atlas at {position_mm:.2f} mm{plane}; {um:.1f} um/px; layers {layers_words}" + (
        f"; borders {border_word}" if border_word else ""
    )
    panels = []
    section = state.by_id(request.sections[0]) if len(request.sections) == 1 else None
    if section is not None:
        from langslice.core.fits.landmarks import draw_landmarks

        panel = {
            "role": "atlas",
            "section": section.id,
            "bounds": [0, 0, *picture.size],
            "pixel_to_native": picture.info["pixel_to_native"],
            "native_size": list(atlas_section(ws, state, position_mm, angles=angles).size),
        }
        panels = [panel]
        picture = picture.copy()
        draw_landmarks(picture, panels, state, section)
    image = caption(picture, label)
    held = snapshot(state, [section.id] if section else [], view_angles=True)
    one = replace(request, positions_mm=(float(position_mm),))
    recipe = _recipe(one, held, base=base, shown=picture.size, um_per_px=um)
    recipe["endpoint_panels"] = panels
    if section:
        recipe["registration_context"] = _landmark_context(state, section)
    note(
        image,
        sections=(section.id,) if section else (),
        mode="atlas",
        recipe=recipe,
        caption=text,
        extra={"position_mm": float(position_mm)},
    )
    return LookPicture(
        image=image,
        caption=text,
        recipe=recipe,
        sections=(section.id,) if section else (),
        mode="atlas",
        um_per_px=um,
        extra={"position_mm": float(position_mm)},
    )


def _positioning(
    ws: Workspace,
    state: Stack,
    request: LookRequest,
    long_edge: int,
    angles: Angles,
    zoom_edge: int | None,
) -> list[LookPicture]:
    records = _records(state, request.sections)
    layers = _atlas_layers(ws, request)
    atlas_options = _options(request, names=(), image="", layers=layers, long_edge=long_edge)
    shown_words: dict[str, str] = {}

    def look_of(record: Section) -> Any:
        names, image = _images_for(ws, record, request.images)
        shown_words.setdefault(record.id, image_words(ws, record, names, image)[0])
        return _options(request, names=names, image=image, layers=(), long_edge=long_edge).look(
            state, record
        )

    for record in records:
        look_of(record)
    words = sorted(set(shown_words.values()))
    try:
        drawn = positioning_pictures(
            ws,
            state,
            records,
            [float(v) for v in request.positions_mm],
            look_of=look_of,
            atlas_options=atlas_options,
            angles=angles,
            tile_edge=long_edge,
            max_width=POSITIONING_MAX_WIDTH,
            part=request.part,
            window=request.zoom,
            zoom_edge=zoom_edge,
            shown=words[0] if len(words) == 1 else "",
        )
    except IndexError as exc:
        raise Refused(RefusalCode.BAD_ARGUMENT, message=str(exc)) from None
    held = snapshot(state, [record.id for record in records], view_angles=True)
    whole = replace(request, sections=tuple(r.id for r in records))
    out: list[LookPicture] = []
    for one in drawn:
        layout = one.layout
        ids = tuple(slot.key for slot in layout.sections)
        rows = ", ".join(f"{slot.label[0]} {slot.position_mm:.2f} mm" for slot in layout.sections)
        text = one.caption + (f". Sections: {rows}" if rows else "")
        counts = [
            f"{section.id}: {len(section.landmarks.pairs)}"
            for section in records
            if section.id in ids and section.landmarks and section.landmarks.pairs
        ]
        if counts:
            counts_text = "landmarks " + ", ".join(counts)
            text += f"; {counts_text}"
        recipe = _recipe(
            replace(whole, part=layout.part),
            held,
            base=(layout.width, layout.height),
            shown=one.content,
            um_per_px=one.um_per_px,
        )
        extra = {
            "part": layout.part,
            "parts": layout.parts,
            "positions_mm": [slot.position_mm for slot in layout.atlas],
            "range_mm": [float(v) for v in layout.range_mm],
        }
        note(one.image, sections=ids, mode="positioning", recipe=recipe, caption=text, extra=extra)
        out.append(
            LookPicture(
                image=one.image,
                caption=text,
                recipe=recipe,
                sections=ids,
                mode="positioning",
                um_per_px=one.um_per_px,
                extra=extra,
            )
        )
    return out


def look(
    ws: Workspace,
    state: Stack,
    request: LookRequest,
    *,
    store: RecordStore | None = None,
    angles: Angles | None = None,
    base: tuple[int, int] | None = None,
    base_um: float | None = None,
    zoom_edge: int | None = None,
) -> list[LookPicture]:
    """The pictures of *request*: one per section (``section``,
    ``overlay``), one per position (``atlas``), one per run of positions
    (``positioning``: one for a short stack; with ``part``, that one).

    *store* (a :class:`~langslice.core.fits.deformation.RecordStore`) gives an
    overlay its applied deformation. *angles* is the plane an atlas picture
    without a section is drawn at (None: ``Stack.view_angles``). *base* /
    *base_um* are the unzoomed picture's content size and scale when the
    request zooms (:func:`redraw` passes the recipe's); without them the
    unzoomed picture is measured first. *zoom_edge* is a positioning zoom's
    long edge (None: the unzoomed picture's).
    """
    if request.mode not in MODES:
        raise Refused(RefusalCode.BAD_ARGUMENT, f"mode must be one of {', '.join(MODES)}")
    if request.warp not in WARP_CHOICES:
        raise Refused(RefusalCode.BAD_ARGUMENT, f"warp must be one of {', '.join(WARP_CHOICES)}")
    long_edge = int(request.long_edge or picture_edge(ws))
    plane = angles if angles is not None else state.view_angles
    if request.mode == "positioning":
        return _positioning(ws, state, request, long_edge, plane, zoom_edge)
    if request.mode == "atlas":
        if request.sections:
            records = _records(state, request.sections)
            own = tuple(record.position_mm for record in records)
            if request.positions_mm and tuple(request.positions_mm) != own:
                raise Refused(
                    RefusalCode.BAD_ARGUMENT,
                    "Name sections or give positions_mm, not both: a "
                    "section's atlas picture is at its own plane.",
                )
            return [
                _atlas_picture(
                    ws,
                    state,
                    record.position_mm,
                    replace(request, sections=(record.id,), positions_mm=(record.position_mm,)),
                    long_edge,
                    base,
                    base_um,
                    record.angles,
                )
                for record in records
            ]
        if not request.positions_mm:
            raise Refused(
                RefusalCode.BAD_ARGUMENT,
                "atlas mode needs positions_mm or sections; nothing was drawn.",
            )
        return [
            _atlas_picture(ws, state, float(mm), request, long_edge, base, base_um, plane)
            for mm in request.positions_mm
        ]
    records = _records(state, request.sections)
    if request.mode == "section":
        return [_section_picture(ws, state, record, request, long_edge, base) for record in records]
    return [
        _overlay_picture(ws, state, record, request, long_edge, base, store) for record in records
    ]


def redraw(
    recipe: dict[str, Any],
    window: Sequence[float],
    ws: Workspace,
    state: Stack,
    *,
    store: RecordStore | None = None,
    zoom_edge: int | None = None,
) -> LookPicture:
    """The picture of *recipe* drawn again at *window* (fractions of its
    unzoomed content; empty: whole), from the stack as it was when the
    recipe was made (``stale`` when it has changed since).
    :class:`Refused` ``NOT_FOUND`` when a section it showed is gone.
    """
    request = replace(LookRequest.from_args(recipe["args"]), zoom=tuple(float(v) for v in window))
    held = recipe.get("state") or {}
    named = {name for name, _number, was_held in request.registrations if not was_held}
    stale = bool(held) and changed_since(state, held, besides=named)
    drawn = restored(state, held) if bool(held) and differs_since(state, held) else state
    angles = held.get("view_angles")
    base = recipe.get("base")
    pictures = look(
        ws,
        drawn,
        request,
        store=store,
        angles=(float(angles[0]), float(angles[1])) if angles else None,
        base=(int(base[0]), int(base[1])) if base else None,
        base_um=_base_um(recipe),
        zoom_edge=zoom_edge,
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
    "ATLAS_LAYER_DEFAULTS",
    "LookPicture",
    "LookRequest",
    "MODES",
    "RENDERER",
    "changed_since",
    "differs_since",
    "image_words",
    "look",
    "redraw",
    "restored",
    "snapshot",
]
