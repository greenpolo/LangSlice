"""The ``look`` tool, and the pictures change tools show of what they wrote.

These save their own pictures: each picture goes through the job's picture
store (:class:`langslice.job.pictures.ViewStore`) with its recipe and
caption, so it gets a picture number at once, and the reply lists
``[{"id", "caption"}]`` per picture. The tool door sends the returned PIL
pictures with those numbers beside them and must not save them again.

- :func:`look_tool` is the tool: it builds the call's
  :class:`~langslice.core.pictures.look.LookRequest` from the model's
  arguments and :func:`look_pictures` checks it and draws one of the four
  modes (:mod:`langslice.core.pictures.look`). A call shows every picture
  drawn, or at most ``Job.picture_cap`` of them (the agent tools and MCP);
  the rest are named in ``not_shown`` with the arguments that get them.
  With ``registrations`` a section is drawn under an earlier registration
  of its own, writing nothing; with ``sheet`` the pictures are saved as
  usual and shown tiled into sheets (:mod:`langslice.core.pictures.sheet`).
- :func:`show_result` draws the picture a change tool shows of what it wrote.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core.atlas.levels import OVERLAY_DEPTH
from langslice.core.pictures import look as looks
from langslice.core.pictures import sheet as sheets
from langslice.core.pictures.drawn import PictureNote, collecting, note_for
from langslice.core.pictures.look import LookRequest
from langslice.core.pictures.sizes import (
    AUTO_RESOLUTION,
    MIN_RESOLUTION,
    clamp_resolution,
)
from langslice.core.refusal import RefusalCode, Refused
from langslice.job import registrations as regs
from langslice.job.pictures import has_frame
from langslice.tools.refusal import unknown_sections
from langslice.tools.run import as_list, pictured

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run
LOOK_TOOL = "look"


@dataclass(frozen=True)
class Looked:
    """What :func:`look_pictures` (and ``grep_atlas_view``) showed."""

    #: The pictures, in order, each saved with a number.
    pictures: list[Image.Image] = field(default_factory=list)
    #: ``{"id", "caption"}`` per picture: its number and index caption.
    entries: list[dict[str, Any]] = field(default_factory=list)
    #: Pictures drawn but left out of this reply: ``{"mode", "sections",
    #: "positions_mm", "caption", "how"}`` each; "how" says how to get it.
    not_shown: list[dict[str, Any]] = field(default_factory=list)
    #: :func:`show_result` only: ``{"id", "error", "message"}`` per section
    #: whose picture could not be drawn (the write stands).
    failed: list[dict[str, Any]] = field(default_factory=list)
    #: ``look`` with ``layers``: per picture and layer asked for, ``{"picture",
    #: "section", "name", "tool", "kind", "drawn"}`` (and ``reason`` for a layer
    #: not drawn).
    layers: list[dict[str, Any]] = field(default_factory=list)
    #: ``look`` with ``masks``: per picture and mask asked for, ``{"picture",
    #: "section", "name", "drawn"}`` (and ``kind``, ``stale`` for a mask drawn, ``reason``
    #: for one not).
    masks: list[dict[str, Any]] = field(default_factory=list)
    #: ``look`` with ``sheet``: the sheets' ``{"id", "caption", "tiles"}``
    #: (``pictures`` then holds the sheets, and ``entries`` the tiled pictures).
    sheets: list[dict[str, Any]] = field(default_factory=list)


# --- saving ------------------------------------------------------------------------------


def _picture_id(name: str) -> int:
    return int(name.split("_", 1)[0])


def save_pictures(
    job: Job, workspace: Workspace, tool: str, drawn: Sequence[Any],
    notes: list[tuple[Image.Image, PictureNote]], arguments: dict[str, Any], *,
    tiles: bool = False,
) -> list[dict[str, Any]]:
    """Save *drawn* (objects with ``image``, ``caption``, ``recipe``,
    ``sections``, ``mode``) through the job's picture store, with the notes
    the core made while drawing them (*notes*, from
    :func:`langslice.core.pictures.drawn.collecting`); ``[{"id", "caption"}]``.
    *tiles*: a sheet shows them
    (:meth:`langslice.job.pictures.ViewStore.save`)."""
    if not drawn:
        return []
    items: list[tuple[Image.Image | bytes, PictureNote | None]] = []
    for picture in drawn:
        held = note_for(picture.image, notes)
        if held is None:
            held = PictureNote(sections=tuple(picture.sections), mode=picture.mode)
        if held.recipe is None:
            held.recipe = picture.recipe
        if held.caption is None:
            held.caption = picture.caption
        items.append((picture.image, held))
    placed = any(has_frame(held) for _image, held in items)
    names = job.views.save(tool=tool, pictures=items, arguments=arguments,
                           atlas=workspace.atlas if placed else None, tiles=tiles)
    return [{"id": _picture_id(name), "caption": picture.caption}
            for name, picture in zip(names, drawn, strict=True)]


def split_shown(job: Job, drawn: Sequence[Any]) -> tuple[list[Any], list[Any]]:
    """The pictures a reply shows and the rest: all of them, or the first
    ``Job.picture_cap``."""
    cap = job.picture_cap
    if cap is None:
        return list(drawn), []
    return list(drawn[:cap]), list(drawn[cap:])


def not_shown(pictures: Sequence[Any], mode: str, *, verb: str = "") -> list[dict[str, Any]]:
    """The pictures left out of a reply, each with the arguments that get it
    (from its recipe: the sections and positions it shows): of this call
    again, or with *verb* of that verb (``look``, for a change tool's
    picture)."""
    out: list[dict[str, Any]] = []
    for picture in pictures:
        args = (picture.recipe or {}).get("args") or {}
        sections = list(picture.sections) or list(args.get("sections") or [])
        positions = [float(v) for v in args.get("positions_mm") or []]
        ask = {"mode": mode, **({"sections": sections} if sections else {}),
               **({"positions_mm": positions} if positions else {})}
        out.append({"mode": mode, "sections": sections, "positions_mm": positions,
                    "caption": picture.caption,
                    "how": (f"{verb} with {_words(ask)}" if verb else
                            f"call again with {_words(ask)} (the other arguments as before)")})
    return out


def _words(arguments: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


# --- look --------------------------------------------------------------------------------


def _strings(value: Any, name: str) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise Refused(RefusalCode.BAD_ARGS, message=f"{name} must be a list.")
    return list(value)


def look_tool(
    run: Run,
    mode: str,
    sections: list[str],
    images: list[str],
    *,
    positions_mm: list[float] | None = None,
    atlas_layers: list[str] | None = None,
    atlas_opacity: float = 0.5,
    warp: str = "applied",
    resolution: int = 0,
    layers: list[str] | None = None,
    depth: int = OVERLAY_DEPTH,
    masks: list[str] | None = None,
    registrations: dict[str, int] | None = None,
    sheet: bool = False,
) -> dict[str, object]:
    """Draw and save the pictures of one ``look`` call; return its reply.

    Parameters
    ----------
    run
        The run the tool is called in.
    mode, sections, images, positions_mm, atlas_layers, atlas_opacity, warp
        The declared arguments as the model sent them
        (``tools/descriptions/looking.py``).
    resolution, layers, depth, masks, registrations, sheet
        More of the declared arguments, as sent.

    Returns
    -------
    dict
        The reply: ``status``, a ``resolution_note`` when the size was
        clamped, the ``layers`` and ``masks`` drawn, the pictures with
        their numbers (``sheet``: the sheets, the tiled pictures under
        ``tiled``) and the pictures left out.
    """
    request = LookRequest(
        mode=str(mode or "").strip().lower(),
        sections=tuple(as_list(sections)),
        positions_mm=tuple(as_list(positions_mm)),
        images=tuple(as_list(images)),
        atlas_layers=tuple(as_list(atlas_layers)) or None,
        atlas_opacity=atlas_opacity,
        warp=str(warp or "applied").strip().lower(),
        long_edge=resolution,
        layers=tuple(as_list(layers)),
        depth=depth,
        masks=tuple(as_list(masks)),
    )
    edge, note = requested_edge(run, request.long_edge)
    sheet_edge = int(run.box.max_view_edge) if sheet else None
    looked = look_pictures(
        run.job,
        run.workspace,
        replace(request, long_edge=edge),
        registrations=dict(registrations or {}),
        sheet_edge=sheet_edge,
    )
    reply: dict[str, object] = {"status": "ok"}
    if note:
        reply["resolution_note"] = note
    if looked.layers:
        reply["layers"] = looked.layers
    if looked.masks:
        reply["masks"] = looked.masks
    reply = pictured(reply, looked)

    # the sheets are attached; the pictures they tile are saved
    if looked.sheets:
        reply["tiled"] = reply["pictures"]
        reply["pictures"] = [
            {
                "id": made["id"],
                "caption": made["caption"],
                "tiles": made["tiles"],
            }
            for made in looked.sheets
        ]
    return reply


def requested_edge(
    run: Run, resolution: float | None
) -> tuple[int | None, str]:
    """Return the long edge a tool's ``resolution`` asks for, and its note.

    Only a run whose caller sizes the pictures (image resolution ``auto``)
    reads it; every other run draws at its level.

    Parameters
    ----------
    run
        The run the tool is called in.
    resolution
        The model's ``resolution``; 0 or None is the default size.

    Returns
    -------
    edge : int or None
        The long edge, held between the smallest picture and the largest
        the driver model takes; None for the run's default.
    note : str
        What the reply says when the edge was clamped; empty otherwise.

    Raises
    ------
    Refused
        ``BAD_ARGS`` when ``resolution`` is not a number.
    """
    if not resolution or run.level != AUTO_RESOLUTION:
        return None, ""
    try:
        return clamp_resolution(resolution, run.box.max_view_edge)
    except (TypeError, ValueError):
        raise Refused(
            RefusalCode.BAD_ARGS, "resolution is a whole number of pixels."
        ) from None


def look_pictures(
    job: Job,
    workspace: Workspace,
    request: LookRequest,
    *,
    registrations: Mapping[str, int] | None = None,
    sheet_edge: int | None = None,
) -> Looked:
    """Draw *request*'s pictures and save each with its recipe and caption.

    The mode is ``section``, ``atlas``, ``overlay`` or ``positioning``
    (:func:`langslice.core.pictures.look.look`).

    The request comes as the door built it and is checked here: its
    ``sections`` are filenames (``Stack.resolve``) and its other fields the
    model's values. No field reads the job's state: no ``sections`` is every
    section, no ``images`` the ``preprocessed`` image (one section image is
    drawn alone, in gray; several raw channels together in their colours), no
    ``atlas_layers`` the mode's default. ``long_edge`` is a picture's long
    edge (clamped up to :data:`~langslice.core.pictures.sizes.MIN_RESOLUTION`);
    a door passes one only where its caller sizes the pictures. At most
    ``Job.picture_cap`` pictures (None: all) are saved and returned; the rest
    are ``not_shown``. Refused: ``BAD_ARGS``, ``UNKNOWN_SLICE_IDS``,
    ``UNKNOWN_MODE``, ``UNKNOWN_IMAGE``, ``MIXED_IMAGES``,
    ``TOO_MANY_CHANNELS``, ``UNKNOWN_LAYER``, ``NO_POSITIONS``,
    ``NO_POSITION``, ``BAD_WARP``, ``OPTION_OFF`` (the job disables the mode:
    ``look:<mode>`` in its ``disabled`` list).

    ``layers`` names the layers tool plugins and scripts left on the sections
    (``"all"``: every one) to draw on the pictures of mode ``section`` or
    ``overlay`` (:mod:`langslice.core.pictures.plugin_layers`); the reply's
    ``layers`` says which were drawn. Refused: ``BAD_ARGS`` in another mode,
    ``UNKNOWN_PLUGIN_LAYER`` for a name no drawn section has.

    ``masks`` names masks of the drawn sections, outlined where their weight
    is 0.5 on pictures of mode ``section`` or ``overlay`` (as a scalar layer);
    the reply's ``masks`` says which were drawn. Refused: ``BAD_ARGS`` in
    another mode, ``UNKNOWN_MASK`` for a name no drawn section holds.

    *registrations* (``{filename: number}``, mode ``section`` or ``overlay``)
    draws each section it names under that registration of its own
    (:func:`langslice.tools.registrations.entry_for`), writing nothing: the
    caption says which, and whether the section holds it; with no
    ``sections`` only those sections are drawn. Refused: ``BAD_ARGS``
    (another mode, a section not among ``sections``),
    ``UNKNOWN_REGISTRATION``, ``REGISTRATION_ELSEWHERE``,
    ``DEFORMATION_UNREADABLE``.

    *sheet_edge* (None: no sheet) shows the pictures tiled into sheets of at
    most that long edge, :data:`langslice.core.pictures.sheet.MAX_TILES` each:
    every picture is saved and numbered as usual (its JPEG an artifact of kind
    ``tile``), and ``pictures`` holds the sheets.
    """
    # the request as the model gave it, checked below
    mode, sections = request.mode, request.sections
    positions_mm, images = request.positions_mm, request.images
    atlas_layers, atlas_opacity = request.atlas_layers, request.atlas_opacity
    warp, resolution, depth = request.warp, request.long_edge, request.depth
    plugin_layers, masks = request.layers, request.masks
    state = job.state
    if str(mode) in looks.MODES and job.spec.off(f"look:{mode}"):
        on = [name for name in looks.MODES if not job.spec.off(f"look:{name}")]
        raise Refused(RefusalCode.OPTION_OFF, option=f"look:{mode}", modes=on,
                      message=f"look mode {mode} is switched off for this job; the modes "
                      f"are {', '.join(on)}.")
    refs = _strings(sections, "sections")
    ids: list[str] = []
    unknown: list[Any] = []
    for ref in refs:
        record = state.resolve(ref)
        if record is None:
            unknown.append(ref)
        else:
            ids.append(record.id)
    if unknown:
        raise unknown_sections(state, [str(ref) for ref in unknown])
    shown_under: tuple[tuple[str, int, bool], ...] = ()
    if registrations:
        state, ids, shown_under = _under_registrations(job, str(mode), ids, registrations)
    try:
        positions = tuple(float(v) for v in _strings(positions_mm, "positions_mm"))
    except (TypeError, ValueError):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="positions_mm must be numbers.") from None
    layers = None if atlas_layers is None else tuple(str(v) for v in _strings(
        atlas_layers, "atlas_layers"))
    if atlas_opacity is not None and not 0.0 <= float(atlas_opacity) <= 1.0:
        raise Refused(RefusalCode.BAD_ARGS,
                      message="atlas_opacity is between 0 and 1.")
    edge = max(int(resolution), MIN_RESOLUTION) if resolution else None
    named = tuple(str(v) for v in _strings(plugin_layers, "layers"))
    if named:
        _check_plugin_layers(job, str(mode), ids, named)
    mask_names = tuple(str(v) for v in _strings(masks, "masks"))
    if mask_names:
        _check_masks(job, str(mode), ids, mask_names)
    if isinstance(depth, bool) or not isinstance(depth, int) or depth < 0:
        raise Refused(RefusalCode.BAD_ARGS,
                      message="depth is a whole number, 0 or more.")
    request = looks.LookRequest(
        mode=str(mode), sections=tuple(ids), positions_mm=positions,
        images=tuple(str(c) for c in _strings(images, "images")), atlas_layers=layers,
        atlas_opacity=None if atlas_opacity is None else float(atlas_opacity),
        warp=str(warp or "applied"), long_edge=edge, layers=named, masks=mask_names, depth=depth,
        registrations=shown_under)
    with collecting() as notes:
        drawn = looks.look(workspace, state, request, store=job.deformations)
    if shown_under:
        numbers = {name: number for name, number, _held in shown_under}
        for picture in drawn:
            held = note_for(picture.image, notes)
            if held is not None and picture.sections and picture.sections[0] in numbers:
                held.extra["registrations"] = {picture.sections[0]: numbers[picture.sections[0]]}
    arguments = request.args()
    if registrations:
        arguments["registrations"] = {name: number for name, number, _held in shown_under}
    if sheet_edge:
        return _sheet_reply(job, workspace, drawn, notes, request.mode, arguments,
                            int(sheet_edge))
    return _reply(job, workspace, LOOK_TOOL, drawn, notes, request.mode, arguments)



def _under_registrations(
    job: Job, mode: str, ids: list[str], registrations: Mapping[str, Any],
) -> tuple[Any, list[str], tuple[tuple[str, int, bool], ...]]:
    """``(state, ids, shown)``: a copy of the job's state with each section
    *registrations* names under that registration of its own, the sections to
    draw, and ``(section, number, held)`` per section named. Nothing is written."""
    from langslice.tools import registrations as ops_registrations

    if mode not in ("section", "overlay"):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="registrations are drawn in mode section or "
                      "overlay.")
    if not isinstance(registrations, Mapping):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="registrations is {filename: number}.")
    trial = copy.deepcopy(job.state)
    shown: list[tuple[str, int, bool]] = []
    for ref, number in registrations.items():
        record, entry, held = ops_registrations.entry_for(job, ref, number)
        if ids and record.id not in ids:
            raise Refused(RefusalCode.BAD_ARGS,
                          message=f"registrations names {record.id}, which is not "
                          "among the sections to draw; add it to sections, or leave sections "
                          "empty.")
        copy_ = trial.by_id(record.id)
        assert copy_ is not None
        copy_.flip = bool(entry.get("flip"))
        copy_.rotation_deg = int(entry.get("rotation_deg") or 0)
        copy_.transform = regs.entry_transform(entry)
        copy_.deformation = regs.entry_deformation(entry)
        shown.append((record.id, int(entry["id"]), held == int(entry["id"])))
    drawn = ids or [name for name, _number, _held in shown]
    return trial, drawn, tuple(shown)


def _check_plugin_layers(job: Job, mode: str, ids: Sequence[str], names: Sequence[str]) -> None:
    from langslice.core.pictures.plugin_layers import known_names

    if mode not in ("section", "overlay"):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="layers are drawn on pictures of mode section or "
                      "overlay.")
    held = known_names(job.state, ids)
    unknown = [name for name in names if name != "all" and name not in held]
    if unknown:
        raise Refused(RefusalCode.UNKNOWN_PLUGIN_LAYER,
                      unknown=unknown, layers=held,
                      message=f"No drawn section has a layer named {', '.join(unknown)}; "
                      f"the layers: {', '.join(held) or 'none'}.")


def _check_masks(job: Job, mode: str, ids: Sequence[str], names: Sequence[str]) -> None:
    if mode not in ("section", "overlay"):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="masks are outlined on pictures of mode section or "
                      "overlay.")
    state = job.state
    records = [r for r in (state.by_id(i) for i in ids) if r is not None] if ids \
        else list(state.slices)
    held = sorted({name for record in records for name in record.masks})
    unknown = [name for name in names if name not in held]
    if unknown:
        raise Refused(RefusalCode.UNKNOWN_MASK, unknown=unknown, masks=held,
                      message=f"No drawn section has a mask named {', '.join(unknown)}; the "
                      f"masks: {', '.join(held) or 'none'}. Make one with make_mask.")


def _layer_notes(shown: Sequence[looks.LookPicture], entries: Sequence[dict[str, Any]],
                 ) -> list[dict[str, Any]]:
    """The layers each shown picture drew or could not draw, with its number."""
    out: list[dict[str, Any]] = []
    for picture, entry in zip(shown, entries, strict=False):
        for item in picture.extra.get("layers") or []:
            out.append({"picture": entry["id"],
                        "section": picture.sections[0] if picture.sections else "", **item})
    return out


def _mask_notes(shown: Sequence[looks.LookPicture], entries: Sequence[dict[str, Any]],
                ) -> list[dict[str, Any]]:
    """The masks each shown picture outlined or could not, with its number."""
    out: list[dict[str, Any]] = []
    for picture, entry in zip(shown, entries, strict=False):
        for item in picture.extra.get("masks") or []:
            out.append({"picture": entry["id"],
                        "section": picture.sections[0] if picture.sections else "", **item})
    return out


def _reply(
    job: Job, workspace: Workspace, tool: str, drawn: list[looks.LookPicture],
    notes: list[tuple[Image.Image, PictureNote]], mode: str, arguments: dict[str, Any],
) -> Looked:
    shown, rest = split_shown(job, drawn)
    entries = save_pictures(job, workspace, tool, shown, notes, arguments)
    return Looked(pictures=[picture.image for picture in shown], entries=entries,
                  not_shown=not_shown(rest, mode), layers=_layer_notes(shown, entries),
                  masks=_mask_notes(shown, entries))


#: The mode a sheet is saved under (its folder ``<n>_look_sheet``).
SHEET_MODE = "sheet"


def _tile_label(picture: looks.LookPicture) -> str:
    what = ", ".join(picture.sections)
    if not what and picture.extra.get("position_mm") is not None:
        what = f"atlas at {float(picture.extra['position_mm']):.2f} mm"
    return f"{what} ({picture.mode})" if what else picture.mode


def _sheet_reply(
    job: Job, workspace: Workspace, drawn: list[looks.LookPicture],
    notes: list[tuple[Image.Image, PictureNote]], mode: str, arguments: dict[str, Any],
    long_edge: int,
) -> Looked:
    """Every picture drawn saved as usual (as tiles), then shown tiled into
    sheets of at most :data:`~langslice.core.pictures.sheet.MAX_TILES`, each
    saved as
    a picture of its own (mode ``sheet``) whose recipe lists its tiles."""
    entries = save_pictures(job, workspace, LOOK_TOOL, drawn, notes, arguments, tiles=True)
    images: list[Image.Image] = []
    made: list[dict[str, Any]] = []
    items: list[tuple[Image.Image | bytes, PictureNote | None]] = []
    parts: list[list[dict[str, Any]]] = []
    for start in range(0, len(drawn), sheets.MAX_TILES):
        part = list(zip(drawn[start:start + sheets.MAX_TILES],
                        entries[start:start + sheets.MAX_TILES], strict=True))
        numbers = [entry["id"] for _picture, entry in part]
        title = (f"sheet of pictures {', '.join(str(n) for n in numbers)}: each tile one "
                 "picture, scaled down; zoom a box on a tile to draw that picture's box")
        sheet = sheets.tile([sheets.Tile(picture.image, entry["id"], _tile_label(picture))
                             for picture, entry in part], long_edge, title=title)
        caption = (f"sheet of {len(part)} {mode} pictures, each scaled to "
                   f"{sheet.scale:.2f} of its size: "
                   + "; ".join(f"picture {entry['id']}: {picture.caption}"
                               for picture, entry in part))
        recipe = sheets.recipe(sheet)
        held = PictureNote(sections=tuple(dict.fromkeys(
            name for picture, _entry in part for name in picture.sections)),
            mode=SHEET_MODE, recipe=recipe, caption=caption)
        items.append((sheet.image, held))
        parts.append([entry for _picture, entry in part])
        made.append({"caption": caption, "tiles": [{"picture": tile["picture"], "box": tile["box"]}
                                                   for tile in sheet.tiles]})
        images.append(sheet.image)
    names = job.views.save(tool=LOOK_TOOL, pictures=items, arguments={**arguments, "sheet": True})
    for name, sheet_entry, tiled in zip(names, made, parts, strict=True):
        sheet_entry["id"] = _picture_id(name)
        for entry in tiled:
            entry["sheet"] = sheet_entry["id"]
    made = [{"id": entry.get("id", 0), "caption": entry["caption"], "tiles": entry["tiles"]}
            for entry in made]
    return Looked(pictures=images, entries=entries, layers=_layer_notes(drawn, entries),
                  masks=_mask_notes(drawn, entries), sheets=made)


# --- a change tool's picture --------------------------------------------------------------


def show_result(
    job: Job, workspace: Workspace, tool: str, mode: str, sections: Sequence[str], *,
    positions_mm: Sequence[float] = (), zooms: Mapping[str, Sequence[float]] | None = None,
    highlight: Sequence[str] = (), resolution: int | None = None,
    masks: Sequence[str] = (), images: Sequence[str] = (),
) -> Looked:
    """The picture a change tool shows of what it wrote, drawn as ``look``
    draws *mode* and saved under *tool* (so it has a number, a caption and a
    recipe, and can be zoomed).

    ``overlay``: one picture per section of *sections* (ids), each zoomed to
    its *zooms* window (``[x0, y0, x1, y1]`` fractions of the unzoomed
    picture) when given, the *highlight* regions' borders thick and the
    others faint (a restricted fit's ``restrict_to``). ``positioning``:
    *sections* and the atlas at
    *positions_mm* along the slicing axis. At most ``Job.picture_cap``
    are shown (None: all); the rest are ``not_shown`` (with the ``look`` call that
    draws them), and a section whose picture fails is ``failed``. *resolution*
    is the long edge of each picture, as ``look``'s (None: the job's level).
    *masks* names masks of the sections outlined on a ``section`` or ``overlay``
    picture (``make_mask``'s picture); *images* the section image drawn
    (``make_image``'s picture; empty: ``preprocessed``).
    """
    state = job.state
    drawn: list[looks.LookPicture] = []
    failed: list[dict[str, Any]] = []
    windows = zooms or {}
    edge = max(int(resolution), MIN_RESOLUTION) if resolution else None
    with collecting() as notes:
        if mode == "positioning":
            request = looks.LookRequest(mode=mode, sections=tuple(sections),
                                        positions_mm=tuple(float(v) for v in positions_mm))
            try:
                drawn = looks.look(workspace, state, request, store=job.deformations)
            except Exception as exc:  # noqa: BLE001 - the write stands; say why
                code = getattr(exc, "code", RefusalCode.RENDER_FAILED)
                failed.append({"error": code, "message": str(exc)})
        else:
            for section in sections:
                window = tuple(float(v) for v in windows.get(section) or ())
                request = looks.LookRequest(mode=mode, sections=(section,), zoom=window,
                                            highlight=tuple(str(v) for v in highlight),
                                            long_edge=edge,
                                            masks=tuple(str(v) for v in masks),
                                            images=tuple(str(v) for v in images))
                try:
                    drawn += looks.look(workspace, state, request, store=job.deformations)
                except Exception as exc:  # noqa: BLE001 - the write stands; say why
                    code = getattr(exc, "code", RefusalCode.RENDER_FAILED)
                    failed.append({"id": section, "error": code,
                                   "message": str(exc)})
    shown, rest = split_shown(job, drawn)
    entries = save_pictures(job, workspace, tool, shown, notes,
                            {"mode": mode, "sections": list(sections),
                             **({"positions_mm": [float(v) for v in positions_mm]}
                                if positions_mm else {})})
    return Looked(pictures=[picture.image for picture in shown], entries=entries,
                  not_shown=not_shown(rest, mode, verb=LOOK_TOOL), failed=failed)


__all__ = [
    "LOOK_TOOL",
    "SHEET_MODE",
    "Looked",
    "look_pictures",
    "look_tool",
    "not_shown",
    "save_pictures",
    "show_result",
]
