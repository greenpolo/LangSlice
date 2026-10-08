"""``look`` and ``zoom``: the two ways to see a stack, as functions on a job.

Unlike the older viewing verbs, these save their own pictures: each picture
goes through the job's picture store (:class:`langslice.job.views.ViewStore`)
with its recipe and caption, so it gets a picture number at once, and the
reply lists ``[{"id", "caption"}]`` per picture. The tool door sends the
returned PIL pictures with those numbers beside them and must not save them
again.

- :func:`look` draws one of the four modes (:mod:`langslice.core.look`). A
  call shows at most :data:`MAX_LOOK_PICTURES` pictures; any more the core
  drew are named in ``not_shown`` with the arguments that get them.
- :func:`zoom` draws a box of an earlier picture again at more detail
  (:mod:`langslice.core.zoom`), from the picture's recipe; a picture with no
  recipe, or whose sections are gone, is cropped from its saved image. The
  picture defaults to the newest one that is not itself a zoom.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.core import look as looks
from langslice.core import zoom as zooms
from langslice.core.layers import PictureNote, collecting, note_for
from langslice.core.sizes import MAX_IMAGES_PER_CALL, MIN_RESOLUTION
from langslice.job.views import PICTURE_FILE, PictureRecord, has_frame
from langslice.ops.refusal import Refused, unknown_sections

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job

#: Pictures one call shows; the rest are named, not drawn into the reply.
MAX_LOOK_PICTURES = MAX_IMAGES_PER_CALL
#: The tool name zoomed pictures are saved under (the picture index skips it
#: when it picks the default picture to zoom).
ZOOM_TOOL = "zoom"
LOOK_TOOL = "look"


@dataclass(frozen=True)
class Looked:
    """What :func:`look` (and ``grep_atlas_view``) showed."""

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


@dataclass(frozen=True)
class Zoomed:
    """What :func:`zoom` drew."""

    pictures: list[Image.Image] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    #: The number of the picture the box was read on.
    picture: int = 0
    #: Whether the zoom was drawn again from the source (False: cropped from
    #: the saved picture, which holds no more detail).
    redrawn: bool = True
    #: The stack has changed since the source picture; it was redrawn as it was.
    stale: bool = False


# --- saving ------------------------------------------------------------------------------


def _picture_id(name: str) -> int:
    return int(name.split("_", 1)[0])


def save_pictures(
    job: Job, workspace: Workspace, tool: str, drawn: Sequence[Any],
    notes: list[tuple[Image.Image, PictureNote]], arguments: dict[str, Any],
) -> list[dict[str, Any]]:
    """Save *drawn* (objects with ``image``, ``caption``, ``recipe``,
    ``sections``, ``mode``) through the job's picture store, with the notes
    the core made while drawing them (*notes*, from
    :func:`langslice.core.layers.collecting`); ``[{"id", "caption"}]``."""
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
                           atlas=workspace.atlas if placed else None)
    return [{"id": _picture_id(name), "caption": picture.caption}
            for name, picture in zip(names, drawn, strict=True)]


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
        raise Refused("BAD_ARGS", message=f"{name} must be a list.")
    return list(value)


def look(
    job: Job,
    workspace: Workspace,
    mode: str,
    sections: Sequence[Any] = (),
    positions_mm: Sequence[float] = (),
    channels: Sequence[str] = (),
    atlas_layers: Sequence[str] | None = None,
    atlas_opacity: float | None = None,
    warp: str = "applied",
    resolution: int | None = None,
) -> Looked:
    """Draw *mode* (``section``, ``atlas``, ``overlay`` or ``positioning``;
    :func:`langslice.core.look.look`) and save every picture with its recipe
    and caption.

    No argument reads the job's state: no *sections* is every section, no
    *channels* every raw channel, no *atlas_layers* the mode's default.
    *sections* are filenames (``StackState.resolve``). *resolution* is a picture's
    long edge (clamped up to :data:`~langslice.core.sizes.MIN_RESOLUTION`); a
    door passes one only where its caller sizes the pictures (image_resolution
    ``auto``, or the agent CLI and the library, whose toolbox works at
    ``auto`` whatever the job's level). At most
    :data:`MAX_LOOK_PICTURES` pictures are saved and returned; the rest are
    ``not_shown``. Refused: ``BAD_ARGS``, ``UNKNOWN_SLICE_IDS``,
    ``UNKNOWN_MODE``, ``UNKNOWN_CHANNEL``, ``MIXED_CHANNELS``,
    ``TOO_MANY_CHANNELS``, ``UNKNOWN_LAYER``, ``NO_POSITIONS``,
    ``NO_POSITION``, ``BAD_WARP``.
    """
    state = job.state
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
    try:
        positions = tuple(float(v) for v in _strings(positions_mm, "positions_mm"))
    except (TypeError, ValueError):
        raise Refused("BAD_ARGS", message="positions_mm must be numbers.") from None
    layers = None if atlas_layers is None else tuple(str(v) for v in _strings(
        atlas_layers, "atlas_layers"))
    if atlas_opacity is not None and not 0.0 <= float(atlas_opacity) <= 1.0:
        raise Refused("BAD_ARGS", message="atlas_opacity is between 0 and 1.")
    edge = max(int(resolution), MIN_RESOLUTION) if resolution else None
    request = looks.LookRequest(
        mode=str(mode), sections=tuple(ids), positions_mm=positions,
        channels=tuple(str(c) for c in _strings(channels, "channels")), atlas_layers=layers,
        atlas_opacity=None if atlas_opacity is None else float(atlas_opacity),
        warp=str(warp or "applied"), long_edge=edge)
    try:
        with collecting() as notes:
            drawn = looks.look(workspace, state, request, store=job.deformations)
    except looks.LookError as exc:
        raise Refused(exc.code, message=str(exc)) from exc
    return _reply(job, workspace, LOOK_TOOL, drawn, notes, request.mode, request.args())


def _reply(
    job: Job, workspace: Workspace, tool: str, drawn: list[looks.LookPicture],
    notes: list[tuple[Image.Image, PictureNote]], mode: str, arguments: dict[str, Any],
) -> Looked:
    shown, rest = drawn[:MAX_LOOK_PICTURES], drawn[MAX_LOOK_PICTURES:]
    entries = save_pictures(job, workspace, tool, shown, notes, arguments)
    return Looked(pictures=[picture.image for picture in shown], entries=entries,
                  not_shown=not_shown(rest, mode))


# --- a change tool's picture --------------------------------------------------------------


def show_result(
    job: Job, workspace: Workspace, tool: str, mode: str, sections: Sequence[str], *,
    positions_mm: Sequence[float] = (), zooms: dict[str, Sequence[float]] | None = None,
) -> Looked:
    """The picture a change tool shows of what it wrote, drawn as ``look``
    draws *mode* and saved under *tool* (so it has a number, a caption and a
    recipe, and can be zoomed).

    ``overlay``: one picture per section of *sections* (ids), each zoomed to
    its *zooms* window (``[x0, y0, x1, y1]`` fractions of the unzoomed
    picture) when given. ``positioning``: *sections* and the atlas at
    *positions_mm* along the slicing axis. At most :data:`MAX_LOOK_PICTURES`
    are shown; the rest are ``not_shown`` (with the ``look`` call that
    draws them), and a section whose picture fails is ``failed``.
    """
    state = job.state
    drawn: list[looks.LookPicture] = []
    failed: list[dict[str, Any]] = []
    windows = zooms or {}
    with collecting() as notes:
        if mode == "positioning":
            request = looks.LookRequest(mode=mode, sections=tuple(sections),
                                        positions_mm=tuple(float(v) for v in positions_mm))
            try:
                drawn = looks.look(workspace, state, request, store=job.deformations)
            except Exception as exc:  # noqa: BLE001 - the write stands; say why
                failed.append({"error": getattr(exc, "code", "RENDER_FAILED"),
                               "message": str(exc)})
        else:
            for section in sections:
                window = tuple(float(v) for v in windows.get(section) or ())
                request = looks.LookRequest(mode=mode, sections=(section,), zoom=window)
                try:
                    drawn += looks.look(workspace, state, request, store=job.deformations)
                except Exception as exc:  # noqa: BLE001 - the write stands; say why
                    failed.append({"id": section, "error": getattr(exc, "code",
                                                                   "RENDER_FAILED"),
                                   "message": str(exc)})
    shown, rest = drawn[:MAX_LOOK_PICTURES], drawn[MAX_LOOK_PICTURES:]
    entries = save_pictures(job, workspace, tool, shown, notes,
                            {"mode": mode, "sections": list(sections),
                             **({"positions_mm": [float(v) for v in positions_mm]}
                                if positions_mm else {})})
    return Looked(pictures=[picture.image for picture in shown], entries=entries,
                  not_shown=not_shown(rest, mode, verb=LOOK_TOOL), failed=failed)


# --- zoom --------------------------------------------------------------------------------


def _saved_image(job: Job, record: PictureRecord) -> Image.Image | None:
    """The picture as it was saved: from memory (a dry run's newest), else
    its JPEG in the job folder; None when neither exists."""
    held = record.image
    try:
        if isinstance(held, Image.Image):
            return held.convert("RGB")
        if isinstance(held, bytes):
            return Image.open(io.BytesIO(held)).convert("RGB")
        if record.path:
            path = Path(job.layout.resolve(record.path)) / PICTURE_FILE
            if path.is_file():
                return Image.open(path).convert("RGB")
    except (OSError, ValueError):
        return None
    return None


def _target(job: Job, picture: int | None) -> PictureRecord:
    if picture is None:
        found = job.views.latest(exclude_tool=ZOOM_TOOL)
        if found is None:
            raise Refused("NO_PICTURE", message="No picture has been shown yet; look first.")
        return found
    try:
        number = int(picture)
    except (TypeError, ValueError):
        raise Refused("BAD_ARGS", message="picture is a picture number.") from None
    found = job.views.lookup(number)
    if found is None:
        if not job.persist:
            raise Refused("PICTURE_NOT_SAVED", picture=number,
                          message=f"Picture {number} is not saved: this job keeps no pictures "
                          "beyond this process.")
        raise Refused("UNKNOWN_PICTURE", picture=number,
                      message=f"There is no picture {number}.")
    return found


def zoom(
    job: Job, workspace: Workspace, box: Sequence[float], picture: int | None = None,
) -> Zoomed:
    """*box* (``[x0, y0, x1, y1]``, pixels of picture *picture* above its
    caption band) drawn again at more detail, saved as a picture of its own
    (tool ``zoom``).

    *picture* defaults to the newest picture that is not itself a zoom. The
    box is redrawn from the picture's recipe, from the stack as it was then
    (``stale`` when it has changed since). A box on a zoom is read in the
    zoom's pixels and maps back onto the picture it came from. A picture
    that cannot be redrawn (no recipe, or its sections are gone) is cropped
    from its saved image and enlarged (``redrawn`` False); a crop's recipe
    names its ``source`` picture, so a crop of a crop is cut from the
    original. Refused: ``NO_PICTURE``, ``UNKNOWN_PICTURE``, ``BAD_ARGS``,
    ``BAD_BOX``, ``EMPTY_BOX``, ``PICTURE_NOT_SAVED`` (nothing to redraw
    from and no saved image), ``UNKNOWN_MODE`` and the other look refusals.
    """
    from langslice.ops import atlas as _atlas  # noqa: F401  (registers its recipe renderer)

    target = _target(job, picture)
    recipe = target.recipe
    renderer = (recipe or {}).get("renderer")
    crop = renderer == zooms.CROP_RENDERER
    source_number = target.seq
    saved: Image.Image | None
    if crop:
        source_number = int((recipe or {}).get("source") or 0)
        original = job.views.lookup(source_number) if source_number else None
        saved = _saved_image(job, original) if original is not None else None
        if saved is None:
            raise Refused("PICTURE_NOT_SAVED", picture=target.seq, message=(
                f"Picture {target.seq} was cropped from picture {source_number or '?'}, "
                "which is not saved."))
    else:
        saved = _saved_image(job, target)
        if saved is None and renderer not in zooms.RENDERERS:
            raise Refused("PICTURE_NOT_SAVED", picture=target.seq,
                          message=f"Picture {target.seq} is not saved and cannot be redrawn.")
    try:
        with collecting() as notes:
            made = zooms.redraw(recipe, box, workspace, state=job.state, store=job.deformations,
                                picture=saved)
    except zooms.ZoomError as exc:
        raise Refused(exc.code, message=str(exc)) from exc
    except looks.LookError as exc:
        raise Refused(exc.code, message=str(exc)) from exc
    if not made.redrawn:
        made.recipe["source"] = source_number
    entries = save_pictures(job, workspace, ZOOM_TOOL, [made], notes,
                            {"box": [float(v) for v in box], "picture": target.seq})
    return Zoomed(pictures=[made.image], entries=entries, picture=target.seq,
                  redrawn=made.redrawn, stale=made.stale)


__all__ = [
    "LOOK_TOOL", "MAX_LOOK_PICTURES", "ZOOM_TOOL", "Looked", "Zoomed", "look", "not_shown",
    "save_pictures", "show_result", "zoom",
]
