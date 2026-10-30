"""The ``zoom`` tool: a box of an earlier picture drawn again.

:func:`zoom_tool` is the tool; :func:`zoom_picture` draws the box on a job (a
sheet's box on the tile under it) and saves it as a picture of its own.
"""

from __future__ import annotations

import io
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from langslice.core.geometry.pixel_frames import source_frame
from langslice.core.pictures import sheet as sheets
from langslice.core.pictures import zoom as zooms
from langslice.core.pictures.drawn import collecting
from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.refusal import RefusalCode, Refused
from langslice.job.pictures import PICTURE_FILE, PictureRecord
from langslice.tools.look import save_pictures
from langslice.tools.run import as_list

if TYPE_CHECKING:
    from langslice.core.workspace import Workspace
    from langslice.job.job import Job
    from langslice.tools.run import Run

#: The tool name zoomed pictures are saved under (the picture index skips it
#: when it picks the default picture to zoom).
ZOOM_TOOL = "zoom"
#: The refusals of a box that, on a zoom, are said in the grid's numbers.
BOX_CODES = (RefusalCode.EMPTY_BOX, RefusalCode.BAD_BOX)


@dataclass(frozen=True)
class Zoomed:
    """What :func:`zoom_picture` drew."""

    pictures: list[Image.Image] = field(default_factory=list)
    entries: list[dict[str, Any]] = field(default_factory=list)
    #: The number of the picture the box was read on.
    picture: int = 0
    #: Whether the zoom was drawn again from the source (False: cropped from
    #: the saved picture, which holds no more detail).
    redrawn: bool = True
    #: The stack has changed since the source picture; it was redrawn as it was.
    stale: bool = False
    #: The sheet the box was given on (None: not a sheet); ``picture`` is then
    #: the tile's picture and ``box`` the box on it.
    sheet: int | None = None
    box: list[float] = field(default_factory=list)


# --- zoom --------------------------------------------------------------------------------


def zoom_tool(
    run: Run, box: Sequence[float], picture: int, *, region: str = ""
) -> dict[str, object]:
    """Draw a box of an earlier picture again; return the ``zoom`` reply.

    Parameters
    ----------
    run
        The run the tool is called in.
    box
        ``[x0, y0, x1, y1]`` in pixels of the picture (on a zoom, the
        numbers its grid prints).
    picture
        The picture's number; 0 is the newest picture that is not a zoom.
    region
        An atlas region whose whole bounds replace the box (an overlay).

    Returns
    -------
    dict
        The reply: the picture the box was read on, whether it was redrawn,
        the sheet and the box on its tile for a box given on a sheet,
        ``stale`` when the stack changed since, and the new picture.

    Raises
    ------
    Refused
        ``BAD_ARGS`` when ``picture`` is not a number; the refusals of
        :func:`zoom_picture`.
    """
    try:
        number = int(picture or 0) or None
        done = zoom_picture(
            run.job, run.ctx, as_list(box), picture=number, region=region
        )
    except Refused:
        raise
    except (TypeError, ValueError):
        raise Refused(
            RefusalCode.BAD_ARGS, "picture is a picture number."
        ) from None
    reply: dict[str, object] = {
        "status": "ok",
        "picture": done.picture,
        "redrawn": done.redrawn,
    }
    if done.sheet is not None:
        reply["sheet"] = done.sheet
        reply["box"] = done.box
    if done.stale:
        reply["stale"] = True
    reply["pictures"] = list(done.entries)
    reply[TOOL_MEDIA_PARTS_KEY] = list(done.pictures)
    return reply


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
            raise Refused(RefusalCode.NO_PICTURE,
                          message="No picture has been shown yet; look first.")
        return found
    try:
        number = int(picture)
    except (TypeError, ValueError):
        raise Refused(RefusalCode.BAD_ARGS,
                      message="picture is a picture number.") from None
    found = job.views.lookup(number)
    if found is None:
        if not job.persist:
            raise Refused(RefusalCode.PICTURE_NOT_SAVED, picture=number,
                          message=f"Picture {number} is not saved: this job keeps no pictures "
                          "beyond this process.")
        raise Refused(RefusalCode.UNKNOWN_PICTURE, picture=number,
                      message=f"There is no picture {number}; use a number "
                      "a reply gave.")
    return found


def _content_box(recipe: dict[str, Any] | None, image: Image.Image) -> list[int] | None:
    """``[x0, y0, x1, y1]``: the real picture inside a zoom's image, which a
    caption band below it and black padding beside a narrow box surround (the
    union of its endpoint panels, else its ``shown`` size from the corner)."""
    panels = (recipe or {}).get("endpoint_panels") or []
    if panels:
        bounds = [p["bounds"] for p in panels]
        box = [min(b[0] for b in bounds), min(b[1] for b in bounds),
               max(b[2] for b in bounds), max(b[3] for b in bounds)]
    elif (recipe or {}).get("shown"):
        box = [0, 0, *(recipe or {})["shown"]]
    else:
        return None
    return [int(round(box[0])), int(round(box[1])),
            min(int(round(box[2])), image.width), min(int(round(box[3])), image.height)]


def _covers_words(covers: Sequence[float]) -> str:
    return f"x {round(covers[0])}-{round(covers[2])}, y {round(covers[1])}-{round(covers[3])}"


def zoom_picture(
    job: Job, workspace: Workspace, box: Sequence[float] = (), picture: int | None = None,
    region: str = "") -> Zoomed:
    """*box* (``[x0, y0, x1, y1]``, pixels of picture *picture* above its
    caption band) drawn again at more detail, saved as a picture of its own
    (tool ``zoom``).

    *region* replaces *box* with one atlas region on an overlay: its whole
    bounds with a small margin and only that region's border. Other modes
    are ``BAD_REGION_PICTURE``, absent anatomy ``REGION_NOT_IN_PLANE``.
    *picture* defaults to the newest picture that is not itself a zoom. The
    box is redrawn from the picture's recipe, from the stack as it was then
    (``stale`` when it has changed since). A box on a zoom is read in the
    numbers its grid prints (the whole picture's pixels), so it maps back onto
    the picture it came from unchanged. A picture
    that cannot be redrawn (no recipe, or its sections are gone) is cropped
    from its saved image and enlarged (``redrawn`` False); a crop's recipe
    names its ``source`` picture, so a crop of a crop is cut from the
    original. Refused: ``NO_PICTURE``, ``UNKNOWN_PICTURE``, ``BAD_ARGS``,
    ``BAD_BOX``, ``EMPTY_BOX``, ``PICTURE_NOT_SAVED`` (nothing to redraw
    from and no saved image), ``UNKNOWN_MODE`` and the other look refusals.
    """
    from langslice.tools import atlas as _atlas  # noqa: F401  (registers its recipe renderer)

    target = _target(job, picture)
    if (target.recipe or {}).get("renderer") == sheets.RENDERER:
        return _zoom_sheet(job, workspace, target, box, region)
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
            raise Refused(RefusalCode.PICTURE_NOT_SAVED,
                          picture=target.seq, message=(
                f"Picture {target.seq} was cropped from picture {source_number or '?'}, "
                "which is not saved."))
    else:
        saved = _saved_image(job, target)
        if saved is None and renderer not in zooms.RENDERERS:
            raise Refused(RefusalCode.PICTURE_NOT_SAVED, picture=target.seq,
                          message=f"Picture {target.seq} is not saved and cannot be redrawn.")
    frame = source_frame(recipe)
    asked = list(box)
    if frame is not None and len(asked) == 4 and not region:
        # On a zoom, a box is in the grid's numbers: the whole picture's pixels.
        try:
            asked = [*frame.edge_to_picture(asked[:2]), *frame.edge_to_picture(asked[2:])]
        except (TypeError, ValueError):
            asked = list(box)
    try:
        with collecting() as notes:
            made = zooms.redraw(recipe, asked, workspace, state=job.state, store=job.deformations,
                                picture=saved, region=region)
    except Refused as refusal:
        if frame is None or refusal.code not in BOX_CODES:
            raise
        text = (f"{refusal.code}: the box {[float(v) for v in box]} is outside this zoom or "
                f"too thin; picture {target.seq} covers {_covers_words(frame.covers())} "
                "(the grid's numbers).")
        raise Refused(refusal.code, message=text) from refusal
    if not made.redrawn:
        made.recipe["source"] = source_number
    entries = save_pictures(job, workspace, ZOOM_TOOL, [made], notes,
                            {"box": [float(v) for v in box], "picture": target.seq,
                             **({"region": region} if region else {})})
    content = _content_box(made.recipe, made.image)
    if content is not None:
        for entry in entries:
            entry["content_box"] = content
    grid = source_frame(made.recipe)
    if grid is not None:
        for entry in entries:
            entry["covers"] = [round(v) for v in grid.covers()]
    return Zoomed(pictures=[made.image], entries=entries, picture=target.seq,
                  redrawn=made.redrawn, stale=made.stale)


def _zoom_sheet(job: Job, workspace: Workspace, target: PictureRecord, box: Sequence[float],
                region: str) -> Zoomed:
    """A zoom on a sheet: the box on the tile that holds most of it, drawn
    as a zoom of that tile's picture; the reply names the sheet."""
    recipe = target.recipe or {}
    numbers = [int(tile["picture"]) for tile in recipe.get("tiles") or []]
    listed = ", ".join(str(n) for n in numbers)
    if region:
        raise Refused(RefusalCode.BAD_REGION_PICTURE,
                      picture=target.seq, message=(
            f"Picture {target.seq} is a sheet; zoom a region on one of its tiles' "
            f"pictures ({listed}) by number."))
    try:
        asked = [float(v) for v in box]
    except (TypeError, ValueError):
        raise Refused(RefusalCode.BAD_BOX,
                      message="box is [x0, y0, x1, y1] in pixels.") from None
    if len(asked) != 4 or asked[2] <= asked[0] or asked[3] <= asked[1]:
        raise Refused(RefusalCode.BAD_BOX,
                      message=f"Picture {target.seq} is a sheet: give a box "
                      "[x0, y0, x1, y1] in its pixels over one tile.")
    held = sheets.tile_under(recipe, asked)
    if held is None:
        raise Refused(RefusalCode.EMPTY_BOX,
                      message=f"The box {asked} covers no tile of sheet "
                      f"{target.seq}; its tiles are pictures {listed}, each at its box: "
                      + "; ".join(f"{t['picture']} at {t['box']}"
                                  for t in recipe.get("tiles") or []) + ".")
    inner = sheets.to_tile(held, asked)
    tile_picture = cast("int", held["picture"])  # a sheet recipe's number
    number = int(tile_picture)
    done = zoom_picture(job, workspace, inner, picture=number)
    return replace(done, sheet=target.seq, box=inner)


__all__ = [
    "ZOOM_TOOL",
    "Zoomed",
    "zoom_picture",
    "zoom_tool",
]
