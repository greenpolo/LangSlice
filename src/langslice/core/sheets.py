"""The stack sheets: captioned section pictures, contact sheets and the spacing plot.

Split out of ``linear/render.py`` (layered refactor, phase 3d).
"""

from __future__ import annotations

from collections.abc import Callable

from PIL import Image, ImageDraw

from langslice.core.affine import resize_long_edge
from langslice.core.appearance import Look, view_look
from langslice.core.captions import _font, caption
from langslice.core.sections import render_slice
from langslice.core.sizes import opening_edge
from langslice.core.state import SliceState, StackState
from langslice.core.status import slice_flags
from langslice.core.workspace import Workspace

#: Largest long edge of the ``view_stack`` contact sheet. Each section tile is
#: drawn at the level's opening size, then shrunk until the whole sheet fits:
#: a 40-section stack at 8 columns lands near ~250 px tiles at every level,
#: while a short stack gets the full tile size. Past this a vision encoder
#: shrinks the sheet itself, captions included.
SHEET_MAX_LONG_EDGE = 2048


def stack_pictures(
    state: StackState,
    ctx: Workspace,
    *,
    long_edge: int | None = None,
    by_position: bool = False,
    under: Callable[[SliceState], Image.Image | None] | None = None,
    look: Callable[[SliceState], Look] | None = None,
) -> list[tuple[str, Image.Image]]:
    """``(label, captioned picture)`` per section, in corrected order.

    *by_position* orders by written position instead (unplaced last) and puts
    each section's position and the signed distance to the next placed one in
    its label. *under* returns a second image to paste beneath a section's
    own in the same picture (None for none), so a section and its atlas match
    travel as ONE captioned image, the atlas drawn to the section's long edge.
    The label is burned into the picture (:func:`caption`), so it survives
    any transport that drops the text next to an attachment. *long_edge*
    None is the run's opening size (:func:`opening_edge`). *look* gives each
    section's look (None: its view appearance).
    """
    long_edge = long_edge or opening_edge(ctx)
    ordered = list(state.in_order())
    if by_position:
        ordered.sort(key=lambda r: (r.position_mm is None, r.position_mm or 0.0))
    out: list[tuple[str, Image.Image]] = []
    for slot, record in enumerate(ordered):
        flags = slice_flags(record)
        label = f"{record.index_corrected}: {record.id}"
        if by_position:
            following = next(
                (r for r in ordered[slot + 1 :] if r.position_mm is not None), None
            )
            if record.position_mm is None:
                label += "  (no position)"
            else:
                label += f"  {record.position_mm:.2f} mm"
                if following is not None and following.position_mm is not None:
                    label += f" ({following.position_mm - record.position_mm:+.2f} to next)"
        if flags:
            label += f"  [{'; '.join(flags)}]"
        picture = render_slice(
            ctx, record, long_edge=long_edge, frame=True,
            look=look(record) if look is not None else view_look(state, record),
        )
        below = under(record) if under is not None else None
        if below is not None:
            picture = stacked(picture, resize_long_edge(below, max(picture.size)))
        out.append((label, caption(picture, label)))
    return out


def reference_slice_picture(
    ctx: Workspace, record: SliceState, *, long_edge: int | None = None,
    look: Look = None,
) -> Image.Image:
    """One captioned, tissue-framed section picture for the comparison tools.

    Ordering and damage annotations do not change the pixels being compared;
    the caption carries the index and flags as they stand now (the cached
    copy keeps its first caption,
    :func:`langslice.core.pictures.reference_section_picture`). Filename
    remains the stable identity. *long_edge* None is the run's opening size.
    """
    long_edge = long_edge or opening_edge(ctx)
    label = f"{record.index_corrected}: {record.id}"
    flags = slice_flags(record)
    if flags:
        label += f"  [{'; '.join(flags)}]"
    return caption(render_slice(ctx, record, long_edge=long_edge, frame=True, look=look), label)


def stack_sheet(
    state: StackState,
    ctx: Workspace,
    *,
    under: Callable[[SliceState], Image.Image | None] | None = None,
    columns: int = 8,
    look: Callable[[SliceState], Look] | None = None,
    tile_edge: int | None = None,
) -> Image.Image:
    """One contact sheet of the stack in written-position order, each
    section (over its atlas match, via *under*) captioned with its label.

    A grid, unlike the opening's strips in corrected order: this is the
    review picture of a stack the model has already read, ordered by written
    position, and one image is what keeps the whole-stack review inside the
    per-call image budget. Detail is one ``view_slices`` call away.

    Each section is drawn at *tile_edge* (None: the run's opening size); when
    the sheet would pass :data:`SHEET_MAX_LONG_EDGE` the tiles are redrawn
    smaller until it fits, so the captions stay at their own font size.
    """
    tile = int(tile_edge or opening_edge(ctx))
    sheet = grid([picture for _, picture in stack_pictures(
        state, ctx, long_edge=tile, by_position=True, under=under, look=look,
    )], columns=columns)
    for _attempt in range(4):
        if max(sheet.size) <= SHEET_MAX_LONG_EDGE or tile <= 64:
            break
        tile = max(64, int(tile * SHEET_MAX_LONG_EDGE / float(max(sheet.size))) - 4)
        sheet = grid([picture for _, picture in stack_pictures(
            state, ctx, long_edge=tile, by_position=True, under=under,
            look=look,
        )], columns=columns)
    return sheet


def grid(images: list[Image.Image], *, columns: int) -> Image.Image:
    """Tile *images* left to right, top to bottom, on black."""
    if not images:
        return Image.new("RGB", (8, 8), (0, 0, 0))
    gap = 6
    cell_w = max(image.width for image in images)
    cell_h = max(image.height for image in images)
    columns = max(1, min(columns, len(images)))
    rows = -(-len(images) // columns)
    out = Image.new(
        "RGB", (columns * cell_w + (columns - 1) * gap, rows * cell_h + (rows - 1) * gap), (0, 0, 0)
    )
    for index, image in enumerate(images):
        x = (index % columns) * (cell_w + gap)
        y = (index // columns) * (cell_h + gap)
        out.paste(image.convert("RGB"), (x, y))
    return out


def beside(left: Image.Image, right: Image.Image) -> Image.Image:
    """*left* next to *right* on black, top-aligned, a thin gap between."""
    gap = 6
    height = max(left.height, right.height)
    out = Image.new("RGB", (left.width + gap + right.width, height), (0, 0, 0))
    out.paste(left.convert("RGB"), (0, 0))
    out.paste(right.convert("RGB"), (left.width + gap, 0))
    return out


def stacked(top: Image.Image, bottom: Image.Image) -> Image.Image:
    """*top* over *bottom* on black, centred, a thin gap between."""
    gap = 6
    width = max(top.width, bottom.width)
    out = Image.new("RGB", (width, top.height + gap + bottom.height), (0, 0, 0))
    out.paste(top.convert("RGB"), ((width - top.width) // 2, 0))
    out.paste(bottom.convert("RGB"), ((width - bottom.width) // 2, top.height + gap))
    return out


def spacing_plot(state: StackState, *, size: tuple[int, int] = (768, 384)) -> Image.Image:
    """Written position against corrected index, one dot per placed section.

    Data only: axes, ticks in millimetres, a dot per section and a line
    through the placed ones in corrected order.
    """
    width, height = size
    img = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(img)
    left, right, top, bottom = 56, width - 16, 16, height - 36
    rows = list(state.in_order())
    placed = [r for r in rows if r.position_mm is not None]
    draw.rectangle((left, top, right, bottom), outline=(0, 0, 0))
    draw.text((left, bottom + 8), "corrected index", fill=(0, 0, 0), font=_font(14))
    draw.text((4, top), "mm", fill=(0, 0, 0), font=_font(14))
    if not placed:
        return img
    values = [float(r.position_mm or 0.0) for r in placed]
    lo, hi = min(values), max(values)
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    n = max(len(rows) - 1, 1)

    def at(record: SliceState) -> tuple[float, float]:
        x = left + (right - left) * record.index_corrected / n
        y = bottom - (bottom - top) * ((record.position_mm or 0.0) - lo) / (hi - lo)
        return x, y

    for tick in range(5):
        mm = lo + (hi - lo) * tick / 4
        y = bottom - (bottom - top) * tick / 4
        draw.line((left - 4, y, left, y), fill=(0, 0, 0))
        draw.text((6, y - 7), f"{mm:.1f}", fill=(0, 0, 0), font=_font(12))
    for index in range(0, len(rows), max(1, len(rows) // 8)):
        x = left + (right - left) * index / n
        draw.line((x, bottom, x, bottom + 4), fill=(0, 0, 0))
        draw.text((x - 6, bottom + 22), str(index), fill=(0, 0, 0), font=_font(12))
    points = [at(r) for r in sorted(placed, key=lambda r: r.index_corrected)]
    if len(points) > 1:
        draw.line(points, fill=(160, 160, 160), width=1)
    for record in placed:
        x, y = at(record)
        colour = (200, 0, 0) if record.damaged else (0, 0, 0)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
    return img
