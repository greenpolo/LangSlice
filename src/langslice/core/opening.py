"""The opening images: the stack as ABBA-style strips.

The seed message (and the Claude host's ``show_stack`` pages) shows the stack
the way ABBA's slice strip does: horizontal strips in corrected order, the
sections along the top row and, directly beneath each one, the atlas at that
section's current position at the stack's cutting angles: a few strips
instead of two images per section.

Each strip's long edge is the model lane's largest image
(:func:`langslice.doors.tools.media.strip_edge`),
its tiles the host's ``image_resolution`` opening size
(:func:`langslice.core.sizes.opening_edge`), so a larger level means fewer
tiles per strip and more strips. A strip also stays inside the lane's patch
budget (:data:`MAX_IMAGE_PATCHES`): past it the vision encoder would shrink
the whole strip, labels included. A stack with no position at all gets
section-only strips followed by the atlas reference (evenly spaced positions)
in the same strip format; a stack where some section lacks a position gets
both the paired strips and the reference.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageDraw

from langslice.core.atlas_fetch import reference_atlas
from langslice.core.captions import angles_label, caption
from langslice.core.scale import atlas_at, pair_scale, section_at
from langslice.core.sections import render_slice
from langslice.core.sizes import opening_edge
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace

#: Longest image edge, in pixels, the OpenAI lanes take in without shrinking
#: (at detail "high"; Codex CLI's own client resize is the same 2048 px).
OPENAI_MAX_IMAGE_EDGE = 2048
#: The OpenAI lanes' per-image budget of 32-px patches at detail "high"
#: (ceil(w/32) * ceil(h/32)); a larger image is downscaled to fit: a
#: 2048-px-wide strip may be at most ~1250 px tall.
OPENAI_MAX_IMAGE_PATCHES = 2500
#: Claude's recommended largest long edge (the MCP host, ``show_stack``):
#: older Claude models shrink anything larger, and a request with more than
#: 20 images caps every image at 2000 px, so 1568 is safe on every model.
CLAUDE_MAX_IMAGE_EDGE = 1568
#: Claude's per-image area at that edge, ~1.15 MP (~1600 tokens at w*h/750),
#: as 32-px patches so one rule serves both lanes.
CLAUDE_MAX_IMAGE_PATCHES = 1_200_000 // (32 * 32)
#: The largest picture a Claude host may ask for per call (``look``'s ``resolution``
#: at image resolution "auto"): a request holding more than 20 images, which
#: any working session does, takes none past 2000 px. Claude 4.7+ reads up
#: to ~2576 px alone; older models shrink anything past 1568 px, which costs
#: bytes, not the request.
CLAUDE_MAX_VIEW_EDGE = 2000

#: ``(long edge, patch budget)`` per canonical provider. A lane not listed
#: (Gemini, a fake test model) uses the OpenAI numbers: unmeasured there.
IMAGE_LIMITS: dict[str, tuple[int, int]] = {
    "openai-oauth": (OPENAI_MAX_IMAGE_EDGE, OPENAI_MAX_IMAGE_PATCHES),
    "openai-api": (OPENAI_MAX_IMAGE_EDGE, OPENAI_MAX_IMAGE_PATCHES),
}
DEFAULT_IMAGE_LIMIT = (OPENAI_MAX_IMAGE_EDGE, OPENAI_MAX_IMAGE_PATCHES)
CLAUDE_IMAGE_LIMIT = (CLAUDE_MAX_IMAGE_EDGE, CLAUDE_MAX_IMAGE_PATCHES)

#: The agent CLI's viewers: the coding agent that opens the picture files a
#: CLI call saves, as ``(opening strip limit, largest later picture)``,
#: the same numbers as the door that serves the same models.
#:
#: - ``claude`` (Claude Code's Read): Read's own resize rule is not
#:   published; the pictures reach the Claude API, which takes no image past
#:   2000 px once a request holds more than 20 (any working session) and
#:   shrinks past 1568 px (~1.15 MP) on models before 4.7 (2576 px on 4.7+),
#:   so the MCP door's numbers: strips at 1568 px, pictures up to 2000.
#: - ``codex`` / ``openai`` (Codex's view_image): resized to fit 2048 px and
#:   2,500 32-px patches (codex-rs ``utils/image``, detail "high", the
#:   default), the OpenAI lanes' numbers.
VIEWER_LIMITS: dict[str, tuple[tuple[int, int], int]] = {
    "claude": (CLAUDE_IMAGE_LIMIT, CLAUDE_MAX_VIEW_EDGE),
    "codex": (DEFAULT_IMAGE_LIMIT, OPENAI_MAX_IMAGE_EDGE),
    "openai": (DEFAULT_IMAGE_LIMIT, OPENAI_MAX_IMAGE_EDGE),
}
#: The viewer of a job that names none.
DEFAULT_VIEWER = "claude"

#: Space between two columns of a strip; a grey separator line runs down
#: its middle so each section reads with the atlas beneath it as one unit.
COLUMN_GAP = 6
SEPARATOR_PX = 2
SEPARATOR_COLOR = (110, 110, 110)
#: Space between a strip's section row and its atlas row.
ROW_GAP = 4


def patches(size: tuple[int, int]) -> int:
    """32-px patches an image of *size* costs: ceil(w/32) * ceil(h/32)."""
    return math.ceil(size[0] / 32) * math.ceil(size[1] / 32)


#: ``SliceState.position_source`` of a starting position the job gave a
#: section at ingest (``job.job.DEFAULT_POSITION``): not yet placed.
STARTING_POSITION = "default"


def strip_layout(edge: int, tile_edge: int) -> tuple[int, int]:
    """``(tiles per strip, tile edge)``: as many tiles as fit along *edge*.

    The tile is the level's opening size, shrunk by at most a few pixels so
    that a full strip of separators and tiles fits *edge* exactly.
    """
    count = max(1, int(edge) // int(tile_edge))
    tile = min(int(tile_edge), (int(edge) - (count - 1) * COLUMN_GAP) // count)
    return count, max(16, tile)


def tile_label(record: SliceState) -> str:
    """The section's filename plus short correction flags (no damage note:
    the note is in the status table, and a long one would wrap the tile)."""
    flags = []
    if record.rotation_deg:
        flags.append(f"rot {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append("damaged")
    return record.id + (f" [{', '.join(flags)}]" if flags else "")


def section_tile(ctx: Workspace, state: StackState, record: SliceState, tile: int
                 ) -> Image.Image:
    """The section as the agent is shown it, tissue-framed, at most *tile*."""
    return render_slice(ctx, record, long_edge=tile, frame=True)


def pair_tiles(ctx: Workspace, state: StackState, record: SliceState, position_mm: float,
               tile: int) -> tuple[Image.Image, Image.Image]:
    """``(section, atlas)``: *record* and the atlas at *position_mm* at its
    own angles, each tissue-framed, at ONE micrometres per pixel, the larger
    of the two at most *tile* (:func:`langslice.core.scale.pair_scale`): the
    section reads at its true size against the atlas, and a small plane (an
    olfactory bulb) is drawn large when its section is small too."""
    shown, working = pair_scale(ctx, state, record, position_mm, tile)
    return (section_at(ctx, record, shown, working_um=working, long_edge=tile),
            atlas_at(ctx, state, position_mm, shown, angles=record.angles))


def _cell(image: Image.Image | None, label: str, width: int) -> Image.Image:
    """*image* centred in a *width*-wide cell, *label* burned above it."""
    height = image.height if image is not None else 1
    body = Image.new("RGB", (width, height), (0, 0, 0))
    if image is not None:
        picture = image.convert("RGB")
        if picture.width > width:
            picture = picture.resize(
                (width, max(1, round(picture.height * width / picture.width))),
                Image.Resampling.LANCZOS,
            )
        body.paste(picture, ((width - picture.width) // 2, 0))
    return caption(body, label)


def compose_strip(columns: Sequence[Sequence[Image.Image]], tile: int) -> Image.Image:
    """Columns of cells (top to bottom) side by side, separators between."""
    rows = max(len(column) for column in columns)
    heights = [max((column[r].height for column in columns if len(column) > r), default=0)
               for r in range(rows)]
    width = len(columns) * tile + (len(columns) - 1) * COLUMN_GAP
    height = sum(heights) + (rows - 1) * ROW_GAP
    out = Image.new("RGB", (width, height), (0, 0, 0))
    draw = ImageDraw.Draw(out)
    for index, column in enumerate(columns):
        x = index * (tile + COLUMN_GAP)
        y = 0
        for row, cell in enumerate(column):
            out.paste(cell, (x, y))
            y += heights[row] + ROW_GAP
        if index:
            line = x - (COLUMN_GAP + SEPARATOR_PX) // 2
            draw.rectangle((line, 0, line + SEPARATOR_PX - 1, height - 1), fill=SEPARATOR_COLOR)
    return out


def jpeg_bytes(image: Image.Image) -> int:
    """The size of *image* in the doors' JPEG encoding (:mod:`langslice.core.jpeg`)."""
    from langslice.core.jpeg import encode_jpeg

    return len(encode_jpeg(image))


def pack_strips(columns: Sequence[Sequence[Image.Image]], tile: int, count: int,
                budget: int, max_bytes: int | None = None) -> list[tuple[int, Image.Image]]:
    """Columns into strips of at most *count*, each within *budget* patches
    and, with *max_bytes*, within that many bytes as the doors encode it (a
    host that caps a reply's size: the MCP door's pages).

    Returns ``(first column index, strip)`` per strip. A column that would
    push a strip past either budget starts the next one (a single column is
    always one strip, whatever it costs), so no strip is shrunk after it is
    drawn: every tile keeps the level's opening size.
    """
    def over(strip: Image.Image) -> bool:
        return patches(strip.size) > budget or (
            max_bytes is not None and jpeg_bytes(strip) > max_bytes)

    strips: list[tuple[int, Image.Image]] = []
    start = 0
    while start < len(columns):
        end = min(len(columns), start + count)
        strip = compose_strip(columns[start:end], tile)
        while end - start > 1 and over(strip):
            end -= 1
            strip = compose_strip(columns[start:end], tile)
        strips.append((start, strip))
        start = end
    return strips


ANGLES_WORDS = " (pitch {:.1f}, yaw {:.1f} degrees)"


def _angles(state: StackState) -> str:
    return angles_label(state.stack_angles, ANGLES_WORDS)


def opening_items(
    state: StackState, ctx: Workspace, *, limit: tuple[int, int] = DEFAULT_IMAGE_LIMIT,
    max_bytes: int | None = None,
) -> list[str | Image.Image]:
    """The stack's opening as strips, each preceded by a short text: texts and
    pictures in the order they are read (a door packages them,
    :func:`langslice.doors.tools.media.opening_parts`).

    *limit* is ``(long edge, patch budget)`` of one image, the model lane's
    (:data:`IMAGE_LIMITS`). Tiles are the run's opening size. *max_bytes*
    caps each strip's encoded size (:func:`pack_strips`; the MCP door's
    page budget): a strip that would pass it holds fewer sections instead
    of being shrunk.
    """
    edge, budget = limit
    count, tile = strip_layout(edge, opening_edge(ctx))
    ordered = list(state.in_order())
    placed = any(record.position_mm is not None for record in ordered)
    unplaced = [record for record in ordered if record.position_mm is None]
    starting = any(record.position_source == STARTING_POSITION for record in ordered)

    columns: list[list[Image.Image]] = []
    for record in ordered:
        position = record.position_mm
        if placed and position is not None:
            section, atlas = pair_tiles(ctx, state, record, position, tile)
            start = " start" if record.position_source == STARTING_POSITION else ""
            column = [_cell(section, tile_label(record), tile),
                      _cell(atlas, f"atlas {position:.2f} mm{start}", tile)]
        else:
            column = [_cell(section_tile(ctx, state, record, tile), tile_label(record), tile)]
            if placed:
                column.append(_cell(None, "no position", tile))
        columns.append(column)
    strips = pack_strips(columns, tile, count, budget, max_bytes)

    planes = ("at that section's own cutting angles (they differ between sections)"
              if state.mixed_angles else f"at the stack's cutting angles{_angles(state)}")
    beneath = (
        " Beneath each section is the atlas at that section's current position "
        f"{planes}, labelled 'atlas <position> mm' "
        "('no position' where it has none)."
        + (" A label ending 'start' marks an evenly spaced starting position the job "
           "gave the section: it is not yet placed." if starting else "")
        if placed else ""
    )
    items: list[str | Image.Image] = [(
        f"The {len(ordered)} sections of the stack follow in {len(strips)} "
        f"strip{'s' if len(strips) != 1 else ''}, in their current order, "
        "left to right and strip after strip. Each section is labelled "
        "by its filename above it and drawn with any rotation and flip "
        f"already applied.{beneath}"
    )]
    bounds = [start for start, _strip in strips] + [len(ordered)]
    for number, (start, strip) in enumerate(strips):
        members = ordered[start:bounds[number + 1]]
        items.append(
            f"Strip {number + 1} of {len(strips)}: "
            + ", ".join(record.id for record in members)
        )
        items.append(strip)

    if unplaced:
        items.extend(reference_items(state, ctx, tile=tile, count=count, budget=budget,
                                     max_bytes=max_bytes))
    return items


def reference_items(
    state: StackState, ctx: Workspace, *, tile: int, count: int, budget: int,
    max_bytes: int | None = None,
) -> list[str | Image.Image]:
    """The atlas reference (evenly spaced positions) as strips of atlas tiles."""
    step, pictures = reference_atlas(ctx, state, long_edge=tile)
    if not pictures:
        return []
    planes = (
        "at the median of the sections' cutting angles"
        + angles_label(state.view_angles, ANGLES_WORDS) if state.mixed_angles
        else f"at the stack's cutting angles{_angles(state)}")
    columns = [[_cell(picture, f"atlas {position:.2f} mm", tile)]
               for position, picture in pictures]
    strips = pack_strips(columns, tile, count, budget, max_bytes)
    items: list[str | Image.Image] = [(
        f"Atlas reference strip{'s' if len(strips) != 1 else ''}: the atlas at "
        f"{len(pictures)} positions, every {step:.2f} mm from {pictures[0][0]:.2f} to "
        f"{pictures[-1][0]:.2f} mm, {planes}, "
        f"in {len(strips)} strip{'s' if len(strips) != 1 else ''}, each atlas section "
        "labelled 'atlas <position> mm' above it."
    )]
    bounds = [start for start, _strip in strips] + [len(pictures)]
    for number, (start, strip) in enumerate(strips):
        first, last = pictures[start][0], pictures[bounds[number + 1] - 1][0]
        items.append(f"Atlas strip {number + 1} of {len(strips)}: {first:.2f} to {last:.2f} mm")
        items.append(strip)
    return items
