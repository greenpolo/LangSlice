"""The opening images: the stack as ABBA-style strips.

The seed message (and the Claude host's ``show_stack`` pages) shows the stack
the way ABBA's slice strip does: horizontal strips in corrected order, the
sections along the top row and, directly beneath each one, the atlas at that
section's current position at the stack's cutting angles. Until 2026-10-03
every section and every atlas section was its own image (~80 images for a
40-section stack); Nash asked for strips instead.

Each strip's long edge is the model lane's largest image
(:func:`langslice.adk.media.strip_edge`),
its tiles the host's ``image_resolution`` opening size
(:func:`langslice.linear.render.opening_edge`), so a larger level means fewer
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

from langslice.affine import resize_long_edge
from langslice.linear.appearance import view_look
from langslice.linear.atlas_fetch import atlas_section, reference_atlas
from langslice.linear.render import caption, opening_edge, render_slice
from langslice.linear.state import SliceState, StackState
from langslice.linear.workspace import Workspace

#: Longest image edge, in pixels, the OpenAI lanes take in without shrinking
#: (gpt-6-astra at detail "high"; Codex CLI's own client resize is the same
#: 2048 px). Verified 2026-10-03.
OPENAI_MAX_IMAGE_EDGE = 2048
#: The OpenAI lanes' per-image budget of 32-px patches at detail "high"
#: (ceil(w/32) * ceil(h/32)); a larger image is downscaled to fit. Verified
#: 2026-10-03: a 2048-px-wide strip may be at most ~1250 px tall.
OPENAI_MAX_IMAGE_PATCHES = 2500
#: Claude's recommended largest long edge (the MCP host, ``show_stack``):
#: older Claude models shrink anything larger, and a request with more than
#: 20 images caps every image at 2000 px, so 1568 is safe on every model.
CLAUDE_MAX_IMAGE_EDGE = 1568
#: Claude's per-image area at that edge, ~1.15 MP (~1600 tokens at w*h/750),
#: as 32-px patches so one rule serves both lanes.
CLAUDE_MAX_IMAGE_PATCHES = 1_200_000 // (32 * 32)
#: The largest picture a Claude host may ask for per call (``view.resolution``
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


def strip_layout(edge: int, tile_edge: int) -> tuple[int, int]:
    """``(tiles per strip, tile edge)``: as many tiles as fit along *edge*.

    The tile is the level's opening size, shrunk by at most a few pixels so
    that a full strip of separators and tiles fits *edge* exactly.
    """
    count = max(1, int(edge) // int(tile_edge))
    tile = min(int(tile_edge), (int(edge) - (count - 1) * COLUMN_GAP) // count)
    return count, max(16, tile)


def tile_label(record: SliceState) -> str:
    """``"<index>: <filename>"`` plus short correction flags (no damage note:
    the note is in the status table, and a long one would wrap the tile)."""
    flags = []
    if record.rotation_deg:
        flags.append(f"rot {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append("damaged")
    label = f"{record.index_corrected}: {record.id}"
    return label + (f" [{', '.join(flags)}]" if flags else "")


def section_tile(ctx: Workspace, state: StackState, record: SliceState, tile: int
                 ) -> Image.Image:
    """The section as the agent is shown it, tissue-framed, at most *tile*."""
    return render_slice(ctx, record, long_edge=tile, frame=True, look=view_look(state, record))


def atlas_tile(ctx: Workspace, state: StackState, position_mm: float, long_edge: int
               ) -> Image.Image:
    """The atlas at *position_mm* and the stack's angles, tissue-framed, drawn
    to *long_edge* (the section's above it), as ``view_stack`` draws it: the
    two read at the same size, so an olfactory-bulb plane (~200 px at 25 um)
    is not a thumbnail under a 500 px section at high."""
    return resize_long_edge(atlas_section(ctx, state, position_mm, frame=True), int(long_edge))


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


def pack_strips(columns: Sequence[Sequence[Image.Image]], tile: int, count: int,
                budget: int) -> list[tuple[int, Image.Image]]:
    """Columns into strips of at most *count*, each within *budget* patches.

    Returns ``(first column index, strip)`` per strip. A column that would
    push a strip past the budget starts the next one (a single column is
    always one strip, whatever it costs).
    """
    strips: list[tuple[int, Image.Image]] = []
    start = 0
    while start < len(columns):
        end = min(len(columns), start + count)
        strip = compose_strip(columns[start:end], tile)
        while end - start > 1 and patches(strip.size) > budget:
            end -= 1
            strip = compose_strip(columns[start:end], tile)
        strips.append((start, strip))
        start = end
    return strips


def _angles(state: StackState) -> str:
    if not state.is_oblique:
        return ""
    return f" (pitch {state.pitch_deg:.1f}, yaw {state.yaw_deg:.1f} degrees)"


def opening_items(
    state: StackState, ctx: Workspace, *, limit: tuple[int, int] = DEFAULT_IMAGE_LIMIT,
) -> list[str | Image.Image]:
    """The stack's opening as strips, each preceded by a short text: texts and
    pictures in the order they are read (a door packages them,
    :func:`langslice.adk.media.opening_parts`).

    *limit* is ``(long edge, patch budget)`` of one image, the model lane's
    (:data:`IMAGE_LIMITS`). Tiles are the run's opening size.
    """
    edge, budget = limit
    count, tile = strip_layout(edge, opening_edge(ctx))
    ordered = list(state.in_order())
    placed = any(record.position_mm is not None for record in ordered)
    unplaced = [record for record in ordered if record.position_mm is None]

    columns: list[list[Image.Image]] = []
    for record in ordered:
        section = section_tile(ctx, state, record, tile)
        column = [_cell(section, tile_label(record), tile)]
        if placed:
            position = record.position_mm
            column.append(
                _cell(atlas_tile(ctx, state, position, max(section.size)),
                      f"atlas {position:.2f} mm", tile)
                if position is not None else _cell(None, "no position", tile)
            )
        columns.append(column)
    strips = pack_strips(columns, tile, count, budget)

    beneath = (
        " Beneath each section is the atlas at that section's current position "
        f"at the stack's cutting angles{_angles(state)}, labelled 'atlas <position> mm' "
        "('no position' where it has none)."
        if placed else ""
    )
    items: list[str | Image.Image] = [(
        f"The {len(ordered)} sections of the stack follow in {len(strips)} "
        f"strip{'s' if len(strips) != 1 else ''}, in their current corrected order, "
        "left to right and strip after strip. Each section is labelled "
        "'<index>: <filename>' above it and drawn with any rotation and flip "
        f"already applied.{beneath}"
    )]
    bounds = [start for start, _strip in strips] + [len(ordered)]
    for number, (start, strip) in enumerate(strips):
        members = ordered[start:bounds[number + 1]]
        items.append(
            f"Strip {number + 1} of {len(strips)}: "
            + ", ".join(f"{record.index_corrected}: {record.id}" for record in members)
        )
        items.append(strip)

    if unplaced:
        items.extend(reference_items(state, ctx, tile=tile, count=count, budget=budget))
    return items


def reference_items(
    state: StackState, ctx: Workspace, *, tile: int, count: int, budget: int,
) -> list[str | Image.Image]:
    """The atlas reference (evenly spaced positions) as strips of atlas tiles."""
    step, pictures = reference_atlas(ctx, state, long_edge=tile)
    if not pictures:
        return []
    columns = [[_cell(picture, f"atlas {position:.2f} mm", tile)]
               for position, picture in pictures]
    strips = pack_strips(columns, tile, count, budget)
    items: list[str | Image.Image] = [(
        f"Atlas reference strip{'s' if len(strips) != 1 else ''}: the atlas at "
        f"{len(pictures)} positions, every {step:.2f} mm from {pictures[0][0]:.2f} to "
        f"{pictures[-1][0]:.2f} mm, at the stack's cutting angles{_angles(state)}, "
        f"in {len(strips)} strip{'s' if len(strips) != 1 else ''}, each atlas section "
        "labelled 'atlas <position> mm' above it."
    )]
    bounds = [start for start, _strip in strips] + [len(pictures)]
    for number, (start, strip) in enumerate(strips):
        first, last = pictures[start][0], pictures[bounds[number + 1] - 1][0]
        items.append(f"Atlas strip {number + 1} of {len(strips)}: {first:.2f} to {last:.2f} mm")
        items.append(strip)
    return items
