"""Rendering the stack: sections as the run sees them, plus the status table.

Every path that shows or measures a section goes through :func:`render_slice`,
so the pixels the agent judges are the pixels a fit is computed on. Corrections
are applied in one order everywhere: ROTATE first, then FLIP left-right.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from langslice.affine import (
    extract_slice_silhouette,
    physical_affine_matrix,
    resize_long_edge,
    silhouette_iou,
)
from langslice.atlas.core import get_reference_slice
from langslice.atlas.render import (
    annotation_slice,
    atlas_um_per_px,
    family_outlines,
    is_dark_background,
    outer_outline,
    region_contours,
)
from langslice.image_prep import (
    adaptive_preprocess,
    custom_appearance,
    foreground_mask,
    prepare_image_for_vlm,
    tissue_box,
)
from langslice.linear.appearance import Look, look_token, view_look
from langslice.linear.state import SliceState, StackState
from langslice.linear.workspace import Workspace
from langslice.space import Plane

#: Working frame for transform fits and their preview panels. A COMPUTE
#: size: fits, calibration and the six stored numbers are normalized against
#: this render, whatever size the pictures are shown at.
PREVIEW_LONG_EDGE = 512

#: Long edge, in pixels, of every picture the agent is SHOWN, per
#: ``JobSpec.image_resolution``: ``(opening, later)``. *Opening* is each tile
#: of the opening strips (every section, every atlas section beneath it or in
#: the atlas reference; :mod:`langslice.linear.opening`);
#: *later* is each picture a tool returns (each panel of a multi-panel
#: picture). "auto" opens at 256 and lets the agent pass ``resolution`` per
#: call (:data:`RESOLUTION_RANGE`), 512 when it does not. The only other
#: bound is the source: nothing is upsampled past the pixels it is drawn from
#: (a section's working copy, the atlas plane at its own voxel size), so a
#: small snapshot stays small. Never a working frame (:data:`PREVIEW_LONG_EDGE`)
#: and never the image model's input.
PICTURE_EDGES: dict[str, tuple[int, int]] = {
    "low": (256, 512),
    "medium": (384, 768),
    "high": (512, 1024),
    "auto": (256, 512),
}
#: The level that lets the agent choose each picture's size.
AUTO_RESOLUTION = "auto"
#: ``(smallest, largest)`` long edge the agent may ask for at "auto"; a value
#: outside is clamped into it and the reply says so.
RESOLUTION_RANGE = (128, 1536)

#: Default image/pair batch size. Separate positioning comparisons may return
#: two references per pair (up to eight images); retained context is governed
#: by the session's image-retention policy, not this per-tool batch size.
MAX_IMAGES_PER_CALL = 4

#: Largest long edge of the ``view_stack`` contact sheet. Each section tile is
#: drawn at the level's opening size, then shrunk until the whole sheet fits:
#: a 40-section stack at 8 columns lands near ~250 px tiles at every level,
#: while a short stack gets the full tile size. Past this a vision encoder
#: shrinks the sheet itself, captions included.
SHEET_MAX_LONG_EDGE = 2048

#: Font size of the label strip :func:`caption` burns into an image.
CAPTION_PX = 14

_ROTATE_OPS = {
    90: Image.Transpose.ROTATE_90,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_270,
}


def resolution_level(ctx: Workspace) -> str:
    """This run's ``image_resolution`` ("low" when the spec has none)."""
    level = str(getattr(getattr(ctx, "spec", None), "image_resolution", "low") or "low")
    return level if level in PICTURE_EDGES else "low"


def opening_edge(ctx: Workspace) -> int:
    """Long edge of each opening-strip tile at this run's level."""
    return PICTURE_EDGES[resolution_level(ctx)][0]


def picture_edge(ctx: Workspace, requested: int | None = None) -> int:
    """Long edge of each later picture: the level's, or *requested* at "auto".

    *requested* must already be clamped into :data:`RESOLUTION_RANGE`
    (:func:`langslice.linear.view_options.parse_view` does it); every level but
    "auto" ignores it.
    """
    level = resolution_level(ctx)
    if level == AUTO_RESOLUTION and requested:
        return int(requested)
    return PICTURE_EDGES[level][1]


@lru_cache(maxsize=1)
def _caption_font() -> Any:
    try:
        return ImageFont.load_default(size=CAPTION_PX)
    except TypeError:  # Pillow < 10.1 has no sized default font
        return ImageFont.load_default()


def caption(image: Image.Image, text: str) -> Image.Image:
    """A COPY of *image* with *text* burned into a dark strip, top-left.

    Tool images reach the model as bare attachments, so the text that binds an
    image to its section or its position has to ride in the pixels. Caption
    only what is SHOWN: an image a fit measures must never be captioned, since
    the strip changes the pixels the fit reads.

    Text wider than the picture wraps (:func:`wrap_caption`) instead of
    running off its right edge, so a small picture keeps its whole label.
    """
    source = image.convert("RGB")
    font = _caption_font()
    text = wrap_caption(text, font, source.width - 6)
    probe = ImageDraw.Draw(source)
    left, top, right, bottom = probe.textbbox((0, 0), text, font=font)
    band = int(bottom - top + 6)
    # The band sits ABOVE the picture, never over it: a caption drawn on the
    # pixels covered exactly the magnified dorsal tissue an agent was reading.
    labelled = Image.new("RGB", (source.width, source.height + band), (0, 0, 0))
    labelled.paste(source, (0, band))
    draw = ImageDraw.Draw(labelled)
    draw.text((3 - left, 3 - top), text, fill=(255, 255, 255), font=font)
    return labelled


def wrap_caption(text: str, font: Any, width: int) -> str:
    """*text* with every line broken to fit *width* pixels in *font*.

    Lines break at spaces; a single word wider than *width* breaks between
    characters. A line that already fits is left exactly as it was, so a
    caption that fit before draws the same pixels.
    """
    width = max(int(width), 24)

    def fits(piece: str) -> bool:
        return float(font.getlength(piece)) <= width

    out: list[str] = []
    for raw in text.split("\n"):
        if fits(raw):
            out.append(raw)
            continue
        line: str | None = None
        for word in raw.split(" "):
            candidate = word if line is None else f"{line} {word}"
            if fits(candidate):
                line = candidate
                continue
            if line is not None and line.strip():
                out.append(line.rstrip())
                line = word or None
            else:
                line = candidate.lstrip() or None
            while line is not None and not fits(line):
                cut = max(1, max((k for k in range(1, len(line) + 1) if fits(line[:k])),
                                 default=1))
                out.append(line[:cut])
                line = line[cut:] or None
        if line is not None and line.strip():
            out.append(line.rstrip())
    return "\n".join(out)


def render_cache_key(
    ctx: Workspace, record: SliceState, *, long_edge: int, frame: bool, look: Look = None,
) -> tuple[str, bool, int, int, str, bool]:
    """The key a render is cached under: the section plus everything it shows.

    The default appearance keeps ``spec.preprocess`` in the look slot, so its
    keys are the ones every earlier caller computed.
    """
    return (record.id, record.flip, record.rotation_deg, long_edge,
            look_token(ctx, look), frame)


def render_slice(
    ctx: Workspace,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    frame: bool = False,
    look: Look = None,
) -> Image.Image:
    """One section as the run sees it: normalized, framed, enhanced, corrected.

    *look* None is the DEFAULT appearance: with ``spec.preprocess == "auto"``
    the section is run through
    :func:`~langslice.image_prep.adaptive_preprocess` — per-channel CLAHE plus a
    DAPI-weighted grayscale blend — so dim fluorescence reads like the atlas
    instead of like a black field; a host's multi-channel snapshot arrives
    already blended (``spec.host_preprocessing``). Any other look
    (:mod:`langslice.linear.appearance`) is drawn from the section's raw
    channels over the SAME frame, crop and size, so geometry never depends on
    appearance. Display only: the user's file is never touched.

    *frame* crops to the tissue plus a small margin before the resize, so the
    section fills its frame about as much as a cropped atlas render does. It is
    off by default because it changes the image's coordinate frame: only the
    paths that SHOW a section to a model set it, never a fit, whose parameters
    are normalized against the render they were computed on.

    *long_edge* is a ceiling, never a target: a source smaller than it is
    returned at its own size (nothing is upsampled).

    Renders are cached on *ctx*, so the returned image is shared: read it,
    never mutate it in place.
    """
    key = render_cache_key(ctx, record, long_edge=long_edge, frame=frame, look=look)
    cached = ctx.render_cache.get(key)
    if cached is not None:
        return cached

    # The working copy, not the file: a whole-slide scan is read once, small.
    source, file_px_per_px = ctx.working_source(record.id)
    working_size = source.size
    box = tissue_box(source) if frame else None
    if box is not None:
        source = source.crop(box)
    prepped = prepare_image_for_vlm(source, max_long_edge=long_edge).image
    # How many FILE pixels one render pixel spans, before any quarter-turn:
    # the section's own micrometres per pixel times this is the canvas's.
    ctx.render_scale[key] = file_px_per_px * source.width / float(prepped.width)
    if look is not None:
        prepped = _look_image(ctx, record, look, working_size, box, prepped.size)
    elif ctx.spec.preprocess == "auto" and ctx.spec.host_preprocessing is None:
        prepped = adaptive_preprocess(prepped)
    rotate = _ROTATE_OPS.get(int(record.rotation_deg) % 360)
    if rotate is not None:
        prepped = prepped.transpose(rotate)
    if record.flip:
        prepped = prepped.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    ctx.render_cache[key] = prepped
    return prepped


def fine_detail(stretched: np.ndarray) -> float:
    """Fine structure of one stretched channel (0..1): the spread of what a
    3 px blur removes, over the pixels brighter than the background. Nuclei,
    layers and fibre edges score high; flat autofluorescence scores low."""
    tissue = stretched > 0.05
    if not tissue.any():
        return 0.0
    fine = stretched - cv2.GaussianBlur(stretched, (0, 0), 3.0)
    return float(fine[tissue].std())


def _look_image(
    ctx: Workspace,
    record: SliceState,
    look: dict[str, Any],
    working_size: tuple[int, int],
    box: tuple[int, int, int, int] | None,
    size: tuple[int, int],
) -> Image.Image:
    """*look* drawn from the raw channels, cropped and sized like the default."""
    names, planes = ctx.section_channels(record.id)

    def at_working(plane: np.ndarray) -> Image.Image:
        image = Image.fromarray(plane)
        if image.size != working_size:
            image = image.resize(working_size, Image.Resampling.LANCZOS)
        return image

    def framed(image: Image.Image) -> Image.Image:
        if box is not None:
            image = image.crop(box)
        if image.size != size:
            image = image.resize(size, Image.Resampling.LANCZOS)
        return image

    def placed(plane: np.ndarray) -> np.ndarray:
        return np.asarray(framed(at_working(plane)), dtype=np.uint8)

    def named(name: str) -> np.ndarray:
        if name not in names:
            raise ValueError(f"{record.id} has no channel {name!r}; channels: {', '.join(names)}")
        return planes[names.index(name)]

    if "channel" in look:
        plane = placed(named(str(look["channel"])))
        return Image.fromarray(np.stack([plane, plane, plane], axis=-1))
    if "overlay" in look:
        # Each channel stretched on its WHOLE working plane (so a framed and
        # an unframed picture share one stretch), then added in its colour.
        from langslice.linear.appearance import OVERLAY_STRETCH, channel_colors

        total = np.zeros((size[1], size[0], 3), dtype=np.float32)
        names_shown = list(look["overlay"])
        # One channel is gray; several are each added in their colour.
        colors = ([(names_shown[0], "gray", (255, 255, 255))] if len(names_shown) == 1
                  else channel_colors(names_shown))
        stretched_planes: list[np.ndarray] = []
        detail: list[float] = []
        for name, _word, _rgb in colors:
            whole = np.asarray(at_working(named(name)), dtype=np.float32)
            low, high = (float(v) for v in np.percentile(whole, OVERLAY_STRETCH))
            if high <= low:
                high = low + 1.0
            stretched = np.clip((whole - low) / (high - low), 0.0, 1.0)
            stretched_planes.append(stretched)
            detail.append(fine_detail(stretched))
        # Several channels: each is dimmed by its fine detail relative to the
        # most detailed one, so a flat autofluorescence channel (stretched to
        # full brightness on its own) cannot wash out the stain under it.
        top = max(detail) if len(colors) > 1 else 0.0
        for (_name, _word, rgb), stretched, amount in zip(
            colors, stretched_planes, detail, strict=True
        ):
            gain = amount / top if top > 0 else 1.0
            shown = np.asarray(framed(Image.fromarray((stretched * 255.0).astype(np.uint8))),
                               dtype=np.float32) / 255.0
            total += gain * shown[..., None] * np.asarray(rgb, dtype=np.float32)
        return Image.fromarray(np.clip(total, 0.0, 255.0).astype(np.uint8), mode="RGB")
    return custom_appearance(
        [placed(plane) for plane in planes],
        channel_weights=look.get("channel_weights"),
        clahe_clip=float(look.get("clahe_clip", 4.0)),
        clahe_tiles=int(look.get("clahe_tiles", 8)),
        n4=bool(look.get("n4")),
        denoise=bool(look.get("denoise")),
    )


def canvas_um_per_px(
    ctx: Workspace,
    record: SliceState,
    *,
    long_edge: int = PREVIEW_LONG_EDGE,
    frame: bool = False,
) -> tuple[float | None, str]:
    """``(micrometres per pixel of the RENDER, source)`` for one section.

    The render is a downsample of the file, so the file's pixel size times
    the downsample factor is the working canvas's. ``(None, "")`` when
    neither the file nor the host supplies one.
    """
    from_file, source = ctx.calibration(record.id)
    if from_file is None:
        return None, source
    key = render_cache_key(ctx, record, long_edge=long_edge, frame=frame)
    if key not in ctx.render_scale:
        render_slice(ctx, record, long_edge=long_edge, frame=frame)
    return from_file * ctx.render_scale.get(key, 1.0), source


def shown_section(
    ctx: Workspace, record: SliceState, section: Image.Image, um_per_px: float,
    look: Look = None, *, long_edge: int = PREVIEW_LONG_EDGE,
) -> tuple[Image.Image, float, tuple[float, float]]:
    """The render a PICTURE of *section* is drawn from, *long_edge* at most.

    *section* is the :data:`PREVIEW_LONG_EDGE` working frame every fit and
    every written transform is computed on, and *um_per_px* its calibration.
    At the working frame's own size with the default *look* this returns them
    unchanged with factors ``(1.0, 1.0)``; otherwise a render of the same
    section in *look* at *long_edge* (never upsampled past the working copy),
    its micrometres per pixel, and the ``(fx, fy)`` that carry working-frame
    pixels onto it. Nothing computed is drawn from here.
    """
    if long_edge == PREVIEW_LONG_EDGE and look is None:
        return section, um_per_px, (1.0, 1.0)
    shown = render_slice(ctx, record, long_edge=int(long_edge), look=look)
    fx = shown.width / float(section.width)
    fy = shown.height / float(section.height)
    return shown, um_per_px / fx, (fx, fy)


def rescale_section_matrix(matrix: Any, fx: float, fy: float) -> np.ndarray:
    """A 2x3 on a section frame, re-expressed on the same frame scaled by (fx, fy)."""
    square = np.vstack([np.asarray(matrix, dtype=np.float64).reshape(2, 3), [0.0, 0.0, 1.0]])
    scale = np.diag([fx, fy, 1.0])
    return (scale @ square @ np.diag([1.0 / fx, 1.0 / fy, 1.0]))[:2]


# --- the status table ----------------------------------------------------


def status_rows(state: StackState) -> list[dict[str, Any]]:
    """One row per section in corrected order. The ``ls`` of the environment.

    ``delta_to_next_mm`` is the SIGNED distance to the next section in
    corrected order that carries a position, and null when this section has
    none or no placed section follows it. ``transform_iou`` and
    ``transform_mirrored`` come off the recorded transform. Data only: no
    comparison against the nominal interval, no verdict.
    """
    ordered = state.in_order()
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(ordered):
        here = record.position_mm
        delta: float | None = None
        if here is not None:
            following = next(
                (
                    value
                    for r in ordered[index + 1 :]
                    if (value := r.position_mm) is not None
                ),
                None,
            )
            if following is not None:
                delta = round(following - here, 3)
        transform = record.transform or {}
        rows.append(
            {
                "index": record.index_corrected,
                "id": record.id,
                "position_mm": round(here, 3) if here is not None else None,
                "delta_to_next_mm": delta,
                "flip": record.flip,
                "rotation_deg": record.rotation_deg,
                "damaged": record.damaged,
                "damage_note": record.damage_note,
                "transform": transform.get("kind"),
                **({"transform_model": (
                    "elastix_bspline" if transform["spline"].get("backend") == "elastix"
                    else "thin_plate_spline"),
                    "landmarks": len(transform["spline"]["source"])}
                   if transform.get("spline") else {}),
                "transform_iou": transform.get("iou"),
                "transform_mirrored": transform.get("mirrored"),
                **({"keep_linear": record.deformation["keep_linear"]}
                   if record.deformation and "keep_linear" in record.deformation
                   else {"deformation_steps": len(record.deformation.get("steps") or [])}
                   if record.deformation else {}),
                "caveats": list(record.caveats),
            }
        )
    return rows


def compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows without their null and empty fields, for a tool payload.

    A position-only run carried null transform fields and empty caveat lists
    on every row of every result, a third of the text the model paid for
    (run 5, 2026-09-09). Absent means null; ``status_text`` keeps the full
    rows.
    """
    return [
        {k: v for k, v in row.items() if v is not None and v != [] and v != ""}
        for row in rows
    ]


def status_text(state: StackState) -> str:
    """The status rows as one line per section, for a text message."""
    lines = [
        "index  id  position_mm  delta_to_next_mm  flags  "
        "transform(kind, iou, mirrored)"
    ]
    for row in status_rows(state):
        flags: list[str] = []
        if row["flip"]:
            flags.append("flipped")
        if row["rotation_deg"]:
            flags.append(f"rotated {row['rotation_deg']}")
        if row["damaged"]:
            note = row["damage_note"]
            flags.append(f"damaged: {note}" if note else "damaged")
        flags.extend(row["caveats"])
        position = (
            "unplaced" if row["position_mm"] is None else f"{row['position_mm']:.3f} mm"
        )
        delta = (
            "-" if row["delta_to_next_mm"] is None else f"{row['delta_to_next_mm']:.3f}"
        )
        transform = ""
        if row["transform"]:
            transform = f"  transform={row['transform']}"
            if row["transform_iou"] is not None:
                transform += f" iou={float(row['transform_iou']):.3f}"
            if row["transform_mirrored"] is not None:
                transform += f" mirrored={bool(row['transform_mirrored'])}"
        if row.get("deformation_steps"):
            transform += f"  deformation={row['deformation_steps']} step(s)"
        lines.append(
            f"{row['index']:>3}  {row['id']}  {position}  {delta}"
            + (f"  [{'; '.join(flags)}]" if flags else "")
            + transform
        )
    return "\n".join(lines)


def slice_flags(record: SliceState) -> list[str]:
    """The section's current corrections, as short human-readable flags."""
    flags: list[str] = []
    if record.rotation_deg:
        flags.append(f"rotated {record.rotation_deg}")
    if record.flip:
        flags.append("flipped")
    if record.damaged:
        flags.append(
            f"damaged: {record.damage_note}" if record.damage_note else "damaged"
        )
    return flags


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
    the caption carries the index and flags as they stand now (the encoded
    copy the doors cache keeps its first caption,
    :func:`langslice.adk.media.reference_slice_part`). Filename remains the
    stable identity. *long_edge* None is the run's opening size.
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


def _font(px: int) -> Any:
    try:
        return ImageFont.load_default(size=px)
    except TypeError:
        return ImageFont.load_default()


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


# --- the physical overlay ------------------------------------------------



#: Black working space on each side of the larger of section and atlas, as a
#: fraction of that extent. Room for x/y moves, like ABBA's viewer.
WORKING_MARGIN = 0.12

@dataclass(frozen=True)
class CanvasGeometry:
    """Where the section and the atlas sit on one millimetre-true canvas.

    The canvas IS the section's frame, grown symmetrically when the atlas
    anatomy at true scale would not fit inside it. Both frames therefore share
    a centre, which is why a transform's rotation centre is the same in either
    (:func:`langslice.affine.normalized_physical_affine`).
    """

    size: tuple[int, int]
    um_per_px: float
    #: Canvas coordinates of the section's top-left corner.
    section_offset: tuple[int, int]
    #: Atlas pixels -> canvas pixels.
    atlas_scale: float
    #: Canvas coordinates of the atlas frame's (0, 0).
    atlas_offset: tuple[float, float]
    annotation: np.ndarray


def canvas_geometry(
    section_size: tuple[int, int],
    section_um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    *,
    pad_to_fit_atlas: bool = True,
    margin: float = WORKING_MARGIN,
) -> CanvasGeometry:
    """Place an atlas section on a section's frame at TRUE physical scale.

    The atlas is scaled by ``atlas um/px / canvas um/px`` — never fitted to
    the canvas, which is not a calibration: measured on LSD_910 M01,
    fit-to-canvas put the atlas plate at 0.69x the tissue where true scale
    puts it at 1.05x. Its ANATOMY (the annotation's bounding box, not the
    atlas frame with its empty margins) is centred on the canvas centre, and
    with *pad_to_fit_atlas* the canvas grows to hold whichever of the section
    and the atlas anatomy is larger, plus *margin* of black on every side —
    ABBA's viewer leaves generous black space around both, and the alignment
    loop needs it: the atlas runs out of frame otherwise and every x/y move
    has to fit inside the tissue's own bounding box.
    """
    ann = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    scale = atlas_um_per_px(atlas) / float(section_um_per_px)
    ys, xs = np.nonzero(ann)
    if ys.size:
        cx = float(xs.min() + xs.max() + 1) / 2.0
        cy = float(ys.min() + ys.max() + 1) / 2.0
        extent = (float(xs.max() - xs.min() + 1), float(ys.max() - ys.min() + 1))
    else:
        cx, cy = ann.shape[1] / 2.0, ann.shape[0] / 2.0
        extent = (float(ann.shape[1]), float(ann.shape[0]))

    width, height = section_size
    if pad_to_fit_atlas:
        body_w = max(width, extent[0] * scale)
        body_h = max(height, extent[1] * scale)
        width = int(np.ceil(body_w * (1.0 + 2.0 * margin)))
        height = int(np.ceil(body_h * (1.0 + 2.0 * margin)))
    return CanvasGeometry(
        size=(width, height),
        um_per_px=float(section_um_per_px),
        section_offset=((width - section_size[0]) // 2, (height - section_size[1]) // 2),
        atlas_scale=scale,
        atlas_offset=(width / 2.0 - cx * scale, height / 2.0 - cy * scale),
        annotation=ann,
    )


#: How the ONE alignment screen may be composed. ``overlay`` is the default and
#: what every earlier run saw.
VIEW_MODES = ("overlay", "side_by_side", "checkerboard", "outlines", "section", "template")

#: Which atlas lines a view draws: every family boundary, the root silhouette
#: alone, or none at all.
OUTLINE_LAYERS = ("all", "outer", "none")

#: Tiles across the width of a ``checkerboard`` view.
CHECKER_TILES = 8

#: Landmark glyph colors (RGB): the section's point, its atlas point, the
#: connector between them. Colored on purpose — a landmark is a click, not
#: anatomy, and the atlas hairlines keep their one neutral grey.
MARKER_SECTION = (80, 220, 255)
MARKER_ATLAS = (255, 170, 60)
MARKER_CONNECTOR = (170, 170, 170)


def _background_color(section: Image.Image) -> tuple[int, int, int]:
    """The section's own border color, so padding does not read as anatomy."""
    arr = np.asarray(section.convert("RGB"), dtype=np.uint8)
    border = np.concatenate([arr[0], arr[-1], arr[:, 0], arr[:, -1]])
    r, g, b = (int(v) for v in np.median(border, axis=0))
    return (r, g, b)


def _shift(offset: tuple[float, float]) -> np.ndarray:
    return np.array([[1.0, 0.0, offset[0]], [0.0, 1.0, offset[1]], [0.0, 0.0, 1.0]])


def _as_3x3(matrix: np.ndarray) -> np.ndarray:
    return np.vstack([np.asarray(matrix, dtype=np.float64), [0.0, 0.0, 1.0]])


def scale_bar_px(um_per_px: float, mm: float = 1.0) -> int:
    """Length in pixels of a *mm*-millimetre bar on a *um_per_px* canvas."""
    return int(round(mm * 1000.0 / float(um_per_px)))


def _draw_scale_bar(canvas: np.ndarray, um_per_px: float, dark: bool) -> None:
    """A 1 mm bar in the bottom-left corner, labelled. Thin, like a viewer's."""
    height, width = canvas.shape[:2]
    length = scale_bar_px(um_per_px)
    if length < 4 or length > width:
        return
    color = (235, 235, 235) if dark else (40, 40, 40)
    thickness = max(1, round(min(height, width) / 400))
    x0, y0 = 12, height - 14
    cv2.rectangle(canvas, (x0, y0), (x0 + length, y0 + thickness), color, -1)
    cv2.putText(
        canvas, "1 mm", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA
    )


def normalize_border_style(
    color: str = "yellow", thickness: float = 0.5,
) -> tuple[tuple[int, int, int], float]:
    """Validate display-only atlas borders; width is in output-image pixels."""
    if not isinstance(color, str):
        raise ValueError("border_color must be a named color or #RRGGBB")
    value = color.strip().lower()
    if not re.fullmatch(r"[a-z]+|#[0-9a-f]{6}", value):
        raise ValueError("border_color must be a named color or #RRGGBB")
    try:
        rgb = ImageColor.getrgb(value)
    except ValueError:
        raise ValueError("border_color must be a named color or #RRGGBB") from None
    if len(rgb) != 3:
        raise ValueError("border_color must be an opaque RGB color")
    if (
        isinstance(thickness, bool) or not isinstance(thickness, (int, float))
        or not math.isfinite(thickness) or not 0.25 <= thickness <= 8
    ):
        raise ValueError("border_thickness must be a number from 0.25 to 8 output pixels")
    return (rgb[0], rgb[1], rgb[2]), float(thickness)


def _tissue_color(dark: bool) -> tuple[int, int, int]:
    """Neutral grey for the section's silhouette, distinct from atlas borders."""
    return (150, 150, 150) if dark else (140, 140, 140)


def _draw_polys(
    canvas: np.ndarray,
    polys: list[np.ndarray],
    color: tuple[int, int, int],
    *,
    thickness: float = 1,
    scale: float = 1.0,
    offset: tuple[float, float] = (0.0, 0.0),
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
    alpha: float = 1.0,
) -> None:
    """Closed x/y polylines with anti-aliased thickness at OUTPUT resolution.

    Each point runs through the same chain the pixels did: its own frame ->
    canvas (*scale*, *offset*), minus the zoom crop's *origin*, times *factor*,
    the canvas-px -> output-px ratio. Sub-pixel via OpenCV's 4-bit shift.
    *alpha* below 1 draws the lines faint (context under highlighted regions).
    """
    ox, oy = float(origin[0]), float(origin[1])
    # Supersample all stroke widths consistently, keeping tissue at its
    # original resolution and compositing each border pixel only once.
    supersample = 8
    target = np.zeros(
        (canvas.shape[0] * supersample, canvas.shape[1] * supersample), dtype=np.uint8,
    )
    for poly in polys:
        points = np.round(
            ((poly * scale + np.asarray(offset, dtype=np.float64)) - (ox, oy))
            * factor * supersample * 16.0
        ).astype(np.int32)
        cv2.polylines(
            target, [points], True, 255,
            max(1, round(thickness * supersample)), cv2.LINE_AA, 4,
        )
    coverage = cv2.resize(
        target, (canvas.shape[1], canvas.shape[0]), interpolation=cv2.INTER_AREA,
    ).astype(np.float32)[..., None] / 255.0 * float(alpha)
    blended = canvas * (1.0 - coverage) + np.asarray(color) * coverage
    canvas[:] = np.rint(blended).astype(np.uint8)


def _draw_outlines(
    canvas: np.ndarray,
    outlines: list[tuple[tuple[int, int, int], np.ndarray]],
    geometry: CanvasGeometry,
    *,
    color: tuple[int, int, int] = (255, 255, 0),
    thickness: float = 1,
    factor: float = 1.0,
    origin: tuple[int, int] = (0, 0),
    alpha: float = 1.0,
) -> None:
    """One atlas-border color, with width set after crop and display resizing."""
    _draw_polys(
        canvas,
        [poly for _color, poly in outlines],
        color,
        thickness=thickness,
        scale=geometry.atlas_scale,
        offset=geometry.atlas_offset,
        origin=origin,
        factor=factor,
        alpha=alpha,
    )


#: Strength of the context outlines drawn under highlighted regions.
REGION_CONTEXT_ALPHA = 0.35


def regions_left(
    atlas: Any, regions: Any, position_mm: float, plane: str, pitch_deg: float,
    yaw_deg: float, native_to_display: np.ndarray,
) -> np.ndarray | None:
    """Native pixels on the picture's left when a region names a side, else None.

    *regions* is ``[(name, ids)]``; *native_to_display* the linear map from the
    native atlas plane to the frame whose left and right the sides name
    (:func:`langslice.atlas.sides.native_left`).
    """
    from langslice.atlas.sides import has_sides, native_left

    if not has_sides([name for name, _ids in regions]):
        return None
    return native_left(atlas, position_mm, plane, pitch_deg, yaw_deg, native_to_display)


def region_polys(
    annotation: np.ndarray, regions: Any, left: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Smoothed outlines of each highlighted region, in atlas-native pixels.

    *regions* is ``[(name, ids)]``; each region is traced as the union of its
    ids (the region and its descendants), with the same tracer and smoothing
    as the family outlines. A name with a side (``"CTX:left"``) keeps only
    that side, *left* being the native pixels on the left (:func:`regions_left`).
    """
    from langslice.atlas.sides import restrict, split_side

    polys: list[np.ndarray] = []
    for name, ids in regions:
        mask = restrict(np.isin(annotation, list(ids)), split_side(name)[1], left)
        if mask.any():
            polys.extend(region_contours(mask.astype(np.int32)).get(1, []))
    return polys


def _patch_window(
    shape: tuple[int, int], patch: np.ndarray, offset: tuple[float, float]
) -> tuple[int, int, np.ndarray]:
    """``(y, x, clipped patch)`` for pasting *patch* at *offset* on *shape*.

    Clips both ways: the canvas grows to hold the ANATOMY, so the atlas
    frame's empty margins may still hang over any edge. An empty patch comes
    back when nothing lands.
    """
    x0, y0 = (int(round(v)) for v in offset)
    sx0, sy0 = max(0, -x0), max(0, -y0)
    x0, y0 = max(0, x0), max(0, y0)
    height, width = shape
    return y0, x0, patch[sy0 : sy0 + height - y0, sx0 : sx0 + width - x0]


def _template_patch(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    geometry: CanvasGeometry,
    picture: Image.Image | None = None,
) -> tuple[int, int, np.ndarray]:
    """The atlas image resized to the canvas's scale, ready to paste.

    *picture* is the atlas image on the native plane grid; None is the atlas
    reference template.
    """
    template = picture if picture is not None else get_reference_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    rows, cols = geometry.annotation.shape[:2]
    if template.size != (cols, rows):
        template = template.resize((cols, rows), Image.Resampling.BILINEAR)
    scale = geometry.atlas_scale
    resized = template.convert("RGB").resize(
        (max(1, round(cols * scale)), max(1, round(rows * scale))),
        Image.Resampling.LANCZOS,
    )
    width, height = geometry.size
    return _patch_window(
        (height, width), np.asarray(resized, dtype=np.uint8), geometry.atlas_offset
    )


def _blend_template(
    canvas: np.ndarray,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    geometry: CanvasGeometry,
    opacity: float = 0.35,
    picture: Image.Image | None = None,
) -> None:
    """Blend the atlas image under the outlines, at the same placement."""
    y0, x0, patch = _template_patch(
        atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, picture
    )
    if patch.size == 0:
        return
    window = canvas[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]]
    lit = patch.max(axis=2) > 0
    window[lit] = (
        window[lit] * (1.0 - opacity) + patch[lit] * opacity
    ).astype(np.uint8)


def template_canvas(
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    geometry: CanvasGeometry,
) -> np.ndarray:
    """The atlas template alone on a black canvas, RGB uint8, at the placement."""
    plate = np.zeros((geometry.size[1], geometry.size[0], 3), dtype=np.uint8)
    _blend_template(plate, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, opacity=1.0)
    return plate


def atlas_mask_canvas(geometry: CanvasGeometry) -> np.ndarray:
    """The atlas anatomy's silhouette on the canvas, as a uint8 0/255 mask.

    The same placement the outlines are drawn at, so an overlap measured
    against it is measured against the lines the model is looking at.
    """
    scale = geometry.atlas_scale
    rows, cols = geometry.annotation.shape[:2]
    resized = cv2.resize(
        (geometry.annotation > 0).astype(np.uint8) * 255,
        (max(1, round(cols * scale)), max(1, round(rows * scale))),
        interpolation=cv2.INTER_NEAREST,
    )
    width, height = geometry.size
    mask = np.zeros((height, width), dtype=np.uint8)
    y0, x0, patch = _patch_window((height, width), resized, geometry.atlas_offset)
    if patch.size:
        mask[y0 : y0 + patch.shape[0], x0 : x0 + patch.shape[1]] = patch
    return mask


def zoom_box(zoom: list[float] | None, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """``[x0, y0, x1, y1]`` fractions of the canvas as a pixel crop box.

    Anything that is not four numbers — an empty list, the default — is the
    whole canvas. The box is ordered, clamped to the canvas and never allowed
    to collapse below 8 px, so a mistyped fraction costs magnification, not a
    crash.
    """
    width, height = size
    if not zoom or len(zoom) != 4:
        return (0, 0, width, height)
    x0, x1 = sorted((float(zoom[0]), float(zoom[2])))
    y0, y1 = sorted((float(zoom[1]), float(zoom[3])))
    bx0 = int(np.clip(round(x0 * width), 0, width - 8))
    by0 = int(np.clip(round(y0 * height), 0, height - 8))
    bx1 = int(np.clip(round(x1 * width), bx0 + 8, width))
    by1 = int(np.clip(round(y1 * height), by0 + 8, height))
    return (bx0, by0, bx1, by1)


def _to_screen(
    canvas: np.ndarray, box: tuple[int, int, int, int], long_edge: int | None
) -> tuple[np.ndarray, float]:
    """Crop to *box* then resize, returning ``(rgb, canvas px -> output px)``.

    Cropping BEFORE the resize is what makes a zoom real magnification: the
    same output budget spent on fewer canvas pixels. Lines and the bar
    are drawn after this, at output resolution.
    """
    x0, y0, x1, y1 = box
    cropped = canvas[y0:y1, x0:x1]
    image = Image.fromarray(cropped, mode="RGB")
    factor = 1.0
    if long_edge is not None:
        image = resize_long_edge(image, long_edge)
        factor = image.width / float(cropped.shape[1])
    return np.asarray(image, dtype=np.uint8).copy(), factor


def _checkerboard(a: np.ndarray, b: np.ndarray, tiles: int = CHECKER_TILES) -> np.ndarray:
    """*a* and *b* in alternating tiles, *tiles* of them across the width."""
    height, width = a.shape[:2]
    size = max(1, int(np.ceil(width / float(tiles))))
    ys, xs = np.mgrid[0:height, 0:width]
    even = ((xs // size + ys // size) % 2 == 0)[..., None]
    return np.where(even, a, b)


def pivot_on_canvas(
    pivot: Any, section: Image.Image, geometry: CanvasGeometry
) -> tuple[float, float] | None:
    """A pivot spec as CANVAS pixels, or ``None`` for the canvas centre.

    ``"canvas"`` (or empty) is the centre, ``"tissue"`` the section's own
    tissue centroid placed on the canvas, and ``[fx, fy]`` fractions of the
    canvas. A section whose tissue cannot be detected falls back to the
    centre; the caller reports the pivot it actually got.

    Raises ``ValueError`` on anything else.
    """
    if pivot is None or (isinstance(pivot, str) and pivot.strip().lower() in ("", "canvas")):
        return None
    width, height = geometry.size
    if isinstance(pivot, str):
        if pivot.strip().lower() != "tissue":
            raise ValueError(f"pivot must be 'canvas', 'tissue' or [fx, fy]; got {pivot!r}")
        mask = foreground_mask(section)
        if mask is None or not mask.any():
            return None
        ys, xs = np.nonzero(mask)
        # The mask is measured on a proxy; scale its centroid onto the section,
        # then place the section on the canvas.
        return (
            float(xs.mean()) * section.width / mask.shape[1] + geometry.section_offset[0],
            float(ys.mean()) * section.height / mask.shape[0] + geometry.section_offset[1],
        )
    values = [float(v) for v in pivot]
    if len(values) != 2:
        raise ValueError("pivot fractions must be [fx, fy] of the canvas")
    return (values[0] * width, values[1] * height)


def _draw_markers(
    canvas: np.ndarray,
    section_pts: np.ndarray,
    atlas_pts: np.ndarray,
    *,
    origin: tuple[int, int] = (0, 0),
    factor: float = 1.0,
) -> None:
    """Landmark pairs: a cross on each section point, a ring on its atlas point.

    Two glyphs and two colors rather than two greys — these are the model's
    own clicks, not anatomy, and they have to be findable against both the
    tissue and the hairlines. A connector runs between the pair, and the pair's
    1-based index is written by the cross so a glyph can be tied to the row in
    the payload.
    """
    def screen(points: np.ndarray) -> np.ndarray:
        return np.round(
            (np.asarray(points, dtype=np.float64) - np.asarray(origin, dtype=np.float64))
            * factor
        ).astype(np.int32)

    # Glyphs are drawn thicker than the atlas hairline on purpose: at the same
    # weight they read as another contour instead of as a marker.
    radius = max(6, round(min(canvas.shape[:2]) / 55))
    weight = max(1, round(min(canvas.shape[:2]) / 350))
    for index, (start, end) in enumerate(
        zip(screen(section_pts), screen(atlas_pts), strict=True), start=1
    ):
        sx, sy = int(start[0]), int(start[1])
        ax, ay = int(end[0]), int(end[1])
        cv2.line(canvas, (sx, sy), (ax, ay), MARKER_CONNECTOR, 1, cv2.LINE_AA)
        cv2.line(
            canvas, (sx - radius, sy), (sx + radius, sy), MARKER_SECTION, weight, cv2.LINE_AA
        )
        cv2.line(
            canvas, (sx, sy - radius), (sx, sy + radius), MARKER_SECTION, weight, cv2.LINE_AA
        )
        cv2.circle(canvas, (ax, ay), radius, MARKER_ATLAS, weight, cv2.LINE_AA)
        cv2.putText(
            canvas,
            str(index),
            (sx + radius + 3, sy - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            MARKER_SECTION,
            weight,
            cv2.LINE_AA,
        )


def _silhouette_polys(mask: np.ndarray) -> list[np.ndarray]:
    """Closed contours of the section's own tissue silhouette, in canvas px.

    Traced by the same smoothed tracer the atlas lines come from: a raw
    findContours boundary on speckled fluorescence is a stipple, and a
    stippled line next to a smooth one reads as texture, not as an edge.
    """
    return region_contours(
        (mask > 0).astype(np.int32), smooth_window=9, min_area_px=64.0
    ).get(1, [])


def physical_views(
    section: Image.Image,
    section_um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    params: dict[str, float] | np.ndarray,
    *,
    mode: str = "overlay",
    zoom: list[float] | None = None,
    atlas_opacity: float = 0.0,
    border_color: str = "yellow",
    border_thickness: float = 0.5,
    outlines: str = "all",
    pad_to_fit_atlas: bool = True,
    pivot: tuple[float, float] | None = None,
    markers: tuple[np.ndarray, np.ndarray] | None = None,
    label: str = "",
    long_edge: int | None = None,
    spline: dict[str, Any] | None = None,
    frames: list[dict[str, Any]] | None = None,
    pivot_in_section: tuple[float, float] | None = None,
    atlas_picture: Image.Image | None = None,
    atlas_name: str = "template",
    regions: Any = (),
    matrix_label: str = "fitted matrix",
    template_lines: bool = False,
) -> tuple[list[Image.Image], float]:
    """The alignment screen in one of :data:`VIEW_MODES`, plus the overlap.

    One composition, several ways of showing it. Every view is built on the
    SAME millimetre-true canvas (:func:`canvas_geometry`) and the same crop, so
    a number read off one is the number on the others:

    * ``overlay`` — the warped section with the atlas family outlines on it,
      and the atlas image blended under them at *atlas_opacity*.
    * ``side_by_side`` — two images, the warped section and the atlas template,
      at the same micrometres per pixel and the same crop, outlines on both.
    * ``checkerboard`` — section and template in alternating tiles.
    * ``outlines`` — the atlas lines and the section's own silhouette contour
      in a second grey, on black. No pixels.

    *zoom* is ``[x0, y0, x1, y1]`` in fractions of the CANVAS; the crop happens
    before the screen is sized, so it is real magnification up to the canvas's
    own pixels: each panel's long edge is *long_edge*, or the crop's when that
    is smaller (never upsampled; ``None`` is canvas pixels one to one). A
    caller wanting a magnified zoom draws the canvas from a larger render.
    The scale bar is redrawn for the magnified micrometres per pixel.

    *outlines* picks which atlas lines are drawn (:data:`OUTLINE_LAYERS`):
    every family boundary, the root silhouette alone, or none. *border_color*
    is a named color or #RRGGBB (default yellow); *border_thickness* is 0.25..8
    output-image pixels (default 0.5), unchanged by zoom or canvas resolution.

    *pivot* is the rotation/scale centre in CANVAS pixels (``None`` is the
    canvas centre), and *markers* is ``(section points, atlas points)`` in
    canvas pixels, drawn on every panel as landmark pairs. *pivot_in_section*
    gives the pivot on the SECTION's frame instead and wins over *pivot*: a
    picture drawn from a larger render (:func:`shown_section`) knows its pivot
    only relative to the section.

    *atlas_picture* is the atlas image on the native plane grid (None: the
    reference template), named *atlas_name* in captions. *regions*
    (``[(name, ids)]``) are drawn at full strength in every mode, the
    *outlines* layer then at :data:`REGION_CONTEXT_ALPHA` for context.
    *matrix_label* names a ready matrix in the caption. ``template`` draws
    the atlas image alone; *template_lines* adds the *outlines* layer to it.

    Returns ``(images, silhouette_iou)`` — every image captioned, and the
    overlap between the warped section's tissue mask and the atlas anatomy at
    this placement.
    """
    line_color, line_width = normalize_border_style(border_color, border_thickness)
    geometry = canvas_geometry(
        section.size,
        section_um_per_px,
        atlas,
        position_mm,
        plane,
        pitch_deg,
        yaw_deg,
        pad_to_fit_atlas=pad_to_fit_atlas,
    )
    fill = _background_color(section)
    canvas = Image.new("RGB", geometry.size, fill)
    canvas.paste(section.convert("RGB"), geometry.section_offset)

    ox, oy = (float(v) for v in geometry.section_offset)
    if isinstance(params, np.ndarray):
        section_matrix = np.asarray(params, dtype=np.float64)
    else:
        section_matrix = physical_affine_matrix(
            size=section.size,
            um_per_px=geometry.um_per_px,
            # The pivot arrives on the canvas; the matrix is built on the
            # section's frame and conjugated onto the canvas below.
            pivot=pivot_in_section if pivot_in_section is not None
            else None if pivot is None else (pivot[0] - ox, pivot[1] - oy),
            **params,
        )
    matrix = (_shift((ox, oy)) @ _as_3x3(section_matrix) @ _shift((-ox, -oy)))[:2]
    if spline is not None:
        from langslice.landmark_warp import warp_section

        warped = warp_section(
            np.asarray(section.convert("RGB"), dtype=np.uint8), spline,
            geometry.size, geometry.section_offset, geometry.um_per_px, fill,
        )
    else:
        warped = cv2.warpAffine(
            np.asarray(canvas, dtype=np.uint8),
            matrix,
            geometry.size,
            flags=cv2.INTER_LINEAR,
            borderValue=fill,
        )

    tissue = extract_slice_silhouette(cv2.cvtColor(warped, cv2.COLOR_RGB2GRAY))
    iou = silhouette_iou(tissue, atlas_mask_canvas(geometry))

    dark = is_dark_background(section)
    layer = str(outlines or "all").strip().lower()
    draw = outer_outline if layer == "outer" else family_outlines
    atlas_lines = (
        []
        if layer == "none"
        else draw(atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg)
    )

    highlighted: list[np.ndarray] = []
    sides_note = ""
    if regions:
        # A side is the SECTION's: carry the native plane onto the section
        # frame (native -> canvas is the atlas scale; section -> canvas the matrix).
        linear = _as_3x3(section_matrix)[:2, :2]
        left = regions_left(atlas, regions, position_mm, plane, pitch_deg, yaw_deg,
                            np.linalg.inv(linear) * geometry.atlas_scale)
        highlighted = region_polys(geometry.annotation, regions, left)
        if left is not None and np.linalg.det(linear) < 0:
            sides_note = " (the section's sides; this placement mirrors it)"
    atlas_head = f"atlas {atlas_name}"

    def _template_canvas() -> np.ndarray:
        plate = np.zeros_like(warped)
        _blend_template(
            plate, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry, opacity=1.0,
            picture=atlas_picture,
        )
        return plate

    silhouette: list[np.ndarray] = []
    lines = True  # atlas outlines drawn on every panel...
    if mode == "section":
        panels = [(warped, label or "section")]
        lines = False  # ...except the clean views, which show one source alone
    elif mode == "template":
        panels = [(_template_canvas(), atlas_head)]
        lines = template_lines
    elif mode == "side_by_side":
        panels = [(warped, label or "section"), (_template_canvas(), atlas_head)]
    elif mode == "checkerboard":
        panels = [(_checkerboard(warped, _template_canvas()), label or "section")]
    elif mode == "outlines":
        silhouette = _silhouette_polys(tissue)
        black = np.zeros_like(warped)
        if atlas_opacity > 0.0:  # an atlas image the call listed, on the black
            _blend_template(
                black, atlas, position_mm, plane, pitch_deg, yaw_deg, geometry,
                opacity=float(np.clip(atlas_opacity, 0.0, 1.0)), picture=atlas_picture,
            )
        panels = [(black, label or "section")]
        dark = True  # the canvas is black whatever the section is
    else:
        if atlas_opacity > 0.0:
            _blend_template(
                warped,
                atlas,
                position_mm,
                plane,
                pitch_deg,
                yaw_deg,
                geometry,
                opacity=float(np.clip(atlas_opacity, 0.0, 1.0)),
                picture=atlas_picture,
            )
        panels = [(warped, label or "section")]

    box = zoom_box(zoom, geometry.size)
    if spline is not None:
        knobs = (
            "landmark Elastix B-spline" if spline.get("backend") == "elastix"
            else "landmark thin-plate spline"
        )
    elif isinstance(params, np.ndarray):
        knobs = matrix_label
    else:
        knobs = (
            f"rot {params['rotation_deg']:.1f}  "
            f"scale {params['scale_x']:.3f}/{params['scale_y']:.3f}  "
            f"shift {params['translate_x_mm']:+.2f}/{params['translate_y_mm']:+.2f} mm"
            + (f"  shear {params['shear']:+.3f}" if params.get("shear") else "")
        )

    # The crop happens first, then the screen is sized: *long_edge*, or the
    # crop's own pixels when fewer (never upsampled). No *long_edge* means
    # canvas pixels one to one (host-side use, never a model's screen).
    edge = None
    if long_edge is not None:
        edge = max(1, min(int(long_edge), max(box[2] - box[0], box[3] - box[1])))
    images: list[Image.Image] = []
    for panel, head in panels:
        screen, factor = _to_screen(panel, box, edge)
        if lines:
            _draw_outlines(
                screen, atlas_lines, geometry, color=line_color, thickness=line_width,
                factor=factor, origin=box[:2],
                alpha=REGION_CONTEXT_ALPHA if regions else 1.0,
            )
        if highlighted:
            _draw_polys(
                screen, highlighted, line_color, thickness=line_width,
                scale=geometry.atlas_scale, offset=geometry.atlas_offset,
                origin=box[:2], factor=factor,
            )
        if silhouette:
            _draw_polys(
                screen,
                silhouette,
                _tissue_color(dark),
                origin=box[:2],
                factor=factor,
            )
        if markers is not None and len(markers[0]):
            _draw_markers(screen, markers[0], markers[1], origin=box[:2], factor=factor)
        _draw_scale_bar(screen, geometry.um_per_px / factor, dark)
        # Two lines by meaning; `caption` wraps any line wider than the panel.
        text = (
            f"{head} @ {position_mm:.3f} mm  pitch {pitch_deg:.2f} yaw {yaw_deg:.2f}\n"
            f"{knobs}  canvas {geometry.um_per_px:.2f} um/px"
        ).strip()
        if mode != "overlay" or box != (0, 0, geometry.size[0], geometry.size[1]):
            zoomed = [
                round(box[0] / geometry.size[0], 3),
                round(box[1] / geometry.size[1], 3),
                round(box[2] / geometry.size[0], 3),
                round(box[3] / geometry.size[1], 3),
            ]
            text += (
                f"\n{mode}  zoom {zoomed}  "
                f"view {geometry.um_per_px / factor:.2f} um/px"
            )
        if layer == "outer" and lines:
            text += "  outlines outer"
        if regions:
            text += "  regions " + ",".join(str(name) for name, _ids in regions) + sides_note
        labelled = caption(Image.fromarray(screen, mode="RGB"), text)
        if frames is not None:
            frames.append({
                "width": labelled.width, "height": labelled.height,
                "content_box": [0, labelled.height - screen.shape[0],
                                labelled.width, labelled.height],
                "canvas_box": list(box), "section_offset": list(geometry.section_offset),
                "section_size": list(section.size), "um_per_px": geometry.um_per_px,
            })
        images.append(labelled)
    return images, iou


def physical_overlay(
    section: Image.Image,
    section_um_per_px: float,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float,
    yaw_deg: float,
    params: dict[str, float] | np.ndarray,
    *,
    atlas_opacity: float = 0.0,
    border_color: str = "yellow",
    border_thickness: float = 0.5,
    pad_to_fit_atlas: bool = True,
    pivot: tuple[float, float] | None = None,
    label: str = "",
    long_edge: int | None = None,
) -> Image.Image:
    """The ONE screen of the alignment loop: section and atlas in millimetres.

    *section* is the display render, *section_um_per_px* the micrometres one
    of its pixels covers. The atlas is drawn at true physical scale on that
    canvas (:func:`canvas_geometry`), as one smoothed outline per FAMILY
    region; leaf boundaries are visual noise at this size and are not drawn.

    *params* is either the five physical knobs (``rotation_deg``,
    ``scale_x``, ``scale_y``, ``translate_x_mm``, ``translate_y_mm``) or a
    ready 2x3 matrix in the SECTION's frame — what a closed-form fit hands
    back — so the interactive loop and ``fit_affine`` draw the same picture.
    The ``overlay`` view of :func:`physical_views`, which is where the other
    views live.
    """
    images, _iou = physical_views(
        section,
        section_um_per_px,
        atlas,
        position_mm,
        plane,
        pitch_deg,
        yaw_deg,
        params,
        atlas_opacity=atlas_opacity,
        border_color=border_color,
        border_thickness=border_thickness,
        pad_to_fit_atlas=pad_to_fit_atlas,
        pivot=pivot,
        label=label,
        long_edge=long_edge,
    )
    return images[0]


def estimate_um_per_px(
    section: Image.Image,
    atlas: Any,
    position_mm: float,
    plane: Plane,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> float | None:
    """Micrometres per pixel guessed from the tissue's width, or None.

    The fallback when no file and no host says: the section's tissue is
    assumed to span the same millimetres as the atlas anatomy at this
    position, which is a SHAPE fit, not a calibration — every caller that
    uses it reports the calibration as "estimated".
    """
    mask = foreground_mask(section)
    if mask is None:
        return None
    xs = np.nonzero(mask.any(axis=0))[0]
    if xs.size == 0:
        return None
    # The mask is measured on a proxy; scale its width back onto the section.
    tissue_px = (xs.max() - xs.min() + 1) * section.width / float(mask.shape[1])
    ann = annotation_slice(
        atlas, position_mm, plane=plane, pitch_deg=pitch_deg, yaw_deg=yaw_deg
    )
    columns = np.nonzero(ann.any(axis=0))[0]
    if columns.size == 0 or tissue_px < 1:
        return None
    atlas_um = (columns.max() - columns.min() + 1) * atlas_um_per_px(atlas)
    return float(atlas_um / tissue_px)
