"""The eight nodes of the whole-brain graph.

``ingest``, ``survey``, ``fix`` and ``emit`` are implemented; the four steps
between ``fix`` and ``emit`` are still stubs that log, mark themselves
complete and fall through to their default successor. Their docstrings are
the contract a later implementation has to satisfy — what it reads from
:class:`~langslice.linear.whole_brain.state.StackState` and what it writes
back.

Every node signature is ``async (state, ctx) -> next_node_name``; return ""
for the default successor.
"""

from __future__ import annotations

import json
import logging
import os
from typing import cast

from PIL import Image, ImageDraw

from langslice.atlas.core import get_position_range_mm
from langslice.image_prep import normalize_image, prepare_image_for_vlm
from langslice.linear.whole_brain.discovery import (
    CONTACT_SHEET_FILENAME,
    discover_slices,
)
from langslice.linear.whole_brain.engine import EngineContext, Node
from langslice.linear.whole_brain.state import SliceState, StackState
from langslice.linear.whole_brain.survey import run_survey_session
from langslice.space import Plane

logger = logging.getLogger(__name__)

_PLANES = ("coronal", "sagittal", "horizontal")


# --- contact sheet -------------------------------------------------------


def build_contact_sheet(
    state: StackState,
    ctx: EngineContext,
    *,
    thumb_px: int = 256,
    columns: int = 6,
) -> str:
    """Render the stack as one labelled thumbnail grid; return its path.

    Rendered through the corrected view — slices appear in
    ``index_corrected`` order and flipped slices are mirrored — so re-running
    it after a fix step shows the stack the agent believes it has. Labels are
    ``<corrected index>: <filename>``.
    """
    ordered = state.in_order()
    if not ordered:
        raise ValueError("cannot build a contact sheet for an empty stack")

    label_px = 14
    cell_w, cell_h = thumb_px, thumb_px + label_px
    cols = max(1, min(columns, len(ordered)))
    rows = (len(ordered) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)

    for position, record in enumerate(ordered):
        with Image.open(ctx.image_path(record.id)) as handle:
            # Detach from the file before it closes: prepare_image_for_vlm can
            # hand back the very object it was given when no resize is needed.
            source = normalize_image(handle.copy())
        thumb = prepare_image_for_vlm(source, max_long_edge=thumb_px).image
        if record.flip:
            thumb = thumb.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        col, row = position % cols, position // cols
        x0, y0 = col * cell_w, row * cell_h
        sheet.paste(
            thumb,
            (x0 + (cell_w - thumb.width) // 2, y0 + (thumb_px - thumb.height) // 2),
        )
        draw.text(
            (x0 + 3, y0 + thumb_px + 2),
            f"{record.index_corrected}: {record.id}",
            fill=(235, 235, 235),
        )

    out_path = os.path.join(
        os.path.dirname(ctx.checkpoint_path), CONTACT_SHEET_FILENAME
    )
    sheet.save(out_path)
    return out_path


# --- nodes ---------------------------------------------------------------


async def ingest(state: StackState, ctx: EngineContext) -> str:
    """Discover the stack and populate state. Plain code, no model.

    Reads: nothing (``ctx.config`` only).
    Writes: atlas/plane/interval/thickness/keep_order, one
    :class:`SliceState` per discovered image (corrected order = discovery
    order, no flips), ``contact_sheet``, and an atlas-range note.
    """
    config = ctx.config
    if config.plane not in _PLANES:
        raise ValueError(f"Unsupported plane {config.plane!r}; expected one of {_PLANES}")

    paths = discover_slices(ctx.image_folder)
    if not paths:
        raise ValueError(f"No slice images found in {ctx.image_folder}")

    atlas = ctx.atlas_loader(config.atlas)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=cast(Plane, config.plane))

    state.image_folder = ctx.image_folder
    state.atlas = config.atlas
    state.plane = config.plane
    state.interval_mm = config.interval_mm
    state.thickness_mm = config.thickness_mm
    state.keep_order = config.keep_order
    state.slices = [
        SliceState(
            id=os.path.basename(path), index_original=index, index_corrected=index
        )
        for index, path in enumerate(paths)
    ]
    state.notes.append(
        f"ingest: {len(paths)} slices, atlas {config.atlas} "
        f"({config.plane}) spans {pos_lo:.2f}-{pos_hi:.2f} mm"
    )

    state.contact_sheet = build_contact_sheet(state, ctx)
    ctx.progress(
        f"[ingest] {len(paths)} slices; contact sheet -> {state.contact_sheet}"
    )
    return ""


async def survey(state: StackState, ctx: EngineContext) -> str:
    """Fused stack review: order, hemisphere flips, damage, gaps.

    Reads: ``contact_sheet`` plus per-slice images on demand, ``keep_order``,
    filenames as ordering context.
    Writes: the agent's tools apply corrections directly — ``index_corrected``,
    ``flip``, ``damaged``/``damage_note`` — and the submission adds
    ``axis_directions``, ``interval_breaks`` and notes.
    Routes: "fix" when this pass corrected something (fix re-renders the stack
    and sends it back for one re-check), "seed" when the stack is clean.
    """
    atlas = ctx.atlas_loader(state.atlas or ctx.config.atlas)
    pos_lo, pos_hi = get_position_range_mm(atlas, plane=cast(Plane, state.plane))
    metadata = getattr(atlas, "metadata", None) or {}
    species = str(metadata.get("species", "mouse"))

    outcome = await run_survey_session(
        state=state,
        ctx=ctx,
        species=species,
        pos_lo=pos_lo,
        pos_hi=pos_hi,
    )

    findings = outcome.findings
    if findings is None:
        state.notes.append(
            f"survey: incomplete — no submission within {outcome.turns} turns"
        )
        clean = True
    else:
        directions = findings.get("axis_directions") or {}
        if isinstance(directions, dict):
            state.axis_directions.update({str(k): str(v) for k, v in directions.items()})
        breaks: set[int] = set()
        for index in findings.get("interval_breaks") or []:
            try:
                breaks.add(int(index))
            except (TypeError, ValueError):
                continue
        state.interval_breaks = sorted(breaks)
        summary = str(findings.get("summary", "")).strip()
        if summary:
            state.notes.append(f"survey: {summary}")
        state.notes.extend(
            f"survey: {note}" for note in findings.get("notes") or [] if str(note).strip()
        )
        # An agent that applied corrections and still called itself clean gets
        # the re-check anyway: the contact sheet it looked at is now stale.
        clean = bool(findings.get("clean")) and not outcome.corrections_applied

    flipped = sum(1 for s in state.slices if s.flip)
    damaged = sum(1 for s in state.slices if s.damaged)
    ctx.progress(
        f"[survey] {outcome.tool_calls} tool calls; {flipped} flipped, "
        f"{damaged} damaged, {len(state.interval_breaks)} interval break(s); "
        f"{'clean' if clean else 'corrections applied'}"
    )

    if not clean:
        return "fix"
    if outcome.corrections_applied:
        # Only reachable when the pass ended without a submission: keep the
        # sheet honest about the corrections its tools did apply.
        state.contact_sheet = build_contact_sheet(state, ctx)
    # Jump past fix — it has nothing to re-render. The engine books skipped
    # nodes as complete so a resumed run does not walk back into them.
    return "seed"


async def fix(state: StackState, ctx: EngineContext) -> str:
    """Re-render the corrected stack for a second look. Plain code, no model.

    The survey's tools already applied its corrections to state, so all that
    is left is a contact sheet that shows the stack as it now stands.
    Routes: "survey" to re-check, until the stack is clean or the survey cycle
    limit is hit (the engine then falls through to seed).
    """
    if not state.slices:
        return ""
    state.contact_sheet = build_contact_sheet(state, ctx)
    flipped = sum(1 for s in state.slices if s.flip)
    damaged = sum(1 for s in state.slices if s.damaged)
    ctx.progress(
        f"[fix] contact sheet rebuilt ({flipped} flipped, {damaged} damaged) "
        f"-> {state.contact_sheet}"
    )
    return "survey"


async def seed(state: StackState, ctx: EngineContext) -> str:
    """Seed initial positions for the stack. STUB.

    Reads: corrected stack order, ``plane``, ``atlas``, damage flags.
    Writes: ``position_mm`` + ``position_source`` ("deepslice" when the
    optional DeepSlice tool ran, otherwise "survey"), and
    ``oblique_angles_deg`` if the seeding tool reported one.
    Routes: "" (position).
    """
    ctx.progress("[seed] not implemented (stub): no positions seeded")
    return ""


async def position(state: StackState, ctx: EngineContext) -> str:
    """Refine positions and estimate the oblique angle. STUB.

    Reads: seeded positions, ``interval_mm``/``thickness_mm``, advisory
    spacing signals from
    :mod:`langslice.linear.whole_brain.signals`, per-slice escalation via
    :func:`langslice.linear.whole_brain.estimation_agents.run_slice_estimation`.
    Writes: ``position_mm`` + ``position_source`` ("anchor" for key slices,
    "refined" elsewhere), ``confidence``, ``interval_breaks``,
    ``oblique_angles_deg``.
    Routes: itself while interval breaks remain (bounded), else "" (transforms).
    """
    ctx.progress("[position] not implemented (stub): positions unchanged")
    return ""


async def transforms(state: StackState, ctx: EngineContext) -> str:
    """Per-slice fan-out: affine for intact slices, interactive for damaged. STUB.

    Reads: per-slice ``position_mm``, ``damaged``, ``flip``.
    Writes: ``affine`` (Elastix parameters, proposed not applied) for intact
    slices; ``interactive_transform`` (rotation_deg, scale_x, scale_y,
    translate_x, translate_y) for damaged ones; ``caveats`` where the fit is
    weak.
    Routes: "" (review).
    """
    ctx.progress("[transforms] not implemented (stub): no transforms proposed")
    return ""


async def review(state: StackState, ctx: EngineContext) -> str:
    """Whole-stack consistency pass. STUB.

    Reads: the full positioned, transformed stack plus the advisory monotone
    fit from :func:`langslice.linear.whole_brain.signals.monotone_fit`.
    Writes: ``confidence``, ``caveats``, ``notes``.
    Routes: "position" once if the stack is inconsistent, else "" (emit).
    The stub falls through to emit.
    """
    ctx.progress("[review] not implemented (stub): stack accepted as-is")
    return ""


async def emit(state: StackState, ctx: EngineContext) -> str:
    """Hand back results. Plain code, no model.

    The serialized :class:`StackState` *is* the result — the same shape the
    checkpoint uses — so the CLI, the checkpoint and any host adapter all
    read one schema.
    """
    os.makedirs(os.path.dirname(os.path.abspath(ctx.results_path)), exist_ok=True)
    with open(ctx.results_path, "w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2)
    ctx.progress(f"[emit] results -> {ctx.results_path}")
    return ""


NODES: list[Node] = [
    ("ingest", ingest),
    ("survey", survey),
    ("fix", fix),
    ("seed", seed),
    ("position", position),
    ("transforms", transforms),
    ("review", review),
    ("emit", emit),
]
