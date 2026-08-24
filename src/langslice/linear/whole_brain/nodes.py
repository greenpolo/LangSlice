"""The eight nodes of the whole-brain graph.

Each node's docstring is its contract: what it reads from
:class:`~langslice.linear.whole_brain.state.StackState`, what it writes back,
and where it can route. The work itself lives beside them — ``survey.py``,
``position.py``, ``transforms.py``, ``review.py`` — so a node stays a routing
decision plus its bookkeeping.

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
from langslice.linear.whole_brain._step_common import render_slice
from langslice.linear.whole_brain.deepslice import deepslice_available
from langslice.linear.whole_brain.discovery import (
    CONTACT_SHEET_FILENAME,
    discover_slices,
)
from langslice.linear.whole_brain.engine import EngineContext, Node
from langslice.linear.whole_brain.position import run_position_session
from langslice.linear.whole_brain.review import run_review_session
from langslice.linear.whole_brain.state import SliceState, StackState
from langslice.linear.whole_brain.survey import run_survey_session
from langslice.linear.whole_brain.transforms import (
    run_affine_pass,
    run_interactive_transforms,
)
from langslice.space import Plane

logger = logging.getLogger(__name__)

_PLANES = ("coronal", "sagittal", "horizontal")


def _atlas_range(state: StackState, ctx: EngineContext) -> tuple[float, float]:
    """Valid position range of the stack's atlas along its slicing plane."""
    atlas = ctx.atlas_loader(state.atlas or ctx.config.atlas)
    return get_position_range_mm(atlas, plane=cast(Plane, state.plane))


def _species(state: StackState, ctx: EngineContext) -> str:
    """Species from the atlas metadata; mouse when the atlas does not say."""
    metadata = getattr(ctx.atlas_loader(state.atlas or ctx.config.atlas), "metadata", None)
    return str((metadata or {}).get("species", "mouse"))


# --- contact sheet -------------------------------------------------------


def build_contact_sheet(
    state: StackState,
    ctx: EngineContext,
    *,
    thumb_px: int = 256,
    columns: int = 6,
) -> str:
    """Render the stack as one labelled thumbnail grid; return its path.

    Thumbnails come from :func:`render_slice`, so the sheet shows the corrected
    view — ``index_corrected`` order, flips mirrored, the same preprocessing the
    per-slice views use. Labels are ``<corrected index>: <filename>``.
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
        thumb = render_slice(ctx, record, long_edge=thumb_px)
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
    pos_lo, pos_hi = _atlas_range(state, ctx)
    outcome = await run_survey_session(
        state=state,
        ctx=ctx,
        species=_species(state, ctx),
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
    """Automatic seeding, if any is available. Plain code, no model.

    There is no automatic seeder installed today: DeepSlice would place a whole
    coronal mouse stack in one shot (see
    :mod:`langslice.linear.whole_brain.deepslice`) but is an optional extra that
    is not wired. Nothing else is prescribed here on purpose — picking key
    sections needs intimate atlas knowledge, so *which* placement strategy to
    use is the positioning agent's decision, not this node's.

    Reads: nothing but the stack size.
    Writes: one note. Positions stay ``None``.
    Routes: "" (position).
    """
    if not state.slices:
        return ""
    if deepslice_available():
        # No integration behind this yet; the seam exists so adding the extra
        # is a one-function change (run_deepslice) rather than a node rewrite.
        reason = "deepslice is installed but not wired"
    else:
        reason = "no automatic seeding available"
    state.notes.append(
        f"seed: {reason}; placement strategy left to the positioning agent"
    )
    ctx.progress(
        f"[seed] {reason} — every section stays unplaced; the positioning "
        f"agent chooses its own strategy"
    )
    return ""


async def position(state: StackState, ctx: EngineContext) -> str:
    """Place the whole stack against the atlas, with the whole stack in context.

    This step owns the placement strategy — key sections plus interpolation,
    estimating every section, or whatever mix the stack calls for — and it
    usually starts from an unplaced stack.

    Reads: any positions already on the stack, ``interval_mm``/``thickness_mm``,
    the contact sheet, advisory spacing signals from
    :mod:`langslice.linear.whole_brain.signals`, and single-slice estimation via
    :func:`langslice.linear.whole_brain.estimation_agents.run_slice_estimation`.
    Writes: ``position_mm`` + ``position_source`` ("refined"), ``confidence``,
    ``interval_breaks`` and notes from the submission.
    Routes: "" (transforms). Oblique-angle estimation is not part of this
    build — it needs atlas re-slicing machinery that does not exist yet.
    """
    if not state.slices:
        return ""
    pos_lo, pos_hi = _atlas_range(state, ctx)
    outcome = await run_position_session(
        state=state,
        ctx=ctx,
        species=_species(state, ctx),
        pos_lo=pos_lo,
        pos_hi=pos_hi,
    )

    findings = outcome.findings
    if findings is None:
        # Budget exhausted: whatever set_positions wrote is kept and the run
        # goes on — a partly refined stack beats no stack.
        state.notes.append(
            f"position: incomplete — no submission within {outcome.turns} turns"
        )
    else:
        breaks: set[int] = set()
        for index in findings.get("interval_breaks") or []:
            try:
                breaks.add(int(index))
            except (TypeError, ValueError):
                continue
        state.interval_breaks = sorted(breaks)
        summary = str(findings.get("summary", "")).strip()
        if summary:
            state.notes.append(f"position: {summary}")
        state.notes.extend(
            f"position: {note}"
            for note in findings.get("notes") or []
            if str(note).strip()
        )

    ctx.progress(
        f"[position] {outcome.tool_calls} tool calls; "
        f"{outcome.positions_written} section(s) repositioned, "
        f"{outcome.estimated} section(s) estimated directly, "
        f"{len(state.interval_breaks)} interval break(s)"
        + ("" if findings is not None else "; incomplete")
    )
    return ""


async def transforms(state: StackState, ctx: EngineContext) -> str:
    """Per-slice fan-out: affine for intact slices, interactive for damaged.

    Reads: per-slice ``position_mm``, ``damaged``, ``flip``.
    Writes: ``affine`` (silhouette-fit parameters, proposed not applied) for
    intact slices; ``interactive_transform`` (rotation_deg, scale_x, scale_y,
    translate_x, translate_y) for damaged ones; ``caveats`` where a fit failed
    or came out weak.
    Routes: "" (review). A failed transform is never fatal — the section keeps
    its position and carries a caveat.
    """
    if not state.slices:
        return ""
    fitted, failed = run_affine_pass(state, ctx)

    pos_lo, pos_hi = _atlas_range(state, ctx)
    interactive, empty = await run_interactive_transforms(
        state, ctx, pos_lo=pos_lo, pos_hi=pos_hi
    )

    state.notes.append(
        f"transforms: {fitted} affine fit(s), {failed} affine failure(s), "
        f"{interactive} interactive transform(s)"
    )
    ctx.progress(
        f"[transforms] {fitted} intact section(s) fitted"
        + (f", {failed} failed" if failed else "")
        + f"; {interactive} damaged section(s) aligned interactively"
        + (f", {empty} without a transform" if empty else "")
    )
    return ""


async def review(state: StackState, ctx: EngineContext) -> str:
    """Whole-stack consistency pass: the last look before hand-back.

    Reads: the full positioned, transformed stack plus the advisory spacing
    signals (interval table, interpolation residuals, monotone fit).
    Writes: ``confidence``, ``caveats``, ``notes``.
    Routes: "position" when the agent refuses the stack (the engine bounds
    that loop; on refusal it falls through), else "" (emit). A pass that runs
    out of turns approves — review can add caveats, never block hand-back.
    """
    if not state.slices:
        return ""
    pos_lo, pos_hi = _atlas_range(state, ctx)
    outcome = await run_review_session(
        state=state,
        ctx=ctx,
        species=_species(state, ctx),
        pos_lo=pos_lo,
        pos_hi=pos_hi,
    )

    findings = outcome.findings
    if findings is None:
        state.notes.append(
            f"review: incomplete — no verdict within {outcome.turns} turns; "
            "stack handed back as it stands"
        )
        approved = True
    else:
        approved = bool(findings.get("approved"))
        summary = str(findings.get("summary", "")).strip()
        if summary:
            state.notes.append(f"review: {summary}")
        # Notes are how a refusal reaches the next positioning pass: its seed
        # message reads them back out of state.
        state.notes.extend(
            f"review: {note}" for note in findings.get("notes") or [] if str(note).strip()
        )

    ctx.progress(
        f"[review] {outcome.tool_calls} tool calls; {outcome.flagged} section(s) "
        f"flagged; {'approved' if approved else 'sent back to position'}"
    )
    return "" if approved else "position"


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
