"""The one toolbox: every tool the linear agent can be given, gated by the spec.

Conventions, applied to every tool: sections are addressed by filename or
corrected index; every write returns the rows it changed (``status`` is the
whole table); every write is undoable; every write checkpoints; fits have a
form that computes without writing.

Tools report data. No advice, no interpretation, no strategy in any payload —
every benchmark failure worth tracing came back to harness text telling the
model what to think. The submit gates are the exception, and a constraint that
states a number is not coaching: refusals name the numbers that caused them and
stop there.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from google.genai import types

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.affine import (
    denormalized_affine,
    normalized_physical_affine,
    physical_affine_matrix,
    resize_long_edge,
)
from langslice.linear.atlas_fetch import (
    ATLAS_LONG_EDGE,
    atlas_section,
    make_fetch_atlas,
)
from langslice.linear.checkpoint import save_checkpoint
from langslice.linear.deepslice import run_deepslice as _run_deepslice
from langslice.linear.render import (
    MAX_IMAGES_PER_CALL,
    OUTLINE_LAYERS,
    OVERLAY_LONG_EDGE,
    PREVIEW_LONG_EDGE,
    VIEW_LONG_EDGE,
    VIEW_MODES,
    CanvasGeometry,
    beside,
    canvas_geometry,
    caption,
    compact_rows,
    image_to_part,
    physical_views,
    pivot_on_canvas,
    render_slice,
    spacing_plot,
    stack_sheet,
    stacked,
    status_rows,
)
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState
from langslice.linear.transform import (
    affine_fit,
    calibrate,
    fit_silhouette,
    physical_decomposition,
    physical_params,
    similarity_fit,
)
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

#: Sections one ``view_slices`` call may return; the same bound applies to
#: every image-returning tool (see :data:`MAX_IMAGES_PER_CALL`).
MAX_VIEW_SLICES = MAX_IMAGES_PER_CALL

#: Overlay panels one ``fit_affine`` call may attach.
MAX_FIT_PANELS = MAX_IMAGES_PER_CALL

#: Undo snapshots kept in memory. Not persisted: a resumed run starts from the
#: checkpoint, which is the state as it stood.
UNDO_DEPTH = 50

#: A reported interval break must be at least this much wider than the stack's
#: own median written spacing.
INTERVAL_BREAK_MIN_RATIO = 1.5

#: Strict-interval tolerance: consecutive spacing must be within this fraction
#: of the nominal interval.
STRICT_INTERVAL_TOLERANCE = 0.10

#: Rotations ``orient_slices`` accepts.
_ROTATIONS = (0, 90, 180, 270)

#: Views ``adjust_transform`` composes on top of the renderer's own
#: (:data:`~langslice.linear.render.VIEW_MODES`): the A/B toggle, which is two
#: renders of one crop rather than one composition.
PREVIEW_MODES = (*VIEW_MODES, "ab")

#: The transform every section starts from, and the B side of an A/B preview
#: when a section carries nothing yet.
IDENTITY_PARAMS: dict[str, float] = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
}


@dataclass
class ToolBox:
    """The tools of one run plus the mutable results the engine reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)
    #: Every parameter set `adjust_transform` was given this run, per section
    #: id, oldest first.
    transform_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)

    @property
    def names(self) -> list[str]:
        return [tool.__name__ for tool in self.tools]


@dataclass(frozen=True)
class _Staged:
    """One section ready to be drawn or measured on its physical canvas."""

    record: SliceState
    section: Any
    um_per_px: float
    calibration_source: str
    geometry: CanvasGeometry
    params: dict[str, float]
    #: Rotation/scale centre in CANVAS pixels; None is the canvas centre.
    pivot: tuple[float, float] | None
    #: The same pivot as fractions of the canvas, for the payload.
    pivot_frac: list[float]
    #: What was asked for: "canvas", "tissue" or "fractions".
    pivot_mode: str

    @property
    def pivot_in_section(self) -> tuple[float, float] | None:
        """The pivot on the SECTION's frame, which the six numbers live on."""
        if self.pivot is None:
            return None
        ox, oy = self.geometry.section_offset
        return (self.pivot[0] - ox, self.pivot[1] - oy)

    @property
    def calibration(self) -> dict[str, Any]:
        return {
            "section_um_per_px": round(self.um_per_px, 4),
            "source": self.calibration_source,
        }

    def matrix(self, params: dict[str, float] | None = None) -> Any:
        """The canvas-pixel 2x3 of *params* (default: the staged ones).

        Built on the canvas frame directly: width and height cancel out of the
        physical translation, so this is the very map the picture shows.
        """
        return physical_affine_matrix(
            size=self.geometry.size,
            um_per_px=self.um_per_px,
            pivot=self.pivot,
            **(params if params is not None else self.params),
        )


# --- submit gates --------------------------------------------------------


def missing_positions(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section has a position, else the rejection."""
    missing = [record.id for record in state.in_order() if record.position_mm is None]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_POSITIONS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) have no "
            "position; every section needs one, damaged ones included."
        ),
    }


def order_position_mismatch(state: StackState) -> dict[str, Any] | None:
    """Positions must run one way along the corrected order (either way)."""
    placed = [s for s in state.in_order() if s.position_mm is not None]
    if len(placed) < 2:
        return None
    first = float(placed[0].position_mm)  # type: ignore[arg-type]
    last = float(placed[-1].position_mm)  # type: ignore[arg-type]
    direction = 1.0 if last >= first else -1.0
    pairs: list[dict[str, Any]] = []
    for before, after in zip(placed, placed[1:], strict=False):
        delta = float(after.position_mm) - float(before.position_mm)  # type: ignore[arg-type]
        if delta * direction < 0:
            pairs.append(
                {
                    "before": before.id,
                    "after": after.id,
                    "before_position_mm": round(float(before.position_mm), 3),  # type: ignore[arg-type]
                    "after_position_mm": round(float(after.position_mm), 3),  # type: ignore[arg-type]
                }
            )
    if not pairs:
        return None
    trend = "increase" if direction > 0 else "decrease"
    return {
        "status": "error",
        "error": "ORDER_POSITION_MISMATCH",
        "pairs": pairs,
        "message": (
            f"Positions {trend} from {first:.3f} mm to {last:.3f} mm along the "
            f"corrected order, but {len(pairs)} neighbour pair(s) run the other "
            "way."
        ),
    }


def strict_interval_error(state: StackState, breaks: list[int]) -> dict[str, Any] | None:
    """Every consecutive spacing within 10% of the interval, no breaks."""
    if breaks:
        return {
            "status": "error",
            "error": "STRICT_INTERVAL",
            "reported_breaks": list(breaks),
            "message": (
                "strict_interval is on: interval_breaks must be empty, "
                f"{len(breaks)} were reported."
            ),
        }
    interval = float(state.interval_mm)
    if interval <= 0:
        return None
    placed = [s for s in state.in_order() if s.position_mm is not None]
    tolerance = interval * STRICT_INTERVAL_TOLERANCE
    failures: list[dict[str, Any]] = []
    for before, after in zip(placed, placed[1:], strict=False):
        spacing = abs(float(after.position_mm) - float(before.position_mm))  # type: ignore[arg-type]
        if abs(spacing - interval) > tolerance:
            failures.append(
                {
                    "between": [before.id, after.id],
                    "spacing_mm": round(spacing, 3),
                }
            )
    if not failures:
        return None
    return {
        "status": "error",
        "error": "STRICT_INTERVAL",
        "interval_mm": round(interval, 3),
        "tolerance_mm": round(tolerance, 3),
        "failures": failures,
        "message": (
            f"strict_interval is on: every consecutive spacing must be within "
            f"{STRICT_INTERVAL_TOLERANCE:.0%} of {interval:.3f} mm; "
            f"{len(failures)} pair(s) are not."
        ),
    }


def unsupported_breaks(state: StackState, breaks: list[int]) -> dict[str, Any] | None:
    """Check reported interval breaks against the spacing that was WRITTEN.

    A break at corrected index *i* claims the gap between *i-1* and *i* is
    larger than the rest of the stack's; the positions on the state make that
    claim checkable without an image.
    """
    if not breaks:
        return None
    ordered = state.in_order()
    deltas = [
        abs(float(b.position_mm) - float(a.position_mm))  # type: ignore[arg-type]
        for a, b in zip(ordered, ordered[1:], strict=False)
        if a.position_mm is not None and b.position_mm is not None
    ]
    if not deltas:
        return None
    median = float(sorted(deltas)[len(deltas) // 2])
    threshold = median * INTERVAL_BREAK_MIN_RATIO

    failures: list[dict[str, Any]] = []
    for index in sorted(set(breaks)):
        position = next(
            (i for i, r in enumerate(ordered) if r.index_corrected == index), None
        )
        if position is None or position == 0:
            failures.append(
                {
                    "index": index,
                    "error": "NOT_A_GAP",
                    "reason": (
                        f"corrected index {index} has no section before it; a "
                        "break index names the section AFTER the gap."
                    ),
                }
            )
            continue
        before, after = ordered[position - 1], ordered[position]
        if before.position_mm is None or after.position_mm is None:
            continue
        written = abs(float(after.position_mm) - float(before.position_mm))
        if written > threshold:
            continue
        failures.append(
            {
                "index": index,
                "error": "NOT_A_GAP",
                "written_interval_mm": round(written, 3),
                "median_interval_mm": round(median, 3),
                "between": [before.id, after.id],
            }
        )
    if not failures:
        return None
    return {
        "status": "error",
        "error": "INTERVAL_BREAKS_UNSUPPORTED",
        "failures": failures,
        "message": (
            f"{len(failures)} reported interval break(s) are not in the "
            "positions you wrote. A break index is accepted only where the "
            f"written interval exceeds {INTERVAL_BREAK_MIN_RATIO:g}x the "
            "stack's median written spacing."
        ),
    }


def missing_transforms(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section carries a transform, else the rejection."""
    missing = [record.id for record in state.in_order() if record.transform is None]
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_TRANSFORMS",
        "missing_ids": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) carry no "
            "transform; every section needs one, damaged ones included."
        ),
    }


def submit_errors(
    state: StackState, spec: JobSpec, breaks: list[int]
) -> dict[str, Any] | None:
    """Every gate that applies to this run, in order."""
    if spec.has("position"):
        refusal = (
            missing_positions(state)
            or order_position_mismatch(state)
            or (
                strict_interval_error(state, breaks)
                if spec.position.strict_interval
                else unsupported_breaks(state, breaks)
            )
        )
        if refusal is not None:
            return refusal
    if spec.has("transform"):
        return missing_transforms(state)
    return None


# --- the toolbox ---------------------------------------------------------


def build_tools(state: StackState, ctx: EngineContext, spec: JobSpec) -> ToolBox:
    """Build the tools this run's spec switches on, closed over *state*."""
    box = ToolBox()
    pos_lo, pos_hi = ctx.position_range

    # --- shared plumbing ------------------------------------------------

    def rows() -> dict[str, Any]:
        return {
            "rows": compact_rows(status_rows(state)),
            "cutting_angles_deg": dict(state.cutting_angles_deg),
            "interval_breaks": list(state.interval_breaks),
        }

    def changed(touched: list[str]) -> dict[str, Any]:
        """The status rows of the sections a write touched, and nothing else.

        A write used to answer with the whole table; on a forty-section stack
        that is thirty-nine rows of noise per call. `status` is still the
        whole table, and is one call away.
        """
        wanted = set(touched)
        return {
            "changed": compact_rows([row for row in status_rows(state) if row["id"] in wanted]),
            "n_sections": len(state.slices),
            "cutting_angles_deg": dict(state.cutting_angles_deg),
            "interval_breaks": list(state.interval_breaks),
        }

    def push(before: dict[str, Any]) -> None:
        """Record one undo step. One tool call = one step, batch included."""
        box.undo_stack.append(before)
        del box.undo_stack[:-UNDO_DEPTH]
        box.redo_stack.clear()

    def snapshot() -> None:
        """Take the undo step, before anything is written."""
        push(state.to_dict())

    def commit(*touched: str) -> dict[str, Any]:
        """Checkpoint the write and answer with the rows it changed."""
        save_checkpoint(state, ctx.checkpoint_path)
        return {"status": "ok", **changed(list(touched))}

    def resolve_many(refs: list[Any]) -> tuple[list[SliceState], list[str]]:
        known: list[SliceState] = []
        unknown: list[str] = []
        for ref in refs:
            record = state.resolve(ref)
            if record is None:
                unknown.append(str(ref))
            else:
                known.append(record)
        return known, unknown

    def renumber(order: list[SliceState]) -> list[str]:
        """Apply a new corrected order; return the ids whose index changed.

        Only ``index_corrected`` moves. Positions and transforms stay where
        they are; ``submit`` is what holds order and position together.
        """
        moved: list[str] = []
        for index, record in enumerate(order):
            if record.index_corrected != index:
                moved.append(record.id)
            record.index_corrected = index
        return moved

    # --- always on ------------------------------------------------------

    def status() -> dict[str, Any]:
        """The stack as it stands: one row per section in corrected order.

        Returns:
            ``rows`` (index, id, position_mm, delta_to_next_mm, flip,
            rotation_deg, damaged, damage_note, transform kind, transform_iou,
            transform_mirrored, caveats), plus the stack's cutting angles and
            interval breaks. Writes return only the rows they changed; this is
            the whole table.
        """
        return {"status": "ok", **rows()}

    def section_part(record: SliceState, *, long_edge: int = VIEW_LONG_EDGE) -> types.Part:
        """One section as corrected, tissue-framed, its index and id burned in."""
        return image_to_part(
            caption(
                render_slice(ctx, record, long_edge=long_edge, frame=True),
                f"{record.index_corrected}: {record.id}",
            )
        )

    def view_slices(slice_ids: list[str]) -> dict[str, Any]:
        """Look at up to 4 named sections at higher resolution.

        Sections are rendered as corrected: any rotation and flip already
        applied, framed to their tissue the same way fetched atlas sections
        are. Each image carries its corrected index and filename burned into
        its top-left corner.

        Args:
            slice_ids: Filenames or corrected indices (max 4 per call).

        Returns:
            status/slice_ids/description plus the images, in the order asked.
        """
        if not slice_ids:
            return {"status": "error", "error": "BAD_ARGS"}
        known, unknown = resolve_many(list(slice_ids)[:MAX_VIEW_SLICES])
        if not known:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        parts = [section_part(record) for record in known]
        return {
            "status": "ok",
            "slice_ids": [record.id for record in known],
            "unknown_ids": unknown,
            # Each image carries its own burned-in label; the ordering note
            # says the same thing in the payload.
            "description": (
                "Attached images are "
                + ", ".join(record.id for record in known)
                + ", in that order, rendered as corrected, each labelled "
                "'<corrected index>: <filename>' in its top-left corner."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    def note(text: str) -> dict[str, Any]:
        """Append one line to the run notes, which are saved with the results.

        Args:
            text: The note.
        """
        cleaned = str(text or "").strip()
        if not cleaned:
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        state.notes.append(cleaned)
        save_checkpoint(state, ctx.checkpoint_path)
        return {"status": "ok", "notes": list(state.notes)}

    def undo() -> dict[str, Any]:
        """Undo the last write. One tool call undoes as one step."""
        if not box.undo_stack:
            return {"status": "error", "error": "NOTHING_TO_UNDO"}
        box.redo_stack.append(state.to_dict())
        state.restore(box.undo_stack.pop())
        save_checkpoint(state, ctx.checkpoint_path)
        return {"status": "ok", "undo_depth": len(box.undo_stack), **rows()}

    def redo() -> dict[str, Any]:
        """Redo the write that ``undo`` reversed."""
        if not box.redo_stack:
            return {"status": "error", "error": "NOTHING_TO_REDO"}
        box.undo_stack.append(state.to_dict())
        del box.undo_stack[:-UNDO_DEPTH]
        state.restore(box.redo_stack.pop())
        save_checkpoint(state, ctx.checkpoint_path)
        return {"status": "ok", "redo_depth": len(box.redo_stack), **rows()}

    def mark_damaged(entries: list[dict[str, str]]) -> dict[str, Any]:
        """Record sections whose shape would break an outline-based fit.

        Damage here means the section outline massively deviates from the
        atlas: large missing chunks, a missing hemisphere or olfactory bulb,
        split or independently rotated hemispheres, displaced fragments.
        Bubbles, stains, low contrast and small tears with an intact outline
        are NOT damage.

        Args:
            entries: ``[{"id": "<filename>", "note": "<what breaks the outline>"}]``

        Returns:
            The rows this call changed.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        marked: list[str] = []
        unknown: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            record.damaged = True
            record.damage_note = str(entry.get("note", "")).strip()
            marked.append(record.id)
        return {"marked": marked, "unknown_ids": unknown, **commit(*marked)}

    def unmark_damaged(slice_ids: list[str]) -> dict[str, Any]:
        """Clear the damaged flag on the named sections.

        Args:
            slice_ids: Filenames or corrected indices.

        Returns:
            The rows this call changed.
        """
        if not slice_ids:
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        known, unknown = resolve_many(list(slice_ids))
        for record in known:
            record.damaged = False
            record.damage_note = ""
        return {
            "unmarked": [record.id for record in known],
            "unknown_ids": unknown,
            **commit(*[record.id for record in known]),
        }

    def submit(
        summary: str,
        notes: list[str],
        interval_breaks: list[int],
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """End the run. Call this exactly once, last.

        Args:
            summary: One or two sentences on what you did.
            interval_breaks: Corrected indices of the sections AFTER a gap you
                conclude is real. Empty if there are none.
            notes: Short observations worth carrying forward.
        """
        breaks: list[int] = []
        for raw in interval_breaks if isinstance(interval_breaks, (list, tuple)) else []:
            try:
                breaks.append(int(raw))
            except (TypeError, ValueError):
                continue
        refusal = submit_errors(state, spec, breaks)
        if refusal is not None:
            return refusal

        snapshot()
        state.interval_breaks = sorted(set(breaks))
        # Model output is a trust boundary: a malformed submission must not
        # take the run down.
        clean_notes = (
            [str(item).strip() for item in notes if str(item).strip()]
            if isinstance(notes, (list, tuple))
            else []
        )
        state.notes.extend(clean_notes)
        summary_text = str(summary or "").strip()
        if summary_text:
            state.notes.append(f"submit: {summary_text}")
        state.submitted = True
        box.submission.update(
            {
                "summary": summary_text,
                "notes": clean_notes,
                "interval_breaks": list(state.interval_breaks),
            }
        )
        save_checkpoint(state, ctx.checkpoint_path)
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", **rows()}

    def validate(interval_breaks: list[int]) -> dict[str, Any]:
        """Run the submit checks without submitting. Writes nothing.

        Args:
            interval_breaks: The corrected indices you would pass to `submit`.
                Empty list for none.

        Returns:
            The refusal `submit` would return, or
            ``{"status": "ok", "would_submit": true}``.
        """
        breaks: list[int] = []
        for raw in interval_breaks if isinstance(interval_breaks, (list, tuple)) else []:
            try:
                breaks.append(int(raw))
            except (TypeError, ValueError):
                continue
        return submit_errors(state, spec, breaks) or {
            "status": "ok",
            "would_submit": True,
        }

    box.tools = [
        status,
        validate,
        view_slices,
        make_fetch_atlas(state, ctx),
        note,
        undo,
        redo,
        mark_damaged,
        unmark_damaged,
    ]

    # --- reorder --------------------------------------------------------

    def orient_slices(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Set the flip and rotation of one or more sections, and show them.

        Corrections are recorded as data; the user's image files are never
        modified. Rotation is applied first, then the flip. A section whose
        orientation changes loses its transform. Determining hemisphere
        orientation (whether a section is mirrored) is only possible when
        there is a visible notch or a noticeable oblique cutting angle that
        produces differences between the hemispheres' anatomy.

        Args:
            entries: ``[{"id": "<filename>", "flip": true|false,
                "rotate_deg": 0|90|180|270}]``. Either key may be omitted to
                leave that correction as it is.

        Returns:
            The rows this call changed, and each changed section rendered as
            it now stands (up to 4), its corrected index and filename burned
            into its top-left corner.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        applied: list[str] = []
        unknown: list[str] = []
        rejected: list[dict[str, Any]] = []
        cleared: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            was = (record.flip, record.rotation_deg)
            if "flip" in entry and entry["flip"] is not None:
                if not spec.reorder.flip:
                    rejected.append({"id": record.id, "error": "FLIP_DISABLED"})
                else:
                    record.flip = bool(entry["flip"])
            if "rotate_deg" in entry and entry["rotate_deg"] is not None:
                try:
                    rotation = int(entry["rotate_deg"]) % 360
                except (TypeError, ValueError):
                    rotation = -1
                if rotation not in _ROTATIONS:
                    rejected.append(
                        {
                            "id": record.id,
                            "error": "BAD_ROTATION",
                            "allowed": list(_ROTATIONS),
                        }
                    )
                else:
                    record.rotation_deg = rotation
            if (record.flip, record.rotation_deg) != was and record.transform is not None:
                # A transform describes the section AFTER its orientation, so
                # an orientation change makes the old one stale.
                record.transform = None
                cleared.append(record.id)
            applied.append(record.id)
        parts: list[types.Part] = []
        failed: list[dict[str, str]] = []
        for name in applied[:MAX_VIEW_SLICES]:
            record = state.by_id(name)
            if record is None:
                continue
            try:
                parts.append(section_part(record))
            except Exception as exc:
                failed.append({"id": name, "message": str(exc)})
        return {
            "applied": applied,
            "cleared_transforms": cleared,
            "unknown_ids": unknown,
            "rejected": rejected,
            **commit(*applied),
            "description": (
                "Attached images are "
                + ", ".join(applied[:MAX_VIEW_SLICES])
                + ", in that order, rendered as they now stand, each labelled "
                "'<corrected index>: <filename>' in its top-left corner."
            ),
            "render_failed": failed,
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    def reorder_slices(new_order: list[str]) -> dict[str, Any]:
        """Set the corrected order of the WHOLE stack in one call.

        Only the corrected index changes: positions and transforms are kept.

        Args:
            new_order: Every section filename exactly once, in the order the
                sections were cut. Filenames, not corrected indices: the
                indices are what this call changes.

        Returns:
            The rows it changed, plus the ids whose corrected index changed.
        """
        ids = [s.id for s in state.slices]
        # Filenames only: a corrected index is exactly what this call changes,
        # so an index-addressed reorder can hit the wrong section next call.
        if sorted(str(item) for item in new_order) != sorted(ids):
            return {
                "status": "error",
                "error": "NOT_A_PERMUTATION",
                "missing_ids": [i for i in ids if i not in set(map(str, new_order))],
                "unknown_ids": [str(i) for i in new_order if str(i) not in set(ids)],
                "message": (
                    f"new_order must list all {len(ids)} section filenames "
                    "exactly once."
                ),
            }
        snapshot()
        ordered = [state.by_id(str(item)) for item in new_order]
        moved = renumber([record for record in ordered if record is not None])
        return {"moved": moved, **commit(*moved)}

    def move_slice(slice_id: str, after: str) -> dict[str, Any]:
        """Move one section to a new place in the corrected order.

        Only corrected indices change: positions and transforms are kept.

        Args:
            slice_id: Filename or corrected index of the section to move.
            after: Filename (or corrected index) of the section it should
                follow, or "start" to put it first.

        Returns:
            The rows it changed, plus the ids whose corrected index changed.
        """
        record = state.resolve(slice_id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [slice_id]}
        target = str(after).strip().lower()
        anchor = None if target == "start" else state.resolve(after)
        if anchor is None and target != "start":
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [after]}
        if anchor is not None and anchor.id == record.id:
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "message": "after names the section itself",
            }

        snapshot()
        ordered = [s for s in state.in_order() if s.id != record.id]
        index = 0 if anchor is None else ordered.index(anchor) + 1
        ordered.insert(index, record)
        moved = renumber(ordered)
        return {"moved": moved, **commit(*moved)}

    if spec.has("reorder"):
        box.tools += [orient_slices, reorder_slices, move_slice]

    # --- position -------------------------------------------------------

    def set_positions(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Write positions for one or more sections, and show each placement.

        Positions are in atlas-native millimetres along the slicing axis. A
        value outside the atlas range is clamped and reported back.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>}]``.

        Returns:
            What was written, what was clamped, the rows it changed, and for
            the first 4 written sections one image each: the section as
            corrected over the atlas section at the position it was given,
            both tissue-framed, labelled in the top-left corner. Sections
            beyond the fourth are written all the same; ``unpictured`` names
            them, and ``compare_placement`` shows any of them on request.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        before = state.to_dict()
        written: list[dict[str, Any]] = []
        clamped: list[dict[str, Any]] = []
        unknown: list[str] = []
        rejected: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                rejected.append({"entry": str(entry), "reason": "not an object"})
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            try:
                requested = float(entry.get("position_mm"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rejected.append({"id": record.id, "reason": "position_mm is not a number"})
                continue
            value = min(pos_hi, max(pos_lo, requested))
            if value != requested:
                clamped.append(
                    {
                        "id": record.id,
                        "requested_mm": round(requested, 3),
                        "clamped_to_mm": round(value, 3),
                    }
                )
            record.position_mm = value
            written.append({"id": record.id, "position_mm": round(value, 3)})

        if not written:
            return {
                "status": "error",
                "error": "NOTHING_WRITTEN",
                "unknown_ids": unknown,
                "rejected": rejected,
            }
        push(before)
        result = {
            "written": written,
            "unknown_ids": unknown,
            "rejected": rejected,
            "clamped": clamped,
            **commit(*[row["id"] for row in written]),
        }
        if clamped:
            result["atlas_range_mm"] = [round(pos_lo, 3), round(pos_hi, 3)]
        shown = written[:MAX_IMAGES_PER_CALL]
        parts: list[types.Part] = []
        failed: list[dict[str, str]] = []
        for row in shown:
            record = state.by_id(row["id"])
            if record is None:
                continue
            position = float(row["position_mm"])
            try:
                picture = stacked(
                    render_slice(ctx, record, long_edge=ATLAS_LONG_EDGE, frame=True),
                    resize_long_edge(
                        atlas_section(ctx, state, position, frame=True), ATLAS_LONG_EDGE
                    ),
                )
                label = f"{record.id} over atlas {position:.2f} mm"
            except Exception as exc:
                failed.append({"id": row["id"], "message": str(exc)})
                continue
            parts.append(image_to_part(caption(picture, label)))
        result["render_failed"] = failed
        result["unpictured"] = [row["id"] for row in written[MAX_IMAGES_PER_CALL:]]
        result["description"] = (
            "Attached: one image per written section, in the order written, "
            "the section as corrected over the atlas section at the position "
            "it was given, for "
            + ", ".join(row["id"] for row in shown)
            + (
                f". {len(result['unpictured'])} more were written without a picture."
                if result["unpictured"]
                else "."
            )
        )
        result[TOOL_MEDIA_PARTS_KEY] = parts
        return result

    def run_deepslice(
        slice_ids: list[str], allow_angle_change: bool, keep: list[str]
    ) -> dict[str, Any]:
        """Seed positions (and optionally angles) with DeepSlice.

        Args:
            slice_ids: Sections to place; empty means every undamaged section.
            allow_angle_change: Whether DeepSlice may set the cutting angles.
            keep: Sections whose current positions must not be overwritten.

        Returns:
            The positions written, or ``UNAVAILABLE`` when DeepSlice is not
            installed or the plane/atlas is unsupported.
        """
        del keep
        return _run_deepslice(
            state,
            ctx,
            slice_ids=[str(item) for item in slice_ids or []],
            allow_angle_change=bool(allow_angle_change),
        )

    def fit_position(slice_id: str, window_mm: float, angles: bool) -> dict[str, Any]:
        """Search the atlas around a section's current position. Writes nothing.

        Scores the section against resampled atlas planes and returns the best
        one it found.

        Args:
            slice_id: Filename or corrected index. The section must already
                have a position.
            window_mm: Half-width of the position search, in millimetres.
            angles: True also searches the cutting angles; False holds them at
                the stack's current ones.

        Returns:
            The best position (and angles) with its score.
        """
        record = state.resolve(slice_id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [slice_id]}
        if record.position_mm is None:
            return {"status": "error", "error": "NO_POSITION", "id": record.id}
        try:
            window = float(window_mm)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}

        from langslice.oblique import fit_oblique

        pitch, yaw = state.pitch_deg, state.yaw_deg
        bounds = ((-15.0, 15.0), (-15.0, 15.0)) if angles else ((pitch, pitch), (yaw, yaw))
        section = render_slice(ctx, record, long_edge=512)
        try:
            fit = fit_oblique(
                ctx.atlas,
                section,
                record.position_mm,
                cast(Plane, state.plane),
                pitch_bounds=bounds[0],
                yaw_bounds=bounds[1],
                position_window_mm=max(0.0, window),
                allow_mirror=False,
            )
        except Exception as exc:
            logger.warning("fit_position failed for %s: %s", record.id, exc)
            return {"status": "error", "error": "FIT_FAILED", "message": str(exc)}
        return {
            "status": "ok",
            "id": record.id,
            "current_position_mm": round(record.position_mm, 3),
            "position_mm": round(float(fit["position_mm"]), 3),
            "pitch_deg": round(float(fit["pitch_deg"]), 3),
            "yaw_deg": round(float(fit["yaw_deg"]), 3),
            "score": round(float(fit["score"]), 4),
            "searched_window_mm": round(max(0.0, window), 3),
            "searched_angles": bool(angles),
        }

    def compare_placement(
        entries: list[dict[str, Any]],
        mode: str = "side_by_side",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        template_opacity: float = 0.0,
        outlines: str = "all",
    ) -> dict[str, Any]:
        """Show sections against the atlas at candidate positions. Writes nothing.

        Each section is drawn as corrected, at true physical scale, on the
        same canvas as the atlas section at each of its candidate positions,
        so the two are directly comparable; the atlas outlines are drawn over
        the section. At most 4 section-position pairs per call, one image
        per pair.

        Args:
            entries: ``[{"id": "<filename or corrected index>",
                "positions_mm": [<mm>, ...]}]``. An empty or missing
                ``positions_mm`` means that section's current position.
            mode: "side_by_side" (the section beside the atlas template in
                one image, same scale and crop), "overlay" (the
                section with the outlines on it), "checkerboard" (section and
                template in alternating tiles), "outlines" (atlas lines and the
                section's silhouette on black), "section" or "template".
            zoom: [x0, y0, x1, y1] as fractions of the canvas; empty is all.
            template_opacity: 0..1, the atlas template blended under the
                outlines in "overlay".
            outlines: "all" (every family boundary), "outer" (the atlas
                outline only) or "none".

        Returns:
            The section-position pairs compared, in order, each with its
            calibration, and the images per pair in that order.
        """
        view = str(mode or "side_by_side").strip().lower()
        if view not in VIEW_MODES:
            return {"status": "error", "error": "BAD_MODE", "modes": list(VIEW_MODES)}
        layer = str(outlines or "all").strip().lower()
        if layer not in OUTLINE_LAYERS:
            return {"status": "error", "error": "BAD_OUTLINES", "layers": list(OUTLINE_LAYERS)}
        try:
            window = [float(value) for value in (zoom or [])]
            opacity = float(template_opacity)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        if window and len(window) != 4:
            return {"status": "error", "error": "BAD_ZOOM",
                    "expected": "[x0, y0, x1, y1] as fractions of the canvas"}

        # Resolve every pair first, so the errors name every problem at once.
        pairs: list[tuple[SliceState, float]] = []
        unknown: list[str] = []
        unplaced: list[str] = []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            try:
                wanted = [float(value) for value in (entry.get("positions_mm") or [])]
            except (TypeError, ValueError):
                return {"status": "error", "error": "BAD_ARGS", "id": record.id}
            if not wanted:
                if record.position_mm is None:
                    unplaced.append(record.id)
                    continue
                wanted = [float(record.position_mm)]
            pairs.extend((record, min(pos_hi, max(pos_lo, value))) for value in wanted)
        if not pairs:
            return {
                "status": "error",
                "error": (
                    "UNKNOWN_SLICE_IDS" if unknown
                    else "NO_POSITION" if unplaced
                    else "BAD_ARGS"
                ),
                "unknown": unknown,
                "no_position": unplaced,
            }
        dropped = len(pairs) - MAX_VIEW_SLICES
        pairs = pairs[:MAX_VIEW_SLICES]

        sections: dict[str, tuple[Any, float, str]] = {}
        compared: list[dict[str, Any]] = []
        parts: list[types.Part] = []
        failed: list[dict[str, Any]] = []
        for record, position in pairs:
            if record.id not in sections:
                section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
                sections[record.id] = (section, *calibrate(state, ctx, record, section))
            section, um_per_px, source = sections[record.id]
            try:
                images, _ = physical_views(
                    section, um_per_px, ctx.atlas, position, cast(Plane, state.plane),
                    state.pitch_deg, state.yaw_deg, dict(IDENTITY_PARAMS),
                    mode=view, zoom=window, template_opacity=opacity, outlines=layer,
                    label=f"{record.id} vs atlas {position:.2f} mm",
                    long_edge=VIEW_LONG_EDGE,  # 512 px panels, ~260 tokens each
                )
            except Exception as exc:
                failed.append({"id": record.id, "position_mm": round(position, 3),
                               "message": str(exc)})
                continue
            if len(images) > 1:
                images = [beside(images[0], images[1])]
            parts.extend(image_to_part(image) for image in images)
            compared.append({
                "id": record.id,
                "position_mm": round(position, 3),
                "current_position_mm": (
                    None if record.position_mm is None else round(record.position_mm, 3)
                ),
                "calibration": {"um_per_px": round(um_per_px, 3), "source": source},
            })
        result: dict[str, Any] = {
            "status": "ok" if parts else "error",
            **({} if parts else {"error": "RENDER_FAILED"}),
            "compared": compared,
            "unknown_ids": unknown,
            "no_position": unplaced,
            "view": {"mode": view, "zoom": window or [0.0, 0.0, 1.0, 1.0], "outlines": layer},
            "render_failed": failed,
            "description": (
                "Compared, in order: "
                + ", ".join(f"{row['id']} at {row['position_mm']:.2f} mm" for row in compared)
                + "; one image per pair in that order, each captioned with "
                "the section and the position it is compared with."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        if dropped > 0:
            result["truncated"] = True
            result["dropped_pairs"] = dropped
        return result

    def view_stack() -> dict[str, Any]:
        """The whole stack ordered by written position, each over its atlas match.

        One contact sheet: every section as a labelled thumbnail, in the
        order of the positions written so far (unplaced sections last), a
        placed section with the atlas section at its position pasted
        directly beneath it. The label carries the corrected index,
        filename, position and the signed distance to the next placed
        section. Then one plot of position against corrected index (damaged
        sections in red). Two images; `view_slices` shows any section large.

        Returns:
            The rows in that order and the two images.
        """
        def atlas_under(record: SliceState) -> Any:
            if record.position_mm is None:
                return None
            try:
                return resize_long_edge(
                    atlas_section(ctx, state, float(record.position_mm), frame=True),
                    ATLAS_LONG_EDGE,
                )
            except Exception as exc:
                logger.warning("view_stack: atlas render failed for %s: %s", record.id, exc)
                return None

        parts = [
            types.Part.from_text(
                text=(
                    f"The {len(state.slices)} sections in the order of their "
                    "written positions (unplaced last), each captioned "
                    "'<index>: <filename>  <position>', over its atlas match:"
                )
            ),
            image_to_part(stack_sheet(state, ctx, under=atlas_under)),
            types.Part.from_text(text="Position against corrected index:"),
            image_to_part(spacing_plot(state)),
        ]
        ordered = sorted(
            status_rows(state), key=lambda r: (r["position_mm"] is None, r["position_mm"] or 0.0)
        )
        return {
            "status": "ok",
            "rows": compact_rows(ordered),
            "description": (
                "Attached: one contact sheet of every section in the order of "
                "its written position, each captioned, with the atlas section "
                "at its position beneath it when it has one; then a plot of "
                "position against corrected index."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    if spec.has("position"):
        box.tools += [set_positions, compare_placement, view_stack]
        if spec.position.deepslice:
            box.tools.append(run_deepslice)
        if spec.position.bayesian:
            box.tools.append(fit_position)

    # --- transform ------------------------------------------------------

    def set_cutting_angles(pitch_deg: float, yaw_deg: float) -> dict[str, Any]:
        """Set the stack-wide cutting angles.

        Subsequent atlas fetches and previews are rendered at these angles.

        Args:
            pitch_deg: Rotation about the plane's column axis, in degrees.
            yaw_deg: Rotation about the plane's row axis, in degrees.

        Returns:
            The rows this call changed.
        """
        try:
            pitch, yaw = float(pitch_deg), float(yaw_deg)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        state.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}
        ctx.render_cache.clear()
        return commit()  # stack-wide: no section row changes

    def fit_affine(slice_ids: list[str], method: str) -> dict[str, Any]:
        """Fit an in-plane affine per section against its atlas section.

        The whole tissue outline is matched against the whole atlas outline;
        a damaged section is refused. Each fit is written as the section's
        transform (undoable, and `adjust_transform` overwrites it).

        Args:
            slice_ids: Filenames or corrected indices; empty means every
                positioned, undamaged section.
            method: "silhouette" (moments fit) or "elastix" (intensity affine).

        Returns:
            Per-section overlap (iou), the transform as the five physical
            knobs about the canvas centre, the calibration the image was
            drawn with, the rows written, and an image of the fitted section
            under the atlas outlines for up to 16 sections.
        """
        # ponytail: spec.transform.elastix is inert until the method lands;
        # asking for it answers UNAVAILABLE either way.
        chosen = str(method or "silhouette").strip().lower()
        if chosen == "elastix":
            return {
                "status": "error",
                "error": "UNAVAILABLE",
                "message": "The elastix affine method is not wired in this build.",
            }
        if chosen != "silhouette":
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "message": "method must be 'silhouette' or 'elastix'.",
            }

        if slice_ids:
            targets, unknown = resolve_many(list(slice_ids))
        else:
            targets = [
                record
                for record in state.in_order()
                if record.position_mm is not None and not record.damaged
            ]
            unknown = []

        results: list[dict[str, Any]] = []
        parts: list[types.Part] = []
        fits: list[tuple[SliceState, dict[str, Any]]] = []
        for record in targets:
            if record.damaged:
                results.append({"id": record.id, "status": "error", "error": "DAMAGED"})
                continue
            outcome = fit_silhouette(state, ctx, record)
            panel = outcome.pop("panel", None)
            results.append(outcome)
            if outcome["status"] != "ok":
                continue
            fits.append((record, outcome))
            if panel is not None and len(parts) < MAX_FIT_PANELS:
                parts.append(image_to_part(panel))

        payload: dict[str, Any] = {
            "status": "ok" if fits else "error",
            "results": results,
            "unknown_ids": unknown,
        }
        if not fits:
            payload["error"] = "NOTHING_FITTED"
            return payload
        if parts:
            payload["description"] = (
                "Attached panels are "
                + ", ".join(record.id for record, _ in fits[: len(parts)])
                + ", in that order; each shows the section under its fitted "
                "transform with the atlas region outlines over it at true "
                "physical scale."
            )
            payload[TOOL_MEDIA_PARTS_KEY] = parts
        snapshot()
        for record, outcome in fits:
            record.transform = {
                "kind": "silhouette",
                "params": outcome["params"],
                "physical": outcome["physical"],
                "iou": outcome["iou"],
                "calibration": outcome["calibration"],
                "mirrored": outcome["mirrored"],
            }
        payload.update(commit(*[record.id for record, _ in fits]))
        return payload

    # --- the interactive transform --------------------------------------

    def stage(
        slice_id: str,
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
        pivot: Any,
    ) -> _Staged | dict[str, Any]:
        """One section, its calibrated canvas and the pivot, or a refusal."""
        record = state.resolve(slice_id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [slice_id]}
        if record.position_mm is None:
            return {"status": "error", "error": "NO_POSITION", "id": record.id}
        try:
            params = {
                "rotation_deg": float(rotation_deg),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "translate_x_mm": float(translate_x_mm),
                "translate_y_mm": float(translate_y_mm),
            }
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
        um_per_px, source = calibrate(state, ctx, record, section)
        try:
            geometry = canvas_geometry(
                section.size,
                um_per_px,
                ctx.atlas,
                record.position_mm,
                cast(Plane, state.plane),
                state.pitch_deg,
                state.yaw_deg,
            )
        except Exception as exc:
            logger.warning("transform: atlas render failed for %s: %s", record.id, exc)
            return {
                "status": "error",
                "error": "ATLAS_RENDER_FAILED",
                "message": str(exc),
            }
        try:
            point = pivot_on_canvas(pivot, section, geometry)
        except ValueError as exc:
            return {"status": "error", "error": "BAD_PIVOT", "message": str(exc)}
        width, height = geometry.size
        centre = point or (width / 2.0, height / 2.0)
        mode = "canvas"
        if isinstance(pivot, str):
            mode = (pivot.strip().lower() or "canvas")
        elif pivot:
            mode = "fractions"
        return _Staged(
            record=record,
            section=section,
            um_per_px=um_per_px,
            calibration_source=source,
            geometry=geometry,
            params=params,
            pivot=point,
            pivot_frac=[round(centre[0] / width, 4), round(centre[1] / height, 4)],
            pivot_mode=mode,
        )

    def views(
        staged: _Staged,
        params: dict[str, float],
        *,
        mode: str,
        zoom: list[float],
        template_opacity: float,
        pivot: tuple[float, float] | None,
        outlines: str = "all",
        markers: Any = None,
        label: str = "",
    ) -> list[Any]:
        images, _iou = physical_views(
            staged.section,
            staged.um_per_px,
            ctx.atlas,
            float(staged.record.position_mm or 0.0),
            cast(Plane, state.plane),
            state.pitch_deg,
            state.yaw_deg,
            params,
            mode=mode,
            zoom=zoom,
            template_opacity=template_opacity,
            outlines=outlines,
            pivot=pivot,
            markers=markers,
            label=label or staged.record.id,
            long_edge=OVERLAY_LONG_EDGE,
        )
        return images

    def adjust_transform(
        slice_id: str,
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
        mode: str = "overlay",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        template_opacity: float = 0.0,
        pivot: str | list[float] = "canvas",
        outlines: str = "all",
        note: str = "",
    ) -> dict[str, Any]:
        """Set one section's in-plane transform and show the result.

        Every call writes the parameters as the section's transform and
        returns the section drawn under them with the atlas outlines. Call it
        as often as you need, on any section that has a position; the last
        call is what stays. The same parameters again only re-draws. The
        section's flip and rotation flags are not touched. Writes,
        checkpoints, and can be undone.

        Args:
            slice_id: Filename or corrected index.
            rotation_deg: Counter-clockwise rotation about the pivot, in
                degrees. Negative turns it clockwise.
            scale_x: Horizontal scale multiplier about the pivot. 1.0 leaves
                the width alone.
            scale_y: Vertical scale multiplier, same convention.
            translate_x_mm: Horizontal shift in millimetres; positive moves
                right.
            translate_y_mm: Vertical shift in millimetres; positive moves down.
            mode: "overlay" (the section with the outlines on it),
                "side_by_side" (two images: the section, then the atlas
                template, same scale and same crop), "checkerboard" (section
                and template in alternating tiles), "outlines" (the atlas
                lines and the section's own silhouette contour on black),
                "section" (the section alone, no lines), "template" (the atlas
                template alone) or "ab" (two overlays at the same crop: these
                parameters, then the section's stored transform, or identity
                when it has none).
            zoom: [x0, y0, x1, y1] as fractions of the CANVAS, cropped before
                the image is sized down, so it is real magnification. An empty
                list is the whole canvas.
            template_opacity: 0..1, how strongly the atlas template is blended
                under the outlines in "overlay". 0.0 draws no template.
            pivot: What the rotation and the scales turn about: "canvas" (the
                canvas centre), "tissue" (the section's tissue centroid) or
                [fx, fy] fractions of the canvas.
            outlines: Which atlas lines to draw: "all" (every family
                boundary), "outer" (the atlas outline only) or "none".
            note: Short remark for the record. May be empty.

        Returns:
            The parameters written, their decomposition (rotation, scales,
            shear, mirrored), the shift in canvas pixels, the pivot used, the
            calibration, every parameter set this section has been given so
            far, the row this call changed, and the image(s) of the view you
            asked for.
        """
        view = str(mode or "overlay").strip().lower()
        if view not in PREVIEW_MODES:
            return {"status": "error", "error": "BAD_MODE", "modes": list(PREVIEW_MODES)}
        layer = str(outlines or "all").strip().lower()
        if layer not in OUTLINE_LAYERS:
            return {
                "status": "error",
                "error": "BAD_OUTLINES",
                "layers": list(OUTLINE_LAYERS),
            }
        try:
            window = [float(value) for value in (zoom or [])]
            opacity = float(template_opacity)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        if window and len(window) != 4:
            return {
                "status": "error",
                "error": "BAD_ZOOM",
                "expected": "[x0, y0, x1, y1] as fractions of the canvas",
            }
        staged = stage(
            slice_id, rotation_deg, scale_x, scale_y, translate_x_mm,
            translate_y_mm, pivot,
        )
        if isinstance(staged, dict):
            return staged
        record = staged.record
        previous = record.transform
        six = normalized_physical_affine(
            size=staged.section.size,
            um_per_px=staged.um_per_px,
            pivot=staged.pivot_in_section,
            **staged.params,
        )
        decomposition = physical_decomposition(six, staged.section.size)
        written = {
            "kind": "interactive",
            "params": six,
            "physical": {**staged.params, "pivot": staged.pivot_frac},
            "calibration": staged.calibration,
            "mirrored": decomposition["mirrored"],
            "note": str(note or "").strip(),
        }
        # The same numbers again are a look, not a write: no undo step for it.
        wrote = previous is None or {**previous, "note": ""} != {**written, "note": ""}
        if wrote:
            snapshot()
            record.transform = written

        stored = (previous or {}).get("physical")
        reference: dict[str, Any] | None = None
        try:
            if view == "ab":
                # The B side is drawn from the six numbers the section carried
                # before this call, when there were any: they are the exact
                # map, shear included, and the five knobs the payload reports
                # cannot carry that shear.
                before = (previous or {}).get("params")
                other: Any = dict(IDENTITY_PARAMS)
                if before is not None and len(before) == 6:
                    other = denormalized_affine(before, staged.section.size)
                elif isinstance(stored, dict):
                    other = {key: float(stored[key]) for key in IDENTITY_PARAMS}
                other_pivot = staged.pivot
                if isinstance(stored, dict) and stored.get("pivot"):
                    fractions = [float(value) for value in stored["pivot"]]
                    other_pivot = (
                        fractions[0] * staged.geometry.size[0],
                        fractions[1] * staged.geometry.size[1],
                    )
                held = previous is not None
                images = views(
                    staged, staged.params, mode="overlay", zoom=window,
                    template_opacity=opacity, pivot=staged.pivot, outlines=layer,
                    label=f"{record.id} candidate",
                ) + views(
                    staged, other, mode="overlay", zoom=window,
                    template_opacity=opacity, pivot=other_pivot, outlines=layer,
                    label=f"{record.id} {'stored' if held else 'identity'}",
                )
                reference = {
                    "source": "stored" if held else "identity",
                    "params": dict(stored) if isinstance(stored, dict) else dict(IDENTITY_PARAMS),
                }
                if held:
                    reference["stored_kind"] = (previous or {}).get("kind")
            else:
                images = views(
                    staged, staged.params, mode=view, zoom=window,
                    template_opacity=opacity, pivot=staged.pivot, outlines=layer,
                )
        except Exception as exc:
            logger.warning("adjust_transform failed for %s: %s", record.id, exc)
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}

        history = box.transform_history.setdefault(record.id, [])
        history.append(dict(staged.params))
        px_per_mm = 1000.0 / staged.um_per_px
        described = {
            "overlay": "the section with the atlas outlines over it",
            "side_by_side": (
                "two images at the same scale and the same crop: the section, "
                "then the atlas template"
            ),
            "checkerboard": "the section and the atlas template in alternating tiles",
            "outlines": (
                "the atlas outlines and the section's own silhouette contour, "
                "on black"
            ),
            "section": "the section alone, no outlines",
            "template": "the atlas template alone, no outlines",
            "ab": (
                "two overlays at the same crop: first these parameters, then "
                + (
                    "the transform the section carried before this call"
                    if previous is not None
                    else "identity"
                )
            ),
        }[view]
        return {
            "id": record.id,
            "position_mm": round(float(record.position_mm or 0.0), 3),
            "params": staged.params,
            "matrix_params": six,
            "decomposition": decomposition,
            "translate_px": {
                "x": round(staged.params["translate_x_mm"] * px_per_mm, 1),
                "y": round(staged.params["translate_y_mm"] * px_per_mm, 1),
                "px_per_mm": round(px_per_mm, 2),
            },
            "pivot": {"mode": staged.pivot_mode, "canvas_frac": staged.pivot_frac},
            "view": {
                "mode": view,
                "zoom": window or [0.0, 0.0, 1.0, 1.0],
                "outlines": layer,
            },
            **({"ab_reference": reference} if reference is not None else {}),
            "history": [dict(entry) for entry in history],
            "calibration": staged.calibration,
            **(commit(record.id) if wrote else {"status": "ok", **changed([])}),
            "description": (
                f"{record.id} under the transform above, {described}. "
                + (
                    "No atlas outlines are drawn."
                    if layer == "none"
                    else (
                        ("The outline is the OUTER boundary of "
                         if layer == "outer"
                         else "The outlines are the FAMILY regions of ")
                        + f"the {state.plane} atlas section at "
                        f"{float(record.position_mm or 0.0):.3f} mm, drawn at "
                        "true physical scale as neutral hairlines."
                    )
                )
            ),
            TOOL_MEDIA_PARTS_KEY: [image_to_part(image) for image in images],
        }

    def landmarks(
        slice_id: str,
        pairs: list[dict[str, list[float]]],
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
        mode: str = "overlay",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        template_opacity: float = 0.0,
        pivot: str | list[float] = "canvas",
    ) -> dict[str, Any]:
        """Measure point pairs under a transform, and fit one to them.

        Each pair is a point on the section and the point of the atlas it
        belongs on, both as fractions of the CANVAS as the last preview drew
        it. Nothing is written.

        Args:
            slice_id: Filename or corrected index.
            pairs: ``[{"section": [fx, fy], "atlas": [fx, fy]}]``, fractions of
                the canvas.
            rotation_deg: Rotation of the transform to measure under.
            scale_x: Horizontal scale of that transform.
            scale_y: Vertical scale of that transform.
            translate_x_mm: Horizontal shift of that transform, in millimetres.
            translate_y_mm: Vertical shift of that transform, in millimetres.
            mode: Any `adjust_transform` view except "ab".
            zoom: [x0, y0, x1, y1] as fractions of the canvas; empty is all.
            template_opacity: 0..1, the template under the outlines in
                "overlay".
            pivot: "canvas", "tissue" or [fx, fy] — the point the fitted
                rotation and scales are reported about, as in
                `adjust_transform`.

        Returns:
            Per pair the distance in millimetres between the section point
            under the given transform and its atlas point, the RMS of those
            distances, the transform fitted to the pairs (a similarity from 2
            points, a full affine from 3) in the same units, and an image with
            the pairs drawn on the current view.
        """
        view = str(mode or "overlay").strip().lower()
        if view not in VIEW_MODES:
            return {"status": "error", "error": "BAD_MODE", "modes": list(VIEW_MODES)}
        section_points: list[list[float]] = []
        atlas_points: list[list[float]] = []
        for pair in pairs or []:
            if not isinstance(pair, dict):
                continue
            try:
                here = [float(value) for value in pair["section"]]
                there = [float(value) for value in pair["atlas"]]
            except (KeyError, TypeError, ValueError):
                continue
            if len(here) == 2 and len(there) == 2:
                section_points.append(here)
                atlas_points.append(there)
        if not section_points:
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "message": (
                    'pairs must be [{"section": [fx, fy], "atlas": [fx, fy]}] '
                    "with fractions of the canvas."
                ),
            }
        try:
            window = [float(value) for value in (zoom or [])]
            opacity = float(template_opacity)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        staged = stage(
            slice_id, rotation_deg, scale_x, scale_y, translate_x_mm,
            translate_y_mm, pivot,
        )
        if isinstance(staged, dict):
            return staged

        width, height = staged.geometry.size
        scale = np.array([width, height], dtype=np.float64)
        source = np.asarray(section_points, dtype=np.float64) * scale
        target = np.asarray(atlas_points, dtype=np.float64) * scale
        matrix = np.asarray(staged.matrix(), dtype=np.float64)
        mapped = source @ matrix[:, :2].T + matrix[:, 2]
        distances = np.linalg.norm(mapped - target, axis=1) * staged.um_per_px / 1000.0

        centre = staged.pivot or (width / 2.0, height / 2.0)
        fitted: dict[str, Any] | None = None
        if len(source) >= 2:
            estimator = affine_fit if len(source) >= 3 else similarity_fit
            estimate = estimator(source, target)
            residual = np.linalg.norm(
                source @ estimate[:, :2].T + estimate[:, 2] - target, axis=1
            ) * staged.um_per_px / 1000.0
            fitted = {
                "kind": "affine" if len(source) >= 3 else "similarity",
                "points": len(source),
                **physical_params(estimate, pivot=centre, um_per_px=staged.um_per_px),
                "rms_mm": round(float(np.sqrt((residual**2).mean())), 4),
            }

        try:
            images = views(
                staged, staged.params, mode=view, zoom=window,
                template_opacity=opacity, pivot=staged.pivot,
                markers=(source, target),
            )
        except Exception as exc:
            logger.warning("landmarks: render failed for %s: %s", staged.record.id, exc)
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}

        return {
            "status": "ok",
            "id": staged.record.id,
            "params": staged.params,
            "pivot": {"mode": staged.pivot_mode, "canvas_frac": staged.pivot_frac},
            "pairs": [
                {
                    "index": index + 1,
                    "section": section_points[index],
                    "atlas": atlas_points[index],
                    "residual_mm": round(float(distances[index]), 4),
                }
                for index in range(len(section_points))
            ],
            "rms_mm": round(float(np.sqrt((distances**2).mean())), 4),
            "fit": fitted,
            "view": {"mode": view, "zoom": window or [0.0, 0.0, 1.0, 1.0]},
            "calibration": staged.calibration,
            "description": (
                f"{staged.record.id} under the transform above, with the "
                f"{len(section_points)} pair(s) drawn: a cross and its number "
                "on each section point, a ring on each atlas point, a line "
                "between them."
            ),
            TOOL_MEDIA_PARTS_KEY: [image_to_part(image) for image in images],
        }

    if spec.has("transform"):
        box.tools += [fit_affine, adjust_transform, landmarks]
        if spec.transform.angles:
            box.tools.append(set_cutting_angles)

    box.tools.append(submit)
    return box
