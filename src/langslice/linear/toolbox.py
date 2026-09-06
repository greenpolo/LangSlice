"""The one toolbox: every tool the linear agent can be given, gated by the spec.

Conventions, applied to every tool: sections are addressed by filename or
corrected index; every write returns the same status rows; every write is
undoable; every write checkpoints; fits have a form that computes without
writing.

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

from google.genai import types

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.linear.atlas_fetch import make_fetch_atlas
from langslice.linear.checkpoint import save_checkpoint
from langslice.linear.deepslice import run_deepslice as _run_deepslice
from langslice.linear.render import (
    VIEW_LONG_EDGE,
    image_to_part,
    render_slice,
    status_rows,
)
from langslice.linear.signals import interpolate_positions
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState, apply_confidence
from langslice.linear.transform import fit_silhouette, run_align_session
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

#: Sections one ``view_slices`` call may return, and panels one ``fit_affine``
#: call may attach.
MAX_VIEW_SLICES = 8

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


@dataclass
class ToolBox:
    """The tools of one run plus the mutable results the engine reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)

    @property
    def names(self) -> list[str]:
        return [tool.__name__ for tool in self.tools]


def _longest_increasing_run(values: list[int]) -> set[int]:
    """Positions of one longest strictly increasing subsequence of *values*.

    O(n^2) patience-free DP; stacks are tens of sections.
    """
    n = len(values)
    if n == 0:
        return set()
    best = [1] * n
    prev = [-1] * n
    for i in range(n):
        for j in range(i):
            if values[j] < values[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
                prev[i] = j
    end = max(range(n), key=lambda i: best[i])
    keep: set[int] = set()
    while end != -1:
        keep.add(end)
        end = prev[end]
    return keep


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


def submit_errors(
    state: StackState, spec: JobSpec, breaks: list[int]
) -> dict[str, Any] | None:
    """Every gate that applies to this run, in order."""
    if not spec.has("position"):
        return None
    return (
        missing_positions(state)
        or order_position_mismatch(state)
        or (
            strict_interval_error(state, breaks)
            if spec.position.strict_interval
            else unsupported_breaks(state, breaks)
        )
    )


# --- the toolbox ---------------------------------------------------------


def build_tools(state: StackState, ctx: EngineContext, spec: JobSpec) -> ToolBox:
    """Build the tools this run's spec switches on, closed over *state*."""
    box = ToolBox()
    pos_lo, pos_hi = ctx.position_range

    # --- shared plumbing ------------------------------------------------

    def rows() -> dict[str, Any]:
        return {
            "rows": status_rows(state),
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

    def commit() -> dict[str, Any]:
        save_checkpoint(state, ctx.checkpoint_path)
        return {"status": "ok", **rows()}

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
        """Apply a new corrected order; return the ids that moved.

        "Moved" is the smallest set of sections whose removal leaves the rest
        in their old relative order (the complement of the longest increasing
        run of old indices) — so moving one section past twenty others moves
        ONE section, not twenty-one. A moved section loses its position and
        its transform: both were decided under different neighbours.
        """
        old = [record.index_corrected for record in order]
        keep = _longest_increasing_run(old)
        moved: list[str] = []
        for index, record in enumerate(order):
            record.index_corrected = index
            if index in keep:
                continue
            record.position_mm = None
            record.transform = None
            moved.append(record.id)
        return moved

    # --- always on ------------------------------------------------------

    def status() -> dict[str, Any]:
        """The stack as it stands: one row per section in corrected order.

        Returns:
            ``rows`` (index, id, position_mm, spacing_to_next_mm, flip,
            rotation_deg, damaged, damage_note, transform kind, confidence,
            caveats), plus the stack's cutting angles and interval breaks.
        """
        return {"status": "ok", **rows()}

    def view_slices(slice_ids: list[str]) -> dict[str, Any]:
        """Look at up to 8 named sections at higher resolution.

        Sections are rendered as corrected: any rotation and flip already
        applied, framed to their tissue the same way fetched atlas sections
        are.

        Args:
            slice_ids: Filenames or corrected indices (max 8 per call).

        Returns:
            status/slice_ids/description plus the images, in the order asked.
        """
        if not slice_ids:
            return {"status": "error", "error": "BAD_ARGS"}
        known, unknown = resolve_many(list(slice_ids)[:MAX_VIEW_SLICES])
        if not known:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        parts = [
            image_to_part(
                render_slice(ctx, record, long_edge=VIEW_LONG_EDGE, frame=True)
            )
            for record in known
        ]
        return {
            "status": "ok",
            "slice_ids": [record.id for record in known],
            "unknown_ids": unknown,
            # The attached images are unlabelled, so this ordering note is the
            # only way the model can tie an image back to a filename.
            "description": (
                "Attached images are "
                + ", ".join(record.id for record in known)
                + ", in that order, rendered as corrected."
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
            The status rows, as they stand after the write.
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
        return {"marked": marked, "unknown_ids": unknown, **commit()}

    def unmark_damaged(slice_ids: list[str]) -> dict[str, Any]:
        """Clear the damaged flag on the named sections.

        Args:
            slice_ids: Filenames or corrected indices.

        Returns:
            The status rows, as they stand after the write.
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
            **commit(),
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

    box.tools = [
        status,
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
        """Set the flip and rotation of one or more sections.

        Corrections are recorded as data; the user's image files are never
        modified. Rotation is applied first, then the flip.

        Args:
            entries: ``[{"id": "<filename>", "flip": true|false,
                "rotate_deg": 0|90|180|270}]``. Either key may be omitted to
                leave that correction as it is.

        Returns:
            The status rows, as they stand after the write.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        applied: list[str] = []
        unknown: list[str] = []
        rejected: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
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
            applied.append(record.id)
        return {
            "applied": applied,
            "unknown_ids": unknown,
            "rejected": rejected,
            **commit(),
        }

    def reorder_slices(new_order: list[str]) -> dict[str, Any]:
        """Set the corrected order of the WHOLE stack in one call.

        A section that moves loses its position and its transform; the rows
        show which.

        Args:
            new_order: Every section filename exactly once, in the order the
                sections were cut.

        Returns:
            The status rows plus the ids that moved.
        """
        ids = [s.id for s in state.slices]
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
        return {"moved": moved, "cleared_positions": moved, **commit()}

    def move_slice(slice_id: str, after: str) -> dict[str, Any]:
        """Move one section to a new place in the corrected order.

        The moved section loses its position and its transform.

        Args:
            slice_id: Filename or corrected index of the section to move.
            after: Filename (or corrected index) of the section it should
                follow, or "start" to put it first.

        Returns:
            The status rows plus the ids that moved.
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
        return {"moved": moved, "cleared_positions": moved, **commit()}

    if spec.has("reorder"):
        box.tools += [orient_slices, reorder_slices, move_slice]

    # --- position -------------------------------------------------------

    def set_positions(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Write positions for one or more sections. Batch: one call, many.

        Positions are in atlas-native millimetres along the slicing axis. A
        value outside the atlas range is clamped and reported back.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>,
                "confidence": "low"|"medium"|"high"}]``; confidence optional.

        Returns:
            What was written, what was clamped, and the status rows.
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
            apply_confidence(record, entry.get("confidence"))
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
            **commit(),
        }
        if clamped:
            result["atlas_range_mm"] = [round(pos_lo, 3), round(pos_hi, 3)]
        return result

    def distribute_spacing(
        fixed: list[dict[str, Any]], keep: list[str], apply: bool
    ) -> dict[str, Any]:
        """Spread positions over the stack from the points you fix.

        Between two anchors the spacing is spread evenly; beyond the outermost
        anchors it steps at the interval those anchors imply. Sections named in
        *keep* hold their current position and act as anchors too.

        Args:
            fixed: ``[{"id": "<filename>", "position_mm": <number>}]``.
            keep: Filenames whose current positions must not move. Pass an
                empty list for none.
            apply: False computes and returns rows without writing; True
                writes the result.

        Returns:
            One position per section, each marked ``fixed``, ``kept`` or
            ``interpolated``, plus the interval used beyond the anchors.
        """
        ordered = state.in_order()
        if not ordered:
            return {"status": "error", "error": "EMPTY_STACK"}
        index_of = {record.id: index for index, record in enumerate(ordered)}
        known: list[float | None] = [None] * len(ordered)
        source = ["interpolated"] * len(ordered)
        unknown: list[str] = []
        rejected: list[dict[str, Any]] = []

        for slice_id in keep or []:
            record = state.resolve(slice_id)
            if record is None:
                unknown.append(str(slice_id))
                continue
            if record.position_mm is None:
                rejected.append({"id": record.id, "reason": "kept section has no position"})
                continue
            known[index_of[record.id]] = float(record.position_mm)
            source[index_of[record.id]] = "kept"

        for entry in fixed or []:
            if not isinstance(entry, dict):
                rejected.append({"entry": str(entry), "reason": "not an object"})
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            try:
                known[index_of[record.id]] = float(entry.get("position_mm"))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                rejected.append({"id": record.id, "reason": "position_mm is not a number"})
                continue
            source[index_of[record.id]] = "fixed"

        anchors = [i for i, value in enumerate(known) if value is not None]
        if len(anchors) < 2:
            return {
                "status": "error",
                "error": "NEED_TWO_ANCHORS",
                "anchors": len(anchors),
                "unknown_ids": unknown,
                "rejected": rejected,
                "message": (
                    "distribute_spacing needs at least two anchors between "
                    "`fixed` and `keep`."
                ),
            }
        first, last = anchors[0], anchors[-1]
        step = (float(known[last]) - float(known[first])) / (last - first)  # type: ignore[arg-type]
        values = interpolate_positions(known, interval_mm=step)
        suggestions = [
            {
                "id": record.id,
                "position_mm": round(min(pos_hi, max(pos_lo, value)), 3),
                "source": source[index],
            }
            for index, (record, value) in enumerate(
                zip(ordered, values, strict=True)
            )
        ]
        result: dict[str, Any] = {
            "status": "ok",
            "suggestions": suggestions,
            "implied_interval_mm": round(abs(step), 3),
            "unknown_ids": unknown,
            "rejected": rejected,
            "applied": bool(apply),
        }
        if not apply:
            return result
        snapshot()
        for record, row in zip(ordered, suggestions, strict=True):
            record.position_mm = float(row["position_mm"])
        result.update(commit())
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

    if spec.has("position"):
        box.tools += [set_positions, distribute_spacing]
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
            The status rows, as they stand after the write.
        """
        try:
            pitch, yaw = float(pitch_deg), float(yaw_deg)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        state.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}
        ctx.render_cache.clear()
        return commit()

    def fit_affine(slice_ids: list[str], method: str, apply: bool) -> dict[str, Any]:
        """Fit an in-plane affine per section against its atlas section.

        Refuses damaged sections: the fit matches the tissue OUTLINE, which is
        what damage destroys.

        Args:
            slice_ids: Filenames or corrected indices; empty means every
                positioned, undamaged section.
            method: "silhouette" (moments fit) or "elastix" (intensity affine).
            apply: True records the fits on the stack; False only measures.

        Returns:
            Per-section overlap (iou) plus an overlay panel for up to 8
            sections.
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
            if panel is not None and len(parts) < MAX_VIEW_SLICES:
                parts.append(image_to_part(panel))

        payload: dict[str, Any] = {
            "status": "ok" if fits else "error",
            "results": results,
            "unknown_ids": unknown,
            "applied": bool(apply) and bool(fits),
        }
        if not fits:
            payload["error"] = "NOTHING_FITTED"
            return payload
        if parts:
            payload["description"] = (
                "Attached panels are "
                + ", ".join(record.id for record, _ in fits[: len(parts)])
                + ", in that order; each shows the transformed section, its "
                "atlas section, and the two overlaid."
            )
            payload[TOOL_MEDIA_PARTS_KEY] = parts
        if not apply:
            return payload
        snapshot()
        for record, outcome in fits:
            record.transform = {
                "kind": "silhouette",
                "params": outcome["params"],
                "iou": outcome["iou"],
            }
        payload.update(commit())
        return payload

    async def align_slice(slice_id: str, notes: str) -> dict[str, Any]:
        """Align ONE section by eye, in a bounded sub-session, and record it.

        The sub-session previews candidate transforms over the section's atlas
        section and submits the alignment it settles on.

        Args:
            slice_id: Filename or corrected index.
            notes: What the sub-session should know about this section. May be
                empty.

        Returns:
            The parameters it settled on plus the final overlay panel.
        """
        record = state.resolve(slice_id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [slice_id]}
        outcome = await run_align_session(state, ctx, record, str(notes or ""))
        panel = outcome.pop("panel", None)
        if outcome["status"] != "ok":
            return outcome
        snapshot()
        record.transform = {
            "kind": "interactive",
            "params": outcome["matrix_params"],
            "note": outcome["note"],
        }
        apply_confidence(record, outcome.get("confidence"))
        outcome.update(commit())
        if panel is not None:
            outcome["description"] = (
                f"The attached panel is the final alignment of {record.id}: "
                "transformed section, atlas section, and the two overlaid."
            )
            outcome[TOOL_MEDIA_PARTS_KEY] = [image_to_part(panel)]
        return outcome

    def copy_transform(from_id: str, to_ids: list[str]) -> dict[str, Any]:
        """Copy one section's transform onto other sections.

        Args:
            from_id: Filename or corrected index of the section to copy from.
            to_ids: Filenames or corrected indices to copy onto.

        Returns:
            The status rows, as they stand after the write.
        """
        source = state.resolve(from_id)
        if source is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [from_id]}
        if source.transform is None:
            return {"status": "error", "error": "NO_TRANSFORM", "id": source.id}
        targets, unknown = resolve_many(list(to_ids or []))
        targets = [record for record in targets if record.id != source.id]
        if not targets:
            return {"status": "error", "error": "BAD_ARGS", "unknown_ids": unknown}
        snapshot()
        for record in targets:
            record.transform = {**source.transform, "copied_from": source.id}
        return {
            "copied_to": [record.id for record in targets],
            "unknown_ids": unknown,
            **commit(),
        }

    if spec.has("transform"):
        box.tools += [fit_affine, copy_transform]
        if spec.transform.angles:
            box.tools.append(set_cutting_angles)
        if spec.transform.subagents:
            box.tools.append(align_slice)

    box.tools.append(submit)
    return box
