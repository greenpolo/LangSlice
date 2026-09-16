"""The one toolbox: every tool the linear agent can be given, gated by the spec.

Conventions, applied to every tool: sections are addressed by filename or
corrected index; ordinary writes return the rows they changed (``status`` is
the whole table); every write is undoable and checkpoints. Transform writes
return their physical result and omit a generic row that would repeat it.

Tools report data. No advice, no interpretation, no strategy in any payload —
every benchmark failure worth tracing came back to harness text telling the
model what to think. The submit gates are the exception, and a constraint that
states a number is not coaching: refusals name the numbers that caused them and
stop there.
"""

from __future__ import annotations

import functools
import inspect
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from google.genai import types

from langslice.adk import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.affine import (
    denormalized_affine,
    normalized_physical_affine,
    physical_affine_matrix,
)
from langslice.linear.atlas_fetch import (
    ATLAS_LONG_EDGE,
    atlas_part,
    atlas_section,
    atlas_sized,
    make_fetch_atlas,
)
from langslice.linear.checkpoint import save_checkpoint
from langslice.linear.deepslice import run_deepslice as _run_deepslice
from langslice.linear.live import LiveCallback, _plain
from langslice.linear.render import (
    MAX_IMAGES_PER_CALL,
    OUTLINE_LAYERS,
    OVERLAY_LONG_EDGE,
    PREVIEW_LONG_EDGE,
    VIEW_LONG_EDGE,
    VIEW_MODES,
    CanvasGeometry,
    canvas_geometry,
    caption,
    compact_rows,
    image_to_part,
    normalize_border_style,
    physical_views,
    pivot_on_canvas,
    reference_slice_part,
    render_slice,
    spacing_plot,
    stack_sheet,
    stacked,
    status_rows,
)
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState
from langslice.linear.transform import (
    calibrate,
    fit_silhouette,
    physical_decomposition,
)
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

#: Sections one ``view_slices`` call may return, or candidate pairs per compare.
#: Separate reference comparisons return up to two images per candidate pair.
MAX_VIEW_SLICES = MAX_IMAGES_PER_CALL

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

#: Views ``adjust_transforms`` composes on top of the renderer's own
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

def _tool_target_ids(state: StackState, name: str, args: dict[str, Any]) -> list[str]:
    """Resolve host display targets before a tool can reorder the stack."""
    if name in {"view_stack", "status", "undo", "redo", "submit",
                "set_cutting_angles", "run_deepslice"}:
        return [record.id for record in state.in_order()]
    if name == "fit_affine" and not args.get("slice_ids"):
        return [record.id for record in state.in_order()
                if record.position_mm is not None and not record.damaged]
    refs = list(args.get("slice_ids") or args.get("new_order") or [])
    if "slice_id" in args:
        refs.append(args["slice_id"])
    for entry in args.get("entries") or []:
        if isinstance(entry, dict) and "id" in entry:
            refs.append(entry["id"])
    if name == "view_slices":
        refs = refs[:MAX_VIEW_SLICES]
    targets: list[str] = []
    for ref in refs:
        record = state.resolve(ref)
        if record is not None and record.id not in targets:
            targets.append(record.id)
    return targets


def _serialized(
    tool: Any, lock: threading.Lock, *, state: StackState | None = None,
    on_event: LiveCallback | None = None,
) -> Any:
    """Serialize execution and its host notifications under the same lock.

    Model tool-call announcements may arrive together. These optional events
    identify the tool actually executing, including stable filenames resolved
    before a reorder. They never enter model context or change its schema.
    """
    signature = inspect.signature(tool)

    def notify(event: dict[str, Any]) -> None:
        if on_event is not None:
            try:
                on_event(event)
            except Exception:
                logger.warning("Tool execution observer failed", exc_info=True)

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        with lock:
            if on_event is None:
                return tool(*args, **kwargs)
            fields: dict[str, Any] = {"name": tool.__name__, "execution_id": uuid.uuid4().hex}
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                context = arguments.pop("tool_context", None)
                fields.update(
                    id=_plain(getattr(context, "function_call_id", None)),
                    args=_plain(arguments),
                    target_ids=_tool_target_ids(state, tool.__name__, arguments)
                    if state is not None else [],
                )
            except Exception:
                # Bad display metadata must never change the tool's behavior.
                logger.warning("Tool execution metadata unavailable", exc_info=True)
                fields.update(id=None, args={}, target_ids=[])
            notify({"kind": "tool_start", **fields})
            try:
                result = tool(*args, **kwargs)
            except Exception as exc:
                notify({"kind": "tool_end", **fields, "response": {
                    "status": "error", "error": type(exc).__name__, "message": str(exc),
                }})
                raise
            notify({"kind": "tool_end", **fields, "response": _plain(result)})
            return result

    return run


@dataclass
class ToolBox:
    """The tools of one run plus the mutable results the engine reads back."""

    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    undo_stack: list[dict[str, Any]] = field(default_factory=list)
    redo_stack: list[dict[str, Any]] = field(default_factory=list)
    #: Every parameter set `adjust_transforms` was given this run, per section
    #: id, oldest first. Kept host-side; the growing history is not repeated
    #: in every tool result because it is already present in the trajectory.
    transform_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    #: Positions each section was compared at since its last write, and
    #: whether `view_stack` has run since the last write (the gates).
    compared: dict[str, set[float]] = field(default_factory=dict)
    #: Placement pictures successfully produced by a tool call but not yet
    #: carried into a later model request. A compare and a write requested in
    #: the same model turn therefore cannot mistake one another for a view the
    #: model has already received.
    pending_placement_views: dict[
        str, set[tuple[str, float, bool, int, float, float]]
    ] = field(default_factory=dict)
    #: Placement pictures delivered in an earlier model request. This is
    #: deliberately a record of what the model has seen, not a render cache.
    seen_placement_views: set[tuple[str, float, bool, int, float, float]] = field(
        default_factory=set
    )
    reviewed: bool = False

    @property
    def names(self) -> list[str]:
        return [tool.__name__ for tool in self.tools]

    def record_placement_view(
        self,
        tool_context: Any,
        key: tuple[str, float, bool, int, float, float],
    ) -> str:
        """Associate a successfully rendered placement with its tool call."""
        call_id = str(getattr(tool_context, "function_call_id", None) or "__direct__")
        self.pending_placement_views.setdefault(call_id, set()).add(key)
        return call_id

    def mark_placement_views_delivered(self, delivery_ids: set[str]) -> None:
        """Promote pictures whose media survived into a model request.

        The id is also embedded in ordinary function-response JSON because
        some ADK backends strip generated ``adk-*`` FunctionResponse ids.
        Historical replay is then harmless: an old result cannot match a new
        pending call merely because both came from the same tool.
        """
        for delivery_id in delivery_ids:
            self.seen_placement_views.update(
                self.pending_placement_views.pop(delivery_id, set())
            )

    def begin_model_call(self) -> None:
        """Promote pictures from direct calls (a convenience for hosts/tests)."""
        direct = self.pending_placement_views.pop("__direct__", None)
        if direct is not None:
            self.seen_placement_views.update(direct)


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


def damaged_transform_error(state: StackState, spec: JobSpec) -> dict[str, Any] | None:
    """Damaged sections require an applied interactive correction, not identity."""
    failures = []
    identity = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    for record in state.in_order():
        if not record.damaged:
            continue
        transform = record.transform or {}
        reason = None
        if not transform:
            reason = "missing_transform"
        elif transform.get("kind") != "interactive":
            reason = "not_interactive"
        else:
            try:
                params = np.asarray(transform.get("params"), dtype=float)
                if params.shape != (6,) or not np.isfinite(params).all():
                    reason = "invalid_transform"
                elif transform.get("spline") is not None:
                    from langslice.landmark_warp import fit_spline

                    spline = transform["spline"]
                    fitted = fit_spline(spline)
                    is_identity = (
                        fitted.is_identity() if spline.get("backend") == "elastix" else
                        np.allclose(spline["source"], spline["target"], rtol=0, atol=1e-9)
                    )
                    if is_identity:
                        reason = "identity_transform"
                elif np.allclose(params, identity, rtol=0, atol=1e-9):
                    reason = "identity_transform"
            except (TypeError, ValueError, KeyError, RuntimeError, np.linalg.LinAlgError):
                reason = "invalid_transform"
        if reason:
            failures.append({"id": record.id, "reason": reason})
    if not failures:
        return None
    return {
        "status": "error",
        "error": "DAMAGED_REQUIRES_MANUAL_TRANSFORM",
        "failures": failures,
        "interactive_enabled": spec.transform.interactive,
        "message": (
            "Damaged sections require a non-identity interactive transform. "
            + (
                "Use adjust_transforms to align the surviving "
                "anatomy, inspect the returned overlays, then submit again."
                if spec.transform.interactive else
                "Interactive transform tools are disabled for this run; the host "
                "must enable interactive transforms to resolve these sections."
            )
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
        return damaged_transform_error(state, spec) or missing_transforms(state)
    return None


# --- the toolbox ---------------------------------------------------------


def build_tools(
    state: StackState, ctx: EngineContext, spec: JobSpec, *,
    on_event: LiveCallback | None = None,
) -> ToolBox:
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

    def placement_view_key(
        record: SliceState, position_mm: float
    ) -> tuple[str, float, bool, int, float, float]:
        """Identity of one placement picture from the model's perspective."""
        return (
            record.id,
            float(position_mm),
            bool(record.flip),
            int(record.rotation_deg),
            float(state.pitch_deg),
            float(state.yaw_deg),
        )

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

    def mark_damaged(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Set or clear damage flags for sections with unreliable outlines.

        Damage here means the section outline massively deviates from the
        atlas: large missing chunks, a missing hemisphere or olfactory bulb,
        split or independently rotated hemispheres, displaced fragments.
        Bubbles, stains, low contrast and small tears with an intact outline
        are NOT damage.

        Args:
            entries: Objects with id, damaged (boolean, default True), and note.
                Set damaged=False to clear the flag and its note.

        Returns:
            The rows this call changed.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        if any(not isinstance(entry, dict)
               or not isinstance(entry.get("damaged", True), bool) for entry in entries):
            return {"status": "error", "error": "BAD_ARGS"}
        snapshot()
        marked: list[str] = []
        unmarked: list[str] = []
        unknown: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            record.damaged = entry.get("damaged", True)
            record.damage_note = str(entry.get("note", "")).strip() if record.damaged else ""
            (marked if record.damaged else unmarked).append(record.id)
        return {"marked": marked, "unmarked": unmarked, "unknown_ids": unknown,
                **commit(*marked, *unmarked)}

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
        if spec.has("position") and spec.position.gated and not box.reviewed:
            return {
                "status": "refused",
                "error": "NOT_REVIEWED",
                "detail": "view_stack has not run since the last set_positions write",
            }

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

    def reorder_slices(new_order: list[str], after: str = "start") -> dict[str, Any]:
        """Place one or more sections together in the requested order.

        Args:
            new_order: Nonempty list of unique section filenames. These sections
                move as one block in the listed order; all unlisted sections
                keep their relative order. List the whole stack to set its order.
                Use filenames, never corrected indices: this call changes indices.
            after: Filename the block should follow, or "start" (default) to
                put it first. The anchor must not be in new_order.

        Returns:
            Changed rows and moved ids. Only corrected indices change; positions
            and transforms are kept. The whole call is one undoable write.
        """
        if (not isinstance(new_order, list) or not new_order
                or any(not isinstance(item, str) for item in new_order)
                or not isinstance(after, str)):
            return {"status": "error", "error": "BAD_ARGS"}
        ids = {record.id for record in state.slices}
        selected = set(new_order)
        unknown = sorted(selected - ids)
        if unknown:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        if len(selected) != len(new_order):
            return {"status": "error", "error": "DUPLICATE_SLICE_IDS"}
        if after != "start" and after not in ids:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [after]}
        if after in selected:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "after must name a section outside new_order"}

        records = {record.id: record for record in state.slices}
        remaining = [record for record in state.in_order() if record.id not in selected]
        index = 0 if after == "start" else remaining.index(records[after]) + 1
        ordered = remaining[:index] + [records[name] for name in new_order] + remaining[index:]
        snapshot()
        moved = renumber(ordered)
        return {"moved": moved, **commit(*moved)}

    if spec.has("reorder"):
        box.tools += [orient_slices, reorder_slices]

    # --- position -------------------------------------------------------

    def set_positions(
        entries: list[dict[str, Any]], tool_context: Any = None
    ) -> dict[str, Any]:
        """Write positions for one or more sections, and show each placement.

        Positions are in atlas-native millimetres along the slicing axis. A
        value outside the atlas range is clamped and reported back.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>}]``.

        Returns:
            What was written, what was clamped, the rows it changed, and one
            image per placement not already seen in a full-canvas,
            atlas-bearing view with this orientation and these cutting
            angles: the section as corrected over the atlas section at the
            position it was given, both tissue-framed and labelled in the
            top-left corner.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        before = state.to_dict()
        written: list[dict[str, Any]] = []
        written_positions: list[float] = []
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
            if spec.position.gated and not box.compared.get(record.id):
                # One compare is enough: Astra's run-8 method confirms each
                # section at ONE hypothesised position (2026-09-09).
                rejected.append(
                    {
                        "id": record.id,
                        "reason": "not compared since its last write; "
                        "compare_placement first",
                    }
                )
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
            written_positions.append(value)
            box.compared.pop(record.id, None)
            box.reviewed = False

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
        # Do not pay for the same placement picture twice. A successful
        # compare is promoted to ``seen`` only at the next model-call
        # boundary, so compare + write tool calls emitted in one round still
        # return the write picture: the model has not received the compare
        # result yet. Orientation and cutting angles are part of the identity.
        shown = [
            (row, position)
            for row, position in zip(written, written_positions, strict=True)
            if (record := state.by_id(row["id"])) is not None
            and placement_view_key(record, position)
            not in box.seen_placement_views
        ]
        parts: list[types.Part] = []
        failed: list[dict[str, str]] = []
        rendered: list[str] = []
        delivery_id: str | None = None
        for row, position in shown:
            record = state.by_id(row["id"])
            if record is None:
                continue
            try:
                picture = stacked(
                    render_slice(ctx, record, long_edge=ATLAS_LONG_EDGE, frame=True),
                    atlas_sized(atlas_section(ctx, state, position, frame=True), ctx.atlas),
                )
                label = f"{record.id} over atlas {position:.2f} mm"
            except Exception as exc:
                failed.append({"id": row["id"], "message": str(exc)})
                continue
            parts.append(image_to_part(caption(picture, label)))
            rendered.append(record.id)
            delivery_id = box.record_placement_view(
                tool_context, placement_view_key(record, position)
            )
        result["render_failed"] = failed
        shown_rows = [row for row, _position in shown]
        suppressed = [row["id"] for row in written if row not in shown_rows]
        result["description"] = (
            (
                "Attached: one placement image per section not already seen "
                "at this position, orientation and cutting angle, in the "
                "order written, the section as corrected over the atlas section "
                "at the position it was given, for "
                + ", ".join(rendered)
                + "."
                if rendered
                else (
                    "No images: placement rendering failed."
                    if failed
                    else "No images: every written placement was already seen."
                )
            )
            + (
                " Already-seen placements not pictured: " + ", ".join(suppressed) + "."
                if suppressed
                else ""
            )
        )
        result["images_suppressed_seen"] = suppressed
        if delivery_id is not None:
            result[TOOL_MEDIA_DELIVERY_ID_KEY] = delivery_id
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
        mode: str = "template",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        template_opacity: float = 0.0,
        outlines: str = "all",
        tool_context: Any = None,
        border_color: str = "yellow",
        border_thickness: float = 0.5,
    ) -> dict[str, Any]:
        """Show sections against the atlas at candidate positions. Writes nothing.

        At most 4 section-position pairs per call. Physical views return one
        image per pair. side_by_side returns separate tissue-framed reference
        images: one original section per distinct id plus one atlas per pair
        (up to 8 images), without overlaying or re-drawing the section.

        Args:
            entries: ``[{"id": "<filename or corrected index>",
                "positions_mm": [<mm>, ...]}]``. An empty or missing
                ``positions_mm`` means that section's current position.
            mode: "template" (default: the atlas at that position on the
                section's own canvas, at the section's scale — the section
                itself is in the opening message and `view_slices`),
                "side_by_side" (separate original section and atlas images,
                independently tissue-framed, not a shared physical canvas),
                "overlay" (the section with the
                outlines on it), "checkerboard" (section and template in
                alternating tiles), "outlines" (atlas lines and the section's
                silhouette on black) or "section".
            zoom: [x0, y0, x1, y1] as fractions of the canvas; empty is all.
                side_by_side supports only the full view.
            template_opacity: 0..1, the atlas template blended under the
                outlines in "overlay".
            border_color: Atlas border color, a named color or #RRGGBB; yellow
                by default. Display only; does not change the transform.
            border_thickness: Atlas border width in output pixels, 0.25..8;
                default 0.5. Fractional widths are antialiased. Applies only
                where atlas outlines are drawn.
            outlines: "all" (every family boundary), "outer" (the atlas
                outline only) or "none". Outlines and template_opacity apply
                only to physical views, not side_by_side reference images.

        Returns:
            The section-position pairs compared, in order, each with its
            calibration, and the images per pair in that order.
        """
        try:
            rgb, border_thickness = normalize_border_style(border_color, border_thickness)
        except ValueError as exc:
            return {"status": "error", "error": "INVALID_BORDER_STYLE", "message": str(exc)}
        border_color = "#" + "".join(f"{channel:02x}" for channel in rgb)
        view = str(mode or "template").strip().lower()
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
        separate = view == "side_by_side"
        if separate and window and window != [0.0, 0.0, 1.0, 1.0]:
            return {"status": "error", "error": "ZOOM_UNSUPPORTED",
                    "mode": view, "supported_zoom": [0.0, 0.0, 1.0, 1.0]}

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
        delivery_id: str | None = None
        full_atlas_view = view != "section" and (
            not window or window == [0.0, 0.0, 1.0, 1.0]
        )
        section_indexes: dict[str, int] = {}
        for record, position in pairs:
            if record.id not in sections:
                section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
                sections[record.id] = (section, *calibrate(state, ctx, record, section))
            section, um_per_px, source = sections[record.id]
            media_indexes: dict[str, int] = {}
            try:
                if separate:
                    # Resolve/encode both before mutating delivery bookkeeping.
                    atlas_image = atlas_part(ctx, state, position)
                    tissue_image = reference_slice_part(ctx, record)
                    if record.id not in section_indexes:
                        section_indexes[record.id] = len(parts)
                        parts.append(tissue_image)
                    media_indexes = {"section": section_indexes[record.id], "atlas": len(parts)}
                    parts.append(atlas_image)
                else:
                    images, _ = physical_views(
                        section, um_per_px, ctx.atlas, position, cast(Plane, state.plane),
                        state.pitch_deg, state.yaw_deg, dict(IDENTITY_PARAMS),
                        mode=view, zoom=window, template_opacity=opacity, outlines=layer,
                        border_color=border_color, border_thickness=border_thickness,
                        label=f"{record.id} vs atlas {position:.2f} mm",
                        long_edge=VIEW_LONG_EDGE,
                    )
                    parts.extend(image_to_part(image) for image in images)
            except Exception as exc:
                failed.append({"id": record.id, "position_mm": round(position, 3),
                               "message": str(exc)})
                continue
            # A failed render is neither a gate-satisfying comparison nor a
            # picture the model could have seen.
            box.compared.setdefault(record.id, set()).add(round(position, 2))
            if full_atlas_view:
                delivery_id = box.record_placement_view(
                    tool_context, placement_view_key(record, position)
                )
            compared.append({
                "id": record.id,
                "position_mm": round(position, 3),
                "current_position_mm": (
                    None if record.position_mm is None else round(record.position_mm, 3)
                ),
                "calibration": {"um_per_px": round(um_per_px, 3), "source": source},
                **({"image_indexes": media_indexes} if separate else {}),
            })
        result: dict[str, Any] = {
            "status": "ok" if parts else "error",
            **({} if parts else {"error": "RENDER_FAILED"}),
            "compared": compared,
            "unknown_ids": unknown,
            "no_position": unplaced,
            "view": {"mode": view, "zoom": window or [0.0, 0.0, 1.0, 1.0],
                     "outlines": layer, "border_color": border_color,
                     "border_thickness": border_thickness},
            "render_failed": failed,
            "description": (
                "Compared, in order: "
                + ", ".join(f"{row['id']} at {row['position_mm']:.2f} mm" for row in compared)
                + ("; separate reference images mapped by zero-based image_indexes. "
                   "Section captions retain their original display index/flags; "
                   "filenames identify sections. Independently tissue-framed, not "
                   "a shared physical canvas. Outlines/opacity do not apply."
                   if separate else "; one image per pair in that order, each captioned "
                   "with the section and the position it is compared with.")
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        if dropped > 0:
            result["truncated"] = True
            result["dropped_pairs"] = dropped
        if delivery_id is not None:
            result[TOOL_MEDIA_DELIVERY_ID_KEY] = delivery_id
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
        box.reviewed = True
        def atlas_under(record: SliceState) -> Any:
            if record.position_mm is None:
                return None
            try:
                return atlas_sized(
                    atlas_section(ctx, state, float(record.position_mm), frame=True),
                    ctx.atlas,
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
        transform (undoable, and `adjust_transforms` overwrites it).

        Args:
            slice_ids: Filenames or corrected indices; empty means every
                positioned, undamaged section.
            method: "silhouette" (moments fit) or "elastix" (intensity affine).

        Returns:
            Per-section overlap (iou), the transform as the five physical
            knobs about the canvas centre, the calibration the image was
            drawn with, and an image of each fitted section under the atlas
            outlines. The generic changed row is omitted because it repeats
            the same fit identifiers and overlap.
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
            if panel is not None:
                # Every fit returns its picture (run 15, 2026-09-10: a
                # 25-section fit pictured 4 and the model never saw 21).
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
                + ", ".join(record.id for record, _ in fits)
                + ", in that order; each shows the section under its fitted "
                "transform with the atlas region outlines over it at true "
                "physical scale."
            )
            payload[TOOL_MEDIA_PARTS_KEY] = parts
        snapshot()
        for record, outcome in fits:
            record.transform = {
                "kind": "silhouette",
                "params": outcome.pop("params"),  # the six raw numbers stay host-side
                "physical": outcome["physical"],
                "iou": outcome["iou"],
                "calibration": outcome["calibration"],
                "mirrored": outcome["mirrored"],
            }
        save_checkpoint(state, ctx.checkpoint_path)
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
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        markers: Any = None,
        label: str = "",
        spline: dict[str, Any] | None = None,
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
            border_color=border_color, border_thickness=border_thickness,
            outlines=outlines,
            pivot=pivot,
            markers=markers,
            label=label or staged.record.id,
            spline=spline,
            long_edge=OVERLAY_LONG_EDGE,
        )
        return images

    batching_adjustments = False

    def _adjust_transform(
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
        border_color: str = "yellow",
        border_thickness: float = 0.5,
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
            border_color: Atlas border color, a named color or #RRGGBB; yellow
                by default. Display only; does not change the transform.
            border_thickness: Atlas border width in output pixels, 0.25..8;
                default 0.5. Fractional widths are antialiased. Applies only
                where atlas outlines are drawn.
            pivot: What the rotation and the scales turn about: "canvas" (the
                canvas centre), "tissue" (the section's tissue centroid) or
                [fx, fy] fractions of the canvas.
            outlines: Which atlas lines to draw: "all" (every family
                boundary), "outer" (the atlas outline only) or "none".
            note: Short remark for the record. May be empty.

        Returns:
            The physical parameters written, whether state changed, the view
            requested, and its image(s). The normalized matrix, decomposition
            and full adjustment history remain in local state rather than
            being repeated in every result.
        """
        try:
            rgb, border_thickness = normalize_border_style(border_color, border_thickness)
        except ValueError as exc:
            return {"status": "error", "error": "INVALID_BORDER_STYLE", "message": str(exc)}
        border_color = "#" + "".join(f"{channel:02x}" for channel in rgb)
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
                    border_color=border_color, border_thickness=border_thickness,
                    label=f"{record.id} candidate",
                ) + views(
                    staged, other, mode="overlay", zoom=window,
                    template_opacity=opacity, pivot=other_pivot, outlines=layer,
                    border_color=border_color, border_thickness=border_thickness,
                    label=f"{record.id} {'stored' if held else 'identity'}",
                    spline=(previous or {}).get("spline"),
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
                    border_color=border_color, border_thickness=border_thickness,
                )
        except Exception as exc:
            logger.warning("adjust_transform failed for %s: %s", record.id, exc)
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}

        # Encoding is part of producing feedback. Finish it before mutating
        # state so an encoder failure cannot leave an uncheckpointed transform,
        # especially inside a multi-section batch.
        try:
            media_parts = [image_to_part(image) for image in images]
        except Exception as exc:
            logger.warning("adjust_transform encode failed for %s: %s", record.id, exc)
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}

        history = box.transform_history.setdefault(record.id, [])
        history.append(dict(staged.params))
        if wrote:
            if not batching_adjustments:
                snapshot()
            record.transform = written
            if not batching_adjustments:
                save_checkpoint(state, ctx.checkpoint_path)
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
            "status": "ok",
            "id": record.id,
            "position_mm": round(float(record.position_mm or 0.0), 3),
            "physical": written["physical"],
            "written": wrote,
            "view": {
                "mode": view,
                "zoom": window or [0.0, 0.0, 1.0, 1.0],
                "outlines": layer,
                "border_color": border_color,
                "border_thickness": border_thickness,
            },
            **({"ab_reference": reference} if reference is not None else {}),
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
                        f"true physical scale with {border_color} borders "
                        f"{border_thickness} output pixel(s) wide."
                    )
                )
            ),
            TOOL_MEDIA_PARTS_KEY: media_parts,
        }

    def adjust_transforms(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Set and show one to four independent sections in one undoable call.

        Each entry replaces the complete transform, including any previous
        spline or shear. A section may appear once per call; inspect its result
        before making a dependent correction in a later call.

        Args:
            entries: One to four objects with id, rotation_deg (counter-clockwise),
                scale_x, scale_y, translate_x_mm (right), translate_y_mm (down).
                Optional per-entry controls: mode (overlay, checkerboard, outlines,
                section, template, side_by_side or ab), zoom ([x0,y0,x1,y1] canvas
                fractions), template_opacity (0..1), pivot (canvas, tissue or
                [fx,fy]), outlines (all, outer or none), note, border_color
                (named color or #RRGGBB), border_thickness (0.25..8 pixels).
                side_by_side shows section and atlas; ab shows new and previous
                transforms. Both return two images. Other modes return one.

        Returns:
            Per-section results with zero-based image_indexes into the attached
            labelled images, in entry order. All writes form one undo step.
            Repeating unchanged parameters only redraws, without an undo step.
        """
        nonlocal batching_adjustments
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        if len(entries) > MAX_VIEW_SLICES:
            return {
                "status": "error",
                "error": "TOO_MANY_ENTRIES",
                "max_entries": MAX_VIEW_SLICES,
            }
        canonical: list[str] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is not None:
                canonical.append(record.id)
        duplicates = sorted({item for item in canonical if canonical.count(item) > 1})
        if duplicates:
            return {
                "status": "error",
                "error": "DUPLICATE_SLICE_IDS",
                "duplicate_ids": duplicates,
                "message": (
                    "A batch may adjust each section once; inspect the first "
                    "result before making a dependent adjustment."
                ),
            }

        before = state.to_dict()
        results: list[dict[str, Any]] = []
        parts: list[types.Part] = []
        wrote_any = False
        batching_adjustments = True
        try:
            for entry in entries:
                if not isinstance(entry, dict):
                    results.append({"status": "error", "error": "BAD_ARGS"})
                    continue
                mode = str(entry.get("mode", "overlay") or "overlay").strip().lower()
                result = _adjust_transform(
                    str(entry.get("id", "")),
                    entry.get("rotation_deg"),  # type: ignore[arg-type]
                    entry.get("scale_x"),  # type: ignore[arg-type]
                    entry.get("scale_y"),  # type: ignore[arg-type]
                    entry.get("translate_x_mm"),  # type: ignore[arg-type]
                    entry.get("translate_y_mm"),  # type: ignore[arg-type]
                    mode,
                    entry.get("zoom", []),  # type: ignore[arg-type]
                    entry.get("template_opacity", 0.0),  # type: ignore[arg-type]
                    entry.get("pivot", "canvas"),  # type: ignore[arg-type]
                    str(entry.get("outlines", "all")),
                    str(entry.get("note", "")),
                    border_color=entry.get("border_color", "yellow"),
                    border_thickness=entry.get("border_thickness", 0.5),
                )
                media = result.pop(TOOL_MEDIA_PARTS_KEY, [])
                result["image_indexes"] = list(range(len(parts), len(parts) + len(media)))
                parts.extend(media)
                wrote_any = wrote_any or bool(result.get("written"))
                results.append(result)
        finally:
            batching_adjustments = False

        if wrote_any:
            push(before)
            save_checkpoint(state, ctx.checkpoint_path)
        successful = [row["id"] for row in results if row.get("status") == "ok"]
        return {
            "status": "ok" if successful else "error",
            **({} if successful else {"error": "NOTHING_ADJUSTED"}),
            "results": results,
            "description": (
                "Attached feedback images (mapped by each result's image_indexes), "
                "in entry order, for "
                + ", ".join(successful)
                + "; each image is labelled with its section and transform."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    if spec.has("transform"):
        if spec.transform.automatic:
            box.tools.append(fit_affine)
        if spec.transform.interactive:
            box.tools.append(adjust_transforms)
        if spec.transform.angles:
            box.tools.append(set_cutting_angles)

    box.tools.append(submit)
    lock = threading.Lock()
    box.tools = [_serialized(tool, lock, state=state, on_event=on_event) for tool in box.tools]
    return box
