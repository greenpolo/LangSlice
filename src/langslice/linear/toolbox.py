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
import importlib.util
import inspect
import logging
import math
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from google.genai import types

from langslice.adk import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.affine import (
    denormalized_affine,
    normalized_physical_affine,
    physical_affine_matrix,
)
from langslice.linear import appearance as looks
from langslice.linear import deformation
from langslice.linear.atlas_fetch import (
    atlas_part,
    make_view_atlas,
)
from langslice.linear.atlas_grep import GREP_ATLAS_LIMIT, grep_structures, plane_structure_ids
from langslice.linear.checkpoint import save_checkpoint
from langslice.linear.deepslice import run_deepslice as _run_deepslice
from langslice.linear.display import (
    CURRENT,
    DisplayOptions,
    atlas_caption,
    atlas_plane_picture,
    framed_atlas,
    framed_section,
    parse_display,
    regions_in_plane,
    resolution_argument,
    with_display_doc,
)
from langslice.linear.live import LiveCallback, _plain
from langslice.linear.render import (
    MAX_IMAGES_PER_CALL,
    PREVIEW_LONG_EDGE,
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
    rescale_section_matrix,
    resolution_level,
    shown_section,
    spacing_plot,
    stack_sheet,
    stacked,
    status_rows,
)
from langslice.linear.spec import MAX_PARALLEL_TRANSFORMS, JobSpec
from langslice.linear.state import SliceState, StackState, normalize_to_atlas_order
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
#: Placement pictures (``view_placement``, ``set_positions``): the physical
#: views plus ``stacked`` — the section over the atlas, each tissue-framed.
PLACEMENT_MODES = ("template", "stacked", *[m for m in VIEW_MODES if m != "template"])
#: Placement modes whose pictures are tissue-framed rather than drawn on the
#: physical canvas (no zoom; outlines default to none).
FRAMED_PLACEMENT_MODES = ("stacked", "side_by_side")
#: Pictures of sections alone and of the atlas alone.
SECTION_MODES = ("section",)
ATLAS_MODES = ("template",)

#: The transform every section starts from, and the B side of an A/B preview
#: when a section carries nothing yet.
IDENTITY_PARAMS: dict[str, float] = {
    "rotation_deg": 0.0,
    "scale_x": 1.0,
    "scale_y": 1.0,
    "translate_x_mm": 0.0,
    "translate_y_mm": 0.0,
}

#: The transform a locked section carries: the host's snapshot is already
#: registered in-plane, so the identity IS its alignment. ``kind`` ``"host"``
#: counts as a transform at submit and is never written back to the host.
HOST_TRANSFORM_KIND = "host"


def locked_ids(spec: JobSpec) -> set[str]:
    """Sections whose flip, rotation and transform the host locked."""
    return {str(name) for name in (spec.inputs or {}).get("locked") or []}


def host_damaged_ids(spec: JobSpec) -> set[str]:
    """Sections the host marked damaged; the agent cannot clear these flags."""
    return {str(name) for name in ((spec.inputs or {}).get("damaged") or {})}


def host_transform() -> dict[str, Any]:
    """The identity transform a locked section carries."""
    return {
        "kind": HOST_TRANSFORM_KIND,
        "params": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "physical": {**IDENTITY_PARAMS, "shear": 0.0, "pivot": [0.5, 0.5]},
        "mirrored": False,
    }


def _tool_target_ids(state: StackState, name: str, args: dict[str, Any]) -> list[str]:
    """Resolve host display targets before a tool can reorder the stack."""
    if name in {"view_stack", "status", "undo", "redo", "submit",
                "set_cutting_angles", "run_deepslice"} or (
                    name == "preprocess" and not args.get("sections")):
        return [record.id for record in state.in_order()]
    if name == "fit_affine" and not args.get("slice_ids"):
        return [record.id for record in state.in_order()
                if record.position_mm is not None and not record.damaged
                and (record.transform or {}).get("kind") != HOST_TRANSFORM_KIND]
    refs = list(args.get("slice_ids") or args.get("new_order") or args.get("sections") or [])
    if "slice_id" in args:
        refs.append(args["slice_id"])
    if "id" in args:
        refs.append(args["id"])
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


def _clears_stale_deformations(tool: Any, state: StackState, ctx: EngineContext) -> Any:
    """After any tool: drop deformations whose linear placement changed, and say so.

    A deformation is fitted on top of one linear placement; a write that moves
    the position, orientation, cutting angles or transform makes it stale. It
    is cleared in the same undo step as that write (the snapshot was taken
    before it), so `undo` restores the placement and the deformation together.
    """

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        result = tool(*args, **kwargs)
        cleared = deformation.clear_stale(state)
        if cleared:
            save_checkpoint(state, ctx.checkpoint_path)
            if isinstance(result, dict):
                result["deformation_cleared"] = cleared
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
    #: Image corrections still running, per section id: the geometry they
    #: were started at and their future. The agent never waits on them;
    #: `submit` and the end of the session do.
    image_jobs: dict[str, tuple[str, Future[dict[str, Any]]]] = field(default_factory=dict)
    image_executor: ThreadPoolExecutor | None = None
    #: Deformable-fit records: cached by input digest, applied ones on disk
    #: under the results folder (`fit_deformable`, placement pictures).
    deformations: deformation.RecordStore | None = None

    @property
    def names(self) -> list[str]:
        return [tool.__name__ for tool in self.tools]

    def start_image_job(
        self, section_id: str, fingerprint: str, job: Any, *, workers: int
    ) -> None:
        """Run one image correction in the background."""
        if self.image_executor is None:
            self.image_executor = ThreadPoolExecutor(
                max_workers=workers, thread_name_prefix="image-correction"
            )
        self.image_jobs[section_id] = (fingerprint, self.image_executor.submit(job))

    def image_job_running(self, section_id: str, fingerprint: str) -> bool:
        running = self.image_jobs.get(section_id)
        return running is not None and running[0] == fingerprint and not running[1].done()

    def settle_image_corrections(self, state: StackState) -> bool:
        """Wait for every running correction and record its result.

        A result lands only on a section that still holds the running record
        for the same geometry; one undone or superseded meanwhile keeps what
        it has (the reply stays on disk and is reused at that geometry).
        Returns whether any section changed.
        """
        changed = False
        for section_id, (fingerprint, future) in list(self.image_jobs.items()):
            changed |= self._land(state, section_id, fingerprint, _job_result(
                section_id, fingerprint, future, None))
        self.image_jobs.clear()
        if self.image_executor is not None:
            self.image_executor.shutdown(wait=True)
            self.image_executor = None
        return changed

    def wait_image_job(self, state: StackState, section_id: str, timeout: float) -> bool:
        """Wait up to *timeout* seconds for one section's running correction.

        Records its result the way :meth:`settle_image_corrections` does and
        returns True once nothing is running for the section; False when the
        call is still running at the timeout.
        """
        running = self.image_jobs.get(section_id)
        if running is None:
            return True
        fingerprint, future = running
        try:
            result = _job_result(section_id, fingerprint, future, max(0.0, timeout))
        except TimeoutError:
            return False
        del self.image_jobs[section_id]
        self._land(state, section_id, fingerprint, result)
        return True

    @staticmethod
    def _land(
        state: StackState, section_id: str, fingerprint: str, result: dict[str, Any],
    ) -> bool:
        """Record a finished correction on a section still waiting for it."""
        record = state.by_id(section_id)
        held = (record.image_correction or {}) if record is not None else {}
        if (record is not None and held.get("status") == "running"
                and held.get("geometry_fingerprint") == fingerprint):
            record.image_correction = result
            return True
        return False

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


#: ``fit_deformable`` docstring passages about traced section images and their
#: wording for a run without the image model (``nonlinear.provider`` "none").
_STAIN_ONLY_DOC: tuple[tuple[str, str], ...] = (
    (
        '            section_image: "fit" (the section\'s fit appearance), a raw channel\n'
        '                name, "traced_borders" (the section\'s trace_borders result at\n'
        '                this placement, its lines turned into named regions; ANTs) or\n'
        '                "traced_lines" (those lines as lines, against atlas borders).\n'
        '                A traced image waits for a trace still running (up to\n'
        '                TRACE_WAIT minutes) and the reply adds the trace drawn on the\n'
        '                section.\n',
        '            section_image: "fit" (the section\'s fit appearance) or a raw\n'
        '                channel name.\n',
    ),
    ('Empty: "borders" for traced images, else "ara".', 'Empty: "ara".'),
    (" Traced\n            sections add `traces`: each one's trace drawn on the section.", ""),
)


def _job_result(
    section_id: str, fingerprint: str, future: Future[dict[str, Any]], timeout: float | None,
) -> dict[str, Any]:
    """A correction job's result; an exception becomes an error result.

    Raises ``TimeoutError`` when *timeout* passes first.
    """
    try:
        return future.result(timeout=timeout)
    except TimeoutError:
        if not future.done():  # the wait ran out; a job's own timeout is a result
            raise
        error: BaseException = future.exception() or TimeoutError()
        return {"id": section_id, "geometry_fingerprint": fingerprint,
                "status": "error", "error": type(error).__name__, "message": str(error)}
    except Exception as exc:  # the job records its own failures; this is a backstop
        return {"id": section_id, "geometry_fingerprint": fingerprint,
                "status": "error", "error": type(exc).__name__, "message": str(exc)}


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
    # A locked section was aligned by the user and cannot be changed here.
    locked = locked_ids(spec)
    for record in state.in_order():
        if not record.damaged or record.id in locked:
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
        refusal = damaged_transform_error(state, spec) or missing_transforms(state)
        if refusal is not None:
            return refusal
    if spec.has("nonlinear"):
        return missing_deformations(state)
    return None


def missing_deformations(state: StackState) -> dict[str, Any] | None:
    """``None`` when every section carries a deformation, or a keep_linear
    reason, at its current linear placement; else the rejection."""
    missing: list[dict[str, str]] = []
    for record in state.in_order():
        held = record.deformation or {}
        if not held:
            missing.append({"id": record.id, "reason": "no deformation and no keep_linear reason"})
        elif held.get("linear_key") != deformation.linear_key(state, record):
            missing.append({"id": record.id, "reason": "made at a different linear placement"})
    if not missing:
        return None
    return {
        "status": "error",
        "error": "MISSING_DEFORMATIONS",
        "sections": missing,
        "message": (
            f"{len(missing)} of {len(state.slices)} section(s) carry no deformation at their "
            "current placement. Apply one fit_deformable result per section, or give "
            "fit_deformable keep_linear with a reason where the linear placement stands."
        ),
    }


# --- the toolbox ---------------------------------------------------------


def build_tools(
    state: StackState, ctx: EngineContext, spec: JobSpec, *,
    on_event: LiveCallback | None = None,
) -> ToolBox:
    """Build the tools this run's spec switches on, closed over *state*."""
    box = ToolBox()
    pos_lo, pos_hi = ctx.position_range
    locked = locked_ids(spec)
    host_damaged = host_damaged_ids(spec)
    #: The image model is part of this run: trace_borders and traced images exist.
    traces_on = spec.nonlinear.uses_image_model
    #: Below the maximum, the most sections one transform call may take.
    transform_cap = (
        spec.transform.max_parallel
        if spec.transform.max_parallel < MAX_PARALLEL_TRANSFORMS else None
    )

    def over_cap(requested: int) -> dict[str, Any] | None:
        """The refusal for a transform call naming more sections than allowed."""
        if transform_cap is None or requested <= transform_cap:
            return None
        return {
            "status": "error",
            "error": "TOO_MANY_SECTIONS",
            "max_sections": transform_cap,
            "requested": requested,
        }

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

    def display(
        modes: tuple[str, ...], args: dict[str, Any], *,
        sections: list[SliceState] = (),  # type: ignore[assignment]
        zoom_ok: bool = True, framed_modes: tuple[str, ...] = (),
    ) -> DisplayOptions | dict[str, Any]:
        """One call's display options, validated once for every picture tool."""
        return parse_display(
            ctx, state, modes=modes,
            mode=args.get("mode", modes[0]),
            zoom=args.get("zoom") or [],
            section_image=args.get("section_image", CURRENT),
            atlas_image=args.get("atlas_image", "ara"),
            atlas_opacity=args.get("atlas_opacity", 0.0),
            regions=args.get("regions") or [],
            outlines=args.get("outlines", ""),
            border_color=args.get("border_color", "yellow"),
            border_thickness=args.get("border_thickness", 0.5),
            sections=sections, zoom_ok=zoom_ok, framed_modes=framed_modes,
            resolution=args.get("resolution", 0),
        )

    def region_names(values: Any, field_name: str) -> tuple[str, ...] | dict[str, Any]:
        """Region acronyms or ids for `include`/`exclude`, checked against the atlas."""
        from langslice.deformable.atlas_images import resolve_structures

        if values is None:
            return ()
        if not isinstance(values, (list, tuple)):
            return {"status": "error", "error": "BAD_ARGS",
                    "message": f"{field_name} must be a list of acronyms or ids"}
        names = tuple(str(value).strip() for value in values if str(value).strip())
        if names:
            try:
                resolve_structures(ctx.atlas, names)
            except ValueError as exc:
                return {"status": "error", "error": "UNKNOWN_REGIONS", "message": str(exc),
                        "argument": field_name}
        return names

    def section_label(record: SliceState, options: DisplayOptions) -> str:
        label = f"{record.index_corrected}: {record.id}"
        if options.section_image != CURRENT:
            label += f"  [{options.section_image} channel]"
        return label

    def section_part(record: SliceState, options: DisplayOptions) -> types.Part:
        """One section as corrected, tissue-framed, its index and id burned in."""
        return image_to_part(
            caption(framed_section(ctx, state, record, options), section_label(record, options))
        )

    def atlas_name(options: DisplayOptions) -> str:
        return "template" if options.atlas_image == "ara" else options.atlas_image

    def draw_canvas(
        record: SliceState,
        section: Any,
        um_per_px: float,
        position: float,
        params: dict[str, float] | np.ndarray,
        options: DisplayOptions,
        *,
        mode: str | None = None,
        pivot: tuple[float, float] | None = None,
        section_offset: tuple[int, int] = (0, 0),
        label: str = "",
        spline: dict[str, Any] | None = None,
        long_edge: int | None = None,
        matrix_label: str = "fitted matrix",
        warp: Any = None,
    ) -> list[Any]:
        """The physical canvas pictures of one section at one placement.

        *section* is the working frame a transform is computed on; the
        picture is drawn from the render the options ask for (the view
        appearance or a raw channel), large enough that each panel, zoom
        included, comes out at *long_edge* (None: ``options.long_edge``)
        unless the section's working copy has fewer pixels, with a matrix and
        the pivot carried onto it. The one renderer of every placement
        picture: `view_placement`, `set_positions`, `adjust_transforms` and
        `fit_affine` all draw through here. *warp* (a `DeformableRecord` on
        this placement) resamples the section into its placed-atlas frame
        first, so the picture shows the full registration: linear placement
        plus deformation.
        """
        edge = int(long_edge or options.long_edge)
        window = options.window
        # The canvas is at least the section on each axis, so a section
        # render 1/span times the panel puts at least the panel's pixels
        # inside the zoom (render_slice stops at the working copy).
        span = (min(max(window[2] - window[0], 1e-3), max(window[3] - window[1], 1e-3), 1.0)
                if window else 1.0)
        render_edge = int(math.ceil(edge / span))
        if render_edge > PREVIEW_LONG_EDGE:
            # Past the working copy every request is the same render: one
            # cache entry, however deep the zoom.
            render_edge = min(render_edge, max(ctx.working_source(record.id)[0].size))
        shown, shown_um, (fx, fy) = shown_section(
            ctx, record, section, um_per_px, options.look(state, record),
            long_edge=render_edge,
        )
        in_section: tuple[float, float] | None = None
        if shown is not section:
            if isinstance(params, np.ndarray):
                params = rescale_section_matrix(params, fx, fy)
            if pivot is not None:
                ox, oy = section_offset
                in_section = ((pivot[0] - ox) * fx, (pivot[1] - oy) * fy)
        if warp is not None:
            from langslice.deformable import warp_section_image

            shown = warp_section_image(shown, warp)
        images, _iou = physical_views(
            shown, shown_um, ctx.atlas, position, cast(Plane, state.plane),
            state.pitch_deg, state.yaw_deg, params,
            mode=mode or options.mode, zoom=options.window,
            atlas_opacity=options.atlas_opacity, outlines=options.outlines,
            border_color=options.border_color, border_thickness=options.border_thickness,
            pivot=pivot if in_section is None else None, pivot_in_section=in_section,
            label=label or record.id, spline=spline, long_edge=edge,
            atlas_picture=atlas_plane_picture(ctx, state, options.atlas_image, position),
            atlas_name=atlas_name(options), regions=options.regions,
            matrix_label=matrix_label,
        )
        return images

    def stored_placement(record: SliceState, section: Any) -> tuple[Any, Any, str]:
        """``(params or matrix, spline, kind)`` of the section's in-plane transform.

        The six stored numbers are the exact map (shear included); a section
        without a transform is drawn at identity.
        """
        transform = record.transform or {}
        values = transform.get("params")
        if values is not None and len(values) == 6:
            return (denormalized_affine(values, section.size), transform.get("spline"),
                    str(transform.get("kind") or "stored"))
        return dict(IDENTITY_PARAMS), None, "identity"

    def deformation_store() -> deformation.RecordStore:
        """Fitted records of this run, saved under the results folder (made on first use)."""
        if box.deformations is None:
            box.deformations = deformation.RecordStore(
                root=Path(ctx.results_path).parent / "deformable")
        return box.deformations

    def current_warp(record: SliceState) -> Any:
        """The section's applied deformation record, or None (or unreadable)."""
        if not record.deformation:
            return None
        try:
            return deformation_store().current(state, record)
        except Exception:  # a missing record must not break a placement picture
            logger.warning("Deformation record unreadable for %s", record.id, exc_info=True)
            return None

    def absent_regions(position: float, options: DisplayOptions) -> list[str]:
        present = regions_in_plane(ctx, state, position, options)
        return [name for name, _ids in options.regions if name not in present]

    @with_display_doc('"section" (the only mode here).')
    def view_slices(
        slice_ids: list[str],
        mode: str = "section",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
    ) -> dict[str, Any]:
        """Look at up to 4 named sections at higher resolution.

        Sections are rendered as corrected: any rotation and flip already
        applied, framed to their tissue the same way atlas sections are.
        Each image carries its corrected index and filename burned into its
        top-left corner.

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
        options = display(SECTION_MODES, dict(
            mode=mode, zoom=zoom, section_image=section_image, atlas_image=atlas_image,
            atlas_opacity=atlas_opacity, regions=regions, outlines=outlines,
            border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=known)
        if isinstance(options, dict):
            return options
        parts = [section_part(record, options) for record in known]
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
            "view": options.echo(),
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
        rejected: list[dict[str, str]] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is None:
                unknown.append(str(entry.get("id", "")))
                continue
            if record.id in host_damaged and not entry.get("damaged", True):
                rejected.append({"id": record.id, "error": "DAMAGE_SET_BY_USER"})
                continue
            record.damaged = entry.get("damaged", True)
            record.damage_note = str(entry.get("note", "")).strip() if record.damaged else ""
            (marked if record.damaged else unmarked).append(record.id)
        return {"marked": marked, "unmarked": unmarked, "unknown_ids": unknown,
                **({"rejected": rejected} if rejected else {}),
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
        # With the image model, missing traces are reported before missing
        # deformations: a deformation may be fitted to its section's trace.
        if refusal is not None and not (
                traces_on and refusal.get("error") == "MISSING_DEFORMATIONS"):
            return refusal
        if spec.has("nonlinear") and traces_on:
            from langslice.registration_tool import correction_fingerprint

            if box.settle_image_corrections(state):
                save_checkpoint(state, ctx.checkpoint_path)
            pending: list[dict[str, str]] = []
            for record in state.in_order():
                result = record.image_correction or {}
                try:
                    current = correction_fingerprint(state, ctx, record.id)
                except (OSError, ValueError) as exc:
                    pending.append({"id": record.id, "reason": str(exc)})
                    continue
                if result.get("status") != "ok":
                    pending.append({"id": record.id, "reason": "No completed image correction"})
                elif result.get("geometry_fingerprint") != current:
                    pending.append({
                        "id": record.id, "reason": "Placement changed since image correction",
                    })
            if pending:
                return {
                    "status": "refused",
                    "error": "MISSING_IMAGE_CORRECTIONS",
                    "sections": pending,
                    "message": "Each section requires a completed image correction at its current "
                    "linear placement. This checks completion and geometry, "
                    "not anatomical quality.",
                }
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
        # Direction is a convention, not an inference: a posterior-first stack
        # is emitted in atlas order without the agent being told about it.
        if normalize_to_atlas_order(state):
            state.notes.append("submit: corrected order reversed to run the atlas way")
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

    # --- appearance (JobSpec.agent_preprocessing) ----------------------

    def channel_summary(records: list[SliceState]) -> Any:
        """The raw channel names: one list when every section shares it."""
        named = {record.id: list(ctx.section_channels(record.id)[0]) for record in records}
        distinct = {tuple(names) for names in named.values()}
        return list(next(iter(distinct))) if len(distinct) == 1 else named

    @with_display_doc('"section" (the only mode here).')
    def preprocess(
        target: str = "both",
        sections: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        channel_weights: list[float] = [],  # noqa: B006
        clahe_clip: float = looks.DEFAULT_CLAHE_CLIP,
        clahe_tiles: int = looks.DEFAULT_CLAHE_TILES,
        n4: bool = False,
        denoise: bool = False,
        reset: bool = False,
        mode: str = "section",
        zoom: list[float] = [],  # noqa: B006
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
    ) -> dict[str, Any]:
        """Set how sections look: for what you view, for what a fit reads, or both.

        Sets the appearance of the whole stack (no sections) or of named
        sections (overriding the stack's), for target "view" (every picture
        you are shown from now on), "fit" (the image a deformable fit reads)
        or "both"; each target keeps its own setting. The image is built from
        the section's raw channels: each channel with weight above zero is
        optionally N4 bias-field corrected and denoised (ANTs), contrast-
        enhanced by CLAHE, then the channels are blended by their weights.
        Until this is called both targets use the default appearance.
        Writes, checkpoints and can be undone; another call replaces the
        setting. Display options never change it.

        Args:
            target: "view", "fit" or "both".
            sections: Filenames or corrected indices; empty sets the stack.
            channel_weights: One weight per raw channel, in the order the
                result's `channels` lists them; 0 leaves a channel out. Give
                the counterstain that lights all the tissue (DAPI, Nissl) the
                most weight and sparse labels (tracers, reporters) little or
                none. Empty: automatic weights by tissue coverage.
            clahe_clip: CLAHE clip limit, 0 (no CLAHE) to 40; the default
                appearance uses 4.
            clahe_tiles: CLAHE tiles per side, 1 to 32; the default uses 8.
            n4: ANTs N4 bias-field correction of uneven illumination.
            denoise: ANTs denoising.
            reset: True returns the target to the default appearance (named
                sections: back to the stack's setting); other settings are
                ignored.

        Returns:
            The settings in force per target, the raw channels, and the
            affected sections (up to 4) rendered as the target now sees them,
            each labelled.
        """
        chosen = str(target or "both").strip().lower()
        if chosen not in looks.TARGET_CHOICES:
            return {"status": "error", "error": "BAD_TARGET",
                    "targets": list(looks.TARGET_CHOICES)}
        targets = list(looks.TARGETS) if chosen == "both" else [chosen]
        if sections:
            scope, unknown = resolve_many(list(sections))
            if unknown or not scope:
                return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        else:
            scope = state.in_order()
        settings: dict[str, Any] | None = None
        if not reset:
            try:
                settings = looks.validate_settings(
                    channel_weights=list(channel_weights or []) or None,
                    clahe_clip=clahe_clip, clahe_tiles=clahe_tiles, n4=n4, denoise=denoise,
                )
            except ValueError as exc:
                return {"status": "error", "error": "BAD_ARGS", "message": str(exc)}
            weights = settings["channel_weights"]
            if weights is not None:
                mismatched = {
                    record.id: list(names) for record in scope
                    if len(names := ctx.section_channels(record.id)[0]) != len(weights)
                }
                if mismatched:
                    return {"status": "error", "error": "CHANNEL_COUNT_MISMATCH",
                            "weights": len(weights), "channels": mismatched}
            if (n4 or denoise) and importlib.util.find_spec("ants") is None:
                return {"status": "error", "error": "UNAVAILABLE",
                        "message": "N4 and denoising need antspyx: install LangSlice's "
                        "'registration' extra (pip install 'langslice[registration]')."}
        if sections:
            shown = scope[:MAX_VIEW_SLICES]
        else:  # up to four sections spread evenly over the stack
            picks = np.linspace(0, len(scope) - 1, min(len(scope), MAX_VIEW_SLICES))
            shown = [scope[index] for index in sorted({int(round(v)) for v in picks})]
        options = display(SECTION_MODES, dict(
            mode=mode, zoom=zoom, section_image=section_image, atlas_image=atlas_image,
            atlas_opacity=atlas_opacity, regions=regions, outlines=outlines,
            border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=shown)
        if isinstance(options, dict):
            return options

        before = state.to_dict()
        ids = [record.id for record in scope] if sections else None
        for name in targets:
            looks.set_settings(state, name, ids, settings)
        pictured = targets[0]  # "both" writes one setting to both targets
        parts: list[types.Part] = []
        try:
            for record in shown:
                label = section_label(record, options) + f"  [{pictured} appearance]"
                parts.append(image_to_part(caption(
                    framed_section(ctx, state, record, options, target=pictured), label,
                )))
        except Exception as exc:
            state.restore(before)
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}
        push(before)
        save_checkpoint(state, ctx.checkpoint_path)
        in_force = {
            name: (
                {"sections": {record.id: looks.section_settings(state, name, record.id)
                              for record in scope}}
                if ids else {"stack": looks.section_settings(state, name, "")}
            )
            for name in targets
        }
        return {
            "status": "ok",
            "targets": targets,
            "scope": ids or "stack",
            "settings": in_force,
            "channels": channel_summary(scope),
            "shown": [record.id for record in shown],
            "description": (
                "Attached: " + ", ".join(record.id for record in shown)
                + f", in that order, as the {pictured} target now sees them; a null "
                "setting is the default appearance."
            ),
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    box.tools = [
        status,
        view_slices,
        make_view_atlas(state, ctx),
        note,
        undo,
        redo,
    ]
    if spec.agent_damage:
        box.tools.append(mark_damaged)
    if spec.agent_preprocessing:
        box.tools.append(preprocess)

    # --- orientation (part of the transform task) ------------------------

    @with_display_doc('"section" (the only mode here).')
    def orient_slices(
        entries: list[dict[str, Any]],
        mode: str = "section",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
    ) -> dict[str, Any]:
        """Set the flip and rotation of one or more sections, and show them.

        Orientation is part of in-plane alignment: a flip is the sign of the
        section's affine. Corrections are recorded as data; the user's image
        files are never modified. Rotation is applied first, then the flip. A
        section whose orientation changes loses its transform. Determining
        hemisphere orientation (whether a section is mirrored) is only
        possible when there is a visible notch or a noticeable oblique cutting
        angle that produces differences between the hemispheres' anatomy.

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
        named, _ = resolve_many([entry.get("id", "") for entry in entries
                                 if isinstance(entry, dict)])
        options = display(SECTION_MODES, dict(
            mode=mode, zoom=zoom, section_image=section_image, atlas_image=atlas_image,
            atlas_opacity=atlas_opacity, regions=regions, outlines=outlines,
            border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=named)
        if isinstance(options, dict):
            return options
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
            if record.id in locked:
                rejected.append({"id": record.id, "error": "LOCKED"})
                continue
            was = (record.flip, record.rotation_deg)
            if "flip" in entry and entry["flip"] is not None:
                if not spec.transform.flip:
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
                parts.append(section_part(record, options))
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
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    # --- reorder --------------------------------------------------------

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
        box.tools.append(reorder_slices)

    # --- position -------------------------------------------------------

    def placement_pictures(
        record: SliceState,
        position: float,
        options: DisplayOptions,
        parts: list[types.Part],
        section_indexes: dict[str, int],
        working: dict[str, tuple[Any, float, str]],
    ) -> dict[str, Any]:
        """Append one section-position pair's pictures to *parts*.

        ``side_by_side``: separate tissue-framed references (one section per
        distinct id, one atlas per pair), mapped by the returned
        ``image_indexes``. ``stacked``: one image, the framed section over the
        framed atlas. Every other mode: the physical canvas, the section under
        its stored in-plane transform (identity when it has none). Returns the
        pair's row fields (calibration, transform drawn, image indexes).
        """
        if record.id not in working:
            section = render_slice(ctx, record, long_edge=PREVIEW_LONG_EDGE)
            working[record.id] = (section, *calibrate(state, ctx, record, section))
        section, um_per_px, source = working[record.id]
        row: dict[str, Any] = {"calibration": {"um_per_px": round(um_per_px, 3), "source": source}}
        default_atlas = (options.atlas_image == "ara" and options.outlines == "none"
                         and not options.regions)
        if options.mode == "side_by_side":
            # Resolve/encode both before mutating delivery bookkeeping.
            atlas_image = (
                atlas_part(ctx, state, position, long_edge=options.long_edge)
                if default_atlas else image_to_part(caption(
                    framed_atlas(ctx, state, position, options),
                    atlas_caption(state, position, options),
                ))
            )
            tissue_image = reference_slice_part(
                ctx, record, long_edge=options.long_edge, look=options.look(state, record),
            )
            if record.id not in section_indexes:
                section_indexes[record.id] = len(parts)
                parts.append(tissue_image)
            row["image_indexes"] = {"section": section_indexes[record.id], "atlas": len(parts)}
            parts.append(atlas_image)
            return row
        if options.mode == "stacked":
            # One picture: the atlas is drawn to the section's long edge so
            # the two read at the same size, as in `view_stack`.
            top = framed_section(ctx, state, record, options)
            picture = stacked(
                top, framed_atlas(ctx, state, position, options, long_edge=max(top.size),
                                  fill=True),
            )
            parts.append(image_to_part(caption(
                picture, f"{record.id} over atlas {position:.2f} mm"
                + ("" if options.atlas_image == "ara" else f" ({options.atlas_image})"),
            )))
            return row
        params, spline, kind = stored_placement(record, section)
        # The section under its full placement: the stored warp too, at the
        # position it was fitted at (the atlas-only view needs no section).
        warp = (current_warp(record)
                if position == record.position_mm and options.mode != "template" else None)
        parts.extend(image_to_part(image) for image in draw_canvas(
            record, section, um_per_px, position, params, options,
            label=f"{record.id} vs atlas {position:.2f} mm", spline=spline,
            matrix_label=f"{kind} transform" + (" + deformation" if warp is not None else ""),
            warp=warp,
        ))
        row["transform"] = kind
        if warp is not None:
            row["deformation_drawn"] = True
        return row

    @with_display_doc(
        '"stacked" (default: the section as corrected over the atlas at the '
        'position given, each tissue-framed), "template", "side_by_side", '
        '"overlay", "checkerboard", "outlines" or "section" — as in '
        '`view_placement`.'
    )
    def set_positions(
        entries: list[dict[str, Any]],
        mode: str = "stacked",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Write positions for one or more sections, and show each placement.

        Positions are in atlas-native millimetres along the slicing axis. A
        value outside the atlas range is clamped and reported back.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>}]``.

        Returns:
            What was written, what was clamped, the rows it changed, and one
            picture per placement not already seen in a full-canvas,
            atlas-bearing view with this orientation and these cutting
            angles, labelled in the top-left corner.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        named, _ = resolve_many([entry.get("id", "") for entry in entries
                                 if isinstance(entry, dict)])
        options = display(PLACEMENT_MODES, dict(
            mode=mode, zoom=zoom, section_image=section_image, atlas_image=atlas_image,
            atlas_opacity=atlas_opacity, regions=regions, outlines=outlines,
            border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=named, framed_modes=FRAMED_PLACEMENT_MODES)
        if isinstance(options, dict):
            return options
        if options.mode in ("stacked", "side_by_side") and not options.full_view:
            return {"status": "error", "error": "ZOOM_UNSUPPORTED", "mode": options.mode,
                    "supported_zoom": [0.0, 0.0, 1.0, 1.0]}
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
                        "view_placement first",
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
        section_indexes: dict[str, int] = {}
        working: dict[str, tuple[Any, float, str]] = {}
        image_indexes: dict[str, Any] = {}
        atlas_bearing = options.mode != "section" and options.full_view
        for row, position in shown:
            record = state.by_id(row["id"])
            if record is None:
                continue
            first = len(parts)
            try:
                extras = placement_pictures(
                    record, position, options, parts, section_indexes, working,
                )
            except Exception as exc:
                del parts[first:]
                failed.append({"id": row["id"], "message": str(exc)})
                continue
            image_indexes[record.id] = extras.get(
                "image_indexes", list(range(first, len(parts))),
            )
            rendered.append(record.id)
            if atlas_bearing:
                delivery_id = box.record_placement_view(
                    tool_context, placement_view_key(record, position)
                )
        result["render_failed"] = failed
        shown_rows = [row for row, _position in shown]
        suppressed = [row["id"] for row in written if row not in shown_rows]
        result["description"] = (
            (
                "Attached: the placement pictures of each section not already "
                "seen at this position, orientation and cutting angle, in the "
                "order written (mapped by image_indexes), for "
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
        if rendered:
            result["image_indexes"] = image_indexes
            result["view"] = options.echo()
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

    def search_position(slice_id: str, window_mm: float, angles: bool) -> dict[str, Any]:
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
            logger.warning("search_position failed for %s: %s", record.id, exc)
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

    @with_display_doc(
        '"template" (default: the atlas image at that position on the '
        "section's own canvas, at the section's scale — the section itself is "
        'in the opening message and `view_slices`), "stacked" (one picture: '
        'the section over the atlas, each tissue-framed), "side_by_side" '
        "(separate original section and atlas images, independently "
        'tissue-framed, full view only), "overlay" (the section under its '
        'transform with the atlas lines on it), "checkerboard" (section and '
        'atlas image in alternating tiles), "outlines" (atlas lines and the '
        "section's silhouette on black) or \"section\"."
    )
    def view_placement(
        entries: list[dict[str, Any]],
        mode: str = "template",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Show sections in their full current placement, or at candidate positions.

        Writes nothing. At most 4 section-position pairs per call. A section
        is drawn under its current in-plane transform (identity when it has
        none) on a millimetre-true canvas: physical views return one image per
        pair. side_by_side returns separate tissue-framed reference images:
        one original section per distinct id plus one atlas per pair (up to 8
        images), without overlaying or re-drawing the section.

        Args:
            entries: ``[{"id": "<filename or corrected index>",
                "positions_mm": [<mm>, ...]}]``. An empty or missing
                ``positions_mm`` means that section's current position.

        Returns:
            The section-position pairs shown, in order, each with its
            calibration and the transform drawn, and the images per pair in
            that order.
        """
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
        options = display(PLACEMENT_MODES, dict(
            mode=mode, zoom=zoom, section_image=section_image, atlas_image=atlas_image,
            atlas_opacity=atlas_opacity, regions=regions, outlines=outlines,
            border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=list({record.id: record for record, _p in pairs}.values()),
            framed_modes=FRAMED_PLACEMENT_MODES)
        if isinstance(options, dict):
            return options
        if options.mode in FRAMED_PLACEMENT_MODES and not options.full_view:
            return {"status": "error", "error": "ZOOM_UNSUPPORTED",
                    "mode": options.mode, "supported_zoom": [0.0, 0.0, 1.0, 1.0]}
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

        working: dict[str, tuple[Any, float, str]] = {}
        compared: list[dict[str, Any]] = []
        parts: list[types.Part] = []
        failed: list[dict[str, Any]] = []
        delivery_id: str | None = None
        full_atlas_view = options.mode != "section" and options.full_view
        section_indexes: dict[str, int] = {}
        for record, position in pairs:
            first = len(parts)
            try:
                extras = placement_pictures(
                    record, position, options, parts, section_indexes, working,
                )
            except Exception as exc:
                del parts[first:]
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
            absent = absent_regions(position, options)
            compared.append({
                "id": record.id,
                "position_mm": round(position, 3),
                "current_position_mm": (
                    None if record.position_mm is None else round(record.position_mm, 3)
                ),
                **extras,
                **({"regions_not_in_plane": absent} if absent else {}),
            })
        separate = options.mode == "side_by_side"
        result: dict[str, Any] = {
            "status": "ok" if parts else "error",
            **({} if parts else {"error": "RENDER_FAILED"}),
            "compared": compared,
            "unknown_ids": unknown,
            "no_position": unplaced,
            "view": options.echo(),
            "render_failed": failed,
            "description": (
                "Shown, in order: "
                + ", ".join(f"{row['id']} at {row['position_mm']:.2f} mm" for row in compared)
                + ("; separate reference images mapped by zero-based image_indexes. "
                   "Section captions retain their original display index/flags; "
                   "filenames identify sections. Independently tissue-framed, not "
                   "a shared physical canvas."
                   if separate else "; the images of each pair in that order, each "
                   "captioned with the section and the position it is shown at.")
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        if dropped > 0:
            result["truncated"] = True
            result["dropped_pairs"] = dropped
        if delivery_id is not None:
            result[TOOL_MEDIA_DELIVERY_ID_KEY] = delivery_id
        return result

    @with_display_doc(
        '"stacked" (the only mode here: each section over the atlas at its '
        "position). zoom is not supported."
    )
    def view_stack(
        mode: str = "stacked",
        zoom: list[float] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
    ) -> dict[str, Any]:
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
        options = display(("stacked",), dict(
            mode=mode, zoom=zoom, section_image=section_image, atlas_image=atlas_image,
            atlas_opacity=atlas_opacity, regions=regions, outlines=outlines,
            border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=state.in_order(), zoom_ok=False, framed_modes=("stacked",))
        if isinstance(options, dict):
            return options
        box.reviewed = True

        def atlas_under(record: SliceState) -> Any:
            if record.position_mm is None:
                return None
            try:
                return framed_atlas(ctx, state, float(record.position_mm), options)
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
            image_to_part(stack_sheet(
                state, ctx, under=atlas_under, section_image=options.section_image,
                tile_edge=options.resolution,
            )),
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
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    if spec.has("position"):
        box.tools += [set_positions, view_placement, view_stack]
        if spec.position.deepslice:
            box.tools.append(run_deepslice)
        if spec.position.bayesian:
            box.tools.append(search_position)

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

    @with_display_doc(
        '"overlay" (default), "side_by_side", "checkerboard", "outlines", '
        '"section" or "template", as in `adjust_transforms`.'
    )
    def fit_affine(
        slice_ids: list[str],
        method: str,
        include: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        exclude: list[str] = [],  # noqa: B006
        mode: str = "overlay",
        zoom: list[float] = [],  # noqa: B006
        section_image: str = "current",
        atlas_image: str = "ara",
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 0.5,
        resolution: int = 0,
    ) -> dict[str, Any]:
        """Fit an in-plane affine per section against its atlas section.

        The tissue outline is matched against the atlas outline (outlines
        only, no internal anatomy). Without regions the whole of both is used
        and a damaged section is refused. Each fit is written as the
        section's transform (undoable, and `adjust_transforms` overwrites it).

        Args:
            slice_ids: Filenames or corrected indices; empty means every
                positioned, undamaged section.
            method: "silhouette" (moments fit) or "elastix" (intensity affine).
            include: Regions (acronyms or ids, descendants included) to fit
                by: only the atlas within 300 um of them, against the tissue
                the fit lays there. Counts only where they reach the outline.
            exclude: Regions removed from the atlas side (e.g. tissue missing
                from the section), descendants included; the tissue the fit
                lays on them is left out too. With regions given, damaged
                sections are fitted.

        Returns:
            Per-section overlap (iou; with regions, of the kept atlas and the
            tissue that corresponds), the transform as the five physical
            knobs about the canvas centre, the calibration the image was
            drawn with, a `regions` report when regions were given, and an
            image of each fitted section under the atlas outlines. The
            generic changed row is omitted because it repeats the same fit
            identifiers and overlap.
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
        kept = region_names(include, "include")
        if isinstance(kept, dict):
            return kept
        dropped = region_names(exclude, "exclude")
        if isinstance(dropped, dict):
            return dropped
        overlap = sorted({name.lower() for name in kept} & {name.lower() for name in dropped})
        if overlap:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "A region cannot be both included and excluded: "
                    + ", ".join(overlap)}
        restricted = bool(kept or dropped)

        if slice_ids:
            targets, unknown = resolve_many(list(slice_ids))
        else:
            targets = [
                record
                for record in state.in_order()
                if record.position_mm is not None and not record.damaged
                and record.id not in locked
            ]
            unknown = []
        refusal = over_cap(len(targets) + len(unknown))
        if refusal is not None:
            return refusal
        options = display(VIEW_MODES, dict(
            mode=mode or "overlay", zoom=zoom, section_image=section_image,
            atlas_image=atlas_image, atlas_opacity=atlas_opacity, regions=regions or list(kept),
            outlines=outlines, border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ), sections=targets)
        if isinstance(options, dict):
            return options

        def draw_fit(record: SliceState) -> Any:
            def draw(section: Any, um_per_px: float, matrix: np.ndarray) -> list[Any]:
                return draw_canvas(
                    record, section, um_per_px, float(record.position_mm or 0.0),
                    matrix, options, label=record.id,
                )
            return draw

        results: list[dict[str, Any]] = []
        parts: list[types.Part] = []
        fits: list[tuple[SliceState, dict[str, Any]]] = []
        for record in targets:
            if record.id in locked:
                results.append({"id": record.id, "status": "error", "error": "LOCKED"})
                continue
            if record.damaged and not restricted:
                results.append({"id": record.id, "status": "error", "error": "DAMAGED"})
                continue
            outcome = fit_silhouette(state, ctx, record, draw=draw_fit(record),
                                     include=kept, exclude=dropped)
            panels = outcome.pop("panels", None) or []
            results.append(outcome)
            if outcome["status"] != "ok":
                continue
            fits.append((record, outcome))
            # Every fit returns its picture (run 15, 2026-09-10: a
            # 25-section fit pictured 4 and the model never saw 21).
            outcome["image_indexes"] = list(range(len(parts), len(parts) + len(panels)))
            parts.extend(image_to_part(panel) for panel in panels)

        payload: dict[str, Any] = {
            "status": "ok" if fits else "error",
            "results": results,
            "unknown_ids": unknown,
            "view": options.echo(),
        }
        if not fits:
            payload["error"] = "NOTHING_FITTED"
            return payload
        if parts:
            payload["description"] = (
                "Attached panels are "
                + ", ".join(record.id for record, _ in fits)
                + ", in that order (mapped by each result's image_indexes); "
                "each shows the section under its fitted transform against "
                "the atlas at true physical scale, in the requested view."
                + (" The whole atlas is drawn; the fit used only the kept regions."
                   if restricted else "")
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
                **({"regions": {"include": list(kept), "exclude": list(dropped)}}
                   if restricted else {}),
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

    def staged_views(
        staged: _Staged,
        params: dict[str, float] | np.ndarray,
        options: DisplayOptions,
        *,
        mode: str,
        pivot: tuple[float, float] | None,
        label: str = "",
        spline: dict[str, Any] | None = None,
    ) -> list[Any]:
        """One staged section's canvas pictures (the write is on its working frame)."""
        return draw_canvas(
            staged.record, staged.section, staged.um_per_px,
            float(staged.record.position_mm or 0.0), params, options,
            mode=mode, pivot=pivot, section_offset=staged.geometry.section_offset,
            label=label, spline=spline,
        )

    batching_adjustments = False

    #: Per-entry keys of `adjust_transforms` beyond the transform itself.
    entry_display_keys = (
        "mode", "zoom", "section_image", "atlas_image", "atlas_opacity", "regions",
        "outlines", "border_color", "border_thickness", "resolution",
    )

    def _adjust_transform(
        slice_id: str,
        rotation_deg: float,
        scale_x: float,
        scale_y: float,
        translate_x_mm: float,
        translate_y_mm: float,
        options: DisplayOptions,
        pivot: str | list[float] = "canvas",
        note: str = "",
    ) -> dict[str, Any]:
        """Set one section's in-plane transform and show the result.

        Every call writes the parameters as the section's transform and
        returns the section drawn under them in the requested view. The same
        parameters again only re-draw. The section's flip and rotation flags
        are not touched. Writes, checkpoints, and can be undone (the batch
        tool owns the undo step).
        """
        view = options.mode
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
                images = staged_views(
                    staged, staged.params, options, mode="overlay", pivot=staged.pivot,
                    label=f"{record.id} candidate",
                ) + staged_views(
                    staged, other, options, mode="overlay", pivot=other_pivot,
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
                images = staged_views(
                    staged, staged.params, options, mode=view, pivot=staged.pivot,
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
        image = "atlas template" if options.atlas_image == "ara" else f"atlas {options.atlas_image}"
        described = {
            "overlay": "the section with the atlas outlines over it",
            "side_by_side": (
                f"two images at the same scale and the same crop: the section, then the {image}"
            ),
            "checkerboard": f"the section and the {image} in alternating tiles",
            "outlines": (
                "the atlas outlines and the section's own silhouette contour, "
                "on black"
            ),
            "section": "the section alone, no outlines",
            "template": f"the {image} alone, no outlines",
            "ab": (
                "two overlays at the same crop: first these parameters, then "
                + (
                    "the transform the section carried before this call"
                    if previous is not None
                    else "identity"
                )
            ),
        }[view]
        layer = options.outlines
        position = float(record.position_mm or 0.0)
        absent = absent_regions(position, options)
        return {
            "status": "ok",
            "id": record.id,
            "position_mm": round(position, 3),
            "physical": written["physical"],
            "written": wrote,
            "view": options.echo(),
            **({"ab_reference": reference} if reference is not None else {}),
            **({"regions_not_in_plane": absent} if absent else {}),
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
                        f"{position:.3f} mm, drawn at "
                        f"true physical scale with {options.border_color} borders "
                        f"{options.border_thickness} output pixel(s) wide."
                    )
                )
            ),
            TOOL_MEDIA_PARTS_KEY: media_parts,
        }

    @with_display_doc(
        'per entry (every display option is an entry key): "overlay" '
        '(default: the section with the outlines on it), "side_by_side" (two '
        "images: the section, then the atlas image, same scale and crop), "
        '"checkerboard", "outlines" (the atlas lines and the section\'s own '
        'silhouette on black), "section", "template" (the atlas image alone) '
        'or "ab" (two overlays at the same crop: these parameters, then the '
        "section's stored transform, or identity when it has none)."
    )
    def adjust_transforms(entries: list[dict[str, Any]]) -> dict[str, Any]:
        """Set and show one to four independent sections in one undoable call.

        Each entry replaces the complete transform, including any previous
        spline or shear. A section may appear once per call; inspect its result
        before making a dependent correction in a later call. Call it as often
        as you need, on any section that has a position; the last call is what
        stays.

        Args:
            entries: One to four objects with id, rotation_deg (counter-clockwise
                about the pivot, degrees), scale_x, scale_y (multipliers about
                the pivot; 1.0 leaves the size alone), translate_x_mm (right),
                translate_y_mm (down). Optional per entry: pivot ("canvas",
                "tissue" or [fx, fy] fractions of the canvas), note, and any
                display option below.

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
        refusal = over_cap(len(entries))
        if refusal is not None:
            return refusal
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
                target = state.resolve(entry.get("id", ""))
                if target is not None and target.id in locked:
                    results.append({"status": "error", "error": "LOCKED", "id": target.id})
                    continue
                options = display(
                    PREVIEW_MODES,
                    {key: entry[key] for key in entry_display_keys if key in entry},
                    sections=[target] if target is not None else [],
                )
                if isinstance(options, dict):
                    results.append({**options, "id": str(entry.get("id", ""))})
                    continue
                result = _adjust_transform(
                    str(entry.get("id", "")),
                    entry.get("rotation_deg"),  # type: ignore[arg-type]
                    entry.get("scale_x"),  # type: ignore[arg-type]
                    entry.get("scale_y"),  # type: ignore[arg-type]
                    entry.get("translate_x_mm"),  # type: ignore[arg-type]
                    entry.get("translate_y_mm"),  # type: ignore[arg-type]
                    options,
                    entry.get("pivot", "canvas"),  # type: ignore[arg-type]
                    str(entry.get("note", "")),
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
        # Orientation (flip + quarter-turn) is part of in-plane alignment.
        box.tools.append(orient_slices)
        if spec.transform.automatic:
            box.tools.append(fit_affine)
        if spec.transform.interactive:
            box.tools.append(adjust_transforms)
        if spec.transform.angles:
            box.tools.append(set_cutting_angles)

    def trace_borders(id: str, prompt: str = "") -> dict[str, Any]:
        """Trace one slice's atlas borders onto its anatomy with the image model.

        Args:
            id: Section filename or corrected index, with a position and linear transform.
            prompt: The full image prompt for this section, edited from the base prompt.

        Starts the image call in the background and returns at once; the result is
        saved and checked at submit. The first result at a placement is reused.
        Does not fit a deformation.
        """
        from langslice import registration_tool

        record = state.resolve(id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SECTION", "id": str(id)}
        try:
            fingerprint = registration_tool.correction_fingerprint(state, ctx, record.id)
            if box.image_job_running(record.id, fingerprint):
                return {"status": "running", "id": record.id,
                        "message": "This section's image correction is already running."}
            result, job = registration_tool.start_correction(
                state, ctx, record.id,
                prompt=prompt,
                out=Path(ctx.results_path).parent / "nonlinear",
                provider=spec.nonlinear.provider,
                image_model=spec.nonlinear.image_model,
            )
        except ValueError as exc:
            return {"status": "error", "error": "INVALID_LINEAR_PLACEMENT",
                    "id": record.id, "message": str(exc)}
        except OSError as exc:
            return {"status": "error", "error": "IMAGE_CORRECTION_IO_ERROR",
                    "id": record.id, "message": str(exc)}
        if job is not None:
            box.start_image_job(
                record.id, result["geometry_fingerprint"], job,
                workers=registration_tool.MAX_CONCURRENT_IMAGE_CALLS,
            )
        if result != record.image_correction:
            snapshot()
            record.image_correction = result
            save_checkpoint(state, ctx.checkpoint_path)
        response = {key: result[key] for key in (
            "status", "error", "message", "cached", "prompt_edited", "attempt",
        ) if key in result}
        response["id"] = record.id
        if job is not None:
            response["message"] = (
                "Image call started in the background. Continue; submit waits for it."
            )
        return response

    def grep_atlas(query: str, section: str = "") -> dict[str, Any]:
        """Look regions up in the atlas hierarchy, like grepping the ontology.

        Args:
            query: Text matched case-insensitively against region acronyms and
                names (substring), or an exact acronym or numeric id.
            section: Optional filename or corrected index of a section with a
                position; each row then says whether the region (or any
                descendant) appears in the atlas plane at that placement.

        Returns:
            Rows of acronym, id, name, ancestry (root to parent, as acronyms)
            and descendant count, capped at 40 with the number left over.
        """
        text = str(query).strip()
        if not text:
            return {"status": "error", "error": "BAD_ARGS", "message": "Empty query."}
        structures = getattr(ctx.atlas, "structures", None)
        entries = list(structures.values()) if structures else []
        if not entries:
            return {"status": "error", "error": "NO_STRUCTURES",
                    "message": "This atlas has no region hierarchy."}
        present: set[int] | None = None
        note = ""
        if section != "":
            record = state.resolve(section)
            if record is None:
                return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [section]}
            if record.position_mm is None:
                note = f"{record.id} has no position yet, so in_section is omitted."
            else:
                present = plane_structure_ids(state, ctx, record.position_mm)
        rows, total = grep_structures(entries, text, present, limit=GREP_ATLAS_LIMIT)
        result: dict[str, Any] = {"status": "ok", "query": text, "matches": total, "rows": rows}
        if total > len(rows):
            result["more"] = total - len(rows)
        if note:
            result["note"] = note
        return result

    # --- the deformable fit (nonlinear) -----------------------------------

    engine_option = spec.nonlinear.engine

    def resolve_choice(values: dict[str, Any]) -> deformation.Choice | dict[str, Any]:
        """One candidate's settings, validated, or the refusal naming the fix."""
        from typing import get_args

        from langslice.deformable.settings import Detail, Stiffness
        from langslice.linear.display import available_atlas_images

        requested = str(values.get("engine") or "").strip().lower()
        if engine_option != "either":
            if requested and requested != engine_option:
                return {"status": "error", "error": "ENGINE_FIXED", "engine": engine_option,
                        "message": f"The user set the engine to {engine_option} for this run."}
            chosen = engine_option
        else:
            chosen = requested or ("ants" if deformation.ants_available() else "elastix")
        if chosen not in deformation.ENGINES:
            return {"status": "error", "error": "BAD_ENGINE", "engines": list(deformation.ENGINES)}
        if chosen == "ants" and not deformation.ants_available():
            return {"status": "error", "error": "UNAVAILABLE", "message": deformation.ANTS_MISSING
                    + ("; engine 'elastix' is available." if engine_option == "either" else ".")}
        stiffness = str(values.get("stiffness") or "medium").strip().lower()
        if stiffness not in get_args(Stiffness):
            return {"status": "error", "error": "BAD_STIFFNESS",
                    "stiffness": list(get_args(Stiffness))}
        detail = str(values.get("detail") or "standard").strip().lower()
        if detail not in get_args(Detail):
            return {"status": "error", "error": "BAD_DETAIL", "detail": list(get_args(Detail))}
        picked = str(values.get("section_image") or deformation.FIT_LOOK).strip()
        traced = picked in deformation.TRACED
        if traced and not traces_on:
            return {"status": "error", "error": "NO_IMAGE_MODEL",
                    "message": "This run has no image model, so there are no traced section "
                    "images; use 'fit' or a raw channel."}
        atlas_kind = str(values.get("atlas_image") or "").strip().lower() or (
            "borders" if traced else "ara")
        if atlas_kind not in deformation.ATLAS_CHOICES:
            return {"status": "error", "error": "BAD_ATLAS_IMAGE",
                    "atlas_images": list(available_atlas_images(ctx))}
        if atlas_kind not in available_atlas_images(ctx):
            return {"status": "error", "error": "ATLAS_IMAGE_UNAVAILABLE",
                    "message": f"The {atlas_kind} atlas image needs ABBA's cached Allen atlas "
                    f"matching {state.atlas}; this host has none.",
                    "available": list(available_atlas_images(ctx))}
        if traced and atlas_kind != "borders":
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "Traced section images are fitted against atlas_image 'borders'."}
        if picked == deformation.TRACED_BORDERS and chosen != "ants":
            return {"status": "error", "error": "LABEL_MAP_ANTS_ONLY",
                    "message": "traced_borders (the traced lines as named regions) needs the "
                    "ANTs engine; traced_lines works with either engine."}
        return deformation.Choice(section_image=picked, atlas_image=atlas_kind, engine=chosen,
                                  stiffness=stiffness, detail=detail)

    def keep_linear_placements(targets: list[SliceState], reason: str) -> dict[str, Any]:
        """Record that each section's linear placement stands: no warp, a reason.

        Held on ``SliceState.deformation`` with the placement's ``linear_key``,
        so it satisfies `submit` like an applied fit, and a later change to the
        placement clears it the same way. One undo step.
        """
        refused = [
            {"id": record.id, "status": "error", "error": "INVALID_LINEAR_PLACEMENT",
             "message": "keep_linear needs a position and a transform."}
            for record in targets if record.position_mm is None or record.transform is None
        ]
        if refused:
            return {"status": "error", "error": "NOTHING_WRITTEN", "results": refused}
        snapshot()
        for record in targets:
            record.deformation = {"keep_linear": reason,
                                  "linear_key": deformation.linear_key(state, record)}
        return {**commit(*(record.id for record in targets)), "applied": True,
                "keep_linear": reason}

    def fit_deformable_impl(
        sections: list[str],
        include: list[str],
        exclude: list[str],
        start: str,
        section_image: str,
        atlas_image: str,
        engine: str,
        stiffness: str,
        detail: str,
        candidates: list[dict[str, Any]],
        mode: str,
        zoom: list[float],
        atlas_opacity: float,
        regions: list[str],
        outlines: str,
        border_color: str,
        border_thickness: float,
        resolution: int = 0,
        keep_linear: str = "",
    ) -> dict[str, Any]:
        store = deformation_store()
        if not isinstance(sections, (list, tuple)) or not sections:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "sections must name one or more sections"}
        named, unknown = resolve_many(list(sections))
        if unknown:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        targets = list({record.id: record for record in named}.values())
        reason = str(keep_linear or "").strip()
        if reason:
            return keep_linear_placements(targets, reason)
        if len(targets) > MAX_VIEW_SLICES:
            return {"status": "error", "error": "TOO_MANY_SECTIONS",
                    "max_sections": MAX_VIEW_SLICES}
        begin = str(start or "linear").strip().lower()
        if begin not in deformation.STARTS:
            return {"status": "error", "error": "BAD_START", "starts": list(deformation.STARTS)}
        kept = region_names(include, "include")
        if isinstance(kept, dict):
            return kept
        dropped = region_names(exclude, "exclude")
        if isinstance(dropped, dict):
            return dropped
        overlap = sorted({name.lower() for name in kept} & {name.lower() for name in dropped})
        if overlap:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "A region cannot be both included and excluded: "
                    + ", ".join(overlap)}
        variants = list(candidates or [])
        if len(variants) > deformation.MAX_CANDIDATES:
            return {"status": "error", "error": "TOO_MANY_CANDIDATES",
                    "max_candidates": deformation.MAX_CANDIDATES}
        allowed = set(deformation.CANDIDATE_KEYS)
        if engine_option != "either":
            allowed.discard("engine")
        for variant in variants:
            if not isinstance(variant, dict) or set(variant) - allowed:
                return {"status": "error", "error": "BAD_CANDIDATE",
                        "candidate_keys": sorted(allowed)}
        base = {"engine": engine, "stiffness": stiffness, "detail": detail,
                "section_image": section_image, "atlas_image": atlas_image}
        choices: list[deformation.Choice] = []
        for variant in variants or [{}]:
            choice = resolve_choice({**base, **variant})
            if isinstance(choice, dict):
                return choice
            choices.append(choice)
        if len(choices) * len(targets) > deformation.MAX_FITS_PER_CALL:
            return {"status": "error", "error": "TOO_MANY_FITS",
                    "max_fits": deformation.MAX_FITS_PER_CALL,
                    "requested": len(choices) * len(targets)}
        options = display(deformation.MODES, dict(
            mode=mode, zoom=zoom, atlas_opacity=atlas_opacity, regions=regions,
            outlines=outlines, border_color=border_color, border_thickness=border_thickness,
            resolution=resolution,
        ))
        if isinstance(options, dict):
            return options
        applying = len(choices) == 1

        rows: list[dict[str, Any]] = []
        jobs: list[deformation.Job] = []
        #: Per traced section, the stain and the trace's lines on the fit grid.
        traced_views: dict[str, tuple[Any, Any]] = {}
        trace_deadline = time.monotonic() + deformation.TRACE_WAIT_S
        for record in targets:
            try:
                grid = deformation.fit_grid(state, ctx, record)
            except (ValueError, OSError) as exc:
                rows.append({"id": record.id, "status": "error",
                             "error": "INVALID_LINEAR_PLACEMENT", "message": str(exc)})
                continue
            previous = None
            previous_key: str | None = None
            if begin == "current":
                previous = store.current(state, record)
                if previous is None:
                    rows.append({"id": record.id, "status": "error", "error": "NO_DEFORMATION",
                                 "message": "start='current' composes onto the section's "
                                 "applied deformation; this section has none."})
                    continue
                previous_key = str((record.deformation or {}).get("key"))
            channels = ctx.section_channels(record.id)[0]
            running = False
            if any(choice.section_image in deformation.TRACED for choice in choices):
                # A traced image waits for the section's trace still running.
                landed = record.id in box.image_jobs
                running = not box.wait_image_job(state, record.id,
                                                 trace_deadline - time.monotonic())
                if landed and not running:
                    save_checkpoint(state, ctx.checkpoint_path)
            for number, choice in enumerate(choices, start=1):
                failure = {"id": record.id, "status": "error", "settings": choice.echo(),
                           **({} if applying else {"candidate": number})}
                if (choice.section_image not in (deformation.FIT_LOOK, *deformation.TRACED)
                        and choice.section_image not in channels):
                    rows.append({**failure, "error": "UNKNOWN_CHANNEL",
                                 "channels": list(channels)})
                    continue
                try:
                    image, identity = deformation.stain_image(ctx, state, grid,
                                                              choice.section_image)
                    lines = None
                    if choice.section_image in deformation.TRACED:
                        lines, trace = deformation.traced_lines(
                            state, ctx, grid, running=running,
                            waited_s=deformation.TRACE_WAIT_S)
                        identity = {**identity, "trace": trace}
                        traced_views.setdefault(record.id, (image, lines))
                    settings = choice.settings(kept, dropped)
                except deformation.FitRefusal as refusal:
                    rows.append({**failure, **refusal.payload, "id": record.id})
                    continue
                except (ValueError, OSError) as exc:
                    rows.append({**failure, "error": "BAD_SETTINGS", "message": str(exc)})
                    continue
                key = deformation.cache_key(state, grid, settings, identity, previous_key)
                cached = store.get(record.id, key)
                jobs.append(deformation.Job(
                    grid=grid, choice=choice, settings=settings, key=key,
                    image_identity=identity, previous=previous, image=image, lines=lines,
                    result=cached, cached=cached is not None,
                ))
                rows.append({"id": record.id, "job": len(jobs) - 1,
                             **({} if applying else {"candidate": number})})
        deformation.run_jobs(ctx, jobs)

        before = state.to_dict()
        wrote_any = False
        parts: list[types.Part] = []
        failed: list[dict[str, str]] = []
        highlight = [name for name, _ids in options.regions] or list(kept)
        color, thickness = normalize_border_style(options.border_color, options.border_thickness)
        style = deformation.Style(
            zoom=() if options.full_view else tuple(options.zoom), highlight=tuple(highlight),
            marked=dropped, outlines=options.outlines, color=color, thickness=thickness,
            atlas_opacity=options.atlas_opacity, long_edge=options.long_edge,
        )
        for row in rows:
            index = row.pop("job", None)
            if index is None:
                continue
            job = jobs[index]
            outcome = job.result
            row["settings"] = job.choice.echo()
            if not isinstance(outcome, deformation.DeformableRecord):
                row.update(status="error", error="FIT_FAILED",
                           message=getattr(outcome, "error", "no result"))
                continue
            store.put(job.key, outcome)
            numbers = deformation.summary(outcome, job.previous)
            record = job.grid.record
            row.update(status="ok", engine_settings=deformation.engine_settings(job.settings),
                       **numbers, runtime_s=round(float(outcome.engine.get("runtime_s", 0.0)), 1),
                       cached=job.cached)
            if applying:
                linear = deformation.linear_key(state, record)
                outcome.provenance = deformation.provenance(
                    job.grid, job.choice, kept, dropped, begin, job.image_identity, linear)
                held = record.deformation or {}
                if held.get("key") == job.key:
                    row["written"] = False
                else:
                    try:
                        folder = store.save(record.id, job.key, outcome)
                    except OSError as exc:
                        row.update(status="error", error="RECORD_WRITE_FAILED", message=str(exc))
                        continue
                    record.deformation = deformation.reference(
                        folder=folder, key=job.key, linear=linear, record=outcome,
                        choice=job.choice, include=kept, exclude=dropped, start=begin,
                        previous=held, numbers=numbers,
                    )
                    row["written"] = True
                    wrote_any = True
                row["steps"] = len((record.deformation or {}).get("steps") or [])
            heading = (f"{record.id}  " + ("applied" if applying else
                       f"candidate {row['candidate']}/{len(choices)}")
                       + f": {job.choice.engine} {job.choice.stiffness} {job.choice.detail}")
            detail_line = (f"{job.choice.section_image} vs {job.choice.atlas_image}, start {begin}"
                           + (f", include {','.join(kept)}" if kept else "")
                           + (f", exclude {','.join(dropped)}" if dropped else ""))
            kind = job.choice.atlas_image
            first = len(parts)
            try:
                images = [deformation.picture(ctx, job.image, outcome, warped=True, style=style,
                                              atlas_image=kind, title=f"{heading}\n{detail_line}")]
                if options.mode == "ab":
                    if job.previous is not None:
                        images.append(deformation.picture(
                            ctx, job.image, job.previous, warped=True, style=style,
                            atlas_image=kind,
                            title=f"{record.id}  before: the deformation it started from"))
                    else:
                        images.append(deformation.picture(
                            ctx, job.image, outcome, warped=False, style=style, atlas_image=kind,
                            title=f"{record.id}  before: the linear placement"))
                parts.extend(image_to_part(image) for image in images)
            except Exception as exc:
                logger.warning("fit_deformable picture failed for %s", record.id, exc_info=True)
                del parts[first:]
                failed.append({"id": record.id, "message": str(exc)})
                continue
            row["image_indexes"] = list(range(first, len(parts)))
        if wrote_any:
            push(before)
            save_checkpoint(state, ctx.checkpoint_path)
        # Each traced section's trace, once per call, so it can be reviewed.
        traces: list[dict[str, Any]] = []
        for section_id, (image, lines) in traced_views.items():
            try:
                picture = deformation.trace_picture(
                    image, lines, style=style,
                    title=f"{section_id}  trace_borders result: the image model's lines")
            except Exception as exc:
                logger.warning("trace picture failed for %s", section_id, exc_info=True)
                failed.append({"id": section_id, "message": str(exc)})
                continue
            traces.append({"id": section_id, "image_indexes": [len(parts)]})
            parts.append(image_to_part(picture))
        succeeded = [row for row in rows if row.get("status") == "ok"]
        view = {key: value for key, value in options.echo().items()
                if key not in ("section_image", "atlas_image")}
        result: dict[str, Any] = {
            "status": "ok" if succeeded else "error",
            **({} if succeeded else {"error": "NOTHING_FITTED"}),
            "applied": applying,
            "results": rows,
            "view": view,
            "render_failed": failed,
            **({"traces": traces} if traces else {}),
            "description": (
                ("Applied: each section's deformation is now this fit (one undo step). "
                 if applying else "Preview: nothing was written. ")
                + "One picture per result (image_indexes): the final atlas borders on the "
                "section image the fit read"
                + (", included/`regions` borders strong over faint outlines" if highlight else "")
                + (", excluded regions in pink" if dropped else "")
                + ("; in ab mode then what the fit started from" if options.mode == "ab" else "")
                + ("; then, per traced section (`traces`), the image model's traced lines "
                   "on the section" if traces else "")
                + "."
            ),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        return result

    def fit_deformable(
        sections: list[str],
        include: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        exclude: list[str] = [],  # noqa: B006
        start: str = "linear",
        section_image: str = "fit",
        atlas_image: str = "",
        engine: str = "",
        stiffness: str = "medium",
        detail: str = "standard",
        candidates: list[dict[str, Any]] = [],  # noqa: B006
        keep_linear: str = "",
        mode: str = "borders",
        zoom: list[float] = [],  # noqa: B006
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 1.0,
        resolution: int = 0,
    ) -> dict[str, Any]:
        """Fit a deformation of the placed atlas onto sections, on top of their linear placement.

        A library engine bends the atlas, as linearly placed, onto the
        section image. A call with several candidates previews them all
        (run concurrently) and writes nothing. A call with exactly one
        setting (no candidates, or one) APPLIES it as each section's
        deformation, reusing the result of an identical earlier fit instead
        of recomputing; that write is undoable and checkpointed. Any later
        change to a section's position, orientation, cutting angles or
        transform clears its deformation. Needs a position and a transform.
        With keep_linear, no fit runs: each named section records that its
        linear placement stands, with that reason.

        Args:
            sections: Filenames or corrected indices (up to 4; at most 8 fits
                per call, sections times candidates).
            include: Regions (acronyms or ids, descendants included) to focus
                on: only they and a 300 um margin are fitted. Empty fits the
                whole section.
            exclude: Regions removed from the atlas side (e.g. tissue that is
                missing from the section), descendants included.
            start: "linear" (from the linear placement) or "current" (compose
                onto the section's applied deformation: region-by-region steps).
            section_image: "fit" (the section's fit appearance), a raw channel
                name, "traced_borders" (the section's trace_borders result at
                this placement, its lines turned into named regions; ANTs) or
                "traced_lines" (those lines as lines, against atlas borders).
                A traced image waits for a trace still running (up to
                TRACE_WAIT minutes) and the reply adds the trace drawn on the
                section.
            atlas_image: "ara", "borders" or "nissl" (hosts with ABBA's
                atlas). Empty: "borders" for traced images, else "ara".
            engine: "ants" or "elastix"; empty is ANTs when installed.
            stiffness: "soft", "medium", "firm" or "stiff".
            detail: "coarse" (40 um), "standard" (20 um) or "fine" (10 um).
            candidates: 2 to 4 objects, each overriding any of stiffness,
                detail, section_image, atlas_image and engine for one variant.
            keep_linear: A reason the named sections' linear placement stands
                without a deformation. Given, nothing is fitted: each section
                records it at its current placement (one undo step; submit
                accepts it like an applied fit; a placement change clears it).
            mode: "borders" (the fitted borders on the section) or "ab" (that,
                then what the fit started from).
            zoom: [x0, y0, x1, y1] fractions of the section; empty is all.
            atlas_opacity: 0..1, the warped atlas image (ara or nissl) under
                the lines.
            regions: Regions drawn at full strength; empty is the include list
                (or every border). Excluded regions are drawn in pink.
            outlines: "all", "outer" or "none" for the other borders.
            border_color: Named or #RRGGBB. border_thickness: 0.25..8 px.

        Returns:
            Per section and candidate: the settings and engine numbers used,
            displacement (max and median, mm, over the tissue), fold fraction,
            plausibility flags (regions compressed, expanded, vanished or
            folded beyond limits) and image_indexes into the pictures: the
            final borders drawn on the section image the fit read. Traced
            sections add `traces`: each one's trace drawn on the section.
        """
        return fit_deformable_impl(
            sections, include, exclude, start, section_image, atlas_image, engine, stiffness,
            detail, candidates, mode, zoom, atlas_opacity, regions, outlines, border_color,
            border_thickness, resolution, keep_linear,
        )

    def fit_deformable_fixed(
        sections: list[str],
        include: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        exclude: list[str] = [],  # noqa: B006
        start: str = "linear",
        section_image: str = "fit",
        atlas_image: str = "",
        stiffness: str = "medium",
        detail: str = "standard",
        candidates: list[dict[str, Any]] = [],  # noqa: B006
        keep_linear: str = "",
        mode: str = "borders",
        zoom: list[float] = [],  # noqa: B006
        atlas_opacity: float = 0.0,
        regions: list[str] = [],  # noqa: B006
        outlines: str = "",
        border_color: str = "yellow",
        border_thickness: float = 1.0,
        resolution: int = 0,
    ) -> dict[str, Any]:
        return fit_deformable_impl(
            sections, include, exclude, start, section_image, atlas_image, "", stiffness,
            detail, candidates, mode, zoom, atlas_opacity, regions, outlines, border_color,
            border_thickness, resolution, keep_linear,
        )

    doc = (fit_deformable.__doc__ or "").replace(
        "TRACE_WAIT minutes", f"{deformation.TRACE_WAIT_S / 60:g} minutes")
    if not traces_on:
        # No image model: no traced section images to offer.
        for traced_text, stain_text in _STAIN_ONLY_DOC:
            doc = doc.replace(traced_text.replace(
                "TRACE_WAIT minutes", f"{deformation.TRACE_WAIT_S / 60:g} minutes"), stain_text)
    fit_deformable.__doc__ = doc
    # The user fixed the engine: the same tool without the engine argument.
    fit_deformable_fixed.__name__ = fit_deformable_fixed.__qualname__ = "fit_deformable"
    fit_deformable_fixed.__doc__ = (fit_deformable.__doc__ or "").replace(
        '            engine: "ants" or "elastix"; empty is ANTs when installed.\n', "",
    ).replace("A library engine", f"The {engine_option} engine")

    if spec.has("nonlinear"):
        if traces_on:
            box.tools.append(trace_borders)
        box.tools.append(grep_atlas)
        box.tools.append(fit_deformable if engine_option == "either" else fit_deformable_fixed)

    box.tools.append(submit)
    lock = threading.Lock()
    # `resolution` exists only where the user left picture size to the agent.
    level = resolution_level(ctx)
    box.tools = [
        _serialized(_clears_stale_deformations(resolution_argument(tool, level), state, ctx),
                    lock, state=state, on_event=on_event)
        for tool in box.tools
    ]
    return box
