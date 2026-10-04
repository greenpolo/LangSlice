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

The tools are a door over the core and the job: they check arguments, keep
the look-before-commit gates and the delivery bookkeeping, call the
operations (:mod:`langslice.ops`) and the core picture builders
(:mod:`langslice.core`), and word the reply. Their pictures are plain PIL
images (and lines of text) under ``TOOL_MEDIA_PARTS_KEY``; the ADK driver
packages them as message parts (:func:`langslice.adk.media.packaged`), the
MCP server as content blocks, so this module never imports ``google.genai``.
"""

from __future__ import annotations

import functools
import importlib.util
import inspect
import logging
import math
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from PIL import Image

from langslice.adk import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.affine import denormalized_affine
from langslice.core import layers, placement
from langslice.core.canvas import VIEW_MODES, normalize_border_style
from langslice.core.captions import caption
from langslice.core.pictures import atlas_view_picture, section_picture, stack_review
from langslice.core.placement import (
    FRAMED_PLACEMENT_MODES,
    PLACEMENT_MODES,
    WARPED_PLACEMENT_MODES,
)
from langslice.core.sections import render_slice
from langslice.core.sizes import MAX_IMAGES_PER_CALL, resolution_level
from langslice.core.status import compact_rows, status_rows
from langslice.linear import appearance as looks
from langslice.linear import deformation
from langslice.linear.arguments import (
    Candidate,
    DamageEntry,
    FixedCandidate,
    OrientEntry,
    PlacementEntry,
    PositionEntry,
    TransformEntry,
    View,
    argument_refusal,
)
from langslice.linear.atlas_grep import GREP_ATLAS_LIMIT, grep_structures, plane_structure_ids
from langslice.linear.deepslice import run_deepslice as _run_deepslice
from langslice.linear.display import (
    MODE_RULES,
    DisplayOptions,
    available_atlas_channels,
    framed_section,
    regions_in_plane,
)
from langslice.linear.job import HOST_TRANSFORM_KIND, Job
from langslice.linear.live import LiveCallback, _plain
from langslice.linear.opening import DEFAULT_IMAGE_LIMIT
from langslice.linear.spec import JobSpec
from langslice.linear.state import (
    IDENTITY_PARAMS,
    SliceState,
    StackState,
    normalize_to_atlas_order,
)
from langslice.linear.transform import (
    FIT_FRAME_KEY,
    FitFrame,
    fit_elastix,
    fit_silhouette,
)
from langslice.linear.view_options import Profile, parse_view, view_edge_limit, view_schema
from langslice.ops import appearance as ops_appearance
from langslice.ops import damage as ops_damage
from langslice.ops import deformable as ops_deformable
from langslice.ops import notes as ops_notes
from langslice.ops import order as ops_order
from langslice.ops import orientation as ops_orientation
from langslice.ops import positions as ops_positions
from langslice.ops import transforms as ops_transforms
from langslice.ops.refusal import Refused
from langslice.space import Plane

if TYPE_CHECKING:  # ponytail: import cycle — engine builds the toolbox
    from langslice.linear.engine import EngineContext

logger = logging.getLogger(__name__)

#: One item of a tool's media list (under ``TOOL_MEDIA_PARTS_KEY``): a
#: picture (captioned PIL image) or a line of text, in reading order. The
#: doors package them: the ADK tools through
#: :func:`langslice.adk.media.packaged`, MCP as content blocks.
Media = Image.Image | str

#: Sections one ``view_slices`` call may return, or candidate pairs per compare.
#: Separate reference comparisons return up to two images per candidate pair.
MAX_VIEW_SLICES = MAX_IMAGES_PER_CALL

#: Views ``adjust_transforms`` composes on top of the renderer's own
#: (:data:`~langslice.core.canvas.VIEW_MODES`): the A/B toggle, which is two
#: renders of one crop rather than one composition.
PREVIEW_MODES = (*VIEW_MODES, "ab")

#: Why a transform write's picture takes no ``deformation``.
_NEW_TRANSFORM = ("this tool writes a new linear transform, which clears any deformation; "
                  "view_placement shows the applied one")

#: Each picture tool's ``view``: its modes (first = default) and the keys that
#: mean something for it (:class:`langslice.linear.view_options.Profile`).
VIEW_SLICES_VIEW = Profile(("section", "channels"), atlas=False)
ORIENT_VIEW = Profile(("section",), atlas=False)
PREPROCESS_VIEW = Profile(
    ("section",), channels=False, atlas=False,
    channels_reason="preprocess shows the target's appearance, before and after this call",
)
PLACEMENT_VIEW = Profile(
    PLACEMENT_MODES, deformation=True,
    zoom_modes=tuple(mode for mode in PLACEMENT_MODES if mode not in FRAMED_PLACEMENT_MODES),
    atlas_defaults={"side_by_side": ("ara",)},
    deformation_modes=WARPED_PLACEMENT_MODES,
)
#: ``set_positions`` draws the same pictures, ``stacked`` by default.
SET_POSITIONS_VIEW = replace(
    PLACEMENT_VIEW,
    modes=("stacked", *[mode for mode in PLACEMENT_MODES if mode != "stacked"]),
)
VIEW_STACK_VIEW = Profile(("stacked",), zoom_modes=())
FIT_AFFINE_VIEW = Profile(VIEW_MODES, deformation_reason=_NEW_TRANSFORM)
ADJUST_VIEW = Profile(PREVIEW_MODES, deformation_reason=_NEW_TRANSFORM)
FIT_DEFORMABLE_VIEW = Profile(
    deformation.MODES, channels=False,
    channels_reason="fit_deformable draws on the image the fit read (fit_section)",
    deformation_reason="fit_deformable's pictures are the fits themselves",
)

# --- view_atlas ----------------------------------------------------------------

#: Atlas sections one ``view_atlas`` call may return. Anything past this is
#: dropped — and reported back, never silently.
MAX_VIEW_POSITIONS = MAX_IMAGES_PER_CALL


def _as_floats(values: list[Any]) -> list[float]:
    """Model output is a trust boundary: keep the numbers, skip the rest.

    One level of nesting is walked: a model occasionally emits
    ``positions_mm=[[1.5, 2.5]]``.
    """
    out: list[float] = []
    for value in values:
        if isinstance(value, (list, tuple)):
            out.extend(_as_floats(list(value)))
            continue
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            continue
    return out


def _clamp_and_dedupe(
    positions: list[float], *, pos_lo: float, pos_hi: float, dedupe_tol: float = 0.02
) -> list[float]:
    out: list[float] = []
    for value in positions:
        clamped = max(pos_lo, min(pos_hi, value))
        if any(abs(clamped - kept) <= dedupe_tol for kept in out):
            continue
        out.append(clamped)
    return out


#: ``view_atlas``'s picture options: the atlas alone, framed to its anatomy.
VIEW_ATLAS_PROFILE_MODES = ("template",)


def make_view_atlas(state: StackState, ctx: EngineContext, max_view_edge: int):
    """Build the ``view_atlas`` tool, closed over the run's atlas and plane;
    *max_view_edge* caps ``view.resolution`` (the driver model's largest image)."""
    pos_lo, pos_hi = ctx.position_range
    profile = Profile(
        VIEW_ATLAS_PROFILE_MODES, channels=False,
        channels_reason="view_atlas draws the atlas alone, no section",
    )

    def view_atlas(
        positions_mm: list[float],
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
    ) -> dict[str, Any]:
        """Look at atlas sections at the positions you name, at most 4 per call.

        Sections are rendered at the stack's current cutting angles, each
        labelled with its position (and the angles, when the stack is oblique)
        in its top-left corner. Ask for more than 4 and only the first 4 are
        shown; the rest come back under ``dropped_positions_mm`` with
        ``truncated: true``. Positions outside the atlas range are clamped,
        and positions within 0.02 mm of one already in the same call are
        coalesced.

        Args:
            positions_mm: Positions along the slicing axis, in millimetres.
            view: Picture options (described once in the job statement).
                Mode "template" only: the atlas alone, framed to its anatomy;
                atlas_channels default ["ara"], add "borders" for the region
                lines. No section is drawn, so channels does not apply.

        Returns:
            status/positions plus the atlas images, in the order requested.
        """
        requested = _as_floats(list(positions_mm or []))
        if not requested:
            return {"status": "error", "error": "BAD_ARGS"}
        options = parse_view(ctx, state, view, profile, max_edge=max_view_edge)
        if isinstance(options, dict):
            return options
        dropped = [round(value, 2) for value in requested[MAX_VIEW_POSITIONS:]]
        positions = _clamp_and_dedupe(
            requested[:MAX_VIEW_POSITIONS], pos_lo=pos_lo, pos_hi=pos_hi
        )
        if not positions:
            return {"status": "error", "error": "EMPTY_RESULT"}

        parts: list[Media] = [
            atlas_view_picture(ctx, state, position, options) for position in positions
        ]
        plural = "s" if len(positions) != 1 else ""
        result: dict[str, Any] = {
            "status": "ok",
            "positions_mm": [round(position, 2) for position in positions],
            "cutting_angles_deg": dict(state.cutting_angles_deg),
            # Each image carries its own burned-in label; the ordering note
            # says the same thing in the payload.
            "description": (
                f"Showing {len(positions)} atlas section{plural}: "
                + ", ".join(f"{position:.2f} mm" for position in positions)
                + ". The attached atlas images appear in that same order, each "
                "labelled with its position in its top-left corner."
            ),
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        if options.regions:
            absent = {
                f"{position:.2f}": missing for position in positions
                if (missing := [
                    name for name, _ids in options.regions
                    if name not in regions_in_plane(ctx, state, position, options)
                ])
            }
            if absent:
                result["regions_not_in_plane"] = absent
        if dropped:
            result["truncated"] = True
            result["dropped_positions_mm"] = dropped
            result["description"] += (
                f" You asked for {len(requested)} positions; only the first "
                f"{MAX_VIEW_POSITIONS} were shown. NOT shown to you: "
                + ", ".join(f"{position:.2f} mm" for position in dropped)
                + "."
            )
        return result

    return view_atlas


def _tool_target_ids(state: StackState, name: str, args: dict[str, Any]) -> list[str]:
    """Resolve host display targets before a tool can reorder the stack."""
    if name in {"view_stack", "status", "undo", "redo", "submit",
                "set_cutting_angles", "run_deepslice"} or (
                    name == "preprocess" and not args.get("slices")):
        return [record.id for record in state.in_order()]
    if name == "fit_affine" and not args.get("slices"):
        return [record.id for record in state.in_order()
                if record.position_mm is not None and not record.damaged
                and (record.transform or {}).get("kind") != HOST_TRANSFORM_KIND]
    refs = list(args.get("slices") or [])
    if args.get("id") not in (None, ""):
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
    on_event: LiveCallback | None = None, before: Callable[[], None] | None = None,
) -> Any:
    """Serialize execution and its host notifications under the same lock.

    Model tool-call announcements may arrive together. These optional events
    identify the tool actually executing, including stable filenames resolved
    before a reorder. They never enter model context or change its schema.
    *before* runs inside the lock ahead of the tool (the job's reload of a
    state file changed on disk).
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
            if before is not None:
                before()
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


def _clears_stale_deformations(tool: Any, job: Job) -> Any:
    """After any tool: drop deformations whose linear placement changed, and say so.

    The rule is the job's (:meth:`~langslice.linear.job.Job.clear_stale_deformations`):
    cleared in the same undo step as the write that made them stale, so
    `undo` restores the placement and the deformation together. The door
    says so in the reply.
    """

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        result = tool(*args, **kwargs)
        cleared = job.clear_stale_deformations()
        if cleared and isinstance(result, dict):
            result["deformation_cleared"] = cleared
        return result

    return run


def _saves_views(tool: Any, job: Job, ctx: Any) -> Any:
    """Save every picture *tool* returns in the job folder, with its layers.

    What the pictures show is noted while the tool runs
    (:func:`langslice.core.layers.collecting`); the job's view store
    (:class:`langslice.job.views.ViewStore`) encodes and writes them in the
    background, in the doors' encoding, so the doors' bytes are untouched.
    A successful `submit` waits for every picture to be written.
    """
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        with layers.collecting() as notes:
            result = tool(*args, **kwargs)
        media = result.get(TOOL_MEDIA_PARTS_KEY) if isinstance(result, dict) else None
        pictures = ([item for item in media if isinstance(item, Image.Image)]
                    if isinstance(media, list) else [])
        if pictures:
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
            except TypeError:
                arguments = dict(kwargs)
            context = arguments.pop("tool_context", None)
            noted = [(picture, layers.note_for(picture, notes)) for picture in pictures]
            placed = any(held is not None and held.panel is not None for _p, held in noted)
            try:
                job.views.save(
                    tool=tool.__name__, pictures=list(noted), arguments=_plain(arguments),
                    call_id=_plain(getattr(context, "function_call_id", None)),
                    atlas=ctx.atlas if placed else None,
                )
            except Exception:  # saving must never break a tool
                logger.warning("Could not queue the pictures of %s", tool.__name__,
                               exc_info=True)
        if tool.__name__ == "submit" and job.state.submitted:
            job.views.flush()
        return result

    return run


def _strict(tool: Any) -> Any:
    """Refuse unknown or misplaced arguments before *tool* runs.

    :func:`langslice.linear.arguments.argument_refusal` against the tool's
    signature as the model sees it: stray top-level names (direct calls; ADK
    and MCP check those at their own doors, since both drop them before a
    tool runs) and stray keys inside ``view`` and every ``entries`` /
    ``candidates`` dict (every caller).
    """
    names = list(inspect.signature(tool, eval_str=True).parameters)

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        given = dict(zip(names, args, strict=False))
        given.update(kwargs)
        refusal = argument_refusal(tool, given)
        if refusal is not None:
            return refusal
        return tool(*args, **kwargs)

    return run


@dataclass
class ToolBox:
    """The tools of one run over its :class:`~langslice.linear.job.Job`, plus
    what only a tool door keeps: the look-before-commit gates and which
    pictures the model has received.

    The job holds the state, undo/redo, the checkpoint, the submit gates and
    the image-correction jobs; the gates and the delivery bookkeeping here
    are consulted by the tools alone, never by a library call.
    """

    job: Job
    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    #: Every parameter set `adjust_transforms` was given this run, per section
    #: id, oldest first. Kept host-side; the growing history is not repeated
    #: in every tool result because it is already present in the trajectory.
    transform_history: dict[str, list[dict[str, float]]] = field(default_factory=dict)
    #: Positions each section was compared at since its last write, and
    #: whether `view_stack` has run since the last write (the gates). A
    #: position that undo, redo or a reload of the state file moves counts as
    #: a write (`forget_looks`); nothing else of this door's record is undone.
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
    #: The largest picture the driver model takes: the cap of
    #: ``view.resolution`` at image resolution "auto".
    max_view_edge: int = DEFAULT_IMAGE_LIMIT[0]

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


#: The ``fit_deformable`` docstring's recommendation for traced fit sections
#: (the 2026-10-01 ceiling test's best pictures and numbers); dropped where it
#: cannot apply (no image model, or the user fixed the engine to Elastix).
_RECOMMENDED_TRACED = (
    " With a\n"
    "                completed trace, traced_borders with the ANTs engine at medium\n"
    "                stiffness is the recommended pairing."
)
#: The ``fit_deformable`` docstring's line on the fit appearance, and the
#: pointer to ``preprocess`` added when the agent may set appearances.
_FIT_LOOK_DOC = (
    '            fit_section: What of the section the fit reads: "fit" (the\n'
    "                section's fit appearance; default"
)
_PREPROCESS_DOC = (
    ";\n                the preprocess tool, target \"fit\", sets it, e.g. to one raw channel"
)
#: ``fit_deformable`` docstring passages about traced fit sections and their
#: wording for a run without the image model (``nonlinear.provider`` "none").
_STAIN_ONLY_DOC: tuple[tuple[str, str], ...] = (
    (
        _FIT_LOOK_DOC + '), "traced_borders" (the\n'
        "                section's trace_borders result at this placement, its lines\n"
        '                turned into named regions; ANTs) or "traced_lines" (those\n'
        "                lines as lines, against atlas borders). A traced fit section\n"
        "                waits for a trace still running (up to TRACE_WAIT minutes)\n"
        "                and the reply adds the trace drawn on the section."
        + _RECOMMENDED_TRACED + "\n",
        _FIT_LOOK_DOC + ").\n",
    ),
    (
        '            fit_atlas: What of the atlas the fit reads. For "fit": "ara" (the\n'
        "                atlas's reference template; default) or \"nissl\" (a\n"
        "                Nissl-stained reference, hosts with ABBA's atlas). For traced\n"
        '                fit sections: "borders" (default).\n',
        '            fit_atlas: What of the atlas the fit reads: "ara" (the atlas\'s\n'
        '                reference template; default) or "nissl" (a Nissl-stained\n'
        "                reference, hosts with ABBA's atlas).\n",
    ),
    (" Traced sections add `traces`: each\n            one's trace drawn on the section.", ""),
)

# --- the toolbox ---------------------------------------------------------


def build_tools(
    state: StackState, ctx: EngineContext, spec: JobSpec, *,
    job: Job | None = None,
    on_event: LiveCallback | None = None,
    max_view_edge: int | None = None,
) -> ToolBox:
    """Build the tools this run's spec switches on, closed over *state*.

    The tools sit on *job*; without one, a job is made around *state* in the
    context's job folder (an empty undo history, nothing written until the
    first write). *max_view_edge* is the largest picture
    the driver model takes (None: the run model's lane,
    :func:`langslice.linear.view_options.view_edge_limit`).
    """
    if job is None:
        job = Job(state, spec, layout=ctx.layout, results_path=ctx.results_path)
    if job.state is not state or job.spec is not spec:
        raise ValueError("build_tools: the job must hold this state and spec")
    box = ToolBox(job, max_view_edge=int(max_view_edge or view_edge_limit(ctx)))
    pos_lo, pos_hi = ctx.position_range
    locked = job.locked
    over_cap = job.over_cap
    #: The image model is part of this run: trace_borders and traced images exist.
    traces_on = spec.nonlinear.uses_image_model

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

    def answered(*touched: str) -> dict[str, Any]:
        """A write's answer: ``ok`` and the rows it changed (the operation
        itself took the undo step and the checkpoint)."""
        return {"status": "ok", **changed(list(touched))}

    def forget_looks(before: dict[str, Any]) -> None:
        """A position moved by undo, redo or a reload is a write to it: that
        section needs a new view, and the stack a new review (the gates)."""
        held = {row["id"]: row.get("position_mm") for row in before.get("slices", [])}
        moved = [record.id for record in state.slices
                 if held.get(record.id) != record.position_mm]
        for name in moved:
            box.compared.pop(name, None)
        if moved:
            box.reviewed = False

    def sync_job() -> None:
        """Before every call: pick up a state file changed on disk."""
        before = job.sync()
        if before is not None:
            forget_looks(before)

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
        profile: Profile, view: Any, *,
        sections: list[SliceState] = (),  # type: ignore[assignment]
    ) -> DisplayOptions | dict[str, Any]:
        """One call's ``view``, validated once for every picture tool."""
        return parse_view(ctx, state, view, profile, max_edge=box.max_view_edge,
                          sections=sections)

    def region_names(values: Any, field_name: str) -> tuple[str, ...] | dict[str, Any]:
        """Region entries for `include`/`exclude`, checked against the atlas.

        An entry is an acronym or id, optionally with a side ("CTX:left",
        :mod:`langslice.atlas.sides`).
        """
        from langslice.atlas.sides import has_sides
        from langslice.deformable.atlas_images import resolve_entries

        if values is None:
            return ()
        if not isinstance(values, (list, tuple)):
            return {"status": "error", "error": "BAD_ARGS",
                    "message": f"{field_name} must be a list of acronyms or ids"}
        names = tuple(str(value).strip() for value in values if str(value).strip())
        if names:
            try:
                resolve_entries(ctx.atlas, names)
            except ValueError as exc:
                return {"status": "error", "error": "UNKNOWN_REGIONS", "message": str(exc),
                        "argument": field_name}
            if state.plane == "sagittal" and has_sides(names):
                return {"status": "error", "error": "NO_SIDES", "argument": field_name,
                        "message": "A sagittal section lies within one hemisphere, so a "
                        "region cannot be limited to one side."}
        return names

    def region_overlap(kept: tuple[str, ...], dropped: tuple[str, ...]) -> dict[str, Any] | None:
        """The refusal for regions both included and excluded (sides respected)."""
        from langslice.atlas.sides import overlapping

        overlap = overlapping(kept, dropped)
        if not overlap:
            return None
        return {"status": "error", "error": "BAD_ARGS",
                "message": "A region cannot be both included and excluded: " + ", ".join(overlap)}

    def section_part(record: SliceState, options: DisplayOptions) -> Image.Image:
        """One section as corrected (:func:`langslice.core.pictures.section_picture`)."""
        return section_picture(ctx, state, record, options)

    def absent_regions(position: float, options: DisplayOptions) -> list[str]:
        present = regions_in_plane(ctx, state, position, options)
        return [name for name, _ids in options.regions if name not in present]

    def view_slices(
        slices: list[str],
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
    ) -> dict[str, Any]:
        """Look at up to 4 named sections at higher resolution.

        Sections are rendered as corrected: any rotation and flip already
        applied, framed to their tissue the same way atlas sections are.
        Each image carries its corrected index and filename burned into its
        top-left corner.

        Args:
            slices: Filenames or corrected indices (max 4 per call).
            view: Picture options (described once in the job statement).
                Modes: "section" (default: the section in view.channels) or
                "channels" (one small tile per raw channel, unmodified, each
                labelled with its name). No atlas is drawn.

        Returns:
            status/slices/description plus the images, in the order asked.
        """
        if not slices:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "slices must name one or more sections"}
        known, unknown = resolve_many(list(slices)[:MAX_VIEW_SLICES])
        if not known:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        options = display(VIEW_SLICES_VIEW, view, sections=known)
        if isinstance(options, dict):
            return options
        parts = [section_part(record, options) for record in known]
        result: dict[str, Any] = {
            "status": "ok",
            "slices": [record.id for record in known],
            "unknown_ids": unknown,
            # Each image carries its own burned-in label; the ordering note
            # says the same thing in the payload.
            "description": (
                "Attached images are "
                + ", ".join(record.id for record in known)
                + ", in that order, rendered as corrected, each labelled "
                "'<corrected index>: <filename>' in its top-left corner"
                + ("; each is a strip of the section's raw channels, unmodified, left to "
                   "right in the order of `channels`." if options.mode == "channels" else ".")
            ),
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }
        if options.mode == "channels":
            result["channels"] = {record.id: list(ctx.section_channels(record.id)[0])
                                  for record in known}
        return result

    def note(text: str) -> dict[str, Any]:
        """Append one line to the run notes, which are saved with the results.

        Args:
            text: The note.
        """
        try:
            return {"status": "ok", "notes": ops_notes.add_note(job, text)}
        except Refused as refusal:
            return refusal.payload()

    def undo() -> dict[str, Any]:
        """Undo the last write. One tool call undoes as one step."""
        before = job.snapshot()
        if not job.undo():
            return {"status": "error", "error": "NOTHING_TO_UNDO"}
        forget_looks(before)
        return {"status": "ok", "undo_depth": len(job.undo_stack), **rows()}

    def redo() -> dict[str, Any]:
        """Redo the write that ``undo`` reversed."""
        before = job.snapshot()
        if not job.redo():
            return {"status": "error", "error": "NOTHING_TO_REDO"}
        forget_looks(before)
        return {"status": "ok", "redo_depth": len(job.redo_stack), **rows()}

    def mark_damaged(entries: list[DamageEntry]) -> dict[str, Any]:
        """Set or clear damage flags for sections with unreliable outlines.

        Damage here means the section outline massively deviates from the
        atlas: large missing chunks, a missing hemisphere or olfactory bulb,
        split or independently rotated hemispheres, displaced fragments.
        Bubbles, stains, low contrast and small tears with an intact outline
        are NOT damage.

        Args:
            entries: Objects with id (filename or corrected index), damaged
                (boolean, default True) and note. Set damaged=False to clear
                the flag and its note.

        Returns:
            The rows this call changed.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        if any(not isinstance(entry, dict)
               or not isinstance(entry.get("damaged", True), bool) for entry in entries):
            return {"status": "error", "error": "BAD_ARGS"}
        done = ops_damage.mark_damaged(job, entries)
        return {"marked": done.marked, "unmarked": done.unmarked, "unknown_ids": done.unknown,
                **({"rejected": done.rejected} if done.rejected else {}),
                **answered(*done.touched)}

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
        refusal = job.submit_errors(breaks)
        # With the image model, missing traces are reported before missing
        # deformations: a deformation may be fitted to its section's trace.
        if refusal is not None and not (
                traces_on and refusal.get("error") == "MISSING_DEFORMATIONS"):
            return refusal
        if spec.has("nonlinear") and traces_on:
            from langslice.registration_tool import correction_fingerprint

            pending = job.missing_image_corrections(
                lambda section_id: correction_fingerprint(state, ctx, section_id))
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

        before = job.snapshot()
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
        job.commit(before)
        if tool_context is not None:
            tool_context.actions.escalate = True
        return {"status": "ok", **rows()}

    # --- appearance (JobSpec.agent_preprocessing) ----------------------

    def channel_summary(records: list[SliceState]) -> Any:
        """The raw channel names: one list when every section shares it."""
        named = {record.id: list(ctx.section_channels(record.id)[0]) for record in records}
        distinct = {tuple(names) for names in named.values()}
        return list(next(iter(distinct))) if len(distinct) == 1 else named

    def preprocess(
        target: str = "both",
        slices: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        channel_weights: list[float] = [],  # noqa: B006
        clahe_clip: float = looks.DEFAULT_CLAHE_CLIP,
        clahe_tiles: int = looks.DEFAULT_CLAHE_TILES,
        n4: bool = False,
        denoise: bool = False,
        reset: bool = False,
        view: View = {},  # noqa: B006
    ) -> dict[str, Any]:
        """Set how sections look: for what you view, for what a fit reads, or both.

        Sets the appearance of the whole stack (no slices) or of named
        sections (overriding the stack's), for target "view" (every picture
        you are shown from now on), "fit" (the image a deformable fit reads)
        or "both"; each target keeps its own setting. The image is built from
        the section's raw channels: each channel with weight above zero is
        optionally N4 bias-field corrected and denoised (ANTs), contrast-
        enhanced by CLAHE, then the channels are blended by their weights.
        Until this is called both targets use the default appearance.
        Writes, checkpoints and can be undone; another call replaces the
        setting. Picture options never change it.

        Args:
            target: "view", "fit" or "both".
            slices: Filenames or corrected indices; empty sets the stack.
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
            view: Picture options (described once in the job statement).
                Mode "section" only; zoom applies. The pictures show the
                target's appearance, so channels and the atlas keys do not
                apply.

        Returns:
            The settings in force per target, the raw channels, and for each
            affected section (up to 4) two labelled pictures: BEFORE (the
            target's appearance before this call) then AFTER (as it is now).
        """
        chosen = str(target or "both").strip().lower()
        if chosen not in looks.TARGET_CHOICES:
            return {"status": "error", "error": "BAD_TARGET",
                    "targets": list(looks.TARGET_CHOICES)}
        targets = list(looks.TARGETS) if chosen == "both" else [chosen]
        if slices:
            scope, unknown = resolve_many(list(slices))
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
        if slices:
            shown = scope[:MAX_VIEW_SLICES]
        else:  # up to four sections spread evenly over the stack
            picks = np.linspace(0, len(scope) - 1, min(len(scope), MAX_VIEW_SLICES))
            shown = [scope[index] for index in sorted({int(round(v)) for v in picks})]
        options = display(PREPROCESS_VIEW, view, sections=shown)
        if isinstance(options, dict):
            return options

        pictured = targets[0]  # "both" writes one setting to both targets
        ids = [record.id for record in scope] if slices else None
        parts: list[Media] = []
        try:
            # BEFORE is drawn first, from the settings as they stand; AFTER
            # from the settings the write will leave. Written only once every
            # picture is drawn.
            earlier = [
                (record, looks.section_settings(state, pictured, record.id),
                 framed_section(ctx, state, record, options,
                                look=looks.section_settings(state, pictured, record.id)))
                for record in shown
            ]
            for record, was, picture in earlier:
                now = ops_appearance.planned_settings(state, pictured, ids, settings, record.id)
                label = f"{record.index_corrected}: {record.id}  {pictured} appearance"
                parts.append(layers.note(
                    caption(picture, f"{label}  BEFORE ({looks.describe(was)})"),
                    sections=(record.id,), mode="before", extra={"target": pictured}))
                parts.append(layers.note(caption(
                    framed_section(ctx, state, record, options, look=now),
                    f"{label}  AFTER ({looks.describe(now)})"),
                    sections=(record.id,), mode="after", extra={"target": pictured}))
        except Exception as exc:
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}
        written = ops_appearance.set_appearance(job, targets, ids, settings)
        return {
            "status": "ok",
            "targets": targets,
            "scope": ids or "stack",
            "settings": written.in_force,
            "channels": channel_summary(scope),
            "shown": [record.id for record in shown],
            "image_indexes": {record.id: {"before": 2 * slot, "after": 2 * slot + 1}
                              for slot, record in enumerate(shown)},
            "description": (
                "Attached, per section in this order: "
                + ", ".join(record.id for record in shown)
                + f" — BEFORE (the {pictured} appearance before this call) then AFTER "
                "(as it is now), each labelled; a null setting is the default appearance."
            ),
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }

    box.tools = [
        status,
        view_slices,
        make_view_atlas(state, ctx, box.max_view_edge),
        note,
        undo,
        redo,
    ]
    if spec.agent_damage:
        box.tools.append(mark_damaged)
    if spec.agent_preprocessing:
        box.tools.append(preprocess)

    # --- orientation (part of the transform task) ------------------------

    def orient_slices(
        entries: list[OrientEntry],
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
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
            view: Picture options (described once in the job statement).
                Mode "section" only: each section as it now stands, in
                view.channels. No atlas is drawn.

        Returns:
            The rows this call changed, and each changed section rendered as
            it now stands (up to 4), its corrected index and filename burned
            into its top-left corner.
        """
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        named, _ = resolve_many([entry.get("id", "") for entry in entries
                                 if isinstance(entry, dict)])
        options = display(ORIENT_VIEW, view, sections=named)
        if isinstance(options, dict):
            return options
        done = ops_orientation.orient_sections(job, entries)
        applied = done.applied
        parts: list[Media] = []
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
            "cleared_transforms": done.cleared_transforms,
            "unknown_ids": done.unknown,
            "rejected": done.rejected,
            **answered(*done.touched),
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

    def reorder_slices(slices: list[str], after: str = "start") -> dict[str, Any]:
        """Place one or more sections together in the requested order.

        Args:
            slices: Nonempty list of unique section filenames. These sections
                move as one block in the listed order; all unlisted sections
                keep their relative order. List the whole stack to set its order.
                Use filenames, never corrected indices: this call changes indices.
            after: Filename the block should follow, or "start" (default) to
                put it first. The anchor must not be in slices.

        Returns:
            Changed rows and moved ids. Only corrected indices change; positions
            and transforms are kept. The whole call is one undoable write.
        """
        try:
            done = ops_order.reorder(job, slices, after)
        except Refused as refusal:
            return refusal.payload()
        return {"moved": done.moved, **answered(*done.touched)}

    if spec.has("reorder"):
        box.tools.append(reorder_slices)

    # --- position -------------------------------------------------------

    def placement_pictures(
        record: SliceState,
        position: float,
        options: DisplayOptions,
        parts: list[Media],
        section_indexes: dict[str, int],
        working: placement.Working,
    ) -> dict[str, Any]:
        """Append one section-position pair's pictures
        (:func:`langslice.core.placement.placement_pictures`) to *parts*.

        ``side_by_side``: the section once per distinct id and one atlas per
        pair, mapped by the returned ``image_indexes``. Returns the pair's
        row fields (calibration, transform drawn, deformation drawn, image
        indexes).
        """
        placed = placement.placement_pictures(ctx, state, record, position, options, working,
                                              store=job.deformations)
        row = placed.row
        if placed.separate:
            # Encode both before mutating delivery bookkeeping.
            tissue_image, atlas_image = placed.images
            if record.id not in section_indexes:
                section_indexes[record.id] = len(parts)
                parts.append(tissue_image)
            row["image_indexes"] = {"section": section_indexes[record.id], "atlas": len(parts)}
            parts.append(atlas_image)
            return row
        parts.extend(placed.images)
        return row

    def set_positions(
        entries: list[PositionEntry],
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Write positions for one or more sections, and show each placement.

        Positions are in atlas-native millimetres along the slicing axis. A
        value outside the atlas range is clamped and reported back.

        Args:
            entries: ``[{"id": "<filename>", "position_mm": <number>}]``.
            view: Picture options (described once in the job statement).
                Modes as in `view_placement`; default "stacked".

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
        options = display(SET_POSITIONS_VIEW, view, sections=named)
        if isinstance(options, dict):
            return options
        wanted: list[tuple[str, float]] = []
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
            wanted.append((record.id, requested))

        done = ops_positions.set_positions(job, ctx, wanted)
        written = [{"id": name, "position_mm": round(value, 3)} for name, value in done.written]
        written_positions = [value for _name, value in done.written]
        clamped = [{"id": name, "requested_mm": round(requested, 3),
                    "clamped_to_mm": round(value, 3)}
                   for name, requested, value in done.clamped]
        for name in done.touched:
            box.compared.pop(name, None)
            box.reviewed = False

        if not written:
            return {
                "status": "error",
                "error": "NOTHING_WRITTEN",
                "unknown_ids": unknown,
                "rejected": rejected,
            }
        result = {
            "written": written,
            "unknown_ids": unknown,
            "rejected": rejected,
            "clamped": clamped,
            **answered(*done.touched),
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
        parts: list[Media] = []
        failed: list[dict[str, str]] = []
        rendered: list[str] = []
        delivery_id: str | None = None
        section_indexes: dict[str, int] = {}
        working: dict[str, tuple[Any, float, str]] = {}
        image_indexes: dict[str, Any] = {}
        atlas_bearing = (options.mode != "section" and options.full_view
                         and bool(options.atlas_channels))
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
        slices: list[str], allow_angle_change: bool, keep: list[str]
    ) -> dict[str, Any]:
        """Seed positions (and optionally angles) with DeepSlice.

        Args:
            slices: Sections to place; empty means every undamaged section.
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
            slice_ids=[str(item) for item in slices or []],
            allow_angle_change=bool(allow_angle_change),
        )

    def search_position(id: str, window_mm: float, angles: bool) -> dict[str, Any]:
        """Search the atlas around a section's current position. Writes nothing.

        Scores the section against resampled atlas planes and returns the best
        one it found.

        Args:
            id: Filename or corrected index. The section must already
                have a position.
            window_mm: Half-width of the position search, in millimetres.
            angles: True also searches the cutting angles; False holds them at
                the stack's current ones.

        Returns:
            The best position (and angles) with its score.
        """
        record = state.resolve(id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [id]}
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

    def view_placement(
        entries: list[PlacementEntry],
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
        tool_context: Any = None,
    ) -> dict[str, Any]:
        """Show sections in their complete current registration, or at candidate positions.

        Writes nothing. At most 4 section-position pairs per call. The
        physical modes draw the section on a millimetre-true canvas under its
        complete current registration: its in-plane transform (identity when
        it has none) and, at its own position, its applied deformation (a
        warped section; view.deformation "none" shows the linear placement
        alone). One image per pair.

        Args:
            entries: ``[{"id": "<filename or corrected index>",
                "positions_mm": [<mm>, ...]}]``. An empty or missing
                ``positions_mm`` means that section's current position.
            view: Picture options (described once in the job statement).
                Modes: "template" (default: the atlas alone on the section's
                own canvas, at its scale; the section is in the opening
                message), "overlay" (the section under its registration with
                the atlas lines on it), "checkerboard" (section and atlas
                image in alternating tiles), "outlines" (atlas lines and the
                section's silhouette on black), "section" (the registered
                section alone), "stacked" (one picture: the section as
                corrected over the atlas, each tissue-framed; no placement
                drawn) or "side_by_side" (separate original section and
                atlas images, tissue-framed, full view only, up to 8
                images). zoom and deformation apply to the physical modes.

        Returns:
            The section-position pairs shown, in order, each with its
            calibration, the transform drawn and whether a deformation was
            drawn, and the images per pair in that order.
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
        options = display(PLACEMENT_VIEW, view,
                          sections=list({record.id: record for record, _p in pairs}.values()))
        if isinstance(options, dict):
            return options
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
        parts: list[Media] = []
        failed: list[dict[str, Any]] = []
        delivery_id: str | None = None
        full_atlas_view = (options.mode != "section" and options.full_view
                           and bool(options.atlas_channels))
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

    def view_stack(
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
    ) -> dict[str, Any]:
        """The whole stack ordered by written position, each over its atlas match.

        One contact sheet: every section as a labelled thumbnail, in the
        order of the positions written so far (unplaced sections last), a
        placed section with the atlas section at its position pasted
        directly beneath it. The label carries the corrected index,
        filename, position and the signed distance to the next placed
        section. Then one plot of position against corrected index (damaged
        sections in red). Two images; `view_slices` shows any section large.

        Args:
            view: Picture options (described once in the job statement).
                Mode "stacked" only; atlas_channels default ["ara"]; no zoom.

        Returns:
            The rows in that order and the two images.
        """
        options = display(VIEW_STACK_VIEW, view, sections=state.in_order())
        if isinstance(options, dict):
            return options
        box.reviewed = True

        sheet, plot = stack_review(ctx, state, options)
        parts = [
            f"The {len(state.slices)} sections in the order of their "
            "written positions (unplaced last), each captioned "
            "'<index>: <filename>  <position>', over its atlas match:",
            sheet,
            "Position against corrected index:",
            plot,
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
    elif spec.has("transform") or spec.has("nonlinear"):
        # The read-only view of the complete registration (placement and
        # applied deformation), wherever there is a placement to look at.
        box.tools.append(view_placement)
    if spec.has("position"):
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
        ops_positions.set_cutting_angles(job, ctx, pitch, yaw)
        return answered()  # stack-wide: no section row changes

    def fit_affine(
        slices: list[str],
        method: str = "elastix",
        fit_atlas: str = "",
        include: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        exclude: list[str] = [],  # noqa: B006
        view: View = {},  # noqa: B006
    ) -> dict[str, Any]:
        """Fit an in-plane affine per section against its atlas section.

        "elastix" (default) refines the section's current transform (none
        yet: from no transform) by matching the section's fit appearance
        against an atlas image, inner anatomy included; it adjusts from there
        and does not search from scratch. "silhouette" fits the tissue
        outline to the atlas outline from scratch (outlines only). Without
        regions a damaged section is refused. Each fit is written as the
        section's transform (undoable, and `adjust_transforms` overwrites it).

        Args:
            slices: Filenames or corrected indices; empty means every
                positioned, undamaged section.
            method: "elastix" (default) or "silhouette".
            fit_atlas: The atlas image the elastix method matches: "ara" (the
                reference template; default) or "nissl" (a Nissl-stained
                reference, hosts with ABBA's atlas). Not for "silhouette".
            include: Regions (acronyms or ids, descendants included) to fit
                by: only the atlas within 300 um of them, against the tissue
                the fit lays there. With "silhouette" they count only where
                they reach the outline.
            exclude: Regions removed from the atlas side (e.g. tissue missing
                from the section), descendants included; the tissue the fit
                lays on them is left out too. With regions given, damaged
                sections are fitted. An include or exclude entry may name one
                side only, "CTX:left" or "CTX:right": left and right of the
                section as view_slices shows it.
            view: Picture options (described once in the job statement).
                Modes as in `adjust_transforms` without "ab"; default
                "overlay". Included regions are highlighted unless
                view.regions names others.

        Returns:
            Per-section overlap (iou; with regions, of the kept atlas and the
            tissue that corresponds), the transform as the five physical
            knobs about the canvas centre, the calibration the image was
            drawn with, a `regions` report when regions were given, and an
            image of each fitted section under its new transform. The
            generic changed row is omitted because it repeats the same fit
            identifiers and overlap.
        """
        chosen = str(method or "elastix").strip().lower()
        if chosen not in ("elastix", "silhouette"):
            return {
                "status": "error",
                "error": "BAD_ARGS",
                "message": "method must be 'elastix' or 'silhouette'.",
            }
        atlas_kind = str(fit_atlas or "").strip().lower()
        if atlas_kind and chosen == "silhouette":
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "fit_atlas is the image the elastix method matches; the "
                    "silhouette method fits outlines only. Leave fit_atlas out."}
        atlas_kind = atlas_kind or "ara"
        offered = [kind for kind in available_atlas_channels(ctx) if kind != "borders"]
        if atlas_kind not in ("ara", "nissl"):
            return {"status": "error", "error": "BAD_FIT_ATLAS", "fit_atlas": offered}
        if atlas_kind not in offered:
            return {"status": "error", "error": "FIT_ATLAS_UNAVAILABLE",
                    "message": f"The {atlas_kind} atlas image needs ABBA's cached Allen atlas "
                    f"matching {state.atlas}; this host has none.", "fit_atlas": offered}
        fitter = (functools.partial(fit_elastix, atlas_image=atlas_kind) if chosen == "elastix"
                  else fit_silhouette)
        kept = region_names(include, "include")
        if isinstance(kept, dict):
            return kept
        dropped = region_names(exclude, "exclude")
        if isinstance(dropped, dict):
            return dropped
        refusal = region_overlap(kept, dropped)
        if refusal is not None:
            return refusal
        restricted = bool(kept or dropped)

        if slices:
            targets, unknown = resolve_many(list(slices))
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
        options = display(FIT_AFFINE_VIEW, view, sections=targets)
        if isinstance(options, dict):
            return options
        if (kept and isinstance(view, dict) and "regions" not in view
                and MODE_RULES[options.mode].atlas):
            # The included regions are highlighted unless the call names others.
            options = display(FIT_AFFINE_VIEW, {**view, "regions": list(kept)}, sections=targets)
            if isinstance(options, dict):
                return options

        def draw_fit(record: SliceState, frame: FitFrame) -> list[Any]:
            """The fitted section, drawn from the fit's working frame and matrix.

            One-sided regions are highlighted with the sides the fit resolved
            (``frame.left``), so the picture shows what the fit used even
            after a large turn.
            """
            return placement.draw_canvas(
                ctx, state, record, frame.section, frame.um_per_px,
                float(record.position_mm or 0.0), frame.matrix, options, label=record.id,
                left=frame.left,
            ).images

        results: list[dict[str, Any]] = []
        parts: list[Media] = []
        fits: list[tuple[SliceState, dict[str, Any]]] = []
        for record in targets:
            if record.id in locked:
                results.append({"id": record.id, "status": "error", "error": "LOCKED"})
                continue
            if record.damaged and not restricted:
                results.append({"id": record.id, "status": "error", "error": "DAMAGED"})
                continue
            try:
                outcome = fitter(state, ctx, record, include=kept, exclude=dropped)
                frame = outcome.pop(FIT_FRAME_KEY, None)
                panels = draw_fit(record, frame) if frame is not None else []
            except Exception as exc:  # nothing is written for this section
                logger.warning("fit_affine failed for %s: %s", record.id, exc)
                results.append({"id": record.id, "status": "error",
                                "error": getattr(exc, "code", "RENDER_FAILED"),
                                "message": str(exc)})
                continue
            results.append(outcome)
            if outcome["status"] != "ok":
                continue
            fits.append((record, outcome))
            # Every fit returns its picture (run 15, 2026-09-10: a
            # 25-section fit pictured 4 and the model never saw 21).
            outcome["image_indexes"] = list(range(len(parts), len(parts) + len(panels)))
            parts.extend(panels)

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
        ops_transforms.set_transforms(job, {
            record.id: ops_transforms.fit_transform(chosen, outcome, include=kept,
                                                    exclude=dropped, fit_atlas=atlas_kind)
            for record, outcome in fits
        })
        for _record, outcome in fits:
            outcome.pop("params")  # the six raw numbers stay host-side
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
        shear: float | None = None,
    ) -> placement.Staged | dict[str, Any]:
        """One section, its calibrated canvas and the pivot
        (:func:`langslice.core.placement.stage`), or a refusal."""
        record = state.resolve(slice_id)
        if record is None:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": [slice_id]}
        if record.position_mm is None:
            return {"status": "error", "error": "NO_POSITION", "id": record.id}
        if shear is None:
            # Left out: the shear the section's transform already has stays.
            held = (record.transform or {}).get("physical") or {}
            shear = held.get("shear") or 0.0
        try:
            params = {
                "rotation_deg": float(rotation_deg),
                "scale_x": float(scale_x),
                "scale_y": float(scale_y),
                "translate_x_mm": float(translate_x_mm),
                "translate_y_mm": float(translate_y_mm),
                "shear": 0.0 if shear is None else float(shear),
            }
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        if not all(math.isfinite(value) for value in params.values()):
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "every transform number must be finite"}
        try:
            return placement.stage(ctx, state, record, params, pivot)
        except placement.StageFailure as failure:
            return {"status": "error", "error": failure.code, "message": str(failure)}

    def staged_views(
        staged: placement.Staged,
        params: dict[str, float] | np.ndarray,
        options: DisplayOptions,
        *,
        mode: str,
        pivot: tuple[float, float] | None,
        label: str = "",
        spline: dict[str, Any] | None = None,
    ) -> list[Any]:
        """One staged section's canvas pictures (the write is on its working frame)."""
        return placement.staged_views(ctx, state, staged, params, options, mode=mode,
                                      pivot=pivot, label=label, spline=spline).images

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
        shear: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """One section's new in-plane transform, drawn: ``(result, record to write)``.

        The record is the stored transform these parameters make
        (:func:`langslice.ops.transforms.interactive_transform`), None when
        the section already holds it (the same numbers again only re-draw)
        or nothing could be drawn. The section's flip and rotation flags are
        not touched. The batch tool writes every record as one undo step.
        """
        view = options.mode
        staged = stage(
            slice_id, rotation_deg, scale_x, scale_y, translate_x_mm,
            translate_y_mm, pivot, shear,
        )
        if isinstance(staged, dict):
            return staged, None
        record = staged.record
        previous = record.transform
        written = ops_transforms.interactive_transform(
            size=staged.section.size, um_per_px=staged.um_per_px,
            calibration=staged.calibration, pivot=staged.pivot_in_section,
            pivot_frac=staged.pivot_frac, knobs=staged.params, note=note,
        )
        wrote = not ops_transforms.same_transform(previous, written)

        stored = (previous or {}).get("physical")
        reference: dict[str, Any] | None = None
        try:
            if view == "ab":
                # The B side is drawn from the six numbers the section carried
                # before this call, when there were any: they are the exact
                # map, where the knobs the payload reports (shear included)
                # are rounded. Knobs alone (no six numbers) are drawn as given.
                before = (previous or {}).get("params")
                other: Any = dict(IDENTITY_PARAMS)
                if before is not None and len(before) == 6:
                    other = denormalized_affine(before, staged.section.size)
                elif isinstance(stored, dict):
                    other = {key: float(stored[key]) for key in IDENTITY_PARAMS}
                    if stored.get("shear"):
                        other["shear"] = float(stored["shear"])
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
            return {"status": "error", "error": "RENDER_FAILED", "message": str(exc)}, None

        history = box.transform_history.setdefault(record.id, [])
        history.append(dict(staged.params))
        image = f"atlas {options.atlas_name()}"
        under = (f", the {image} blended under it at opacity {options.atlas_opacity:g}"
                 if options.atlas_images and options.atlas_opacity > 0 else "")
        described = {
            "overlay": f"the section with the atlas lines over it{under}",
            "side_by_side": (
                f"two images at the same scale and the same crop: the section, then the {image}"
            ),
            "checkerboard": f"the section and the {image} in alternating tiles",
            "outlines": (
                f"the atlas lines and the section's own silhouette contour, on black{under}"
            ),
            "section": "the section alone, no atlas",
            "template": f"the {image} alone",
            "ab": (
                f"two overlays at the same crop{under}: first these parameters, then "
                + (
                    "the transform the section carried before this call"
                    if previous is not None
                    else "identity"
                )
            ),
        }[view]
        position = float(record.position_mm or 0.0)
        absent = absent_regions(position, options)
        if options.borders:
            lines = (
                ("The outline is the OUTER boundary of "
                 if options.outlines == "outer"
                 else "The outlines are the FAMILY regions of ")
                + f"the {state.plane} atlas section at {position:.3f} mm, drawn at "
                f"true physical scale with {options.border_color} borders "
                f"{options.border_thickness} output pixel(s) wide"
                + (", the named regions at full strength." if options.regions else ".")
            )
        elif options.regions:
            lines = ("Only the named regions' outlines are drawn, "
                     f"{options.border_color}, {options.border_thickness} output pixel(s) wide.")
        else:
            lines = "No atlas outlines are drawn."
        return {
            "status": "ok",
            "id": record.id,
            "position_mm": round(position, 3),
            "physical": written["physical"],
            "written": wrote,
            **({"ab_reference": reference} if reference is not None else {}),
            **({"regions_not_in_plane": absent} if absent else {}),
            "description": f"{record.id} under the transform above, {described}. {lines}",
            TOOL_MEDIA_PARTS_KEY: images,
        }, (written if wrote else None)

    def adjust_transforms(
        entries: list[TransformEntry],
        view: View = {},  # noqa: B006 — read, never mutated; ADK wants a value
    ) -> dict[str, Any]:
        """Set and show one to four independent sections in one undoable call.

        Each entry replaces the complete transform, including any previous
        spline; a shear left out is kept from the current transform. A section may
        appear once per call; inspect its result before making a dependent
        correction in a later call. Call it as often as you need, on any
        section that has a position; the last call is what stays.

        Args:
            entries: One to four objects with id, rotation_deg (counter-clockwise
                about the pivot, degrees), scale_x, scale_y (multipliers about
                the pivot; 1.0 leaves the size alone), translate_x_mm (right),
                translate_y_mm (down). Optional per entry: shear (a unitless
                slant applied before the rotation: each point moves sideways
                by shear times its distance below the pivot, in units of
                scale_x; the number fit_affine reports; left out, the
                section's current shear is kept, 0 sets none), pivot
                ("canvas", "tissue" or [fx, fy] fractions of the canvas) and
                note.
            view: Picture options (described once in the job statement), one
                for every entry's picture. Modes: "overlay" (default: the section
                with the atlas lines on it), "side_by_side" (two images: the
                section, then the atlas image, same scale and crop),
                "checkerboard", "outlines" (the atlas lines and the section's
                own silhouette on black), "section", "template" (the atlas
                alone) or "ab" (two overlays at the same crop: these
                parameters, then the section's stored transform, or identity
                when it has none).

        Returns:
            Per-section results with zero-based image_indexes into the attached
            labelled images, in entry order, and the `view` they were drawn
            with. All writes form one undo step. Repeating unchanged
            parameters only redraws, without an undo step.
        """
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
        named: list[SliceState] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            record = state.resolve(entry.get("id", ""))
            if record is not None:
                canonical.append(record.id)
                named.append(record)
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
        options = display(ADJUST_VIEW, view, sections=named)
        if isinstance(options, dict):
            return options

        results: list[dict[str, Any]] = []
        parts: list[Media] = []
        writes: dict[str, dict[str, Any]] = {}
        for entry in entries:
            if not isinstance(entry, dict):
                results.append({"status": "error", "error": "BAD_ARGS"})
                continue
            target = state.resolve(entry.get("id", ""))
            if target is not None and target.id in locked:
                results.append({"status": "error", "error": "LOCKED", "id": target.id})
                continue
            result, record_to_write = _adjust_transform(
                str(entry.get("id", "")),
                entry.get("rotation_deg"),  # type: ignore[arg-type]
                entry.get("scale_x"),  # type: ignore[arg-type]
                entry.get("scale_y"),  # type: ignore[arg-type]
                entry.get("translate_x_mm"),  # type: ignore[arg-type]
                entry.get("translate_y_mm"),  # type: ignore[arg-type]
                options,
                entry.get("pivot", "canvas"),  # type: ignore[arg-type]
                str(entry.get("note", "")),
                entry.get("shear"),
            )
            media = result.pop(TOOL_MEDIA_PARTS_KEY, [])
            result["image_indexes"] = list(range(len(parts), len(parts) + len(media)))
            parts.extend(media)
            if record_to_write is not None:
                writes[str(result["id"])] = record_to_write
            results.append(result)
        # Every section of the call, one undo step.
        ops_transforms.set_transforms(job, writes)
        successful = [row["id"] for row in results if row.get("status") == "ok"]
        return {
            "status": "ok" if successful else "error",
            **({} if successful else {"error": "NOTHING_ADJUSTED"}),
            "results": results,
            "view": options.echo(),
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
            if job.image_job_running(record.id, fingerprint):
                return {"status": "running", "id": record.id,
                        "message": "This section's image correction is already running."}
            result, call = registration_tool.start_correction(
                state, ctx, record.id,
                prompt=prompt,
                calls_dir=job.layout.image_correction_dir(record.id),
                provider=spec.nonlinear.provider,
                image_model=spec.nonlinear.image_model,
            )
        except ValueError as exc:
            return {"status": "error", "error": "INVALID_LINEAR_PLACEMENT",
                    "id": record.id, "message": str(exc)}
        except OSError as exc:
            return {"status": "error", "error": "IMAGE_CORRECTION_IO_ERROR",
                    "id": record.id, "message": str(exc)}
        if call is not None:
            job.start_image_job(
                record.id, result["geometry_fingerprint"], call,
                workers=registration_tool.MAX_CONCURRENT_IMAGE_CALLS,
            )
        result = job.portable(result)
        if result != record.image_correction:
            before = job.snapshot()
            record.image_correction = result
            job.commit(before)
        response = {key: result[key] for key in (
            "status", "error", "message", "cached", "prompt_edited", "attempt",
        ) if key in result}
        response["id"] = record.id
        if call is not None:
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

        from langslice.deformable.settings import Stiffness

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
        picked = str(values.get("fit_section") or deformation.FIT_LOOK).strip()
        traced = picked in deformation.TRACED
        if traced and not traces_on:
            return {"status": "error", "error": "NO_IMAGE_MODEL",
                    "message": "This run has no image model, so there are no traced fit "
                    "sections; use 'fit'."}
        if picked not in deformation.FIT_SECTIONS:
            offered = [deformation.FIT_LOOK, *(deformation.TRACED if traces_on else ())]
            return {"status": "error", "error": "BAD_FIT_SECTION", "fit_sections": offered,
                    "message": "A fit reads the section's fit appearance"
                    + (" or its traced borders" if traces_on else "")
                    + (". To fit one raw channel, set the fit appearance with preprocess "
                       "(target 'fit')." if spec.agent_preprocessing else ".")}
        available = available_atlas_channels(ctx)
        atlas_kind = str(values.get("fit_atlas") or "").strip().lower() or (
            "borders" if traced else "ara")
        if atlas_kind not in deformation.FIT_ATLASES:
            return {"status": "error", "error": "BAD_FIT_ATLAS",
                    "fit_atlases": list(available)}
        if atlas_kind not in available:
            return {"status": "error", "error": "FIT_ATLAS_UNAVAILABLE",
                    "message": f"The {atlas_kind} atlas image needs ABBA's cached Allen atlas "
                    f"matching {state.atlas}; this host has none.",
                    "available": list(available)}
        if traced and atlas_kind != "borders":
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "Traced fit sections are fitted against fit_atlas 'borders'."}
        if not traced and atlas_kind == "borders":
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "Atlas borders are for traced fit sections (the image model's "
                    "lines against the atlas's lines); the fit appearance is fitted against "
                    "a grayscale atlas image, 'ara' or 'nissl'."}
        if picked == deformation.TRACED_BORDERS and chosen != "ants":
            return {"status": "error", "error": "LABEL_MAP_ANTS_ONLY",
                    "message": "traced_borders (the traced lines as named regions) needs the "
                    "ANTs engine; traced_lines works with either engine."}
        return deformation.Choice(fit_section=picked, fit_atlas=atlas_kind, engine=chosen,
                                  stiffness=stiffness)

    def fit_deformable_impl(
        slices: list[str],
        include: list[str],
        exclude: list[str],
        start: str,
        fit_section: str,
        fit_atlas: str,
        engine: str,
        stiffness: str,
        candidates: list[Any],
        keep_linear: str,
        view: Any,
    ) -> dict[str, Any]:
        if not isinstance(slices, (list, tuple)) or not slices:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "slices must name one or more sections"}
        named, unknown = resolve_many(list(slices))
        if unknown:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        targets = list({record.id: record for record in named}.values())
        reason = str(keep_linear or "").strip()
        if reason:
            given = [name for name, value, default in (
                ("include", include, None), ("exclude", exclude, None),
                ("start", start, "linear"), ("fit_section", fit_section, "fit"),
                ("fit_atlas", fit_atlas, ""), ("engine", engine, ""),
                ("stiffness", stiffness, "medium"), ("candidates", candidates, None),
                ("view", view, None),
            ) if (value if default is None else (value or default) != default)]
            if given:
                return {"status": "error", "error": "BAD_ARGS", "given": given,
                        "message": "keep_linear records that the linear placement stands: it "
                        "runs no fit and draws no picture, so leave the fit settings, "
                        "candidates and view out."}
            try:
                kept_linear = ops_deformable.keep_linear(job, targets, reason)
            except Refused as refusal:
                return refusal.payload()
            return {**answered(*kept_linear.touched), "applied": True,
                    "keep_linear": kept_linear.reason}
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
        refusal = region_overlap(kept, dropped)
        if refusal is not None:
            return refusal
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
        base = {"engine": engine, "stiffness": stiffness,
                "fit_section": fit_section, "fit_atlas": fit_atlas}
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
        options = display(FIT_DEFORMABLE_VIEW, view, sections=targets)
        if isinstance(options, dict):
            return options
        done = ops_deformable.fit_deformable(job, ctx, targets, choices, include=kept,
                                             exclude=dropped, start=begin)
        applying = done.applied
        rows = done.rows

        parts: list[Media] = []
        failed: list[dict[str, str]] = []
        highlight = [name for name, _ids in options.regions] or list(kept)
        color, thickness = normalize_border_style(options.border_color, options.border_thickness)
        style = deformation.Style(
            zoom=() if options.full_view else tuple(options.zoom), highlight=tuple(highlight),
            marked=dropped, outlines=options.layer, color=color, thickness=thickness,
            atlas_opacity=options.atlas_opacity, long_edge=options.long_edge,
        )
        for fitted in done.fitted:
            row, fit, outcome = fitted.row, fitted.fit, fitted.outcome
            record = fit.grid.record
            heading = (f"{record.id}  " + ("applied" if applying else
                       f"candidate {row['candidate']}/{len(choices)}")
                       + f": {fit.choice.engine} {fit.choice.stiffness}")
            detail_line = (f"{fit.choice.fit_section} vs {fit.choice.fit_atlas}, start {begin}"
                           + (f", include {','.join(kept)}" if kept else "")
                           + (f", exclude {','.join(dropped)}" if dropped else ""))
            shown_atlas = options.atlas_images
            first = len(parts)
            try:
                images = [deformation.picture(ctx, fit.image, outcome, warped=True, style=style,
                                              atlas_images=shown_atlas,
                                              title=f"{heading}\n{detail_line}")]
                if options.mode == "ab":
                    if fit.previous is not None:
                        images.append(deformation.picture(
                            ctx, fit.image, fit.previous, warped=True, style=style,
                            atlas_images=shown_atlas,
                            title=f"{record.id}  before: the deformation it started from"))
                    else:
                        images.append(deformation.picture(
                            ctx, fit.image, outcome, warped=False, style=style,
                            atlas_images=shown_atlas,
                            title=f"{record.id}  before: the linear placement"))
                for index, image in enumerate(images):
                    layers.note(image, sections=(record.id,),
                                mode=("after" if index == 0 else "before")
                                if options.mode == "ab" else options.mode)
                parts.extend(images)
            except Exception as exc:
                logger.warning("fit_deformable picture failed for %s", record.id, exc_info=True)
                del parts[first:]
                failed.append({"id": record.id, "message": str(exc)})
                continue
            row["image_indexes"] = list(range(first, len(parts)))
        # Each traced section's trace, once per call, so it can be reviewed.
        traces: list[dict[str, Any]] = []
        for section_id, (image, lines) in done.traced.items():
            try:
                picture = deformation.trace_picture(
                    image, lines, style=style,
                    title=f"{section_id}  trace_borders result: the image model's lines")
            except Exception as exc:
                logger.warning("trace picture failed for %s", section_id, exc_info=True)
                failed.append({"id": section_id, "message": str(exc)})
                continue
            traces.append({"id": section_id, "image_indexes": [len(parts)]})
            parts.append(layers.note(picture, sections=(section_id,), mode="trace"))
        succeeded = [row for row in rows if row.get("status") == "ok"]
        result: dict[str, Any] = {
            "status": "ok" if succeeded else "error",
            **({} if succeeded else {"error": "NOTHING_FITTED"}),
            "applied": applying,
            "results": rows,
            "view": options.echo(),
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
        slices: list[str],
        include: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        exclude: list[str] = [],  # noqa: B006
        start: str = "linear",
        fit_section: str = "fit",
        fit_atlas: str = "",
        engine: str = "",
        stiffness: str = "medium",
        candidates: list[Candidate] = [],  # noqa: B006
        keep_linear: str = "",
        view: View = {},  # noqa: B006
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
            slices: Filenames or corrected indices (up to 4; at most 8 fits
                per call, slices times candidates).
            include: Regions (acronyms or ids, descendants included) to focus
                on: only they and a 300 um margin are fitted. Empty fits the
                whole section.
            exclude: Regions removed from the atlas side (e.g. tissue that is
                missing from the section), descendants included. An include
                or exclude entry may name one side only, "CTX:left" or
                "CTX:right": left and right of the section as this tool's
                pictures show it.
            start: "linear" (from the linear placement) or "current" (compose
                onto the section's applied deformation: region-by-region steps).
            fit_section: What of the section the fit reads: "fit" (the
                section's fit appearance; default), "traced_borders" (the
                section's trace_borders result at this placement, its lines
                turned into named regions; ANTs) or "traced_lines" (those
                lines as lines, against atlas borders). A traced fit section
                waits for a trace still running (up to TRACE_WAIT minutes)
                and the reply adds the trace drawn on the section. With a
                completed trace, traced_borders with the ANTs engine at medium
                stiffness is the recommended pairing.
            fit_atlas: What of the atlas the fit reads. For "fit": "ara" (the
                atlas's reference template; default) or "nissl" (a
                Nissl-stained reference, hosts with ABBA's atlas). For traced
                fit sections: "borders" (default).
            engine: "ants" or "elastix"; empty is ANTs when installed.
            stiffness: "soft", "medium" (default) or "firm".
            candidates: 2 to 4 objects, each overriding any of stiffness,
                fit_section, fit_atlas and engine for one variant.
            keep_linear: A reason the named sections' linear placement stands
                without a deformation. Given, nothing is fitted and nothing is
                drawn: each section records it at its current placement (one
                undo step; submit accepts it like an applied fit; a placement
                change clears it).
            view: Picture options (described once in the job statement).
                Modes: "borders" (default: the fitted borders on the image the
                fit read) or "ab" (that, then what the fit started from).
                atlas_channels default ["borders"]; add "ara" or "nissl" to
                see that atlas image, warped, under the lines at
                atlas_opacity. view.regions is drawn at full strength (empty:
                the include list); excluded regions are drawn in pink. The
                pictures are the image the fit read, so channels does not
                apply.

        Returns:
            Per section and candidate: the settings and engine numbers used,
            displacement (max and median, mm, over the tissue), fold fraction,
            plausibility flags (regions compressed, expanded, vanished or
            folded beyond limits; displacement outsized for the section) and
            image_indexes into the pictures: the final borders drawn on the
            section image the fit read. Traced sections add `traces`: each
            one's trace drawn on the section.
        """
        return fit_deformable_impl(
            slices, include, exclude, start, fit_section, fit_atlas, engine, stiffness,
            list(candidates or []), keep_linear, view,
        )

    def fit_deformable_fixed(
        slices: list[str],
        include: list[str] = [],  # noqa: B006 — read, never mutated; ADK wants a value
        exclude: list[str] = [],  # noqa: B006
        start: str = "linear",
        fit_section: str = "fit",
        fit_atlas: str = "",
        stiffness: str = "medium",
        candidates: list[FixedCandidate] = [],  # noqa: B006
        keep_linear: str = "",
        view: View = {},  # noqa: B006
    ) -> dict[str, Any]:
        return fit_deformable_impl(
            slices, include, exclude, start, fit_section, fit_atlas, "", stiffness,
            list(candidates or []), keep_linear, view,
        )

    doc = (fit_deformable.__doc__ or "").replace(
        "TRACE_WAIT minutes", f"{deformation.TRACE_WAIT_S / 60:g} minutes")
    if not traces_on:
        # No image model: no traced fit sections to offer.
        for traced_text, stain_text in _STAIN_ONLY_DOC:
            doc = doc.replace(traced_text.replace(
                "TRACE_WAIT minutes", f"{deformation.TRACE_WAIT_S / 60:g} minutes"), stain_text)
    if spec.agent_preprocessing:
        doc = doc.replace(_FIT_LOOK_DOC, _FIT_LOOK_DOC + _PREPROCESS_DOC)
    fit_deformable.__doc__ = doc
    # The user fixed the engine: the same tool without the engine argument.
    fit_deformable_fixed.__name__ = fit_deformable_fixed.__qualname__ = "fit_deformable"
    fixed_doc = (fit_deformable.__doc__ or "").replace(
        '            engine: "ants" or "elastix"; empty is ANTs when installed.\n', "",
    ).replace("A library engine", f"The {engine_option} engine").replace(
        "fit_section, fit_atlas and engine for one variant", "fit_section and fit_atlas for one "
        "variant")
    if engine_option != "ants":
        # traced_borders is ANTs-only, so its recommendation does not apply.
        fixed_doc = fixed_doc.replace(_RECOMMENDED_TRACED, "")
    fit_deformable_fixed.__doc__ = fixed_doc

    if spec.has("nonlinear"):
        if traces_on:
            box.tools.append(trace_borders)
        box.tools.append(grep_atlas)
        box.tools.append(fit_deformable if engine_option == "either" else fit_deformable_fixed)

    box.tools.append(submit)
    lock = threading.Lock()
    # `view.resolution` exists only where the user left picture size to the agent.
    level = resolution_level(ctx)
    box.tools = [
        _serialized(
            _saves_views(_clears_stale_deformations(_strict(view_schema(tool, level)), job),
                         job, ctx),
            lock, state=state, on_event=on_event, before=sync_job)
        for tool in box.tools
    ]
    return box
