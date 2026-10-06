"""The one toolbox: every tool the linear agent can be given, gated by the spec.

Conventions, applied to every tool: sections are addressed by filename or
corrected index; ordinary writes return the rows they changed (``status`` is
the whole table); every write is undoable and checkpoints. Transform writes
return their physical result and omit a generic row that would repeat it.

Tools report data: no advice, no interpretation, no strategy in any payload
(the job statement's "Method:" section is the one place for strategy). A
constraint that states a number is not coaching: refusals name the numbers
that caused them and stop there.

The tools are a door over the core and the job: they check arguments, keep
the look-before-commit gates and the delivery bookkeeping, call the
operations (:mod:`langslice.ops`) and the core picture builders
(:mod:`langslice.core`), and word the reply. Their pictures are plain PIL
images (and lines of text) under ``TOOL_MEDIA_PARTS_KEY``; the ADK driver
packages them as message parts (:func:`langslice.doors.tools.media.packaged`), the
MCP server as content blocks, so this module never imports ``google.genai``.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import inspect
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from langslice.agent.live import LiveCallback, plain
from langslice.core import appearance as looks
from langslice.core import deformation, layers
from langslice.core.canvas import VIEW_MODES
from langslice.core.captions import caption
from langslice.core.display import (
    MODE_RULES,
    DisplayOptions,
    available_atlas_channels,
)
from langslice.core.opening import DEFAULT_IMAGE_LIMIT
from langslice.core.placement import (
    FRAMED_PLACEMENT_MODES,
    PLACEMENT_MODES,
    WARPED_PLACEMENT_MODES,
)
from langslice.core.sizes import AUTO_RESOLUTION, MAX_IMAGES_PER_CALL, resolution_level
from langslice.core.spec import JobSpec
from langslice.core.state import (
    IDENTITY_KNOBS,
    SliceState,
    StackState,
)
from langslice.core.status import compact_rows
from langslice.doors.declarations import Variant, declare
from langslice.doors.tools import TOOL_MEDIA_DELIVERY_ID_KEY, TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.arguments import (
    Candidate,
    DamageEntry,
    OrientEntry,
    PlacementEntry,
    PositionEntry,
    TransformEntry,
    View,
    argument_refusal,
)
from langslice.doors.tools.view_options import Profile, parse_view, view_edge_limit
from langslice.job.job import HOST_TRANSFORM_KIND, Job
from langslice.job.views import PICTURE_FILE as VIEW_PICTURE_FILE
from langslice.job.views import captured as captured_views
from langslice.ops import appearance as ops_appearance
from langslice.ops import atlas as ops_atlas
from langslice.ops import damage as ops_damage
from langslice.ops import deformable as ops_deformable
from langslice.ops import exports as ops_exports
from langslice.ops import history as ops_history
from langslice.ops import notes as ops_notes
from langslice.ops import order as ops_order
from langslice.ops import orientation as ops_orientation
from langslice.ops import positions as ops_positions
from langslice.ops import submit as ops_submit
from langslice.ops import traces as ops_traces
from langslice.ops import transforms as ops_transforms
from langslice.ops import views as ops_views
from langslice.ops.refusal import Refused
from langslice.ops.registry import VERBS, enabled
from langslice.providers.registry import ImageModel, resolve_image_model

if TYPE_CHECKING:  # import cycle: the engine builds the toolbox
    from langslice.agent.engine import EngineContext

logger = logging.getLogger(__name__)

#: One item of a tool's media list (under ``TOOL_MEDIA_PARTS_KEY``): a
#: picture (captioned PIL image) or a line of text, in reading order. The
#: doors package them: the ADK tools through
#: :func:`langslice.doors.tools.media.packaged`, MCP as content blocks.
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
#: mean something for it (:class:`langslice.doors.tools.view_options.Profile`).
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


def make_view_atlas(job: Job, ctx: EngineContext, max_view_edge: int, level: str | None = None):
    """Build the ``view_atlas`` tool body, closed over the run's atlas and plane;
    *max_view_edge* caps ``view.resolution`` (the driver model's largest
    image), *level* is the picture-size level (None: the run's)."""
    state = job.state
    pos_lo, pos_hi = ctx.position_range
    profile = Profile(
        VIEW_ATLAS_PROFILE_MODES, channels=False,
        channels_reason="view_atlas draws the atlas alone, no section",
    )

    def view_atlas(
        positions_mm: list[float],
        view: View,
    ) -> dict[str, Any]:
        requested = _as_floats(list(positions_mm or []))
        if not requested:
            return {"status": "error", "error": "BAD_ARGS"}
        options = parse_view(ctx, state, view, profile, max_edge=max_view_edge, level=level)
        if isinstance(options, dict):
            return options
        dropped = [round(value, 2) for value in requested[MAX_VIEW_POSITIONS:]]
        positions = _clamp_and_dedupe(
            requested[:MAX_VIEW_POSITIONS], pos_lo=pos_lo, pos_hi=pos_hi
        )
        if not positions:
            return {"status": "error", "error": "EMPTY_RESULT"}

        shown = ops_views.view_atlas(job, ctx, positions, options)
        parts: list[Media] = list(shown.pictures)
        # The plane drawn: the stack's angles as stored, or, when the
        # sections differ, the median (StackState.view_angles).
        drawn_angles = (
            dict(zip(("pitch", "yaw"), state.view_angles, strict=True))
            if state.mixed_angles else dict(state.cutting_angles_deg))
        plural = "s" if len(positions) != 1 else ""
        result: dict[str, Any] = {
            "status": "ok",
            "positions_mm": [round(position, 2) for position in positions],
            "cutting_angles_deg": drawn_angles,
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
        if state.mixed_angles:
            result["description"] += (
                " The sections' cutting angles differ; these atlas sections are drawn "
                f"at their median, pitch {drawn_angles['pitch']:.2f} yaw "
                f"{drawn_angles['yaw']:.2f} degrees."
            )
        if shown.regions_not_in_plane:
            result["regions_not_in_plane"] = shown.regions_not_in_plane
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
                "set_cutting_angles"} or (
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
    on_event: LiveCallback | None = None,
    guard: Callable[[], contextlib.AbstractContextManager[Any]] | None = None,
) -> Any:
    """Serialize execution and its host notifications under the same lock.

    Model tool-call announcements may arrive together. These optional events
    identify the tool actually executing, including stable filenames resolved
    before a reorder. They never enter model context or change its schema.
    ``tool_end`` carries ``views``: the paths of the pictures the call saved
    in the job folder (``job.views``), never their bytes.
    *guard* is entered inside the lock around the tool (the job's write
    lock and its reload of a state file changed on disk).
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
        with lock, (guard() if guard is not None else contextlib.nullcontext()):
            if on_event is None:
                return tool(*args, **kwargs)
            fields: dict[str, Any] = {"name": tool.__name__, "execution_id": uuid.uuid4().hex}
            try:
                bound = signature.bind(*args, **kwargs)
                bound.apply_defaults()
                arguments = dict(bound.arguments)
                context = arguments.pop("tool_context", None)
                fields.update(
                    id=plain(getattr(context, "function_call_id", None)),
                    args=plain(arguments),
                    target_ids=_tool_target_ids(state, tool.__name__, arguments)
                    if state is not None else [],
                )
            except Exception:
                # Bad display metadata must never change the tool's behavior.
                logger.warning("Tool execution metadata unavailable", exc_info=True)
                fields.update(id=None, args={}, target_ids=[])
            notify({"kind": "tool_start", **fields})
            try:
                with captured_views() as saved:
                    result = tool(*args, **kwargs)
            except Exception as exc:
                notify({"kind": "tool_end", **fields, "response": {
                    "status": "error", "error": type(exc).__name__, "message": str(exc),
                }})
                raise
            # The pictures as saved in the job folder (written in the
            # background): hosts show them from there, never from bytes.
            notify({"kind": "tool_end", **fields, "response": plain(result),
                    "views": [str(item.folder / VIEW_PICTURE_FILE) for item in saved]})
            return result

    return run


def _clears_stale_deformations(tool: Any, job: Job) -> Any:
    """After any tool: drop deformations whose linear placement changed, and say so.

    The rule is the job's (:meth:`~langslice.job.job.Job.clear_stale_deformations`):
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

    The hook is the job's (:meth:`langslice.job.views.ViewStore.shown`),
    the same for every door: this wrapper only says which pictures the tool
    returned and with which arguments. The store encodes and writes them in
    the background, in the doors' encoding, so the doors' bytes are
    untouched; a successful `submit` (:func:`langslice.ops.submit.submit`)
    waits for every picture to be written.
    """
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        with job.views.shown(tool.__name__, atlas_of=lambda: ctx.atlas) as shown:
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
                shown.show(pictures, arguments=plain(arguments),
                           call_id=plain(getattr(context, "function_call_id", None)))
        return result

    return run


def _opening_gate(tool: Any, box: ToolBox) -> Any:
    """Refuse a write while the door's opening-read gate is armed and some
    opening page is unread (:meth:`ToolBox.opening_refusal`)."""

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        refusal = box.opening_refusal()
        if refusal is not None:
            return refusal
        return tool(*args, **kwargs)

    return run


def _strict(tool: Any) -> Any:
    """Refuse unknown or misplaced arguments before *tool* runs.

    :func:`langslice.doors.tools.arguments.argument_refusal` against the tool's
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
    """The tools of one run over its :class:`~langslice.job.job.Job`, plus
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
    #: The opening-read gate of a door that delivers the opening pictures
    #: on request (MCP's ``show_stack`` pages): the pages not read yet,
    #: ``None`` while the door has not armed it (:meth:`require_opening`).
    #: Every write is refused (``OPENING_NOT_READ``) until the set is empty.
    opening_unread: set[int] | None = None
    #: Calls of a door whose host may send several at once (MCP) that are
    #: running now (:meth:`in_flight`).
    _flying: int = field(default=0, repr=False)
    _flight: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def names(self) -> list[str]:
        return [tool.__name__ for tool in self.tools]

    def require_opening(self, pages: int) -> None:
        """Arm the opening-read gate: every write is refused until each of
        the *pages* opening pages was read (:meth:`opening_read`). Armed
        once per toolbox; a host that already read them is not asked again."""
        if self.opening_unread is None:
            self.opening_unread = set(range(1, int(pages) + 1))

    def opening_read(self, page: int) -> None:
        """The host fetched opening page *page*."""
        if self.opening_unread is not None:
            self.opening_unread.discard(int(page))

    def opening_refusal(self) -> dict[str, Any] | None:
        """The opening-read gate's refusal, or None when it is open."""
        if not self.opening_unread:
            return None
        pages = sorted(self.opening_unread)
        return {"status": "refused", "error": "OPENING_NOT_READ", "pages": pages,
                "detail": "Read every opening page before the first write: "
                + ", ".join(f"show_stack(page={page})" for page in pages)
                + " (the pictures of the stack the job statement asks for)."}

    @contextlib.contextmanager
    def in_flight(self) -> Iterator[None]:
        """Around one call of a door whose host may send several at once:
        while it runs, :meth:`begin_model_call` promotes nothing."""
        with self._flight:
            self._flying += 1
        try:
            yield
        finally:
            with self._flight:
                self._flying -= 1

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
        """Promote pictures from direct calls (a convenience for hosts/tests):
        a new call from the host means the pictures of the calls before it
        reached it, unless another call is still in flight (:meth:`in_flight`:
        the host sent them together, so it has seen neither's pictures)."""
        with self._flight:
            if self._flying:
                return
            direct = self.pending_placement_views.pop("__direct__", None)
        if direct is not None:
            self.seen_placement_views.update(direct)


# --- the toolbox ---------------------------------------------------------


def build_tools(
    state: StackState, ctx: EngineContext, spec: JobSpec, *,
    job: Job | None = None,
    on_event: LiveCallback | None = None,
    max_view_edge: int | None = None,
    image_model: ImageModel | None = None,
    gates: bool = True,
    level: str | None = None,
    scripting: bool = False,
    image_model_connected: bool = True,
    door: str = "agent",
) -> ToolBox:
    """Build the tools this run's spec switches on, closed over *state*.

    Which tools, in which order, is the registry's
    (:func:`langslice.ops.registry.enabled`); each tool's name, arguments and
    description are its declaration (:func:`langslice.doors.declarations.declare`);
    what is defined here is each tool's body. The tools sit on *job*;
    without one, a job is made around *state* in the context's job folder (an
    empty undo history, nothing written until the first write).
    *max_view_edge* is the largest picture the driver model takes (None: the
    run model's lane, :func:`langslice.doors.tools.view_options.view_edge_limit`).
    *image_model* is the image model ``trace_borders`` calls when the run has
    one (None: this door resolves the spec's ``nonlinear.provider`` and
    ``image_model`` through
    :func:`langslice.providers.registry.resolve_image_model`, the binding
    every operation below receives). *gates* False drops the
    look-before-commit gates of ``position.gated`` (the CLI door: gates are
    tool-only). *level* is the picture-size level the tools work at (None:
    the run's ``image_resolution``; the CLI passes "auto": its caller sizes
    every picture). *scripting* adds the verbs only scripts get
    (``Verb.scripting``: ``export_maps``, and the hidden ``trace_from_atlas``,
    called by name, listed nowhere), for the CLI and the library.
    *image_model_connected* False: the door cannot reach the spec's image
    model (MCP with none connected, ``doors.api.setup.image_model_connected``),
    so the tools are those of a run without one: no ``trace_borders``,
    ``fit_deformable`` declared and checked for the stain alone, ``submit``
    not waiting for traces. The spec itself is left as it is. *door* is who
    reads the declarations (``agent``, ``mcp``, ``cli``:
    :data:`langslice.doors.declarations.DOORS`).
    """
    if job is None:
        job = Job(state, spec, layout=ctx.layout, results_path=ctx.results_path)
    if job.state is not state or job.spec is not spec:
        raise ValueError("build_tools: the job must hold this state and spec")
    if job.workspace is None:
        job.workspace = ctx
    box = ToolBox(job, max_view_edge=int(max_view_edge or view_edge_limit(ctx)))
    #: The picture-size level, and whether this door keeps the look gates.
    level = level or resolution_level(ctx)
    gated = gates and spec.position.gated
    pos_lo, pos_hi = ctx.position_range
    over_cap = job.over_cap
    #: The image model is part of this run: trace_borders and traced images exist.
    traces_on = spec.nonlinear.uses_image_model and image_model_connected
    if traces_on and image_model is None:
        image_model = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)

    # --- shared plumbing ------------------------------------------------

    def rows() -> dict[str, Any]:
        table = ops_views.status(job)
        return {
            "rows": compact_rows(table.rows),
            "cutting_angles_deg": table.cutting_angles_deg,
            "interval_breaks": table.interval_breaks,
        }

    def changed(touched: list[str]) -> dict[str, Any]:
        """The status rows of the sections a write touched, and nothing else.

        A write used to answer with the whole table; on a forty-section stack
        that is thirty-nine rows of noise per call. `status` is still the
        whole table, and is one call away.
        """
        wanted = set(touched)
        table = ops_views.status(job)
        return {
            "changed": compact_rows([row for row in table.rows if row["id"] in wanted]),
            "n_sections": len(state.slices),
            "cutting_angles_deg": table.cutting_angles_deg,
            "interval_breaks": table.interval_breaks,
        }

    def answered(*touched: str) -> dict[str, Any]:
        """A write's answer: ``ok`` and the rows it changed (the operation
        itself took the undo step and the checkpoint)."""
        return {"status": "ok", **changed(list(touched))}

    def forget_looks(moved: list[str]) -> None:
        """A position moved by undo, redo or a reload is a write to it: that
        section needs a new view, and the stack a new review (the gates)."""
        for name in moved:
            box.compared.pop(name, None)
        if moved:
            box.reviewed = False

    def sync_job() -> None:
        """Before every call: pick up a state file changed on disk."""
        before = job.sync()
        if before is not None:
            forget_looks(ops_history.moved_positions(before, state))

    @contextlib.contextmanager
    def guarded(name: str) -> Iterator[None]:
        """Around every call: a verb runs whole under the job folder's write
        lock, after the state is brought up to date (:meth:`Job.writing`); a
        long verb (``VERBS[name].long``) only syncs, computes outside the
        lock and takes it itself to apply, re-checking its sections."""
        if VERBS[name].long:
            sync_job()
            yield
            return
        with job.writing() as before:
            if before is not None:
                forget_looks(ops_history.moved_positions(before, state))
            yield

    def placement_view_key(
        record: SliceState, position_mm: float
    ) -> tuple[str, float, bool, int, float, float]:
        """Identity of one placement picture from the model's perspective."""
        return (
            record.id,
            float(position_mm),
            bool(record.flip),
            int(record.rotation_deg),
            float(record.pitch_deg),
            float(record.yaw_deg),
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
        return {"status": "ok", **rows()}

    def display(
        profile: Profile, view: Any, *,
        sections: list[SliceState] = (),  # type: ignore[assignment]
    ) -> DisplayOptions | dict[str, Any]:
        """One call's ``view``, validated once for every picture tool."""
        return parse_view(ctx, state, view, profile, max_edge=box.max_view_edge,
                          sections=sections, level=level)

    def region_names(values: Any, field_name: str) -> tuple[str, ...] | dict[str, Any]:
        """Region entries for `include`/`exclude`, checked against the atlas.

        An entry is an acronym or id, optionally with a side ("CTX:left",
        :mod:`langslice.core.atlas.sides`).
        """
        from langslice.core.atlas.sides import has_sides
        from langslice.core.deformable.atlas_images import resolve_entries

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
        from langslice.core.atlas.sides import overlapping

        overlap = overlapping(kept, dropped)
        if not overlap:
            return None
        return {"status": "error", "error": "BAD_ARGS",
                "message": "A region cannot be both included and excluded: " + ", ".join(overlap)}

    def view_slices(
        slices: list[str],
        view: View,
    ) -> dict[str, Any]:
        if not slices:
            return {"status": "error", "error": "BAD_ARGS",
                    "message": "slices must name one or more sections"}
        known, unknown = resolve_many(list(slices)[:MAX_VIEW_SLICES])
        if not known:
            return {"status": "error", "error": "UNKNOWN_SLICE_IDS", "unknown": unknown}
        options = display(VIEW_SLICES_VIEW, view, sections=known)
        if isinstance(options, dict):
            return options
        shown = ops_views.view_slices(job, ctx, known, options)
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
            TOOL_MEDIA_PARTS_KEY: list(shown.pictures),
        }
        if options.mode == "channels":
            result["channels"] = shown.channels
        return result

    def note(text: str) -> dict[str, Any]:
        try:
            return {"status": "ok", "notes": ops_notes.add_note(job, text)}
        except Refused as refusal:
            return refusal.payload()

    def undo() -> dict[str, Any]:
        step = ops_history.undo(job)
        if not step.done:
            return {"status": "error", "error": "NOTHING_TO_UNDO"}
        forget_looks(step.moved)
        return {"status": "ok", "undo_depth": step.depth, **rows()}

    def redo() -> dict[str, Any]:
        step = ops_history.redo(job)
        if not step.done:
            return {"status": "error", "error": "NOTHING_TO_REDO"}
        forget_looks(step.moved)
        return {"status": "ok", "redo_depth": step.depth, **rows()}

    def mark_damaged(entries: list[DamageEntry]) -> dict[str, Any]:
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
        def look_gate() -> dict[str, Any] | None:
            if spec.has("position") and gated and not box.reviewed:
                return {
                    "status": "refused",
                    "error": "NOT_REVIEWED",
                    "detail": "view_stack has not run since the last set_positions write",
                }
            return None

        try:
            done = ops_submit.submit(
                job, summary=summary, notes=notes, interval_breaks=interval_breaks,
                traces=traces_on, workspace=ctx, gate=look_gate,
            )
        except Refused as refusal:
            return refusal.payload()
        box.submission.update(
            {
                "summary": done.summary,
                "notes": done.notes,
                "interval_breaks": done.interval_breaks,
            }
        )
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
        target: str,
        slices: list[str],
        channel_weights: list[float],
        clahe_clip: float,
        clahe_tiles: int,
        n4: bool,
        denoise: bool,
        reset: bool,
        view: View,
    ) -> dict[str, Any]:
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

        ids = [record.id for record in scope] if slices else None
        try:
            # BEFORE is drawn first, from the settings as they stand; AFTER
            # from the settings the write will leave. Written only once every
            # picture is drawn.
            done = ops_appearance.preprocess(job, ctx, targets, ids, settings, shown=shown,
                                             options=options)
        except Refused as refusal:
            return refusal.payload()
        pictured = done.target  # "both" writes one setting to both targets
        parts: list[Media] = []
        for pair in done.pictures:
            record = pair.record
            label = f"{record.index_corrected}: {record.id}  {pictured} appearance"
            parts.append(layers.note(
                caption(pair.before, f"{label}  BEFORE ({looks.describe(pair.before_settings)})"),
                sections=(record.id,), mode="before", extra={"target": pictured}))
            parts.append(layers.note(
                caption(pair.after, f"{label}  AFTER ({looks.describe(pair.after_settings)})"),
                sections=(record.id,), mode="after", extra={"target": pictured}))
        written = done.written
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


    # --- orientation (part of the transform task) ------------------------

    def orient_slices(
        entries: list[OrientEntry],
        view: View,
    ) -> dict[str, Any]:
        if not entries:
            return {"status": "error", "error": "BAD_ARGS"}
        named, _ = resolve_many([entry.get("id", "") for entry in entries
                                 if isinstance(entry, dict)])
        options = display(ORIENT_VIEW, view, sections=named)
        if isinstance(options, dict):
            return options
        done = ops_orientation.orient_sections(job, entries, workspace=ctx, options=options)
        applied = done.applied
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
            "render_failed": done.render_failed,
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: list(done.pictures),
        }

    # --- reorder --------------------------------------------------------

    def reorder_slices(slices: list[str], after: str) -> dict[str, Any]:
        try:
            done = ops_order.reorder(job, slices, after)
        except Refused as refusal:
            return refusal.payload()
        return {"moved": done.moved, **answered(*done.touched)}


    # --- position -------------------------------------------------------

    def set_positions(
        entries: list[PositionEntry],
        view: View,
        tool_context: Any = None,
    ) -> dict[str, Any]:
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
            if gated and not box.compared.get(record.id):
                # One compare is enough: a section is confirmed at one
                # hypothesised position.
                rejected.append(
                    {
                        "id": record.id,
                        "reason": "not compared since its last write; "
                        "view_placement first",
                    }
                )
                continue
            wanted.append((record.id, requested))

        atlas_bearing = (options.mode != "section" and options.full_view
                         and bool(options.atlas_channels))
        # Do not pay for the same placement picture twice. A successful
        # compare is promoted to ``seen`` only at the next model-call
        # boundary, so compare + write tool calls emitted in one round still
        # return the write picture: the model has not received the compare
        # result yet. Orientation and cutting angles are part of the identity.
        done = ops_positions.set_positions(
            job, ctx, wanted, options=options,
            show=lambda record, position: placement_view_key(record, position)
            not in box.seen_placement_views,
        )
        written = [{"id": name, "position_mm": round(value, 3)} for name, value in done.written]
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
        pictured = done.view or ops_views.PlacementView()
        delivery_id: str | None = None
        image_indexes: dict[str, Any] = {}
        rendered: list[str] = []
        for pair in pictured.shown:
            image_indexes[pair.record.id] = pair.indexes
            rendered.append(pair.record.id)
            if atlas_bearing:
                delivery_id = box.record_placement_view(
                    tool_context, placement_view_key(pair.record, pair.position)
                )
        failed = [{"id": record.id, "message": message} for record, _p, message in pictured.failed]
        result["render_failed"] = failed
        suppressed = done.not_shown
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
        result[TOOL_MEDIA_PARTS_KEY] = list(pictured.pictures)
        return result

    def search_position(id: str, window_mm: float, angles: bool) -> dict[str, Any]:
        try:
            found = ops_positions.search_position(job, ctx, id, window_mm, angles=bool(angles))
        except Refused as refusal:
            return refusal.payload()
        return {"status": "ok", **found}

    def view_placement(
        entries: list[PlacementEntry],
        view: View,
        tool_context: Any = None,
    ) -> dict[str, Any]:
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

        full_atlas_view = (options.mode != "section" and options.full_view
                           and bool(options.atlas_channels))
        shown = ops_views.view_placement(job, ctx, pairs, options)
        parts: list[Media] = list(shown.pictures)
        failed = [{"id": record.id, "position_mm": round(position, 3), "message": message}
                  for record, position, message in shown.failed]
        delivery_id: str | None = None
        compared: list[dict[str, Any]] = []
        for pair in shown.shown:
            record, position = pair.record, pair.position
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
                **pair.row,
                **({"regions_not_in_plane": pair.regions_not_in_plane}
                   if pair.regions_not_in_plane else {}),
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
        view: View,
    ) -> dict[str, Any]:
        options = display(VIEW_STACK_VIEW, view, sections=state.in_order())
        if isinstance(options, dict):
            return options
        box.reviewed = True

        review = ops_views.view_stack(job, ctx, options)
        sheet, plot = review.sheet, review.plot
        parts = [
            f"The {len(state.slices)} sections in the order of their "
            "written positions (unplaced last), each captioned "
            "'<index>: <filename>  <position>', over its atlas match:",
            sheet,
            "Position against corrected index:",
            plot,
        ]
        return {
            "status": "ok",
            "rows": compact_rows(review.rows),
            "description": (
                "Attached: one contact sheet of every section in the order of "
                "its written position, each captioned, with the atlas section "
                "at its position beneath it when it has one; then a plot of "
                "position against corrected index."
            ),
            "view": options.echo(),
            TOOL_MEDIA_PARTS_KEY: parts,
        }


    # --- transform ------------------------------------------------------

    def set_cutting_angles(pitch_deg: float, yaw_deg: float) -> dict[str, Any]:
        try:
            pitch, yaw = float(pitch_deg), float(yaw_deg)
        except (TypeError, ValueError):
            return {"status": "error", "error": "BAD_ARGS"}
        ops_positions.set_cutting_angles(job, ctx, pitch, yaw)
        return answered()  # stack-wide: no section row changes

    def fit_affine(
        slices: list[str],
        method: str,
        fit_atlas: str,
        include: list[str],
        exclude: list[str],
        view: View,
    ) -> dict[str, Any]:
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
                    "message": f"The {atlas_kind} atlas image is offered on Allen mouse atlases "
                    f"only; {state.atlas} does not cover the CCFv3 grid.", "fit_atlas": offered}
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
            targets, unknown = ops_transforms.fit_targets(job), []
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

        done = ops_transforms.fit_affine(
            job, ctx, targets, method=chosen, fit_atlas=atlas_kind, include=kept,
            exclude=dropped, options=options,
        )
        fits = [row for row in done.rows if row.get("status") == "ok"]
        parts: list[Media] = list(done.pictures)
        payload: dict[str, Any] = {
            "status": "ok" if fits else "error",
            "results": done.rows,
            "unknown_ids": unknown,
            "view": options.echo(),
        }
        if not fits:
            payload["error"] = "NOTHING_FITTED"
            return payload
        if parts:
            payload["description"] = (
                "Attached panels are "
                + ", ".join(row["id"] for row in fits)
                + ", in that order (mapped by each result's image_indexes); "
                "each shows the section under its fitted transform against "
                "the atlas at true physical scale, in the requested view."
                + (" The whole atlas is drawn; the fit used only the kept regions."
                   if restricted else "")
            )
            payload[TOOL_MEDIA_PARTS_KEY] = parts
        for row in fits:
            row.pop("params")  # the six raw numbers stay host-side
        return payload

    # --- the interactive transform --------------------------------------

    def adjusted(entry: ops_transforms.Adjustment, options: DisplayOptions) -> dict[str, Any]:
        """One written (or redrawn) entry's result, in words and numbers.

        The ``ab`` reference reported is what the B picture drew: the
        section's transform before the call, or identity.
        """
        staged, previous = entry.staged, entry.previous
        assert staged is not None and entry.transform is not None
        record = staged.record
        view = options.mode
        stored = (previous or {}).get("physical")
        reference: dict[str, Any] | None = None
        if view == "ab":
            held = previous is not None
            reference = {
                "source": "stored" if held else "identity",
                "params": dict(stored) if isinstance(stored, dict) else dict(IDENTITY_KNOBS),
            }
            if held:
                reference["stored_kind"] = (previous or {}).get("kind")
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
        absent = ops_views.regions_not_in_plane(job, ctx, position, options, record.angles)
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
            "physical": entry.transform["physical"],
            "written": entry.written,
            **({"ab_reference": reference} if reference is not None else {}),
            **({"regions_not_in_plane": absent} if absent else {}),
            "description": f"{record.id} under the transform above, {described}. {lines}",
        }

    def adjust_transforms(
        entries: list[TransformEntry],
        view: View,
    ) -> dict[str, Any]:
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

        done = ops_transforms.adjust_transforms(job, ctx, entries, options=options)
        results: list[dict[str, Any]] = []
        parts: list[Media] = []
        for entry in done.entries:
            if entry.error is not None:
                results.append(dict(entry.error) if entry.early
                               else {**entry.error, "image_indexes": []})
                continue
            result = adjusted(entry, options)
            result["image_indexes"] = list(range(len(parts), len(parts) + len(entry.pictures)))
            parts.extend(entry.pictures)
            results.append(result)
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


    def trace_borders(id: str, prompt: str, include: list[str],
                      exclude: list[str]) -> dict[str, Any]:
        assert image_model is not None  # the tool exists only when traces are on
        kept = region_names(include, "include")
        if isinstance(kept, dict):
            return kept
        dropped = region_names(exclude, "exclude")
        if isinstance(dropped, dict):
            return dropped
        refusal = region_overlap(kept, dropped)
        if refusal is not None:
            return refusal
        try:
            done = ops_traces.trace_borders(job, ctx, id, prompt=prompt, image_model=image_model,
                                            include=kept, exclude=dropped)
        except Refused as refusal:
            return refusal.payload()
        if done.running:
            return {"status": "running", "id": done.id,
                    "message": "This section's image correction is already running."}
        result = done.record
        response = {key: result[key] for key in (
            "status", "error", "message", "cached", "prompt_edited", "attempt",
            "include", "exclude",
        ) if key in result}
        response["id"] = done.id
        if done.started:
            response["message"] = (
                "Image call started in the background. Continue; submit waits for it."
            )
        return response

    def trace_from_atlas(slices: list[str], passes: int) -> dict[str, Any]:
        assert image_model is not None  # the verb exists only when traces are on
        try:
            done = ops_traces.trace_from_atlas(job, ctx, list(slices or []),
                                               image_model=image_model, passes=passes)
        except Refused as refusal:
            return refusal.payload()
        failed = [row for row in done.rows if row.get("status") == "error"]
        return {
            "status": "error" if len(failed) == len(done.rows) else "ok",
            **({"error": "NOTHING_TRACED"} if len(failed) == len(done.rows) else {}),
            "results": done.rows,
            "message": "Image calls run in the background; submit waits for them.",
        }

    def grep_atlas(query: str, section: str) -> dict[str, Any]:
        try:
            return {"status": "ok", **ops_atlas.grep_atlas(job, ctx, query, section)}
        except Refused as refusal:
            return refusal.payload()

    # --- the deformable fit (nonlinear) -----------------------------------

    engine_option = spec.nonlinear.engine

    def resolve_choice(values: dict[str, Any]) -> deformation.Choice | dict[str, Any]:
        """One candidate's settings, validated, or the refusal naming the fix."""
        from typing import get_args

        from langslice.core.deformable.settings import Stiffness

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
                    "message": f"The {atlas_kind} atlas image is offered on Allen mouse atlases "
                    f"only; {state.atlas} does not cover the CCFv3 grid.",
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

    def fit_deformable(
        slices: list[str],
        include: list[str],
        exclude: list[str],
        start: str,
        fit_section: str,
        fit_atlas: str,
        stiffness: str,
        candidates: list[Candidate],
        keep_linear: str,
        view: View,
        engine: str = "",  # not declared where the user fixed the engine
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
                                             exclude=dropped, start=begin, options=options)
        applying = done.applied
        rows = done.rows
        parts: list[Media] = list(done.pictures)
        failed = done.render_failed
        traces = done.traces
        highlight = [name for name, _ids in options.regions] or list(kept)
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

    # --- for scripts only (Verb.scripting) ------------------------------

    def export_maps(slices: list[str], full_resolution: bool) -> dict[str, Any]:
        try:
            done = ops_exports.export_maps(job, ctx, list(slices or []),
                                           full_resolution=bool(full_resolution))
        except Refused as refusal:
            return refusal.payload()
        return {
            "status": "ok" if done.sections or not done.skipped else "error",
            **({} if done.sections or not done.skipped else {"error": "NOTHING_EXPORTED"}),
            "written": done.sections, "skipped": done.skipped,
            "full_resolution": done.full_resolution, "files_written": done.written,
            "files": [{"path": path, "kind": kind} for path, kind in done.files],
            "seconds": round(done.seconds, 2),
        }

    bodies: dict[str, Callable[..., Any]] = {
        "status": status, "view_slices": view_slices,
        "view_atlas": make_view_atlas(job, ctx, box.max_view_edge, level), "note": note,
        "undo": undo, "redo": redo, "mark_damaged": mark_damaged, "preprocess": preprocess,
        "reorder_slices": reorder_slices, "set_positions": set_positions,
        "view_placement": view_placement, "view_stack": view_stack,
        "search_position": search_position,
        "orient_slices": orient_slices, "fit_affine": fit_affine,
        "adjust_transforms": adjust_transforms, "set_cutting_angles": set_cutting_angles,
        "trace_borders": trace_borders, "trace_from_atlas": trace_from_atlas,
        "grep_atlas": grep_atlas,
        "fit_deformable": fit_deformable, "submit": submit, "export_maps": export_maps,
    }
    # `view.resolution` exists only where the caller chooses the picture size.
    variant = Variant.of(spec, auto=level == AUTO_RESOLUTION, image_model=traces_on, door=door)
    lock = threading.Lock()
    def behind_gate(name: str, tool: Any) -> Any:
        """A write behind the door's opening-read gate (armed by MCP only)."""
        return _opening_gate(tool, box) if VERBS[name].kind == "write" else tool

    box.tools = [
        _serialized(
            _saves_views(_clears_stale_deformations(
                _strict(behind_gate(name, declare(name, bodies[name], variant))), job), job, ctx),
            lock, state=state, on_event=on_event, guard=functools.partial(guarded, name))
        for name in enabled(spec, scripting=scripting, image_model=image_model_connected,
                            hidden=scripting)
    ]
    return box
