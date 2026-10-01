"""The run: ingest, one agent session, results.

``run(spec)`` is the whole of ``langslice linear``. There is no node graph: one
state, one toolbox, one job statement, and a session that ends at ``submit`` or
the turn budget. Every write tool checkpoints, so a run that dies mid-way
resumes from the checkpoint with the state it had — the agent is re-seeded, not
replayed.
"""

from __future__ import annotations

import contextlib
import copy
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from google.genai import types
from PIL import Image

from langslice.atlas.core import get_position_range_mm, load_atlas
from langslice.image_prep import (
    channel_planes,
    host_preprocess,
    normalize_image,
    read_pixel_size_um,
    read_working_image,
    read_working_pages,
)
from langslice.linear.atlas_fetch import atlas_strip_parts
from langslice.linear.checkpoint import (
    default_checkpoint_path,
    load_checkpoint,
    observe_checkpoints,
    save_checkpoint,
)
from langslice.linear.discovery import discover_slices
from langslice.linear.display import display_facts
from langslice.linear.live import LiveCallback
from langslice.linear.prompt import build_job_statement
from langslice.linear.render import stack_image_parts, status_text
from langslice.linear.session import (
    DEFAULT_MAX_ITERATIONS,
    build_agent,
    run_agent_session,
)
from langslice.linear.spec import JobSpec
from langslice.linear.state import SliceState, StackState
from langslice.linear.toolbox import ToolBox, build_tools
from langslice.space import Plane, atlas_space_context, slice_axis_ends

logger = logging.getLogger(__name__)

RESULTS_FILENAME = "linear_results.json"

_RUN_LABEL = "linear_stack"

_NUDGE_NO_TOOL = (
    "You did not call a tool. Continue with the tools rather than in prose; "
    "call `submit` when the job is done."
)
_NUDGE_CONTINUE = "Continue; call `submit` when the job is done."

#: Asked once, after submit, in the same context. The answer is recorded on the
#: state (``debrief``) for the people building this environment; nothing about
#: the run changes. Answer in text: tools are still live, and a call here would
#: be a write after submit.
DEBRIEF_PROMPT = (
    "The job is submitted and nothing you say now changes it. This is a "
    "debrief for the people building this tool environment; answer in text "
    "and do not call any tool.\n"
    "1. Which tools or pieces of information did you reach for, or wish "
    "existed, that were not available — including while aligning sections: "
    "views, overlays, controls, measurements?\n"
    "2. Which tools behaved differently from what you expected, or were "
    "awkward to use as specified?\n"
    "3. What did you have to work around?\n"
    "4. Which tool environments you know does this resemble, and what did "
    "those have that this lacks?\n"
    "5. Your wishlist: what would you want in this environment to do this "
    "job well?"
)


def _log_progress(message: str) -> None:
    logger.info(message)


@dataclass
class EngineContext:
    """Everything a tool needs that is not stack state."""

    spec: JobSpec
    image_folder: str
    checkpoint_path: str
    results_path: str
    model: str
    emit: Callable[[str], None] = _log_progress
    #: Atlas accessor, injectable so tests (and offline hosts) can supply one.
    atlas_loader: Callable[[str], Any] = load_atlas
    #: Rendered sections, keyed ``(id, flip, rotation, long_edge, preprocess,
    #: frame)``. The user's files never change during a run, and a correction
    #: or a different size is a different key. Cached images are shared —
    #: callers read them, never mutate them.
    render_cache: dict[tuple[str, bool, int, int, str, bool], Image.Image] = field(
        default_factory=dict, repr=False
    )
    #: Same keys as ``render_cache``: how much that render shrank the file's
    #: pixels, so a known micrometres-per-pixel can follow the image down.
    render_scale: dict[tuple[str, bool, int, int, str, bool], float] = field(
        default_factory=dict, repr=False
    )
    #: Each file's working copy (:func:`langslice.image_prep.read_working_image`)
    #: and how many file pixels one of its pixels spans. Every render above
    #: is drawn from it, so a whole-slide scan is read once, small.
    source_cache: dict[str, tuple[Image.Image, float]] = field(
        default_factory=dict, repr=False
    )
    #: Each file's raw channels at working size: names and 8-bit planes
    #: (:func:`langslice.image_prep.channel_planes`). Shared: read only.
    channel_cache: dict[str, tuple[tuple[str, ...], list[Any]]] = field(
        default_factory=dict, repr=False
    )
    #: Encoded, captioned reference images shared by the seed and comparison tools.
    reference_parts: dict[tuple[Any, ...], types.Part] = field(default_factory=dict, repr=False)
    _atlas: Any = field(default=None, repr=False)
    #: ABBA's cached Allen atlas when it matches this run's atlas (the
    #: ``nissl`` atlas image); looked up once.
    _abba: Any = field(default=None, repr=False)
    _abba_checked: bool = field(default=False, repr=False)
    _range: tuple[float, float] | None = field(default=None, repr=False)
    _pixel_sizes: dict[str, float | None] = field(default_factory=dict, repr=False)

    def progress(self, message: str) -> None:
        self.emit(message)
        logger.info(message)

    def image_path(self, slice_id: str) -> str:
        """Absolute path of a section image. Ids are folder-relative names."""
        return os.path.join(self.image_folder, slice_id)

    def working_source(self, slice_id: str) -> tuple[Image.Image, float]:
        """``(working copy, file pixels per working pixel)`` of one section.

        Shared: read it, never mutate it in place.
        """
        cached = self.source_cache.get(slice_id)
        if cached is None:
            settings = self.spec.host_preprocessing
            if settings is not None:
                # A host's snapshot, one page per channel: the default
                # appearance is the host's blend of the pages (what
                # ``preprocess.preview`` shows), drawn at working size. The
                # pages stay readable as channels.
                pages, factor = read_working_pages(self.image_path(slice_id))
                blended = host_preprocess(pages, settings).convert("L")
                cached = (normalize_image(blended), factor)
            else:
                cached = read_working_image(self.image_path(slice_id))
            self.source_cache[slice_id] = cached
        return cached

    def section_channels(self, slice_id: str) -> tuple[tuple[str, ...], list[Any]]:
        """``(names, raw 8-bit planes)`` of one section at working size.

        A single colour file is red/green/blue (``gray`` when the three are
        equal); a multi-page file is one plane per page, named by the host's
        ``inputs["channel_names"]`` when it gave one per page, else ch1, ch2...
        """
        cached = self.channel_cache.get(slice_id)
        if cached is None:
            pages, _factor = read_working_pages(self.image_path(slice_id))
            names = (self.spec.inputs or {}).get("channel_names")
            cached = channel_planes(pages, list(names) if isinstance(names, list) else None)
            self.channel_cache[slice_id] = cached
        return cached

    @property
    def abba_atlas(self) -> Any:
        """ABBA's cached Allen atlas when present and matching, else None."""
        if not self._abba_checked:
            self._abba_checked = True
            try:
                from langslice.deformable.abba_atlas import AbbaAtlas

                found = AbbaAtlas.find()
                self._abba = found if found is not None and found.compatible(self.atlas) else None
            except Exception:  # an unreadable cache or a non-BrainGlobe atlas: no Nissl
                logger.debug("ABBA atlas lookup failed", exc_info=True)
                self._abba = None
        return self._abba

    @property
    def atlas(self) -> Any:
        if self._atlas is None:
            self._atlas = self.atlas_loader(self.spec.atlas)
        return self._atlas

    @property
    def position_range(self) -> tuple[float, float]:
        if self._range is None:
            self._range = get_position_range_mm(
                self.atlas, plane=cast(Plane, self.spec.plane)
            )
        return self._range

    @property
    def axis_ends(self) -> tuple[str, str]:
        """(low, high) anatomical ends of this run's slicing axis."""
        return slice_axis_ends(
            atlas_space_context(self.atlas), cast(Plane, self.spec.plane)
        )

    def calibration(self, slice_id: str) -> tuple[float | None, str]:
        """``(micrometres per pixel of the FILE, source)`` for one section.

        The host's ``inputs["pixel_size_um"]`` wins when it is given — that is
        what ``--pixel-size-um`` is for — otherwise the file's own tags answer
        (:func:`langslice.image_prep.read_pixel_size_um`). ``(None, "")`` when
        nothing says; the caller estimates and says that it estimated.
        """
        host = (self.spec.inputs or {}).get("pixel_size_um")
        if host:
            try:
                value = float(host)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0:
                return value, "host"
        if slice_id not in self._pixel_sizes:
            self._pixel_sizes[slice_id] = read_pixel_size_um(self.image_path(slice_id))
        from_file = self._pixel_sizes[slice_id]
        return (from_file, "file") if from_file else (None, "")

    @property
    def species(self) -> str:
        metadata = getattr(self.atlas, "metadata", None)
        return str((metadata or {}).get("species", "mouse"))


def build_context(
    spec: JobSpec,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
) -> EngineContext:
    """Assemble an :class:`EngineContext` from a job spec."""
    from langslice.providers.openai_oauth import DEFAULT_REVIEW_MODEL

    folder = os.path.abspath(spec.image_folder)
    return EngineContext(
        spec=spec,
        image_folder=folder,
        checkpoint_path=default_checkpoint_path(folder),
        results_path=spec.out or os.path.join(folder, RESULTS_FILENAME),
        model=spec.model or DEFAULT_REVIEW_MODEL,
        emit=emit or _log_progress,
        atlas_loader=atlas_loader or load_atlas,
    )


# --- ingest --------------------------------------------------------------


def ingest(spec: JobSpec, ctx: EngineContext) -> StackState:
    """Discover the folder and build the stack state. Plain code, no model."""
    paths = discover_slices(ctx.image_folder)
    if not paths:
        raise ValueError(f"No slice images found in {ctx.image_folder}")

    pos_lo, pos_hi = ctx.position_range
    state = StackState(
        image_folder=ctx.image_folder,
        atlas=spec.atlas,
        plane=spec.plane,
        interval_mm=spec.interval_mm,
        thickness_mm=spec.thickness_mm,
        spec=spec.to_dict(),
        slices=[
            SliceState(
                id=os.path.basename(path), index_original=index, index_corrected=index
            )
            for index, path in enumerate(paths)
        ],
    )
    state.notes.append(
        f"ingest: {len(paths)} sections, atlas {spec.atlas} ({spec.plane}) "
        f"spans {pos_lo:.2f}-{pos_hi:.2f} mm"
    )
    ctx.progress(f"[ingest] {len(paths)} sections from {ctx.image_folder}")
    return state


def apply_host_inputs(state: StackState, spec: JobSpec) -> None:
    """Write the host's answers for the tasks that are switched off.

    Order arrives as a list of filenames, positions as a filename -> mm
    mapping, angles as ``{"pitch": deg, "yaw": deg}``, damage as a filename
    -> note mapping, and transforms as filename -> stored transform dictionaries.
    Anything the host
    supplies for a task that IS on is applied too — it is a starting point,
    not a constraint. Two exceptions are constraints: ``damaged`` flags the
    agent cannot clear, and ``locked`` sections (a list of filenames) whose
    flip, rotation and transform the agent cannot change; a locked section
    without a supplied transform carries the ``"host"`` identity
    (:func:`langslice.linear.toolbox.host_transform`), because its snapshot
    is already aligned.
    """
    inputs = spec.inputs or {}

    order = inputs.get("order") or []
    if order:
        known = [state.by_id(str(name)) for name in order]
        missing = [str(name) for name, hit in zip(order, known, strict=True) if hit is None]
        if missing:
            raise ValueError(f"inputs.order names sections that are not here: {missing}")
        tail = [s for s in state.in_order() if s.id not in {str(name) for name in order}]
        for index, record in enumerate([r for r in known if r is not None] + tail):
            record.index_corrected = index
        state.notes.append(f"inputs: order set by the host ({len(order)} sections)")

    positions = inputs.get("positions") or {}
    if positions:
        applied = 0
        for name, value in positions.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.positions names an unknown section: {name!r}")
            record.position_mm = float(value)
            applied += 1
        state.notes.append(f"inputs: {applied} position(s) set by the host")

    angles = inputs.get("angles") or {}
    if angles:
        state.cutting_angles_deg = {
            "pitch": float(angles.get("pitch", 0.0)),
            "yaw": float(angles.get("yaw", 0.0)),
        }
        state.notes.append(
            f"inputs: cutting angles set by the host "
            f"(pitch {state.pitch_deg:.2f}, yaw {state.yaw_deg:.2f})"
        )

    transforms = inputs.get("transforms") or {}
    if transforms:
        for name, value in transforms.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.transforms names an unknown section: {name!r}")
            if not isinstance(value, dict):
                raise ValueError(f"inputs.transforms[{name!r}] must be a transform dictionary")
            # Preserve complete historical mappings, including splines. The image
            # correction handoff explicitly refuses unsupported spline inputs.
            record.transform = copy.deepcopy(value)
        state.notes.append(f"inputs: {len(transforms)} transform(s) set by the host")

    damaged = inputs.get("damaged") or {}
    if damaged:
        # Damage is normally the agent's own classification; a host (or a
        # benchmark) may assert it up front so the automatic fits refuse the
        # section and it is aligned by hand.
        for name, note in damaged.items():
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.damaged names an unknown section: {name!r}")
            record.damaged = True
            record.damage_note = str(note or "")
        state.notes.append(f"inputs: {len(damaged)} section(s) marked damaged by the host")

    locked = inputs.get("locked") or []
    if locked:
        from langslice.linear.toolbox import host_transform

        if not isinstance(locked, (list, tuple)):
            raise ValueError("inputs.locked must be a list of section filenames")
        for name in locked:
            record = state.by_id(str(name))
            if record is None:
                raise ValueError(f"inputs.locked names an unknown section: {name!r}")
            if record.transform is None:
                record.transform = host_transform()
        state.notes.append(
            f"inputs: {len(locked)} section(s) locked by the host (in-plane alignment done)"
        )


# --- the session ---------------------------------------------------------


def build_seed_message(state: StackState, ctx: EngineContext) -> types.Content:
    """Every section as its own labelled image, the atlas strip, the table."""
    parts: list[types.Part] = stack_image_parts(state, ctx)
    parts.extend(atlas_strip_parts(ctx, state))
    parts.append(
        types.Part.from_text(
            text=(
                "Status table (corrected index, filename, position, spacing to "
                "the next placed section, flags):\n"
                f"{status_text(state)}\n\n"
                "Recent run notes:\n"
                + ("\n".join(f"- {note}" for note in state.notes[-12:]) or "- (none)")
            )
        )
    )
    return types.Content(role="user", parts=parts)


async def run_session(
    state: StackState,
    ctx: EngineContext,
    spec: JobSpec,
    box: ToolBox,
    *,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    on_event: LiveCallback | None = None,
) -> tuple[int, int]:
    """Drive the one stack session; return ``(tool_calls, turns)``."""
    pos_lo, pos_hi = ctx.position_range
    agent = build_agent(
        model=ctx.model,
        name="linear_stack",
        instruction=build_job_statement(
            spec,
            state,
            tool_names=box.names,
            species=ctx.species,
            pos_lo=pos_lo,
            pos_hi=pos_hi,
            axis_ends=ctx.axis_ends,
            **display_facts(ctx, state),
        ),
        tools=box.tools,
        reasoning=spec.reasoning,
    )
    sink: list[str] = []
    outcome = await run_agent_session(
        agent=agent,
        seed_message=build_seed_message(state, ctx),
        done=lambda: state.submitted,
        nudge_no_tool=_NUDGE_NO_TOOL,
        nudge_continue=_NUDGE_CONTINUE,
        max_iterations=max_iterations,
        run_label=_RUN_LABEL,
        debrief=DEBRIEF_PROMPT if spec.debrief else None,
        debrief_sink=sink,
        progress=ctx.progress,
        max_input_tokens=spec.max_input_tokens,
        max_quota_percent=spec.max_quota_percent,
        tool_media_delivered=box.mark_placement_views_delivered,
        on_event=on_event,
    )
    if sink and sink[0]:
        state.debrief = sink[0]
        save_checkpoint(state, ctx.checkpoint_path)
    return outcome


# --- results -------------------------------------------------------------


def emit_results(state: StackState, ctx: EngineContext) -> StackState:
    """Write the results JSON — the same shape as the checkpoint."""
    os.makedirs(os.path.dirname(os.path.abspath(ctx.results_path)), exist_ok=True)
    with open(ctx.results_path, "w", encoding="utf-8") as handle:
        json.dump(state.to_dict(), handle, indent=2)
    ctx.progress(f"[emit] results -> {ctx.results_path}")
    return state


async def run(
    spec: JobSpec,
    *,
    emit: Callable[[str], None] | None = None,
    atlas_loader: Callable[[str], Any] | None = None,
    on_write: Callable[[StackState], None] | None = None,
    on_event: LiveCallback | None = None,
) -> StackState:
    """Run one linear job and return the final stack state.

    *on_write* is a host adapter that wants to watch the run live (the ABBA
    mirror, :mod:`langslice.integrations.abba_linear`): it is called once
    with the state as ingested, then again after every checkpoint the
    session writes (:func:`langslice.linear.checkpoint.observe_checkpoints`),
    through to the final result.
    """
    ctx = build_context(spec, emit=emit, atlas_loader=atlas_loader)

    state = load_checkpoint(ctx.checkpoint_path) if spec.resume else None
    if state is not None:
        ctx.progress(f"[ingest] resuming from {ctx.checkpoint_path}")
        state.spec = spec.to_dict()
        state.submitted = False
    else:
        state = ingest(spec, ctx)
        apply_host_inputs(state, spec)
    save_checkpoint(state, ctx.checkpoint_path)
    if on_write is not None:
        on_write(state)

    box = build_tools(state, ctx, spec, on_event=on_event)
    watch = observe_checkpoints(on_write) if on_write is not None else contextlib.nullcontext()
    with watch:
        try:
            tool_calls, turns = await run_session(state, ctx, spec, box, on_event=on_event)
        finally:
            # Background image corrections finish and are recorded even when
            # the session ends without a submit.
            if box.settle_image_corrections(state):
                save_checkpoint(state, ctx.checkpoint_path)
        ctx.progress(
            f"[session] {tool_calls} tool call(s) over {turns} turn(s); "
            + ("submitted" if state.submitted else "no submission")
        )
        if not state.submitted:
            state.notes.append(f"session: no submission within {turns} turns")

        return emit_results(state, ctx)
