"""The one toolbox: every tool the agent can be given, gated by the spec.

Conventions, applied to every tool: sections are addressed by filename or
index; ordinary writes return the rows they changed (``status`` is the whole
table); every write is undoable and checkpoints. Every picture a tool sends
is saved in the job folder with a number, and the reply lists the numbers
under ``pictures``, in the order the pictures are attached.

Tools report data: no advice, no interpretation, no strategy in any payload.
A constraint that states a number is not coaching: refusals name the
numbers that caused them and stop there.

The tools are a door over the core and the job: they check arguments, keep
the look-before-commit gates, call the operations (:mod:`langslice.ops`) and
word the reply. The bodies are in :mod:`.looking`, :mod:`.changing` and
:mod:`.fitting`, closures over one :class:`~langslice.doors.tools.door.Door`;
this module wraps each one (:func:`build_tools`). Their pictures are plain
PIL images under ``TOOL_MEDIA_PARTS_KEY``; the ADK driver packages them as
message parts (:func:`langslice.doors.tools.media.packaged`), the MCP
server as content blocks, so this module never imports ``google.genai``.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from PIL import Image

from langslice.agent.live import LiveCallback, plain
from langslice.core.opening import DEFAULT_IMAGE_LIMIT
from langslice.core.sizes import AUTO_RESOLUTION, resolution_level
from langslice.core.spec import JobSpec
from langslice.core.state import StackState
from langslice.doors.declarations import Variant, declare
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY, changing, fitting, looking
from langslice.doors.tools.arguments import argument_refusal
from langslice.doors.tools.door import Door
from langslice.doors.tools.view_options import view_edge_limit
from langslice.job.job import HOST_TRANSFORM_KIND, Job
from langslice.job.views import PICTURE_FILE as VIEW_PICTURE_FILE
from langslice.job.views import captured as captured_views
from langslice.ops import history as ops_history
from langslice.ops.registry import VERBS, enabled
from langslice.providers.registry import ImageModel, resolve_image_model

if TYPE_CHECKING:  # import cycle: the engine builds the toolbox
    from langslice.agent.engine import EngineContext

logger = logging.getLogger(__name__)

#: Tools whose host display targets are the whole stack.
_STACK_TOOLS = frozenset({"status", "undo", "redo", "submit"})


def _tool_target_ids(state: StackState, name: str, args: dict[str, Any]) -> list[str]:
    """Resolve host display targets before a tool can reorder the stack."""
    every = [record.id for record in state.in_order()]
    sections = args.get("sections")
    if name in _STACK_TOOLS:
        return every
    if name == "position_sections" and not sections:
        return every  # the cutting angles alone: every section
    if name == "set_preprocessed_channel_properties" and not sections:
        return every
    if name == "look" and not sections and args.get("mode") != "atlas":
        return every
    if name == "elastix_affine" and not sections:
        return [record.id for record in state.in_order()
                if record.position_mm is not None
                and (record.transform or {}).get("kind") != HOST_TRANSFORM_KIND]
    refs: list[Any] = []
    for entry in sections if isinstance(sections, (list, tuple)) else []:
        refs.append(entry.get("id", "") if isinstance(entry, dict) else entry)
    for key in ("section", "id"):
        if args.get(key) not in (None, ""):
            refs.append(args[key])
    for entry in args.get("slices") or []:
        refs.append(entry)
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
    """Save every picture *tool* returns in the job folder, with its layers,
    and list their numbers under ``pictures``.

    A reply that already lists ``pictures`` was saved by its operation
    (``look``, ``zoom``, the change tools' pictures: :mod:`langslice.ops.look`)
    and is left as it is. Otherwise the hook is the job's
    (:meth:`langslice.job.views.ViewStore.shown`), the same for every door:
    this wrapper says which pictures the tool returned and with which
    arguments, and the store numbers them at once and writes them in the
    background, in the doors' encoding, so the doors' bytes are untouched.
    """
    signature = inspect.signature(tool)

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        with job.views.shown(tool.__name__, atlas_of=lambda: ctx.atlas) as shown:
            result = tool(*args, **kwargs)
            media = result.get(TOOL_MEDIA_PARTS_KEY) if isinstance(result, dict) else None
            pictures = ([item for item in media if isinstance(item, Image.Image)]
                        if isinstance(media, list) else [])
            if pictures and "pictures" not in result:
                try:
                    bound = signature.bind(*args, **kwargs)
                    bound.apply_defaults()
                    arguments = dict(bound.arguments)
                except TypeError:
                    arguments = dict(kwargs)
                context = arguments.pop("tool_context", None)
                shown.show(pictures, arguments=plain(arguments),
                           call_id=plain(getattr(context, "function_call_id", None)))
        if shown.numbers and isinstance(result, dict):
            result["pictures"] = [
                {"id": number, **({"caption": text} if text else {})}
                for number, text in zip(shown.numbers, shown.captions or
                                        [None] * len(shown.numbers), strict=False)]
        return result

    return run


def _announces_work(tool: Any, job: Job) -> Any:
    """Open every reply with the notices of the background work that
    finished since the last reply (:meth:`langslice.job.background.BackgroundWork.notices`),
    and attach that work's pictures after the reply's own: ``background``
    holds the notices, ``pictures`` gains each work picture's number."""

    @functools.wraps(tool)
    def run(*args: Any, **kwargs: Any) -> Any:
        result = tool(*args, **kwargs)
        if not isinstance(result, dict):
            return result
        return with_notices(job, result)

    return run


def with_notices(job: Job, result: dict[str, Any]) -> dict[str, Any]:
    """*result* opened with the notices of the background work finished
    since they were last handed out, the work's pictures after its own."""
    finished = job.background.notices()
    if not finished:
        return result
    media = list(result.get(TOOL_MEDIA_PARTS_KEY) or [])
    listed = list(result.get("pictures") or [])
    for work in finished:
        for number, image in zip(work.pictures, work.images, strict=False):
            media.append(image)
            listed.append({"id": number, "work": work.id})
    out = {"background": [work.notice for work in finished], **result}
    if listed:
        out["pictures"] = listed
    if media:
        out[TOOL_MEDIA_PARTS_KEY] = media
    return out


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
    tool runs) and stray keys inside every typed-dict argument (``sections``
    entries, ``cutting_angles``, ``left_linear``; every caller).
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
    what only a tool door keeps: the look-before-commit gates.

    The job holds the state, undo/redo, the checkpoint, the submit gates,
    the image-correction jobs and the background work; the gates here are
    consulted by the tools alone, never by a library call.
    """

    job: Job
    tools: list[Any] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    #: Positions each section was looked at (``look`` in mode overlay or
    #: positioning) since its last write, and whether a positioning look of
    #: every section has run since the last write (the gates). A position
    #: that undo, redo or a reload of the state file moves counts as a
    #: write (``Door.forget_looks``); nothing else of this door's record is
    #: undone.
    compared: dict[str, set[float]] = field(default_factory=dict)
    reviewed: bool = False
    #: The largest picture the driver model takes: the cap of ``look``'s
    #: ``resolution`` at image resolution "auto".
    max_view_edge: int = DEFAULT_IMAGE_LIMIT[0]
    #: The opening-read gate of a door that delivers the opening pictures
    #: on request (MCP's ``show_stack`` pages): the pages not read yet,
    #: ``None`` while the door has not armed it (:meth:`require_opening`).
    #: Every write is refused (``OPENING_NOT_READ``) until the set is empty.
    opening_unread: set[int] | None = None

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
    each tool's body is in :mod:`.looking`, :mod:`.changing` or :mod:`.fitting`.
    The tools sit on *job*; without one, a job is made around *state* in the
    context's job folder (an empty undo history, nothing written until the
    first write). *max_view_edge* is the largest picture the driver model
    takes (None: the run model's lane,
    :func:`langslice.doors.tools.view_options.view_edge_limit`). *image_model*
    is the image model ``trace_borders`` calls when the run has one (None:
    this door resolves the spec's ``nonlinear.provider`` and ``image_model``
    through :func:`langslice.providers.registry.resolve_image_model`). *gates*
    False drops the look-before-commit gates of ``position.gated`` (the CLI
    door: gates are tool-only). *level* is the picture-size level the tools
    work at (None: the run's ``image_resolution``; the CLI passes "auto": its
    caller sizes every picture). *scripting* adds the verbs only scripts get
    (``Verb.scripting``: ``export_maps``, and the hidden ``trace_from_atlas``,
    called by name, listed nowhere), for the CLI and the library.
    *image_model_connected* False: the door cannot reach the spec's image
    model (MCP with none connected), so the tools are those of a run without
    one: no ``trace_borders``, ``submit`` not waiting for traces. The spec
    itself is left as it is. *door* is who reads the declarations
    (``agent``, ``mcp``, ``cli``: :data:`langslice.doors.declarations.DOORS`).

    Every tool is wrapped, innermost first: the opening-read gate (writes),
    the strict argument check, the stale-deformation sweep, the picture
    saving (``pictures``), the background-work notices at the head of the
    reply, and the serialization lock with the host events.
    """
    if job is None:
        job = Job(state, spec, layout=ctx.layout, results_path=ctx.results_path)
    if job.state is not state or job.spec is not spec:
        raise ValueError("build_tools: the job must hold this state and spec")
    if job.workspace is None:
        job.workspace = ctx
    box = ToolBox(job, max_view_edge=int(max_view_edge or view_edge_limit(ctx)))
    level = level or resolution_level(ctx)
    traces_on = spec.nonlinear.uses_image_model and image_model_connected
    if traces_on and image_model is None:
        image_model = resolve_image_model(spec.nonlinear.provider, spec.nonlinear.image_model)
    run = Door(job=job, ctx=ctx, spec=spec, box=box, level=level,
               gated=bool(gates and spec.position.gated), traces_on=traces_on,
               image_model=image_model, over_cap=job.over_cap)

    def sync_job() -> None:
        """Before every call: pick up a state file changed on disk."""
        before = job.sync()
        if before is not None:
            run.forget_looks(ops_history.moved_positions(before, job.state))

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
                run.forget_looks(ops_history.moved_positions(before, job.state))
            yield

    bodies: dict[str, Callable[..., Any]] = {
        **looking.bodies(run), **changing.bodies(run), **fitting.bodies(run),
    }
    prompt: str | None = None
    if traces_on and image_model is not None:
        from typing import cast

        from langslice.core.nonlinear.registration_tool import profile_prompt
        from langslice.core.space import Plane

        prompt = profile_prompt(image_model, cast(Plane, state.plane))[0]
    variant = Variant.of(spec, auto=level == AUTO_RESOLUTION, image_model=traces_on,
                         door=door, prompt=prompt)
    lock = threading.Lock()

    def behind_gate(name: str, tool: Any) -> Any:
        """A write behind the door's opening-read gate (armed by MCP only)."""
        return _opening_gate(tool, box) if VERBS[name].kind == "write" else tool

    box.tools = [
        _serialized(
            _announces_work(_saves_views(_clears_stale_deformations(
                _strict(behind_gate(name, declare(name, bodies[name], variant))), job),
                job, ctx), job),
            lock, state=state, on_event=on_event, guard=functools.partial(guarded, name))
        for name in enabled(spec, scripting=scripting, image_model=image_model_connected,
                            hidden=scripting)
    ]
    return box
