"""The tool pipeline: what every tool call goes through, in order.

A door (the built-in agent, MCP, the agent CLI, the library) calls a tool
as a plain function with the tool's declared signature and description
(:func:`declared_tool`); a plugin's tool is one more such function. Each
call becomes a :class:`ToolCall` and runs through :data:`TOOL_STEPS`
(:func:`run_call`): each step's part before the body in the order below,
then the tool's body, then each step's part after the body in reverse
order.

``_serialize``
    One call at a time, the job held as the tool needs it: a write under
    the job folder's write lock, a read as the state stands.
``_log_events``
    ``tool_start`` to the observers before; ``tool_end`` after, with the
    pictures the call saved.
``_announce_work``
    After: the reply opens with the notices of the background work that
    finished since the last reply.
``_save_pictures``
    After: the reply's pictures saved in the job folder, their numbers
    listed under ``pictures``.
``_name_registrations`` (writes)
    After: the registrations the write left, named.
``_clear_stale_deformations`` (writes)
    After: the deformations whose linear placement changed, dropped in the
    same undo step.
``_answer_refusals``
    A refusal raised by a later step or the body becomes the reply. It sits
    inside ``_clear_stale_deformations``, so a refused write still runs the
    sweep, which finds nothing changed.
``_strict_arguments``
    Before: unknown or misplaced arguments refused.
``_opening_gate`` (writes)
    Before: a write refused while an opening page is unread (MCP).

Tools report data: no advice, no interpretation, no strategy in any reply.
Their pictures are plain PIL images under ``TOOL_MEDIA_PARTS_KEY``; each
door packages them (ADK message parts, MCP content blocks, file paths).
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging
import threading
import uuid
from collections.abc import Callable, Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from PIL import Image

from langslice.core.pictures.media_keys import TOOL_MEDIA_PARTS_KEY
from langslice.core.pictures.opening import DEFAULT_IMAGE_LIMIT
from langslice.core.refusal import (
    FAILED_STATUSES,
    RefusalCode,
    Refused,
    ReplyStatus,
)
from langslice.job.job import Job
from langslice.job.pictures import PICTURE_FILE, captured
from langslice.tools.arguments import argument_refusal
from langslice.tools.catalog import Verb
from langslice.tools.descriptions.description import DescriptionFacts
from langslice.tools.descriptions.render import declare
from langslice.tools.events import (
    LiveCallback,
    call_id,
    plain,
    reply_status,
    target_ids,
)

if TYPE_CHECKING:  # import cycle: a run holds its toolbox
    from langslice.tools.run import Run

logger = logging.getLogger(__name__)


@dataclass
class ToolBox:
    """The tools of one run over its job, and what only the tools keep.

    The job holds the state, undo and redo, the checkpoint, the submit
    gates, the image-correction jobs and the background work; the toolbox
    holds the tools, the submission, the opening-read gate and what every
    call's steps share (the lock, the observers, the verbs).
    """

    job: Job
    #: The facts of the run the tools are declared in: what ``TOOLS.md``
    #: is written from.
    variant: DescriptionFacts
    tools: list[Callable[..., object]] = field(default_factory=list)
    submission: dict[str, Any] = field(default_factory=dict)
    #: The largest picture the driver model takes: the cap of ``look``'s
    #: ``resolution`` at image resolution "auto".
    max_view_edge: int = DEFAULT_IMAGE_LIMIT[0]
    #: The opening-read gate of a door that delivers the opening pictures
    #: on request (MCP's ``show_stack`` pages): the pages not read yet,
    #: ``None`` while the door has not armed it (:meth:`require_opening`).
    #: Every write is refused (``OPENING_NOT_READ``) until the set is empty.
    opening_unread: set[int] | None = None
    #: The provenance of the tool plugins the job loaded (module, factory,
    #: versions, source digests, each tool's declaration as the model sees
    #: it); empty without plugins.
    plugins: list[dict[str, Any]] = field(default_factory=list)
    #: What every tool body of the box shares (the job, the workspace, the
    #: picture level, the image model): a script's ``job.context`` sits on it.
    run: Run | None = None
    #: Every verb the job can have, by name: read or write, long or not.
    verbs: Mapping[str, Verb] = field(default_factory=dict[str, Verb])
    #: The door the job folder's event log names for every call.
    door: str = "agent"
    #: Who is sent each call's events: a host's display, the event log.
    observers: tuple[LiveCallback, ...] = ()
    #: One call at a time: a model's several calls of one turn may arrive
    #: together, and every tool shares one state.
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def names(self) -> list[str]:
        """Return the tools' names, in the order they are declared."""
        return [tool.__name__ for tool in self.tools]

    def require_opening(self, pages: int) -> None:
        """Arm the opening-read gate: refuse writes until each page is read.

        Armed once per toolbox; a host that already read the pages is not
        asked again.

        Parameters
        ----------
        pages
            How many opening pages the host delivers.
        """
        if self.opening_unread is None:
            self.opening_unread = set(range(1, int(pages) + 1))

    def opening_read(self, page: int) -> None:
        """Mark an opening page as read.

        Parameters
        ----------
        page
            The page the host fetched, from 1.
        """
        if self.opening_unread is not None:
            self.opening_unread.discard(int(page))

    def notify(self, event: dict[str, Any]) -> None:
        """Send an event to every observer, skipping one that fails.

        Parameters
        ----------
        event
            A call's ``tool_start`` or ``tool_end`` event.
        """
        for observer in self.observers:
            try:
                observer(event)
            except Exception:  # a host's display gone: the call goes on
                logger.warning("Tool execution observer failed", exc_info=True)

    def opening_refusal(self) -> Refused | None:
        """Return the opening-read gate's refusal.

        Returns
        -------
        Refused or None
            ``OPENING_NOT_READ`` with the unread pages, or None when the gate
            is open.
        """
        if not self.opening_unread:
            return None
        pages = sorted(self.opening_unread)
        calls = ", ".join(f"show_stack(page={page})" for page in pages)
        return Refused(
            RefusalCode.OPENING_NOT_READ,
            status=ReplyStatus.REFUSED,
            pages=pages,
            detail=f"Read every opening page before the first write: {calls}"
            " (the pictures of the stack the job statement asks for).",
        )


@dataclass(eq=False)
class ToolCall:
    """One call of one tool on its way through :data:`TOOL_STEPS`.

    The steps read the call and set its reply; a step that refuses the
    call before the body leaves the body unrun.
    """

    #: The tool's name, as the model calls it.
    name: str
    #: The tool's declaration over its body: binds the arguments to the
    #: declared signature, fills the defaults and runs the body.
    declared: Callable[..., object]
    #: The toolbox the tool belongs to.
    box: ToolBox
    #: The arguments as the door passed them, by position and by name.
    args: tuple[object, ...]
    kwargs: dict[str, object]
    #: The reply, once the body or a step has answered: the tool's own
    #: (a dict of plain values and pictures, for every native tool).
    reply: object = None
    #: The paths of the pictures the call saved in the job folder.
    pictures: list[str] = field(default_factory=list[str])

    @property
    def verb(self) -> Verb:
        """Return what the tool is: read or write, long or not."""
        return self.box.verbs[self.name]

    def bound(self) -> inspect.BoundArguments:
        """Bind the arguments to the declaration and fill the defaults.

        Returns
        -------
        inspect.BoundArguments
            Every declared argument by name, ADK's ``tool_context`` among
            them when the tool takes it.

        Raises
        ------
        TypeError
            When the arguments do not bind to the declared signature.
        """
        signature = inspect.signature(self.declared)
        bound = signature.bind(*self.args, **self.kwargs)
        bound.apply_defaults()
        return bound


# --- the steps, in the order a call enters them ------------------------------


@contextlib.contextmanager
def _serialize(call: ToolCall) -> Generator[None]:
    """Run one call at a time, with the job held as the tool needs it.

    A write runs whole under the job folder's write lock, after the state
    is brought up to date; a read never waits for another process's writer
    (the state as it stands); a long tool only syncs, computes outside the
    lock and takes it itself to apply, re-checking its sections. The job
    names the registrations the call leaves for the tool.
    """
    job = call.box.job
    with call.box.lock, job.acting(call.name):
        if call.verb.long:
            job.sync()
            yield
        elif call.verb.kind == "read":
            with job.reading():
                yield
        else:
            with job.writing():
                yield


@contextlib.contextmanager
def _log_events(call: ToolCall) -> Generator[None]:
    """Send ``tool_start`` before the call and ``tool_end`` after it.

    ``tool_end`` names the pictures the call saved and whether it failed;
    a crash is sent as the refusal its exception makes and raised on. An
    observer's failure never changes the call.
    """
    box = call.box
    fields: dict[str, Any] = {
        "name": call.name,
        "execution_id": uuid.uuid4().hex,
    }
    try:
        arguments: dict[str, object] = dict(call.bound().arguments)
        context = arguments.pop("tool_context", None)
        fields.update(
            id=call_id(context),
            args=plain(arguments),
            target_ids=target_ids(box.job.state, call.name, arguments),
        )
    except Exception:  # bad display metadata must never change the call
        logger.warning("Tool execution metadata unavailable", exc_info=True)
        fields.update(id=None, args={}, target_ids=[])
    box.notify({"kind": "tool_start", **fields})
    try:
        with captured() as saved:
            yield
    except Exception as error:
        crash = Refused.from_exception(error).payload()
        end = {"kind": "tool_end", **fields, "failed": True, "response": crash}
        box.notify(end)
        raise
    call.pictures = [str(item.folder / PICTURE_FILE) for item in saved]
    response = plain(call.reply)
    failed = reply_status(response) in FAILED_STATUSES
    end = {"kind": "tool_end", **fields, "failed": failed}
    box.notify({**end, "response": response, "views": call.pictures})


@contextlib.contextmanager
def _announce_work(call: ToolCall) -> Generator[None]:
    """Open the reply with the notices of the work finished since the last.

    Background work (a fit, a trace) that finished since the last reply
    is announced once, its pictures after the reply's own
    (:func:`with_notices`).
    """
    yield
    reply = call.reply
    if isinstance(reply, dict):
        fields = cast("dict[str, Any]", reply)  # a tool's reply: by name
        call.reply = with_notices(call.box.job, fields)


@contextlib.contextmanager
def _save_pictures(call: ToolCall) -> Generator[None]:
    """Save the pictures the reply carries and list their numbers.

    A reply that already lists ``pictures`` was saved by its operation
    (``look``, ``zoom``, the change tools' pictures) and is left as it is.
    Otherwise the job's picture store numbers the pictures at once and
    writes them, with their layers, in the background.
    """
    job = call.box.job
    run = call.box.run
    # a box without its run (before build_tools sets it) saves no layers
    atlas_of = None
    if run is not None:
        atlas_of = functools.partial(getattr, run.ctx, "atlas")
    with job.views.shown(call.name, atlas_of=atlas_of) as shown:
        yield
        pictures = _new_pictures(call.reply)
        if pictures:
            try:
                arguments: dict[str, object] = dict(call.bound().arguments)
            except TypeError:
                arguments = dict(call.kwargs)
            context = arguments.pop("tool_context", None)
            shown.show(
                pictures, arguments=plain(arguments), call_id=call_id(context)
            )
    reply = call.reply
    if shown.numbers and isinstance(reply, dict):
        captions = shown.captions or [None] * len(shown.numbers)
        listed: list[dict[str, object]] = []
        for number, text in zip(shown.numbers, captions, strict=False):
            caption = {"caption": text} if text else {}
            listed.append({"id": number, **caption})
        fields = cast("dict[str, object]", reply)  # a tool's reply: by name
        fields["pictures"] = listed


@contextlib.contextmanager
def _name_registrations(call: ToolCall) -> Generator[None]:
    """Name the registration each section the write changed now holds."""
    mark = len(call.box.job.registered)
    yield
    call.reply = registrations_named(call.box.job, call.reply, mark)


@contextlib.contextmanager
def _clear_stale_deformations(call: ToolCall) -> Generator[None]:
    """Drop the deformations whose linear placement the write changed.

    The rule is the job's: cleared in the same undo step as the write that
    made them stale, so ``undo`` restores the placement and the deformation
    together; the reply lists them under ``deformation_cleared``.
    """
    yield
    cleared = call.box.job.clear_stale_deformations()
    reply = call.reply
    if cleared and isinstance(reply, dict):
        fields = cast("dict[str, object]", reply)  # a tool's reply: by name
        fields["deformation_cleared"] = cleared


@contextlib.contextmanager
def _answer_refusals(call: ToolCall) -> Generator[None]:
    """Answer a refusal raised by a later step or the body with its reply.

    The tools' one conversion of a refusal to a reply
    (:meth:`Refused.payload`), so the steps above it see a reply, as for
    any call.
    """
    try:
        yield
    except Refused as refused:
        call.reply = refused.payload()


@contextlib.contextmanager
def _strict_arguments(call: ToolCall) -> Generator[None]:
    """Refuse unknown or misplaced arguments before the tool runs.

    Stray top-level names (ADK and MCP check those at their own doors,
    since both drop them before a tool runs) and stray keys inside every
    typed-dict argument (``sections`` entries, ``cutting_angles``), against
    the tool's signature as the model sees it.
    """
    names = list(inspect.signature(call.declared).parameters)
    given = {**dict(zip(names, call.args, strict=False)), **call.kwargs}
    refusal = argument_refusal(call.declared, given)
    if refusal is not None:
        raise refusal
    yield


@contextlib.contextmanager
def _opening_gate(call: ToolCall) -> Generator[None]:
    """Refuse a write while some opening page is unread (MCP's gate)."""
    refusal = call.box.opening_refusal()
    if refusal is not None:
        raise refusal
    yield


#: What one step does around the body: before its ``yield``, then after.
StepFunction = Callable[[ToolCall], contextlib.AbstractContextManager[None]]


@dataclass(frozen=True)
class Step:
    """One step of :data:`TOOL_STEPS`, and the calls it runs for."""

    #: What the step does around the body.
    around: StepFunction
    #: Whether the step runs for write tools only.
    writes_only: bool = False


#: The steps every tool call goes through, in the order it enters them; the
#: module docstring says what each does.
TOOL_STEPS: tuple[Step, ...] = (
    Step(_serialize),
    Step(_log_events),
    Step(_announce_work),
    Step(_save_pictures),
    Step(_name_registrations, writes_only=True),
    Step(_clear_stale_deformations, writes_only=True),
    Step(_answer_refusals),
    Step(_strict_arguments),
    Step(_opening_gate, writes_only=True),
)


def run_call(call: ToolCall) -> object:
    """Run a call through every step that applies to it, then its body.

    Parameters
    ----------
    call
        The call, as the tool's door made it.

    Returns
    -------
    object
        The reply, as the steps left it.
    """
    writes = call.verb.kind == "write"
    with contextlib.ExitStack() as entered:
        for step in TOOL_STEPS:
            if writes or not step.writes_only:
                entered.enter_context(step.around(call))
        call.reply = call.declared(*call.args, **call.kwargs)
    return call.reply


def declared_tool(
    name: str, body: Callable[..., object], box: ToolBox
) -> Callable[..., object]:
    """Return a tool body as a declared tool whose calls run the pipeline.

    Parameters
    ----------
    name
        The tool's name; its declaration in the box's variant gives the
        signature and the description ADK and FastMCP read.
    body
        The tool's body, taking every declared argument by name.
    box
        The toolbox the tool belongs to.

    Returns
    -------
    callable
        The tool: each call is a :class:`ToolCall` through
        :func:`run_call`.
    """
    declared = declare(name, body, box.variant)

    @functools.wraps(declared)
    def tool(*args: object, **kwargs: object) -> object:
        return run_call(ToolCall(name, declared, box, args, kwargs))

    return tool


# --- what the steps add to a reply -------------------------------------------


def _new_pictures(reply: object) -> list[Image.Image]:
    """Return the pictures a reply carries that no operation saved yet.

    Parameters
    ----------
    reply
        A tool's reply.

    Returns
    -------
    list of PIL.Image.Image
        The pictures under the media key, or none when the reply already
        lists its ``pictures``.
    """
    if not isinstance(reply, dict):
        return []
    fields = cast("dict[str, object]", reply)  # a tool's reply: by name
    media = fields.get(TOOL_MEDIA_PARTS_KEY)
    if "pictures" in fields or not isinstance(media, list):
        return []
    items = cast("list[object]", media)  # pictures and lines of text
    return [item for item in items if isinstance(item, Image.Image)]


def registrations_named(job: Job, reply: object, mark: int) -> object:
    """Name the registrations a write left that its reply's rows do not show.

    Parameters
    ----------
    job
        The job the write was made in.
    reply
        The write's reply; a reply with a ``registration`` of its own keeps
        it.
    mark
        ``len(job.registered)`` before the write.

    Returns
    -------
    object
        The reply, with ``registration`` ``{section: number}``: the
        registrations this thread's writes left since *mark*, less the ones
        its status rows (``changed``, ``rows``) show.
    """
    left = job.registered_since(mark)
    if not left or not isinstance(reply, dict):
        return reply
    fields = cast("dict[str, object]", reply)  # a tool's reply: by name
    if "registration" in fields:
        return fields
    in_rows: dict[object, object] = {}
    for key in ("changed", "rows"):
        rows = cast("list[object]", fields.get(key) or [])  # status rows
        for row in rows:
            if isinstance(row, dict):
                columns = cast("dict[str, object]", row)  # a row by column
                in_rows[columns.get("id")] = columns.get("registration")
    named = {
        section: number
        for section, number in left.items()
        if in_rows.get(section) != number
    }
    if named:
        fields["registration"] = named
    return fields


def with_notices(job: Job, result: dict[str, Any]) -> dict[str, Any]:
    """*result* opened with the notices of the background work finished
    since they were last handed out, the work's pictures after its own."""
    finished = job.background.notices()
    if not finished:
        return result
    media = list(result.get(TOOL_MEDIA_PARTS_KEY) or [])
    listed = list(result.get("pictures") or [])
    for work in finished:
        captions = list(work.captions) + [""] * len(work.pictures)
        for number, image, text in zip(
            work.pictures, work.images, captions, strict=False
        ):
            media.append(image)
            listed.append(
                {
                    "id": number,
                    "work": work.id,
                    **({"caption": text} if text else {}),
                }
            )
    out = {"background": [work.notice for work in finished], **result}
    if listed:
        out["pictures"] = listed
    if media:
        out[TOOL_MEDIA_PARTS_KEY] = media
    return out


def work_pictures(
    job: Job, reply: object, *, start: int
) -> list[dict[str, object]]:
    """Return the saved pictures of the background work a reply announces.

    Parameters
    ----------
    job
        The job whose picture store holds them.
    reply
        The reply; its ``pictures`` entries with a ``work`` are the work's.
    start
        The index of the first, after the call's own pictures.

    Returns
    -------
    list of dict
        One artifact per picture on disk: ``path``, ``kind``, ``index``,
        ``label``.
    """
    out: list[dict[str, object]] = []
    if not isinstance(reply, dict):
        return out
    fields = cast("dict[str, object]", reply)  # a tool's reply: by name
    entries = cast("list[object]", fields.get("pictures") or [])  # listed
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        picture = cast("dict[str, object]", entry)  # id, work, caption
        work = picture.get("work")
        if not work:
            continue
        number = cast("int", picture["id"])  # a picture number
        record = job.views.lookup(int(number))
        if record is None or not record.path:
            continue
        path = Path(job.layout.resolve(record.path)) / PICTURE_FILE
        if path.exists():
            label = f"{work} ({record.mode or record.tool})"
            out.append(
                {
                    "path": str(path),
                    "kind": "view",
                    "index": start + len(out),
                    "label": label,
                }
            )
    return out
