"""Every picture the model was shown, saved in the job folder with its layers.

Per picture, one folder ``<seq>_<tool>[_<mode>]`` (``seq`` zero-padded, one
number per picture across the job's life):

- ``view.jpg`` — the exact JPEG bytes the model received (the doors' one
  encoding, :func:`langslice.core.jpeg.encode_jpeg`, applied to the same
  picture; a door that sends its own bytes, the opening, hands them over);
- for a placement picture (a section on its physical canvas), lossless
  layers on the same pixel grid (:mod:`langslice.core.layers`):
  ``labels.tif`` (atlas ids, uint32, deflate-compressed) and
  ``borders.png`` (the drawn borders' coverage, 8-bit);
- ``view.json`` — the tool, call, arguments, sections and mode, the
  picture's size and, for a placement picture, its frame: plane, µm/px,
  placement, ``pixel_to_atlas_um`` (BrainGlobe µm, atlas axis order) and
  the applied deformation's folder (relative to the job folder).
  :func:`langslice.core.layers.coordinate_map` turns it into a per-pixel
  atlas coordinate map on demand. It also holds the picture's ``caption``,
  its ``step`` (the history depth when it was saved) and its ``recipe``
  (``{"renderer": name, "args": {...}}``, enough to redraw it; None when the
  caller gave none).

A picture of one section goes under ``sections/<name>/views/``, one of
several (or none: an atlas section) under ``views/``. ``views.jsonl`` at the
top of the job folder lists every saved picture, one line each, appended in
order, with the same ``caption``, ``step`` and ``recipe`` (each left out of
a line while None).

Every picture can be looked up by its ``seq``: :meth:`ViewStore.lookup` and
:meth:`ViewStore.latest` read the index (:class:`PictureRecord`);
:class:`DiscardedViews` (a lean job) keeps the same index in memory, with the
newest pictures themselves, for the life of the process.

Saving never touches what the doors send and never slows a tool: the tool
thread only numbers the pictures and queues them; one background thread per
store encodes and writes, in order, and ends when the queue is empty.
:meth:`ViewStore.flush` waits for it (the job calls it at submit, results and
close; the library after every verb; :func:`flush_all` runs at interpreter
exit, through :func:`at_exit`, while the writer can still start the threads
its encoders use). A failed write is logged and skipped; it never reaches a
tool.
"""

from __future__ import annotations

import atexit
import concurrent.futures.thread  # noqa: F401  (its exit hook is registered first: at_exit)
import contextlib
import contextvars
import json
import logging
import queue
import re
import threading
import time
import weakref
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from langslice.core.jpeg import encode_jpeg
from langslice.core.layers import (
    RESIDUAL_LAYER,
    PictureNote,
    collecting,
    note_for,
    picture_layers,
    warp_layers,
)
from langslice.job.layout import JobLayout

#: Beside ``views.jsonl``: the numbers handed out so far (``seq``: the next
#: picture's, ``call``: the last call's), and its lock ``views.seq.lock``.
VIEWS_COUNTER = "views.seq"
#: How long a store waits for another to finish numbering its pictures.
NUMBERING_TIMEOUT_S = 30.0

logger = logging.getLogger(__name__)

#: ``view.json``'s format.
VIEW_FORMAT_VERSION = 1
VIEW_FILE = "view.json"
PICTURE_FILE = "view.jpg"
LABELS_FILE = "labels.tif"
BORDERS_FILE = "borders.png"
RESIDUAL_FILE = RESIDUAL_LAYER

_NAME_PART = re.compile(r"[^a-zA-Z0-9_]+")

#: Every store with pictures queued at some point (for :func:`flush_all`).
_STORES: weakref.WeakSet[ViewStore] = weakref.WeakSet()
#: The open :func:`captured` lists of this context, innermost last.
_CAPTURES: contextvars.ContextVar[tuple[list[Saved], ...]] = contextvars.ContextVar(
    "langslice_view_captures", default=())


@dataclass(frozen=True)
class Saved:
    """One queued picture: its folder (absolute) and whether it gets layers."""

    folder: Path
    name: str
    layers: bool
    #: A deformable-fit picture with its residual drawn (``residual.tif``).
    residual: bool = False
    #: What the picture shows, from its note: the sections and the mode.
    sections: tuple[str, ...] = ()
    mode: str | None = None
    #: Its place among the call's pictures (zero-based, the order a reply's
    #: ``image_indexes`` count in).
    index: int = 0
    #: The picture's number (``view.json``'s ``seq``): what :meth:`ViewStore.lookup` takes.
    seq: int = 0

    def files(self) -> list[tuple[Path, str]]:
        """``(path, kind)`` of every file the picture's folder will hold:
        ``view`` (the JPEG), ``view_json``, for a placement or deformable-fit
        picture ``labels`` and ``borders``, and ``residual`` for a
        deformable-fit picture showing its residual."""
        listed = [(self.folder / PICTURE_FILE, "view"), (self.folder / VIEW_FILE, "view_json")]
        if self.layers:
            listed += [(self.folder / LABELS_FILE, "labels"), (self.folder / BORDERS_FILE,
                                                               "borders")]
        if self.residual:
            listed.append((self.folder / RESIDUAL_FILE, "residual"))
        return listed


@dataclass(frozen=True)
class PictureRecord:
    """One indexed picture (a ``views.jsonl`` line, or a lean job's memory)."""

    seq: int
    name: str
    tool: str
    call: int
    #: The picture's folder relative to the job folder ("" in a lean job).
    path: str
    sections: tuple[str, ...] = ()
    mode: str | None = None
    layers: bool = False
    caption: str | None = None
    step: int | None = None
    recipe: dict[str, Any] | None = None
    #: The picture itself, kept only by :class:`DiscardedViews`.
    image: Image.Image | bytes | None = field(default=None, compare=False, repr=False)

    @classmethod
    def from_entry(cls, entry: dict[str, Any]) -> PictureRecord:
        step = entry.get("step")
        return cls(seq=int(entry["seq"]), name=str(entry.get("name", "")),
                   tool=str(entry.get("tool", "")), call=int(entry.get("call", 0)),
                   path=str(entry.get("path", "")),
                   sections=tuple(entry.get("sections") or ()), mode=entry.get("mode"),
                   layers=bool(entry.get("layers")), caption=entry.get("caption"),
                   step=int(step) if step is not None else None,
                   recipe=entry.get("recipe"))


def _newest(records: Iterable[PictureRecord], exclude_tool: str | None) -> PictureRecord | None:
    found: PictureRecord | None = None
    for record in records:
        if record.tool != exclude_tool and (found is None or record.seq > found.seq):
            found = record
    return found


def artifacts(saved: Iterable[Saved], tool: str) -> tuple[list[dict[str, Any]], list[str]]:
    """``(artifacts, warnings)`` of a call's saved pictures, once the store
    is flushed: every file a picture's folder holds, ``{"path", "kind",
    "index"}`` (``index`` its place among the call's pictures, the number a
    reply's ``image_indexes`` give), the ``view`` with a ``label`` (the
    sections and mode it shows, else *tool*); a warning per picture whose
    JPEG is missing. What the doors that answer with file paths (the agent
    CLI, the library) report."""
    listed: list[dict[str, Any]] = []
    warnings: list[str] = []
    for picture in saved:
        label = ", ".join(picture.sections) + (f" ({picture.mode})" if picture.mode else "")
        for path, kind in picture.files():
            if path.exists():
                listed.append({"path": str(path), "kind": kind, "index": picture.index,
                               **({"label": label or tool} if kind == "view" else {})})
            elif kind == "view":
                warnings.append(f"picture not saved: {path}")
    return listed, warnings


def has_frame(note: PictureNote | None) -> bool:
    """Whether a picture's note carries a frame its layers are drawn from: a
    placement picture's canvas, or a deformable-fit picture's record."""
    return note is not None and ((note.panel is not None and note.frame is not None)
                                 or note.warp is not None)


@contextlib.contextmanager
def captured() -> Iterator[list[Saved]]:
    """Collect every picture any store queues in this block (this thread or
    task), in order: what a door that answers with file paths (the CLI)
    reports. The files exist once the store is flushed."""
    found: list[Saved] = []
    token = _CAPTURES.set((*_CAPTURES.get(), found))
    try:
        yield found
    finally:
        _CAPTURES.reset(token)


def flush_all() -> None:
    """Wait until every store's queued pictures are written."""
    for store in list(_STORES):
        store.flush()


def at_exit(function: Callable[[], None]) -> None:
    """Run *function* when the interpreter exits, while threads still run.

    ``atexit`` is too late for work that starts threads: by then
    ``concurrent.futures`` takes no new work ("cannot schedule new futures
    after interpreter shutdown"), and the picture writer's TIFF encoder and
    an image-model call use thread pools. Python's threading exit hooks run
    before that one, last registered first; ``concurrent.futures.thread`` is
    imported above so that its hook is registered before any of ours. Falls
    back to ``atexit`` where the hook does not exist."""
    register = getattr(threading, "_register_atexit", None)
    if register is None:
        atexit.register(function)
        return
    try:
        register(function)
    except RuntimeError:  # the interpreter is already shutting down
        atexit.register(function)


at_exit(flush_all)
# And once more after every thread has been joined, for pictures queued by a
# thread that was still running a tool when the first flush ran.
atexit.register(flush_all)


def _items(
    seq: int, tool: str, pictures: list[tuple[Image.Image | bytes, PictureNote | None]],
    step: int | None, recipes: list[dict[str, Any] | None] | None,
    captions: list[str | None] | None,
) -> list[_Picture]:
    """The numbered pictures of one call, the first numbered *seq*. A recipe
    or caption comes from the picture's note, else from *recipes* /
    *captions* (one per picture)."""
    items: list[_Picture] = []
    for index, (image, held) in enumerate(pictures):
        mode = held.mode if held is not None else None
        recipe = held.recipe if held is not None else None
        caption = held.caption if held is not None else None
        if recipe is None and recipes is not None and index < len(recipes):
            recipe = recipes[index]
        if caption is None and captions is not None and index < len(captions):
            caption = captions[index]
        items.append(_Picture(seq + index, view_name(seq + index, tool, mode), index, image,
                              held, step, recipe, caption))
    return items


def view_name(seq: int, tool: str, mode: str | None) -> str:
    """``<seq>_<tool>[_<mode>]``: six-digit sequence, then what drew it."""
    parts = [f"{seq:06d}", _NAME_PART.sub("-", tool).strip("-") or "picture"]
    if mode:
        parts.append(_NAME_PART.sub("-", str(mode)).strip("-"))
    return "_".join(part for part in parts if part)


@dataclass
class _Picture:
    seq: int
    name: str
    index: int
    image: Image.Image | bytes
    note: PictureNote | None
    step: int | None = None
    recipe: dict[str, Any] | None = None
    caption: str | None = None


@dataclass
class _Call:
    tool: str
    call: int
    call_id: str | None
    arguments: Any
    pictures: list[_Picture]
    atlas: Any


@dataclass
class Shown:
    """What one door call showed (:meth:`ViewStore.shown`): the pictures, in
    the order the model receives them, the call's arguments and its id."""

    pictures: list[Image.Image] = field(default_factory=list)
    arguments: Any = None
    call_id: str | None = None

    def show(self, pictures: Iterable[Image.Image], *, arguments: Any = None,
             call_id: str | None = None) -> None:
        """Record the pictures the door sends (and the call they answer)."""
        self.pictures.extend(pictures)
        self.arguments = arguments
        self.call_id = call_id


class ViewStore:
    """The job folder's saved pictures (see the module text)."""

    def __init__(self, layout: JobLayout) -> None:
        self.layout = layout
        self._lock = threading.Lock()
        self._queue: queue.Queue[_Call] = queue.Queue()
        self._running = False
        self._seq: int | None = None
        self._call = 0
        #: Asked for the history depth when a picture is saved (the job sets
        #: it); None: ``step`` stays unrecorded unless :meth:`save` is given one.
        self.step_source: Callable[[], int] | None = None
        #: Bytes of the views index already read for the numbering.
        self._index_read = 0
        #: Seconds the calling threads spent in :meth:`save` (numbering and
        #: queueing), and the writer spent per call: kept for measuring.
        self.queue_seconds = 0.0
        self.write_seconds = 0.0

    # --- numbering ----------------------------------------------------------------

    def _next_numbers(self) -> tuple[int, int]:
        """The next picture and call numbers, continuing the index's: a
        reopened job's, and another store's on the same folder (a running
        agent and a CLI call), read as the index grows."""
        if self._seq is None:
            self._seq, self._call, self._index_read = 1, 0, 0
        try:
            with self.layout.views_index.open("rb") as handle:
                handle.seek(self._index_read)
                for line in handle:
                    if not line.endswith(b"\n"):
                        break  # a line still being written: read it next time
                    self._index_read += len(line)
                    try:
                        entry = json.loads(line)
                    except ValueError:
                        continue
                    self._seq = max(self._seq, int(entry.get("seq", 0)) + 1)
                    self._call = max(self._call, int(entry.get("call", 0)))
        except OSError:
            pass
        return self._seq, self._call

    @contextlib.contextmanager
    def _numbering(self) -> Iterator[tuple[int, int]]:
        """The next picture and call numbers, reserved across processes.

        Numbers handed out are recorded in ``views.seq`` (the next picture
        and the last call) under its own file lock, before any picture is
        written, so two stores on one folder (two processes) that queue
        pictures before either writes never share a number. The block
        records the numbers it used (``self._seq`` / ``self._call``) on
        exit. Without a usable folder: the index alone, as before."""
        from filelock import FileLock, Timeout

        from langslice.job.checkpoint import write_json_atomic

        counter = self.layout.folder / VIEWS_COUNTER
        try:
            lock: Any = FileLock(str(counter) + ".lock", timeout=NUMBERING_TIMEOUT_S)
            lock.acquire()
        except (OSError, Timeout):
            yield self._next_numbers()
            return
        try:
            seq, call = self._next_numbers()
            try:
                held = json.loads(counter.read_text(encoding="utf-8"))
                seq = max(seq, int(held.get("seq", 0)))
                call = max(call, int(held.get("call", 0)))
            except (OSError, ValueError, TypeError, AttributeError):
                pass
            self._seq, self._call = seq, call
            yield seq, call
            write_json_atomic(str(counter), {"seq": self._seq, "call": self._call})
        finally:
            lock.release()

    # --- saving -------------------------------------------------------------------

    def save(
        self, *, tool: str, pictures: list[tuple[Image.Image | bytes, PictureNote | None]],
        arguments: Any = None, call_id: str | None = None, atlas: Any = None,
        step: int | None = None, recipes: list[dict[str, Any] | None] | None = None,
        captions: list[str | None] | None = None,
    ) -> list[str]:
        """Queue one call's pictures (in the order the model received them).

        Each is a PIL picture (encoded here as the doors encode it) or the
        JPEG bytes a door sent, with its note (None: nothing known but the
        tool). *atlas* draws placement pictures' layers. Returns the
        pictures' names, at once; the files follow in the background.

        A picture's ``recipe`` and ``caption`` come from its note, else from
        *recipes* / *captions* (one per picture); *step* is the history depth
        (default: ``step_source()``).
        """
        if not pictures:
            return []
        started = time.perf_counter()
        step = self._step(step)
        with self._lock, self._numbering() as (seq, call):
            call += 1
            items = _items(seq, tool, pictures, step, recipes, captions)
            seq += len(items)
            for capture in _CAPTURES.get():
                capture.extend(
                    Saved(self._folder(item.note, item.name), item.name,
                          has_frame(item.note) and atlas is not None,
                          item.note is not None and item.note.warp is not None
                          and item.note.warp.warped and atlas is not None,
                          sections=tuple(item.note.sections) if item.note else (),
                          mode=item.note.mode if item.note else None, index=item.index,
                          seq=item.seq)
                    for item in items)
            self._seq, self._call = seq, call
            self._queue.put(_Call(tool, call, call_id, arguments, items, atlas))
            if not self._running:
                self._running = True
                _STORES.add(self)
                threading.Thread(target=self._work, name="langslice-views",
                                 daemon=True).start()
        self.queue_seconds += time.perf_counter() - started
        return [item.name for item in items]

    def _step(self, step: int | None) -> int | None:
        if step is None and self.step_source is not None:
            with contextlib.suppress(Exception):
                return int(self.step_source())
        return step

    # --- the index ----------------------------------------------------------------

    def records(self) -> list[PictureRecord]:
        """Every indexed picture, in order (queued pictures are written first)."""
        self.flush()
        found: list[PictureRecord] = []
        try:
            with self.layout.views_index.open("rb") as handle:
                for line in handle:
                    try:
                        found.append(PictureRecord.from_entry(json.loads(line)))
                    except (ValueError, KeyError, TypeError):
                        continue
        except OSError:
            pass
        return found

    def lookup(self, seq: int) -> PictureRecord | None:
        """The picture numbered *seq*, or None."""
        for record in self.records():
            if record.seq == seq:
                return record
        return None

    def latest(self, exclude_tool: str | None = "zoom") -> PictureRecord | None:
        """The newest picture not drawn by *exclude_tool*, or None."""
        return _newest(self.records(), exclude_tool)

    @contextlib.contextmanager
    def shown(
        self, tool: str, *, atlas_of: Callable[[], Any] | None = None,
    ) -> Iterator[Shown]:
        """Save the pictures a door shows from this block, with their layers.

        The one hook every door uses (the agent tools and the MCP door
        through the tool wrapper, ``toolbox._saves_views``; the CLI and
        scripts the same way): run the operation inside the block, then hand
        the pictures it shows to :meth:`Shown.show`. What the core noted
        about each while the block ran (:func:`langslice.core.layers.collecting`)
        decides its folder, its mode and, for a placement picture, its layers
        (the atlas from *atlas_of*, asked only then). A block that raises
        saves nothing; saving never raises into the door.
        """
        call = Shown()
        with collecting() as notes:
            yield call
        if not call.pictures:
            return
        noted = [(picture, note_for(picture, notes)) for picture in call.pictures]
        placed = any(has_frame(held) for _p, held in noted)
        try:
            self.save(tool=tool, pictures=list(noted), arguments=call.arguments,
                      call_id=call.call_id,
                      atlas=atlas_of() if placed and atlas_of is not None else None)
        except Exception:  # saving must never break a tool
            logger.warning("Could not queue the pictures of %s", tool, exc_info=True)

    def flush(self) -> None:
        """Wait until every queued picture is written."""
        self._queue.join()

    # --- the writer ------------------------------------------------------------------

    def _work(self) -> None:
        """Write queued calls in order; end once the queue is empty."""
        while True:
            with self._lock:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    self._running = False
                    return
            try:
                started = time.perf_counter()
                self._write_call(item)
                self.write_seconds += time.perf_counter() - started
            except Exception:
                logger.warning("Could not save the pictures of a %s call", item.tool,
                               exc_info=True)
            finally:
                self._queue.task_done()

    def _folder(self, note: PictureNote | None, name: str) -> Path:
        sections = note.sections if note is not None else ()
        if len(sections) == 1:
            return self.layout.section_views_dir(sections[0]) / name
        return self.layout.views_dir / name

    def _write_call(self, call: _Call) -> None:
        lines: list[str] = []
        for picture in call.pictures:
            try:
                lines.append(self._write_picture(call, picture))
            except Exception:
                logger.warning("Could not save picture %s", picture.name, exc_info=True)
        if lines:
            with self.layout.views_index.open("a", encoding="utf-8") as handle:
                handle.write("".join(lines))

    def _write_picture(self, call: _Call, picture: _Picture) -> str:
        import tifffile

        note = picture.note
        folder = self._folder(note, picture.name)
        folder.mkdir(parents=True, exist_ok=True)
        data = (picture.image if isinstance(picture.image, bytes)
                else encode_jpeg(picture.image))
        (folder / PICTURE_FILE).write_bytes(data)
        if isinstance(picture.image, bytes):
            from io import BytesIO

            with Image.open(BytesIO(data)) as opened:
                size = list(opened.size)
        else:
            size = list(picture.image.size)
        record: dict[str, Any] = {
            "format_version": VIEW_FORMAT_VERSION,
            "seq": picture.seq, "name": picture.name, "tool": call.tool,
            "call": call.call, "call_id": call.call_id,
            "index": picture.index, "of": len(call.pictures),
            "arguments": call.arguments,
            "caption": picture.caption, "step": picture.step, "recipe": picture.recipe,
            "sections": list(note.sections) if note is not None else [],
            "mode": note.mode if note is not None else None,
            "picture": {"file": PICTURE_FILE, "size": size, "bytes": len(data)},
            "layers": {},
            "frame": None,
        }
        if note is not None and note.extra:
            record["extra"] = note.extra
        if has_frame(note) and call.atlas is not None:
            assert note is not None
            residual = None
            if note.warp is not None:
                labels, borders, frame, residual = warp_layers(call.atlas, note,
                                                                (size[0], size[1]))
            else:
                labels, borders, frame = picture_layers(call.atlas, note)
            tifffile.imwrite(folder / LABELS_FILE, labels, compression="zlib")
            Image.fromarray(np.ascontiguousarray(borders)).save(
                folder / BORDERS_FILE, format="PNG", optimize=True)
            record["layers"] = {"labels": LABELS_FILE, "borders": BORDERS_FILE}
            if residual is not None:
                tifffile.imwrite(folder / RESIDUAL_FILE,
                                 np.ascontiguousarray(np.moveaxis(residual, -1, 0)),
                                 compression="zlib")
                record["layers"]["residual"] = RESIDUAL_FILE
            record["frame"] = frame
        (folder / VIEW_FILE).write_text(json.dumps(record, indent=1, default=str) + "\n",
                                        encoding="utf-8")
        entry = {"seq": picture.seq, "name": picture.name,
                 "path": self.layout.relative(folder), "tool": call.tool, "call": call.call,
                 "sections": record["sections"], "mode": record["mode"],
                 "layers": bool(record["layers"])}
        # Left out of the index line when absent (view.json always has them).
        entry.update({key: record[key] for key in ("caption", "step", "recipe")
                      if record[key] is not None})
        return json.dumps(entry) + "\n"


class DiscardedViews(ViewStore):
    """A store that saves nothing to disk: a job that writes nothing
    (``Job.persist`` False: the CLI's dry run) or keeps the results only
    (``Job.lean``). It indexes the pictures in memory, so :meth:`lookup` and
    :meth:`latest` work within the process; the newest :data:`KEPT_PICTURES`
    keep the picture itself (``PictureRecord.image``)."""

    #: How many of the newest pictures keep their image in memory.
    KEPT_PICTURES = 24

    def __init__(self, layout: JobLayout) -> None:
        super().__init__(layout)
        self._memory: list[PictureRecord] = []

    def save(
        self, *, tool: str, pictures: list[tuple[Image.Image | bytes, PictureNote | None]],
        arguments: Any = None, call_id: str | None = None, atlas: Any = None,
        step: int | None = None, recipes: list[dict[str, Any] | None] | None = None,
        captions: list[str | None] | None = None,
    ) -> list[str]:
        if not pictures:
            return []
        step = self._step(step)
        with self._lock:
            call = max((r.call for r in self._memory), default=0) + 1
            first = max((r.seq for r in self._memory), default=0) + 1
            for item in _items(first, tool, pictures, step, recipes, captions):
                held = item.note
                self._memory.append(PictureRecord(
                    seq=item.seq, name=item.name, tool=tool, call=call, path="",
                    sections=tuple(held.sections) if held else (),
                    mode=held.mode if held else None, caption=item.caption,
                    step=item.step, recipe=item.recipe, image=item.image))
            for at in range(max(0, len(self._memory) - self.KEPT_PICTURES)):
                old = self._memory[at]
                if old.image is not None:
                    self._memory[at] = replace(old, image=None)
        return []

    def records(self) -> list[PictureRecord]:
        with self._lock:
            return list(self._memory)
