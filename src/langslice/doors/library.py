"""The script door: ``langslice.open_job(folder)`` and the verbs as methods.

A script (or a coding agent's Python) opens a job folder and calls the same
verbs the agent tools and the CLI offer, by the same names and arguments
(:mod:`langslice.doors.declarations`)::

    import langslice

    job = langslice.open_job("/data/brain1")       # the job folder or its images
    job.status()["rows"]
    reply = job.position_sections(sections=[{"id": "s01.tif", "position_mm": 5.2}])
    reply["artifacts"]                              # the pictures' files
    reply.images                                    # the pictures, as PIL images

Each method returns the tool's reply as a plain, JSON-safe dict
(:class:`Reply`): the result the agent CLI gives under ``result`` (whole,
not shortened), every status row with every field (null where absent), and
under ``artifacts`` the files of its pictures, saved in the job folder with
their layers as every door saves them, as the CLI lists them (``path``,
``kind``, ``index``, ``label``). A method returns once its pictures are on
disk; the pictures themselves, as PIL images, are on the reply's
``images`` attribute (not a key). The look-before-commit gates do not apply
(gates are tool-only) and ``look``'s ``resolution`` takes any size from
128 px to the source's own pixels. A retired verb's name raises an
``AttributeError`` naming the verb to use instead. Writes go through the job: one undo step each,
checkpointed, and picked up by an agent working on the same folder
(``Job.sync``), whose writes this handle picks up before each call.

A script makes a job with :func:`create_job` (a folder of sections and the
settings ``langslice-job FOLDER init`` takes, or a :class:`JobSpec`), the
same ingest every host uses, and gets the same handle. Both take
``image_model=``: the model ``trace_borders`` calls, a provider name, a
model profile (:func:`langslice.providers.profiles.image_model`) or a model
of the caller's own (``docs/library.md``).

An image-model verb (``trace_borders``) returns once its call has started;
the call lands in the background. :meth:`JobHandle.close` (or leaving the
``with`` block) waits for it, and a script that exits without either has
every open job closed for it at exit, while threads can still start
(:func:`langslice.job.views.at_exit`).

Nothing here loads the agent framework or a model client.
"""

from __future__ import annotations

import functools
import logging
import math
import os
import weakref
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.spec import JobSpec
from langslice.doors.jobs import Opened, open_folder

if TYPE_CHECKING:
    from langslice.providers.registry import ImageModel

logger = logging.getLogger(__name__)

#: ``job.json``'s record of the image model a library job was made with
#: (provider, model, profile, tested, the profile prompt's digest).
IMAGE_MODEL_KEY = "image_model"


class Reply(dict):  # type: ignore[type-arg]
    """A verb's reply: a plain dict of JSON values (``json.dumps(reply)``
    works), its pictures' files under ``artifacts``; the pictures
    themselves, as PIL images in the order a model receives them, on
    :attr:`images`, which is not a key."""

    images: list[Any]

    def __init__(self, body: Mapping[str, Any] | None = None,
                 images: Sequence[Any] = ()) -> None:
        super().__init__(body or {})
        self.images = list(images)


def plain(value: Any) -> Any:
    """*value* as JSON values: mappings with text keys, lists, numbers,
    text, booleans and null; a numpy value as its Python value, anything
    else as its text (as the agent CLI prints it)."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = sorted(value, key=str) if isinstance(value, (set, frozenset)) else value
        return [plain(item) for item in items]
    import numpy as np

    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    return str(value)


def library_reply(verb: str, reply: Any, saved: Sequence[Any]) -> Reply:
    """A tool's reply for a script: the pictures out of it (their files
    under ``artifacts``, :func:`langslice.job.views.artifacts`; the images
    on :attr:`Reply.images`; their lines of text under ``media_texts``), the
    status rows uniform (:func:`langslice.core.status.with_uniform_rows`),
    everything else JSON values (:func:`plain`); a picture that could not be
    saved is a line under ``warnings``."""
    from langslice.core.status import with_uniform_rows
    from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
    from langslice.job.views import artifacts

    body = dict(reply) if isinstance(reply, Mapping) else {"result": reply}
    media = body.pop(TOOL_MEDIA_PARTS_KEY, None)
    media = media if isinstance(media, list) else []
    texts = [item for item in media if isinstance(item, str)]
    if texts:
        body["media_texts"] = texts
    out = plain(with_uniform_rows(body))
    out["artifacts"], unsaved = artifacts(saved, verb)
    if unsaved:
        out["warnings"] = unsaved
    return Reply(out, [item for item in media if not isinstance(item, str)])


#: The jobs the library holds open, closed at interpreter exit (:func:`_close_open`).
_OPEN: dict[int, weakref.ref[Opened]] = {}
_EXIT_HOOKED = False


def _track(opened: Opened) -> None:
    global _EXIT_HOOKED
    if not _EXIT_HOOKED:
        from langslice.job.views import at_exit

        at_exit(_close_open)
        _EXIT_HOOKED = True
    key = id(opened)
    _OPEN[key] = weakref.ref(opened, lambda _ref: _OPEN.pop(key, None))


def _close_open() -> None:
    """Close every job a script left open: its image-model calls land and
    its pictures are written before the interpreter stops taking work."""
    for ref in list(_OPEN.values()):
        opened = ref()
        if opened is None:
            continue
        try:
            opened.close()
        except Exception:  # an exiting script has no one to raise to
            logger.warning("Could not close the job %s at exit", opened.job.folder,
                           exc_info=True)
    _OPEN.clear()


class JobHandle:
    """One open job folder; every verb its settings have is a method."""

    def __init__(self, opened: Opened, imported: dict[str, Any] | None = None) -> None:
        self._opened = opened
        self._tools = {tool.__name__: self._method(tool) for tool in opened.tools().tools}
        self._imported = imported
        _track(opened)

    def _method(self, tool: Callable[..., Any]) -> Callable[..., Reply]:
        """The verb *tool* as a method: the tool's call, then its pictures
        written, then its reply for a script (:func:`library_reply`)."""
        from langslice.job.views import captured

        job = self._opened.job
        name = tool.__name__

        @functools.wraps(tool)
        def method(*args: Any, **kwargs: Any) -> Reply:
            with captured() as saved:
                reply = tool(*args, **kwargs)
            job.views.flush()  # the pictures are on disk before the method returns
            return library_reply(name, reply, saved)

        return method

    # --- the verbs ------------------------------------------------------------------

    @property
    def verbs(self) -> list[str]:
        """The verbs this job has (its tasks decide), in registry order; a
        hidden verb (``Verb.hidden``) is callable by name but not listed."""
        return self._opened.listed_verbs()

    def __getattr__(self, name: str) -> Callable[..., Reply]:
        tools = self.__dict__.get("_tools") or {}
        if name in tools:
            return tools[name]
        from langslice.ops.registry import VERBS, retired_payload

        if name in VERBS:
            raise AttributeError(f"This job's settings have no {name!r} (tasks "
                                 f"{self._opened.spec.tasks}); see .verbs")
        retired = retired_payload(name)
        if retired is not None:  # a retired verb says what replaced it
            raise AttributeError(retired["message"])
        raise AttributeError(name)

    def __dir__(self) -> list[str]:
        return sorted({*super().__dir__(), *self.verbs})

    # --- the job ----------------------------------------------------------------------

    @property
    def folder(self) -> str:
        """The job folder."""
        return str(self._opened.job.folder)

    @property
    def image_model(self) -> Any:
        """The image model ``trace_borders`` calls when one was handed in
        (:class:`langslice.providers.registry.ImageModel`), else None."""
        return self._opened.image_model

    @property
    def imported(self) -> dict[str, Any] | None:
        """What :func:`create_job`'s *registration* file placed (the import
        report: format, per section how it matched and where it was placed,
        ``unmatched`` entries, ``missing`` sections, ``refused`` ones with
        the reason, the pixel size, ``warnings``); None for a job made
        without one, or opened with :func:`open_job`."""
        return self._imported

    @property
    def job(self) -> Any:
        """The job itself (:class:`langslice.job.job.Job`: state, undo, gates)."""
        return self._opened.job

    @property
    def state(self) -> Any:
        """The stack as it stands (:class:`langslice.core.state.StackState`);
        read it, write through the verbs."""
        return self._opened.job.state

    @property
    def workspace(self) -> Any:
        """The atlas and the section files (:class:`langslice.core.workspace.Workspace`)."""
        return self._opened.ctx

    def close(self) -> None:
        """Finish the job's background work: its image-model calls land and
        its pictures are written. The job stays usable; a script that exits
        without closing has this done for it."""
        self._opened.close()

    def __enter__(self) -> JobHandle:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<langslice-job {self.folder} ({len(self.state.slices)} sections)>"


def as_image_model(value: Any) -> ImageModel | None:
    """*value* as the image model the tools call: None stays None; a provider
    name, an ``ImageModel``, an object with a ``call`` or a function goes
    through :func:`langslice.providers.profiles.image_model`."""
    if value is None:
        return None
    from langslice.providers.profiles import image_model

    return image_model(value)


def profile_record(model: ImageModel) -> dict[str, Any]:
    """What ``job.json`` records of the image model a job was made with."""
    import hashlib

    record: dict[str, Any] = {"provider": model.provider, "model": model.model,
                              "profile": model.profile, "tested": bool(model.tested)}
    if model.prompt is not None:
        record["prompt_sha256"] = hashlib.sha256(model.prompt.encode("utf-8")).hexdigest()
    if model.photograph_first is not None:
        record["photograph_first"] = model.photograph_first
    return record


def _check_model_fits(spec: JobSpec, model: ImageModel | None) -> None:
    if model is not None and not spec.nonlinear.uses_image_model:
        raise ValueError("This job runs without an image model (nonlinear provider 'none'); "
                         "create it with image_model= to trace borders")


def open_job(
    folder: str | os.PathLike[str], *,
    image_model: Any = None,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> JobHandle:
    """Open the job in *folder* (the job folder, or the image folder beside
    it). Create one with :func:`create_job` or ``langslice-job <images> init``.

    *image_model* is the model ``trace_borders`` calls (a provider name, a
    profile, a model of your own: :func:`langslice.providers.profiles.image_model`);
    None: the job's provider as its settings name it. A job made with an
    untested profile (``job.json`` ``image_model``) opened without one offers
    no ``trace_borders``, so a reopen never traces with another prompt than
    the one the job was made with. *atlas_loader* replaces BrainGlobe's
    (tests, offline hosts)."""
    from langslice.job.layout import read_job_file

    model = as_image_model(image_model)
    opened = open_folder(folder, atlas_loader=atlas_loader, emit=emit, image_model=model)
    try:
        _check_model_fits(opened.spec, model)
        held = (read_job_file(opened.job.layout) or {}).get(IMAGE_MODEL_KEY) or {}
        if model is None and held and not held.get("tested", True):
            opened.traces_off = True
            opened.ctx.progress(f"[job] made with the untested image-model profile "
                                f"{held.get('profile')!r}: pass image_model= to trace borders")
        return JobHandle(opened)
    except BaseException:
        opened.close()
        raise


def _transform(name: str, value: Any) -> dict[str, Any]:
    """A supplied in-plane transform: LangSlice's normalized six numbers
    ``[a, b, tx, c, d, ty]`` (``langslice.core.affine``), or a transform
    dictionary kept as given."""
    if isinstance(value, Mapping):
        return dict(value)
    try:
        params = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"transforms[{name!r}] must be six numbers [a, b, tx, c, d, ty] "
                         "or a transform dictionary") from exc
    if len(params) != 6 or not all(math.isfinite(item) for item in params):
        raise ValueError(f"transforms[{name!r}] must be six finite numbers [a, b, tx, c, d, ty]")
    return {"kind": "interactive", "params": params,
            "mirrored": params[0] * params[4] - params[1] * params[3] < 0}


def _angles(value: Mapping[str, Any]) -> dict[str, Any]:
    """Supplied cutting angles in either form (``inputs.angles``), checked
    (:func:`langslice.core.spec.supplied_angles`) and as plain floats."""
    from langslice.core.spec import supplied_angles

    stack, sections = supplied_angles(dict(value))
    if stack is not None:
        return {"pitch": stack[0], "yaw": stack[1]}
    return {name: {"pitch": pitch, "yaw": yaw} for name, (pitch, yaw) in sections.items()}


def pipeline_tasks(images: str | os.PathLike[str], transforms: Mapping[str, Any]) -> list[str]:
    """The tasks a scripted registration needs: ``nonlinear``, plus
    ``transform`` (the automatic linear alignment, ``elastix_affine``) unless
    every section in *images* has a supplied transform."""
    from langslice.core.discovery import discover_slices

    names = {os.path.basename(path) for path in discover_slices(os.fspath(images))}
    return ["nonlinear"] if names and names <= set(transforms) else ["transform", "nonlinear"]


def create_job(
    images: str | os.PathLike[str] | JobSpec,
    *,
    atlas: str = "allen_mouse_25um",
    plane: str = "coronal",
    tasks: Sequence[str] | None = None,
    image_model: Any = None,
    job_dir: str | os.PathLike[str] | None = None,
    output: str = "full",
    positions: Mapping[str, float] | None = None,
    transforms: Mapping[str, Any] | None = None,
    angles: Mapping[str, Any] | None = None,
    orientation: Mapping[str, Mapping[str, Any]] | None = None,
    pixel_size_um: float | None = None,
    inputs: Mapping[str, Any] | None = None,
    registration: str | os.PathLike[str] | None = None,
    fresh: bool = False,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
    **settings: Any,
) -> JobHandle:
    """Create (or continue) the job of a folder of section images; the same
    handle as :func:`open_job`.

    *images* is the folder of sections (or a :class:`JobSpec`, used as it
    is: then only *image_model*, *atlas_loader* and *emit* apply). The job
    folder is *job_dir*, else ``<images>/langslice``. *output* "full" keeps
    every file a job writes, "lean" the results only (no pictures, no undo
    history on disk, no logs, no reference card: ``JobSpec.output_level``).

    The registration you supply, by section filename: *positions* (atlas
    millimetres from the anterior edge of the volume), *transforms* (six
    numbers ``[a, b, tx, c, d, ty]``, LangSlice's normalized in-plane affine
    on the oriented section, or a transform dictionary), *orientation*
    (``{"flip": bool, "rotation_deg": 0|90|180|270}``), the cutting
    *angles* (``{"pitch": deg, "yaw": deg}`` for the whole stack, or
    ``{filename: {"pitch": deg, "yaw": deg}}`` for each section its own; a
    section the per-section form does not name is flat:
    :func:`langslice.core.spec.supplied_angles`) and *pixel_size_um*; *inputs*
    takes any other key of ``JobSpec.inputs``. *tasks* None: ``nonlinear``,
    plus ``transform`` (``elastix_affine``) unless every section has a supplied
    transform.

    Or *registration*: a linear registration made elsewhere (a QuickNII or
    VisuAlign JSON/XML, a DeepSlice CSV/JSON/XML, a LangSlice
    ``registration.json``), imported as every section's position, cutting
    angles, orientation and transform (``doors.jobs.with_registration``; not
    with *positions*, *transforms*, *angles* or *orientation*; VisuAlign
    markers are not imported). *tasks* None is then ``["nonlinear"]``: the
    imported placement kept as it is. The handle's :attr:`JobHandle.imported`
    says what was placed; the warnings go to *emit* as well.

    *image_model* as in :func:`open_job`; None makes a job
    without one (nonlinear provider ``none``). *fresh* starts over instead
    of continuing the folder's job. *settings* are further
    :class:`JobSpec` fields (``preprocess``, ``interval_um`` lives in
    ``position``...).
    """
    from langslice.doors.api.setup import load_credentials
    from langslice.doors.jobs import REGISTRATION_EXCLUDES, create, with_registration
    from langslice.job.layout import write_job_file

    load_credentials()  # the image model's keys, as open_job loads them
    model = as_image_model(image_model)
    if isinstance(images, JobSpec):
        given = {name: value for name, value in {
            "positions": positions, "transforms": transforms, "angles": angles,
            "orientation": orientation, "pixel_size_um": pixel_size_um, "inputs": inputs,
            "tasks": tasks, "job_dir": job_dir,
            "registration": registration}.items() if value is not None}
        if given or settings or output != "full" or fresh:
            raise ValueError("create_job takes a JobSpec as it is: set "
                             f"{sorted({*given, *settings})} on the spec instead")
        spec = images
    else:
        folder = Path(os.path.abspath(os.path.expanduser(os.fspath(images))))
        supplied: dict[str, Any] = dict(inputs or {})
        if positions:
            supplied["positions"] = {str(name): float(mm) for name, mm in positions.items()}
        if transforms:
            supplied["transforms"] = {str(name): _transform(str(name), value)
                                      for name, value in transforms.items()}
        if angles:
            supplied["angles"] = _angles(angles)
        if orientation:
            supplied["orientation"] = {str(name): dict(value)
                                       for name, value in orientation.items()}
        if pixel_size_um is not None:
            supplied["pixel_size_um"] = float(pixel_size_um)
        if registration is not None:
            clashes = sorted(key for key in REGISTRATION_EXCLUDES if supplied.get(key))
            if clashes:
                raise ValueError(f"registration= supplies every section's placement; "
                                 f"it cannot be combined with {clashes}")
        nonlinear = dict(settings.pop("nonlinear", None) or {})
        if model is not None:
            nonlinear.update(provider=model.provider, image_model=model.model)
        else:
            nonlinear.setdefault("provider", "none")
        unknown = sorted(set(settings) - set(JobSpec.__dataclass_fields__))
        if unknown:
            raise ValueError(f"Unknown job setting(s) {unknown}")
        spec = JobSpec.from_dict({
            "atlas": atlas, "plane": plane, "debrief": False, **settings,
            "image_folder": str(folder),
            "tasks": list(tasks) if tasks is not None else ["nonlinear"]
            if registration is not None else pipeline_tasks(
                folder, supplied.get("transforms") or {}),
            "nonlinear": nonlinear, "inputs": supplied,
            "job_dir": None if job_dir is None else os.path.abspath(
                os.path.expanduser(os.fspath(job_dir))),
            "output_level": output, "resume": not fresh,
        })
    _check_model_fits(spec, model)
    imported: dict[str, Any] | None = None
    if registration is not None:
        spec, imported = with_registration(spec, registration, atlas_loader=atlas_loader,
                                           emit=emit)
    opened = create(spec, atlas_loader=atlas_loader, emit=emit, image_model=model)
    try:
        if model is not None:
            write_job_file(opened.job.layout, **{IMAGE_MODEL_KEY: profile_record(model)})
        return JobHandle(opened, imported)
    except BaseException:
        opened.close()
        raise
