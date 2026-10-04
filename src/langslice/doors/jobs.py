"""Opening a job folder without an agent: the CLI's and the library's way in.

The agent driver opens its job through ``agent.engine.build_context``,
which brings the agent framework along. A coding agent's CLI call and a
script open the same job folder here instead, with nothing but the core and
the job layer loaded: :func:`open_folder` reads an existing job as it stands
(``job.job.Job.load``: nothing rewritten, so a call beside a running
agent changes nothing until its first write) and :func:`create` makes one
from an image folder through the same ingest every host uses
(``Job.open``). Both write the job folder's reference card
(:mod:`langslice.doors.card`). The tools on top are the same toolbox the
agent and MCP doors use (:func:`tools`), with the look-before-commit gates
off (gates are tool-only) and the picture size the caller's.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from langslice.core.spec import JobSpec
from langslice.core.workspace import Workspace, log_progress
from langslice.doors.card import write_card
from langslice.job.job import Job
from langslice.job.layout import (
    JobLayout,
    held_image_folder,
    job_folder_for,
    locate_job_folder,
    read_job_file,
)

if TYPE_CHECKING:
    from langslice.doors.tools.toolbox import ToolBox
    from langslice.providers.registry import ImageModel

#: The largest picture a CLI or script call may ask for (``view.resolution``):
#: no model's limit applies, only the source's own pixels (nothing is
#: upsampled past them) and :data:`langslice.core.sizes.MIN_RESOLUTION`.
OPEN_MAX_VIEW_EDGE = 1 << 15


class NoJob(LookupError):
    """The folder holds no job (and is no image folder with one beside it)."""


@dataclass(kw_only=True)
class JobContext(Workspace):
    """A :class:`Workspace` and where its job's files live (the job folder
    and the results path the job is opened at). The agent driver's
    ``EngineContext`` adds the model."""

    job_folder: str
    results_path: str

    @property
    def layout(self) -> JobLayout:
        return JobLayout(Path(self.job_folder), Path(self.image_folder))

    @property
    def checkpoint_path(self) -> str:
        """The job folder's state checkpoint."""
        return str(self.layout.state_file)


def context(
    spec: JobSpec, job_folder: str | os.PathLike[str], *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> JobContext:
    """The workspace of *spec*'s images with its job folder *job_folder*."""
    folder = Path(job_folder)
    kwargs: dict[str, Any] = {}
    if atlas_loader is not None:
        kwargs["atlas_loader"] = atlas_loader
    return JobContext(
        spec=spec, image_folder=os.path.abspath(spec.image_folder), job_folder=str(folder),
        results_path=spec.out or str(JobLayout(folder).results_file),
        emit=emit or log_progress, **kwargs,
    )


def find(path: str | os.PathLike[str]) -> Path:
    """The job folder *path* names: *path* itself when it holds ``job.json``,
    else the job folder beside the images (``<path>/langslice``). ``NoJob``
    otherwise."""
    given = Path(os.path.abspath(os.path.expanduser(os.fspath(path))))
    for folder in (given, job_folder_for(given)):
        if (folder / "job.json").is_file() and (folder / "state.json").is_file():
            return folder
    raise NoJob(f"No LangSlice job in {given} (neither a job folder nor an image folder "
                "with one beside it)")


def read_spec(job_folder: Path) -> JobSpec:
    """The job's spec as its ``job.json`` holds it, opened as a resume.

    The image folder is resolved from the job folder
    (:func:`~langslice.job.layout.held_image_folder`: the default job folder's
    images are its parent, wherever it moved). ``NoJob``, with how to
    reattach the job, when its images are not where it says."""
    record = read_job_file(JobLayout(job_folder)) or {}
    data = dict(record.get("spec") or {})
    if not data:
        raise NoJob(f"{job_folder / 'job.json'} holds no job settings")
    images = held_image_folder(job_folder, record)
    if images is None or not images.is_dir():
        raise NoJob(
            f"The images of the job in {job_folder} are not in {images} any more. If they "
            "moved, reattach the job from their new folder: langslice job NEW_IMAGE_FOLDER "
            f"init --job-dir {job_folder} (or move the job folder back beside them, as "
            "<images>/langslice).")
    data["image_folder"] = str(images)
    spec = JobSpec.from_dict(data)
    spec.resume = True
    return spec


@dataclass
class Opened:
    """One job open for a CLI call or a script: the job, its workspace."""

    job: Job
    ctx: JobContext
    #: The image model ``trace_borders`` calls (None: the spec's provider,
    #: resolved by the tool door; a ``custom`` provider without one offers no
    #: ``trace_borders``).
    image_model: ImageModel | None = None
    #: No ``trace_borders`` whatever the spec says (a library job made with
    #: an untested profile, reopened without it).
    traces_off: bool = False
    _box: ToolBox | None = field(default=None, repr=False)

    @property
    def spec(self) -> JobSpec:
        return self.job.spec

    def tools(self) -> ToolBox:
        """The verbs this job's spec has, as the tool door builds them (one
        toolbox per open job), with the gates off, the pictures sized by
        the caller (``view.resolution`` up to :data:`OPEN_MAX_VIEW_EDGE`) and
        the scripting verbs (``export_maps``) added; ``trace_borders`` calls
        :attr:`image_model` when one was handed in."""
        if self._box is None:
            from langslice.core.provider_names import CUSTOM_PROVIDER, canonical_provider
            from langslice.core.sizes import AUTO_RESOLUTION
            from langslice.doors.tools.toolbox import build_tools

            custom = canonical_provider(self.spec.nonlinear.provider) == CUSTOM_PROVIDER
            self._box = build_tools(self.job.state, self.ctx, self.spec,  # type: ignore[arg-type]
                                    job=self.job, max_view_edge=OPEN_MAX_VIEW_EDGE,
                                    gates=False, level=AUTO_RESOLUTION, scripting=True,
                                    image_model=self.image_model,
                                    image_model_connected=not self.traces_off and (
                                        self.image_model is not None or not custom))
        return self._box

    def close(self) -> None:
        """Finish the job's background writes (pictures, image corrections)."""
        if self.job.image_jobs:
            self.job.settle_image_corrections()
        self.job.close()


def open_folder(
    path: str | os.PathLike[str], *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
    persist: bool = True,
    image_model: ImageModel | None = None,
) -> Opened:
    """Open the job *path* names (:func:`find`) as it stands on disk.

    *persist* False opens it for a dry run: nothing is written. Otherwise the
    reference card is brought up to date (not in a lean job). *image_model*
    is the model ``trace_borders`` calls (None: the spec's provider).
    """
    from langslice.doors.api.setup import load_credentials

    folder = find(path)
    spec = read_spec(folder)
    load_credentials()  # the image model's keys, as every door loads them
    ctx = context(spec, folder, atlas_loader=atlas_loader, emit=emit)
    job = Job.load(spec, ctx, folder=folder, results_path=ctx.results_path, persist=persist)
    if persist and not job.lean:
        write_card(job.layout)
    return Opened(job, ctx, image_model)


def create(
    spec: JobSpec, *,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
    image_model: ImageModel | None = None,
) -> Opened:
    """A job for *spec*'s image folder, as every host makes one
    (``Job.open``: the job folder beside the images or ``spec.job_dir``,
    ingest, host inputs, first checkpoint; a resume when ``spec.resume``),
    with its reference card (not in a lean job). *image_model* as in
    :func:`open_folder`."""
    folder, _fallback = locate_job_folder(spec.image_folder, spec.job_dir,
                                          emit=emit or log_progress)
    ctx = context(spec, folder, atlas_loader=atlas_loader, emit=emit)
    job = Job.open(spec, ctx, folder=folder, results_path=ctx.results_path)
    if not job.lean:
        write_card(job.layout)
    return Opened(job, ctx, image_model)
