"""The scripted nonlinear registration: the image model's border trace and the
deformable fit, run on a job without the agent.

Two entry points, one sequence of the ordinary verbs on the ordinary job
folder (no second pipeline):

- :func:`register_job`, the pipeline path: a job made with
  :func:`langslice.create_job` over a folder of sections, every section run
  through the sequence below, then ``submit`` (which writes the maps and
  the exports).
- :func:`register_section`, the convenience call for one section: a job in
  a folder of its own (the section image linked, copied or, for an array,
  written there), then :func:`register_job`.

The sequence, per section: ``elastix_affine`` (the automatic linear
alignment the agent's tool runs, here run with no agent) when the section
has no in-plane transform yet; ``trace_borders`` with the job's image model
(when it has one: the image model's trace, then ANTs' fit of it, applied in
the background and waited for), else ``ants_syn`` on the section's
preprocessed channel; then ``submit``, a section whose deformation failed
named in its ``left_linear`` with the reason. Each step is the verb every
door calls, so the job folder holds what an agent's run would. Both
deformable steps need ANTs (antspyx).
"""

from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langslice.doors.library import JobHandle, create_job

#: The ``ants_syn`` arguments ``register_job`` fits with when the job has no
#: image model (``fit=`` replaces any of them).
STAIN_FIT: dict[str, Any] = {"stiffness": "medium", "atlas_image": "template"}
#: Most sections one ``ants_syn`` call takes.
FIT_BATCH = 4
#: The section image an array is written as (a lossless TIFF, its dtype kept).
ARRAY_NAME = "section.tif"


class RegistrationError(RuntimeError):
    """A section could not be registered; ``problems`` says why, per section."""

    def __init__(self, message: str, problems: Mapping[str, str]) -> None:
        super().__init__(message)
        self.problems = dict(problems)


@dataclass
class SectionOutput:
    """One section's results in the job folder (paths absolute)."""

    id: str
    #: ``sections/<stem>/``.
    folder: Path
    #: Atlas micrometres per pixel, float32 (``coords.tif``).
    coords: Path
    #: Atlas ids per pixel, uint32 (``labels.tif``).
    labels: Path
    #: The maps' record (``maps.json``).
    maps: Path
    #: The deformation as a displacement field (``residual.tif``), when one is applied.
    residual: Path | None
    #: Every map file by kind (coords, labels, labels_fiji, labels_csv, tissue,
    #: residual, maps).
    files: dict[str, Path]
    #: The trace record (``image_correction`` in the state), None without one.
    trace: dict[str, Any] | None
    #: The trace was made with an untested model profile.
    untested: bool
    #: Why the section is not fully registered, None when it is.
    problem: str | None = None
    #: The maps as arrays, when asked for (:meth:`read`): ``coords`` (rows,
    #: cols, 3) float32, ``labels`` (rows, cols) uint32, ``residual`` (rows,
    #: cols, 2) float32 when present.
    arrays: dict[str, Any] | None = None

    def read(self) -> dict[str, Any]:
        """Load the maps (see :attr:`arrays`), keep and return them."""
        import numpy as np
        import tifffile

        loaded: dict[str, Any] = {"coords": _channels_last(tifffile.imread(self.coords)),
                                  "labels": np.asarray(tifffile.imread(self.labels))}
        if self.residual is not None and self.residual.exists():
            loaded["residual"] = _channels_last(tifffile.imread(self.residual))
        self.arrays = loaded
        return loaded


def _channels_last(stack: Any) -> Any:
    import numpy as np

    array = np.asarray(stack, dtype=np.float32)
    return np.moveaxis(array, 0, -1) if array.ndim == 3 else array


@dataclass
class RegistrationResult:
    """What :func:`register_job` / :func:`register_section` produced."""

    #: The job folder.
    job_folder: Path
    #: ``registration.json``: every section's parameters and mapping.
    registration: Path
    #: ``exports/quicknii.json`` and ``exports/visualign.json`` by kind, when written.
    exports: dict[str, Path]
    #: Per section, in stack order.
    sections: list[SectionOutput] = field(default_factory=list)
    #: Whether the job was submitted (every section registered).
    submitted: bool = False

    @property
    def problems(self) -> dict[str, str]:
        """Section id -> why it is not fully registered."""
        return {section.id: section.problem for section in self.sections
                if section.problem is not None}

    @property
    def ok(self) -> bool:
        return self.submitted and not self.problems

    def section(self, section_id: str) -> SectionOutput:
        for section in self.sections:
            if section.id == section_id:
                return section
        raise KeyError(section_id)


def _message(reply: Mapping[str, Any]) -> str:
    return str(reply.get("message") or reply.get("error") or reply.get("status") or reply)


def _rows(reply: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("id")): row for row in reply.get("results") or []
            if isinstance(row, Mapping)}


def register_job(
    job: JobHandle,
    *,
    sections: Sequence[str] | None = None,
    fit: Mapping[str, Any] | None = None,
    full_resolution: bool = False,
    arrays: bool = False,
    summary: str = "Scripted registration (langslice.register_job).",
) -> RegistrationResult:
    """Run the scripted registration on *job*'s sections (*sections*: their
    filenames; None: every section) and submit.

    Per section: ``elastix_affine`` when it has no in-plane transform;
    ``trace_borders`` when the job has an image model (its fit lands in the
    background; this waits for it), else ``ants_syn`` with *fit* (default
    :data:`STAIN_FIT`; any of its arguments: ``stiffness``, ``atlas_image``,
    ``restrict_to``). Then ``submit``, which writes every section's maps and
    the exports, a section whose deformation failed named in its
    ``left_linear`` with the reason; *full_resolution* rewrites the maps on
    the image files' own pixels instead of their working copies. A section
    that fails a step is reported in the result's ``problems``; one with no
    position or transform keeps the job from being submitted (the maps of
    the placed sections are still written, ``export_maps``). *arrays* loads
    each section's maps into ``SectionOutput.arrays``.
    """
    state = job.state
    wanted = [state.resolve(name) for name in sections] if sections else state.in_order()
    unknown = [str(name) for name, record in zip(sections or [], wanted, strict=False)
               if record is None]
    if unknown:
        raise ValueError(f"No such section(s) in {job.folder}: {unknown}")
    ids = [record.id for record in wanted if record is not None]
    problems: dict[str, str] = {}

    # 1. The automatic linear alignment, where no transform is supplied.
    unplaced = [name for name in ids if state.by_id(name).position_mm is None]  # type: ignore[union-attr]
    for name in unplaced:
        problems[name] = "no position: supply one (positions=) before registering"
    to_align = [name for name in ids if name not in problems
                and job.state.by_id(name).transform is None]  # type: ignore[union-attr]
    if to_align:
        if "elastix_affine" not in job.verbs:
            raise ValueError(f"Sections {to_align} have no in-plane transform and this job "
                             "has no elastix_affine (task 'transform'); supply transforms= "
                             "or create the job with the 'transform' task")
        rows = _rows(job.elastix_affine(sections=to_align))
        for name in to_align:
            if job.state.by_id(name).transform is None:  # type: ignore[union-attr]
                problems[name] = "elastix_affine: " + _message(rows.get(name) or {})
    placed = [name for name in ids if name not in problems]

    # 2. The deformation: the image model's trace and its fit, in the
    # background for every section, or ANTs on the stain.
    failed: dict[str, str] = {}
    if "trace_borders" in job.verbs:
        for name in placed:
            reply = job.trace_borders(section=name)
            if reply.get("status") not in ("started", "running", "ok"):
                failed[name] = "trace_borders: " + _message(reply)
        job.job.settle_image_corrections()
        for work in job.job.background.wait_all():
            for name in work.sections:
                if work.status != "done" and name in placed:
                    failed.setdefault(name, "trace_borders: " + work.notice)
    else:
        settings = {**STAIN_FIT, **dict(fit or {})}
        for start in range(0, len(placed), FIT_BATCH):
            batch = placed[start:start + FIT_BATCH]
            reply = job.ants_syn(sections=batch, **settings)
            rows = _rows(reply)
            for name in batch:
                if (rows.get(name) or {}).get("status") != "ok":
                    failed[name] = "ants_syn: " + _message(rows.get(name) or reply)
    for name in placed:
        held = job.state.by_id(name).deformation or {}  # type: ignore[union-attr]
        if not held and name not in failed:
            failed[name] = "no deformation was applied"
    problems.update(failed)

    # 3. Submit (maps and exports), a failed deformation left linear with its
    # reason; with a section unplaced, the maps of what is placed.
    submitted = False
    unplaced_or_unaligned = [name for name in ids if name not in placed]
    if not unplaced_or_unaligned and len(ids) == len(job.state.slices):
        left = [{"id": name, "reason": reason} for name, reason in failed.items()]
        reply = job.submit(summary=summary, notes=[], interval_breaks=[],
                           **({"left_linear": left} if left else {}))
        submitted = reply.get("status") == "ok"
        if not submitted:
            for name in ids:
                problems.setdefault(name, "submit: " + _message(reply))
    if not submitted or full_resolution:
        job.export_maps(slices=ids, full_resolution=full_resolution)
    job.close()
    return _result(job, ids, problems, submitted, arrays)


def _result(job: JobHandle, ids: Sequence[str], problems: Mapping[str, str], submitted: bool,
            arrays: bool) -> RegistrationResult:
    from langslice.job.formats import QUICKNII_FILE, SECTION_FILES, VISUALIGN_FILE

    layout = job.job.layout
    outputs: list[SectionOutput] = []
    for name in ids:
        record = job.state.by_id(name)
        assert record is not None
        folder = layout.section_dir(name)
        files = {kind: folder / file for kind, file in SECTION_FILES.items()
                 if (folder / file).exists()}
        trace = dict(record.image_correction) if record.image_correction else None
        problem = problems.get(name)
        if problem is None and "coords" not in files:
            problem = "no maps were written (see registration.json, its 'problem')"
        output = SectionOutput(
            id=name, folder=folder, coords=folder / SECTION_FILES["coords"],
            labels=folder / SECTION_FILES["labels"], maps=folder / SECTION_FILES["maps"],
            residual=files.get("residual"), files=files, trace=trace,
            untested=bool(trace and trace.get("untested")), problem=problem)
        if arrays and "coords" in files:
            output.read()
        outputs.append(output)
    exports = {kind: layout.exports_dir / file for kind, file in (
        ("quicknii", QUICKNII_FILE), ("visualign", VISUALIGN_FILE))
        if (layout.exports_dir / file).exists()}
    return RegistrationResult(job_folder=layout.folder,
                              registration=layout.folder / "registration.json",
                              exports=exports, sections=outputs, submitted=submitted)


def _section_name(image: Any, name: str | None) -> str:
    """The section's filename (its id) in its folder: *name*, else the
    image file's own name, else :data:`ARRAY_NAME` for an array."""
    if name:
        return name
    if isinstance(image, (str, os.PathLike)):
        return Path(os.fspath(image)).name
    return ARRAY_NAME


def _place_image(image: Any, folder: Path, filename: str) -> None:
    """Put the section image in *folder* as *filename*. A file already there
    under that name is used when it is the same image, and refused (never
    replaced) otherwise."""
    import numpy as np

    target = folder / filename
    if isinstance(image, (str, os.PathLike)):
        source = Path(os.path.expanduser(os.fspath(image))).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"No section image at {source}")
        if target.exists():
            if os.path.samefile(target, source):
                return
            raise ValueError(f"{folder} already holds a different {filename}; give "
                             "register_section another folder or name")
        try:
            os.link(source, target)  # no copy of a whole-slide scan where possible
        except OSError:
            shutil.copy2(source, target)
        return
    array = np.asarray(image)
    if array.ndim not in (2, 3) or (array.ndim == 3 and array.shape[-1] not in (1, 3, 4)):
        raise ValueError("A section array must be (rows, cols) or (rows, cols, 1|3|4); "
                         f"got shape {array.shape}")
    import tifffile

    if Path(filename).suffix.lower() not in (".tif", ".tiff"):
        raise ValueError(f"An array is written as a TIFF; name {filename!r} must end in .tif")
    if target.exists():
        try:
            same = np.array_equal(tifffile.imread(target), array)
        except Exception:
            same = False
        if same:
            return
        raise ValueError(f"{folder} already holds a different {filename}; give "
                         "register_section another folder or name")
    tifffile.imwrite(target, array, photometric="rgb" if array.ndim == 3
                     and array.shape[-1] in (3, 4) else "minisblack")


def register_section(
    image: Any,
    *,
    position_mm: float,
    image_model: Any = None,
    atlas: str = "allen_mouse_25um",
    plane: str = "coronal",
    pitch_deg: float | None = None,
    yaw_deg: float | None = None,
    transform: Sequence[float] | Mapping[str, Any] | None = None,
    flip: bool | None = None,
    rotation_deg: int | None = None,
    pixel_size_um: float | None = None,
    folder: str | os.PathLike[str] | None = None,
    job_dir: str | os.PathLike[str] | None = None,
    output: str = "full",
    name: str | None = None,
    fit: Mapping[str, Any] | None = None,
    full_resolution: bool = False,
    arrays: bool = False,
    atlas_loader: Callable[[str], Any] | None = None,
    emit: Callable[[str], None] | None = None,
) -> RegistrationResult:
    """Register one section with the image model, without the agent.

    *image* is the section: a file path (linked, else copied, into
    *folder*) or an array ``(rows, cols)`` / ``(rows, cols, 3)`` (written into
    *folder* as the lossless TIFF *name*, default ``section.tif``, its dtype
    kept). Its placement: the *atlas* and *plane*, *position_mm* (atlas
    millimetres from the anterior edge of the volume), the cutting angles
    *pitch_deg* / *yaw_deg*, the in-plane *transform* (LangSlice's
    normalized six numbers ``[a, b, tx, c, d, ty]`` on the oriented section,
    or a transform dictionary), its orientation (*flip* left-right,
    *rotation_deg* counter-clockwise quarter turns, applied before the
    transform) and *pixel_size_um* (None: read from the file, else assumed).

    With a *transform*: ``trace_borders`` (its fit waited for), ``submit``.
    With a position only: first ``elastix_affine``, the automatic linear
    alignment, then the same. *image_model* is the model or profile
    (:func:`langslice.image_model`; None: ``ants_syn`` on the stain, with
    *fit*, no trace).

    *folder* is the section's own folder (default: a new temporary folder;
    it must hold no other section images); the job folder is *job_dir*, else
    ``<folder>/langslice``. *output* "full" or "lean"
    (:func:`langslice.create_job`). The folder is yours to delete when
    done (``shutil.rmtree(result.job_folder.parent)`` for the default
    layout). Raises :class:`RegistrationError` when the section could not
    be registered (its job folder is kept, for a look).
    """
    from langslice.core.discovery import discover_slices

    place = Path(tempfile.mkdtemp(prefix="langslice-section-")) if folder is None else \
        Path(os.path.abspath(os.path.expanduser(os.fspath(folder))))
    place.mkdir(parents=True, exist_ok=True)
    filename = _section_name(image, name)
    others = [os.path.basename(path) for path in discover_slices(str(place))
              if os.path.basename(path) != filename]
    if others:  # checked before anything is written into the folder
        raise ValueError(f"{place} holds other section images {others[:5]}; give "
                         "register_section a folder of its own (or use create_job and "
                         "register_job for a folder of sections)")
    _place_image(image, place, filename)
    orientation: dict[str, Any] = {}
    if flip is not None:
        orientation["flip"] = bool(flip)
    if rotation_deg is not None:
        orientation["rotation_deg"] = int(rotation_deg)
    angles: dict[str, Any] | None = None
    if pitch_deg is not None or yaw_deg is not None:
        # The section's own plane (inputs.angles' per-section form).
        angles = {filename: {"pitch": float(pitch_deg or 0.0), "yaw": float(yaw_deg or 0.0)}}
    job = create_job(
        place, atlas=atlas, plane=plane, image_model=image_model, job_dir=job_dir,
        output=output, positions={filename: float(position_mm)},
        transforms=None if transform is None else {filename: transform},
        angles=angles, orientation={filename: orientation} if orientation else None,
        pixel_size_um=pixel_size_um, fresh=True, atlas_loader=atlas_loader, emit=emit)
    result = register_job(job, fit=fit,
                          full_resolution=full_resolution, arrays=arrays,
                          summary="Scripted registration (langslice.register_section).")
    if result.problems:
        raise RegistrationError(
            f"{filename} was not registered: {result.problems[filename]} "
            f"(job folder {result.job_folder})", result.problems)
    return result


__all__ = [
    "RegistrationError", "RegistrationResult", "SectionOutput", "register_job",
    "register_section",
]
