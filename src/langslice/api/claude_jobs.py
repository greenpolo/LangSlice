"""Saved Claude jobs: a job folder next to the images, found by id.

``prepare_claude`` (ABBA's Claude mode) and ``prepare_folder`` (``langslice
claude prepare``) save a job the MCP server opens later by id. The job lives
in the job folder next to its images (``<images>/langslice/``,
:mod:`langslice.job.layout`): ``job.json`` there holds the settings and,
under ``host``, the kind (``host`` or ``folder``), the host's parameters,
the notes and the trace folder; ``prompt.txt`` the copy prompt. The id leads
there through the index (:mod:`langslice.job.index`,
``~/.langslice/jobs/<id>.json``), which also holds the host's loopback
channel. A phase-2 saved job (the whole job under ``~/.langslice/jobs/<id>/``)
is moved into its job folder on first open. No model or credential access.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langslice.api.abba_worker import prepare_linear
from langslice.job import index, migrate
from langslice.job.layout import (
    JobLayout,
    check_owner,
    job_folder_for,
    locate_job_folder,
    read_job_file,
    write_job_file,
)

#: The saved job's own format inside ``job.json`` (``host``). 2 (2026-10-03):
#: the job folder next to the images; 1 was the whole job under
#: ``~/.langslice/jobs/<id>/``.
FORMAT_VERSION = 2


def jobs_root() -> Path:
    """Where the id index lives (``~/.langslice/jobs``)."""
    return index.default_root()


def validate_channel(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != {"address", "port", "token"}:
        raise ValueError("host_channel requires address, port and token")
    if value["address"] not in {"127.0.0.1", "::1"}:
        raise ValueError("The host channel must use a literal loopback address")
    port, token = value["port"], value["token"]
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError("The host channel port must be an integer from 1 to 65535")
    if not isinstance(token, str) or len(token) < 32 or len(token) > 256:
        raise ValueError("The host channel token must contain 32 to 256 characters")
    return dict(value)


def copy_prompt(job_id: str, spec: Any, notes: str) -> str:
    labels = {"reorder": "section order", "position": "positioning",
              "transform": "linear alignment"}
    selected = ", ".join(labels[task] for task in spec.tasks)
    lines = [f"Use the LangSlice connector for job {job_id}.",
             f'Call start_job(job_id="{job_id}") first.',
             f"Selected tasks: {selected}; atlas: {spec.atlas}; plane: {spec.plane}.",
             "Use only the LangSlice tools and finish with submit."]
    if spec.has("position"):
        lines.append(
            f"Positioning: section thickness {spec.position.thickness_um:g} µm, "
            f"interval {spec.position.interval_um:g} µm."
        )
    if spec.has("transform"):
        lines.append(
            f"Linear: automatic affine {'enabled' if spec.transform.automatic else 'disabled'}, "
            f"up to {spec.transform.max_parallel} sections per interactive adjustment; "
            f"hemisphere flipping {'enabled' if spec.reorder.flip else 'disabled'}."
        )
    inputs = spec.inputs or {}
    if inputs.get("locked"):
        lines.append("Existing alignment is locked for: " + ", ".join(inputs["locked"]) + ".")
    if inputs.get("damaged"):
        lines.append("User-marked damage: " + "; ".join(
            f"{name}: {note}" for name, note in inputs["damaged"].items()
        ) + ".")
    from langslice.linear.prompt import task_notes
    lines.extend(task_notes(spec))
    if notes.strip():
        lines.extend(["User notes:", notes])
    return "\n".join(lines)


def prepare_claude(params: dict[str, Any]) -> dict[str, Any]:
    notes = params.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be text")
    channel = validate_channel(params.get("host_channel"))
    run_params = {key: value for key, value in params.items()
                  if key not in {"notes", "host_channel"}}
    if "nonlinear" in (run_params.get("spec") or {}).get("tasks", []):
        raise ValueError("Image generation is unavailable in Claude mode")
    prepared = prepare_linear(run_params)
    if prepared.spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable in Claude mode")
    job_id, layout = _write_job(
        prepared.folder, {"kind": "host", "params": run_params, "notes": notes},
        spec=prepared.spec.to_dict(), channel=channel,
    )
    return {"job_id": job_id, "job_dir": str(layout.folder),
            "prompt": _save_prompt(layout, copy_prompt(job_id, prepared.spec, notes))}


def prepare_folder(spec: Any, notes: str = "", trace_dir: str | None = None) -> dict[str, Any]:
    """A saved job for a plain folder of sections: the CLI's Copy prompt.

    No host and no live channel. The job folder next to the sections holds
    it; the server resumes it from the checkpoint there (a job folder that
    already holds a checkpoint is continued: one folder, one job).
    """
    from langslice.linear.discovery import discover_slices

    if spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable in Claude mode")
    folder = Path(spec.image_folder).expanduser().resolve(strict=True)
    if not discover_slices(str(folder)):
        raise ValueError(f"No section images found in {folder}")
    spec.image_folder = str(folder)
    job_id, layout = _write_job(
        folder, {"kind": "folder", "notes": notes,
                 "trace_dir": str(Path(trace_dir).expanduser().resolve()) if trace_dir else None},
        spec=spec.to_dict(), channel=None,
    )
    return {"job_id": job_id, "job_dir": str(layout.folder),
            "prompt": _save_prompt(layout, copy_prompt(job_id, spec, notes))}


def _save_prompt(layout: JobLayout, prompt: str) -> str:
    """Keep the prompt in the job folder, so it can be copied (or run) again."""
    layout.prompt_file.write_text(prompt + "\n", encoding="utf-8")
    return prompt


def _write_job(
    image_folder: Path, host: dict[str, Any], *, spec: dict[str, Any],
    channel: dict[str, Any] | None,
) -> tuple[str, JobLayout]:
    job_id = index.new_id()
    images = Path(image_folder).expanduser().resolve()
    folder, fallback = locate_job_folder(images, spec.get("job_dir"), root=jobs_root(),
                                         job_id=job_id, register=False)
    layout = JobLayout(folder, images)
    check_owner(layout)
    if layout.folder == job_folder_for(images):
        migrate.migrate_beside_images(layout)
    layout.ensure()
    write_job_file(layout, job_id=job_id, spec=spec,
                   host={"format": FORMAT_VERSION, **host})
    index.register(jobs_root(), job_id, layout.folder, host_channel=channel, fallback=fallback)
    return job_id, layout


def load_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    """``(job folder, record)`` of a saved job; the record is the job file's
    ``host`` fields plus ``job_id``, ``spec`` and ``host_channel``.

    A phase-2 saved job is moved into its job folder first.
    """
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{12}", job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    root = jobs_root()
    entry = index.lookup(root, job_id)
    if entry is None:
        if not (index.legacy_dir(root, job_id) / "job.json").exists():
            raise ValueError(f"No saved LangSlice job {job_id}")
        entry = migrate.migrate_saved_job(root, job_id)
    layout = JobLayout(Path(entry["job_folder"]))
    data = read_job_file(layout)
    if data is None or not isinstance(data.get("host"), dict):
        raise ValueError(f"Saved job {job_id}: {layout.folder} holds no saved job")
    if data.get("job_id") != job_id:
        raise ValueError(f"Saved job {job_id} was replaced by job {data.get('job_id')} in "
                         f"{layout.folder} (one job folder per image folder)")
    host = dict(data["host"])
    if int(host.get("format", 1)) > FORMAT_VERSION:
        raise ValueError("This saved job was written by a newer LangSlice. Update LangSlice "
                         "to open it.")
    record = {**host, "job_id": job_id, "spec": data.get("spec"),
              "host_channel": validate_channel(entry.get("host_channel"))}
    record.setdefault("kind", "host")
    if record["kind"] not in ("host", "folder"):
        raise ValueError(f"Unknown LangSlice job kind {record['kind']!r}")
    return layout.folder, record
