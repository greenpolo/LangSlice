"""Saved ABBA jobs: a job folder next to the snapshots, found by id.

``prepare_saved_job`` (the engine method ``mcp.prepare``: ABBA's Claude
mode) saves a job an MCP host opens later by id. The job lives in the job
folder next to its images (``<images>/langslice/``,
:mod:`langslice.job.layout`): ``job.json`` there holds the settings and,
under ``host``, the kind (``host``), the host's parameters and the notes;
``prompt.txt`` the copy prompt. The id leads there through the index
(:mod:`langslice.job.index`, ``~/.langslice/jobs/<id>.json``), which also
holds the host's loopback channel. No model access; the image provider's
key or login is only checked for presence (``setup.image_model_connected``).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from langslice.core.spec import JobSpec
from langslice.doors.api import setup as provider_setup
from langslice.doors.api.abba_worker import prepare_linear
from langslice.job import index
from langslice.job.layout import (
    JobLayout,
    check_owner,
    locate_job_folder,
    read_job_file,
    write_job_file,
)

#: The saved job's own format inside ``job.json`` (``host``); a newer one is
#: refused.
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


def copy_prompt(job_id: str, spec: Any) -> str:
    """The prompt the user pastes into the MCP host: the job id, its settings and
    how to begin. The user's notes are not repeated here: ``start_job``'s
    statement carries them (``job.json`` ``notes``), as every door's does."""
    labels = {"position": "positioning",
              "transform": "linear alignment", "nonlinear": "nonlinear (deformable) alignment"}
    selected = ", ".join(labels[task] for task in spec.tasks)
    lines = [f"Use the LangSlice connector for job {job_id}.",
             f'Call start_job(job_id="{job_id}") first.',
             f"Selected tasks: {selected}; atlas: {spec.atlas}; plane: {spec.plane}.",
             "Use only the LangSlice tools and finish with submit."]
    if spec.has("position"):
        thickness = (f"{spec.position.thickness_um:g} µm"
                     if spec.position.thickness_um is not None else "not supplied")
        interval = (f"{spec.position.interval_um:g} µm"
                    if spec.position.interval_um is not None else "not supplied")
        lines.append(f"Positioning: section thickness {thickness}, interval {interval}.")
    if spec.has("transform"):
        lines.append(
            f"Linear: automatic affine {'enabled' if spec.transform.automatic else 'disabled'}, "
            f"up to {spec.transform.max_parallel} sections per interactive adjustment; "
            f"hemisphere flipping {'enabled' if spec.transform.flip else 'disabled'}."
        )
    if spec.has("nonlinear"):
        if not spec.nonlinear.uses_image_model:
            lines.append("Nonlinear: deformations are fitted to the stain alone "
                         "(no image model chosen).")
        elif provider_setup.image_model_connected(spec.nonlinear.provider):
            lines.append(f"Nonlinear: the image model ({spec.nonlinear.provider}) traces "
                         "region borders with trace_borders.")
        else:
            lines.append("Nonlinear: the image-model tool (trace_borders) is off: no image "
                         "model is connected to LangSlice (no key or login for "
                         f"{spec.nonlinear.provider}); deformations are fitted to the stain.")
    inputs = spec.inputs or {}
    if inputs.get("locked"):
        lines.append("Existing alignment is locked for: " + ", ".join(inputs["locked"]) + ".")
    if inputs.get("damaged"):
        lines.append("User-marked damage: " + "; ".join(
            f"{name}: {note}" for name, note in inputs["damaged"].items()
        ) + ".")
    from langslice.agent.prompt import task_notes
    lines.extend(task_notes(spec))
    return "\n".join(lines)


def prepare_saved_job(params: dict[str, Any]) -> dict[str, Any]:
    notes = params.get("notes", "")
    if not isinstance(notes, str):
        raise ValueError("notes must be text")
    channel = validate_channel(params.get("host_channel"))
    run_params = {key: value for key, value in params.items()
                  if key not in {"notes", "host_channel"}}
    prepared = prepare_linear(run_params)
    job_id, layout = _write_job(
        prepared.folder, {"kind": "host", "params": run_params},
        spec=prepared.spec, channel=channel, notes=notes,
    )
    return {"job_id": job_id, "job_dir": str(layout.folder),
            "prompt": _save_prompt(layout, copy_prompt(job_id, prepared.spec))}


def _save_prompt(layout: JobLayout, prompt: str) -> str:
    """Keep the prompt in the job folder, so it can be copied (or run) again."""
    layout.prompt_file.write_text(prompt + "\n", encoding="utf-8")
    return prompt


def _write_job(
    image_folder: Path, host: dict[str, Any], *, spec: JobSpec,
    channel: dict[str, Any] | None, notes: str = "",
) -> tuple[str, JobLayout]:
    """Save the job: its folder located and checked (another image
    folder's job is refused before anything is written), ``job.json`` (the
    user's *notes* under ``notes``, which every door reads:
    :func:`langslice.doors.statement.read_notes`) and the index entry."""
    job_id = index.new_id()
    images = Path(image_folder).expanduser().resolve()
    folder, fallback = locate_job_folder(images, spec.job_dir, root=jobs_root(),
                                         job_id=job_id, register=False)
    layout = JobLayout(folder, images)
    check_owner(layout)
    layout.ensure()
    write_job_file(layout, job_id=job_id, spec=spec.to_dict(), notes=notes,
                   host={"format": FORMAT_VERSION, **host})
    index.register(jobs_root(), job_id, layout.folder, host_channel=channel, fallback=fallback)
    from langslice.doors.card import write_card

    write_card(layout)
    return job_id, layout


def load_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    """``(job folder, record)`` of a saved job; the record is the job file's
    ``host`` fields plus ``job_id``, ``spec`` and ``host_channel``."""
    if not isinstance(job_id, str) or re.fullmatch(r"[0-9a-f]{12}", job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    root = jobs_root()
    entry = index.lookup(root, job_id)
    if entry is None:
        raise ValueError(f"No saved LangSlice job {job_id}")
    layout = JobLayout(Path(entry["job_folder"]))
    data = read_job_file(layout)
    if data is None or not isinstance(data.get("host"), dict):
        raise ValueError(f"Saved job {job_id}: {layout.folder} holds no saved job")
    if data.get("job_id") != job_id:
        raise ValueError(f"Saved job {job_id} was replaced by job {data.get('job_id')} in "
                         f"{layout.folder} (one job folder per image folder)")
    host = dict(data["host"])
    if int(host.get("format", FORMAT_VERSION)) > FORMAT_VERSION:
        raise ValueError("This saved job was written by a newer LangSlice. Update LangSlice "
                         "to open it.")
    record = {**host, "job_id": job_id, "spec": data.get("spec"),
              "host_channel": validate_channel(entry.get("host_channel"))}
    if record.setdefault("kind", "host") != "host":
        raise ValueError(f"Saved job {job_id} is no ABBA job; open its image folder "
                         f"instead: start_job(image_folder=...)")
    return layout.folder, record
