"""Local, immutable host job handoffs; no model or credential access."""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langslice.api.abba_worker import prepare_linear

FORMAT_VERSION = 1


def jobs_root() -> Path:
    return Path.home() / ".langslice" / "jobs"


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
    job_id, folder = _write_job(
        {"kind": "host", "params": run_params, "notes": notes, "host_channel": channel}
    )
    return {"job_id": job_id, "job_dir": str(folder),
            "prompt": _save_prompt(folder, copy_prompt(job_id, prepared.spec, notes))}


def prepare_folder(spec: Any, notes: str = "", trace_dir: str | None = None) -> dict[str, Any]:
    """A saved job for a plain folder of sections: the CLI's Copy prompt.

    No host and no live channel. The server resumes it from its own
    checkpoint in the job directory, so the user's folder is never written.
    """
    from langslice.linear.discovery import discover_slices

    if spec.has("nonlinear"):
        raise ValueError("Image generation is unavailable in Claude mode")
    folder = Path(spec.image_folder).expanduser().resolve(strict=True)
    if not discover_slices(str(folder)):
        raise ValueError(f"No section images found in {folder}")
    spec.image_folder = str(folder)
    job_id, job_dir = _write_job(
        {"kind": "folder", "spec": spec.to_dict(), "notes": notes, "host_channel": None,
         "trace_dir": str(Path(trace_dir).expanduser().resolve()) if trace_dir else None}
    )
    return {"job_id": job_id, "job_dir": str(job_dir),
            "prompt": _save_prompt(job_dir, copy_prompt(job_id, spec, notes))}


def _save_prompt(folder: Path, prompt: str) -> str:
    """Keep the prompt beside the job, so it can be copied (or run) again."""
    (folder / "prompt.txt").write_text(prompt + "\n", encoding="utf-8")
    return prompt


def _write_job(fields: dict[str, Any]) -> tuple[str, Path]:
    job_id = uuid.uuid4().hex[:12]
    folder = jobs_root() / job_id
    folder.mkdir(mode=0o700, parents=True)
    record = {"format_version": FORMAT_VERSION, "job_id": job_id,
              "created_at": datetime.now(timezone.utc).isoformat(), **fields}
    path = folder / "job.json"
    with path.open("x", encoding="utf-8") as handle:
        path.chmod(0o600)
        json.dump(record, handle, indent=2)
    return job_id, folder


def load_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{12}", job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    folder = jobs_root() / job_id
    record = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    if record.get("format_version") != FORMAT_VERSION or record.get("job_id") != job_id:
        raise ValueError("Unsupported or mismatched LangSlice job file")
    validate_channel(record.get("host_channel"))
    record.setdefault("kind", "host")
    if record["kind"] not in ("host", "folder"):
        raise ValueError(f"Unknown LangSlice job kind {record['kind']!r}")
    return folder, record
