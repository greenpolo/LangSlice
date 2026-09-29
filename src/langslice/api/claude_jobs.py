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
    labels = {"reorder": "section order and orientation", "position": "Positioning",
              "transform": "Linear alignment"}
    selected = ", ".join(labels[task] for task in spec.tasks)
    lines = [f"Use the LangSlice connector for job {job_id}.",
             f'Call start_job(job_id="{job_id}") first.',
             f"Selected tasks: {selected}; atlas: {spec.atlas}; plane: {spec.plane}.",
             "The saved job enforces the dialog settings; "
             "use only LangSlice tools and finish with submit."]
    if spec.has("position"):
        lines.append(
            f"Positioning: section thickness {spec.position.thickness_um:g} µm, "
            f"interval {spec.position.interval_um:g} µm; "
            f"hemisphere flipping {'enabled' if spec.reorder.flip else 'disabled'}."
        )
    if spec.has("transform"):
        lines.append(
            f"Linear: automatic affine {'enabled' if spec.transform.automatic else 'disabled'}, "
            f"up to {spec.transform.max_parallel} sections per interactive adjustment."
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
    job_id = uuid.uuid4().hex[:12]
    folder = jobs_root() / job_id
    folder.mkdir(mode=0o700, parents=True)
    record = {"format_version": FORMAT_VERSION, "job_id": job_id,
              "created_at": datetime.now(timezone.utc).isoformat(),
              "params": run_params, "notes": notes, "host_channel": channel}
    path = folder / "job.json"
    with path.open("x", encoding="utf-8") as handle:
        path.chmod(0o600)
        json.dump(record, handle, indent=2)
    return {"job_id": job_id, "job_dir": str(folder),
            "prompt": copy_prompt(job_id, prepared.spec, notes)}


def load_job(job_id: str) -> tuple[Path, dict[str, Any]]:
    if re.fullmatch(r"[0-9a-f]{12}", job_id) is None:
        raise ValueError("Invalid LangSlice job id")
    folder = jobs_root() / job_id
    record = json.loads((folder / "job.json").read_text(encoding="utf-8"))
    if record.get("format_version") != FORMAT_VERSION or record.get("job_id") != job_id:
        raise ValueError("Unsupported or mismatched LangSlice job file")
    validate_channel(record.get("host_channel"))
    return folder, record
