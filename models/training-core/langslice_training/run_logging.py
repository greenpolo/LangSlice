"""Run-card / launch-snapshot writer for SFT + GRPO runs.

LangSlice training configs default to ``report_to = "none"`` (no trackio/wandb),
so a run's documentation is on-disk. At launch we drop two files into the run's
output dir:

* ``run_card.json`` — machine-readable: kind, run name, timestamp, git commit,
  base/resume model, dataset/data-source, full CLI args, and the *fully-resolved*
  config (TOML defaults + CLI overrides merged). This is the answer to "what
  hyperparameters did this run actually use?".
* ``RUN.md`` — the same, human-readable, so the run dir documents itself instead
  of carrying TRL's auto-generated boilerplate README.

Run *status* and *final metrics* are intentionally NOT written here — they are
derived at read time from ``checkpoint-*/trainer_state.json`` + ``launch.log`` by
``scripts/docker-training/runs_index.ps1``, so they stay correct even when a run
is killed -9 and never gets to run any finalizer.

Everything here is BEST-EFFORT: ``write_run_card`` swallows all exceptions and
returns None on failure. Call sites must additionally wrap the import so a missing
module can never abort training. Logging must never break a run.
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def _git_commit(repo_hint: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_hint), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def write_run_card(
    *,
    output_dir: str | Path,
    kind: str,
    config_path: str | Path | None,
    resolved: dict[str, Any],
    cli_args: dict[str, Any],
    base_model: str | None = None,
    dataset: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Write ``run_card.json`` + ``RUN.md`` into ``output_dir``.

    ``resolved`` is a mapping of config-section name → resolved dict (e.g.
    ``{"grpo": grpo_cfg, "lora": lora_cfg, ...}``). Returns the card dict, or
    None if anything went wrong (never raises).
    """
    try:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        card = {
            "schema_version": SCHEMA_VERSION,
            "kind": kind,
            "run_name": out.name,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "git_commit": _git_commit(out),
            "config_path": str(config_path) if config_path is not None else None,
            "base_model": base_model,
            "dataset": dataset,
            "cli_args": {k: _jsonable(v) for k, v in cli_args.items()},
            "resolved_config": {s: {k: _jsonable(v) for k, v in d.items()} for s, d in resolved.items()},
            "extra": extra or {},
        }
        (out / "run_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
        (out / "RUN.md").write_text(_render_md(card), encoding="utf-8")
        print(f"[run_logging] wrote run card → {out / 'run_card.json'}", file=sys.stderr)
        return card
    except Exception as exc:  # never let logging break training
        print(f"[run_logging] WARN: failed to write run card: {exc!r}", file=sys.stderr)
        return None


def _jsonable(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return str(v)


def _render_md(card: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append(f"# Run: {card['run_name']}")
    lines.append("")
    lines.append("> Auto-written at launch by `langslice_training.run_logging`.")
    lines.append("> Status + final metrics are derived from `trainer_state.json` /")
    lines.append("> `launch.log` (see `scripts/docker-training/runs_index.ps1`).")
    lines.append("")
    lines.append("| field | value |")
    lines.append("| --- | --- |")
    lines.append(f"| kind | `{card['kind']}` |")
    lines.append(f"| created_at | {card['created_at']} |")
    lines.append(f"| git_commit | `{card.get('git_commit')}` |")
    lines.append(f"| base / resume model | `{card.get('base_model')}` |")
    lines.append(f"| dataset / data-source | `{card.get('dataset')}` |")
    lines.append(f"| config | `{card.get('config_path')}` |")
    lines.append("")
    for section, d in card.get("resolved_config", {}).items():
        lines.append(f"## resolved [{section}]")
        lines.append("")
        lines.append("```toml")
        for k, v in d.items():
            lines.append(f"{k} = {json.dumps(v)}")
        lines.append("```")
        lines.append("")
    lines.append("## CLI args")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(card.get("cli_args", {}), indent=2))
    lines.append("```")
    lines.append("")
    return "\n".join(lines)
