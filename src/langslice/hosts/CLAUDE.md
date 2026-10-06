# LangSlice `hosts/` — host connectors in LangSlice's own environment

Package guide for `src/langslice/hosts/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The top layer of the layered core: code that connects LangSlice to another
program and runs in LangSlice's own Python environment. What is installed into someone else's program lives in
`connectors/` at the repo root instead (the Fiji/ABBA connector, the Claude
Desktop configuration).

## The layer rule

A host module may import every layer below it: the core, the job layer,
the operations, the doors, the agent driver and the providers. Nothing
below imports a host (import-linter's layers contract, `pyproject.toml`,
run by `tests/test_import_layers.py`, no exceptions listed). The `langslice`
command reaches the host commands by module path only (`hosts/cli.py`,
`doors.cli.HOST_COMMANDS`).

## Files and sub-packages

- `cli.py` — the host commands `abba` (ABBA 0.24.x started from Python with
  the Fiji connector jar on the classpath and the passive Python viewer and
  log listening to its runs; `--connector-jar`, `--no-viewer`, `--no-log`)
  and `serve` (the engine service), registered with the `langslice`
  command by module path.

- `integrations/` — `abba_launch.py` (what `langslice abba` starts, and
  the listener of the connector's run messages), the ABBA agent viewer
  (`abba_follow.py`, `abba_compare.py`, `abba_overview.py`) and activity log
  (`abba_chat.py` + `static/`, `abba_activity.py`): `integrations/CLAUDE.md`.
  The Fiji connector is the one ABBA integration; this package only follows
  its runs.
- `api/` — `service.py` (`langslice serve --stdio`: the JSON-lines engine
  service the Fiji connector starts). The protocol models, the runtime
  handlers, setup, saved ABBA jobs and the JVM-free snapshot worker
  (`abba_worker.py`, whose checkpoints carry the affine and warp rows) are
  door-level, in `doors/api/`. `docs/abba_plugin_design.md`,
  `docs/abba_installation.md`.
