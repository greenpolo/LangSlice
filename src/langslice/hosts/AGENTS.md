# LangSlice `hosts/` — host connectors in LangSlice's own environment

Package guide for `src/langslice/hosts/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The top layer of the layered core (folder move, 2026-10-04): code that
connects LangSlice to another program and runs in LangSlice's own Python
environment. What is installed into someone else's program lives in
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

- `cli.py` — the host commands `abba` (ABBA with LangSlice installed,
  optionally with the linear agent's live mirror) and `serve` (the engine
  service), registered with the `langslice` command by module path.

- `integrations/` — the ABBA registration plugin, the live linear mirror,
  the ABBA viewer and activity log (formerly top-level `integrations/`):
  `integrations/CLAUDE.md`. The QUINT writer that used to sit there is
  `job/quint.py` (an operation writes it).
- `api/` — `service.py` (`langslice serve --stdio`: the JSON-lines engine
  service the Fiji connector starts). Its `nonlinear.abba` worker
  (`nonlinear_worker.py`), whose only caller was the Fiji connector's unused
  Java `nonlinear(...)`, was removed 2026-10-04; ABBA's nonlinear route is the
  abba-python registration plugin (`integrations/abba.py`). The protocol models, the runtime handlers, setup, saved
  Claude jobs and the JVM-free linear snapshot worker are door-level, in
  `doors/api/` (moved down 2026-10-04). `docs/abba_plugin_design.md`,
  `docs/abba_installation.md`.
