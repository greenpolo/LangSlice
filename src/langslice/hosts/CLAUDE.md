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
run by `tests/test_import_layers.py`). Known violations, listed in that
contract's `ignore_imports` and to be removed by moving code: the CLI's host
commands (`doors/cli/__init__.py`, `doors/cli/hosts.py`,
`doors/cli/register.py`) and the MCP door (`doors/mcp/server.py`,
`doors/mcp/host_channel.py`) import `hosts.api` / `hosts.integrations`.

## Sub-packages

- `integrations/` — the ABBA registration plugin, the live linear mirror,
  the ABBA viewer and activity log (formerly top-level `integrations/`):
  `integrations/CLAUDE.md`. The QUINT writer that used to sit there is
  `job/quint.py` (an operation writes it).
- `api/` — the JSON-lines worker protocol (`models.py`, `service.py`),
  desktop setup and authentication (`setup.py`), the ABBA worker
  (`abba_worker.py`: `prepare_linear`, live checkpoints), saved Claude jobs
  (`claude_jobs.py`) and the nonlinear registration runtime the worker and
  `langslice nonlinear register` call (`runtime.py`); formerly top-level
  `api/`. `docs/abba_plugin_design.md`, `docs/abba_installation.md`.
