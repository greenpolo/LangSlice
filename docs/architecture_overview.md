# Architecture

LangSlice is one Python package, `langslice` (`src/langslice/`). Each layer
imports only the layers below it; import-linter enforces this
(`[tool.importlinter]` in `pyproject.toml`, run by `tests/test_import_layers.py`),
and a violation is fixed by moving code, never by an exception.

| Layer | Package | Holds |
|---|---|---|
| hosts | `hosts/` | host connectors in LangSlice's own environment: the JSON-lines service, the `abba` and `serve` commands, the agent viewer and log |
| doors, agent | `doors/`, `agent/` | the ways in: native agent tools, MCP server, agent CLI, Python library, engine contract; and the ADK driver of the agent |
| ops | `ops/` | the verbs: every write and every viewing read as a function on a job, listed by `registry.py` |
| job | `job/` | one `Job` owning the state, undo, checkpoint and submit gates; the job folder's layout and public files |
| core | `core/` | state, spec, renders, in-plane fits, atlas access, deformable engine, image-model border route |
| providers | `providers/` | model access (`gemini-api`, `openai-api`, `openai-oauth`); beside the layers, imported by doors only |

Core, job and ops import no provider, agent framework or model client
(`google`, `litellm`, `openai`, `mcp`); an image model is passed in as an
argument. Each package has a
`CLAUDE.md` code map (`AGENTS.md` is its identical twin).

## One job, many doors

A job is a folder `<images>/langslice/` plus the spec that made it
(`core/spec.py`, `JobSpec`). Every way of working on it runs the same verbs on
the same `Job`:

- `langslice linear run FOLDER`: the built-in agent (ADK) in one session.
- `langslice mcp`: the verbs as MCP tools for a host that brings its own model
  (Claude Desktop), opening a job folder made with `langslice-job FOLDER init`.
- `langslice-job FOLDER VERB`: the agent CLI, for coding agents with a shell.
- `langslice.open_job`, `create_job`: the Python library.
- `langslice serve --stdio`: the JSON-lines worker the Fiji connector starts.

One declaration per verb (`doors/declarations.py`) feeds all of them. Every
write is one undo step and checkpoints `state.json`; writers of different
processes serialize on the job folder's lock and pick up each other's changes.
Image files are never modified.

## Control flow

1. A door fills a `JobSpec` and opens the `Job` (ingesting the folder, or
   resuming the checkpoint).
2. Tasks switch tools on: `reorder`, `position`, `transform`, and the opt-in
   `nonlinear`. A task that is off builds no tools and takes its answer from
   the spec's `inputs` ([linear_design.md](linear_design.md)).
3. The agent (or script) calls verbs: order, atlas position and cutting
   angles, an in-plane affine per section, and with `nonlinear` a deformable
   fit per section, optionally reading an image-model border trace
   ([nonlinear_design.md](nonlinear_design.md)).
4. `submit` checks the gates and writes the results: `registration.json`, each
   section's coordinate and label maps, and the QuickNII / VisuAlign exports
   ([file_formats.md](file_formats.md)).

## Hosts

- Fiji / ABBA: the connector in `connectors/fiji/` starts `langslice serve
  --stdio` and applies the worker's checkpoints as ABBA registration steps
  ([abba_plugin_design.md](abba_plugin_design.md)).
- Claude Desktop: `connectors/claude-desktop/`; Claude Code and Codex:
  `connectors/claude-code/`, `connectors/codex/`.
- `langslice abba` starts ABBA from Python with the connector, the agent viewer
  and the agent log.

## Debugging

`LANGSLICE_TRACE_DIR` (or `langslice linear run --trace-dir`, or the worker's
`trace_dir`) writes a full-content JSONL trace of each agent session.
`LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` writes redacted ADK request captures.
