# LangSlice

LangSlice registers histology section images to BrainGlobe atlases: an agent
(or a script) orders the sections, places each at its atlas position and
cutting angles, aligns it in-plane, and optionally fits a deformation, using an image model
to trace borders when useful. The CLI is `langslice`; the package is
`src/langslice/` (imports are `langslice.*`). `AGENTS.md` is a byte-identical
copy of this file, and each package guide `CLAUDE.md` under `src/langslice/`
has an `AGENTS.md` twin: edit one, then `cp CLAUDE.md AGENTS.md` in that folder.

## Layers

`core` < `job` < `ops` < `doors` : `agent` < `hosts`; each layer imports only
the layers below it. `providers/` (model access) sits beside them: core, job and
ops never import it, nor `google.*`, `litellm`, `openai` or `mcp`; an image
model is passed in as an argument. import-linter enforces this
(`[tool.importlinter]` in `pyproject.toml`, run by `tests/test_import_layers.py`
and CI) with no exceptions: a violation is fixed by moving code.

| Package | Role | Guide |
|---|---|---|
| `core/` | state, job spec, renders and pictures, in-plane fits, atlas access, deformable engine, image-model border route | `core/CLAUDE.md`, `core/atlas/`, `core/deformable/`, `core/nonlinear/` |
| `job/` | the `Job`: state, undo, checkpoint, submit gates; the job folder `<images>/langslice/` and its public files | `job/CLAUDE.md` |
| `ops/` | the verbs, as functions on a job; `registry.py` lists them for every door | `ops/CLAUDE.md` |
| `doors/` | one declaration per verb feeds the native agent tools (`tools/`), MCP (`mcp/`), the agent CLI and the other commands (`cli/`), the engine contract and worker (`api/`), and the library (`library.py`) | `doors/CLAUDE.md` |
| `agent/` | the ADK driver of `langslice linear run` | `agent/CLAUDE.md` |
| `providers/` | model access: `gemini-api`, `openai-api`, `openai-oauth`, `none`; no task logic | |
| `hosts/` | host code in LangSlice's own environment: the JSON-lines service, `abba` / `gui` / `serve` commands, the LangSlice window (`gui/`), the ABBA agent viewer and log | `hosts/CLAUDE.md`, `hosts/gui/CLAUDE.md`, `hosts/integrations/CLAUDE.md` |

`connectors/` holds what is installed into someone else's program:
`fiji/` (the Java connector for ABBA 0.24, which starts `langslice serve
--stdio`), `claude-desktop/` (configuration for `langslice mcp`),
`claude-code/`, `antigravity/` and `codex/` (skills and agents over the agent CLI). `packaging/`
and `environment.yml` hold the worker's source installation.

## Rules that hold in the code

- Every way in goes through one job folder and its verbs: the agent run
  (`langslice linear run`), the agent CLI (`langslice-job FOLDER VERB`, `ops`,
  `schema`), the library (`langslice.open_job`, `create_job`),
  MCP and the Fiji worker.
- Tasks `position` / `transform` / `nonlinear` are switched on in
  `JobSpec.tasks`; a task that is off builds no tools and takes its answer from
  `JobSpec.inputs`. Nonlinear needs a linear placement first.
- A host or benchmark only configures LangSlice, with three settings:
  `inputs` (what is already known per section), `frozen` (which of a
  section's `position`, `angles`, `orientation`, `affine`, `deformation` the
  agent may not change; every write op refuses a change, `FROZEN`) and
  `disabled` (which tools exist). A task that is off freezes its properties
  everywhere.
- The toolbox (`ops/registry.py`, `docs/registration.md`): looking and channel
  tools shared by every run, and one distinct tool per registration method
  (`position_sections`, `interactive_transform`, `elastix_affine`, `fireants_syn`,
  `set_landmarks` / `landmark_triangulation`, `trace_borders`); a task switches its tools on or off whole. Sections are
  named by filename only.
- A tool plugin (`JobSpec.plugins`, `docs/plugins.md`) adds first-class tools
  through `langslice.plugin_api`; it loads only where the user chose it
  (library, `linear run --tool-plugin`), never through MCP or the ABBA worker,
  writes job state only through the context's operations and files only under
  `plugins/<tool>/`; a job without plugins shows no trace of the seam.
- Any tool, or option of a tool, is switched off per job by `JobSpec.disabled`
  (`tool` or `tool:option`; `core/toolset.py` lists the valid names, the
  spec refuses an unknown name or a list that cannot work). Every sentence
  that names a tool or an option is written with what it needs
  (`core/needs.py`), so no model-facing text names what the job disables;
  `tests/test_disabled_tools.py` checks it. A new tool or option is a row in
  `core/toolset.py`, and its text carries its marks.
- Every write is one undo step and checkpoints `state.json`; `registration.json`,
  pictures and maps are derived from it and never read back.
- Positions are atlas millimetres from the anterior edge of the volume; axis
  conventions are centralized in `core/space.py`.
- Model-facing text (tool descriptions, job statements, refusals) is pinned
  by a text snapshot (`tests/test_agent_text_snapshot.py`,
  `tests/snapshots/agent_text/`); an intended change regenerates it with
  `LANGSLICE_UPDATE_SNAPSHOT=1` and the diff is reviewed. Pictures are
  pinned by no test: a change to them is judged by looking (below).
- Image-model prompts live in `core/nonlinear/prompts.py`; every sentence of an
  edit is audited for a second reading, and no border line is made conditional
  on visibility (`src/langslice/core/nonlinear/CLAUDE.md`).
- Judge image output by looking at it: open the raw output, describe what is
  anatomically right and wrong, and report a disagreement between the picture
  and a metric as a finding. Metrics are secondary diagnostics.
- Markdown is literal to the code: a behavior change updates the relevant page
  in `docs/` in the same change. Docs state what the code does now, without
  plans or history. Do not document or commit `_local/`, `references/`, `out/`
  or `archive/`.
- Debug output: `langslice linear run` writes a full-content JSONL trace of
  agent sessions to `<job folder>/trace`, and the same file to `--trace-dir`,
  else `LANGSLICE_TRACE_DIR`, when one is set; every tool call on every door
  is a `tool_start` / `tool_end` line in `<job folder>/logs/events.jsonl`;
  `LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` writes redacted ADK request captures.

## Docs

`docs/` (built by MkDocs, `mkdocs.yml`, strict): `index.md`, `agents.md`,
`abba.md`, `registration.md`, `library.md`, `plugins.md`, `file_formats.md`,
`architecture.md`.

## Environment and checks

Linux + bash. The project environment is a uv venv at `.venv` (Python 3.11):
`uv venv --python 3.11 .venv && uv pip install -e ".[dev]"`; the deformable engine
(FireANTs on torch) and SimpleITK are core dependencies.

```bash
source .venv/bin/activate
python -m pytest              # in parallel (pytest-xdist); `slow` is deselected by default
python -m ruff check .
python -m basedpyright
lint-imports                  # the layer contracts
langslice version
```
