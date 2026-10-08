# LangSlice

LangSlice registers histology section images to BrainGlobe atlases: an agent
(or a script) orders the sections, places each at its atlas position and
cutting angles, aligns it in-plane, and optionally corrects atlas borders with
an image model and fits a deformation. The CLI is `langslice`; the package is
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
| `doors/` | one declaration per verb feeds the native agent tools (`tools/`), MCP (`mcp/`), the agent CLI and the other commands (`cli/`), the engine contract and worker (`api/`), and the library (`library.py`, `pipeline.py`) | `doors/CLAUDE.md` |
| `agent/` | the ADK driver of `langslice linear run` | `agent/CLAUDE.md` |
| `providers/` | model access: `gemini-api`, `openai-api`, `openai-oauth`, `none`; no task logic | |
| `hosts/` | host code in LangSlice's own environment: the JSON-lines service, `abba` / `serve` commands, the ABBA agent viewer and log | `hosts/CLAUDE.md`, `hosts/integrations/CLAUDE.md` |

`connectors/` holds what is installed into someone else's program:
`fiji/` (the Java connector for ABBA 0.24, which starts `langslice serve
--stdio`), `claude-desktop/` (configuration for `langslice mcp`),
`claude-code/` and `codex/` (skills and agents over the agent CLI). `packaging/`
and `environment.yml` hold the worker's source installation.

## Rules that hold in the code

- Every way in goes through one job folder and its verbs: the agent run
  (`langslice linear run`), the agent CLI (`langslice-job FOLDER VERB`, `ops`,
  `schema`), the library (`langslice.open_job`, `create_job`),
  MCP and the Fiji worker. `trace_from_atlas` is an internal scripting verb
  (`Verb.hidden` in `ops/registry.py`) in no listing.
- Tasks `reorder` / `position` / `transform` / `nonlinear` are switched on in
  `JobSpec.tasks`; a task that is off builds no tools and takes its answer from
  `JobSpec.inputs`. Nonlinear needs a linear placement first.
- The toolbox (`ops/registry.py`, `docs/linear_design.md`): looking and channel
  tools shared by every run, and one distinct tool per registration method
  (`position_sections`, `interactive_transform`, `elastix_affine`, `ants_syn`,
  `trace_borders`); a task switches its tools on or off whole. Sections are
  named by filename only.
- Every write is one undo step and checkpoints `state.json`; `registration.json`,
  pictures and maps are derived from it and never read back.
- Positions are atlas millimetres from the anterior edge of the volume; axis
  conventions are centralized in `core/space.py`.
- Model-facing text (tool descriptions, job statement, tool replies) and the
  pictures are pinned by golden tests (`tests/test_golden_linear_tools.py`,
  `tests/golden/`); an intended change re-records them with
  `LANGSLICE_UPDATE_GOLDEN=1` (see `tests/golden/record.py`) and the diff is
  reviewed. The golden test needs `antspyx`, `itk-elastix`, `mcp` and
  `google-adk`, and skips without them.
- Image-model prompts live in `core/nonlinear/prompts.py`; every sentence of an
  edit is audited for a second reading, and no border line is made conditional
  on visibility (`docs/nonlinear_design.md`).
- Judge image output by looking at it: open the raw output, describe what is
  anatomically right and wrong, and report a disagreement between the picture
  and a metric as a finding. Metrics are secondary diagnostics.
- Markdown is literal to the code: a behavior change updates the relevant page
  in `docs/` in the same change. Docs state what the code does now, without
  plans or history. Do not document or commit `_local/`, `references/`, `out/`
  or `archive/`.
- Debug output: `langslice linear run` writes a full-content JSONL trace of
  agent sessions to `--trace-dir`, else `LANGSLICE_TRACE_DIR`, else
  `<job folder>/trace`;
  `LANGSLICE_ADK_CAPTURE_REQUESTS_DIR` writes redacted ADK request captures.

## Docs

`docs/` (built by MkDocs, `mkdocs.yml`, strict): `index.md`, `abba_installation.md`,
`cli.md`, `agent_cli.md`, `library.md`, `file_formats.md`,
`architecture_overview.md`, `linear_design.md`, `nonlinear_design.md`,
`abba_plugin_design.md`.

## Environment and checks

Linux + bash. The project environment is a uv venv at `.venv` (Python 3.11):
`uv venv --python 3.11 .venv && uv pip install -e ".[dev]"`; the ANTs engine
(antspyx) is a core dependency below Python 3.14.

```bash
source .venv/bin/activate
python -m pytest              # markers: `slow` is deselected by default
python -m ruff check .
python -m basedpyright
lint-imports                  # the layer contracts
langslice version
```
