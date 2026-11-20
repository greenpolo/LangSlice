# LangSlice

LangSlice registers histology section images to BrainGlobe atlases. An agent
(or a script) orders the sections, places each at its atlas position and
cutting angles, aligns it in plane, and optionally fits a deformation, using an
image model to trace borders when useful. The command is `langslice`, the
package `src/langslice/` (imports are `langslice.*`). `AGENTS.md` beside this
file and beside each package note is a byte-identical copy
(`tests/test_markdown_files.py` checks it): edit `CLAUDE.md`, then
`cp CLAUDE.md AGENTS.md` in that folder.

## Layers

`hosts > doors > engines > plugins > tools > job > core`: a layer imports only
the layers below it. The one rule that matters: domain code (`core`, `job`,
`tools`, `plugins`, `engines`) never imports `providers`, `google.*`, `litellm`,
`openai` or `mcp`, and nothing below `hosts` imports a host; an image model is
passed in as an argument. import-linter enforces this
(`[tool.importlinter]` in `pyproject.toml`, run by `tests/test_import_layers.py`
and CI) with no exceptions: a violation is fixed by moving code.

| Package | Role |
|---|---|
| `core/` | the science and the stack's shapes: stack, geometry, atlas, section files and renders, pictures, fits (affine, deformable engine, landmarks, the image-model border trace), maps; `refusal.py` |
| `job/` | the `Job` and its folder: state, undo, checkpoint, inputs, submit gates; `settings/` (`JobSpec`, the tool table `tool_table.py`, `ToolsOn` in `tools_on.py`) |
| `tools/` | the operations, as functions on a job; `catalog.py` lists them for every door; the declarations and the job statement (`descriptions/`) |
| `plugins/` | the tool-plugin machinery and `ToolContext`; the public facade is `plugin_api.py` |
| `engines/` | the first-party engines' tools (DeepSlice, Brainways, STalign, SPOmiAlign, emlddmm), each declared once in the tool table |
| `doors/` | the ways in: the built-in agent (`agent/`), MCP (`mcp/`), the `langslice` command with its `job` group for coding agents (`commands/`, `commands/job/`), the library (`library/`) |
| `providers/` | model access (Gemini, OpenAI, OpenAI OAuth, none); imports `core` only |
| `hosts/` | code in LangSlice's own environment: the `gui` command and the window (`gui/`) |
| `_vendor/` | STalign and emlddmm as published; never edited |

`connectors/` holds what is installed into someone else's program:
`claude-desktop/` (MCP configuration) and the `claude-code/`, `antigravity/`
and `codex/` plugins. `packaging/` and `environment.yml` hold the source
installation.

## Rules that hold in the code

- Every way in goes through one job folder (`<images>/langslice/`) and its
  verbs: `langslice linear run`, the agent CLI, the library
  (`langslice.open_job`, `create_job`) and MCP. A door translates; it holds no
  registration logic.
- The tasks `position`, `transform` and `nonlinear` are switched on in
  `JobSpec.tasks`; a task that is off builds no tools and takes its answer from
  `JobSpec.inputs`. A host or benchmark only configures LangSlice with `inputs`
  (what is known per section), `frozen` (which of a section's `position`,
  `angles`, `orientation`, `affine`, `deformation` the agent may not change;
  every write refuses with `FROZEN`) and `disabled` (which tools exist). A task
  that is off freezes its properties everywhere. Nonlinear needs a linear
  placement first.
- A tool is one row of `job/settings/tool_table.py`. Text that names a tool is
  a Python function of the job's `ToolsOn`, so no model-facing sentence names a
  tool the job disables. There is no text macro language and no `tool:option`
  switch.
- A tool of the user's own (`JobSpec.plugins`, `docs/developers/plugins.md`)
  is written against `langslice.plugin_api`, the API the engines use; it
  writes job state only through the `ToolContext` parts (`reads`, `frames`,
  `proposals`, `fits`, `writes`, `pictures`) and files only under
  `plugins/<tool>/`.
- Every write is one undo step and checkpoints `state.json`. `registration.json`,
  pictures and maps are derived from it and never read back.
- A refusal is `Refused(code, message, **facts)` with one of the twelve
  `RefusalCode`s of `core/refusal.py`; each door converts it once. The message
  says what to do next.
- Positions are atlas millimetres from the anterior edge of the volume; axis
  conventions are in `core/space.py`; coordinate contracts are in
  `docs/reference/file_formats.md`.
- ABBA, QuickNII, VisuAlign and DMC-BrainMap are file exports: LangSlice writes
  the file and the program opens it. Nothing drives another program live.
- Pictures are drawn from layers (`core/pictures/canvas.py` `PlacementLayers`,
  `placement.placement_layers`); `core/sections/tissue.py` is the one tissue
  mask. The core returns a picture, never encodes one; the doors package it.
- Judge image output by looking at it: open the raw output, describe what is
  anatomically right and wrong, and report a disagreement between the picture
  and a metric as a finding. Metrics are secondary diagnostics.
- Image-model prompts live in `core/fits/trace/prompts.py`; read
  `src/langslice/core/fits/trace/CLAUDE.md` before changing one. The built-in
  agent's rules are in `src/langslice/doors/agent/CLAUDE.md`, the window's in
  `src/langslice/hosts/gui/CLAUDE.md`.
- New code follows `STYLE.md`. ruff (100 columns), basedpyright (standard mode)
  and import-linter run with no baseline: a finding is fixed in the code, never
  waived and never answered with a new lint rule where the design is what is
  wrong.
- No code comments and no waivers (`tests/test_no_comments_or_waivers.py`, over
  `src/langslice` and `tests`): only docstrings stay. A name says what a line
  or a constant is, the docstring says why. A `# noqa`, `# type: ignore`,
  `# pyright: ignore`, `# pragma` or `# fmt` switch is never added: the code is
  changed until the tool has nothing to report. Related constants are grouped
  in a frozen dataclass whose docstring explains them. Vendored code stays as
  published.
- No Markdown file is written unless the owner asked for that file: no notes,
  plans, summaries or reports in the repository. The allowed kinds are listed
  in `tests/test_markdown_files.py`, which fails on any other; a new kind is
  added there only on the owner's word.
- Markdown states what the code does now, without plans or history, and is
  literal to the code: a behavior change updates the relevant page in `docs/`
  in the same change. Do not document or commit `_local/`, `references/`,
  `out/` or `archive/`.
- The skills `registration-playbook`, `placement`, `damage-and-masks`,
  `landmarks` and `choosing-engines` of `connectors/*/skills/` are generated
  from `src/langslice/skills/`: edit the source, then run
  `python packaging/build_connector_skills.py`
  (`tests/test_connector_skills.py` fails when a copy differs).
  `register-brain` and `setup` exist per connector and are edited in place.
- Debug output: `langslice linear run` writes a JSONL trace of the agent
  session to `<job folder>/trace`; every tool call on every door is a
  `tool_start` / `tool_end` line in `<job folder>/logs/events.jsonl`.

## Docs

`docs/` is built by Sphinx with MyST (`docs/conf.py`, warnings are errors). The
site is the bare minimum a user needs: `index.md`, `getting-started/install.md`,
`register.md`, `abba.md`, `tools.md`, `troubleshooting.md` and the integration
pages. The reference for agents and developers stays in the repository, read on
GitHub and left out of the site (`exclude_patterns`): `docs/agents/`,
`docs/reference/` (`tools.md`, `settings.md`, `file_formats.md`) and
`docs/developers/` (`library.md`, `plugins.md`, `architecture.md`, `gui.md`).
A site page links to them by their GitHub address. `README.md` and
`CONTRIBUTING.md` are for people.

## Environment and checks

The project environment is a uv venv at `.venv` (Python 3.11) at the versions in
`uv.lock`. `--torch-backend=auto` installs the CUDA build of torch that fits
the GPU (PyPI's torch on Windows is CPU-only):

```bash
uv venv --python 3.11 .venv
uv export --frozen --extra dev --no-hashes --no-emit-project -o "$TMP/langslice-lock.txt"
uv pip install -r "$TMP/langslice-lock.txt" --torch-backend=auto
uv pip install --no-deps -e .
```

On Windows, long paths must be on (`LongPathsEnabled`) and Developer Mode lets
the tests make symlinks.

```bash
uv run python -m pytest
uv run python -m ruff check .
uv run python -m ruff format --check .
uv run python -m basedpyright
uv run lint-imports
```

A change that alters pictures or model-facing text is also checked by running
a real brain and looking.
