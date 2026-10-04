# LangSlice `doors/` — one vocabulary, every door

Package guide for `src/langslice/doors/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

Phase 5 of the layered refactor (2026-10-04). A verb (an agent tool) has ONE
name, one argument list and one description; every door is generated from
them and from the registry (`ops/registry.py`: `VERBS`, `enabled(spec)`):

| Door | Built by | Driver |
|---|---|---|
| agent tools (ADK) | `doors/tools/toolbox.py` `build_tools`: the verbs `enabled(spec)` names, each `declarations.declare`d | LangSlice's agent |
| MCP tools | the same toolbox (`doors/mcp/server.py`), plus the door's `start_job`, `show_stack`; `readOnlyHint` = read verbs; `trace_borders` only when the job's image model is connected (`server.image_model_off`, `api.setup.image_model_connected`; else `build_tools(image_model_connected=False)`) | Claude Desktop, Claude Code locked to it |
| agent CLI | `cli/job.py` over the same toolbox, gates off, `level="auto"`, plus the scripting verbs (`export_maps`) | Claude Code, Codex |
| library | `library.py` (`langslice.open_job`) over the same toolbox, the scripting verbs included | a script |

A verb is never renamed once shipped (scripts and agents call it by name).
The goldens (`tests/golden/linear_tools/*_declarations_*`) pin what the ADK
and MCP doors declare, byte for byte.

## The layer rule

Doors translate; they hold no registration logic. They import the core, the
job layer, the operations, the agent driver (`agent/`, the same layer: the
CLI's `linear run` and the MCP server start its engine) and the providers
(a door resolves a provider name); never a host (`hosts/`). Only the ADK
packaging (`tools/media.py`) imports `google.*`. import-linter's layers
contract (`pyproject.toml`, `tests/test_import_layers.py`) checks it, with
no exceptions listed. The host commands (`abba`, `serve`) live in
`hosts/cli.py`; `cli/__init__.py` names them by module path
(`HOST_COMMANDS`) and imports that module when it builds the parser, never
statically.
`tests/test_core_imports.py` loads `declarations`, `jobs`, `library`, `card`,
`cli`, `cli.job`, `tools.toolbox` and `tools.view_options` in a fresh
interpreter and checks, and runs `import langslice;
langslice.open_job(...)` with a verb or two. An operation (`ops/`) never
imports a door.

Two sub-packages moved in with the folder move (2026-10-04), each described
in the linear agent environment's guide (`src/langslice/agent/CLAUDE.md`):
`tools/` (formerly in `linear/` and `adk/`): `toolbox.py` (the tool
bodies), `arguments.py`, `view_options.py`, `media.py` (the ADK message
parts) and, in `__init__.py`, the media keys re-exported from
`core/media_keys.py`; `mcp/` (formerly
`mcp_server/`): the MCP server (`server.py`, `prompt.py`,
`host_channel.py`; `connectors/claude-desktop/`). And `api/` (formerly in
`hosts/api/`, moved down 2026-10-04 because the MCP door and the CLI use
it and none of it drives a host): `models.py` (the engine contract's
Pydantic models, `export_schema_bundle`), `runtime.py` (`register.run`,
`quick_affine.run`, `export.run`, `nonlinear register`'s runtime),
`setup.py` (offline setup status, saved credentials, login, and
`image_model_connected(provider)`: the provider is not `none` and its key
or login is present, an offline presence check the MCP door and Claude
mode ask before offering `trace_borders`),
`claude_jobs.py` (saved Claude jobs: the id index and the host channel)
and `abba_worker.py` (the JVM-free linear snapshot worker:
`prepare_linear`, `checkpoint_callback` with its ABBA-world host rows,
`run_linear`, `preview_preprocess`). The engine service and the ABBA
plugin's `nonlinear.abba` worker stay in `hosts/api/`.

## Files

- `declarations.py` — each verb's declaration: a stub function per verb
  (signature = the arguments, docstring = the description a model reads).
  `model_doc` re-indents the docstring to the eight spaces the tool closures
  gave it, so models read the same bytes as before phase 5. `Variant`
  (`traces`, `preprocessing`, `engine`, `auto`; `Variant.of(spec, auto=)`)
  is what of a run changes a declaration: `view` typed `ViewAuto` (with
  `resolution`) where the caller sizes pictures; `fit_deformable`'s
  description without the image model (`_STAIN_ONLY_DOC`), with
  `preprocess` (`_PREPROCESS_DOC`), and with the engine fixed (no `engine`
  argument, candidates `FixedCandidate`, `_RECOMMENDED_TRACED` dropped
  unless ANTs). `declaration(name, variant)` (cached), `summary(name)`,
  `declare(name, body, variant)` (a function with the declared name, doc
  and signature that binds the call, fills the declared defaults and hands
  every argument to the body by name; ADK's `tool_context` is added when
  the body takes it; a body lacking a declared argument is a `TypeError`),
  `arguments_schema(name, variant)` (pydantic JSON schema, unknown keys
  refused). `FULL` is the variant of a caller without a job.
- `jobs.py` — opening a job without the agent: `JobContext` (the
  workspace plus the job folder and results path; the driver's
  `EngineContext` adds only the model), `find(path)` (a job folder, or the
  image folder beside one; `NoJob`), `read_spec` (`job.json`'s spec, as a
  resume), `open_folder(path, persist=)` (`Job.load`: nothing rewritten;
  the card brought up to date; the model keys loaded by
  `api.setup.load_credentials`, `.env` then the keys saved by setup, the one
  loader the CLI's `main` uses too), `create(spec)` (`Job.open`: the ingest
  every host uses; the card), `Opened.tools()` (the toolbox with
  `gates=False`, `level="auto"`, `scripting=True`, `max_view_edge` `OPEN_MAX_VIEW_EDGE`:
  no model's cap, the source's pixels bound every picture), `Opened.close`
  (image corrections settled, pictures flushed).
- `library.py` — `open_job(folder, atlas_loader=, emit=)` -> `JobHandle`:
  every verb the job has as a method (the tool itself: same arguments,
  the reply dict with plain PIL pictures under `images`, saved like every
  door's; the scripting verbs too), `verbs`, `folder`, `job`, `state`,
  `workspace`, `close`, a
  context manager. `langslice/__init__.py` exposes `open_job`,
  `coordinate_map` (`core.layers`) and `load_atlas` (`core.atlas.core`), each
  imported on first use.
- `card.py` — the job folder's reference card, `AGENTS.md` and
  `CLAUDE.md` (identical; Codex reads one, Claude Code the other):
  `card_text(layout)` (one screen: the folder's files, one line each for
  `registration.json` and each section's maps, state as truth and the
  rest derived, the coordinate map and convention, the CLI with
  every verb from the registry, the Python entry point), `write_card`
  (writes where missing or worded differently; never raises). Written by
  every door that opens or makes a job: the CLI and the library
  (`jobs.open_folder`, `jobs.create`), the agent run (`engine.run`), the MCP
  door (`open_job`) and a saved Claude job (`doors.api.claude_jobs._write_job`).
- `cli/` — every `langslice` command, one module per group;
  `langslice/cli.py` keeps the entry point `langslice.cli:main`.
  `__init__.py` (`build_parser`, `main`: the agent commands return their
  exit code), `linear.py` (`linear run`, `linear quick-affine`, and the job
  flags every stack-opening command shares: `add_linear_arguments`,
  `build_linear_spec`), `register.py` (`nonlinear register`), `claude.py`
  (`mcp`, `claude prepare`), the host commands by module path
  (`HOST_COMMANDS`: `abba`, `serve` in `hosts/cli.py`), and the agent CLI
  (`docs/agent_cli.md`):
  - `envelope.py` — `Envelope` (`ok`, `result`, `artifacts`, `warnings`,
    `next`, `error` {code, message, fix}), `EXIT_OK` 0, `EXIT_ARGUMENTS` 2,
    `EXIT_REFUSED` 3, `EXIT_INTERNAL` 4; `exit_code(code)` (`BAD_*`,
    `UNKNOWN_*`, `TOO_MANY_*` and `ARGUMENT_CODES` are 2, every other
    refusal 3); `FIXES` per code; `stdout_to_stderr()` (Python and native
    stdout to stderr while a verb runs, so stdout holds the envelope only).
  - `catalog.py` — `langslice ops` (verbs: name, kind, group, summary; the
    job commands) and `langslice schema [VERB] [--job FOLDER]`
    (`SCHEMA_VERSION` 1; `canonical_verb`: kebab-case accepted).
  - `job.py` — `langslice job FOLDER VERB`: `execute` (never raises),
    `parse` (`--args`, `--name value`, `--dry-run`, `--background`,
    `--verbose`, `--timeout`, the child's `--run-id`), `arguments_for`
    (flags read as the verb declares them; `argument_refusal`, missing
    arguments, `normalize_arguments`), `call` (open, `VERB_OFF`, background
    start, dry run, run), `_run` (the tool inside `job.views.captured()`;
    `trace_borders` settled before answering; `submit` writes the results;
    pictures flushed and listed as artifacts; `would_change` from the state
    before and after on a job that writes nothing), `shape` (concise:
    no `description`, a write's whole-stack `rows` as `n_rows`; verbose:
    everything and the picture texts), `changes`, `init` (the job flags of
    `linear run`, `jobs.create`), `runs` (`runs [ID]`, `wait [ID]`; `status`
    is only the verb).
    `CHECKED_ONLY`: `trace_borders` and `fit_deformable` are checked, not
    run, by `--dry-run`. After `submit` the derived files
    (`job.formats.derived_files`) and after `export_maps` the files it
    wrote are listed as artifacts by kind.
  - `background.py` — `--background`: `start` (a record in
    `logs/runs/<id>.json`, then `CHILD_COMMAND` + `job FOLDER VERB --args
    ... --run-id ID` detached, stderr in `<id>.log`), `begin` / `finish`
    (in the child: its pid; its envelope and exit), `read` (a running run
    whose process is gone, or that never started within `START_GRACE_S`,
    is `lost`; the process is probed with a null signal, on Windows with
    psutil when installed, else `OpenProcess`/`GetExitCodeProcess`), `listing`,
    `latest`, `wait` (without a timeout it returns once the run is lost).

## Live shared editing

Each CLI call and each `open_job` opens the job as it stands (`Job.load`),
and every tool call runs `Job.sync` first, so an agent run and CLI calls on
one folder see each other's writes, history included
(`tests/test_agent_cli.py` interleaves them). Every write holds the job
folder's lock (`job/lock.py`, `Job.writing`: lock, sync, apply, commit):
the tool door wraps every verb in it, except the long ones
(`VERBS[name].long`: `fit_affine`, `fit_deformable`, `trace_borders`,
`export_maps`),
which compute outside it and take it to apply, refusing a section whose
inputs changed (`ops.inputs`, `STALE_INPUT`); `job.lock` timing out is
`JOB_BUSY` (exit 3). `tests/test_job_concurrency.py`: an agent write during
a fit survives, a moved section is refused, concurrent CLI processes all
land.
