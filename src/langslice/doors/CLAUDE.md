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
| agent CLI | `cli/job.py` over the same toolbox, gates off, `level="auto"`, plus the scripting verbs (`export_maps`, and the hidden `trace_from_atlas`) | Claude Code, Codex |
| library | `library.py` (`langslice.open_job`, `langslice.create_job`) over the same toolbox, the scripting verbs included; `pipeline.py` (`register_section`, `register_job`) calls those verbs in a fixed order | a script, a scripted pipeline |

A hidden verb (`registry.Verb.hidden`: `trace_from_atlas`, the
placement-free image-model trace kept for experiments) is built by the
scripting doors and called by name (`langslice job FOLDER
trace_from_atlas`, `job.trace_from_atlas(...)`), but listed nowhere: not
in `langslice ops`, `langslice schema` without a verb (`schema
trace_from_atlas` answers by name), the card, the library's `verbs` or
`dir()`, the CLI's `verbs` lists, nor the public docs
(`registry.listed()`).

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
`tests/test_core_imports.py` loads `declarations`, `jobs`, `library`,
`pipeline`, `providers.profiles`, `card`, `cli`, `cli.job`, `tools.toolbox`
and `tools.view_options` in a fresh interpreter (this tree's `src/` first on
the path) and checks, and runs `import langslice; langslice.open_job(...)`
with a verb or two, then a scripted `register_section` with a model of the
script's own. An operation (`ops/`) never
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
Pydantic models, `export_schema_bundle`), `runtime.py` (`version`,
`export.run`; `register.run` and the `nonlinear register` command, a
one-shot pipeline outside the job, and `quick_affine.run` with `langslice
linear quick-affine`, a one-shot silhouette alignment outside the job
(the job verb `fit_affine` covers it), were removed 2026-10-04),
`setup.py` (offline setup status, saved credentials, login, and
`image_model_connected(provider)`: the provider is not `none` and its key
or login is present, an offline presence check the MCP door and Claude
mode ask before offering `trace_borders`),
`claude_jobs.py` (saved Claude jobs: the id index and the host channel;
`prepare_folder` refuses a folder whose checkpoint was made from other
inputs, `job.refuse_changed_inputs`, and with `--fresh` marks the job
`fresh` so `mcp.server.open_folder_job`'s first open starts it over and
clears the mark)
and `abba_worker.py` (the JVM-free linear snapshot worker:
`prepare_linear`, `checkpoint_callback` with its ABBA-world host rows,
`run_linear`, `preview_preprocess`; ABBA shows one atlas angle per stack,
so `checkpoint_callback` refuses a state whose sections differ in cutting
angle before emitting anything (`refuse_mixed_angles`,
`ABBA_MIXED_ANGLES`: the job was made elsewhere with an angle per section,
which ABBA cannot show), which also fails `run_linear`'s final checkpoint
and the MCP door's opening of a saved ABBA job (`server.open_saved_job`);
`refuse_mixed_job(spec)` refuses differing supplied per-section angles, and
a resumed checkpoint that has them, for the Python launcher. The worker's
own jobs are flat). The engine service stays in
`hosts/api/`.

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
  every host uses; the card), both taking `image_model=` (kept on
  `Opened.image_model`) and writing no card for a lean job, `Opened.tools()`
  (the toolbox with `gates=False`, `level="auto"`, `scripting=True`,
  `max_view_edge` `OPEN_MAX_VIEW_EDGE`: no model's cap, the source's pixels
  bound every picture; `image_model` handed to `build_tools`; without one a
  `custom`-provider job, or one with `Opened.traces_off`, gets
  `image_model_connected=False`: no `trace_borders`), `Opened.close`
  (image corrections settled, pictures flushed). `with_registration(spec,
  file, target=, atlas_loader=, emit=)`: the spec with a registration made
  elsewhere as its supplied inputs (`job.imports.registration_inputs` on
  the spec's workspace, `context`) and the import report, each warning
  said through `emit`; refuses a spec that already supplies any of
  `REGISTRATION_EXCLUDES` (positions, transforms, angles, orientation).
  Every door that takes `--registration` / `registration=` goes through it.
- `library.py` — `open_job(folder, image_model=, atlas_loader=, emit=)` ->
  `JobHandle`: every verb the job has as a method (the tool itself: same
  arguments, the reply dict with plain PIL pictures under `images`, saved
  like every door's; the scripting verbs too, the hidden one by name),
  `verbs` (the listed ones), `folder`,
  `image_model`, `job`, `state`, `workspace`, `close`, a context manager.
  `create_job(images | JobSpec, atlas=, plane=, tasks=, image_model=,
  job_dir=, output=, positions=, transforms=, angles=, orientation=,
  pixel_size_um=, inputs=, registration=, fresh=, **JobSpec fields)`: the job of a folder
  through `jobs.create` (keys loaded as `open_job` loads them), the same
  handle; supplied transforms as six numbers become `{"kind":
  "interactive", "params", "mirrored"}` (`_transform`); `angles` in either
  `inputs.angles` form, the stack's or per section (`_angles`, checked by
  `core.spec.supplied_angles`); `registration=` a file made elsewhere
  (`jobs.with_registration`; not with the four placement arguments; tasks
  None is then `["nonlinear"]`; the report on `JobHandle.imported`);
  otherwise `tasks` None is
  `pipeline_tasks` (`nonlinear`, plus `transform` unless every section has
  a transform); `image_model` None is provider `none`, else the model's
  provider (`custom` for a model of the caller's own) and `job.json`
  records the profile under `image_model` (`profile_record`). `open_job` of
  a job whose record says untested, without `image_model=`, sets
  `Opened.traces_off`. `as_image_model` normalizes every `image_model=`
  through `providers.profiles.image_model`. `langslice/__init__.py` exposes
  `open_job`, `create_job`, `image_model`, `default_prompt`
  (`providers.profiles`), `register_section`, `register_job`,
  `RegistrationError` (`pipeline`), `coordinate_map` (`core.layers`) and
  `load_atlas` (`core.atlas.core`), each imported on first use.
- `pipeline.py` — the scripted nonlinear registration on the job layer
  (`docs/library.md`). `register_job(job, sections=, affine_method=, fit=,
  full_resolution=, arrays=)`: per section `fit_affine` where no transform,
  `trace_borders` when the job has the verb, `fit_deformable` applied
  (`TRACED_FIT`: traced lines, Elastix, medium; `STAIN_FIT` without an image
  model; `FIT_BATCH` 4 per call), then `submit` (or, with a problem,
  `export_maps`); each step checked on the state, a failure kept per
  section in `problems`. `register_section(image, position_mm=, ...)`: the
  section linked, copied or (an array) written as a TIFF into a folder of
  its own (`_place_image`; other sections there refused), `create_job` (its
  `pitch_deg`/`yaw_deg` that section's own angles)
  (`fresh=True`), `register_job`; `RegistrationError` on a problem.
  `RegistrationResult` / `SectionOutput` (paths, trace, `untested`,
  `problem`, `read()` the maps as arrays).
- `card.py` — the job folder's reference card, `AGENTS.md` and
  `CLAUDE.md` (identical; Codex reads one, Claude Code the other):
  `card_text(layout)` (one screen: the folder's files, one line each for
  `registration.json` and each section's maps, state as truth and the
  rest derived, the coordinate map and convention, the CLI with
  every listed verb from the registry, `registry.listed()`, the Python
  entry point), `write_card`
  (writes where missing or worded differently; never raises). Written by
  every door that opens or makes a job: the CLI and the library
  (`jobs.open_folder`, `jobs.create`), the agent run (`engine.run`), the MCP
  door (`open_job`) and a saved Claude job (`doors.api.claude_jobs._write_job`).
- `cli/` — every `langslice` command, one module per group;
  `langslice/cli.py` keeps the entry point `langslice.cli:main`.
  `__init__.py` (`build_parser`, `main`: the agent commands return their
  exit code), `linear.py` (`linear run` and the job
  flags every stack-opening command shares: `add_linear_arguments`,
  `build_linear_spec`; `--tasks` defaults to `DEFAULT_TASKS`, or with
  `--registration FILE` to `REGISTRATION_TASKS` (`nonlinear`);
  `spec_from_args` refuses `--registration` with any of
  `REGISTRATION_CLASHES`, and `build_job_spec` imports the file through
  `jobs.with_registration` and returns its report), `claude.py`
  (`mcp`, `claude prepare`), the host commands by module path
  (`HOST_COMMANDS`: `abba`, `serve` in `hosts/cli.py`), and the agent CLI
  (`docs/agent_cli.md`):
  - `envelope.py` — `Envelope` (`ok`, `result`, `artifacts`, `warnings`,
    `next`, `error` {code, message, fix}), `EXIT_OK` 0, `EXIT_ARGUMENTS` 2,
    `EXIT_REFUSED` 3, `EXIT_INTERNAL` 4; `exit_code(code)` (`BAD_*`,
    `UNKNOWN_*`, `TOO_MANY_*` and `ARGUMENT_CODES` are 2, every other
    refusal 3); `FIXES` per code; `stdout_to_stderr()` (Python and native
    stdout to stderr while a verb runs, so stdout holds the envelope only).
  - `catalog.py` — `langslice ops` (the listed verbs: name, kind, group,
    summary; the job commands) and `langslice schema [VERB] [--job FOLDER]`
    (`SCHEMA_VERSION` 1; `canonical_verb`: kebab-case accepted; every
    listed verb, or one verb by name, a hidden one included).
  - `job.py` — `langslice job FOLDER VERB`: `execute` (never raises),
    `parse` (`--args`, `--name value`, `--dry-run`, `--background`,
    `--verbose`, `--timeout`, the child's `--run-id`), `arguments_for`
    (flags read as the verb declares them; `argument_refusal`, missing
    arguments, `normalize_arguments`), `call` (open, `VERB_OFF`, background
    start, dry run, run; a hidden verb is callable, `VERB_OFF` lists only
    the listed ones), `_run` (the tool inside `job.views.captured()`;
    an image-model verb's calls (`Verb.image_model`: `trace_borders`,
    `trace_from_atlas`) settled before answering, each landed outcome
    shown; `submit` writes the results;
    pictures flushed and listed as artifacts; `would_change` from the state
    before and after on a job that writes nothing), `shape` (concise:
    no `description`, a write's whole-stack `rows` as `n_rows`; verbose:
    everything and the picture texts), `changes`, `init` (the job flags of
    `linear run`, `jobs.create`; with `--registration` the import report
    under `result.registration` and its warnings as the envelope's,
    `BAD_REGISTRATION` when the file cannot be read, matched one to one or
    places nothing), `runs` (`runs [ID]`, `wait [ID]`; `status`
    is only the verb).
    `CHECKED_ONLY`: `trace_borders`, `trace_from_atlas` and
    `fit_deformable` are checked, not run, by `--dry-run`. After `submit` the derived files
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
`trace_from_atlas`, `export_maps`),
which compute outside it and take it to apply, refusing a section whose
inputs changed (`ops.inputs`, `STALE_INPUT`); `job.lock` timing out is
`JOB_BUSY` (exit 3). `tests/test_job_concurrency.py`: an agent write during
a fit survives, a moved section is refused, concurrent CLI processes all
land.
