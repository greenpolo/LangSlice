# LangSlice `doors/` — one vocabulary, every door

Package guide for `src/langslice/doors/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

A verb (an agent tool) has ONE name, one argument list and one description;
every door is generated from them and from the registry (`ops/registry.py`:
`VERBS`, `enabled(spec)`):

| Door | Built by | Driver |
|---|---|---|
| agent tools (ADK) | `tools/toolbox.py` `build_tools`: the verbs `enabled(spec)` names, each `declarations.declare`d | LangSlice's agent (`agent/`) |
| MCP tools | the same toolbox (`mcp/server.py`, `door="mcp"`), plus the door's `start_job` and `show_stack`; `readOnlyHint` on the read verbs; `trace_borders` only when the job's image model is connected (`jobs.image_model_off`); every reply within `tools.reply.REPLY_BYTES`; the opening-read gate armed by `start_job` | Claude Desktop, Claude Code locked to it |
| agent CLI (`langslice-job`) | `cli/job.py` over the same toolbox, gates off, `level="auto"`, pictures capped at the job's viewer's (`jobs.job_viewer`), `trace_borders` only when the image model is connected (`Opened.image_model_connected`), plus the scripting verbs (`export_maps`, the hidden `trace_from_atlas`) and `brief` | Claude Code, Codex |
| library | `library.py` (`langslice.open_job`, `langslice.create_job`) over the same toolbox, the scripting verbs included; `pipeline.py` (`register_section`, `register_job`) calls those verbs in a fixed order | a script |

A hidden verb (`registry.Verb.hidden`: `trace_from_atlas`, the
placement-free image-model trace kept for experiments) is built by the
scripting doors and called by name (`langslice-job FOLDER trace_from_atlas`,
`job.trace_from_atlas(...)`), but listed nowhere: not in `langslice-job ops`,
`langslice-job schema` without a verb (`schema trace_from_atlas` answers by
name), the card, the library's `verbs`, the CLI's `verbs` lists, the job
statement, nor the public docs (`registry.listed()`).

A verb is never renamed once shipped (scripts and agents call it by name).
The goldens (`tests/golden/linear_tools/*_declarations_*`) pin what the ADK
and MCP doors declare, byte for byte.

## The layer rule

Doors translate; they hold no registration logic. They import the core, the
job layer, the operations, the agent driver (`agent/`, the same layer: the
CLI's `linear run` and the MCP server start its engine) and the providers (a
door resolves a provider name); never a host (`hosts/`). Only the ADK
packaging (`tools/media.py`) imports `google.*`. import-linter's layers
contract (`pyproject.toml`, `tests/test_import_layers.py`) checks it, with no
exceptions listed. The host commands (`abba`, `serve`) live in
`hosts/cli.py`; `cli/__init__.py` names them by module path
(`HOST_COMMANDS`) and imports that module when it builds the parser.
`tests/test_core_imports.py` loads the doors' agent-free modules in a fresh
interpreter and runs `import langslice; langslice.open_job(...)` and a
scripted `register_section`. An operation (`ops/`) never imports a door.

What every door shares beyond the verbs: the job statement
(`statement.py`), the ending of a job (`jobs.close_job`), the host trace
(`trace.py`), the reply byte budget and paging (`tools/reply.py`), the
image-model check (`jobs.provider_connected`, `jobs.image_model_off`), and in
the `ToolBox` the in-flight rule (`in_flight`, `begin_model_call`) and the
opening-read gate (`require_opening`, `opening_read`, `opening_refusal`;
armed by the MCP door only). Deliberately different: the host owns the loop
for MCP and the CLI (no turn budget, nudges, debrief, image working set or
quota accounting there); the look-before-commit gates are the agent and MCP
tools' only; pictures are files and artifacts in the CLI, inline elsewhere;
the CLI is a process per call.

## Files

- `declarations.py` — each verb's declaration: a stub function per verb
  (signature = the arguments, docstring = the description a model reads,
  re-indented by `model_doc`). `Variant` (`traces`, `preprocessing`,
  `engine`, `auto`, `door`; `Variant.of(spec, auto=, door=, image_model=)`)
  is what of a run changes a declaration: the door that reads it (`DOORS`:
  `agent`, `mcp`, `cli`; `_DOOR_DOCS`: where `view_placement` says the
  section is, and the CLI's trace verbs, which answer once their calls have
  landed); `view` typed `ViewAuto` (with `resolution`) where the caller
  sizes pictures; `fit_deformable`'s description without the image model
  (`_STAIN_ONLY_DOC`), with `preprocess` (`_PREPROCESS_DOC`), and with the
  engine fixed (no `engine` argument, candidates `FixedCandidate`,
  `_RECOMMENDED_TRACED` dropped unless ANTs). `declaration(name, variant)`
  (cached), `summary(name)`, `declare(name, body, variant)` (a function with
  the declared name, doc and signature that binds the call, fills the
  declared defaults and hands every argument to the body by name; ADK's
  `tool_context` is added when the body takes it; a body lacking a declared
  argument is a `TypeError`), `arguments_schema(name, variant)` (pydantic
  JSON schema, unknown keys refused). `FULL` is the variant of a caller
  without a job.
- `statement.py` — the job statement every door gives a registration agent:
  `job_statement(spec, state, ctx, door=, tool_names=, opening=, notes=,
  max_resolution=, image_model_off=, auto=, gates=)` (`agent.prompt.build_job_statement`
  worded for the door, the door's opening paragraph, `IMAGE_MODEL_OFF` when
  the image model is not connected, the user's notes, `status_and_notes`).
  `opening_for_mcp` (the `show_stack` pages), `opening_for_cli` (the saved
  picture files, `--dry-run`, `--background` for `long_verbs()`, parallel
  calls). `status_and_notes(state)` (the status table and the newest
  `RECENT_NOTES` run notes) is the ADK seed message's text too.
  `read_notes(layout)`: the user's notes, `job.json` `notes` (written by
  ABBA's Claude mode and `job FOLDER init --notes`), read by every door. `image_model_state(spec, connected=)`: the
  image model as the CLI's `status` and `brief` report it.
- `jobs.py` — opening a job without the agent: `JobContext` (the workspace
  plus the job folder and results path; the driver's `EngineContext` adds
  the model), `find(path)` (a job folder, or the image folder beside one;
  `NoJob`), `read_spec` (`job.json`'s spec, as a resume), `open_folder(path,
  persist=, door=)` (`Job.load`: nothing rewritten; the card brought up to
  date; the model keys loaded by `api.setup.load_credentials`),
  `create(spec)` (`Job.open`, the ingest every host uses; the card), both
  taking `image_model=` and writing no card for a lean job. `Opened.tools()`
  (the toolbox with `gates=False`, `level="auto"`, `scripting=True`,
  `max_view_edge` the viewer's or `OPEN_MAX_VIEW_EDGE` for a script; one per
  open job), `Opened.listed_verbs()`, `Opened.image_model_connected` (a model
  handed in, else `provider_connected(spec)`; never with
  `Opened.traces_off`), `Opened.close` (`close_job`: image corrections
  settled, pictures flushed; every door ends a job through it).
  `provider_connected(spec)`: the spec names no image model, or its
  provider (never `custom`, a script's own) has its key or login here.
  `image_model_off(spec, connected)`: the nonlinear task names an image
  model that is not connected. `door` (`agent`, or `cli`: its declarations
  worded for the CLI) and `viewer` (the CLI's: `job_viewer(layout)`,
  `job.json` `viewer`, default `claude`; `core.opening.VIEWER_LIMITS`).
  `with_registration(spec, file, target=, atlas_loader=, emit=)`: the spec
  with a registration made elsewhere as its supplied inputs
  (`job.imports.registration_inputs`) and the import report; refuses a spec
  that already supplies any of `REGISTRATION_EXCLUDES`. Every door that
  takes `--registration` / `registration=` goes through it.
- `library.py` — `open_job(folder, image_model=, atlas_loader=, emit=)` ->
  `JobHandle`: every verb the job has as a method (the tool, same
  arguments, run inside `job.views.captured()`, then the pictures flushed,
  so a method returns once its pictures are on disk; the hidden verb by
  name), `verbs`, `folder`, `image_model`, `job`, `state`, `workspace`,
  `imported`, `close` (image-model calls settled, pictures flushed), a
  context manager. A method's reply is `library_reply`: a `Reply` (a dict
  of JSON values, `plain`; its status rows `core.status.with_uniform_rows`;
  the pictures' files under `artifacts` as the CLI lists them,
  `job.views.artifacts`; their text lines under `media_texts`; an unsaved
  picture under `warnings`) whose `images` ATTRIBUTE holds the PIL
  pictures. Every handle's job is tracked (`_track`) and closed at
  interpreter exit by `_close_open`, registered with `job.views.at_exit`
  (before `concurrent.futures` stops taking work), so a script that never
  calls `close` keeps its `trace_borders` results and pictures. `create_job(images | JobSpec, atlas=, plane=, tasks=,
  image_model=, job_dir=, output=, positions=, transforms=, angles=,
  orientation=, pixel_size_um=, inputs=, registration=, fresh=, **JobSpec
  fields)`: six-number transforms become `{"kind": "interactive", "params",
  "mirrored"}` (`_transform`); `angles` in either `inputs.angles` form
  (`_angles`); `registration=` a file made elsewhere (`tasks` None is then
  `["nonlinear"]`); otherwise `tasks` None is `pipeline_tasks`
  (`nonlinear`, plus `transform` unless every section has a transform).
  `image_model` None is provider `none`; else the model's provider (`custom`
  for a model of the caller's own) and `job.json` records the profile
  (`profile_record`). `open_job` of a job whose record says untested,
  without `image_model=`, sets `Opened.traces_off`. `langslice/__init__.py`
  exposes `open_job`, `create_job`, `image_model`, `default_prompt`,
  `register_section`, `register_job`, `RegistrationError`, `coordinate_map`
  and `load_atlas`, each imported on first use.
- `pipeline.py` — the scripted nonlinear registration (`docs/library.md`).
  `register_job(job, sections=, affine_method=, fit=, full_resolution=,
  arrays=)`: per section `fit_affine` where no transform, `trace_borders`
  when the job has the verb, `fit_deformable` applied (`TRACED_FIT`: traced
  lines, Elastix, medium, which does not depend on the optional ANTs
  install; `STAIN_FIT` without an image model; `FIT_BATCH` 4 per call), then
  `submit` (with a problem, `export_maps`); a failure kept per section in
  `problems`. `register_section(image, position_mm=, ...)`: the section
  linked, copied or (an array) written as a TIFF into a folder of its own
  (the folder checked first: other section images there, or a different
  file under the section's name, are refused and nothing is written), then
  `create_job(fresh=True)` and `register_job`; `RegistrationError` on a
  problem. `RegistrationResult` / `SectionOutput` (paths, trace,
  `untested`, `problem`, `read()` the maps as arrays).
- `card.py` — the job folder's reference card, `AGENTS.md` and `CLAUDE.md`
  (identical; Codex reads one, Claude Code the other): `card_text(layout)`
  (the folder's files, the maps and their coordinate convention, the CLI
  with every listed verb, `long` marked and its per-call limits
  (`limits_note` of `registry.Verb.limits`; `transform_cap_line`: the job's
  own `transform.max_parallel` when lower), `schema init` and `--help`, the
  Python entry point and what a method returns, and to keep the agent's own
  files in `scripts/` and `scratch/` of the job folder, not `/tmp`; first,
  run `brief` and read `BRIEF.md`; under 85 lines, `tests/test_agent_cli.py`),
  `write_card` (writes where missing or worded differently; never raises).
  Written by every door that opens or makes a job.
- `trace.py` — `TRACE_DIR_ENV` (`LANGSLICE_TRACE_DIR`), `HostTrace` (the MCP
  door's trace: one JSON line per record, images as descriptors; one file
  per session), `cli_trace(job_folder, trace_dir)` (the agent CLI's: one
  file per job folder) and `log_call` (`logs/calls.jsonl`).
- `tools/` — the tool door: `toolbox.py` (`build_tools`, `ToolBox`: the tool
  bodies, each wrapped by `_serialized`, `_strict`, `_saves_views` and
  `_clears_stale_deformations`; see `agent/CLAUDE.md`), `arguments.py` (the
  argument shapes, typed dicts with `extra="forbid"`; `argument_refusal`, the
  one strictness rule every door applies; `normalize_arguments` for a door
  that validates first), `view_options.py` (`parse_view` against each
  tool's `Profile`, `image_limit` / `view_edge_limit` of the model lane,
  `clamp_resolution`), `media.py` (the ADK message parts: `packaged`,
  `package_result`, `opening_parts`), `reply.py` (`REPLY_BYTES` 680 KB,
  `fit_reply` shrinks a reply's pictures together, `paged`, `strip_bytes`,
  `shrunk_note`; no model framework) and, in `__init__.py`, the media keys
  from `core/media_keys.py`.
- `mcp/` — the MCP server (`server.py`; `connectors/claude-desktop/`):
  `open_job` (the job opened as the engine does, the toolbox at
  `CLAUDE_MAX_VIEW_EDGE`, `trace_borders` only when the image model is
  connected), `Session`, `briefing` and `opening_pages` (the opening strips
  at `CLAUDE_IMAGE_LIMIT`, each composed within a page's byte budget, paged
  under `PAGE_BYTES`, a strip and its text kept together), `save_page`,
  `result_blocks` (every reply through `fit_reply`; a shrunk reply says so),
  `host_tool` (each call `in_flight`, so a sibling call promotes no picture
  as seen), `strict_arguments` (FastMCP drops unknown arguments; refused
  first, a nested object sent as a JSON string parsed), `open_saved_job`
  (a saved ABBA job by id), `open_folder` (the job of a named image or job
  folder as saved: `doors.jobs.find` / `read_spec`, its notes; a folder
  without one gets a new job from the server's job flags), `build_server`,
  `serve`. `EventRelay` forwards a saved ABBA job's
  tool events (and one `seed` per `show_stack` page) over its host channel
  (`host_channel.py`) as `agent_event`s; `tool_end` events carry `views`,
  the saved pictures' paths.
- `api/` — what the MCP door, the CLI and the engine service share (the
  service itself is `hosts/api/`): `models.py` (the engine contract's
  Pydantic models, `export_schema_bundle`), `runtime.py` (`version`),
  `setup.py` (offline setup status, saved credentials, login;
  `load_credentials`: `.env`, then the keys saved by setup, an explicit
  environment setting winning, an unknown or unreadable entry skipped with a
  warning; `image_model_connected(provider)`: provider not `none` and its
  key or login present, nothing contacted; `image_model_choices()` /
  `IMAGE_MODEL_CHOICES`: the dialog's image models, each `connected` or not,
  with the models to offer), `saved_jobs.py` (saved ABBA jobs,
  `prepare_saved_job` for the engine method `mcp.prepare`: the job folder
  next to the snapshots, the id index, the host channel, the copy prompt) and
  `abba_worker.py` (the JVM-free snapshot worker of the Fiji connector:
  `prepare_linear` (snapshots, BrainGlobe AP positions, `angles_deg` as the
  stack-wide `inputs.angles`, `z_offset_mm` / `existing_warp` kept as the
  job's `host.abba`, `locked`, `damaged`, `existing_warp` & `locked` ->
  `inputs.keep_warp`, `nonlinear_skip` -> `inputs.nonlinear_skip`,
  `channel_names`, `preprocessing`), `checkpoint_callback` (a
  `HostCheckpoints` tracker: ABBA-world linear rows; with the `nonlinear`
  task `warp` rows from `core/abba_warp.py`, each logged with its measured
  `max_error_mm` / `p99_error_mm`; `host_angles` on a stack-wide angle
  change), `run_linear` (ends with a checkpoint of the final state),
  `public_event`, `preview_preprocess`). ABBA shows one atlas angle per
  stack, so the tracker refuses a state whose sections differ in cutting
  angle (`refuse_mixed_angles`, `ABBA_MIXED_ANGLES`), which also fails the
  MCP door's opening of such a saved ABBA job.
- `cli/` — the `langslice` and `langslice-job` commands, one module per
  group; `langslice/cli.py` keeps the entry point `langslice.cli:main`.
  `__init__.py` (`build_parser`, `main`: `linear run`, `mcp`, the host
  commands, `login`, `version`), `linear.py` (`linear run` and the job flags every
  stack-opening command shares: `add_linear_arguments`, `build_linear_spec`,
  `spec_from_args`; `--tasks` defaults to `DEFAULT_TASKS`, or with
  `--registration FILE` to `REGISTRATION_TASKS`; `--registration` with any
  of `REGISTRATION_CLASHES` is refused; a bad flag value ends the command
  with a message), `mcp.py` (`mcp`), and the agent CLI
  (`docs/agent_cli.md`), a command of its own so that allowing it allows
  no agent, server, host or login:
  - `jobcli.py` — `langslice-job` (entry point `langslice.doors.cli.jobcli:main`,
    also `python -m langslice.doors.cli.jobcli`): `ops` and `schema` first
    (run inside `stdout_to_stderr`, their envelope printed after),
    anything else `FOLDER VERB`; returns the exit code.
  - `envelope.py` — `Envelope` (`ok`, `result`, `artifacts`, `warnings`,
    `next`, `error` {code, message, fix}), `EXIT_OK` 0, `EXIT_ARGUMENTS` 2,
    `EXIT_REFUSED` 3, `EXIT_INTERNAL` 4; `exit_code(code)` (`BAD_*`,
    `UNKNOWN_*`, `TOO_MANY_*` and `ARGUMENT_CODES` are 2, every other refusal
    3); `FIXES` per code; `stdout_to_stderr()` (stdout holds the envelope
    only).
  - `brief.py` — `brief` (and `init`'s statement): `build(opened, pictures=)`
    -> `Brief`: the statement for door `cli` (no gates, `auto` sizing),
    the opening strips saved as the job's `opening` views (artifacts with
    `index` and `label`), the facts (`viewer`, `resolution`, `image_model`),
    all written to `BRIEF.md`.
  - `catalog.py` — `ops` (the listed verbs: name, kind, group, summary,
    `long`; `JOB_COMMANDS`) and `schema [VERB] [--job FOLDER]`
    (`SCHEMA_VERSION` 2; `canonical_verb` accepts kebab-case; per verb
    `Declared.entry`: `summary`, `description`, `kind`, `group`, `long`,
    `arguments`, and for a picture verb `picture_options`; declared for the
    job given or of the current folder, else `FULL` with a `hint`; a job
    that cannot be read answers `JOB_UNREADABLE`; a job command,
    `command_names()`, answers `command_entry`: summary, usage and `flags`,
    `init`'s read off `job.init_parser` by `flags_table`). Both return the
    envelope. `describe(name, folder)`: what `FOLDER NAME --help` answers,
    `schema NAME` declared for FOLDER's job when it has one.
  - `job.py` — `langslice-job FOLDER VERB`: `execute` (never raises; every
    call logged by `record` and, with `LANGSLICE_TRACE_DIR`, traced; any
    `HELP_FLAGS` token, `--help` / `-h`, answers `catalog.describe` and runs
    nothing),
    `parse` (`--args`, `--name value`, `--dry-run`, `--background`,
    `--verbose`, `--timeout`, the child's `--run-id`), `arguments_for` (flags
    read as the verb declares them), `call` (open, `VERB_OFF` /
    `IMAGE_MODEL_OFF`, `--dry-run` with `--background` refused, background
    start, dry run, run), `_run` (the tool inside `job.views.captured()`; an
    image-model verb's calls settled before answering, each landed outcome
    shown; `submit` writes the results; pictures listed as artifacts with
    `index` and `label`, `job.views.artifacts`; `would_change` on a dry
    run), `shape` (status rows uniform, `core.status.with_uniform_rows`;
    concise: a reply with pictures keeps its description as `picture_note`,
    a write's whole-stack `rows` as `n_rows`), `changes`, `init` (its
    parser `init_parser`: the job flags of `linear run`, `--notes`,
    `--viewer`; `--registration`'s report under `result.registration`,
    `BAD_REGISTRATION` when the file cannot be read or places nothing),
    `brief`, `runs` (`runs [ID]`, `wait [ID]`). `CHECKED_ONLY`:
    `trace_borders`, `trace_from_atlas` and `fit_deformable` are checked,
    not run, by `--dry-run`.
  - `background.py` — `--background`: `start` (a record in
    `logs/runs/<id>.json`, then `CHILD_COMMAND` (`jobcli` as a module) + `FOLDER VERB --args ...
    --run-id ID` detached, stderr in `<id>.log`), `begin` / `finish` (in the
    child), `read` (a running run whose process is gone, or that never
    started within `START_GRACE_S`, is `lost`; the process probed with a null
    signal, on Windows with psutil when installed, else
    `OpenProcess`/`GetExitCodeProcess`), `listing`, `latest`, `wait`.

## Live shared editing

Each CLI call and each `open_job` opens the job as it stands (`Job.load`),
and every tool call runs `Job.sync` first, so an agent run and CLI calls on
one folder see each other's writes, history included
(`tests/test_agent_cli.py` interleaves them). Every write holds the job
folder's lock (`job/lock.py`, `Job.writing`: lock, sync, apply, commit): the
tool door wraps every verb in it, except the long ones (`VERBS[name].long`:
`fit_affine`, `fit_deformable`, `trace_borders`, `trace_from_atlas`,
`export_maps`), which compute outside it and take it to apply, refusing a
section whose inputs changed (`ops.inputs`, `STALE_INPUT`); the lock timing
out is `JOB_BUSY` (exit 3). `tests/test_job_concurrency.py`: an agent write
during a fit survives, a moved section is refused, concurrent CLI processes
all land.
