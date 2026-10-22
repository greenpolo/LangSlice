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
| agent CLI (`langslice-job`) | `cli/job.py` over the same toolbox, gates off, `level="auto"`, pictures capped at the job's viewer's (`jobs.job_viewer`), `trace_borders` only when the image model is connected (`Opened.image_model_connected`), plus the scripting verb `export_maps`, `brief`, `env` and `python SCRIPT`; each call answered by the job's warm process when it is up (`cli/warm.py`) | Claude Code, Antigravity, Codex |
| library | `library.py` (`langslice.open_job`, `langslice.create_job`) over the same toolbox, the scripting verbs included; `job.context` (`ScriptContext`) for a method of the script's own | a script |

A job whose spec names tool plugins (`JobSpec.plugins`) has their tools after
the native ones on every door: the toolbox (agent and MCP), `ops` / `schema`
(for the job in the folder), the card and the library's `verbs` read the
per-spec view (`registry.verbs_for(spec)`).
A job without plugins declares, replies and draws exactly as before.

The text snapshot (`tests/snapshots/agent_text/`) pins what the doors
declare.

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
interpreter and runs `import langslice; langslice.open_job(...)` and
`create_job` with a script's own image model. An operation (`ops/`) never imports a door.

What every door shares beyond the verbs: the job statement
(`statement.py`), the ending of a job (`jobs.close_job`), the host trace
(`trace.py`), the reply byte budget and paging (`tools/reply.py`), the
image-model check (`jobs.provider_connected`, `jobs.image_model_off`), and in
the `ToolBox` the opening-read gate (`require_opening`, `opening_read`, `opening_refusal`;
armed by the MCP door only). Deliberately different: the host owns the loop
for MCP and the CLI (no turn budget, nudges, debrief, image working set or
quota accounting there); pictures are files and artifacts in the CLI, inline elsewhere;
each CLI call is a process of its own (cold, or the child the job's warm
process forks for it, `cli/warm.py`).

## Files

- `declarations.py` — each verb's declaration: a stub function per verb
  (signature = the arguments, docstring = the description a model reads,
  re-indented by `model_doc`). The description is the tool's only
  description: the job statement names the tools and describes none.
  `Variant` (`auto`, `door`, `uncapped`, `forced_view`, `positions`,
  `prompt`, `brain_scale`, `absent` (the tools a task that is off leaves out: plugin
  stubs render by it, native text does not); `Variant.of(spec, auto=, image_model=, door=, prompt=,
  uncapped=, brain_scale=)`) is
  what of a run changes a declaration (`dropped_arguments`, each dropped
  argument's `Args` entry cut from the description): `off` (the job's
  disabled list): the modes it disables are out of `look`'s `mode` enum
  (`narrowed`), the arguments only those modes served (`atlas_opacity`,
  `warp`, `positions_mm`, `atlas_layers`, `zoom`'s `region`) are dropped,
  `interactive_transform`'s sections lose `flip` and `rotate_quarter`
  (`SectionTransformNoFlip`, `NoRotate`, `NoOrientation`),
  and every passage of a description written with `[[?needs ...]]` marks
  (`core/needs.py`) is dropped or swapped; `look`'s `resolution`
  only where the caller sizes pictures; the change tools' `view`
  (`VIEW_TOOLS`) not offered where the host forces their pictures on
  (`JobSpec.force_view`); `position_sections`' `sections` only with the
  Positioning task; `trace_borders` carries the run's base image prompt
  (the image model's profile prompt, else the job's own
  `nonlinear.image_prompt`, else `base_prompt`; none without an
  image model) in place of `BASE_PROMPT`; and the door that reads it
  (`DOORS`: `agent`, `mcp`, `cli`; `_DOOR_DOCS`: the CLI's background
  verbs answer once their work has landed); `uncapped` (the CLI and the
  library, `build_tools(scripting=True)`): `_UNCAPPED_DOCS` words the
  picture and section counts of `look`, `grep_atlas_view`,
  `make_image`, `make_mask`, `interactive_transform` and
  `elastix_affine` without the ADK and MCP cap; `brain_scale` (the job's
  atlas brain's, `core.atlas.size`, passed by the toolbox and the CLI's
  `schema` for a job): `_SCALED_DOCS` states the distances a description
  gives (`elastix_affine`'s 300 um neighbourhood) at that scale, unchanged at 1. `declaration(name, variant)`
  (cached), `summary(name)`, `declare(name, body, variant)` (a function with
  the declared name, doc and signature that binds the call, fills the
  declared defaults and hands every argument to the body by name; ADK's
  `tool_context` is added when the body takes it; a body lacking a declared
  argument is a `TypeError`), `arguments_schema(name, variant)` (pydantic
  JSON schema, unknown keys refused), `argument_texts(doc)` (each argument's
  `Args` entry on one line). `FULL` is the variant of a caller
  without a job (LangSlice's coronal base prompt for the OpenAI models).
- `plugins/` — tool plugins (`docs/plugins.md`); `langslice/plugin_api.py`
  re-exports its names. `loader.py`: `load(ref, config)` imports
  `module:factory`, calls it, checks every tool (`core.plugins.check_tool`),
  builds the provenance (module, factory, versions, source
  SHA-256s) and registers the `LoadedPlugin` with the core (`install()` runs
  at import); `as_refs` (the library's objects: factory functions by
  `module:qualname`, ToolSpecs as `inline:<digest>`).
  `context.py`: `Caller` (who a context works for: `Caller.of_tool(spec)` a
  plugin tool, labels `plugin:<tool>`; `Caller.script()` a script's
  `job.context`, label `script`, every property writable the job leaves free,
  no `work_dir` / `start_work` (`NOT_FOR_SCRIPTS`), `in_tool_call` False: its
  reads sync the job first and its placement writes drop the deformations
  they made stale, `merge_calls` False: each outermost write one undo step;
  `provenance`: a deformation record's `plugin` or `script` entry, the latter
  with `argv0`); `ToolContext(door, caller, call)` (reads `sections`, `section_image`, `atlas`,
  `fit_frame` (`placed_atlas_to_section`: the current placement; `image=` a
  section image as `section_image` with `section_image_ref`), `atlas_slice`,
  `section_inputs`, `channels` (a section's raw channel names), `image_names`
  (its section images), `working_grid`,
  `read_registration` (each section in its current orientation, `params`, `at=` a
  fixed plane, `registered_on="oriented"` for a file made on the oriented
  render); `section_raster(images=, raw=)`;
  conversions `transform_from_matrix` / `transform_from_params`, and
  `deformation_from_map` (a full fit-grid-to-native map as a
  `DeformationProposal`, `ops.deformable.warp_from_map`); writes
  `set_positions`, `set_transforms`, `set_deformations`
  (`ops.deformable.apply_warps`, maps converted before the lock),
  `mark_damage` through `ops/`, `add_image` (a made section image,
  `ops.images.add_image`), `add_layer`
  through `job/plugin_store` (a script's under `plugins/script/`), `expect=` fingerprints refusing `STALE_INPUT`
  rows; `writing()` holds the lock around several writes; a plugin call's writes are
  one undo step unless another write came between; `atlas_plane` /
  `atlas_volume` (`AtlasPlane`: a plane at any placement), `to_atlas_um` /
  `from_atlas_um` / `plane_distance_um` / `region_at` (`ops.points`),
  `write_frame` / `map_from_itk` / `transform_from_itk` (`core.itk_io`: an
  outside engine's inputs and answer), `fit_fireants` / `fit_elastix`
  (`ops.deformable.fit_record`, `ops.transforms.elastix_fit`: the native
  engines with settings and a mask of the caller's own, a proposal not
  written, the settings in its `meta` / the record's `fit`); `overlay_picture` and
  `positioning_picture` (a proposal drawn on a copy of the state), with a
  deformation proposal drawn through an in-memory record store) and
  its views. `runtime.py`: the body of a plugin tool (limits, the context, a
  write tool's call as one undo step, `ToolError` / `Refused` / any exception
  as a reply, the reply and pictures from the `ToolResult`). `provenance.py`:
  `job.json`'s `plugins` and the declarations as the model sees them (also
  `ToolBox.plugins`, the trace's `plugins` record). `mcp.py`: the generic MCP
  adapter plugin (LangSlice as a client of a configured server, allow-list,
  `server__tool` names, `inputs` / `results` mapping through the context).
  `declarations.Variant.plugins` makes `declaration` read a plugin stub;
  `look` takes `layers` on every job (plugins and scripts leave them; dropped
  without the `section` and `overlay` modes).
- `statement.py` — the job statement every door gives a registration agent:
  `job_statement(spec, state, ctx, door=, tool_names=, opening=, notes=,
  max_resolution=, image_model_off=, auto=)` (`agent.prompt.build_job_statement`
  worded for the door, the door's opening paragraph, `IMAGE_MODEL_OFF`
  (`image_model_off_text(disabled)`) when the image model is not connected,
  the user's notes, `status_and_notes`).
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
  taking `image_model=`. `Opened.tools()`
  (the toolbox with `level="auto"`, `scripting=True`,
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
  `with_registration(spec, file, atlas_loader=, emit=)`: the spec
  with a registration made elsewhere as its supplied inputs
  (`job.imports.registration_inputs`) and the import report; refuses a spec
  that already supplies any of `REGISTRATION_EXCLUDES`. Every door that
  takes `--registration` / `registration=` goes through it.
- `library.py` — `open_job(folder, image_model=, atlas_loader=, emit=)` ->
  `JobHandle`: every verb the job has as a method (the tool, same
  arguments, run inside `job.views.captured()`, then the pictures flushed,
  so a method returns once its pictures are on disk), `verbs`, `folder`, `image_model`, `job`, `state`, `workspace`,
  `imported`, `wait(timeout=None)` (waits for background work and returns its landing as the CLI does:
  `background` notices, `work`, `running`, pictures; uses `toolbox.work_pictures`), `context` (one `ScriptContext` per handle, on the toolbox's
  `door`: the plugin `ToolContext` with `Caller.script()`; each outermost
  write gets a call id `script-<pid>-<n>`, a conversion before it shares the
  write's, and a landed write a `script_write` line in `logs/events.jsonl`;
  `preview_overlay` / `preview_positioning` draw a proposal through
  `overlay_picture` / `positioning_picture`, caption it `NOT written` and save
  it as tool `script` with a `transport_snapshot` recipe, flushed before they
  return `SavedPicture`s (a linear proposal's overlay keeps its frame and
  layers, a deformation's none); its docstring is the how-to the card points
  to), `section_maps(section, full_resolution=)` (a section's maps in
  memory, `ops.exports.section_maps`), `export_stack()` (what `submit`
  writes after its step, `ops.exports.export_stack`), `close` (image-model calls settled, pictures flushed), a
  context manager. A method's reply is `library_reply`: a `Reply` (a dict
  of JSON values, `plain`; its status rows `core.status.with_uniform_rows`;
  the pictures' files under `artifacts` as the CLI lists them,
  `job.views.artifacts`; their text lines under `media_texts`; an unsaved
  picture under `warnings`) whose `images` ATTRIBUTE holds the PIL
  pictures. Every handle's job is tracked (`_track`) and closed at
  interpreter exit by `_close_open`, registered with `job.views.at_exit`
  (before `concurrent.futures` stops taking work), so a script that never
  calls `close` keeps its `trace_borders` results and pictures. `create_job(images | JobSpec, atlas=, plane=, tasks=,
  image_model=, job_dir=, positions=, transforms=, angles=,
  orientation=, pixel_size_um=, inputs=, registration=, fresh=, **JobSpec
  fields)`: six-number transforms become `{"kind": "interactive", "params",
  "mirrored"}` (`_transform`); `angles` in either `inputs.angles` form
  (`_angles`); `registration=` a file made elsewhere (`tasks` None is then
  `["nonlinear"]`); otherwise `tasks` None is `default_tasks`
  (`nonlinear`, plus `transform` unless every section has a transform).
  `image_model` None is provider `none`; else the model's provider (`custom`
  for a model of the caller's own) and `job.json` records the profile
  (`profile_record`). `open_job` of a job whose record says untested,
  without `image_model=`, sets `Opened.traces_off`. `langslice/__init__.py`
  exposes `open_job`, `create_job`, `image_model`, `default_prompt`,
  `coordinate_map` and `load_atlas`, each imported on first use.
- `card.py` — the job folder's reference card, `AGENTS.md` and `CLAUDE.md`
  (identical; Codex reads one, Claude Code the other): `card_text(layout)`
  (the folder's files, the maps and their coordinate convention, the CLI
  with every listed verb, `long` marked and its per-call limits
  (`limits_note` of `registry.Verb.limits`: `fireants_syn`'s and a plugin's;
  the CLI and the library cap no pictures or transform sections, so the card
  states none; `transform_cap_line`: the job's
  own `transform.max_parallel` when lower; less the verbs its `disabled` list
  names, `disabled_names`), how a long verb's call answers (within 30 s, its
  run going on), `picture_files`, `FOLDER schema VERB [VERB ...]`, `schema
  init`, the
  Python entry point and what a method returns, and to keep the agent's own
  files in `scripts/` and `scratch/` of the job folder, not `/tmp`; one line
  pointing a method of the agent's own to `help(job.context)`; first,
  run `brief` and read `BRIEF.md`; under 90 lines, `tests/test_agent_cli.py`),
  `write_card` (writes where missing or worded differently; never raises).
  Written by every door that opens or makes a job.
- `trace.py` — `TRACE_DIR_ENV` (`LANGSLICE_TRACE_DIR`), `HostTrace` (the MCP
  door's trace: one JSON line per record, images as descriptors; one file
  per session), `cli_trace(job_folder, trace_dir)` (the agent CLI's: one
  file per job folder) and `log_call` (`logs/calls.jsonl`).
- `tools/` — the tool door: `toolbox.py` (`build_tools`, `ToolBox`, which
  keeps the `Door` it built as `door`; sets
  `Job.picture_cap` to `MAX_IMAGES_PER_CALL` (4), or to None with
  `scripting=True` (the CLI and the library); every tool wrapped,
  innermost first, by the opening gate (writes), `_strict`,
  for a write `_clears_stale_deformations` and `_names_registrations`
  (`registrations_named`: `registration` `{section: number}`, the registrations
  the call's writes left in this thread, `Job.registered_since`, for the
  sections whose status row in the reply does not show it; a reply with a
  `registration` of its own keeps it), `_saves_views` (the pictures a body returns
  saved and numbered, their numbers and captions under `pictures`; a reply
  that already lists `pictures`, saved by its operation, is left as it is),
  `_announces_work` (`with_notices`: the notices of background work
  finished since the last reply first, under `background`, and its pictures
  after the reply's own) and `_serialized`; `_tool_target_ids`;
  `with_job_prompt`: the image model with the job's own base prompt,
  `nonlinear.image_prompt`, as an untested `custom` profile), `door.py`
  (`Door`, what every tool body of a run shares: the job, the workspace,
  the picture level, the image model, `rows`, `changed`,
  `answered`, `resolve_many`; `pictured`, `as_list`, `spread`: the sections
  a change tool pictures when the call names none), the
  tool bodies (`looking.py`: `look` (`images`: one section image alone, or
  raw channels together; empty draws `preprocessed`), `zoom`, `make_image`
  (its `section` pictures through `ops.look.show_result`, none where
  `look:section` is off, which also drops its `view`), `grep_atlas`,
  `grep_atlas_view`, `status` (`raw_channels`, `section_images`:
  `core.section_images.summary`), the job-folder tools;
  `changing.py`: `position_sections`, `interactive_transform`,
  `mark_damage`, `make_mask` (per-section `rows`), `note`, `undo`, `redo` (`steps`, default 1: that many in
  one call, `ops.history`), `restore_registration` (0 lists,
  `ops.registrations.listing`; a number puts it back and shows the
  section's overlay), `submit`; `fitting.py`:
  `elastix_affine` (its `turn_deg` and plausibility `flags` worded in the
  row's `warning`, `affine_flag_words`), `fireants_syn`, `trace_borders`,
  `export_maps`; a change tool's picture is drawn after its write by
  `ops.look.show_result` unless its `view` is false and the host does not
  force it (a fit's and an `em` mask's picture drawn on the section image
  it read, `door.drawn_image`); `interactive_transform` and `elastix_affine` drop the
  deformations their write made stale before drawing it, so the picture
  shows the section as it now stands; a restricted fit's picture
  highlights its `restrict_to` regions and zooms to them only past
  `ops.transforms.MIN_ZOOM_GAIN`; `with_notices` lists each background
  picture with its caption; `work_pictures` is the saved files of those pictures, for the CLI and `JobHandle.wait`), `arguments.py` (the argument shapes, typed dicts with
  `extra="forbid"`: `SectionPosition`, `CuttingAngles`, `SectionTransform`;
  `argument_refusal`, the one strictness rule every door
  applies; `normalize_arguments` for a door that validates first),
  `view_options.py` (`image_limit` / `view_edge_limit` of the model lane,
  `clamp_resolution` of `look`'s `resolution`), `media.py` (the ADK message parts: `packaged`,
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
  `host_tool` (each call serialized on the session's lock), `strict_arguments` (FastMCP drops unknown arguments; refused
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
  Pydantic models, `export_schema_bundle`),
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
  job's `host.abba`, `damaged`, `channel_names`, `preprocessing`; what the
  agent may not change is the host's `spec.frozen`, its names checked against
  the snapshots, and no host row is ever emitted for a property frozen on a
  section, `_host_updates(..., frozen=spec.frozen_for)`), `checkpoint_callback` (a
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
  commands, `session-end`, `login`, `version`), `linear.py` (`linear run` and the job flags every
  stack-opening command shares: `add_linear_arguments` (`--disable
  NAME[,NAME...]`: the job's `disabled` list), `build_linear_spec`,
  `spec_from_args` (`--frozen JSON`: the job's `frozen`); `--tasks` defaults to `DEFAULT_TASKS`, or with
  `--registration FILE` to `REGISTRATION_TASKS`; `--registration` with any
  of `REGISTRATION_CLASHES` is refused; a bad flag value ends the command
  with a message; `linear run` alone also takes `--tool-plugin
  MODULE:FACTORY` and `--tool-plugin-config REF=JSON`, `with_tool_plugins`),
  `mcp.py` (`mcp`), and the agent CLI
  (`docs/agents.md`), a command of its own so that allowing it allows
  no agent, server, host or login:
  - `jobcli.py` — `langslice-job` (entry point `langslice.doors.cli.jobcli:main`,
    also `python -m langslice.doors.cli.jobcli`): `main()` without argv
    (the process's own command line) asks the job's warm process first
    (`warm.run`), importing nothing else until that is decided (the
    package's `__init__` imports its command groups where it builds its
    parser); `run(argv)`, the call in this process (a cold call, a warm
    child's): `ops` and `schema` first (run inside `stdout_to_stderr`,
    their envelope printed after), anything else `FOLDER VERB`; returns the
    exit code.
  - `warm.py` — the warm process (`docs/agents.md`, "The warm process"):
    one fork server per job folder and Python environment (`identity`:
    interpreter, flags, import path, LangSlice's folder), standard library
    only on the client's side. Client `run(argv)`: `job_folder` (not
    `init`, `ops`, `schema`), `request` (argv, cwd, environment, `envkey`,
    the streams' encodings and buffering), the fds of stdin/stdout/stderr
    by `SCM_RIGHTS`, `ready` / `go` / `started` / `exit` or `signal`;
    `FORWARDED` signals passed on; None (run cold) before a call starts, and
    `start_server` (detached, `BOOT`; not while one holds `<key>.lock`, nor
    for `REFUSED_S` after one could not serve). `runtime_dir`
    (`$XDG_RUNTIME_DIR/langslice-warm`, else `/tmp/langslice-warm-<uid>`,
    0700), `Place` (`.sock`, `.lock`, `.log`, `.refused`). Server
    `serve_main`: the lock, `warm_up` (the CLI's modules,
    `core.deformable.engines.preload`, the job opened read-only, its tools,
    atlas arrays, `core.maps.share_frames`, the oblique template volumes when
    its sections are tilted; `_quiet_pools` stops dask's, zarr's,
    fsspec's and the sampling threads; `gc.freeze`), `Server` (one thread,
    `selectors`; `refusal`: protocol, identity, `Sources.changed` (the
    loaded first-party modules, the import path's folders, the atlas
    folder: `stale`, it stops listening and exits when its calls end),
    `ENV_PREFIXES` / `HOME` (`envkey`); a fork per call, its process group
    killed when the client goes (`SIGKILL`) or signals; `reap`; idle
    `check` (stale code, a gone job, frames shared again when `state.json`
    changed, unless the job is now submitted, `_submitted`); `IDLE_S` (10
    min), `SUBMITTED_IDLE_S` (60 s) once the job is submitted). Child `_child`: `setpgid`, `PR_SET_PDEATHSIG`, the
    fds on 0-2, cwd, environment, `sys.argv` / `sys.path` / `sys.executable`,
    `_streams`, `_fresh_process` (tempdir, tz, logging, `formats.source`,
    numpy's seed), `jobcli.run`, `as_python` (exit codes, tracebacks,
    Ctrl-C as Python gives them), `_end` (threads, `atexit`, flush,
    `os._exit`). `in_child()`. `run_script(interpreter, rest)`: the job
    environment's python asked for its path (`PROBE`), `_cannot_run_here`
    (another Python, or a package of the environment's own already loaded
    here: then `os.execv` that python), its `.pth` files (not
    `PARENT_PTH`), `_run_main` (a fresh `__main__`, `__file__` absolute,
    `sys.argv[0]` as given).
  - `envelope.py` — the `FIXES` texts are written with `[[?needs ...]]`
    marks and rendered for the job's disabled list (`DISABLED`, set by
    `job.call` once the job is open);
    `Envelope` (`ok`, `result`, `artifacts`, `warnings`,
    `next`, `error` {code, message, fix}; printed with `picture_files`, the
    paths of the artifacts of `PICTURE_KINDS`, second when there are any),
    `EXIT_OK` 0, `EXIT_ARGUMENTS` 2,
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
    `long`; `JOB_COMMANDS`) and `schema [VERB ...] [--job FOLDER]`
    (`SCHEMA_VERSION` 3; `canonical_verb` accepts kebab-case, `_names`
    comma-separated lists; per verb
    `Declared.entry`: `summary`, `description`, `kind`, `group`, `long`,
    `arguments`; one verb's entry as the result, several (or every) verbs'
    under `verbs`; declared for the
    job given or of the current folder, only its verbs (`Opened.listed_verbs`;
    another answers `VERB_OFF`), else `FULL` with a `hint`; a name that is no
    verb answers `UNKNOWN_VERB`, which without a job says how to name one (its
    plugin verbs are known only with it); a job
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
    read as the verb declares them; an unknown one refused with
    `corrected_call` as the fix: every argument with the first sentence of its
    `Args` entry, `declarations.argument_texts`, and its default, then the
    call without the unknown ones, a short signature's others at their
    defaults), `FOLDER schema [VERB ...]` (catalog's `schema` for that job),
    `call` (open, `VERB_OFF` /
    `IMAGE_MODEL_OFF`, `--dry-run` with `--background` refused, background
    start; a long verb without `--background` where `background.detaching()`
    runs as a background run the call waits for, `_detached`: at most
    `background.WAIT_S`, then answered as `wait` answers, Ctrl-C ending the run
    too; dry run, run), `_run` (the tool inside `job.views.captured()`; an
    image-model verb's calls settled before answering, each landed outcome
    shown; `submit` writes the results; pictures listed as artifacts with
    `index` and `label`, `job.views.artifacts`; an image-model verb waits
    for the background work it started and answers with its notices, the
    work's pictures listed after the call's own, `_work_pictures`;
    `would_change` on a dry run), `shape` (status rows uniform,
    `core.status.with_uniform_rows`;
    concise: a reply with pictures keeps its description as `picture_note`,
    a write's whole-stack `rows` as `n_rows`), `changes`, `init` (its
    parser `init_parser`: the job flags of `linear run`, `--notes`,
    `--viewer`; `--registration`'s report under `result.registration`,
    `BAD_REGISTRATION` when the file cannot be read or places nothing),
    `brief`, `runs` (`runs [ID]`, `wait [ID]`: `--timeout`, default
    `background.WAIT_S` (30 s), then `run_answer`: the run's own envelope, or
    while it runs `ok` with its state, `waited_s`, a message and the command
    that waits again (exit 0), or `RUN_LOST`), and `env` (`jobenv.command`). `CHECKED_ONLY`:
    `trace_borders`, `elastix_affine` and `fireants_syn`
    are checked, not run, by `--dry-run`, and so is every plugin tool (a verb not in `VERBS`).
  - `jobenv.py` — `env` (`langslice-job FOLDER env [install PACKAGE ...|remove]`):
    the job's own Python environment `env/` (uv's or `venv`'s), a `.pth`
    (`PARENT_PTH`) adding the site-packages of the environment LangSlice runs
    in, so its python imports LangSlice; `install` (uv pip, else pip) and
    `scripts/requirements.txt` (`REQUIREMENTS`: the environment's own
    packages, pinned, read from its `*.dist-info`); `remove` keeps that list,
    from which `ensure` rebuilds; `ensure` makes it once under
    `logs/env.lock` (`ENV_LOCK`), however many calls ask together, and marks
    it made (`env/.langslice-ready`, `READY`). `ENV_FAILED` when a step fails.
    `python(folder, rest)` — `langslice-job FOLDER python SCRIPT [ARGS ...]`
    (handled by `job.run` before any parsing, the script's arguments
    verbatim; `FOLDER python --help` answers as a job command;
    `script_path`: a relative SCRIPT the working folder does not hold is the
    job folder's, made absolute): the
    environment made when missing, the call logged, then the script in this
    warm child (`warm.run_script`) or this process replaced by the
    environment's python (`os.execv`; `subprocess.call` on Windows); a call
    that cannot start prints its refusal's envelope (`NO_JOB`,
    `BAD_ARGUMENTS`, `ENV_FAILED`).
  - `transcripts.py` — `langslice session-end --harness claude-code|antigravity`,
    the hook of the Claude Code (`SubagentStop`) and Antigravity (`Stop`)
    plugins: the hook's JSON on stdin, the transcript (`transcript_path`), the
    job folders it names (`job_folders`: `langslice-job FOLDER` and
    `open_job(...)`, through `jobs.find`), each given the transcript as
    `trace/<harness>_<session>.jsonl` with inline image data replaced by its
    size (`slim`); a transcript that submitted (`SUBMIT_PATTERN`) also removes
    the job's `env/`. Never raises; prints `{}`.
  - `background.py` — `--background`: `start` (a record in
    `logs/runs/<id>.json`, then `CHILD_COMMAND` (`jobcli` as a module, so
    the job's warm process answers it too) + `FOLDER VERB --args ...
    --run-id ID` detached, stderr in `<id>.log`), `begin` / `finish` (in the
    child), `read` (a running run whose process is gone, or that never
    started within `START_GRACE_S`, is `lost`; the process probed with a null
    signal, on Windows with psutil when installed, else
    `OpenProcess`/`GetExitCodeProcess`), `listing`, `latest`, `wait`
    (`WAIT_S`: what `wait` waits without `--timeout`, and a detached long
    call, well inside the time a harness gives a command), `cancel` (the run's
    process sent `SIGTERM`), `detaching()` (`DETACH_ENV`, `LANGSLICE_DETACH`:
    `0` no, `1` yes; unset, yes unless `CLAUDECODE` is set: Claude Code's
    sandbox ends what a command leaves running).

## Live shared editing

Each CLI call and each `open_job` opens the job as it stands (`Job.load`;
a warm call too, in its forked child),
and every tool call runs `Job.sync` first, so an agent run and CLI calls on
one folder see each other's writes, history included
(`tests/test_agent_cli.py` interleaves them). Every write holds the job
folder's lock (`job/lock.py`, `Job.writing`: lock, sync, apply, commit): the
tool door wraps every verb in it, except the long ones (`VERBS[name].long`:
`elastix_affine`, `fireants_syn`, `trace_borders`,
`export_maps`), which compute outside it and take it to apply, refusing a
section whose inputs changed (`ops.inputs`, `STALE_INPUT`), and the reads
(`kind` "read"), which never wait for another process's writer
(`Job.reading`: the state as last saved); the lock timing
out is `JOB_BUSY` (exit 3). Every call runs as its verb (`Job.acting`), so the
registrations its writes leave are listed as made by it. `tests/test_job_concurrency.py`: an agent write
during a fit survives, a moved section is refused, concurrent CLI processes
all land.

## Landmark tool declarations

`set_landmarks` and `landmark_triangulation` are ordinary Nonlinear write verbs
on all doors. Typed `LandmarkPair` and `LandmarkPoint` arguments expose explicit
atlas/section endpoints and x,y PIXELS in a referenced picture; nested unknown
keys are checked recursively. `set_landmarks` defaults to one contextual
side-by-side template/preprocessed crop for each edited pair (at most four
with `view=True`; larger batches can set `view=False`); application shows the
resulting overlay, at `look`'s size (`landmark_triangulation` takes `resolution`
where `look` does: `dropped_arguments` drops both together; `TOO_MANY_LANDMARK_VIEWS`
says how many pairs were sent and what to resend; the landmark codes' `FIXES`
are specific, written with needs marks; `ATLAS_ENDPOINT_OUTSIDE`, an argument
code (exit 2), resends the pairs at the inside point its message gives). `grep_atlas`'s `points` (`PicturePoint`) is
the option `grep_atlas:points`. The CLI reads a list of numbers from one
comma-separated flag (`--box 960,230,1300,940`) and returns `grep_atlas`'s rows
unshortened (`ROW_VERBS`). `status`' `can_change` names the
landmark tools when they are on. Both honor forced
views and undo/checkpoint normally. `undo` / `redo` answer with `undone` /
`redone` (sections, fields), the rows of those sections and the depth, not the
whole table; with `steps` above 1, also `steps` taken and, when the history
held fewer, a `message`.

MCP reply-budget shrinking saves each delivered JPEG with a distinct immutable
picture number and resized endpoint geometry. `extra.transport_source_picture`
names the original. Look recipes retain native redraw; other pictures become
`transport_snapshot` recipes. Returned picture lists (including batch results)
refer to the delivered IDs; landmark input provenance retains its original IDs.
