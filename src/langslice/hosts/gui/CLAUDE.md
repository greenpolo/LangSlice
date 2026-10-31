# LangSlice `hosts/gui/` — the LangSlice window

Package guide for `src/langslice/hosts/gui/`. The repo-level `CLAUDE.md` holds
the project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror
to the other. The user's page is `docs/developers/gui.md`.

One local window, titled "LangSlice", with three tabs: Agent viewer, Prompt
generator and Settings (top right). The agent viewer is a preliminary layout in the
ABBA log's styling (its palette and type tokens now live in `static/app.css`,
kept after the log's deletion; the comparison window's blue-grey picture
panel); Nash designs the final one.

## The rule

The GUI is a separate process from every run (the built-in agent, an MCP
server, the agent CLI's calls, a SliceBench subprocess), so it learns
everything by watching files: no socket, no relay, no import of the run. It
imports the core (`core.stack.stack`, `core.stack.status`, `core.atlas.details`,
`core.connections`, `core.stack.json_reads`)
and the job layer (`job.settings.tool_table`, `job.settings.spec`,
`job.settings.schema`, `job.settings.labels`, `job.layout` names and
`read_job_file`, `job.migrations.viewer_stack` and `without_removed_settings`,
`job.history`), the providers' image-model lists (`providers.connections`,
`providers.openai_api`), and for the Settings tab the doors' setup
(`doors.credentials`: key presence, `save_api_key`, `login_oauth`;
`image_model_connected` for the generator) and the
user's settings file (`doors.user_settings`). Nothing imports it but
`hosts/cli.py` (`langslice gui`); `doors/commands/linear.py`'s `--gui` starts that
command as a process (`open_gui`). The viewer and the prompt generator are
read-only: every job and generator endpoint is GET only, nothing is written to
a job folder. The Settings tab's three POSTs are the only writes (the user's
settings file, a saved key, the ChatGPT login); sign-in runs in this window's
process, never inside an agent tool call. Nothing in it is model-facing: the
prompt generator's message is the person's to send, and its labels
(`job.settings.labels`) are for people.

## Settings

Connections (`openai-oauth` signed in by the browser flow; `openai-api`,
`anthropic-api`, `gemini-api` keys in `~/.langslice/provider_credentials.json`,
an Anthropic key also counted from `~/.langslice/anthropic_api_key`), one chosen;
"Registration runs in" (`app` / `langslice`), greyed until the chosen
connection is present, `langslice` until the user picks; the chosen
connection's agent model and reasoning. Saved in `doors/user_settings.py`'s file
(`~/.langslice/settings.json`, `LANGSLICE_SETTINGS` another file or `off`;
the test suite sets it off). `runs_in` is read by NOTHING yet: acting on it is
Phase 5 (model-facing, awaiting Nash's approval). The agent entry IS read:
`langslice linear run` with neither `--model` nor `--reasoning` runs on the
chosen connection's saved model and effort (`apply_saved_agent_choice`); no
file, no connection or no saved model leaves its default. Choosing a
connection saves LangSlice's default model for it where there is one
(`doors.user_settings.agent_models`: the ChatGPT account's from
`providers.connections`, the same model for the OpenAI API key; Claude and
Gemini keys list the models `doors.agent.spend.PRICES` prices, with no default).

## Files

- `__init__.py` — `serve(folders, port=, window=, tab=)`: the job list
  (the recent jobs file, `job.recent.recent_file()`), the server, the window
  on `window.start_address` (`langslice gui --tab viewer|prompt`); runs until Ctrl-C or until
  the app window's process has been gone for `LINGER_S` (5 s); a tab or no
  window serves until Ctrl-C.
- `watch.py` — `JobWatcher(folder)`: per job, `_Tail`s (byte offsets, an
  unterminated last line kept for the next round, a file grown shorter starts
  the watch over: a new `generation`) on `logs/events.jsonl`, `views.jsonl` and
  the newest `trace/*.jsonl`; `_Whole`s (size and mtime, one retry on a file
  caught mid-rename) on `state.json`, `history/undo.json`, `job.json`,
  `exports/spend.json`. `poll` at most every `POLL_S` (0.25 s), run by a
  page's snapshot request, so only watched jobs are read. One round's trace
  records and events are interleaved by time (`_in_time_order`: an event's
  `at` is UTC, the trace's local; within a second a model turn before the
  calls it made, a session's end after them), then the pictures. Items: `tool`
  (`tool_start` / `tool_end` by `execution_id`; `args` an object or the cut
  JSON string), `picture` (number, view folder, tool, call, sections, caption
  from `view.json`; a path outside the folder is refused), `agent` and
  `reasoning` (the trace's `model` `text` / `thought`), `note` (open, plugins,
  script writes, session start and end, a trace error). The sections table:
  `core.stack.status.status_rows` of `job.migrations.viewer_stack` (an older
  format migrated in memory as every door opens it, nothing written) with
  each section's registration number (`job.history.SectionHistories`); a state this
  version does not read (a newer or a pre-release format) shows its raw
  `slices` fields with a note. Undo labels: `_label_step` follows each `tool_end`'s `step` (the undo
  depth after the call): a call that deepens it names the new depth (a first
  event of any tool but a read, `job.settings.tool_table` kind `read`, names
  it too; a plugin's or an older version's tool counts as a write); `undo` /
  `redo` move names between the sides; a depth no event names is `Step n`. `status`:
  Submitted (`state.submitted`), Failed (a trace `error`, or a `summary`
  without a submit), Running (a `tool_start` without its end, or a file
  changed within `RECENT_S`), else Waiting, and a sentence. `snapshot(since,
  generation, state_version)`: the items after the cursor (all of them for
  another generation), the header, the status, and the tables only when
  `state_version` moved. `JobList`: the folders given, then the recent-jobs
  file's (`folder` of each line, newest first, existing folders, at most
  `MAX_RECENT`), indexed in the order added; `listing` peeks at `job.json`
  and `state.json` only. The recent jobs file is the job layer's
  (`job/recent.py`: `~/.langslice/recent_jobs.jsonl`, or
  `LANGSLICE_RECENT_JOBS`, another file or `off`).
- `server.py` — `GuiServer(jobs, port=, settings=)`: `ThreadingHTTPServer` on
  127.0.0.1, every path under a random token, the `Host` header checked
  against the port, `HEADERS` (CSP, `no-store`, `nosniff`, no referrer).
  `answer(path, host)` (GET): the
  static files, `jobs`, `generator` (`generator.payload()`; `?job=<n>` adds
  `prefill`, `generator.prefill` of that listed job as plain data, null
  without a `job.json`, 404 for an index not listed),
  `jobs/<n>/snapshot?since=&gen=&sv=`,
  `jobs/<n>/picture/<number>` (the `view.jpg` of a picture the job's
  `views.jsonl` lists, read from disk), `settings`; anything else 404.
  `post(path, headers, body)`: only `settings.POSTS` (any other path 405),
  with the token and `Host`, `Origin` exactly `http://127.0.0.1:<port>` and
  `Sec-Fetch-Site` same-origin when sent (403), `Content-Type`
  `application/json` (415), at most `MAX_BODY` bytes (413); a refusal is 400
  `{"error"}`. PUT, DELETE, PATCH, HEAD: 405. The Settings code in `server.py`,
  `index.html` and `app.css` sits between `Settings tab` marker comments.
- `generator.py` — the prompt generator's choices, `payload()` (`GET
  generator`): `tools()` (`tool_table.catalog()` without the scripting verbs,
  each with its `tool_labels` label, its options' names and labels, and
  `runs_in`: the task choices, comma-joined, whose `JobSpec` has it by
  `ToolDef.when`), the tasks (label, sentence, default), the groups (label,
  task), `TASK_WRITERS`, `KEEP` (the properties the form can keep on every
  section, `angles` / `orientation`, with their tasks) and `frozen_off` (what
  a freeze of each switches off, `JobSpec.switched_off`), the atlases
  (`core.atlas.details.descriptions()`, each with BrainGlobe's paragraph and
  `species`: Mouse, Rat or Other by name) and the default atlas, the
  planes, `flags`: each setting's flag as the settings table spells it
  (`job.settings.schema.flag_spellings`, setting name -> flag), which the
  page writes the command with, and `image_models` (`ImageModelChoice` per
  provider name `core.connections.CANONICAL_PROVIDERS` holds, the one
  `NonlinearSpec` checks `--nonlinear-provider` against: the job's default
  first, the one with no image model last; its models from
  `providers.connections` / `providers.openai_api`, its default model,
  whether this computer has its key or sign-in). Each tool row also says
  whether it calls the image model (`ToolDef.image_model`).
  `prefill(job_folder)` -> `Prefill` (None without a readable `job.json`):
  the form as `job.json` holds it, read through `read_job_file`,
  `without_removed_settings` and `JobSpec.from_dict` (the image folder
  resolved from the job folder), `notes`, and for a `host.kind == "abba"`
  job `AbbaBlock` (`host.abba`: ABBA's atlas name, file -> slice name,
  registered, existing warp, pixel size, thickness and interval, read with
  `core.stack.json_reads`); an ABBA job's thickness and interval are ABBA's
  where the spec has none; `frozen` the per-section freezes (`"*"` is
  `keep`'s).
- `settings.py` — `Settings(login=)`: `state()` (each connection's presence
  word from `doors.credentials.connections`, never a key or token; the saved
  choices; `runs_in` `{saved, enabled, value}`; per connection `models`,
  `default`, `model`, `reasoning`; the sign-in's `running` / `url` /
  `error`), `post(resource, body)`: `settings` (`connection`, `runs_in`,
  `model`, `reasoning`), `settings/key` (`setup.save_api_key`; the refusal
  never repeats the input), `settings/signin` (`setup.login_oauth` in a
  thread, `SIGNIN_TIMEOUT_S`; a failure is worded without the provider's
  reply); `SettingsError`.
- `window.py` — `find_browser` (`PATH_NAMES` on the PATH, then
  `candidates(platform, env, home)`: Windows `WINDOWS_ROOTS` x
  `WINDOWS_PATHS`, macOS `/Applications` and `~/Applications` x `MAC_APPS`),
  `open_window` (`--app=URL`, a private `--user-data-dir`, `--window-size`;
  a browser that exits at once, or none, falls back to `webbrowser.open`,
  then to printing the address) -> `Opened` (`app`, `tab`, `none`);
  `start_address(url, tab, folders=)`: the viewer's address as it is, the
  prompt generator's with `#tab=prompt`, plus `&job=0` (the first folder
  given) when folders were given.
- `static/generator.js`, `static/generator.css` — the prompt generator tab
  (loaded by one delimited block in `index.html`). `Generator`, pure
  functions on the payload and a plain form object (`folder`, `atlas`,
  `plane`, `thickness`, `interval`, `pixelSize`, `tasks`, `off`, `ordered`,
  `damaged`, `keep`, `notes`, `imageProvider`, `imageModel`, and from a
  job's prefill `abba`, `keepRegistrations`, `frozen`), exported to node for
  the tests: `startForm`, `prefilled(data, prefill)` (the form a
  `prefill` starts), `message` -> `{text, problems, warnings}`. A form with
  `abba` (a job prepared in ABBA) leaves the folder's atlas, plane and
  pixel size out of the command and of the left-out sentence and never
  checks the pixel size (the job holds them; their inputs are disabled);
  `--frozen` is then the kept properties on `"*"` plus the job's own
  per-section freezes while "keep the registrations" is ticked, and
  `'{}'` (or `"*"` alone) when it is unticked. The image-model row writes
  `--nonlinear-provider` / `--nonlinear-image-model` after `--disabled`,
  only while the run warps and a tool that calls the image model is on;
  provider `none` takes those tools out of the run. The text is one request sentence, a
  sentence naming what the form leaves to the agent (atlas, plane,
  thickness, interval, pixel size, the tasks when none is chosen) and one on
  bash quoting when a word is quoted, then a fenced `bash` block: the
  `langslice-job FOLDER init` command, one flag per line, only the fields
  set (spelled from the payload's `flags`: `--tasks`, `--atlas`, `--plane`, `--position-thickness-um`, `--position-interval-um`,
  `--pixel-size-um`, `--disabled` in table order: the tools of this run off
  whole, then the options off of tools still on, a name a kept property
  already switches off left out; `--frozen {"*": [...]}`, `--damaged`
  `{file: note}` from `file: note` lines, `--facts` for the cutting order,
  `--notes`). `problems` (no message while one stands) mirror
  `tool_table.refusal` on the form: the folder, whole micrometres, `look`
  keeps a mode, each chosen task keeps a writer, each tool on keeps what it
  needs (one sentence per needed group). The form's DOM is built in the
  file (the From ABBA block: fixed facts, the keep-registrations box, the
  file -> slice table; the Image model fieldset after the warp's tools);
  every text goes in by `textContent`. It loads `generator?job=<n>` when
  the address carries `#…&job=<n>`.
- `static/` — `index.html`, `app.css` (the palette tokens), `app.js`: the
  tabs (the first one `#tab=` names, `startTab`), the job list
  (polled every 2 s; the most recently changed job is shown until the user
  picks one), the snapshot poll (300 ms), the picture pane (follows the
  newest, previous/next, one section's pictures, enlarge), the activity list
  (tool rows with their pictures as thumbnails, pictures no event claims
  grouped by call, the opening first; the agent's text and reasoning; raw
  inputs under Details), the sections table, undo and redo, notes and spend;
  `settings.js` (the Settings tab: built once, updated from each answer, so a
  key being typed survives a refresh; refreshed when the tab is opened, every
  second while a sign-in runs; on the first load with no connection chosen it
  opens the tab, unless the address named a tab). Every text goes in by
  `textContent`.

## Tests

`tests/test_gui_watch.py` (appended and half-written lines, a state replaced
whole, a trace appearing mid-run, time order, missing files, the raw-state
fallback, undo labels, a path outside the folder, the job list,
a truncated file; a `slow` replay of a real folder, `LANGSLICE_GUI_REPLAY_JOB`),
`tests/test_gui_server.py` (token and Host, GET only, snapshot and pictures,
refused paths), `tests/test_gui_generator.py` (the catalog-driven tools, a
tool the tool table loses is not offered, every name labelled, no label in the
pinned model-facing text and `tool_labels` imported only here, the payload
and its endpoint; under node, skipped without it: golden messages for a
default and a filled form, read back by `init`'s parser into a `JobSpec`, a
tool of a task not chosen left out, refused tool choices giving no message
and refused by the spec too, warping alone; the prefill of a job prepared
in ABBA (a fixture writing `job.json` as `abba.prepare` does) and of an
opened job, `generator?job=n`, the ABBA messages with the registrations kept
and freed, read back by `init`; the image models are the providers `init`
takes, and their flags), `tests/test_gui_window.py` (the candidates per platform with a
fake `which` and filesystem, the app window, both fallbacks, the start
address per tab, `--tab` reaching `serve`),
`tests/test_gui_settings.py` (the settings file in a temporary home, the
environment override, the POST checks, a key never sent back, the Claude key
loaded for the agent only, the run location's greying, models per
connection, sign-in with a fake login, `linear run`'s model unchanged with no
file or settings off).
