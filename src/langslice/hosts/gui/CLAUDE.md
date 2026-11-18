# LangSlice `hosts/gui/` — the LangSlice window

Package guide for `src/langslice/hosts/gui/`. The repo-level `CLAUDE.md` holds
the project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror
to the other. The user's page is `docs/developers/gui.md`.

One local window, titled "LangSlice", with three tabs: Agent viewer, Prompt
generator and Settings (top right). The agent viewer has two views (Simple |
Extended, `v`, kept in `localStorage` as `langslice.detail`). Simple, the
start, shows three things: the agent speaking (the LangSlice mark, the
model's name and its state beside a speech bubble with its latest words and
the call it is making in plain words), the sections it is working on, and the
cutting axis under them. Extended is made for a person who wants the whole
run, read top to bottom: the same speaker with the progress meters, up to
four panels showing the sections the agent works on under their current
registration with the cutting axis and the atlas's pictures under them, and
on the right the journal (the run so far as a timeline of the agent's words,
its tool calls quiet lines between them). The window is neutral
graphite around greyscale pictures with one accent colour; the agent's words
are in a serif face, the window's in a sans-serif face
(`static/tokens.css`). Both views have **2D | 3D** beside Simple |
Extended (one setting, kept in `localStorage` as `langslice.stage`): 3D
takes the cutting axis's place, showing the sections in focus in a narrow
column at the left and the atlas brain in three dimensions, with every
placed section in it, filling the rest (`src/space/`).

## The rule

The GUI is a separate process from every run (the built-in agent, an MCP
server, the agent CLI's calls, a SliceBench subprocess), so it learns
everything by watching files: no socket, no relay, no import of the run. It
imports the core (`core.stack.stack`, `core.stack.status`, `core.atlas.details`,
`core.connections`, `core.stack.json_reads`; for the viewer's pictures
`core.pictures.look` (`look`, `LookRequest`, `snapshot`),
`core.sections.renders.render_slice`, `core.fits.deformation.RecordStore`,
`core.pictures.jpeg.encode_jpeg`, `core.pictures.captions.view_angles_label`)
and the job layer (`job.settings.tool_table`, `job.settings.spec`,
`job.settings.schema`, `job.settings.labels`, `job.layout` names and
`read_job_file`, `job.migrations.viewer_stack` and `without_removed_settings`,
`job.history`, `job.review` for the review comments), the providers' image-model lists (`providers.connections`,
`providers.openai_api`), the doors' job reading for the viewer's pictures
(`doors.jobs.read_spec`, `context`: the job read as the agent CLI reads it,
never opened), and for the Settings tab the doors' setup
(`doors.credentials`: key presence, `save_api_key`, `login_oauth`;
`image_model_connected` for the generator) and the
user's settings file (`doors.user_settings`). Nothing imports it but
`hosts/cli.py` (`langslice gui`); `doors/commands/linear.py`'s `--gui` starts that
command as a process (`open_gui`). The viewer and the prompt generator change
no registration: every job and generator endpoint is GET only but the review
comments' POST and the Export menu's, and nothing else is written to a job
folder (the viewer's pictures are kept in memory; the atlas's own disk cache
is the only file the core may write, as for any reader). The writes are the
Settings tab's three POSTs (the user's settings file, a saved key, the
ChatGPT login), the review comments' (`jobs/<n>/comments`: the job's
`review.json`, through `job.review`) and the Export menu's two (a job's
export under its `exports/` folder and its `registration.json`, never its
state); sign-in runs in this window's
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

The page is a React app (`web/`, below), built into `static/`.

- `web/` — the page's source: React 19, TypeScript, Motion (`package.json`
  pins them; `npm install`, `npx tsc --noEmit`, `npx vite build`). Vite
  builds it into `static/index.html`, `static/ui.js` and `static/ui.css`
  (fixed names directly in `static/`, no inline script or style, so the
  content policy stays `'self'`), which are committed: installing LangSlice
  needs no Node. `src/run.ts` is the polled store (the job list every 2 s,
  the shown job's snapshot every 300 ms, the strip when the sections change;
  `useRun(select)`; the run's `log` for the journal; `sendComment`);
  `src/nav.ts` keeps the tab and the view in the address
  (`#tab=viewer|prompt|settings&view=simple|extended&job=n`);
  `src/pictures.ts` loads the sections' quiet pictures (`?style=calm`) at
  most three at once, lowest rank first, decoded and kept by address, with
  each section's newest picture as a stand-in (`newestOf`), sizes snapped
  to `EDGES`, and `warm` for the finished run's grid while nothing else
  loads; `src/motion.ts` is the motion family (the same values as
  `tokens.css`'s `--ease-*` and `--dur-*`). `src/Viewer.tsx`: Simple or
  Extended, following the agent until a section is picked (`f` follows
  again, `v` switches, `r` reviews). `src/simple/`: `Speaker` (the mark,
  the agent's figure with one gesture per state, its name and state, the
  fixed-height bubble whose old and new lines share one cell, the call in
  plain words), `Stage` (`geometry` places every tile; `useDirector` holds a
  focus `DWELL_MS`, a grid `DECK_DWELL_MS`, and shows the next once its
  pictures are ready, at most `READY_WAIT_MS`; Motion layout animations move
  a section still shown), `Picture` (canvas layers crossfading on a
  redraw), `CuttingAxis` (under the pictures in both views: a millimetre
  ruler over the placed sections' span plus a margin, the atlas's pictures
  above it in Extended only, every placed section's thumbnail in one row in
  the order of its position, joined to its key on the ruler by a line, two
  half-offset rows when one row would make them narrower than
  `ONE_ROW_MIN_W`; a thumbnail glides to its new place when the agent moves
  it; wheel zoom, drag pan, double-click for the whole span, hover names,
  click picks). `src/extended/`: `ExtendedView` (the progress box:
  one meter per step the job's `tasks` ask for, counted over its sections;
  four sections a page; in 3D no cutting axis), `Journal`, `Lightbox`. `src/space/`:
  the 3D view (three.js). `mode.ts` keeps 2D | 3D (`useStageMode`,
  `setStageMode`; the switch is in `Viewer.tsx`, its buttons pressed to 0.97; the
  stage switched to comes in with a 240 ms ease-out crossfade and a little blur,
  `justSwitched`, the cutting axis too when 2D comes back, never when the
  window opens); `SpaceStage.tsx` is both
  views' 3D stage (the focus Stage in a column, the Volume beside it, no
  panel or border around it). `scene.ts` (`Space`) maps atlas micrometres to
  scene millimetres by the orientation letters (anterior +z, superior +y,
  the animal's right -x, centred on the atlas), draws BrainGlobe's root mesh
  as brainrender shows it (matte, see-through, lit over the viewer's
  shoulder, faces turned outward, with a thin grey silhouette where the
  surface turns away, drawn `INFLATE_MM` outside it so sections reaching the
  brain's edge do not hide it) and each placed section as an opaque slab
  whose basis is its thumbnail's corners (`TR-TL`, `BL-TL`, the normal times
  the thickness; the shear the decomposition leaves is kept), only its
  picture's ground left out by a mask made once per picture (`tissueImage`:
  brighter than the corners' luminance over a 5x5 mean; smooth patches
  touching the ground outside, glare or grey smears, cut; specks dropped,
  the dark outer edge peeled, pinholes filled; uploaded as a `DataTexture`,
  never through a canvas, whose premultiplied alpha would blacken the cut
  ground into a rim), a hard edge, no mipmaps (they blur the tissue into
  the ground at a slant); picking follows the same mask. A section
  without a written transform sits where the core's identity puts it: its
  picture as the job keeps it (no turn or flip beyond the job's), centred
  on the atlas plane at its position, at its true pixel size. A section
  still at the job's initial spacing (`position_source` `initial`) is not drawn; a newly placed one appears where it is,
  whole (never part see-through); a changed placement is glided to on critically damped springs
  (position, size, rotation); the sections a call changed come in stack
  order over `BATCH_S`; nothing else moves. The sections in focus lift by
  `LIFT` of the brain's height (an ease-out over `FOCUS_S`) and brighten a
  little, and the others grey a little; sections are never see-through.
  The brain's nearest surface only is drawn (a depth pass first), and
  inside it, behind everything, its far wall in the sections' mean tissue
  colour where sections stand close (`WALL_REACH_MM`), so the gaps of a
  thin-section stack do not read as black ribs. OrbitControls with damping, a lateral three-quarter view (`VIEW`) fitted
  to the mesh; reduced motion moves everything at once; a click on a slab
  picks its section. `Volume.tsx` fetches the shell once per job, the space
  again when a row's `render_key` changes (at most every `SPACE_MS`), the
  calm thumbnails through `pictures.ts` (`THUMB_RANK`, after the stage's),
  and shows the placed count, the thickness, a hint and the hovered
  section's name; in 3D the journal's column is narrower. `src/review/Review`:
  one section, arrow keys, wheel zoom, a dragged box and its comment.
  `src/prompt/`, `src/settings/`, `src/export/` (with `controls.tsx`): the
  Prompt generator (its message from `static/generator-logic.js`, the same
  file the tests run under node), Settings and the Export menu, as the
  endpoints below describe.
- `__init__.py` — `serve(folders, port=, window=, tab=)`: the job list
  (the recent jobs file, `job.recent.recent_file()`), the server, the window
  on `window.start_address` (`langslice gui --tab viewer|prompt`); runs until Ctrl-C or until
  the app window's process has been gone for `LINGER_S` (5 s); a tab or no
  window serves until Ctrl-C. The address is printed and flushed at once; a
  `--port` in use ends with one sentence (`Port N is in use: ...`;
  `server.port_in_use`, which counts Windows' WSAEACCES for a port another
  program listens on), any other bind error with its reason.
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
  JSON string; `title`, the tool's plain-language label from
  `job.settings.labels`), `picture` (number, view folder, tool and its `title`, call, sections, caption
  from `view.json`; a path outside the folder is refused), `agent` and
  `reasoning` (the trace's `model` `text` / `thought`; for a folder with no
  trace, the same records from Claude Code's transcript, `transcripts.py`),
  `note` (open, plugins, script writes, session start and end, a trace
  error). The sections table:
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
  changed within `RECENT_S`), Idle (nothing changed for `IDLE_S`, 600 s:
  "No activity for 3 h. Last: ..."), else Waiting, and a sentence. Each
  sections row also carries `render_key` (`render.render_key`; null in the
  raw-fields fallback) and `position_mm` (a number or null). `focus`: the
  agent's focus (`focus.started_focus` of each `tool_start`; before any,
  every section). `current_stack()`: the stack as last read (polled first;
  replaced whole, never edited in place), what the renderer draws.
  `snapshot(since, generation, state_version)`: the items after the cursor
  (all of them for another generation), the header (`name`, `path`: the
  images folder, ...), the status, the focus, and the tables only when
  `state_version` moved. `JobList`: the folders given, then the recent-jobs
  file's (`folder` of each line, newest first, existing folders, at most
  `MAX_RECENT`), indexed in the order added; `listing` peeks at `job.json`
  and `state.json` only: `index`, `name` (`_job_name`: the images folder and
  the folder above it, `M01/images`, an images folder's own name often being a
  generic one; a job folder made elsewhere, its own name and the one above),
  `path`, `folder`, `atlas`, `changed_at`, `status`, `given`. The recent jobs file is the job layer's
  (`job/folder/recent.py`: `~/.langslice/recent_jobs.jsonl`, or
  `LANGSLICE_RECENT_JOBS`, another file or `off`).
- `transcripts.py` — a coding agent's words. `TranscriptWords(root)`
  (`projects_root()`: `~/.claude/projects`, or `LANGSLICE_CLAUDE_TRANSCRIPTS`,
  another folder or `off`; the test suite sets it off) `.read(job_folder,
  job_file, until)`: until the job's transcript is known, every `SEARCH_S`
  it reads on in the candidates (`*/*.jsonl` and `*/*/subagents/*.jsonl`
  last written between `EARLY_S` before the job's first tool call,
  `first_call_at`, and `LATE_S` after its last change; at most
  `MAX_CANDIDATES`, nearest that change first; each by byte offset, so a
  growing file is never read twice) for an assistant turn whose `start_job`
  call names one of `job_folders` (the job folder, its images folder, the one
  `job.json` names; `folder_key`); then it tails that file. Only lines
  holding `"type":"assistant"` are parsed (`Turn`). It gives trace records:
  one `session` (`model_name` of the model id, agent `Claude Code`), then a
  `model` record per reasoning summary that is not empty (`thought`) and per
  message (`text`), times local as the trace's (`local_time`). Read-only.
- `server.py` — `GuiServer(jobs, port=, settings=)`: `ThreadingHTTPServer` on
  127.0.0.1, every path under a random token, the `Host` header checked
  against the port, `HEADERS` (CSP, `no-store`, `nosniff`, no referrer).
  `answer(path, host)` (GET): the
  static files (`.ico` among them: the icon), `jobs` (with `home`, the home folder, which the page writes `~` in the paths it shows), `generator` (`generator.payload()`; `?job=<n>` adds
  `prefill`, `generator.prefill` of that listed job as plain data, null
  without a `job.json`, 404 for an index not listed),
  `jobs/<n>/snapshot?since=&gen=&sv=`,
  `jobs/<n>/picture/<number>` (the `view.jpg` of a picture the job's
  `views.jsonl` lists, read from disk), `jobs/<n>/strip` (`JobRenderer.strip`), `jobs/<n>/space` (`JobRenderer.space`,
  `space.job_space`: the thickness, the job's or `DEFAULT_THICKNESS_UM`, and
  whether it was given; the plane's axis; each section's id, order and
  `corners_um`, its thumbnail's top-left, top-right and bottom-left corners
  in atlas micrometres in the atlas's axis order, null while unplaced; 409
  without a state this LangSlice reads) and `jobs/<n>/shell`
  (`JobRenderer.shell`, `space.atlas_shell`: the atlas's orientation and
  extent, and its `root` mesh (`mesh_from_structure`, the mesh brainrender
  draws, in the atlas micrometres the corners use): the vertex and triangle
  counts, `positions` base64 of little-endian float32 x, y, z and `indices`
  of uint32 triangles; 409 for an atlas without one),
  `jobs/<n>/render/overlay|thumb/<section>?edge=&key=` (the section's filename
  URL-encoded and one of the job's sections, else 404; `key`, the row's
  `render_key`, is for the browser's cache only), `jobs/<n>/render/atlas/<k>?edge=`,
  `settings`; anything else 404. The snapshot answer also carries
  `comments`: the job's review comments (`comments.CommentRow` each, through
  one `CommentWatch` per folder). A picture that cannot be drawn is 409
  `{"error"}` with the reason, one still being drawn after `render.WAIT_S` 503.
  The static files: `""` is `index.html`, and every other file directly in
  `static/` whose name `STATIC_NAME` matches is served with its type from its
  extension (`STATIC_TYPES`), so a new script or stylesheet needs no server
  change. `GuiServer(..., renders=)` takes a `RenderPool` (tests).
  `post(path, headers, body)`: only `settings.POSTS` and `jobs/<n>/comments`
  (`_comment`: `{"section", "box", "text"}` keeps a comment
  (`job.review.add_comment` on the watcher's `current_stack`), `{"remove": id}`
  takes one away; the answer is the comments as they now stand, a
  `ReviewError` 400 `{"error"}`; any other path 405),
  with the token and `Host`, `Origin` exactly `http://127.0.0.1:<port>` and
  `Sec-Fetch-Site` same-origin when sent (403), `Content-Type`
  `application/json` (415), at most `MAX_BODY` bytes (413); a refusal is 400
  `{"error"}`. PUT, DELETE, PATCH, HEAD: 405. The Settings code in `server.py` and
  `index.html` sits between `Settings tab` marker comments.
- `comments.py` — the review comments as the viewer shows them:
  `CommentRow` (the comment's fields and `revised`: its section changed
  since, `job.review.revised`), `CommentWatch(folder).rows(state, version)`
  (read again only when `review.json`'s stamp or the watcher's
  `state_version` moved).
- `calm.py` — the Simple view's pictures (`render/overlay|thumb/...?style=calm`,
  `JobRenderer.overlay(..., calm=True)`, cached under their own kind):
  `main_channel` (a channel named for a nuclear or Nissl stain, else
  `blue`, else the first), `draw_calm_overlay` (two `look` calls on that
  channel, with and without the borders; their difference is the border
  layer, so `look`'s scale bar goes), `draw_calm_thumb`, `calm` (the tissue
  `tissue_mask`ed off the tile floor, `scale_bar` cleared, a soft knee, a
  cool tint on `STAGE`, the borders at `LINE_STRENGTH`, fainter when small).
- `export.py` — the Export menu: `FORMATS` (`abba`, `quicknii`,
  `visualign`); `run_export(folder, kind)` -> `Written` writes one through
  the library (`langslice.open_job`: `export_abba`, or `export_stack` for
  both QUINT files, naming *kind*'s); `Exports(run=)` keeps each job's last
  `ExportState` (`ExportStatus` idle, running, done, failed; path, message,
  sections left out, time) and runs each export on a thread of its own
  (`start`, `ExportError` for an unknown format or one already running);
  `reveal(folder)` opens the job's `exports/` in the system's file browser.
  Served as `GET jobs/<n>/export` (the state), `POST jobs/<n>/export`
  (`{"format", "landmarks"}`: an ABBA project's most landmark pairs per
  warp, `MIN_LANDMARKS` (100) to 1089, `ExportError` outside; `abba_linear`
  writes no warp steps) and `POST jobs/<n>/export/reveal`, with the Settings POSTs'
  checks (`server.GuiServer.post`). `web/src/export/`: the top bar's Export button, its menu (ABBA, warps
  of 1089 landmarks; ABBA editable, 100; ABBA linear only; QuickNII; VisuAlign; a row for an
  ABBA project with a count of the person's own, 100 to 1089), and a note that
  says the export is running, then its path with Open folder and Copy path,
  or why it failed.
- `focus.py` — the agent's focus, as ABBA's follower followed it: `Focus`
  (`sections`, `tool`, `call`, `at`); `is_followed(name)` (a tool of kind
  `write` in `job.settings.tool_table`, a plugin's or an unknown tool, and
  `look` and `zoom`, `FOLLOWED_READS`); `started_focus(record)`: a followed
  `tool_start` with targets gives its `target_ids` in order without repeats.
- `space.py` — the 3D view's numbers: `atlas_shell(workspace)` ->
  `AtlasShell` (`ShellError` without a root mesh), `job_space(workspace, state, thickness_um=)` -> `Space`
  (`SpaceSection` per section in stack order), `section_corners` (from
  `core.maps.section_maps.section_frame`, the frame `registration.json` is
  written from, through the thumbnail's turn and flip, `turned_corners`).
- `render.py` — the viewer's pictures of a job's current registration.
  `RenderPool(atlas_loader=, workers=WORKERS)`: one `ThreadPoolExecutor` (2)
  for every job and one `JobRenderer` per job folder (`renderer`); the jobs
  not drawn lately beyond `KEPT_JOBS` let go of their workspace (`release`).
  `JobRenderer` reads the job for drawing without opening it (`_open_job`:
  `doors.jobs.read_spec` and `context`, a `RecordStore` on the sections'
  folders; read again when `job.json` changes; a spec, image or atlas that
  cannot be read is `RenderError` with the reason) and draws from the
  watcher's `current_stack`: `overlay(id, edge)` (`look` mode `overlay`, the
  applied deformation),
  `thumb(id, edge)` (`render_slice`, oriented); both draw the section's raw
  channels in their own colours (`drawing.py`: `draw_overlay`, `draw_thumb`,
  `draw_atlas`, `raw_channels`, at most `MAX_OVERLAY_CHANNELS`, `content`),
  not the `preprocessed` image the fits read,
  whose contrast enhancement makes a scanner's tile mosaic stand out,
  `atlas_thumb(k, edge)` (`look` mode `atlas` at the stack's view angles: the
  template, `look`'s own atlas default, because the borrowed Nissl of the
  Allen atlases would add over a gigabyte), each with its caption band cut off
  (the recipe's `shown` content) and `encode_jpeg`d; `strip()` -> `Strip`
  (`Workspace.position_range` and `axis_ends`, `step_mm` by `strip_step`:
  `NOMINAL_STEP_MM` times `brain_scale`, rounded to `ROUND_STEPS_MM` so
  `MIN_THUMBNAILS` to `MAX_THUMBNAILS` cover the axis, the atlas positions,
  the view angles in words, each section's id, order, position, render key
  and thumb key). A job draws one picture at a time (a workspace's caches are not made
  for concurrent writers); concurrent requests for one picture share one
  drawing; a queued overlay or thumbnail whose section's key has changed
  by its turn is not drawn (answered `RenderBusy`; asked again, it draws
  the section as it is then); a request waits `WAIT_S` (60 s), then
  `RenderBusy`. The JPEGs are
  kept in an LRU of `CACHE_ENTRIES` per job, keyed by kind, section or
  position, edge and render key (a thumbnail by its thumb key), and the workspace's caches are trimmed after
  each drawing (`KEPT_SOURCES`, `KEPT_PLANES`, `KEPT_RENDERS`).
  `render_key(state, id)`: `KEY_DIGITS` hex of the section's `look.snapshot`,
  changing exactly when its overlay would; `thumb_key(state, id)`: the
  same of its flip, quarter turn and made images' digests, so a new
  placement does not draw the strip's thumbnail again; `edge_px` (`MIN_EDGE_PX` to
  `MAX_EDGE_PX` in steps of `EDGE_STEP_PX`, `DEFAULT_EDGE_PX` without one);
  `UnknownSection`.
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
  `NonlinearSpec` checks `--nonlinear-provider` against: the ChatGPT sign-in
  first, the one with no image model last; its models from
  `providers.connections` / `providers.openai_api`, its default model,
  whether this computer has its key or sign-in). Each tool row also says
  whether it calls the image model (`ToolDef.image_model`) and whether it
  starts off (`default_off`: its box is unticked until the person ticks it,
  the form's `on`, written as `--enabled`: `trace_borders`,
  `deepslice_place`, `brainways_ap`; `runs_in` is computed with an image
  model named). The image-model row's `default` is the connection the job
  would take with none chosen (`doors.credentials.saved_image_provider`,
  else `none`).
  `integrations`: `INTEGRATIONS`, the programs the results can go to
  (QUINT, VERSO, DMC-BrainMap; ABBA is not one), each `Integration`'s
  `name`, its box's `label` and the `words` the request names it with.
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
- `app.py` — LangSlice's own window (`python -m langslice.hosts.gui.app URL
  [WIDTH HEIGHT]`): pywebview (the `window` extra; imported only here, inside
  `main`) shows the page in the system's web view, titled `TITLE` with `ICON`
  (`static/langslice.ico`, made from `assets/logo/logomark_dark_512.png` with
  rounded corners); on Windows `own_taskbar_identity` (`APP_ID`, so the
  taskbar files it under LangSlice, not Python) and `dark_title_bar`. A
  process of its own, private mode: nothing is kept between windows.
- `window.py` — `own_window_command(url, size)` (the command that starts
  `app.py`, None without pywebview), `find_browser` (`PATH_NAMES` on the PATH, then
  `candidates(platform, env, home)`: Windows `WINDOWS_ROOTS` x
  `WINDOWS_PATHS`, macOS `/Applications` and `~/Applications` x `MAC_APPS`),
  `open_window` (LangSlice's own window first, unless a browser was named:
  one that ends within `OWN_WINDOW_START_S` falls through; then a browser's
  `--app=URL`, a private `--user-data-dir`, `--window-size`;
  a browser that exits at once, or none, falls back to `webbrowser.open`,
  then to printing the address) -> `Opened` (`app`, `tab`, `none`);
  `start_address(url, tab, folders=)`: the address with `#tab=<tab>&job=0`
  (the first folder given) when folders were given; without folders the
  viewer's address as it is (the page then opens Settings while no
  connection is chosen) and the prompt generator's with `#tab=prompt`.
- `static/generator-logic.js` — the prompt generator's message, imported by
  `web/src/prompt/` (typed by `generator-logic.d.ts`). `Generator`, pure
  functions on the payload and a plain form object (`folder`, `atlas`,
  `plane`, `thickness`, `interval`, `pixelSize`, `tasks`, `off`, `on`, `ordered`,
  `damaged`, `keep`, `notes`, `imageProvider`, `imageModel`,
  `integrations`, and from a
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
  thickness, interval, pixel size, the tasks when none is chosen), one
  naming the programs the results go to (`integrationWords`: the ticked
  `integrations` in the payload's order, `When the registration is done,
  prepare the results for ...`; none for a form with `abba`; for the main
  agent only, never in the command) and one on
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
  needs (one sentence per needed group). The form (`web/src/prompt/`: the
  From ABBA block: fixed facts, the keep-registrations box, the
  file -> slice table; the Image model fieldset after the warp's tools)
  loads `generator?job=<n>` when
  the address carries `#…&job=<n>`. The layout: numbered cards (1 Images, 2
  Atlas, 3 Sections, 4 What to do, 5 What you already know, 6 Notes for the
  agent, 7 Where you'll use the results (a box per payload integration,
  none ticked; left out for a job prepared in ABBA), then Advanced, closed;
  the From ABBA card first when there is one)
  beside a sticky **Your message** panel (the whole message, the command in
  violet, Copy, problems in pink, warnings muted), one column below 1080 px;
  the plane as three buttons and *Let the agent decide*; the Advanced tools
  as a table of switches, each greyed row saying why.
- `static/` — what the server serves: the built page (`index.html`,
  `ui.js`, `ui.css`), `generator-logic.js`, the fonts, the icon and the logos.
  The CSP allows no inline script or style, no `style` attribute and no
  `data:` or `blob:` image.
  - `tokens.css` — the window's look, the only place a colour, the font, a
    size or a spacing is written: three faces, the system's sans-serif UI
    face for the window's own words (`--sans`), a system serif face for the
    agent's own words (`--serif`: the narration and the journal) and the
    monospace face (JetBrains Mono, bundled as `jetbrains-mono-*.woff2`, SIL
    OFL 1.1, `jetbrains-mono-ofl.txt`) for literal text only (paths,
    commands, tool inputs); neutral graphite grounds (`--ground`, panels and
    controls a step lighter in `--panel`, black `--picture` behind pictures
    only), 1 px edges, small rounded corners (`--radius`, `--radius-sm`), no
    glow, gradients or shadows, colours each with one job (off-white names
    and the agent's latest words, lavender the one accent: what is happening
    now and what can be clicked, also `--focus`, the sections in focus;
    grey-violet literal text, pink problems, green Submitted; `--clawd`
    is the mascot's own). `:root[data-theme="light"]` is the light look:
    paper grounds and dark words, pictures still on black, and the brain
    strip on a dark ground of its own (`--stage*`). The page keeps the choice in `localStorage` and swaps
    the lockup. `lockup.svg` and `logomark.svg` are `assets/logo/`'s dark
    ones, `lockup-light.svg` the light one.

## Tests

`tests/test_gui_watch.py` (appended and half-written lines, a state replaced
whole, a trace appearing mid-run, time order, missing files, the raw-state
fallback, undo labels, a path outside the folder, the job list,
a truncated file, the focus, Idle, the rows' render keys and positions, job
names),
`tests/test_gui_transcripts.py` (the transcript found by its `start_job`
call among others and tailed, a half-written line, an old transcript and a
job with no call left alone, `off`, the watcher showing the words),
`tests/test_gui_render.py` (every picture drawn with nothing written to the
folder, an overlay as `look`'s with its deformation and without its caption,
the render key, the strip and its spacing, one drawing for concurrent requests,
busy after the wait, what cannot be drawn says why, the workspace trimmed, jobs
let go), `tests/test_gui_server.py` (token and Host, GET only, snapshot and
pictures, refused paths, static names, the strip and the pictures, 404, 409,
503), `tests/test_gui_generator.py` (the catalog-driven tools, a
tool the tool table loses is not offered, every name labelled, no label in the
pinned model-facing text and `tool_labels` imported only here, the payload
and its endpoint; under node, skipped without it: golden messages for a
default and a filled form, read back by `init`'s parser into a `JobSpec`, a
tool of a task not chosen left out, refused tool choices giving no message
and refused by the spec too, warping alone; the prefill of a job prepared
in ABBA (a fixture writing `job.json` as `abba.prepare` does) and of an
opened job, `generator?job=n`, the ABBA messages with the registrations kept
and freed, read back by `init`; the image models are the providers `init`
takes, and their flags; the programs ticked named in the request only,
the command the same as without them, none for a job prepared in ABBA; the
tab's stylesheet uses only `tokens.css` names and
no colour value, and its script builds no HTML from strings), `tests/test_gui_window.py` (the candidates per platform with a
fake `which` and filesystem, LangSlice's own window first and the browser when it cannot start, its icon shipped, the app window, both fallbacks, the start
address per tab, `--tab` reaching `serve`, a port in use in one sentence, the
address flushed),
`tests/test_review_comments.py` (a comment kept with nothing else
written to the job folder, the refusals, Origin needed, removal; the tool's
pictures with the boxes outlined, open and revised after a write, the
statement's line),
`tests/test_gui_settings.py` (the settings file in a temporary home, the
environment override, the POST checks, a key never sent back, the Claude key
loaded for the agent only, the run location's greying, models per
connection, sign-in with a fake login, `linear run`'s model unchanged with no
file or settings off; the tab's stylesheet uses only `tokens.css` names and no
colour value, and its script builds no HTML from strings).
