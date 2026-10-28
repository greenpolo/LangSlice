# LangSlice `hosts/gui/` — the LangSlice window

Package guide for `src/langslice/hosts/gui/`. The repo-level `CLAUDE.md` holds
the project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror
to the other. The user's page is `docs/gui.md`.

One local window, titled "LangSlice", with three tabs: Agent viewer, Prompt
generator and Settings (top right). The prompt generator and settings are a
placeholder sentence each. The agent viewer is a preliminary layout in the
ABBA log's styling (`hosts/integrations/static/chat.css`'s palette and type,
the comparison window's blue-grey picture panel); Nash designs the final one.

## The rule

The GUI is a separate process from every run (the built-in agent, an MCP
server, the agent CLI's calls, a SliceBench subprocess), so it learns
everything by watching files: no socket, no relay, no import of the run. It
imports the core (`core.state`, `core.status`, `core.toolset`) and the job
layer (`job.layout` names, `job.checkpoint.current_state`, `job.history`,
`job.registrations`) only. Nothing imports it but `hosts/cli.py`
(`langslice gui`); `doors/cli/linear.py`'s `--gui` starts that command as a
process (`open_gui`). It is read-only: no command endpoints, nothing written to
a job folder. Nothing in it is model-facing.

## Files

- `__init__.py` — `serve(folders, port=, window=)`: the job list
  (`recent_jobs_file()`), the server, the window; runs until Ctrl-C or until
  the app window's process has been gone for `LINGER_S` (5 s); a tab or no
  window serves until Ctrl-C.
- `watch.py` — `JobWatcher(folder)`: per job, `_Tail`s (byte offsets, an
  unterminated last line kept for the next round, a file grown shorter starts
  the watch over: a new `generation`) on `logs/events.jsonl`, `views.jsonl` and
  the newest `trace/*.jsonl`; `_Whole`s (size and mtime, one retry on a file
  caught mid-rename) on `state.json`, `history/index.json`, `job.json`,
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
  `core.status.status_rows` of `StackState.from_dict(current_state(...))` with
  each section's registration number (`job.registrations`); a state this
  version does not read (another format) shows its raw `slices` fields with a
  note. Undo labels: `_label_step` follows each `tool_end`'s `step` (the undo
  depth after the call, capped at `UNDO_DEPTH`): a call that deepens it names
  the new depth; at the cap a call of any tool but a read (`core.toolset` kind
  `read`; a plugin's or an older version's tool counts as a write) drops the
  oldest name; `undo` / `redo` move names between the sides; a depth no event
  names is `Step n`. `status`:
  Submitted (`state.submitted`), Failed (a trace `error`, or a `summary`
  without a submit), Running (a `tool_start` without its end, or a file
  changed within `RECENT_S`), else Waiting, and a sentence. `snapshot(since,
  generation, state_version)`: the items after the cursor (all of them for
  another generation), the header, the status, and the tables only when
  `state_version` moved. `JobList`: the folders given, then the recent-jobs
  file's (`folder` of each line, newest first, existing folders, at most
  `MAX_RECENT`), indexed in the order added; `listing` peeks at `job.json`
  and `state.json` only. `recent_jobs_file()`: `~/.langslice/recent_jobs.jsonl`,
  or `LANGSLICE_RECENT_JOBS` (another file, or `off`), read as the job
  layer writes it.
- `server.py` — `GuiServer(jobs, port=)`: `ThreadingHTTPServer` on
  127.0.0.1, every path under a random token, the `Host` header checked
  against the port, `HEADERS` (CSP, `no-store`, `nosniff`, no referrer; as
  `abba_chat.ChatServer`), GET only (others 405). `answer(path, host)`: the
  static files, `jobs`, `jobs/<n>/snapshot?since=&gen=&sv=`,
  `jobs/<n>/picture/<number>` (the `view.jpg` of a picture the job's
  `views.jsonl` lists, read from disk); anything else 404.
- `window.py` — `find_browser` (`PATH_NAMES` on the PATH, then
  `candidates(platform, env, home)`: Windows `WINDOWS_ROOTS` x
  `WINDOWS_PATHS`, macOS `/Applications` and `~/Applications` x `MAC_APPS`),
  `open_window` (`--app=URL`, a private `--user-data-dir`, `--window-size`;
  a browser that exits at once, or none, falls back to `webbrowser.open`,
  then to printing the address) -> `Opened` (`app`, `tab`, `none`).
- `static/` — `index.html`, `app.css`, `app.js`: the tabs, the job list
  (polled every 2 s; the most recently changed job is shown until the user
  picks one), the snapshot poll (300 ms), the picture pane (follows the
  newest, previous/next, one section's pictures, enlarge), the activity list
  (tool rows with their pictures as thumbnails, pictures no event claims
  grouped by call, the opening first; the agent's text and reasoning; raw
  inputs under Details), the sections table, undo and redo, notes and spend.
  Every text goes in by `textContent`.

## Tests

`tests/test_gui_watch.py` (appended and half-written lines, a state replaced
whole, a trace appearing mid-run, time order, missing files, the raw-state
fallback, undo labels and the cap, a path outside the folder, the job list,
a truncated file; a `slow` replay of a real folder, `LANGSLICE_GUI_REPLAY_JOB`),
`tests/test_gui_server.py` (token and Host, GET only, snapshot and pictures,
refused paths), `tests/test_gui_window.py` (the candidates per platform with a
fake `which` and filesystem, the app window, both fallbacks).
