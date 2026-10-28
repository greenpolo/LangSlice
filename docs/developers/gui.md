# The LangSlice window

The LangSlice window shows a registration while it runs: the pictures the
agent looks at, what it is doing, the sections' current placements and the
steps it can undo. It reads the job folder, so it works the same whether the
run is LangSlice's own agent, a coding agent (Claude Code, Codex,
Antigravity) or a script. Closing the window does not stop the run.

## Opening it

```bash
langslice gui                       # the jobs opened recently on this computer
langslice gui ~/data/brain1         # an image folder (its langslice/ job folder) ...
langslice gui ~/data/brain1/langslice  # ... or the job folder itself
langslice linear run ~/data/brain1 --gui   # start a run and open the window on it
```

LangSlice serves the window from this computer only and opens it as a browser
window without an address bar, using Chrome, Chromium, Edge or Brave. Without
one of those it opens a tab in the default browser; if that fails too it
prints the address to open. `--no-window` only prints the address, and
`--port N` serves on a fixed port. The command keeps running until Ctrl-C, or
until the window has been closed for five seconds.

## What it shows

The window has three tabs: **Agent viewer**, **Prompt generator** and
**Settings** (top right). The prompt generator and settings are not built yet.

The agent viewer lists the jobs on the left: the folders given on the command
line and the jobs opened recently on this computer
(`~/.langslice/recent_jobs.jsonl`; `LANGSLICE_RECENT_JOBS` names another file,
or `off`). It shows the most recently changed job, and new jobs as they
appear. For the chosen job:

- **Status line.** Running, Waiting, Submitted or Failed, with the tool running
  now or the last one, and the atlas, plane, model and door when known.
- **Picture pane.** The newest saved picture, large, with its caption; earlier
  and later pictures, one section's pictures only, and a click to enlarge.
  *Follow* returns to the newest.
- **Activity.** One row per tool call with its sections and the pictures it
  saved (a click shows one in the pane); a failed call is red. When the job
  folder holds LangSlice's agent trace, the agent's text and its reasoning
  summaries appear between the calls. A coding agent's own reasoning is not
  in the job folder while it runs, so it is not shown. Each row's inputs are
  under **Details**.
- **Sections.** Order, filename, position, cutting angles, flip, transform,
  deformation method, damage and registration number. A job saved by an older
  LangSlice is shown from its raw fields.
- **Undo** and **Redo.** The steps that can be undone and redone, each named by
  the call that made it.
- **Notes** (yours from the job's settings, and the run's) and **Spend**
  (tokens and the dollar estimate, when the run records them).

The viewer is read-only: nothing in it changes the job.

## How it works

The window is a page served by `langslice gui` on `127.0.0.1` under a random
address, reachable only from this computer. It watches the job folder's files
(`logs/events.jsonl`, `views.jsonl`, `trace/`, `state.json`,
`history/index.json`, `job.json`, `exports/spend.json`; see
[file formats](../file_formats.md)) four times a second while a page looks at the
job, and serves only the folders it lists.
