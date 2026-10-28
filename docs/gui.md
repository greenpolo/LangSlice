# The LangSlice window

The LangSlice window writes the message that starts a registration in your
agent app, and shows the registration while it runs: the pictures the
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
**Settings** (top right). Settings is not built yet.

## The agent viewer

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

## The prompt generator

The prompt generator turns what you know about a brain into one message to
paste into your agent app (Claude Code, Codex or Antigravity, with LangSlice
installed). Only the folder of section images is needed; anything you leave
blank, the agent decides or asks you about.

- **Images.** The folder of section images. *How to copy a folder's path*
  says how on Windows, macOS and Linux; the quotes Windows' "Copy as path"
  adds are taken off.
- **Atlas** and **cutting plane.** Every atlas BrainGlobe describes, grouped
  into mouse, rat and other, starting on `allen_mouse_25um`, with
  BrainGlobe's own description beneath it.
- **Sections.** Thickness and interval in micrometres (whole numbers), and the
  pixel size, needed only when the image files do not record it.
- **What to do.** Place along the atlas, align in the plane, warp to fit
  (the job's tasks; the first two are on to start with).
- **What you already know.** That the file names sort in cutting order; that
  the sections were cut straight (the cutting plane stays untilted); that
  every section is the right way up and unmirrored (no flips or quarter
  turns); and damaged sections, one per line as `file name: what is wrong`.
- **Notes for the agent.** Saved with the job; the registration agent reads
  them.
- **Advanced** (closed to start with): every tool the registration agent can
  use, grouped by task, each with a short description and its own name, all
  on. Turning a tool or one of its options off takes it away from the agent.
  The tools of a task you did not choose are greyed; so are the options that
  something you said in *What you already know* leaves nothing to do. Some
  tools cannot be turned off.

The message on the right updates as you type, and **Copy** puts it on the
clipboard. It is a short request and the job's `langslice-job FOLDER init`
command, holding only what you set:

````text
Please register my brain sections with LangSlice, following its register-brain skill: create the job with the init command below, then launch the registration and check the result. The command leaves out the pixel size (when the image files do not record it): choose it yourself, or ask me when the images do not tell you.

```bash
langslice-job /data/brain1 init \
  --tasks position,transform,nonlinear \
  --atlas allen_mouse_25um \
  --plane coronal \
  --thickness 40 \
  --interval 120 \
  --disable zoom,trace_borders \
  --frozen '{"*":["orientation"]}' \
  --damaged '{"s012.tif":"left cortex torn"}' \
  --fact 'The file names sort in the order the sections were cut.' \
  --notes 'Injection in the right striatum.'
```
````

Each line of the command is one of `init`'s settings (`langslice-job schema
init` lists them all): `--tasks` what to do, `--disable` the tools and
options turned off, `--frozen` what stays as it is on every section,
`--damaged` the damage notes, `--fact` the cutting order and `--notes` your
notes. A choice of tools the job would refuse (a task with none of the tools
that do it, a tool without the pictures it needs, `look` with no kind of
picture) shows in red instead of a message, with what to turn back on. The
command is written for bash; on Windows the agent adjusts its quoting.

## How it works

The window is a page served by `langslice gui` on `127.0.0.1` under a random
address, reachable only from this computer. The agent viewer watches the job folder's files
(`logs/events.jsonl`, `views.jsonl`, `trace/`, `state.json`,
`history/index.json`, `job.json`, `exports/spend.json`; see
[file formats](file_formats.md)) four times a second while a page looks at the
job, and serves only the folders it lists. The prompt generator's choices
(the tools, the tasks, the atlases) come from the installed LangSlice, so a
tool a newer version renames or removes is never offered.
