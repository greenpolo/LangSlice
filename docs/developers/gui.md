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
**Settings** (top right). The prompt generator is not built yet. The first time
the window opens with no connection chosen, it opens on Settings.

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

## Settings

Only you set these, in the window; no agent tool reads or changes them.

- **Connections.** How LangSlice reaches a model: your ChatGPT account (the
  **Sign in with ChatGPT** button opens the sign-in page in your browser), or
  an OpenAI, Claude (Anthropic) or Gemini API key pasted into its row. Each row
  says whether it is signed in, has a key saved, or has a key set in the
  environment. Choose one with its round button. Keys are saved on this
  computer in `~/.langslice/provider_credentials.json`, the sign-in in
  `~/.langslice/openai_auth.json` (`LANGSLICE_OPENAI_AUTH` names another
  file); a saved key is never shown again, and saving a new one replaces it.
  A key set in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`,
  `GEMINI_API_KEY`) wins over a saved one.
- **Registration runs in.** *This app* (your agent app runs registration) or
  *LangSlice, using the connection below* (LangSlice's own agent runs it). It
  is greyed until the chosen connection is signed in or has a key, and then
  starts at LangSlice. The choice is saved, but nothing acts on it yet: agent
  apps do not read it in this version.
- **Agent model.** The model and reasoning effort for the chosen connection,
  starting at LangSlice's default where it has one (the ChatGPT account's
  `gpt-6.1-sol`, also offered for the OpenAI API key). The Claude and Gemini
  keys have no LangSlice default: choose a model. `langslice linear run` uses
  the chosen connection's model and effort when it is started with neither
  `--model` nor `--reasoning` (`--reasoning` alone keeps the saved model);
  without a choice saved it runs on its own default as before.

The choices are saved in `~/.langslice/settings.json`
([file formats](../reference/file_formats.md)); `LANGSLICE_SETTINGS` names another file, or
`off` for none.

## How it works

The window is a page served by `langslice gui` on `127.0.0.1` under a random
address, reachable only from this computer. It watches the job folder's files
(`logs/events.jsonl`, `views.jsonl`, `trace/`, `state.json`,
`history/index.json`, `job.json`, `exports/spend.json`; see
[file formats](../reference/file_formats.md)) four times a second while a page looks at the
job, and serves only the folders it lists. The viewer's addresses only read;
the Settings tab's changes are the only requests that write, and the window
accepts them only from its own page.
