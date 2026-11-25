# `hosts/gui/`: the LangSlice window

A loopback page with the agent viewer (Simple and Extended views, 2D and 3D),
the prompt generator and the Settings tab. The page is a React app in `web/`
built into `static/` (fixed names, no inline script or style, so the content
policy stays `'self'`); the build is committed, so installing LangSlice needs no
Node. The user's page is `docs/developers/gui.md`.

## Layout

- `server.py` (the loopback HTTP server), `window.py` and `app.py` (the
  window around the page), `__init__.py` (`serve`).
- `watch/`: what the window learns from a job folder's files (`jobs.py`, the
  agent's focus, its words from the transcript, the review comments).
- `pictures/`: the pictures the window shows, drawn from the core's layers
  (`render.py` queues and keeps them; the looks `drawing`, `calm`, `paper`),
  the 3D view's cut-outs and numbers (`cutout.py`, `space.py`).
- `forms/`: the prompt generator (its choices and its message), the Settings
  tab and the Export menu.
- `web/src/`: `data/` (the polled store and the picture loader), `ui/`
  (controls, motion, the window's CSS), `stage/` (what the views share: the
  section pictures, the cutting axis, the speaker, the 3D stage), `views/`
  (Viewer, Simple, Extended, Review), `panels/` (prompt generator, Settings,
  Export).
- The page computes nothing the library knows: the prompt generator's
  message and checks, the export formats, the 3D view's frame and the
  atlas's cross-sections all come from the server.

## Rules

- The window is a separate process from every run (the built-in agent, an MCP
  server, the agent CLI's calls, a benchmark subprocess). It learns everything
  by watching job folders' files: no socket, no relay, no import of a run. It
  reads a job as the agent CLI does, never opens it, and nothing imports it but
  `hosts/cli.py`.
- It changes no registration. Its only writes: the Settings tab's three POSTs
  (the user's settings file, a saved key, the ChatGPT sign-in), the review
  comments (`review.json`, through `job.review`) and the Export menu (a job's
  `exports/` folder and its `registration.json`, never its state). Every other
  endpoint is GET. Sign-in runs in the window's process, never inside an agent
  tool call. A key is never sent back to the page.
- Nothing in the window is model-facing. The prompt generator's message is the
  person's to send, and its labels (`job.settings.labels`) are for people.
- The viewer draws from the same layers and the one tissue mask as everything
  else (`core.pictures`, `core.sections.tissue`); it adds no segmentation and no
  registration arithmetic of its own.
- A section is never drawn see-through in 3D, and a section still at the job's
  initial spacing (`position_source` `initial`) is not drawn there.
- Look: neutral graphite around greyscale pictures with one accent colour; the
  agent's words in a serif face, the window's in a sans-serif face
  (`static/tokens.css`). Motion values live in `web/src/ui/motion.ts` and
  `tokens.css` and are the same family.
- The chosen connection's `runs_in` in the user's settings is saved but read by
  nothing; the agent model and effort are read by `langslice linear run` when
  neither `--model` nor `--reasoning` is given.
- Pictures are read through the page's polled store (`web/src/data/run.ts`): the job
  list every 2 s, the shown job every 300 ms. Do not add a second channel.
