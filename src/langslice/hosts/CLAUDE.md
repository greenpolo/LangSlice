# LangSlice `hosts/` — host connectors in LangSlice's own environment

Package guide for `src/langslice/hosts/`. The repo-level `CLAUDE.md` holds the
project-wide rules. `AGENTS.md` here is a verbatim copy — edit one, mirror to
the other.

The top layer of the layered core: code that connects LangSlice to another
program and runs in LangSlice's own Python environment. What is installed into someone else's program lives in
`connectors/` at the repo root instead (the Fiji/ABBA connector, the Claude
Desktop configuration).

## The layer rule

A host module may import every layer below it: the core, the job layer,
the tools, the doors (the built-in agent among them) and the providers. Nothing
below imports a host (import-linter's layers contract, `pyproject.toml`,
run by `tests/test_import_layers.py`, no exceptions listed). The `langslice`
command reaches the host commands by module path only (`hosts/cli.py`,
`doors.commands.parser.HOST_COMMANDS`).

## Files and sub-packages

- `cli.py` — the host commands `gui` (the LangSlice window on job folders:
  `FOLDER ...`, an image folder standing for its `langslice/` job folder;
  `--port`, `--no-window`, `--tab viewer|prompt`, the prompt generator filled
  in from the first folder's job) and `serve` (the engine service),
  registered with the `langslice` command by module path.

- `gui/` — the LangSlice window (`langslice gui`, `linear run --gui`): a
  loopback page with the agent viewer, which learns everything by watching
  job folders' files and draws each section's current registration from
  them, the prompt generator, which writes the message (a
  request and a `langslice-job FOLDER init` command) a person pastes into
  their agent app, and the Settings tab (connections, run location, agent
  model), which writes only the user's settings and credentials; the viewer
  and the generator are read-only. Only `cli.py` imports it.
  `gui/CLAUDE.md`, `docs/developers/gui.md`.

- `service.py` — `langslice serve --stdio`: the JSON-lines engine
  service the Fiji connector starts (protocol 3: `setup.status`,
  `abba.prepare`, `abba.follow`). A long request (`abba.follow`, the
  follower that mirrors a job's checkpoints into ABBA) runs until the
  service's input closes: a thread reads the rest of the input and sets the
  request's stop event at its end. The protocol models, the preparation of
  a job from ABBA's snapshots, the follower and its checkpoint rows are
  door-level, in `doors/abba/`. `docs/developers/architecture.md`,
  `docs/abba.md`.
