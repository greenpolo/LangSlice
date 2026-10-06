# LangSlice

LangSlice registers histological brain sections to BrainGlobe atlases. An agent
orders the sections, places each at its atlas position and aligns it in-plane;
optionally it then deforms the atlas onto each section with ANTs or Elastix,
choosing the regions and settings itself and, where it chooses, reading
borders an image model has traced on the tissue. The README has the install and
the ways to use it.

## Using LangSlice

- [Install in ABBA](abba_installation.md): the Fiji connector and the Python environment.
- [CLI usage](cli.md): `langslice linear run`, the other commands, traces.
- [Agent CLI](agent_cli.md): `langslice job FOLDER VERB` for Claude Code, Codex and scripts.
- [Python library](library.md): `open_job`, `create_job`, `register_section`, `register_job`.
- [File formats](file_formats.md): `registration.json`, the coordinate and label maps, the QuickNII / VisuAlign exports, importing a registration.

## How it works

- [Architecture](architecture_overview.md): the layers and how the doors share one job.
- [The linear job](linear_design.md): job spec, state, tools, submit gates.
- [Nonlinear](nonlinear_design.md): the agent's deformation toolbox, with the optional image model.
- [Fiji connector and worker](abba_plugin_design.md): the worker protocol and the dialog-to-spec mapping.

## Repository

- `src/langslice/`: the package (`core`, `job`, `ops`, `doors`, `agent`, `providers`, `hosts`, plus compatibility shims).
- `connectors/`: the Fiji connector, the Claude Desktop MCP configuration, the Claude Code and Codex setups.
- `packaging/`, `environment.yml`: the worker's source installation.
- `tests/`: pytest.
