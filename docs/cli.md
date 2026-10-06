# CLI usage

```text
langslice linear run FOLDER        the built-in agent over a folder of sections
langslice mcp                      the verbs as an MCP server (stdio)
langslice claude prepare FOLDER    save a job for Claude Desktop / Code and print its prompt
langslice abba                     start ABBA from Python with the connector, viewer and log
langslice serve --stdio            JSON-lines worker (what the Fiji connector starts)
langslice login                    sign in with ChatGPT (OAuth)
langslice version

langslice-job FOLDER VERB          one verb on a job folder (the agent CLI)
langslice-job ops | schema [VERB]  the verbs; one verb's arguments (JSON)
```

`langslice-job` is a command of its own: the MCP tools as shell commands,
for a coding agent ([agent_cli.md](agent_cli.md)). It starts no agent, server
or host and signs nobody in, so allowing it allows only work on job folders.
The commands live in `src/langslice/doors/cli/` and `src/langslice/hosts/cli.py`.
`langslice linear run --help` lists every flag.

## The agent run

```bash
langslice linear run sections/                       # order, position, in-plane transform
langslice linear run sections/ --tasks nonlinear     # image-model borders + deformable fit on a saved linear placement
langslice linear run sections/ --registration quicknii.json   # import a registration made elsewhere, then nonlinear
```

One agent session over the whole folder (a single section is a stack of
one) ends at `submit` or at a budget. `--tasks` picks which of `reorder`,
`position`, `transform` and `nonlinear` are on (default: the first three);
see [linear_design.md](linear_design.md). A task that is off takes its answer
from the host: `--order`, `--positions`, `--transforms`, `--orientation`,
`--section-angles`, `--pitch`/`--yaw`, `--pixel-size-um`, `--locked`,
`--damaged`, or a whole registration with `--registration`. Each takes a JSON
file path or inline JSON.

Common options:

| Option | Meaning |
|---|---|
| `--atlas`, `--plane` | BrainGlobe atlas (default `allen_mouse_25um`) and slicing plane |
| `--model`, `--reasoning` | agent model (`openai-oauth/<model>`, ...) and reasoning effort |
| `--image-provider`, `--image-model` | image model for `trace_borders`: `openai-oauth` (default), `openai-api`, `gemini-api`, or `none` (the deformable fit then reads the stain alone) |
| `--engine` | deformable engine: `ants`, `elastix` or `either` (the agent chooses) |
| `--image-resolution` | size of the pictures the agent sees: `low` (256 / 512 px long edge for opening tiles / later pictures), `medium` (384 / 768), `high` (512 / 1024), `auto` (the agent asks per call) |
| `--preprocess auto\|none` | display preprocessing of what the agent sees; image files are never modified |
| `--agent-preprocessing` | also offer the `preprocess` tool |
| `--angles`, `--bayesian` | let the agent set the cutting angles; offer `search_position` |
| `--no-flip`, `--hemisphere-cue`, `--thickness`, `--interval`, `--strict-interval`, `--fact` | facts and limits passed to the agent |
| `--job-dir`, `--out`, `--fresh`, `--trace-dir` | where the job folder and results go; start over; trace directory |
| `--max-quota-percent` (default 25), `--max-input-tokens`, `--gates`, `--playbook`, `--no-debrief` | budgets and run behavior |

A run that dies resumes from its checkpoint; resuming with different supplied
inputs is refused, naming the ones that differ. Everything a run writes goes
in the job folder `<image_folder>/langslice/` (`--job-dir` moves it; a
read-only image folder falls back to `~/.langslice/jobs/<id>/`):
`state.json` (the checkpoint, and the truth), `history/` (undo), `sections/`,
`views/`, `exports/` and `registration.json` ([file_formats.md](file_formats.md)).

Each model call prints a `[tokens]` line (request input, cached input,
output, cumulative input, peak). `--max-quota-percent` ends the session when
the run's share of the provider's usage window reaches that percent;
`--max-input-tokens` stops after one request reports more than N input tokens.
Both allow one final `submit`; writes made before the stop are kept.

## Traces

`langslice linear run` always writes a trace: to `--trace-dir PATH`, else
`LANGSLICE_TRACE_DIR`, else `<job folder>/trace`. One JSONL file per agent
session, `<run_label>_<8 hex>.jsonl`, one record per event: `session`, `seed`
(the first message; images as descriptors), `model` (text, thought summary,
function calls, token usage), `tool_result`, `nudge`, `summary`. Images are
never written, only mime type, byte count and size; no credentials are read.

## Other commands

- **Agent CLI**: [agent_cli.md](agent_cli.md). **Python library**:
  [library.md](library.md).
- **MCP / Claude**: [connectors/claude-desktop/README.md](https://github.com/greenpolo/LangSlice/blob/main/connectors/claude-desktop/README.md).
  `langslice claude prepare FOLDER` takes the same job flags as `linear run`.
- **ABBA**: [abba_installation.md](abba_installation.md). `langslice abba` needs
  Java 21 (fetched on first start) and the connector jar (`--connector-jar`
  or `$LANGSLICE_CONNECTOR_JAR`); `--no-viewer` / `--no-log` turn off the companions.
- **`langslice login`** runs the ChatGPT OAuth (PKCE) flow in a browser
  (callback on `localhost:1455`) and writes `~/.langslice/openai_auth.json`
  (mode 600). It is the only credentials file read; a Codex login is never
  used. `LANGSLICE_OPENAI_AUTH` names another file for one process. API keys
  (`OPENAI_API_KEY`, `GEMINI_API_KEY` / `GOOGLE_API_KEY`) come from the
  environment, a `.env` file (`.env.example`) or the ABBA setup dialog.
- **`langslice serve --stdio`** is a newline-delimited JSON service. Its
  methods are `version`, `setup.status`, `setup.login`, `setup.api_key`,
  `linear.run`, `linear.estimate`, `preprocess.preview` and `claude.prepare`;
  the contract is the Pydantic models in `src/langslice/doors/api/models.py`
  ([abba_plugin_design.md](abba_plugin_design.md)).
