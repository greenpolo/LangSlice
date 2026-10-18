# Agents, CLI and MCP

LangSlice gives an agent tools to inspect and register a stack of sections.
Choose who runs the conversation; every option uses the same registration
operations and saves work in a [job folder](file_formats.md).

| Interface | Who drives it | Entry point |
|---|---|---|
| Agent CLI | A coding agent with a shell, such as Claude Code, Antigravity or Codex | `langslice-job FOLDER VERB` |
| MCP | An agent in an MCP host, such as Claude Desktop | `langslice mcp` |
| Built-in agent | LangSlice's agent session | `langslice linear run FOLDER` |
| Python | A coding agent writing its own workflow | [Python library](library.md) |
| ABBA | The agent works on sections selected in ABBA | [ABBA setup and use](abba.md) |

Install from the source checkout with `conda env create -f environment.yml`,
then `conda activate langslice`. Atlases download on first use. For model
access, `langslice login` signs in with ChatGPT; API providers use
`OPENAI_API_KEY` or `GEMINI_API_KEY` / `GOOGLE_API_KEY` (also read from `.env`).
MCP uses its host's agent and account. An optional image model has its own
provider setting and credentials.

Claude models run in `langslice linear run` through the Anthropic API:
`--model anthropic-api/<model id>` (install the extra with
`pip install -e ".[anthropic]"`). The key is `ANTHROPIC_API_KEY`, else the
file `~/.langslice/anthropic_api_key`. `--reasoning` sets Claude's
effort (`low`, `medium`, `high`, `xhigh`, `max`) with adaptive thinking whose
summary is written to the run trace, and the system prompt and the newest
message are marked for prompt caching.

## Coding agents: `langslice-job`

Initialize a job, read its brief, then choose tools from the live schema:

```bash
langslice-job sections/ init --tasks position,transform,nonlinear --viewer codex
langslice-job sections/ brief
langslice-job ops
langslice-job schema look --job sections/
langslice-job sections/ look --mode positioning
```

`FOLDER` can be the image folder or its job folder. `init` runs no agent.
The job contains identical `AGENTS.md` and `CLAUDE.md` reference cards;
`brief` writes `BRIEF.md` and returns the opening pictures as file paths.
The coding agent opens those pictures with its image reader and decides what
to do next. Keep its scripts in the job's `scripts/` and other work in
`scratch/`.

A job made with [tool plugins](plugins.md) (`langslice linear run FOLDER
--tool-plugin MODULE:FACTORY`, or `create_job(plugins=...)` in Python) has
their tools after the native ones in `ops`, `schema` and the reference card,
and calls them by name like any other.

Tool names and arguments are shared with MCP and the built-in agent.
`langslice-job schema VERB --job FOLDER` gives the exact arguments enabled
for that job; `schema init` lists initialization flags. Sections are named
by image filename (a unique stem also works), never by number.

```bash
langslice-job sections/ position_sections --sections '[{"id":"s01.tif","position_mm":5.2}]'
langslice-job sections/ look --mode overlay --sections s01.tif --resolution 1200
langslice-job sections/ zoom --region CA1:left
langslice-job sections/ zoom --picture 12 --box 960,230,1300,940
```

Arguments can be separate flags, a JSON object with `--args '{...}'`, or a
file with `--args @arguments.json`. Separate flags override `--args`.
List arguments accept repeated flags or a JSON list; a list of numbers also
takes one comma-separated flag (`zoom --box 960,230,1300,940`); a list of
objects (`set_landmarks` pairs) goes through `--args '{...}'` or
`--args @pairs.json`. A boolean flag alone means true. Kebab-case and
snake_case are accepted. `grep_atlas` returns its rows by default.

Long calls can run detached:

```bash
langslice-job sections/ fireants_syn --sections s01.tif --background
langslice-job sections/ runs
langslice-job sections/ wait --timeout 600
```

`runs ID` and `wait ID` address a particular run. `--dry-run` previews a
write without saving; expensive fits are validated without running them.
`export_maps` is available through this CLI and Python to write the job's
maps from its present state, submitted or not (`submit` writes no maps);
with the maps it writes each traced
section's image-model reply to `exports/traces/<name>.npz`
([format](file_formats.md#tracesnamenpz)).

### Replies and pictures

Stdout contains one JSON envelope; diagnostics go to stderr:

```json
{"ok":true,"result":{"status":"ok"},"artifacts":[],"warnings":[],"next":[]}
```

`result` contains the tool reply; `--verbose` includes its full text and
stack rows. `artifacts` lists absolute file paths, kinds and picture numbers.
Open `view` images to assess alignment. `zoom` uses a saved picture number,
or the latest non-zoom picture when omitted. A failed call has an `error`
with a `code`, `message` and `fix`.

| Exit code | Meaning |
|---|---|
| 0 | Successful call |
| 2 | Invalid command, arguments, section or job path |
| 3 | Job refusal, write-lock timeout or background work still running at the wait timeout |
| 4 | Internal error or background run that ended without an answer |

`init --viewer claude` caps later pictures at 2000 px; `codex` and `openai`
cap them at 2048 px. `look --resolution N` requests a long edge of at least
128 px, bounded by the viewer and source. A clamped request is reported.
Picture resolution does not change the registration.

The CLI and the library cap no pictures or sections per call: `look`,
`grep_atlas_view`, `set_preprocessed_channel_properties` and the change tools
save and return every picture they draw, and `interactive_transform` and
`elastix_affine` take any number of sections. What still limits a call is the
job's own `transform.max_parallel` for those two, `fireants_syn`'s section cap
and any limit a tool plugin declares; the job card states them. (LangSlice's
own agent and MCP keep four pictures, and four sections, per call.)

### The job's own Python environment

`langslice-job FOLDER env` makes `env/` in the job folder when it is missing
and answers with its `python`: a virtual environment that also imports every
package of the environment LangSlice runs in (LangSlice included), with its
own packages first. `env install PACKAGE [PACKAGE ...]` installs pip
packages into it (uv when it is on the PATH, which links each package from
its cache, so a package used by many jobs is stored once; else pip), and lists
the environment's packages, pinned, in `scripts/requirements.txt`.
`env remove` deletes `env/` and keeps that list; the next `env` rebuilds the
environment from it. LangSlice's own environment is never changed.

### Transcripts

`langslice session-end --harness claude-code|antigravity` is the hook the
Claude Code and Antigravity plugins run when a registration agent's session
ends. It reads the hook's JSON on stdin, finds the job folders the transcript
names (its `langslice-job FOLDER ...` commands and `open_job(...)` calls) and
copies the transcript into each one's `trace/<harness>_<session id>.jsonl`,
beside the built-in agent's traces, inline image data replaced by its size.
When the transcript submitted the job, the job's `env/` is removed (its
package list kept). It never fails the session and prints `{}`.

Ready-made setups: [Claude Code](https://github.com/greenpolo/LangSlice/blob/main/connectors/claude-code/README.md),
[Antigravity](https://github.com/greenpolo/LangSlice/blob/main/connectors/antigravity/README.md)
and [Codex](https://github.com/greenpolo/LangSlice/blob/main/connectors/codex/README.md).

## MCP

Configure an MCP host to launch the `langslice` executable from its installed
environment, with `mcp` as its argument. In Claude Desktop, open Settings >
Developer > Edit Config and add this to `claude_desktop_config.json`, then
restart Desktop:

```json
{
  "mcpServers": {
    "langslice": {
      "command": "/absolute/path/to/langslice-env/bin/langslice",
      "args": ["mcp"]
    }
  }
}
```

Use the environment's `Scripts/langslice.exe` on Windows. The source
environment includes MCP; a minimal pip installation needs the `mcp` extra.

Create a job with `langslice-job FOLDER init`, then ask the host's agent to
call `start_job(image_folder="/absolute/path/to/FOLDER")`. For a job prepared
in ABBA, paste ABBA's copy prompt, which supplies `start_job(job_id=...)`.
The agent must read every `show_stack(page=...)` page before writing.
Tool replies include images directly; pages and replies have a size budget,
and a reply reports when its images were reduced to fit.

Opening an existing job keeps its saved settings and checkpoint. For a
folder without a job, the server's job flags set the defaults. Use one
conversation per job. The MCP host controls the conversation, model budget
and image history; LangSlice supplies the tools and submit checks.

## Built-in agent

```bash
langslice linear run sections/ --tasks position,transform
langslice linear run sections/ --tasks position,transform,nonlinear --image-provider none
langslice linear run sections/ --registration quicknii.json
```

One session handles the whole stack and ends at `submit`.
The default tasks are `position,transform`; `--registration` imports a linear
placement and defaults to `nonlinear`. The agent chooses its sequence of
tools, regions and fit settings. Nonlinear registration can fit to the stain
alone or use optional image-model border traces.

### Job settings

`langslice linear run --help` and `langslice-job schema init` are the complete
flag reference. The main choices are:

| Setting | Flags |
|---|---|
| Atlas and plane | `--atlas` (default `allen_mouse_25um`), `--plane` |
| Tasks | `--tasks position,transform,nonlinear`; include only the tasks wanted |
| Agent (`linear run` only) | `--model` (default `openai-oauth/gpt-6.1-sol`), `--reasoning` (OAuth default `medium`), `--no-debrief`, `--force-view` (change and fitting tools always return their picture), `--require-deformation` (Nonlinear: `submit` is refused until every section carries a deformation) |
| Optional image model | `--image-provider openai-oauth\|openai-api\|gemini-api\|none`, `--image-model`, `--image-prompt FILE` (the base prompt `trace_borders` sends, in place of LangSlice's own) |
| Calibration and section spacing | `--pixel-size-um`, `--thickness`, `--interval`, `--strict-interval` |
| Supplied placement | `--registration`, or `--positions`, `--transforms`, `--orientation`, `--section-angles`, `--pitch`, `--yaw` |
| Constraints and context | `--hemisphere-cue`, `--frozen`, `--damaged`, `--fact`, `--position-notes`, `--transform-notes`, `--nonlinear-notes` (the user's notes for a task, shown to the agent with it) |
| Tools and options off | `--disable NAME[,NAME...]`: a tool (`zoom`) or `tool:option` (`look:positioning`, `interactive_transform:flip`); [valid names](registration.md#switching-tools-and-options-off) |
| Pictures | `--image-resolution low\|medium\|high\|auto`, `--preprocess auto\|none` |
| Storage | `--job-dir`, `--out`, `--fresh` (`linear run`), `--trace-dir` (`linear run`, `mcp`) |

`--pixel-size-um` is one number for every file, or a JSON mapping (inline or a file) of filenames to micrometres per pixel; a file it does not name reads its own TIFF/OME tags. `--thickness` and `--interval` are optional, in micrometres. Omit unknown
values: the agent infers positions and spacing from anatomy and records
estimates or uncertainty. Supplied values are advisory unless
`--strict-interval` is enabled; that option requires `--interval`.

Structured placement inputs take inline JSON or a JSON file. `--angles`
lets the agent adjust cutting angles. A disabled task keeps the supplied
answer and removes its tools; the [registration guide](registration.md)
explains these constraints. `init --notes TEXT` saves instructions for the
agent with the job.

`--image-prompt FILE` replaces the base prompt `trace_borders` sends the
image model for this job (`NonlinearSpec.image_prompt`, kept in the job's
settings); `{plane}` in it becomes the section plane. Start from
`langslice.default_prompt(provider, plane)`. It must describe the provider's
attachment order: Image 1 the clean photograph and Image 2 the placed
borders for the OpenAI providers, the other way round otherwise. The agent
is shown it as the base it may edit per section, and its traces are recorded
as an untested profile (`custom`).

An interrupted job resumes from its checkpoint. Resuming with different
supplied inputs is refused with `INPUTS_CHANGED`, and `init` with other job
settings than the saved job's with `SETTINGS_CHANGED`; `--fresh` starts the job
over. Every write is checkpointed and undoable. Calls through different
interfaces pick up each other's saved changes.

The built-in agent prints token usage per model call. LangSlice sets no limit
on model turns, tool calls, tokens or time, and nothing stops a run for what it
has spent. Spend is reported to whoever runs LangSlice, never to the model:

- On the ChatGPT OAuth lane, which reports the account's usage window, a
  `[spend]` warning is logged each time this run's share of the window (its
  rise since the run's first reading) passes another 10 percentage points.
- On the API-key lanes (`anthropic-api`, `openai-api`, `gemini-api`) a running
  dollar estimate from per-token list prices (input, cached input, output;
  thinking tokens as output; Claude input that is not a cache read as a cache
  write) is warned at $5, $10, $20, $40 and so on, doubling. A model without a
  price in `agent/spend.py` is said so once, with no estimate.
- Each warning also reaches the live event stream as a `spend` event (the ABBA
  agent viewer shows it) and the trace as a `spend` record. The run's final
  account is written to `exports/spend.json` in the job folder, also when the
  session fails part-way, and is the `spend` field of the trace's `summary`
  and of the `complete` event.

## Logs and credentials

The built-in agent always writes a JSONL trace: `--trace-dir`, then
`LANGSLICE_TRACE_DIR`, then the job's `trace/` folder. It records model and
tool events; images are described by type, size and byte count rather than
embedded. MCP traces record calls and results, not the host model's private
conversation. The agent CLI logs every call in `logs/calls.jsonl` and uses
`LANGSLICE_TRACE_DIR` for additional tool traces. The Claude Code and
Antigravity plugins copy a registration agent's own transcript into the
job's `trace/` folder when its session ends (Transcripts, above).

The `openai-oauth` agent transport automatically retries connection failures,
timeouts, HTTP 408/409/429 and 5xx errors (including 507), and early
`response.failed` events reporting server overload, server errors or temporary
rate limits. HTTP and event failures share a budget of three retries; event
failures are retried only before any model output arrives. It logs each retry and waits
2, 4 and 8 seconds with jitter, honoring a longer `Retry-After` up to 60
seconds. Longer server delays, explicit quota exhaustion, certificate errors
and other permanent failures stop the run. A 401 can refresh credentials once.
Stream failures are surfaced without replaying partially delivered tool calls;
image edits are not covered by these request retries.

`langslice login` stores OAuth credentials in
`~/.langslice/openai_auth.json`; `LANGSLICE_OPENAI_AUTH` overrides that path.
LangSlice uses its own login. ABBA setup can also save API keys in
`~/.langslice/provider_credentials.json`; explicit environment variables
take precedence. Credential presence is checked offline when deciding
whether to offer `trace_borders`.
