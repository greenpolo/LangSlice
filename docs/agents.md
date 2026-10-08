# Agents, CLI and MCP

LangSlice gives an agent tools to inspect and register a stack of sections.
Choose who runs the conversation; every option uses the same registration
operations and saves work in a [job folder](file_formats.md).

| Interface | Who drives it | Entry point |
|---|---|---|
| Agent CLI | A coding agent with a shell, such as Claude Code or Codex | `langslice-job FOLDER VERB` |
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

Tool names and arguments are shared with MCP and the built-in agent.
`langslice-job schema VERB --job FOLDER` gives the exact arguments enabled
for that job; `schema init` lists initialization flags. Sections are named
by image filename (a unique stem also works), never by number.

```bash
langslice-job sections/ position_sections --sections '[{"id":"s01.tif","position_mm":5.2}]'
langslice-job sections/ look --mode overlay --sections s01.tif --resolution 1200
langslice-job sections/ zoom --region CA1:left
```

Arguments can be separate flags, a JSON object with `--args '{...}'`, or a
file with `--args @arguments.json`. Separate flags override `--args`.
List arguments accept repeated flags or a JSON list; a boolean flag alone
means true. Kebab-case and snake_case are accepted.

Long calls can run detached:

```bash
langslice-job sections/ ants_syn --sections s01.tif --background
langslice-job sections/ runs
langslice-job sections/ wait --timeout 600
```

`runs ID` and `wait ID` address a particular run. `--dry-run` previews a
write without saving; expensive fits are validated without running them.
`export_maps` is available through this CLI and Python to export the job's
present state without submitting it.

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

Ready-made setups: [Claude Code](https://github.com/greenpolo/LangSlice/blob/main/connectors/claude-code/README.md)
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

One session handles the whole stack and ends at `submit` or a budget limit.
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
| Agent | `--model` (default `openai-oauth/gpt-6.1-sol`), `--reasoning` (OAuth default `medium`) |
| Optional image model | `--image-provider openai-oauth\|openai-api\|gemini-api\|none`, `--image-model` |
| Calibration and section spacing | `--pixel-size-um`, `--thickness`, `--interval`, `--strict-interval` |
| Supplied placement | `--registration`, or `--positions`, `--transforms`, `--orientation`, `--section-angles`, `--pitch`, `--yaw` |
| Constraints and context | `--no-flip`, `--hemisphere-cue`, `--locked`, `--damaged`, `--fact` |
| Pictures | `--image-resolution low\|medium\|high\|auto`, `--preprocess auto\|none` |
| Storage | `--job-dir`, `--out`, `--trace-dir`, `--fresh` |

`--thickness` and `--interval` are optional, in micrometres. Omit unknown
values: the agent infers positions and spacing from anatomy and records
estimates or uncertainty. Supplied values are advisory unless
`--strict-interval` is enabled; that option requires `--interval`.

Structured placement inputs take inline JSON or a JSON file. `--angles`
lets the agent adjust cutting angles. A disabled task keeps the supplied
answer and removes its tools; the [registration guide](registration.md)
explains these constraints. `init --notes TEXT` saves instructions for the
agent with the job.

An interrupted job resumes from its checkpoint. Resuming with different
supplied inputs is refused with `INPUTS_CHANGED`; `--fresh` starts the job
over. Every write is checkpointed and undoable. Calls through different
interfaces pick up each other's saved changes.

The built-in agent prints token usage per model call.
There is no limit on model turns or tool calls.
`--max-quota-percent` (default 25) limits the run's share of a provider usage
window when the provider reports it. `--max-input-tokens` limits the input
reported by one request. A budget stop allows a final submit call and keeps
completed writes.

## Logs and credentials

The built-in agent always writes a JSONL trace: `--trace-dir`, then
`LANGSLICE_TRACE_DIR`, then the job's `trace/` folder. It records model and
tool events; images are described by type, size and byte count rather than
embedded. MCP traces record calls and results, not the host model's private
conversation. The agent CLI logs every call in `logs/calls.jsonl` and uses
`LANGSLICE_TRACE_DIR` for additional tool traces.

The `openai-oauth` agent transport automatically retries connection failures,
timeouts, HTTP 408/409/429 and 5xx errors (including 507) up to three times
before a successful response stream starts. It logs each retry and waits
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
