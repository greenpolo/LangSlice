# Agents, CLI and MCP

LangSlice gives an agent tools to inspect and register a stack of sections.
Choose who runs the conversation; every option uses the same registration
operations and saves work in a [job folder](file_formats.md).

| Interface | Who drives it | Entry point |
|---|---|---|
| Agent CLI | A coding agent with a shell, such as Claude Code, Antigravity or Codex | `langslice-job FOLDER VERB` |
| MCP | An agent in an MCP host, such as Claude Desktop, or a coding agent's `register` role | `langslice mcp` |
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

Initialize a job, then choose tools from the live schema:

```bash
langslice-job sections/ init --tasks position,transform,nonlinear --viewer codex
langslice-job sections/ ops
langslice-job sections/ schema look zoom fireants_syn
langslice-job sections/ look --mode positioning
```

`FOLDER` can be the image folder or its job folder. `init` runs no agent.
The job contains identical `AGENTS.md` and `CLAUDE.md` reference cards;
`init` writes `TOOLS.md` beside them: every verb of the job with its description and
its arguments (type and default), the words `schema` gives as JSON, so an
agent reads them once from a file instead of one `--help` per verb.
`langslice-job FOLDER ops` lists the job's own verbs (`langslice-job ops`
lists every verb).
A registering agent's instructions are never in a command's output it reads
itself: they are the message that launched it. `langslice-job FOLDER launch
--role register|register-scripting` prints that message as `result.prompt`
and saves the exact text in `logs/launches/<six-digit number>_<role>.md`; the
main session passes it unchanged as the registration subagent's whole prompt.
`register` (the MCP subagent) gets one line naming the job folder and
`start_job`, then the job statement worded for MCP; `register-scripting` gets
one line naming the folder, then the statement worded for the command line,
whose opening paragraph lists the opening pictures (drawn and saved now as the
job's opening views, each strip's text then the picture's path) in reading
order. `launch` also writes `TOOLS.md`. Run it again for each launch: it
reflects the job as it stands. The subagent's system prompt is one fixed text,
the same for every job and for the built-in agent.
The coding agent opens those pictures with its image reader and decides what
to do next. Keep its scripts in the job's `scripts/` and other work in
`scratch/`, and run them with `langslice-job FOLDER python scripts/NAME.py`.

A job made with [tool plugins](plugins.md) (`langslice linear run FOLDER
--tool-plugin MODULE:FACTORY`, or `create_job(plugins=...)` in Python) has
their tools after the native ones in `ops`, `schema` and the reference card,
and calls them by name like any other.

Tool names and arguments are shared with MCP and the built-in agent.
`langslice-job FOLDER schema VERB [VERB ...]` (or `langslice-job schema VERB
... --job FOLDER`) gives the exact arguments enabled for that job, its tool
plugins' verbs included: one verb's entry, or several verbs' (separate or
comma-separated) under `verbs`, every verb's without one. Without a job a
plugin's verb is unknown, and the refusal says how to name the job.
`schema init` lists initialization flags. Sections are named
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
snake_case are accepted. `grep_atlas` returns its rows by default. An
unknown argument is refused (`UNKNOWN_ARGUMENTS`) with a `fix` that lists
every argument the verb takes, with the first sentence of its description
and its default, and the call without the unknown ones (for a verb of five
arguments or fewer, the others at their defaults); no other names are
accepted.

Long calls run detached:

```bash
langslice-job sections/ fireants_syn --sections s01.tif
langslice-job sections/ fireants_syn --sections s01.tif --background
langslice-job sections/ runs
langslice-job sections/ wait
```

A long verb (the fits, `trace_borders`, `export_maps`, every plugin tool)
called without `--background` starts a background run and waits for it, at
most 30 seconds: it answers with the run's answer (its `result` carries `run`),
or, while the run still runs, with `ok` true, `result` the run's `id`,
`state` `running`, `waited_s` and a message, and `next` the command that
waits again. The run goes on whatever happens to the command, so a harness
that ends a command it has left running (Codex does, about ten seconds after
the agent's own code stops waiting on it, however long the command had run)
ends only the waiting; `runs` lists the run and `wait` collects it. Ctrl-C on
the call ends the run too. `--background` answers at once with the run id.
`runs ID` and `wait ID` address a particular run. `wait` answers within 30
seconds the same way: the run's answer, or its state `running` (`ok`, exit 0)
and the command that waits again, so an agent polls instead of holding a
command its harness may end as too long; a run whose process ended without an
answer is `RUN_LOST` (exit 4). `--timeout SECONDS` sets how long it waits.
A detached run needs a shell that lets a process outlive the command that
started it; a sandbox that ends them when the command returns (Claude Code's)
loses every run (`RUN_LOST`). `LANGSLICE_DETACH=0` runs a long verb called
without `--background` in the calling process, as one call, and `1` detaches
it; unset, it is detached except under Claude Code (`CLAUDECODE` set), whose
agents run a long verb in the harness's own background mode (the Claude Code
agents say so). `--dry-run` previews a
write without saving; expensive fits (`elastix_affine`, `fireants_syn`, `trace_borders`
and every plugin tool, engines included) are validated without running them.
`export_maps` is available through this CLI and Python to write the job's
maps from its present state, submitted or not (`submit` writes no maps);
with the maps it writes each traced
section's image-model reply to `exports/traces/<name>.npz`
([format](file_formats.md#tracesnamenpz)).

### Replies and pictures

Stdout contains one JSON envelope; diagnostics go to stderr:

```json
{"ok":true,"picture_files":["/job/views/000012_look/view.jpg"],"result":{"status":"ok"},"artifacts":[{"path":"/job/views/000012_look/view.jpg","kind":"view","index":0}],"warnings":[],"next":[]}
```

`result` contains the tool reply; `--verbose` includes its full text and
stack rows. `artifacts` lists absolute file paths, kinds and picture numbers.
`picture_files`, second in the envelope and only when the call drew pictures,
repeats the paths of the pictures to open (`view` and `opening` JPEGs, in
order), so an agent reading the reply finds them first; the envelope's other
keys are unchanged. A `look` with `--sheet` attaches its sheets: the pictures
they tile are saved and numbered too, listed as artifacts of kind `tile`, not
in `picture_files`. Open `view` images to assess alignment. `zoom` uses a saved picture number,
or the latest non-zoom picture when omitted. A failed call has an `error`
with a `code`, `message` and `fix`.

| Exit code | Meaning |
|---|---|
| 0 | Successful call, or a long run still running when the call or `wait` answered |
| 2 | Invalid command, arguments, section or job path |
| 3 | Job refusal or write-lock timeout |
| 4 | Internal error or background run that ended without an answer |

What the registration engines print while they fit (the GPU compiler's
warnings on the first FireANTs fit, torch's memory allocator, Elastix) goes to
the job's `logs/engine.log`, never to the call's stderr or reply.

`init --viewer claude` caps later pictures at 2000 px; `codex` and `openai`
cap them at 2048 px. `look --resolution N` requests a long edge of at least
128 px, bounded by the viewer and source. A clamped request is reported.
Picture resolution does not change the registration.

A coding agent that writes its own method (an affine, a displacement field, an
outside engine's answer) puts its result into the job through the library's
`job.context` ([your own method](library.md#your-own-method)); the job folder's
card points to it.

The CLI and the library cap no pictures or sections per call: `look`,
`grep_atlas_view`, `make_image`, `make_mask` and the change tools
save and return every picture they draw, and `interactive_transform` and
`elastix_affine` take any number of sections. What still limits a call is the
job's own `transform.max_parallel` for those two, `fireants_syn`'s section cap
(20: each fitted section is held in memory until the call writes them all)
and any limit a tool plugin declares; the job card states them, and the
refusal (`TOO_MANY_SECTIONS`) says why. (LangSlice's
own agent and MCP keep four pictures, and four transform sections, per call.)

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

`langslice-job FOLDER python SCRIPT [ARGS ...]` runs a script with that
python (making `env/` first when it is missing). A relative SCRIPT that is not
a file from the working folder is the job folder's, so `scripts/NAME.py` runs
the job's script from any folder (`sys.argv[0]` is then its absolute path).
Its stdout, stderr and exit
code are the script's, as `env/bin/python SCRIPT ARGS` gives them: `sys.argv`,
the working folder, the script's folder first on `sys.path`, then the
environment's own packages, then LangSlice's. It is the faster way to run
one: in the job's warm process (below) the script starts with LangSlice, the
engines and the atlas already loaded.

### The warm process

Each `langslice-job FOLDER ...` call is a process of its own, and most of a
cold call's time is starting: importing ITK's Elastix, torch and FireANTs,
and loading the atlas. The first call on a job folder runs cold and starts
the job's warm process, detached; it loads all of that once, computes
every placed section's frame, and then answers later
calls on the folder by forking a child per call. The child runs the cold
call's own code with the caller's arguments, working folder, environment
variables and standard streams, so a warm call writes what a cold one writes
on stdout and stderr, ends with the same exit code, takes the job folder's
lock as its own process, and runs in parallel with other calls. Ctrl-C and
`SIGTERM` sent to the call reach the child; a call whose `langslice-job`
process is killed is killed too. `init`, `ops` and `schema` always run cold,
and so does a call whose environment differs where the loaded libraries read
it at start (`PYTHON*`, `OMP_*`, `CUDA_*`, `TORCH*`, `HOME` and similar), or
whose environment's own packages shadow ones the warm process has loaded
(a `python` script, which then runs in its environment's python).

The warm process never runs stale code: before each call it checks the
files of the modules it loaded from outside the standard library and
site-packages (LangSlice's own, a tool plugin's), the import path's folders
and the atlas's folder. When one changed, that call runs cold, the warm
process stops taking calls and exits once its running calls end, and the
call starts a new one. It exits after 10 minutes without a call, or one minute once the job is submitted. It never
touches the GPU, so each call initialises CUDA itself as a cold call does.
`LANGSLICE_WARM=0` runs every call cold. It is Linux only; elsewhere every
call is cold. Its socket, lock and log are in `$XDG_RUNTIME_DIR/langslice-warm/`
(else `/tmp/langslice-warm-<uid>/`), readable by the user alone, and it
serves only that user's processes. `logs/calls.jsonl` marks each call `warm`
or not.

### Transcripts

`langslice session-end --harness claude-code|antigravity` is the hook the
Claude Code and Antigravity plugins run when a registration agent's session
ends. It reads the hook's JSON on stdin, finds the job folders the transcript
names (its `langslice-job FOLDER ...` commands, `open_job(...)` calls and MCP
`start_job` calls, by `image_folder` or a saved job's `job_id`) and
copies the transcript into each one's `trace/<harness>_<session id>.jsonl`,
beside the built-in agent's traces, inline image data replaced by its size.
When the transcript submitted the job (a `submit` command, call or MCP tool
call), the job's `env/` is removed (its package list kept).

Beside each copy it writes `trace/<harness>_<session id>.launch.json`: the
agent's first user message compared with the job's saved launch prompts
(`logs/launches/`). `verbatim` is true when it equals one (trailing whitespace
aside); `launch` names that file, or the closest; `similarity` is difflib's
ratio over the words; `first_difference` is the character index where they
part (null when verbatim); `expected_chars` and `got_chars` are the lengths.
It is `{"checked": false, "reason": ...}` when the job has no saved launch
prompt or the transcript has no first user message that can be read (Claude
Code's user records and Antigravity's `USER_INPUT` steps are read). It never
fails the session and prints `{}`.

Ready-made setups: [Claude Code](https://github.com/greenpolo/LangSlice/blob/main/connectors/claude-code/README.md),
[Antigravity](https://github.com/greenpolo/LangSlice/blob/main/connectors/antigravity/README.md)
and [Codex](https://github.com/greenpolo/LangSlice/blob/main/connectors/codex/README.md).
Each gives the coding agent two registration roles, each an agent file in the
connector:

| Role | Tools |
|---|---|
| `register` | LangSlice's MCP tools only, no shell. The Claude Code and Antigravity plugins ship the MCP server; for Codex it goes in the session's `config.toml` with `[features] mcp_tool_list_changed = true`, so the job's tools appear after `start_job`. The server holds one job at a time: one `register` per session |
| `register-scripting` (`register_scripting` in Codex) | a shell, `langslice-job` and the [Python library](library.md) |

## MCP

Configure an MCP host to launch the `langslice` executable from its installed
environment, with `mcp` as its argument. The coding-agent connectors' `register`
role works through this server (Ready-made setups, above). The server
advertises that its tool list changes, so a host takes the job's tools once
`start_job` has opened it. In Claude Desktop, open Settings >
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

Create a job with `langslice-job FOLDER init`, then launch the host's agent
with `langslice-job FOLDER launch --role register`'s `result.prompt`, which
names the folder and holds the job statement. The agent calls
`start_job(image_folder="/absolute/path/to/FOLDER")`; its reply says only that
the job is open, its section count and the pages of its opening (the server's
instructions are one short text, the same for every job). For a job prepared
in ABBA, paste ABBA's copy prompt, which supplies `start_job(job_id=...)`; that
call answers with the whole job statement, as that path has no launching
message. The agent must read every `show_stack(page=...)` page before writing.
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

One session handles the whole stack and ends at `submit`. Its system prompt
is one short text, the same for every job (the coding-agent connectors use the
same text); the job statement, the user's notes, the opening pictures and the
status table are its first message.
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
| Watching (`linear run` only) | `--gui` opens the [LangSlice window](gui.md) on the job |

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
interfaces pick up each other's saved changes. A write waits for the job
folder's lock while another holds it (up to 300 s, then `JOB_BUSY`); a read
(`look`, `zoom`, `status`, the atlas and job-folder tools) never waits: it
reads the job as it was last saved.

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
conversation. The agent CLI logs every call in `logs/calls.jsonl` (the verb,
its arguments as given, the outcome, an error's code and message, the
artifacts, the seconds) and keeps each call's whole envelope, as printed, in
`logs/calls/<n>_<verb>.json`, which the call's line names under `reply`, so a
reply the harness cut short or the agent did not keep is still in the job; it
uses `LANGSLICE_TRACE_DIR` for additional tool traces. The Claude Code and
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
