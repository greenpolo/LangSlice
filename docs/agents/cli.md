# The agent CLI: `langslice-job`

`langslice-job FOLDER VERB` is the command line a coding agent with a shell
(Claude Code, Antigravity, Codex) uses to make a job, launch a registration
agent on it and call its tools. Every call prints one JSON envelope.

## Start a job

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

## Launch prompts

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

A job made with [tool plugins](../developers/plugins.md) (`langslice linear run FOLDER
--tool-plugin MODULE:FACTORY`, or `create_job(plugins=...)` in Python) has
their tools after the native ones in `ops`, `schema` and the reference card,
and calls them by name like any other.

## Calling verbs

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
([format](../file_formats.md#tracesnamenpz)).

## Replies and pictures

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
`job.context` ([your own method](../developers/library.md#your-own-method)); the job folder's
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

## The job's own Python environment

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

## The warm process

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

## Transcripts

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

## Ready-made setups

[Claude Code](https://github.com/greenpolo/LangSlice/blob/main/connectors/claude-code/README.md),
[Antigravity](https://github.com/greenpolo/LangSlice/blob/main/connectors/antigravity/README.md)
and [Codex](https://github.com/greenpolo/LangSlice/blob/main/connectors/codex/README.md).
Each gives the coding agent two registration roles, each an agent file in the
connector:

| Role | Tools |
|---|---|
| `register` | LangSlice's MCP tools only, no shell. The Claude Code and Antigravity plugins ship the MCP server; for Codex it goes in the session's `config.toml` with `[features] mcp_tool_list_changed = true`, so the job's tools appear after `start_job`. The server holds one job at a time: one `register` per session |
| `register-scripting` (`register_scripting` in Codex) | a shell, `langslice-job` and the [Python library](../developers/library.md) |

