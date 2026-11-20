# The job commands: `langslice job`

`langslice job FOLDER COMMAND` is the command line of the session that sets a
registration up and checks it: a coding agent with a shell (Claude Code,
Antigravity, Codex) makes a job with it, writes the message that launches a
registration agent, reads the job's status and exports its results. It does
not register: a registration agent works through the
[MCP tools](mcp.md), and through scripts of its own with the
[Python library](../developers/library.md) when it has a shell. Every call
prints one JSON envelope.

## Start a job

```bash
langslice job sections/ init --tasks position,transform,nonlinear
langslice job sections/ launch --role register
langslice job sections/ status
langslice job sections/ export_maps
```

`FOLDER` can be the image folder or its job folder. `init` runs no agent.
The job contains identical `AGENTS.md` and `CLAUDE.md` reference cards;
`init` writes `TOOLS.md` beside them: every tool of the job with its
description and its arguments (type and default), the words `schema` gives
as JSON. `langslice job FOLDER schema init` lists the initialization flags.

| Command | What it does |
|---|---|
| `init` | Create the job for a folder of section images, or continue the one there |
| `launch --role register\|register-scripting` | The message that launches a registration agent |
| `status` | The job's sections, their positions, transforms and deformations, and whether it is submitted |
| `export_maps` | Write the job's maps and the QuickNII and VisuAlign files from its present state |
| `runs [ID]`, `wait [ID]` | A long call's background run |
| `env`, `python SCRIPT` | The job's own Python environment |
| `ops`, `schema [TOOL ...]` | What the command line runs, and every tool's description and arguments |

## Launch prompts

A registering agent's instructions are never in a command's output it reads
itself: they are the message that launched it. `langslice job FOLDER launch
--role register|register-scripting` prints that message as `result.prompt`
and saves the exact text in `logs/launches/<six-digit number>_<role>.md`; the
main session passes it unchanged as the registration subagent's whole prompt.
`register` (the agent with the MCP tools alone) gets one line naming the job
folder and `start_job`, then the job statement worded for MCP.
`register-scripting` (the agent with the MCP tools and a shell) gets the same
message with a paragraph on scripts after the opening's: the interpreter of
the job's own Python environment (`launch` makes `env/` when it is missing),
the command that installs a package into it, how a script opens the job
(`langslice.open_job`: every tool as a method, `job.context` for a method of
its own), that the tools and the scripts see each other's writes, and where
its files go (`scripts/`, `scratch/`). Run `launch` again for each launch: it
reflects the job as it stands. The subagent's system prompt is one fixed
text, the same for every job and for the built-in agent.

Every job has the in-tree engines' tools (STalign, SPOmiAlign, emlddmm;
DeepSlice and Brainways with `init --enabled deepslice_place` or
`--enabled brainways_ap`) after the others in `schema`, `TOOLS.md` and the
reference card. A job made with tools of your own
([plugins](../developers/plugins.md): `langslice linear run FOLDER --plugins
MODULE:FACTORY`, or `create_job(plugins=...)` in Python) has those after them.

## The tools are not commands

`langslice job FOLDER TOOL` runs `status`, `export_maps` and `review_comments`
(the comments left in the LangSlice window, with their pictures saved as
files) only. Any other
tool (`look`, `position_sections`, the fits, `undo`, `submit` and the rest)
is refused with `TOOL_OFF`, its message saying where it is: a registration
agent's MCP tool, or `langslice.open_job(FOLDER).TOOL(...)` in a script.
`langslice job FOLDER ops` lists what the command line runs.

Tool names and arguments are shared by MCP, the library and the built-in
agent. `langslice job FOLDER schema TOOL [TOOL ...]` gives the exact arguments
enabled for that job, its tool plugins' included: one tool's entry, or several (separate or
comma-separated) under `verbs`, every tool's without one. A FOLDER that holds
no job declares every argument, and a plugin's tool is unknown there. Sections
are named by image filename (a unique stem also works), never by number.

`export_maps` writes the job's maps from its present state, submitted or not
(`submit` writes no maps); with the maps it writes each traced section's
image-model reply to `exports/traces/<name>.npz`
([format](../reference/file_formats.md#tracesnamenpz)). It is a long call:

```bash
langslice job sections/ export_maps
langslice job sections/ export_maps --background
langslice job sections/ runs
langslice job sections/ wait
```

Called without `--background` it starts a background run and waits for it, at
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
and the command that waits again; a run whose process ended without an
answer is `INTERNAL` (exit 4). `--timeout SECONDS` sets how long it waits.
A detached run needs a shell that lets a process outlive the command that
started it; a sandbox that ends them when the command returns (Claude Code's)
loses every run (`INTERNAL`). `LANGSLICE_DETACH=0` runs the call in the
calling process, as one call, and `1` detaches it; unset, it is detached
except under Claude Code (`CLAUDECODE` set).

## Replies

Stdout contains one JSON envelope; diagnostics go to stderr:

```json
{"ok":true,"result":{"status":"ok"},"artifacts":[],"warnings":[],"next":[]}
```

`result` contains the reply; `--verbose` includes its full text and stack
rows. `artifacts` lists absolute file paths and kinds. A failed call has an
`error` with a `code` and a `message` that says what to do next (an
unknown flag adds a `fix`, the corrected call).

| Exit code | Meaning |
|---|---|
| 0 | Successful call, or a long run still running when the call or `wait` answered |
| 2 | Invalid command, arguments, section or job path |
| 3 | Job refusal or write-lock timeout |
| 4 | Internal error or background run that ended without an answer |

A coding agent that writes its own method (an affine, a displacement field, an
outside engine's answer) puts its result into the job through the library's
`job.context` ([your own method](../developers/library.md#your-own-method)); the job folder's
card points to it.

The library caps no pictures or sections per call: `look`,
`grep_atlas_view`, `make_image`, `make_mask` and the change tools
save and return every picture they draw, and `interactive_transform` and
`elastix_affine` take any number of sections. What still limits a call is the
job's own `transform.max_parallel` for those two, `fireants_syn`'s section cap
(20: each fitted section is held in memory until the call writes them all)
and any limit a tool plugin declares; the job card states them, and the
refusal (`BAD_ARGUMENT`) says why. (LangSlice's
own agent and MCP keep four pictures, and four transform sections, per call.)

## The job's own Python environment

`langslice job FOLDER env` makes `env/` in the job folder when it is missing
and answers with its `python`: a virtual environment that also imports every
package of the environment LangSlice runs in (LangSlice included), with its
own packages first. `env install PACKAGE [PACKAGE ...]` installs pip
packages into it (uv when it is on the PATH, which links each package from
its cache, so a package used by many jobs is stored once; else pip), and lists
the environment's packages, pinned, in `scripts/requirements.txt`.
`env remove` deletes `env/` and keeps that list; the next `env` rebuilds the
environment from it. LangSlice's own environment is never changed.

`langslice job FOLDER python SCRIPT [ARGS ...]` runs a script with that
python (making `env/` first when it is missing). A relative SCRIPT that is not
a file from the working folder is the job folder's, so `scripts/NAME.py` runs
the job's script from any folder (`sys.argv[0]` is then its absolute path).
Its stdout, stderr and exit
code are the script's, as `env/bin/python SCRIPT ARGS` gives them: `sys.argv`,
the working folder, the script's folder first on `sys.path`, then the
environment's own packages, then LangSlice's.

## Transcripts

`langslice session-end --harness claude-code|antigravity` is the hook the
Claude Code and Antigravity plugins run when a registration agent's session
ends. It reads the hook's JSON on stdin, finds the job folders the transcript
names (its `langslice job FOLDER ...` commands, `open_job(...)` calls and MCP
`start_job` calls, by `image_folder`) and
copies the transcript into each one's `trace/<harness>_<session id>.jsonl`,
beside the built-in agent's traces, inline image data replaced by its size.
When the transcript submitted the job (a `submit` command, call or MCP tool
call), or the job's own `state.json` says it is submitted (a submit the
transcript does not show: one made in a script, or through a shell variable),
the job's `env/` is removed (its package list kept).

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
| `register` | LangSlice's MCP tools only, no shell. The Claude Code and Antigravity plugins ship the MCP server; for Codex the plugin's `setup` skill writes it into Codex's `config.toml` with `[features] mcp_tool_list_changed = true`, so the job's tools appear after `start_job`. The server holds one job at a time: one `register` per session |
| `register-scripting` (`register_scripting` in Codex) | LangSlice's MCP tools, and a shell and file tools for scripts of its own with the [Python library](../developers/library.md), run with the job's own Python |

