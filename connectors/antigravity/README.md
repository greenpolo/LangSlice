# LangSlice plugin for Antigravity

Registers histology sections to a BrainGlobe atlas from Google's Antigravity
(the `agy` CLI, the Antigravity app and IDE), on the user's own Antigravity
account. Like the Claude Code plugin ([`../claude-code/`](../claude-code/README.md)),
it is skills for the main session plus two registration agents that each
register exactly one brain: `register` through LangSlice's MCP tools, and
`register-scripting` through the same MCP tools and scripts of its own with
the Python library. The main session sets a job up and checks it with the
`langslice-job` command line ([`docs/agents/cli.md`](../../docs/agents/cli.md)),
which runs no registration tool.

The registration skills (`registration-playbook`, `placement`,
`damage-and-masks`, `landmarks`, `choosing-engines`) teach the agents how to
register, with example pictures they open with `view_file`. Both agents list
`skills/registration-playbook` in their frontmatter `skills`. The skills' one
source is `src/langslice/skills/`; each connector carries a copy, checked by
`tests/test_connector_skills.py`.

## Install

LangSlice must be installed first (the conda environment of
`environment.yml`). Antigravity must then find `langslice` and
`langslice-job` on the PATH it reads from the login shell (`$SHELL -ilc`;
on Windows, the user and system environment variables): the plugin's MCP
server runs `langslice mcp`, the skills run `langslice-job`, and
the plugin's hook runs `langslice`. The `setup` skill (below) puts them
there.

```bash
agy plugin install https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
```

Or ask Antigravity's `/plugin` skill, in a chat, to install that URL. To try
it from a checkout: `agy plugin install connectors/antigravity`.

`agy plugin install` copies the folder; install again after a change, and
restart Antigravity. The plugin brings its own MCP server
([`mcp_config.json`](mcp_config.json): `langslice mcp` over stdio), started
while the plugin is enabled; no `agy mcp add` is needed. Antigravity prefixes
a plugin's servers with the plugin's name, so the server is `langslice_mcp`,
and its tools need the allow rule `mcp(langslice_mcp/*)` (below).

After installing and restarting, ask Antigravity to "Set up LangSlice.": the
plugin's `setup` skill finds `langslice` (on PATH, else in a conda
environment named `langslice`), appends its folder to PATH (a guarded line in
`~/.zprofile` and `~/.zshrc` for zsh; in `~/.bashrc` and the bash login file
for bash; the user `Path` on Windows), and adds the permission rules below to
`settings.json` with this machine's folders. Until then the MCP server fails
to start. Restart Antigravity again afterwards; then `start_job` with no
arguments tests the connection.

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| skill `register-brain` | main session | prepares the job from the user's choices, starts one registration agent with `invoke_subagent`, then checks the job and handles export; it never registers itself |
| skill `setup` | main session | once per computer: puts `langslice`'s folder on PATH and adds LangSlice's permission rules (below) to `settings.json` |
| skill `abba` | main session | jobs started from ABBA's Register › LangSlice…, ABBA 0.25's own commands, reaching a running ABBA through abba-mcp, an ABBA registration into a job |
| MCP server `langslice_mcp` | `register` | `langslice mcp`: `start_job`, `show_stack` and the job's tools, whose replies carry the pictures |
| agent `register` | registers one brain | the LangSlice MCP tools and `view_file`; no shell, no file writes |
| agent `register-scripting` | registers one brain | the LangSlice MCP tools, and `run_command`, `view_file`, `write_to_file`, `replace_file_content` and the web for scripts of its own with the LangSlice Python library, run with the job's own Python (`env/` in the job folder), into which it may install packages |
| `assets/logo.svg` | the plugin list | LangSlice's logo, a copy of `assets/logo/logomark.svg` |
| hook `langslice-transcript` | every session | at the end of a session, `langslice session-end --harness antigravity` copies the transcript into the job's `trace/` folder |

## The two agents

An agent's `tools` list is its whole toolset, whether the main session starts
it with `invoke_subagent` or a run starts it with `--agent`; the session's
other tools are not offered to it.

- `register` lists `view_file` and `send_message` (its reply to the session
  that started it) and sets `inheritMcp: true`, which gives it
  `call_mcp_tool` on the session's MCP servers, among them `langslice_mcp`.
  It has no `run_command` and no file-writing tool, so it works only through
  the LangSlice tools. A tool reply with several pictures reaches it with the
  last picture inline and the others saved under
  `~/.gemini/antigravity-cli/brain/<conversation>/.system_generated/`; the
  reply names each file, and `view_file` opens it.
- `register-scripting` lists the shell, file and web tools and inherits the
  MCP servers too (`inheritMcp: true`).

Both agents are `mainAgent` and `subagent`: the main session starts one with
`invoke_subagent`, and a run without a chat starts one directly, in the job
folder, with the launch prompt as its message:

```bash
cd /data/brains/M04/langslice     # the job folder (made by `langslice-job ... init`)
agy -p "$(langslice-job . launch --role register | jq -r .result.prompt)" \
  --agent register --model gemini-4-argon-high --output-format stream-json \
  > run.jsonl
```

`--agent register-scripting` runs the scripting agent the same way. `agy
models` lists the models. An agent's `model: inherit` takes the session's
model (`--model`, or the chat's).

## Sandbox and permissions

Antigravity runs an agent's commands in an OS sandbox (Linux namespaces):
the working folder and temporary folders are writable, the paths that
`read_file` rules allow are visible read-only, and network access is limited
to the domains `read_url` rules allow. `langslice-job` reads the atlas and the
images, so the main session runs it outside the sandbox (`BypassSandbox`),
which a `command(langslice-job)` rule allows without asking. A
`register-scripting` agent's script reads them too: it runs the job's own
Python (its launch message names the interpreter) outside the sandbox, which
asks each time unless a `command(...)` rule names that interpreter. The MCP server is a process Antigravity starts, not an agent's
command, so it runs outside the sandbox; an `mcp(langslice_mcp/*)` rule lets
an agent call its tools without asking.
Permissions live in one file, `~/.gemini/antigravity-cli/settings.json`, for
every session (a headless `agy -p` cannot ask, so anything not allowed is
denied). The settings these agents need, which the `setup` skill adds with
this machine's paths:

```json
{
  "enableTerminalSandbox": true,
  "toolPermission": "proceed-in-sandbox",
  "permissions": {
    "allow": [
      "mcp(langslice_mcp/*)",
      "command(langslice-job)",
      "read_url(*)",
      "read_file(/path/to/LangSlice)",
      "read_file(/path/to/the/python/LangSlice/runs/on)",
      "read_file(~/.brainglobe)"
    ]
  }
}
```

`register` needs only the `mcp(...)` rule. `read_url(*)` gives sandboxed
commands the web (package downloads, papers, documentation). The scripting
agent runs its scripts with the job's own Python, outside the sandbox when
allowed (above). The `read_file`
rules let a script run in the sandbox with the job's own environment's python
import LangSlice: its checkout or installation, the Python it runs on (for a
uv environment, `~/.local/share/uv/python`), and the atlases. Antigravity on
Linux ignores `unsandboxed(...)` rules.
