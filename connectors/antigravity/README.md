# LangSlice plugin for Antigravity

Registers histology sections to a BrainGlobe atlas from Google's Antigravity
(the `agy` CLI, the Antigravity app and IDE), on the user's own Antigravity
account. Like the Claude Code plugin ([`../claude-code/`](../claude-code/README.md)),
it is skills for the main session plus two registration agents that each
register exactly one brain: `register` through LangSlice's MCP tools, and
`register-scripting` through the `langslice-job` command line and the Python
library ([`docs/agents/cli.md`](../../docs/agents/cli.md)).

## Install

LangSlice must be installed first, with `langslice` and `langslice-job` on
the PATH of the shell that starts Antigravity (`langslice version` must work):
the plugin's MCP server runs `langslice mcp`, `register-scripting` runs
`langslice-job`, and the plugin's hook runs `langslice`.

```bash
agy plugin install connectors/antigravity      # from a checkout
agy plugin install https://github.com/greenpolo/LangSlice   # or from GitHub
```

`agy plugin install` copies the folder; install again after changing it, and
restart Antigravity. The plugin brings its own MCP server
([`mcp_config.json`](mcp_config.json): `langslice mcp` over stdio), started
while the plugin is enabled; no `agy mcp add` is needed. Antigravity prefixes
a plugin's servers with the plugin's name, so the server is `langslice_mcp`,
and its tools need the allow rule `mcp(langslice_mcp/*)` (below).

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| skill `register-brain` | main session | prepares the job from the user's choices, starts one registration agent with `invoke_subagent`, then checks the job and handles export; it never registers itself |
| skill `abba` | main session | ABBA's own 0.24.x commands, how to reach a running Fiji, moving files between ABBA and a job |
| MCP server `langslice_mcp` | `register` | `langslice mcp`: `start_job`, `show_stack` and the job's tools, whose replies carry the pictures |
| agent `register` | registers one brain | the LangSlice MCP tools and `view_file`; no shell, no file writes |
| agent `register-scripting` | registers one brain | `run_command`, `view_file`, `write_to_file`, `replace_file_content` and the web: the `langslice-job` command line, scripts with the LangSlice Python library, packages installed in the job's own environment (`langslice-job FOLDER env`) |
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
- `register-scripting` lists the shell, file and web tools and no MCP access.

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
images and calls the image model, so `register-scripting` runs it outside the
sandbox (`BypassSandbox`), which a `command(langslice-job)` rule allows
without asking. The MCP server is a process Antigravity starts, not an agent's
command, so it runs outside the sandbox; an `mcp(langslice_mcp/*)` rule lets
an agent call its tools without asking.
Permissions live in one file, `~/.gemini/antigravity-cli/settings.json`, for
every session (a headless `agy -p` cannot ask, so anything not allowed is
denied). The settings these agents need, with the paths of this machine
replaced by yours:

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
agent runs its scripts with `langslice-job FOLDER python SCRIPT`, a
`langslice-job` command, so they run outside the sandbox too. The `read_file`
rules let a script run in the sandbox with the job's own environment's python
import LangSlice: its checkout or installation, the Python it runs on (for a
uv environment, `~/.local/share/uv/python`), and the atlases. Antigravity on
Linux ignores `unsandboxed(...)` rules.
