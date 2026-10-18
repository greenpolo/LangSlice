# LangSlice plugin for Antigravity

Registers histology sections to a BrainGlobe atlas from Google's Antigravity
(the `agy` CLI, the Antigravity app and IDE), on the user's own Antigravity
account. Like the Claude Code plugin ([`../claude-code/`](../claude-code/README.md)),
it is skills for the main session plus registration agents that each register
exactly one brain, over the `langslice-job` command line
([`docs/agents.md`](../../docs/agents.md)).

## Install

LangSlice must be installed first, with `langslice` and `langslice-job` on
the PATH of the shell that starts Antigravity (`langslice version` must work):
the agents run `langslice-job`, and the plugin's hook runs `langslice`.

```bash
agy plugin install connectors/antigravity      # from a checkout
agy plugin install https://github.com/greenpolo/LangSlice   # or from GitHub
```

`agy plugin install` copies the folder; install again after changing it.

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| skill `register-brain` | main session | prepares the job from the user's choices, starts one registration agent with `invoke_subagent`, then checks the job and handles export; it never registers itself |
| skill `abba` | main session | ABBA's own 0.24.x commands, how to reach a running Fiji, moving files between ABBA and a job |
| agent `register-cli` | registers one brain | `run_command`, `view_file` and the web: the `langslice-job` command line and the pictures it saves |
| agent `register-scripting` | registers one brain | also `write_to_file` and `replace_file_content`: scripts with the LangSlice Python library, packages installed in the job's own environment (`langslice-job FOLDER env`) |
| hook `langslice-transcript` | every session | at the end of a session, `langslice session-end --harness antigravity` copies the transcript into the job's `trace/` folder |

Both agents are `mainAgent` and `subagent`: the main session starts one with
`invoke_subagent`, and a run without a chat starts one directly:

```bash
cd /data/brains/M04/langslice     # the job folder (made by `langslice-job ... init`)
agy -p "Register the brain in job folder /data/brains/M04/langslice" \
  --agent register-cli --model gemini-4-argon-high --output-format stream-json \
  > run.jsonl
```

`agy models` lists the models. An agent's `model: inherit` takes the
session's model (`--model`, or the chat's).

## Sandbox and permissions

Antigravity runs an agent's commands in an OS sandbox (Linux namespaces):
the working folder and temporary folders are writable, the paths that
`read_file` rules allow are visible read-only, and network access is limited
to the domains `read_url` rules allow. `langslice-job` reads the atlas and the
images and calls the image model, so the agents run it outside the sandbox
(`BypassSandbox`), which a `command(langslice-job)` rule allows without asking.
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
      "command(langslice-job)",
      "read_url(*)",
      "read_file(/path/to/LangSlice)",
      "read_file(/path/to/the/python/LangSlice/runs/on)",
      "read_file(~/.brainglobe)"
    ]
  }
}
```

`read_url(*)` gives sandboxed commands the web (package downloads, papers,
documentation). The scripting agent runs its scripts with `langslice-job
FOLDER python SCRIPT`, a `langslice-job` command, so they run outside the
sandbox too. The `read_file` rules let a script run in the sandbox with the
job's own environment's python import LangSlice: its checkout or installation, the
Python it runs on (for a uv environment, `~/.local/share/uv/python`), and the
atlases. Antigravity on Linux ignores `unsandboxed(...)` rules.

An agent's `tools` list does not narrow the tools of an agent started with
`--agent` (the session's tools stay available); the agents are given the
same freedom as the session.
