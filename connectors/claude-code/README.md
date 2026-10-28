# LangSlice plugin for Claude Code

Registers histology sections to a BrainGlobe atlas from Claude Code, on the
user's own Claude subscription (LangSlice never handles Claude credentials).
The plugin is skills for the normal Claude Code session plus registration
subagents that each register exactly one brain. It bundles LangSlice's MCP
server (`.mcp.json`: the `langslice` executable with `mcp`, the server
Claude Desktop uses, [`docs/agents/mcp.md`](../../docs/agents/mcp.md)), and
the scripting subagent and the skills use the `langslice-job` command line
([`docs/agents/cli.md`](../../docs/agents/cli.md)).

## Install

LangSlice itself must be installed first (the conda environment of
`environment.yml`) with `langslice` on the PATH of the shell that starts
Claude Code (`langslice version` must work).

```text
/plugin marketplace add greenpolo/LangSlice
/plugin install langslice@langslice
```

The marketplace is `.claude-plugin/marketplace.json` at the repository root;
it points at this folder. To try it from a checkout:
`claude --plugin-dir connectors/claude-code`.

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| skill `langslice:register-brain` | main session | prepares the job from the user's choices (tasks, atlas, image model or none, `--registration FILE`, pixel size), launches one subagent, then checks `status`, `registration.json`, the exports and the pictures, and handles export. It never registers itself. |
| skill `langslice:abba` | main session | ABBA's own 0.24.x commands (import, state, export), how to reach a running Fiji, moving files between ABBA and a job |
| agent `langslice:register` | registers one brain | only LangSlice's MCP tools (`mcp__plugin_langslice_langslice__*`), whose replies carry the pictures; no shell, no file writes, no web |
| agent `langslice:register-scripting` | registers one brain | Bash, Read, Write, Edit and the web: the `langslice-job` command line plus scripts with the LangSlice Python library, and packages installed in the job's own environment (`langslice-job FOLDER env`) |
| MCP server `plugin:langslice:langslice` | the `register` agent | `langslice mcp`, started with the session; its tools are also listed in the main session, which leaves them to the subagent |
| hook `SubagentStop` | every registration subagent | `langslice session-end --harness claude-code` copies the subagent's transcript into the job's `trace/` folder |

The role split is deliberate. The main session owns everything around the
registration (choices, job, checking, import and export with ABBA and other
software); registration is always a subagent's. A subagent has one job and a
one fixed system prompt, the same for every job; the job itself (LangSlice's
job statement, the user's notes) is the message that launched them, made by
`langslice-job <folder> launch --role register` (or `register-scripting`) and
passed on unchanged. The job card (`AGENTS.md` in the job folder) is the
scripting subagent's reference. The skills are not preloaded into the
subagents (`skills` is empty and `Skill` is not in their `tools`).

`register` starts with `start_job(image_folder=<folder>)`, named in its launch
message, which also holds LangSlice's job statement for the job (the one its
own agent gets) and the user's notes; it reads the opening pictures with
`show_stack`. `register-scripting`'s launch message holds the same statement,
the status table and the opening pictures listed as saved files. Each launch
message is also saved in the job's `logs/launches/`, and the hook checks the
subagent's first message against those files
([`docs/agents/cli.md`](../../docs/agents/cli.md)).

The MCP server holds one job at a time for the session, so one `register`
subagent runs at a time per Claude Code session. It sends no progress during
a long verb; `.mcp.json` sets the server's `timeout` to 24 hours, which also
keeps Claude Code from aborting a call that is silent for 30 minutes (its
idle limit for stdio servers), so a call longer than 24 hours is aborted.

## Sandboxing

Each agent has a strict `tools` allowlist, `model: inherit` and
`omitClaudeMd: true` (the subagent starts without your user, project and local
CLAUDE.md files; managed policy files still load).

- `register`: `mcp__plugin_langslice_langslice__*`.
- `register-scripting`: `Bash, Read, Write, Edit, WebFetch, WebSearch`.

Claude Code asks before each MCP tool call unless a permission rule allows
it; `"allow": ["mcp__plugin_langslice_langslice"]` in a settings file allows
every LangSlice MCP tool.

Claude Code can also attach its advisor tool (a second model the agent may
consult) to every agent, these included; neither `tools` nor
`disallowedTools` in an agent file removes it. Turn it off with
`CLAUDE_CODE_DISABLE_ADVISOR_TOOL=1`, in the shell or in a settings file's
`env` (`"env": {"CLAUDE_CODE_DISABLE_ADVISOR_TOOL": "1"}`, honored from
project settings too). It applies to the whole session, main agent included.

The main session may pass a model and an effort level when it starts a
subagent (the Agent tool's `model` and `effort`); otherwise the subagent
inherits the session's.

A plugin agent cannot restrict Bash to one command (a `Bash(langslice *)`
entry in `tools` is not honored per command; `permissionMode` and `hooks` are
ignored in plugin agents). To make `register-scripting` run only `langslice-job`, add
permission rules to a settings file, for example `.claude/settings.local.json`
of a dedicated working directory. They apply to the whole session, main agent
included:

```json
{
  "permissions": {
    "defaultMode": "dontAsk",
    "allow": [
      "Bash(langslice-job *)",
      "Read(//data/brains/M04/**)"
    ]
  }
}
```

`dontAsk` denies every call that no rule allows, so only `langslice-job` commands
(each part of a compound command must match) and reading under the named path
run. Replace the path with your job folder (`//` starts an absolute path).
It also needs `Edit(//data/brains/M04/**)`, and the main
session needs whatever else it should be allowed to do. The scripting
agent runs its scripts with `langslice-job FOLDER python SCRIPT`, so allowing
`langslice-job` allows any script it writes.
