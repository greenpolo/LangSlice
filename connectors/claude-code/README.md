# LangSlice plugin for Claude Code

Registers histology sections to a BrainGlobe atlas from Claude Code, on the
user's own Claude subscription (LangSlice never handles Claude credentials).
The plugin is skills for the normal Claude Code session plus registration
subagents that each register exactly one brain. It uses the `langslice-job`
command line ([`docs/agents.md`](../../docs/agents.md)); it bundles no
MCP server. The MCP configuration for Claude Desktop is in
[`connectors/claude-desktop/`](../claude-desktop/README.md).

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
| agent `langslice:register-cli` | registers one brain | Bash and Read: the `langslice-job` command line and the pictures it saves |
| agent `langslice:register-scripting` | registers one brain | Bash, Read, Write, Edit: the CLI plus scripts with the LangSlice Python library in the job folder |

The role split is deliberate. The main session owns everything around the
registration (choices, job, checking, import and export with ABBA and other
software); registration is always a subagent's. A subagent has one job and a
minimal system prompt that defers to LangSlice's own job statement and the
job card (`AGENTS.md` in the job folder). The skills are not preloaded into the
subagents (`skills` is empty and `Skill` is not in their `tools`).

The subagents start with `langslice-job <folder> brief`: LangSlice's job
statement for the job (the one its own agent gets), the user's notes, the
status table and the opening pictures saved as files, also written to
`BRIEF.md` in the job folder ([`docs/agents.md`](../../docs/agents.md)).

## Sandboxing

Each agent has a strict `tools` allowlist, `model: inherit` and
`omitClaudeMd: true` (the subagent starts without your user, project and local
CLAUDE.md files; managed policy files still load).

- `register-cli`: `Bash, Read`.
- `register-scripting`: `Bash, Read, Write, Edit`.

A plugin agent cannot restrict Bash to one command (a `Bash(langslice *)`
entry in `tools` is not honored per command; `permissionMode` and `hooks` are
ignored in plugin agents). To make a subagent run only `langslice-job`, add
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
The scripting variant also needs `Edit(//data/brains/M04/**)`, and the main
session needs whatever else it should be allowed to do.
