# LangSlice plugin for Claude Code

Registers histology sections to a BrainGlobe atlas from Claude Code, on the
user's own Claude subscription (LangSlice never handles Claude credentials).
The plugin is skills for the normal Claude Code session plus registration
subagents that each register exactly one brain.

## Install

LangSlice itself must be installed first (the conda environment of
`environment.yml`, with the MCP extra: `pip install "langslice[mcp]"`), and
`langslice` must be on the PATH of the shell that starts Claude Code
(`langslice version` must work). If you would rather not change the PATH,
edit `.mcp.json` in the installed plugin so `command` is the absolute path of
`langslice` in that environment (as in `connectors/claude-desktop/`).

```text
/plugin marketplace add greenpolo/LangSlice
/plugin install langslice@langslice
```

The marketplace is `.claude-plugin/marketplace.json` at the repository root;
it points at this folder. To try it from a checkout:
`claude --plugin-dir connectors/claude-code`, or
`claude plugin marketplace add ./` from the repository root.

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| skill `langslice:register-brain` | main session | creates the job from the user's choices (tasks, atlas, image model or none, `--registration FILE`, pixel size), launches one subagent, then checks `status`, `registration.json`, the exports and the pictures. It never registers itself. Export is folded in here. |
| skill `langslice:abba` | main session | ABBA 0.24.x imports, state, exports, and how to reach a running Fiji |
| agent `langslice:register-mcp` | registers one brain | only the LangSlice MCP tools |
| agent `langslice:register-cli` | registers one brain | Bash and Read: the `langslice job` command line |
| agent `langslice:register-scripting` | registers one brain | Bash, Read, Write, Edit: the CLI plus scripts with the Python library |
| `.mcp.json` | | the LangSlice MCP server (`langslice mcp`), shown as `plugin:langslice:langslice` |

The role split is deliberate. The main session owns everything around the
registration (choices, job, checking, exports, ABBA). A registration subagent
has one job and a minimal system prompt that only defers to LangSlice's own
job statement (MCP: returned by `start_job`) or job card (CLI: `AGENTS.md` in
the job folder), so it works like the registration agent LangSlice runs in
ABBA and Claude Desktop: same tools, same statement. The skills are not
preloaded into the subagents (their `skills` field is empty and `Skill` is
not in their `tools`).

## Variants and sandboxing

Each agent has a strict `tools` allowlist, so what a subagent can do is what
you allow:

- `register-mcp`: `mcp__plugin_langslice_langslice` (every tool of the
  bundled server, nothing else). No shell, no file edits, no web.
- `register-cli`: `Bash, Read`.
- `register-scripting`: `Bash, Read, Write, Edit`.

All three use `model: inherit` and `omitClaudeMd: true` (the subagent starts
without your user, project and local CLAUDE.md files, so your project rules do
not reach it; managed policy files still load).

A plugin agent cannot restrict Bash to one command: a `Bash(langslice *)`
entry in `tools` or `disallowedTools` is not honored per command, and a
plugin agent's `permissionMode` and `hooks` are ignored. To make a CLI
subagent run only `langslice`, add permission rules to a settings file
(for example `.claude/settings.local.json` of a dedicated working directory).
They apply to the whole session, main agent included:

```json
{
  "permissions": {
    "defaultMode": "dontAsk",
    "allow": [
      "Bash(langslice *)",
      "Read(//data/brains/M04/**)"
    ]
  }
}
```

`dontAsk` denies every call that is not allowed by a rule, so only
`langslice` commands (each part of a compound command must match) and reading
under the named path run. Replace the path with your job folder (`//` starts
an absolute path). The `Write`/`Edit` of the scripting variant needs its own
`Edit(//data/brains/M04/**)` rule.

## Uncertain

- Not run against a live model here (that costs money): `claude plugin
  validate` and JSON checks passed, but whether a subagent sees the MCP
  tools that `start_job` adds after it starts, and whether the server-level
  entry `mcp__plugin_langslice_langslice` resolves in a plugin agent's
  `tools`, is not yet confirmed. If the MCP subagent lists no tools, start
  the server with the job: change `args` in `.mcp.json` to
  `["mcp", "--job", "<job id>"]`.
- The CLI has no job statement: the CLI variants work from the job card and
  `status`, which is less than the MCP variant gets.
