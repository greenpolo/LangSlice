# LangSlice for Codex

The closest Codex equivalent of the Claude Code plugin
(`connectors/claude-code/`): skills for the main session and one custom agent
per access level that registers one brain. It uses the `langslice job` command
line ([`docs/agent_cli.md`](../../docs/agent_cli.md)) and bundles no MCP
server (MCP is for desktop apps without a shell). Codex documents no way to
package custom agents in a plugin, so these are files you copy into place;
nothing here is a Codex plugin.

LangSlice must be installed (conda env of `environment.yml`) with `langslice`
on PATH.

## Install

```bash
# skills (Codex scans .agents/skills in the repo, then ~/.agents/skills)
cp -r skills/* ~/.agents/skills/
# custom agents (personal: ~/.codex/agents/, project: .codex/agents/)
mkdir -p ~/.codex/agents && cp agents/*.toml ~/.codex/agents/
# optional command rules
mkdir -p ~/.codex/rules && cp rules/langslice.rules ~/.codex/rules/
```

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| `skills/register-brain` (`$register-brain`) | main session | prepares the job from the user's choices, launches one registration agent, checks results and exports; never registers itself |
| `skills/abba` | main session | ABBA's own 0.24.x commands, how to reach a running Fiji, moving files between ABBA and a job |
| `agents/register_cli.toml` | registers one brain | the `langslice job` CLI, workspace-write sandbox |
| `agents/register_scripting.toml` | registers one brain | CLI plus Python-library scripts in the job folder, workspace-write sandbox |
| `rules/langslice.rules` | sandboxing | allows `langslice` commands outside the sandbox, forbids `curl` |

The role split is the same as in Claude Code: the main session owns setup,
checking and import/export; each agent file has a minimal
`developer_instructions` that defers to LangSlice's job statement and the job
card (`AGENTS.md` in the job folder). Ask Codex to spawn the agent by name,
for example "have register_cli register the job in /data/M04".

The agents start with `langslice job <folder> brief`: LangSlice's job
statement for the job (the one its own agent gets), the user's notes, the
status table and the opening pictures saved as files, also written to
`BRIEF.md` in the job folder ([`docs/agent_cli.md`](../../docs/agent_cli.md)).
Create the job with `init --viewer codex`, so the pictures are sized for
Codex's `view_image` (2048 px).

## Sandboxing, and the gaps

- Codex has no per-agent tool allowlist like Claude Code's `tools`. It has
  `sandbox_mode` per agent file (read-only, workspace-write,
  danger-full-access) and the `features.shell_tool` switch. Both agents need
  the shell, so the sandbox is the boundary.
- Rules (`prefix_rule`) apply only to commands that would run outside the
  sandbox; they are not an allowlist of everything the shell may run. A
  "`langslice` only" shell is not possible.
- Custom agents inherit the parent's sandbox and approval mode unless the file
  sets `sandbox_mode`; unattended, an action needing approval fails and the
  error goes back to the parent.
- No equivalent of `omitClaudeMd` was found, so a project `AGENTS.md` still
  reaches the agents. `[[skills.config]]` with `enabled = false` can disable
  a skill per agent file.

## Status

The TOML files parse and `codex execpolicy check` matches the rules; the
agents have not yet been used for a full registration. No Codex plugin
package is shipped.
