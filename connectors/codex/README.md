# LangSlice for Codex

The closest Codex equivalent of the Claude Code plugin
(`connectors/claude-code/`): skills for the main session, one custom agent per
access level that registers one brain, and the LangSlice MCP server. Codex
documents no way to package custom agents in a plugin, so these are files you
copy into place; nothing here is a Codex plugin.

LangSlice must be installed (conda env of `environment.yml`, plus
`pip install "langslice[mcp]"`) with `langslice` on PATH, or give the absolute
path of the environment's `langslice` as `command` in the files below.

## Install

```bash
# skills (Codex scans .agents/skills in the repo, then ~/.agents/skills)
cp -r skills/* ~/.agents/skills/
# custom agents (personal: ~/.codex/agents/, project: .codex/agents/)
mkdir -p ~/.codex/agents && cp agents/*.toml ~/.codex/agents/
# optional command rules
mkdir -p ~/.codex/rules && cp rules/langslice.rules ~/.codex/rules/
```

For the main session's own use of the LangSlice MCP tools (not needed when
the `register_mcp` agent carries the server itself):
`codex mcp add langslice -- langslice mcp`, or the equivalent
`[mcp_servers.langslice]` table in `~/.codex/config.toml`.

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| `skills/register-brain` (`$register-brain`) | main session | creates the job from the user's choices, launches one registration agent, checks results; never registers itself |
| `skills/abba` | main session | ABBA 0.24.x commands and how to reach a running Fiji |
| `agents/register_mcp.toml` | registers one brain | LangSlice MCP tools, read-only sandbox |
| `agents/register_cli.toml` | registers one brain | `langslice job` CLI, workspace-write sandbox |
| `agents/register_scripting.toml` | registers one brain | CLI plus Python-library scripts, workspace-write sandbox |
| `rules/langslice.rules` | sandboxing | allows `langslice` commands outside the sandbox, forbids `curl` |
| `langslice-mcp.config.toml` | `codex exec` | profile for a headless MCP-only one-brain run |

The role split is the same as in Claude Code: the main session owns the setup
and the checking; each agent file has a minimal `developer_instructions` that
only defers to LangSlice's own job statement (`start_job`) or job card
(`AGENTS.md` in the job folder). Ask Codex to spawn the agent by name, for
example "have register_cli register the job in /data/M04".

## Headless one-brain run

```bash
langslice claude prepare /data/M04 --tasks position,transform   # prints the job id
codex exec -p langslice-mcp "Call start_job(job_id=\"<id>\") and register that brain. Use only the LangSlice tools and finish with submit."
```

## Sandboxing, and the gaps

- Codex has no per-agent tool allowlist like Claude Code's `tools`. What it
  has: `sandbox_mode` per agent file (read-only, workspace-write,
  danger-full-access), the `features.shell_tool = false` switch, and per-server
  `enabled_tools` / `disabled_tools` for MCP tools.
- `register_mcp` is therefore only "MCP-only" by convention and sandbox: the
  agent file sets a read-only sandbox, but does not remove the shell tool. The
  `langslice-mcp.config.toml` profile does (`features.shell_tool = false`) for
  `codex exec`. Whether `features.shell_tool` is honored inside a custom agent
  file is not documented; it is left out of the agent file.
- Rules (`prefix_rule`) apply only to commands that would run outside the
  sandbox; they are not an allowlist of everything the shell may run. A
  "`langslice` only" shell is not possible; the sandbox is the boundary.
- Custom agents inherit the parent's sandbox and approval mode unless the file
  sets `sandbox_mode`; unattended, an action needing approval fails and the
  error goes back to the parent.
- No equivalent of `omitClaudeMd` or of keeping skills out of an agent was
  found, though `[[skills.config]]` with `enabled = false` can disable a
  skill per agent file. A project `AGENTS.md` therefore still reaches the
  agents.

## Uncertain

- Not run against a live model (that costs money). The TOML files parse and
  `codex execpolicy check` matches the rules; agent behavior, and whether the
  `mcp_servers` table in an agent file is honored, are untested.
- Codex's two plugin documents disagreed about the manifest (`.codex-plugin/`
  or a root `plugin.json`), so no plugin package is shipped.
- Documentation read from developers.openai.com/codex (redirects to
  learn.chatgpt.com/docs): skills, subagents, MCP, rules, config reference,
  plugins/build. The profile `-p NAME` loading `$CODEX_HOME/NAME.config.toml`
  is from `codex exec --help` of codex-cli 0.160.0.
