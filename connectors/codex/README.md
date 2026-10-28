# LangSlice for Codex

Skills for the main session and two custom agents that register one brain:
`register` works only through LangSlice's MCP tools (`langslice mcp`), and
`register_scripting` works through the shell, the `langslice-job` command line
([`docs/agents.md`](../../docs/agents.md)) and the Python library. Install the
supplied skill, agent and rule files and the MCP server entry directly; this
folder contains no plugin package.

LangSlice must be installed (conda env of `environment.yml`) with `langslice`
and `langslice-job` on PATH.

## Install

From the repository root:

```bash
cd connectors/codex
# skills (Codex scans .agents/skills in the repo, then ~/.agents/skills)
mkdir -p ~/.agents/skills
cp -r skills/* ~/.agents/skills/
# custom agents (personal: ~/.codex/agents/, project: .codex/agents/)
mkdir -p ~/.codex/agents && cp agents/*.toml ~/.codex/agents/
# optional command rules
mkdir -p ~/.codex/rules && cp rules/langslice.rules ~/.codex/rules/
```

Then add the LangSlice MCP server to `~/.codex/config.toml` (or a trusted
project's `.codex/config.toml`), with the full path of `langslice` (`which
langslice`):

```toml
[mcp_servers.langslice]
command = "/path/to/env/bin/langslice"
args = ["mcp"]
tool_timeout_sec = 3600
default_tools_approval_mode = "approve"

[features]
mcp_tool_list_changed = true
```

- `args = ["mcp"]` names no folder: the agent opens its job with
  `start_job(image_folder=...)`, which opens the job `langslice-job FOLDER
  init` created as saved. The job's tools appear only then, so Codex must
  follow MCP tool-list changes: `mcp_tool_list_changed` is a Codex feature
  marked "under development" (codex-cli 0.164.0), and without it the agent
  sees `start_job` alone.
- `tool_timeout_sec` replaces Codex's 60 s default, which is shorter than a
  deformable fit.
- `default_tools_approval_mode = "approve"` lets LangSlice's tools run
  without a prompt for each call.
- No picture-size flag is needed: `langslice mcp` has none. Its pictures are
  drawn at Claude's sizes (opening strips within 1568 px; later pictures at
  the job's `--image-resolution`, at most 2000 px), within the 2048 px Codex
  accepts. `init --viewer codex` sizes the pictures of the `langslice-job`
  command line only.

## What is inside

| Part | For | What it is |
| --- | --- | --- |
| `skills/register-brain` (`$register-brain`) | main session | prepares the job from the user's choices, launches one registration agent, checks results and exports; never registers itself |
| `skills/abba` | main session | ABBA's own 0.24.x commands, how to reach a running Fiji, moving files between ABBA and a job |
| `agents/register.toml` | registers one brain | LangSlice's MCP tools only; no shell |
| `agents/register_scripting.toml` | registers one brain | the `langslice-job` CLI plus Python-library scripts in the job folder, workspace-write sandbox |
| `rules/langslice.rules` | sandboxing | allows `langslice-job` commands outside the sandbox, forbids `curl` |

The role split is the same as in Claude Code: the main session owns setup,
checking and import/export; each agent file has the same fixed
`developer_instructions`, and the job statement comes in the launch message
(`langslice-job <folder> launch --role register`, or `register-scripting`),
which the main session passes on unchanged. Ask Codex
to spawn the agent by name, for example "have register register the job in
/data/M04". `register` is the default; `register_scripting` is for a user who
wants the agent to write scripts.

- `register` starts with `start_job(image_folder=<folder>)`, named in its
  launch message (which holds LangSlice's job statement), reads every
  `show_stack` page (the opening pictures, carried in the tool replies) and
  finishes with `submit`.
- `register_scripting`'s launch message holds the same job statement, the
  user's notes, the status table and the opening pictures listed as saved
  files ([`docs/agents.md`](../../docs/agents.md)). `AGENTS.md` in the job folder
  is its reference card. Create the job with `init --viewer codex`, so these
  pictures are sized for Codex's `view_image` (2048 px).

## Sandboxing, and the gaps

- Codex has no per-agent tool allowlist like Claude Code's `tools`. An agent
  file can switch off a few features; `register` sets `shell_tool = false`
  (no `exec_command`, and an agent it starts has no shell either) and
  `apps = false`, `plugins = false`.
- What `register` can still do, all without a shell: write files with
  `apply_patch` where the session's sandbox allows writes, search and read the
  web (`web__run`), generate images (`image_gen`), open local images
  (`view_image`), read MCP resources, and start other agents. Codex ignores
  the agent-file switches for these. Running the main session read-only
  (`codex --sandbox read-only`) removes the file writes; the `langslice-job`
  rule still lets the main session's setup commands run.
- `sandbox_mode`, `approval_policy` and `mcp_servers` in an agent file do not
  override the main session: every agent sees the session's MCP servers and
  runs in its sandbox and approval mode. Unattended, an action needing
  approval fails and the error goes back to the parent.
- `register_scripting` needs the shell, so the sandbox is its boundary. Rules
  (`prefix_rule`) apply only to commands that would run outside the sandbox;
  they are not an allowlist of everything the shell may run, so a
  "`langslice-job` only" shell is not possible. `rules/langslice.rules` lets
  every `langslice-job` command run outside the sandbox, and the scripting
  agent runs its scripts with `langslice-job FOLDER python SCRIPT`: those
  scripts run outside the sandbox too.
- The main session sees LangSlice's MCP tools as well; the `register-brain`
  skill keeps it from registering.
- The MCP server holds one open job, and `start_job` switches it: run one
  `register` agent at a time in a session.
- No equivalent of `omitClaudeMd` was found, so a project `AGENTS.md` still
  reaches the agents. `[[skills.config]]` with `enabled = false` can disable
  a skill per agent file.
- This connector has no session-end hook: an agent's transcript stays in
  Codex's own session files (`$CODEX_HOME/sessions`) and is not copied into
  the job's `trace/`.
