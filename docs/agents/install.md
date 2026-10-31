# Install LangSlice (for agents)

Instructions for an AI agent installing LangSlice on the user's computer and
connecting it to the user's app. Run every step. Stop and tell the user when
a check fails. Ask before replacing anything that already exists.

LangSlice installs from PyPI into a conda environment named `langslice`
(Python 3.11).

## 1. Check the prerequisites

```bash
conda --version
```

- No `conda`: install Miniforge from
  <https://github.com/conda-forge/miniforge>, following its README for the
  user's operating system, then open a new shell. On Windows, use the
  Miniforge Prompt for the commands below.
- Disk: the environment downloads several gigabytes (PyTorch among them).
  Atlases download on first use into `~/.brainglobe/`: `allen_mouse_25um`
  takes about 600 MB, and the Nissl image LangSlice borrows for it
  (`ccfv3augmented_mouse_25um`) about 1.3 GB.
- A GPU is optional. Deformable fits run on the CPU without one, slower.

## 2. Create the environment

If `conda env list` already shows an environment named `langslice`, ask the
user before removing or replacing it. To install beside it, use another name,
such as `langslice-2`, below.

```bash
conda create -n langslice python=3.11 -y
conda run -n langslice python -m pip install "langslice[registration,mcp]"
```

The environment holds LangSlice with its registration engines, the MCP
server and the worker the ABBA connector starts.

## 3. Or install from the source

For the version on GitHub, instead of step 2:

```bash
git clone https://github.com/greenpolo/LangSlice.git
cd LangSlice
conda env create -f environment.yml
```

Keep the folder: an update reinstalls LangSlice from it (step 7).

## 4. Verify

```bash
conda run -n langslice langslice version
conda run -n langslice langslice-job ops
```

Expected: the first prints `langslice 0.1.0` (or a later version). The
second prints one JSON object with `"ok": true` and the list of verbs.

## 5. Connect the user's app

Install the plugin for the app the user uses. Each gives the main session
three skills (`setup`, `register-brain`, `abba`) and two registration agents
(`register`, which uses only LangSlice's MCP tools, and `register-scripting`,
which uses a shell and `langslice-job`). The user-facing steps for each app's
GUI, with screenshots, are in [Connect your AI app](../getting-started/connect.md).

Then run the plugin's `setup` skill (`skills/setup/SKILL.md` in the plugin,
or ask the app "Set up LangSlice."). In every app it first finds `langslice`
(on `PATH`, else in a conda environment named `langslice`) and puts its
folder on the user's `PATH`, because the apps and LangSlice's agents run
`langslice` and `langslice-job` by name. Claude Desktop and Antigravity read
`PATH` from `$SHELL -ilc` on macOS and Linux and from the user and system
environment variables on Windows. The skill appends the folder (the
environment's `bin`, or `Scripts` on Windows), without activating the
environment:

- zsh: a guarded `export PATH="$PATH:<folder>"` line in `~/.zprofile` and
  `~/.zshrc`.
- bash: the same line in `~/.bashrc` and in the login file bash reads
  (`~/.bash_profile`, else `~/.bash_login`, else `~/.profile`).
- Windows: the folder appended to the user `Path` with
  `[Environment]::SetEnvironmentVariable('Path', …, 'User')`.

A login, interactive shell (the apps), a login shell that is not interactive
(`-lc`, as agents' commands may run) and a new Linux terminal (interactive,
not login) each read a different file, hence two. The skill skips a file that
already holds its line, and the whole step when a new login shell already
finds `langslice`. The rest of each skill is per app, below. The user quits
and reopens the app after setup.

### Codex (ChatGPT)

```bash
codex plugin marketplace add greenpolo/LangSlice
codex plugin add langslice@langslice
```

In the ChatGPT desktop app the same is **Plugins** ‣ **Add** ‣ **Add a
marketplace**, Source `greenpolo/LangSlice`, then install LangSlice from the
plugin list.

A Codex plugin cannot carry custom agents or command rules, and the
`register` agent does not see a plugin's MCP server. The plugin's `setup`
skill puts them in Codex's own folder. In a Codex session, ask
"Set up LangSlice." After the `PATH` step, the skill copies
`agents/register.toml`, `agents/register_scripting.toml` and
`rules/langslice.rules` into `~/.codex/agents/` and `~/.codex/rules/`. On
Windows the folder is `%USERPROFILE%\.codex`, and `$CODEX_HOME` replaces it
when set. It then adds this to `~/.codex/config.toml`:

```toml
[mcp_servers.langslice]
command = "/absolute/path/to/env/bin/langslice"
args = ["mcp"]
tool_timeout_sec = 3600
default_tools_approval_mode = "approve"

[features]
mcp_tool_list_changed = true
```

`mcp_tool_list_changed` is required: the job's tools appear only after
`start_job`, and without it the `register` agent sees `start_job` alone.
`tool_timeout_sec` replaces Codex's 60 s default, which is shorter than a
deformable fit. The writes are outside the workspace, so Codex asks for
approval. Restart the app afterwards. Full notes:
`connectors/codex/README.md`.

### Antigravity (Gemini)

```bash
agy plugin install https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
```

In the app, the same is the `plugin` skill: type `/plugin`, pick **plugin**,
and send "Please install this plugin:" followed by the address above.

Restart Antigravity, then follow the plugin's `setup` skill
(`skills/setup/SKILL.md`, or ask the app "Set up LangSlice."). After the
`PATH` step, it adds these rules to `permissions.allow` in `~/.gemini/antigravity-cli/settings.json`,
merging with what is there:

```json
{
  "permissions": {
    "allow": [
      "mcp(langslice_mcp/*)",
      "command(langslice-job)"
    ]
  }
}
```

`mcp(langslice_mcp/*)` is all the `register` agent needs.
`register-scripting` needs `command(langslice-job)`, plus the `read_file`
and `read_url` rules in `connectors/antigravity/README.md` for scripts run
in the sandbox; setup adds those too. Restart Antigravity.

### Claude Code (Claude)

```bash
claude plugin marketplace add greenpolo/LangSlice
claude plugin install langslice@langslice
```

In Claude Desktop's **Code** tab the same is **Customize** ‣ **Plugins** ‣
**Add** ‣ **Add marketplace** ‣ **Add from a repository**, URL
`greenpolo/LangSlice`, **Sync**, then install LangSlice from the list. On
Windows, Claude Desktop needs Git for Windows for marketplace actions.

The plugin starts its MCP server as `langslice mcp`, so the server fails to
start until `langslice` is on `PATH`. Then follow the plugin's `setup` skill
(`skills/setup/SKILL.md`, or ask the app "Set up LangSlice."; it shows as
`langslice:setup`). After the `PATH` step, it adds this rule to
`permissions.allow` in `~/.claude/settings.json` (`$CLAUDE_CONFIG_DIR`
replaces `~/.claude` when set), merging with what is there, so every
LangSlice MCP tool runs without a prompt:

```json
{"permissions": {"allow": ["mcp__plugin_langslice_langslice"]}}
```

Restart the app afterwards.

Full notes: `connectors/claude-code/README.md`.

## 6. Check the connection

1. Start a new session in the app.
2. List the MCP tools available to the main session. `start_job` must be
   among them.
3. Run `langslice-job ops` from the app's shell. It must print
   `"ok": true`.

If either check fails, the app cannot find the executables: run the
plugin's `setup` skill again, restart the app and check again.

## 7. After an update

```bash
conda run -n langslice python -m pip install -U "langslice[registration,mcp]"
```

From the source instead, in the LangSlice folder: `git pull`, then
`conda run -n langslice python -m pip install ".[registration,mcp]"`.

The plugins are installed from GitHub, so update them too:

- Claude Code: `claude plugin marketplace update langslice`, then
  `claude plugin update langslice@langslice`.
- Codex: `codex plugin marketplace upgrade`, then the `setup` skill again,
  which copies the agents and rules.
- Antigravity: the `agy plugin install` command of step 5 again.

Restart the app, and run step 4 to verify.
