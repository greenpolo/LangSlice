# Install LangSlice (for agents)

Instructions for an AI agent installing LangSlice on the user's computer and
connecting it to the user's app. Run every step. Stop and tell the user when
a check fails. Ask before replacing anything that already exists.

LangSlice installs from source into a conda environment named `langslice`.
It is not on PyPI or conda-forge.

## 1. Check the prerequisites

```bash
git --version
conda --version
```

- No `conda`: install Miniforge from
  <https://github.com/conda-forge/miniforge>, following its README for the
  user's operating system, then open a new shell. On Windows, use the
  Miniforge Prompt for the commands below.
- No `git`: download <https://github.com/greenpolo/LangSlice/archive/refs/heads/main.zip>
  instead of cloning, and extract it.
- Disk: the environment downloads several gigabytes (PyTorch among them).
  Atlases download on first use into `~/.brainglobe/`: `allen_mouse_25um`
  takes about 600 MB, and the Nissl image LangSlice borrows for it
  (`ccfv3augmented_mouse_25um`) about 1.3 GB.
- A GPU is optional. Deformable fits run on the CPU without one, slower.

## 2. Get the source

```bash
git clone https://github.com/greenpolo/LangSlice.git
cd LangSlice
```

Keep this folder: an update reinstalls LangSlice from it (step 7).

## 3. Create the environment

In the folder that holds `environment.yml`:

```bash
conda env create -f environment.yml
```

If `conda env list` already shows an environment named `langslice`, ask the
user before removing or replacing it. To install beside it, add
`-n langslice-2` and use that name below.

The environment holds LangSlice with its registration engines, the MCP
server and the ABBA launcher.

## 4. Verify

```bash
conda run -n langslice langslice version
conda run -n langslice langslice-job ops
```

Expected: the first prints `langslice 0.1.0` (or a later version). The
second prints one JSON object with `"ok": true` and the list of verbs.

## 5. Put `langslice` on the user's PATH

Every app starts `langslice` by name (Codex's setup skill looks it up on the
`PATH` first), so `langslice` and `langslice-job` must be on the `PATH` the
app starts with. Claude Desktop reads the login shell's `PATH` on macOS and
Linux, and the user and system environment variables on Windows. Find the
executables:

```bash
conda run -n langslice python -c "import shutil; print(shutil.which('langslice')); print(shutil.which('langslice-job'))"
```

On Linux and macOS this is `<env>/bin/langslice`; on Windows,
`<env>\Scripts\langslice.exe`. Then:

- macOS: `echo 'conda activate langslice' >> ~/.zshrc`
- Linux: `echo 'conda activate langslice' >> ~/.bashrc`
- Windows: add `<env>\Scripts` to the user `Path` (in the GUI: **Edit
  environment variables for your account** ‣ **User variables** ‣ **Path**
  ‣ **Edit** ‣ **New**).

Ask the user before editing a shell startup file or `Path` that already
names another LangSlice environment. Check from a new shell, without
activating the environment: `langslice version` prints the version. The user
quits and reopens the app after this change.

## 6. Connect the user's app

Install the plugin for the app the user uses. Each gives the main session
two skills (`register-brain`, `abba`) and two registration agents
(`register`, which uses only LangSlice's MCP tools, and `register-scripting`,
which uses a shell and `langslice-job`). The user-facing steps for each app's
GUI, with screenshots, are in [Connect your AI app](../getting-started/connect.md).

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
skill (Codex only) puts them in Codex's own folder. In a Codex session, ask
"Set up LangSlice." The skill finds `langslice` and copies
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

Add these rules to `permissions.allow` in
`~/.gemini/antigravity-cli/settings.json`, merging with what is there:

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
in the sandbox. Restart Antigravity.

### Claude Code (Claude)

```bash
claude plugin marketplace add greenpolo/LangSlice
claude plugin install langslice@langslice
```

In Claude Desktop's **Code** tab the same is **Customize** ‣ **Plugins** ‣
**Add** ‣ **Add marketplace** ‣ **Add from a repository**, URL
`greenpolo/LangSlice`, **Sync**, then install LangSlice from the list. On
Windows, Claude Desktop needs Git for Windows for marketplace actions.

To let every LangSlice MCP tool run without a prompt, add to the user's
Claude Code settings:

```json
{"permissions": {"allow": ["mcp__plugin_langslice_langslice"]}}
```

Full notes: `connectors/claude-code/README.md`.

## 7. Check the connection

1. Start a new session in the app.
2. List the MCP tools available to the main session. `start_job` must be
   among them.
3. Run `langslice-job ops` from the app's shell. It must print
   `"ok": true`.

If either check fails, the app cannot find the executables: redo step 5,
restart the app and check again. For Codex, run the `setup` skill again.

## 8. After an update

In the LangSlice folder:

```bash
git pull
conda run -n langslice python -m pip install ".[registration,mcp,abba]"
```

pip rebuilds and reinstalls LangSlice from the folder. The plugins are
installed from GitHub, so update them too:

- Claude Code: `claude plugin marketplace update langslice`, then
  `claude plugin update langslice@langslice`.
- Codex: `codex plugin marketplace upgrade`, then the `setup` skill again,
  which copies the agents and rules.
- Antigravity: the `agy plugin install` command of step 6 again.

Restart the app, and run step 4 to verify.
