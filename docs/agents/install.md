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

Keep this folder. The Antigravity and Codex connectors are installed from it.

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

Find the absolute path of the executables, needed below:

```bash
conda run -n langslice python -c "import shutil; print(shutil.which('langslice')); print(shutil.which('langslice-job'))"
```

On Linux and macOS this is `<env>/bin/langslice`; on Windows,
`<env>\Scripts\langslice.exe`.

## 5. Connect the user's app

Install the connector for the app the user uses. Each gives the main session
two skills (`register-brain`, `abba`) and two registration agents
(`register`, which uses only LangSlice's MCP tools, and `register-scripting`,
which uses a shell and `langslice-job`).

### Codex (ChatGPT)

From the LangSlice folder:

```bash
cd connectors/codex
mkdir -p ~/.agents/skills && cp -r skills/* ~/.agents/skills/
mkdir -p ~/.codex/agents && cp agents/*.toml ~/.codex/agents/
mkdir -p ~/.codex/rules && cp rules/langslice.rules ~/.codex/rules/
```

Add to `~/.codex/config.toml`, with the absolute path from step 4. Merge
into an existing `[features]` table rather than adding a second one:

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
deformable fit. Full notes: `connectors/codex/README.md`.

### Antigravity (Gemini)

From the LangSlice folder:

```bash
agy plugin install connectors/antigravity
```

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

Ask the user to type these two commands into a Claude Code session:

```text
/plugin marketplace add greenpolo/LangSlice
/plugin install langslice@langslice
```

To let every LangSlice MCP tool run without a prompt, add to the user's
Claude Code settings:

```json
{"permissions": {"allow": ["mcp__plugin_langslice_langslice"]}}
```

Full notes: `connectors/claude-code/README.md`.

## 6. Check the connection

The Antigravity and Claude Code connectors start `langslice` by name, so
`langslice` and `langslice-job` must be on the `PATH` the app starts with.
Codex uses the absolute path from its `config.toml`.

1. Start a new session in the app.
2. List the MCP tools available to the main session. `start_job` must be
   among them.
3. Run `langslice-job ops` from the app's shell. It must print
   `"ok": true`.

If either check fails, the app cannot find the executables. Put them on the
app's `PATH`: link both into a folder already on it, or add the environment's
`bin` folder (`Scripts` on Windows) to the user's `PATH`. Restart the app and
check again.

## 7. After an update

In the LangSlice folder:

```bash
git pull
conda run -n langslice python -m pip install ".[registration,mcp,abba]"
```

pip rebuilds and reinstalls LangSlice from the folder. Then install the
connector again as in step 5: the Antigravity plugin and the Codex files are
copies, so a change to the connector reaches the app only when it is
installed again. Run step 4 to verify.
