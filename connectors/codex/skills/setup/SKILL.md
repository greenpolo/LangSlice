---
name: setup
description: Set up LangSlice in Codex once the LangSlice plugin is installed: find the langslice program, copy LangSlice's two registration agents and its command rules into Codex's folder, and add LangSlice's MCP server to Codex's config.toml. Use when the user asks to set up LangSlice, connect LangSlice, or install LangSlice's agents.
---

# Set up LangSlice in Codex

For the main session, once per computer. A Codex plugin cannot carry custom
agents or command rules, and the `register` agent does not see a plugin's MCP
server, so you put these in Codex's own folder. Do the steps in order.

Codex's folder is `~/.codex` on macOS and Linux and `%USERPROFILE%\.codex`
(`$HOME\.codex` in PowerShell) on Windows; if the `CODEX_HOME` environment
variable is set, use that folder instead everywhere below. Writing there is
outside the workspace: ask for approval when the sandbox refuses a write.

## 1. Find `langslice`

- macOS and Linux: `command -v langslice`. Windows: `where langslice`.
- If that finds nothing, look for the conda environment named `langslice`:
  - macOS and Linux: `envs/langslice/bin/langslice` under the folder
    `conda info --base` prints, then under `~/miniforge3`, `~/mambaforge`,
    `~/miniconda3` and `~/anaconda3`.
  - Windows: `%USERPROFILE%\miniforge3\envs\langslice\Scripts\langslice.exe`,
    then the same under `mambaforge`, `miniconda3`, `anaconda3` and the
    folder `conda info --base` prints.
- If none exists, tell the user to finish LangSlice's install first
  (https://langslice.readthedocs.io/en/latest/getting-started/install.html)
  and stop.
- Run `<path> version`. If it fails, show the user the error and stop.

Keep this absolute path for step 3.

## 2. Copy the agents and the rules

The plugin folder is two levels above the folder of this `SKILL.md`
(`skills/setup/` sits inside it). It holds `agents/register.toml`,
`agents/register_scripting.toml` and `rules/langslice.rules`.

macOS and Linux:

```bash
PLUGIN="<plugin folder>"
mkdir -p ~/.codex/agents ~/.codex/rules
cp "$PLUGIN/agents/register.toml" "$PLUGIN/agents/register_scripting.toml" ~/.codex/agents/
cp "$PLUGIN/rules/langslice.rules" ~/.codex/rules/
```

Windows (PowerShell):

```powershell
$plugin = "<plugin folder>"
New-Item -ItemType Directory -Force "$HOME\.codex\agents", "$HOME\.codex\rules" | Out-Null
Copy-Item -Force "$plugin\agents\register.toml", "$plugin\agents\register_scripting.toml" "$HOME\.codex\agents\"
Copy-Item -Force "$plugin\rules\langslice.rules" "$HOME\.codex\rules\"
```

## 3. Add LangSlice to `config.toml`

Edit `config.toml` in Codex's folder (create it if it is missing). Keep
everything else in the file as it is.

- If it has no `[mcp_servers.langslice]` table, add this one, with the path
  from step 1:

  ```toml
  [mcp_servers.langslice]
  command = "/absolute/path/to/langslice"
  args = ["mcp"]
  tool_timeout_sec = 3600
  default_tools_approval_mode = "approve"
  ```

  On Windows, write the path in single quotes so its backslashes stay as
  they are: `command = 'C:\Users\you\miniforge3\envs\langslice\Scripts\langslice.exe'`.
- If the table is there and its `command` is a different path, change only
  `command`.
- Put `mcp_tool_list_changed = true` in the `[features]` table: add the line
  to the existing `[features]` table, or set it to `true` if it is there; if
  the file has no `[features]` table, add one with that line. A TOML file may
  hold only one `[features]` table.
- Check the result: `codex mcp list`, if the `codex` command runs here,
  lists `langslice` with the path from step 1 (and fails if the file does not
  parse). Otherwise read the file back.

## 4. Restart and test

Tell the user:

1. Quit the Codex app completely and open it again: Codex reads its agents,
   rules and settings when it starts.
2. In a new chat, send: "Call LangSlice's start_job tool with no arguments
   and tell me what it says." LangSlice answers that no job is named yet and
   asks for an image folder; that answer means LangSlice is connected.
3. To register, ask Codex to register the brain in an image folder (the
   `register-brain` skill).
