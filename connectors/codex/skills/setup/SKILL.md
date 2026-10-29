---
name: setup
description: Set up LangSlice in Codex once the LangSlice plugin is installed: find the langslice program, put its folder on PATH, copy LangSlice's two registration agents and its command rules into Codex's folder, and add LangSlice's MCP server to Codex's config.toml. Use when the user asks to set up LangSlice, connect LangSlice, or install LangSlice's agents.
---

# Set up LangSlice in Codex

For the main session, once per computer. A Codex plugin cannot carry custom
agents or command rules, and the `register` agent does not see a plugin's MCP
server, so you put these in Codex's own folder. Do the steps in order.

Codex's folder is `~/.codex` on macOS and Linux and `%USERPROFILE%\.codex`
(`$HOME\.codex` in PowerShell) on Windows; if the `CODEX_HOME` environment
variable is set, use that folder instead everywhere below. Codex's folder and
the shell files of step 2 are outside the workspace: ask for approval when the
sandbox refuses a write.

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

Keep this absolute path for steps 2 and 4.

## 2. Put LangSlice's folder on PATH

Apps and LangSlice's agents run `langslice` and `langslice-job` by name, so
the folder that holds them must be on PATH everywhere, not only in a terminal
where the LangSlice environment is active. That folder is the one holding the
path from step 1 (the environment's `bin`, or `Scripts` on Windows). Add it
to the end of PATH and change nothing else: no `conda activate`.

macOS and Linux:

- Skip to step 3 if a new login shell already finds it:
  `"$SHELL" -ilc 'command -v langslice'` prints the path from step 1.
- Otherwise add these two lines, with the folder in place of `FOLDER`, to
  the end of each file the table names for the user's shell (`$SHELL`).
  Create a file that is missing; leave alone a file that already holds the
  comment line.

  ```sh
  # Added by LangSlice setup: lets apps find langslice and langslice-job
  case ":$PATH:" in *":FOLDER:"*) ;; *) export PATH="$PATH:FOLDER" ;; esac
  ```

  | Shell | Files |
  | --- | --- |
  | zsh | `~/.zprofile` and `~/.zshrc` |
  | bash | `~/.bashrc`, and `~/.bash_profile` if it exists, else `~/.bash_login` if it exists, else `~/.profile` |
  | another | tell the user the folder to add to PATH, and go on |

  Both files, because each kind of shell reads a different one: the apps
  read PATH from a login, interactive shell (`$SHELL -ilc`), agents' commands
  may run in a login shell that is not interactive (`-lc`), and a new
  terminal on Linux is interactive but not a login shell.
- Check: `"$SHELL" -ilc 'command -v langslice'` and
  `"$SHELL" -lc 'command -v langslice'` each print the path from step 1.

Windows (PowerShell), with the folder in place of `FOLDER`:

```powershell
$dir = 'FOLDER'
$user = [Environment]::GetEnvironmentVariable('Path', 'User')
$all = "$user;" + [Environment]::GetEnvironmentVariable('Path', 'Machine')
if (($all -split ';') -notcontains $dir) {
  $parts = @($user -split ';' | Where-Object { $_ }) + $dir
  [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
}
```

This appends the folder to the user's Path and keeps every entry already
there. Check: `[Environment]::GetEnvironmentVariable('Path', 'User')` ends
with the folder (or the folder was already on Path).

## 3. Copy the agents and the rules

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

## 4. Add LangSlice to `config.toml`

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

## 5. Restart and test

Tell the user:

1. Quit the Codex app completely and open it again: Codex reads PATH, its
   agents, rules and settings when it starts.
2. In a new chat, send: "Call LangSlice's start_job tool with no arguments
   and tell me what it says." LangSlice answers that no job is named yet and
   asks for an image folder; that answer means LangSlice is connected.
3. To register, ask Codex to register the brain in an image folder (the
   `register-brain` skill).
