---
name: setup
description: Set up LangSlice in Claude Code once the LangSlice plugin is installed: find the langslice program, put its folder on PATH so the app can start LangSlice's MCP server, and let LangSlice's tools run without asking. Use when the user asks to set up LangSlice or connect LangSlice, or when LangSlice's tools are missing or its server failed to start.
---

# Set up LangSlice in Claude Code

For the main session, once per computer. The plugin starts LangSlice's MCP
server as `langslice mcp`, by name, so the app must find `langslice` on PATH;
and Claude Code asks before each MCP tool call unless a rule in its settings
allows it. A plugin can set neither, so you set both. Do the steps in order.

Until this is done, LangSlice's MCP server may have failed to start and its
tools may be missing from this session; that is expected. The files below
are outside the project folder: ask the user to approve each write.

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

Keep this absolute path for step 2.

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

## 3. Let LangSlice's tools run without asking

The settings file is `~/.claude/settings.json` on macOS and Linux and
`%USERPROFILE%\.claude\settings.json` on Windows; if the `CLAUDE_CONFIG_DIR`
environment variable is set, it is `settings.json` in that folder.

Read the file (it may be missing or empty) and keep everything in it as it is.

- Add `"mcp__plugin_langslice_langslice"` to the `permissions.allow` list,
  creating the `permissions` object and its `allow` list if they are missing,
  and skipping it if it is already there.
- Write the file back and read it again: it must parse as JSON.

The rule allows every tool of the plugin's LangSlice MCP server, all the
`langslice:register` agent needs.

## 4. Restart and test

Tell the user:

1. What you changed, in one or two lines.
2. Quit the Claude app completely and open it again: the app reads PATH and
   its settings when it starts.
3. In a new session, send: "Call LangSlice's start_job tool with no
   arguments and tell me what it says." LangSlice answers that no job is
   named yet and asks for an image folder; that answer means LangSlice is
   connected. If the tools are still missing, see
   https://langslice.readthedocs.io/en/latest/troubleshooting.html#the-app-cannot-start-langslice.
4. To register, ask Claude to register the brain in an image folder (the
   `langslice:register-brain` skill).
