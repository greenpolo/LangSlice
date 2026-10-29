---
name: setup
description: Set up LangSlice in Antigravity once the LangSlice plugin is installed: find the langslice program, put its folder on PATH so Antigravity can start LangSlice's MCP server, and add the permission rules LangSlice's tools and agents need to Antigravity's settings.json. Use when the user asks to set up LangSlice, connect LangSlice, or allow LangSlice's tools, or when LangSlice's tools are missing or its server failed to start.
---

# Set up LangSlice in Antigravity

For the main session, once per computer. The plugin starts LangSlice's MCP
server as `langslice mcp`, by name, so Antigravity must find `langslice` on
PATH; and Antigravity keeps its permission rules in one settings file for
every session. A plugin can set neither, so you set both. Do the steps in
order.

Until this is done, LangSlice's MCP server may have failed to start and its
tools may be missing from this conversation; that is expected. The settings
file is `~/.gemini/antigravity-cli/settings.json` on macOS and Linux and
`%USERPROFILE%\.gemini\antigravity-cli\settings.json` on Windows. It and the
shell files of step 2 are outside the workspace: ask the user to approve each
write when it is refused.

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

Keep this absolute path for steps 2 and 3.

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

## 3. Find the folders LangSlice reads

Find the Python that runs `langslice`, from the path of step 1:

- macOS and Linux: the first line of that file is `#!` followed by the
  Python's path (`head -1 <path>`).
- Windows: `langslice.exe` sits in the environment's `Scripts` folder; the
  Python is `python.exe` one folder up.

Run it:

```bash
PYTHON -c "import sys, langslice; print(sys.prefix); print(sys.base_prefix); print(langslice.__file__)"
```

The folders to allow are:

- the first line (the environment), and the second if it differs (the Python
  the environment was made from);
- the LangSlice checkout, when the third line is not inside the first: the
  folder above `src`, which holds `pyproject.toml`;
- `~/.brainglobe`, where the atlases are.

On Windows, write every folder with forward slashes, for example
`C:/Users/you/miniforge3/envs/langslice`.

## 4. Add the rules to `settings.json`

Read the file (it may be missing or empty) and keep everything in it as it is.

- Add each of these to the `permissions.allow` list, creating the
  `permissions` object and its `allow` list if they are missing, and skipping
  any already there:

  ```text
  mcp(langslice_mcp/*)
  command(langslice-job)
  read_url(*)
  read_file(FOLDER)      one rule for each folder from step 3
  ```

- If the file has no `enableTerminalSandbox` key, add
  `"enableTerminalSandbox": true`; if it has no `toolPermission` key, add
  `"toolPermission": "proceed-in-sandbox"`. Leave either alone if it is there.
- Write the file back and read it again: it must parse as JSON.

What the rules do: `mcp(langslice_mcp/*)` lets an agent call LangSlice's tools
without asking, all the `register` agent needs. `command(langslice-job)` lets
`langslice-job` run outside the sandbox, where it can read the atlas and the
images; `register-scripting` runs its scripts through it. `read_url(*)` gives
commands in the sandbox the web. The `read_file` rules let a script in the
sandbox import LangSlice and read the atlases.

## 5. Restart and test

Tell the user:

1. What you changed, in one or two lines.
2. Quit Antigravity completely and open it again: Antigravity reads PATH and
   its settings file when it starts.
3. In a new conversation, send: "Call LangSlice's start_job tool with no
   arguments and tell me what it says." LangSlice answers that no job is
   named yet and asks for an image folder; that answer means LangSlice is
   connected. If the tools are still missing, see
   https://langslice.readthedocs.io/en/latest/troubleshooting.html#the-app-cannot-start-langslice.
4. To register, ask Antigravity to register the brain in an image folder (the
   `register-brain` skill).
