---
name: setup
description: Set up LangSlice in Antigravity once the LangSlice plugin is installed: check that the langslice program runs, add the permission rules LangSlice's tools and agents need to Antigravity's settings.json, and test the connection. Use when the user asks to set up LangSlice, connect LangSlice, or allow LangSlice's tools.
---

# Set up LangSlice in Antigravity

For the main session, once per computer. Antigravity keeps its permission
rules in one settings file for every session, and a plugin cannot carry them,
so you add LangSlice's rules there. Do the steps in order.

The settings file is `~/.gemini/antigravity-cli/settings.json` on macOS and
Linux and `%USERPROFILE%\.gemini\antigravity-cli\settings.json` on Windows.
Writing there is outside the workspace: ask the user to approve the write
when it is refused.

## 1. Check `langslice`

- macOS and Linux: `command -v langslice`. Windows: `where langslice`.
- If that finds nothing, the plugin's MCP server cannot start either: tell
  the user to finish step 6 of LangSlice's install
  (https://langslice.readthedocs.io/en/latest/getting-started/install.html#step-6-let-your-ai-app-find-langslice),
  quit Antigravity completely, open it again and ask you again. Stop.
- Run `langslice version`. If it fails, show the user the error and stop.

## 2. Find the folders LangSlice reads

Find the Python that runs `langslice`:

- macOS and Linux: the first line of the `langslice` file is `#!` followed
  by that Python's path (`head -1 "$(command -v langslice)"`).
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

## 3. Add the rules to `settings.json`

Read the file (it may be missing or empty) and keep everything in it as it is.

- Add each of these to the `permissions.allow` list, creating the
  `permissions` object and its `allow` list if they are missing, and skipping
  any already there:

  ```text
  mcp(langslice_mcp/*)
  command(langslice-job)
  read_url(*)
  read_file(FOLDER)      one rule for each folder from step 2
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

## 4. Test and finish

Call LangSlice's `start_job` tool (server `langslice_mcp`) with no arguments.
LangSlice answers that no job is named yet and asks for an image folder; that
answer means LangSlice is connected. If the server is missing or failed to
start, tell the user to quit Antigravity completely, open it again, and ask
you to test again; if it still fails, point them to
https://langslice.readthedocs.io/en/latest/troubleshooting.html#the-app-cannot-start-langslice.

Then tell the user:

1. What you added to `settings.json`, in one or two lines.
2. Quit Antigravity completely and open it again: Antigravity reads the
   settings file when it starts.
3. To register, ask Antigravity to register the brain in an image folder (the
   `register-brain` skill).
