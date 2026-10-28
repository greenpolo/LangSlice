# Troubleshooting

Each heading is a symptom or the message you see, quoted as it appears. Use
your browser's search (**Ctrl+F**, or **Cmd+F** on a Mac) to find yours. In
the messages, `…` stands for a folder or name that differs on your computer.

Messages from a registration usually reach you through the agent, which may
reword them. Search for a few distinctive words.

## Installing

### "'conda' is not recognized as an internal or external command, operable program or batch file."

You are in a Windows terminal that does not know Miniforge. Open **Miniforge
Prompt** from the Start menu and run the command there.

### "zsh: command not found: conda" or "bash: conda: command not found"

Miniforge is not set up in this terminal, on macOS or Linux. Close the
terminal and open a new one. If the message stays, run this once, then open a
new terminal:

```bash
~/miniforge3/bin/conda init "$(basename "$SHELL")"
```

### Conda cannot find `environment.yml`

The terminal is not in the LangSlice folder. Go to it as in
[step 3 of the install](getting-started/install.md#step-3-open-a-terminal-in-the-langslice-folder),
then run `conda env create -f environment.yml` again.

### Conda says the `langslice` environment already exists

LangSlice was installed before. To replace it, for example with a newer
download, remove the old environment and create it again:

```bash
conda env remove -n langslice
conda env create -f environment.yml
```

Your AI app settings stay valid, because the new environment has the same
name and location.

### "'langslice' is not recognized as an internal or external command" or "command not found: langslice"

The LangSlice environment is not active in this terminal. Run
`conda activate langslice` and try again. The line where you type then starts
with `(langslice)`.

### The first registration waits a long time before anything happens

LangSlice is downloading the atlas, about 600 MB for the Allen mouse atlas at
25 µm. This happens once per atlas; later registrations reuse it.

## Connecting your AI app

### The app cannot start LangSlice

The app says the LangSlice server failed to start, or its agent says it has no
LangSlice tools.

- **Codex:** the `command` line in `config.toml` does not name the `langslice`
  program. Run `where langslice` (Windows) or `which langslice` (macOS,
  Linux) with the environment active, and put that exact line in `command`
  ([step 3](getting-started/connect-codex.md#step-3-add-langslice-to-codexs-settings)).
- **Antigravity and Claude Code:** the app cannot find `langslice` outside the
  LangSlice environment. Open a new terminal without running
  `conda activate langslice`, and run `langslice version`. If it fails, redo
  "Let the app find LangSlice" for
  [Antigravity](getting-started/connect-antigravity.md#step-2-let-antigravity-find-langslice)
  or [Claude Code](getting-started/connect-claude-code.md#step-2-let-claude-find-langslice).
- **Every app:** quit the app completely and open it again after a change. It
  reads its settings when it starts.

### Codex sees only `start_job`, and the registration agent cannot find the job's tools

LangSlice shows a job's tools only once the agent has opened the job, and
Codex follows that change only with `mcp_tool_list_changed = true` under
`[features]` in `config.toml`. Add it
([step 3](getting-started/connect-codex.md#step-3-add-langslice-to-codexs-settings)),
then restart the app.

### Codex stops a LangSlice tool after 60 seconds

Codex's own time limit for a tool is 60 seconds, shorter than a deformation
fit. Add `tool_timeout_sec = 3600` under `[mcp_servers.langslice]` in
`config.toml`, then restart the app.

### Codex reports an error in `config.toml` after you edited it

Two common causes:

- The file has two `[features]` lines. Keep the first, and move the lines
  under the second one up under it.
- On Windows, the `command` location is in double quotes. Write it in single
  quotes, `'C:\Users\…\langslice.exe'`: inside double quotes, a backslash
  starts a special character.

### The app asks permission before every LangSlice tool

- **Codex:** add `default_tools_approval_mode = "approve"` under
  `[mcp_servers.langslice]` in `config.toml`.
- **Antigravity:** add the rule `"mcp(langslice_mcp/*)"` to the `allow` list in
  its settings file ([step 4](getting-started/connect-antigravity.md#step-4-allow-langslices-tools)).
- **Claude Code:** add the rule `"mcp__plugin_langslice_langslice"` to
  `permissions.allow` in a Claude Code settings file
  ([step 4](getting-started/connect-claude-code.md#step-4-test-the-connection)).

### A second registration in the same conversation took over the first one's job

LangSlice's tools hold one job at a time in each conversation. Register one
brain per conversation, and start a new conversation for the next brain.

## While a registration runs

### "No LangSlice job in … (neither a job folder nor an image folder with one beside it)."

The folder named has no LangSlice job yet. The agent creates the job first
(`langslice-job <image folder> init`), then registers it. Check that the
folder you gave is the one that holds your section images.

### "No section images in …."

The folder has no section images LangSlice can read. Give a folder of TIFF,
PNG or JPEG files, one file per section.

### "The images of the job in … are not in … any more."

The section images moved after the job was made. Move the job folder
(`langslice`) back beside the images. Or ask the agent to reattach the job from
the images' new folder; the message gives the command.

### "The job in this folder was made with other settings: …."

The folder already holds a job with other settings, such as another atlas or
other tasks. Continue it with the settings it was made with (they are in
`job.json` in the job folder), or start it over, which discards its work.

### "The job in … was made from other supplied inputs"

The job was made from other known values (positions, transforms, cutting
angles). Continue it with the values it was made from, or start a new job from
the new ones, which replaces the old job's work.

### "… is locked by another writer (waited 300 s)"

Another program was writing to the job folder, such as a second agent working
on the same brain. Wait for it to finish, then try again. Run one registration
per job folder at a time.

### "… needs the job's image model (…), which is not connected to LangSlice here (no key or login)."

The job uses an optional image model to trace borders, and LangSlice has no
login or key for it on this computer. Sign in with `langslice login` (for the
ChatGPT image model) or set the provider's key, then try again. Without an
image model, the agent fits each section's deformation to the stain alone.

### "Run … ended without an answer."

A long fit ran in the background and its process ended early. The agent can
run it again. If every long fit ends this way, the app stops programs that a
command leaves running. The agent then runs long fits with
`LANGSLICE_DETACH=0` in front of the command.

### "Unknown job setting(s) […]"

The job folder was made by an earlier LangSlice whose settings this version no
longer has. Open it with the LangSlice version that made it, or start a new
job on the same images.

### The agent says it cannot change a section because you froze it

When the job was made, that property was marked as fixed: position, cutting
angles, orientation, in-plane transform or deformation. LangSlice refuses
every change to it. To let the agent change it, make a new job without that
setting.
