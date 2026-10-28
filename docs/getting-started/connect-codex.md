# Connect Codex (ChatGPT)

Codex is the coding agent inside the ChatGPT desktop app. On this page you
install the app, copy LangSlice's skill and registration agents into Codex's
folders, and add LangSlice to Codex's settings file.

**Before you start:** [install LangSlice](install.md), and have the location
of `langslice` from its last step at hand.

```{note}
The steps on this page follow LangSlice's Codex connector. They are untested
in the desktop app on Windows and macOS.
```

## Step 1: Install the ChatGPT desktop app

1. Download the ChatGPT desktop app from <https://chatgpt.com/download> and
   install it.
2. Open it and sign in with your ChatGPT account.
3. Choose **Codex**, then **New chat**.

:::{admonition} Screenshot to come
The ChatGPT desktop app with **Codex** selected and a new chat open.
:::

## Step 2: Copy LangSlice's skill and agents into Codex

Open a terminal in the `LangSlice-main` folder, as in
[step 3 of the install](install.md#step-3-open-a-terminal-in-the-langslice-folder).
Then copy and run the commands for your system.

On Windows, in Miniforge Prompt:

```bat
mkdir "%USERPROFILE%\.agents\skills" "%USERPROFILE%\.codex\agents" "%USERPROFILE%\.codex\rules"
xcopy /E /I /Y connectors\codex\skills "%USERPROFILE%\.agents\skills"
copy /Y connectors\codex\agents\*.toml "%USERPROFILE%\.codex\agents\"
copy /Y connectors\codex\rules\langslice.rules "%USERPROFILE%\.codex\rules\"
```

On macOS and Linux, in Terminal:

```bash
mkdir -p ~/.agents/skills ~/.codex/agents ~/.codex/rules
cp -r connectors/codex/skills/* ~/.agents/skills/
cp connectors/codex/agents/*.toml ~/.codex/agents/
cp connectors/codex/rules/langslice.rules ~/.codex/rules/
```

These commands print little or nothing when they work. On Windows, `mkdir`
says a folder already exists if it does; that is fine.

<details>
<summary>What these files are</summary>

- `skills/register-brain` is the skill the main agent follows: it asks you
  about the registration, prepares it, starts one registration agent and
  checks the result. It never registers sections itself.
- `skills/abba` tells the main agent how to work with ABBA.
- `agents/register.toml` is the registration agent that works only through
  LangSlice's tools, with no terminal.
- `agents/register_scripting.toml` is the registration agent that can also
  write and run its own scripts.
- `rules/langslice.rules` lets `langslice-job` commands run without asking you
  each time.

</details>

## Step 3: Add LangSlice to Codex's settings

Codex keeps its settings in a file named `config.toml`, in a folder named
`.codex` in your home folder. The ChatGPT desktop app and Codex's other
versions share it.

### Windows

1. In Miniforge Prompt, run:

   ```bat
   notepad "%USERPROFILE%\.codex\config.toml"
   ```

   If Notepad asks whether to create the file, click **Yes**.

2. Paste this at the end of the file:

   ```toml
   [mcp_servers.langslice]
   command = 'C:\Users\you\miniforge3\envs\langslice\Scripts\langslice.exe'
   args = ["mcp"]
   tool_timeout_sec = 3600
   default_tools_approval_mode = "approve"

   [features]
   mcp_tool_list_changed = true
   ```

3. Replace `C:\Users\you\miniforge3\envs\langslice\Scripts\langslice.exe` with
   the line `where langslice` printed. Keep the single quotes around it.
4. Save the file and close Notepad.

### macOS and Linux

With the LangSlice environment active (the line starts with `(langslice)`),
run this in Terminal. It adds the settings to the end of the file, with the
location of `langslice` filled in:

```bash
mkdir -p ~/.codex
cat >> ~/.codex/config.toml <<EOF

[mcp_servers.langslice]
command = "$(which langslice)"
args = ["mcp"]
tool_timeout_sec = 3600
default_tools_approval_mode = "approve"

[features]
mcp_tool_list_changed = true
EOF
```

To check it, run `cat ~/.codex/config.toml`. The `command` line shows the full
location of `langslice`, ending in `/envs/langslice/bin/langslice`.

### If the file already had a `[features]` line

The file may hold only one `[features]` section. If it had one before you
pasted, delete the `[features]` line you added and move
`mcp_tool_list_changed = true` under the earlier `[features]` line.

<details>
<summary>What each setting does</summary>

- `command` and `args` start LangSlice's MCP server, `langslice mcp`.
- `tool_timeout_sec = 3600` lets one LangSlice tool run for up to an hour.
  Codex's own limit is 60 seconds, shorter than a deformation fit.
- `default_tools_approval_mode = "approve"` lets LangSlice's tools run without
  asking you before each call.
- `mcp_tool_list_changed = true` lets Codex pick up the tools a registration
  needs. LangSlice shows them only once the agent has opened the job; without
  this setting the agent sees only `start_job`.

</details>

## Step 4: Restart the app

Quit the ChatGPT desktop app completely and open it again. Codex reads its
settings when it starts.

## Step 5: Test the connection

1. Start a new Codex chat.
2. Type `/mcp` and press **Enter**. The list of connected servers includes
   `langslice`.
3. Send this message:

   ```text
   Call LangSlice's start_job tool with no arguments and tell me what it says.
   ```

Codex calls the tool and reports LangSlice's answer: no job is named yet, and
LangSlice asks for an image folder (`start_job(image_folder=...)`). That
answer comes from LangSlice running on your computer, so the connection works.

:::{admonition} Screenshot to come
The Codex chat after the test message, showing the `start_job` call and the
reply about naming an image folder.
:::

If Codex says it has no LangSlice tools, or the `langslice` server failed to
start, see [Troubleshooting](../troubleshooting.md#the-app-cannot-start-langslice).

## Next

Your app can now use LangSlice. To register sections, open a new chat and ask
Codex to register the brain in your image folder. It follows the
`register-brain` skill and asks you what it needs.
