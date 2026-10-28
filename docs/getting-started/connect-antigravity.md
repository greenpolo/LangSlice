# Connect Antigravity (Gemini)

Antigravity is Google's app for AI agents that work on your computer, with
Gemini models. On this page you install Antigravity, make LangSlice findable by
it, install LangSlice's plugin, and allow LangSlice's tools.

**Before you start:** [install LangSlice](install.md).

```{note}
The steps on this page follow LangSlice's Antigravity connector, which is
tested from Antigravity's command-line version on Linux. They are untested in
the app on Windows and macOS.
```

## Step 1: Install Antigravity

1. Download Antigravity from <https://antigravity.google> and install it.
2. Open it and sign in with your Google account.

:::{admonition} Screenshot to come
Antigravity's first screen after signing in.
:::

## Step 2: Let Antigravity find LangSlice

Antigravity starts LangSlice by the name `langslice`, so that name has to work
outside the LangSlice environment too.

### Windows

1. Take the line `where langslice` printed in
   [step 6 of the install](install.md#step-6-note-where-langslice-is-installed),
   and remove `\langslice.exe` from its end. For example,
   `C:\Users\you\miniforge3\envs\langslice\Scripts`.
2. Open the Start menu, type `environment variables`, and choose **Edit
   environment variables for your account**.
3. Under **User variables**, select **Path** and click **Edit**.
4. Click **New**, paste the folder from item 1, and click **OK** twice.

:::{admonition} Screenshot to come
The **Edit environment variable** window with the LangSlice `Scripts` folder
added.
:::

### macOS

In Terminal, run:

```bash
echo 'conda activate langslice' >> ~/.zshrc
```

### Linux

In a terminal, run:

```bash
echo 'conda activate langslice' >> ~/.bashrc
```

On macOS and Linux, every new terminal now starts in the LangSlice
environment, and apps started from your desktop find `langslice` too.

On every system, quit Antigravity completely and open it again so it picks up
the change.

## Step 3: Install the LangSlice plugin

Open a terminal in the `LangSlice-main` folder, as in
[step 3 of the install](install.md#step-3-open-a-terminal-in-the-langslice-folder),
and run:

```bash
agy plugin install connectors/antigravity
```

`agy` is Antigravity's command-line tool. The plugin brings LangSlice's skill,
its two registration agents and its MCP server, named `langslice_mcp` in
Antigravity. Quit Antigravity and open it again to load the plugin.

<details>
<summary>What the plugin holds</summary>

- skill `register-brain`: the main agent's instructions. It asks you about the
  registration, prepares it, starts one registration agent and checks the
  result. It never registers sections itself.
- skill `abba`: how to work with ABBA.
- MCP server `langslice_mcp`: LangSlice's tools, started by Antigravity as
  `langslice mcp`.
- agent `register`: registers one brain through LangSlice's tools only, with
  no terminal and no file writing.
- agent `register-scripting`: registers one brain with LangSlice's
  command-line tool and scripts of its own.
- hook `langslice-transcript`: when a session ends, copies its transcript into
  the job folder's `trace/` folder.

</details>

## Step 4: Allow LangSlice's tools

Antigravity runs an agent's commands in a sandbox, a walled-off space that
cannot read most of your files. It also asks before using a tool it has no
rule for.
LangSlice's commands read the atlas and your images, so they need rules.
The rules live in one settings file for every session:
`settings.json` in the folder `.gemini/antigravity-cli` in your home folder.

1. Open that file in a text editor with the command for your system. If the
   file does not exist yet, the commands create it.

   On Windows, in Miniforge Prompt (click **Yes** if Notepad offers to create
   the file):

   ```bat
   mkdir "%USERPROFILE%\.gemini\antigravity-cli"
   notepad "%USERPROFILE%\.gemini\antigravity-cli\settings.json"
   ```

   On macOS, in Terminal:

   ```bash
   mkdir -p ~/.gemini/antigravity-cli && touch ~/.gemini/antigravity-cli/settings.json
   open -e ~/.gemini/antigravity-cli/settings.json
   ```

   On Linux, in a terminal:

   ```bash
   mkdir -p ~/.gemini/antigravity-cli && touch ~/.gemini/antigravity-cli/settings.json
   xdg-open ~/.gemini/antigravity-cli/settings.json
   ```

2. If the file is empty or new, paste the block below. If it already has
   content, add the lines inside `"allow": [ ... ]` to its own `"allow"`
   list.

   ```json
   {
     "enableTerminalSandbox": true,
     "toolPermission": "proceed-in-sandbox",
     "permissions": {
       "allow": [
         "mcp(langslice_mcp/*)",
         "command(langslice-job)",
         "read_url(*)",
         "read_file(/Users/you/Documents/LangSlice-main)",
         "read_file(/Users/you/miniforge3/envs/langslice)",
         "read_file(~/.brainglobe)"
       ]
     }
   }
   ```

3. Replace the two example folder locations with your own: the `LangSlice-main` folder,
   and the LangSlice environment folder (the location of `langslice` without
   `/bin/langslice` at the end). On Windows, write them with forward
   slashes, for example `C:/Users/you/miniforge3/envs/langslice`.
4. Save the file, then quit Antigravity and open it again.

<details>
<summary>What each rule does</summary>

- `mcp(langslice_mcp/*)` lets an agent call LangSlice's tools without asking.
  This is the only rule the `register` agent needs.
- `command(langslice-job)` lets `langslice-job` commands run outside the
  sandbox, where they can read the atlas and your images. The scripting
  agent runs its scripts through `langslice-job` too.
- `read_url(*)` gives commands in the sandbox the web, for package downloads
  and documentation.
- The `read_file` rules let a script in the sandbox read LangSlice, the
  Python it runs on, and the atlases.

</details>

## Step 5: Test the connection

1. Start a new conversation in Antigravity.
2. Send this message:

   ```text
   Call LangSlice's start_job tool with no arguments and tell me what it says.
   ```

Antigravity calls the tool and reports LangSlice's answer: no job is named
yet, and LangSlice asks for an image folder (`start_job(image_folder=...)`).
That answer comes from LangSlice running on your computer, so the connection
works.

:::{admonition} Screenshot to come
The Antigravity conversation after the test message, showing the `start_job`
call and the reply about naming an image folder.
:::

If Antigravity says it has no LangSlice tools, or the server failed to start,
see [Troubleshooting](../troubleshooting.md#the-app-cannot-start-langslice).

## Next

Your app can now use LangSlice. To register sections, start a new
conversation and ask Antigravity to register the brain in your image folder.
It follows the `register-brain` skill and asks you what it needs.
