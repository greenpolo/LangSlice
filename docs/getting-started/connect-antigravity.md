# Connect Antigravity (Gemini)

Antigravity is Google's app for AI agents that work on your computer, with
Gemini models. On this page you install Antigravity, install LangSlice's
plugin, and allow LangSlice's tools. A **plugin** is a package of add-ons
that an app installs in one go.

**Before you start:** [install LangSlice](install.md), including
[step 6, which lets your AI app find LangSlice](install.md#step-6-let-your-ai-app-find-langslice).

```{note}
The pictures on this page are from the app on Linux. The steps are untested
on Windows and macOS.
```

## Step 1: Install Antigravity

1. Download Antigravity from <https://antigravity.google> and install it.
2. Open it and sign in with your Google account.

:::{admonition} Screenshot to come
Antigravity's first screen after signing in.
:::

## Step 2: Install the LangSlice plugin

Antigravity installs plugins through a skill of its own, named `plugin`.

1. Click **New Conversation**.
2. In the message box, type `/plugin`. A list of skills opens above the box.
   Click **plugin**.

   ```{figure} img/antigravity-plugin-skill.png
   :alt: The message box with /plugin typed and the list of matching skills above it; the plugin skill is marked.

   Type `/plugin` and pick the **plugin** skill.
   ```

3. After **plugin** in the message box, type this message and send it:

   ```text
   Please install this plugin: https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
   ```

   ```{figure} img/antigravity-install-message.png
   :alt: The message box holding the plugin skill followed by the install request with LangSlice's GitHub address.

   The message, ready to send.
   ```

4. Antigravity installs the plugin from GitHub and says so. Quit Antigravity
   completely and open it again to load the plugin.

:::{admonition} Screenshot to come
Antigravity's reply after installing the plugin.
:::

The plugin brings LangSlice's skill, its two registration agents and its MCP
server, named `langslice_mcp` in Antigravity.

<details>
<summary>The same install from a terminal</summary>

Antigravity's command-line tool, `agy`, installs the plugin with one command.
Run it in any terminal, then quit Antigravity and open it again:

```bash
agy plugin install https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
```

</details>

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

## Step 3: Allow LangSlice's tools

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

## Step 4: Test the connection

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
