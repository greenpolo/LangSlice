# Connect Antigravity (Gemini)

Antigravity is Google's app for AI agents that work on your computer, with
Gemini models. On this page you install Antigravity, install LangSlice's
plugin, and ask Antigravity to finish setting LangSlice up. A **plugin** is a
package of add-ons that an app installs in one go.

**Before you start:** [install LangSlice](install.md).

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

The plugin brings LangSlice's skills, its two registration agents and its MCP
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
- skill `setup`: lets Antigravity find LangSlice, and adds LangSlice's
  permission rules to Antigravity's settings (step 3).
- MCP server `langslice_mcp`: LangSlice's tools, started by Antigravity as
  `langslice mcp`.
- agent `register`: registers one brain through LangSlice's tools only, with
  no terminal and no file writing.
- agent `register-scripting`: registers one brain with LangSlice's
  command-line tool and scripts of its own.
- hook `langslice-transcript`: when a session ends, copies its transcript into
  the job folder's `trace/` folder.

</details>

## Step 3: Ask Antigravity to set up LangSlice

The plugin starts LangSlice by its name, `langslice`, and so far that name
works only in a terminal where the LangSlice environment is active. Also,
Antigravity asks before using a tool it has no rule for, and runs an agent's
commands in a sandbox, a walled-off space that cannot read most of your files.
LangSlice's registration agents need rules that let them use LangSlice's tools
and read the atlas and your images. A skill in the plugin, named `setup`,
fixes both.

1. Click **New Conversation**.
2. Send this message:

   ```text
   Set up LangSlice.
   ```

Antigravity follows the `setup` skill. It finds `langslice` on your
computer, adds its folder to the list of places your computer looks for
programs, finds the folders LangSlice reads, and adds LangSlice's rules to
Antigravity's settings file. When it is done, it tells you to restart the
app.

Antigravity may ask permission to write files outside the workspace, the
folder the conversation works in. Allow it. The files are your shell's
startup files in your home folder (on Windows, your user environment
variables instead), and `settings.json` in the folder
`.gemini/antigravity-cli` in your home folder, where Antigravity keeps its
own settings.

In this first conversation, Antigravity may say that LangSlice's server
failed to start or that it has no LangSlice tools. That is expected until
setup is done and the app restarts.

:::{admonition} Screenshot to come
The setup conversation: Antigravity asking to write the settings file, then
reporting what it set up.
:::

<details>
<summary>What setup changed</summary>

- `PATH`, the list of folders your computer searches for programs, gets the
  folder that holds `langslice` and `langslice-job`, added at the end. On
  macOS and Linux, setup adds these two lines, with that folder, to your
  shell's startup files (`~/.zprofile` and `~/.zshrc` for zsh, the default
  on macOS; `~/.bashrc` and `~/.bash_profile` or `~/.profile` for bash):

  ```sh
  # Added by LangSlice setup: lets apps find langslice and langslice-job
  case ":$PATH:" in *":/Users/you/miniforge3/envs/langslice/bin:"*) ;; *) export PATH="$PATH:/Users/you/miniforge3/envs/langslice/bin" ;; esac
  ```

  On Windows, setup adds the folder, such as
  `C:\Users\you\miniforge3\envs\langslice\Scripts`, to the `Path` in your
  user environment variables. Setup does not activate the LangSlice
  environment and changes nothing else.

Setup adds these rules to the `allow` list in `settings.json`, with the
folders of your computer, and keeps everything else in the file:

```json
{
  "enableTerminalSandbox": true,
  "toolPermission": "proceed-in-sandbox",
  "permissions": {
    "allow": [
      "mcp(langslice_mcp/*)",
      "command(langslice-job)",
      "read_url(*)",
      "read_file(/Users/you/miniforge3/envs/langslice)",
      "read_file(~/.brainglobe)"
    ]
  }
}
```

- `mcp(langslice_mcp/*)` lets an agent call LangSlice's tools without asking.
  This is the only rule the `register` agent needs.
- `command(langslice-job)` lets `langslice-job` commands run outside the
  sandbox, where they can read the atlas and your images. The scripting
  agent runs its scripts through `langslice-job` too.
- `read_url(*)` gives commands in the sandbox the web, for package downloads
  and documentation.
- The `read_file` rules let a script in the sandbox read LangSlice, the
  Python it runs on, and the atlases.
- `enableTerminalSandbox` and `toolPermission` are added only if the file
  does not set them already.

</details>

## Step 4: Restart Antigravity

Quit Antigravity completely and open it again. Antigravity reads `PATH` and
its settings file when it starts.

## Step 5: Test the connection

1. Click **New Conversation**.
2. Send this message:

   ```text
   Call LangSlice's start_job tool with no arguments and tell me what it says.
   ```

Antigravity calls the tool and reports LangSlice's answer: no job is named
yet, and LangSlice asks for an image folder (`start_job(image_folder=...)`).
That answer comes from LangSlice running on your computer, so the connection
works.

:::{admonition} Screenshot to come
The conversation after the test message, showing the reply about naming an
image folder.
:::

If Antigravity says it has no LangSlice tools, or the server failed to start,
see [Troubleshooting](../troubleshooting.md#the-app-cannot-start-langslice).

## Next

Your app can now use LangSlice. To register sections, start a new
conversation and ask Antigravity to register the brain in your image folder.
It follows the `register-brain` skill and asks you what it needs.
