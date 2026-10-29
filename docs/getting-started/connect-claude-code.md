# Connect Claude Code (Claude)

Claude Code is the coding agent in the **Code** tab of the Claude desktop app.
On this page you install the app, add LangSlice's plugin to it, and ask
Claude to finish setting LangSlice up. A **plugin** is a package of add-ons
that an app installs in one go.

**Before you start:**

- [Install LangSlice](install.md).
- Have a paid Claude plan. Claude Code is not part of the free Claude plan.
  On the free plan, use [Codex](connect-codex.md) with a free ChatGPT account
  instead.
- On Windows, install Git for Windows from <https://git-scm.com/downloads/win>,
  accepting the default choices. The Claude app uses it to download
  LangSlice's plugin.

```{note}
The pictures on this page are from the app on Linux. The steps are untested
on Windows and macOS.
```

## Step 1: Install the Claude desktop app

1. Download the Claude desktop app from <https://claude.com/download> and
   install it.
2. Open it and sign in with your Claude account.
3. Click the **Code** tab, the `</>` button at the top of the left-hand
   column.
4. Under the message box, choose **Local**, then **Select folder…**, and pick
   a folder to work in, such as the folder that holds your section images.

```{figure} img/claude-code-tab.png
:alt: The Claude app on the Code tab, with the Code button at the top left and the Local and Select folder buttons above the message box marked.
:width: 80%

The **Code** tab. **Local** and **Select folder…** sit above the message
box.
```

## Step 2: Add LangSlice's plugin

LangSlice's plugin comes from a **marketplace**: a list of plugins that
someone publishes, here LangSlice's own, on GitHub.

1. In the left-hand column, click **Customize**, then **Plugins** at the top
   of the page.
2. Click **Add**, then **Add marketplace**.

   ```{figure} img/claude-plugins-add-menu.png
   :alt: The Customize page with Customize, the Plugins tab, the Add button and the menu item Add marketplace marked.

   **Customize**, **Plugins**, **Add** and **Add marketplace**.
   ```

3. Click **Add from a repository**.

   ```{figure} img/claude-add-marketplace.png
   :alt: The Add marketplace window with two choices; Add from a repository is marked.
   :width: 70%

   Choose **Add from a repository**.
   ```

4. In **URL**, type `greenpolo/LangSlice`, then click **Sync**.

   ```{figure} img/claude-marketplace-url.png
   :alt: The Add marketplace window with a warning to add only trusted marketplaces, and the URL field marked.
   :width: 70%

   The **URL** field, where `greenpolo/LangSlice` goes.
   ```

5. Find **LangSlice** in the list of plugins and install it.

:::{admonition} Screenshot to come
LangSlice in the plugin list, with its install button.
:::

<details>
<summary>What the plugin holds</summary>

- skill `langslice:register-brain`: the main agent's instructions. It asks you
  about the registration: tasks, atlas, image model, an earlier registration
  and pixel size. It prepares the job, starts one registration agent, checks
  the result and handles export. It never registers sections itself.
- skill `langslice:abba`: how to work with ABBA.
- skill `langslice:setup`: lets the app find LangSlice and lets LangSlice's
  tools run without asking (step 3).
- agent `langslice:register`: registers one brain through LangSlice's tools
  only, with no terminal, no file writing and no web.
- agent `langslice:register-scripting`: registers one brain with LangSlice's
  command-line tool and scripts of its own.
- MCP server `langslice`: LangSlice's tools, started with the session as
  `langslice mcp`.
- hook `SubagentStop`: when a registration agent finishes, copies its
  transcript into the job folder's `trace/` folder.

</details>

## Step 3: Ask Claude to set up LangSlice

The plugin starts LangSlice by its name, `langslice`, and so far that name
works only in a terminal where the LangSlice environment is active. Claude
Code also asks before each use of a LangSlice tool. A skill in the plugin,
named `setup`, fixes both.

1. Start a new session in the **Code** tab.
2. Send this message:

   ```text
   Set up LangSlice.
   ```

Claude follows the `langslice:setup` skill. It finds `langslice` on your
computer, adds its folder to the list of places your computer looks for
programs, and adds a rule to Claude Code's settings that lets LangSlice's
tools run without asking. When it is done, it tells you to restart the app.

Claude asks permission before it changes files outside the folder the
session works in. Allow it. The files are your shell's startup files in your
home folder (on Windows, your user environment variables instead), and
`settings.json` in the `.claude` folder in your home folder, where Claude
Code keeps its own settings.

In this first session, Claude may say that LangSlice's server failed to start
or that it has no LangSlice tools. That is expected until setup is done and
the app restarts.

:::{admonition} Screenshot to come
The setup session: Claude asking to change a file outside the folder, then
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
- `settings.json` in the `.claude` folder gets this rule, and keeps everything
  else in the file:

  ```json
  {
    "permissions": {
      "allow": ["mcp__plugin_langslice_langslice"]
    }
  }
  ```

  The rule lets all of LangSlice's tools run without asking you before each
  call.

</details>

## Step 4: Restart the app

Quit the Claude desktop app completely and open it again. The app reads
`PATH` and Claude Code's settings when it starts.

## Step 5: Test the connection

1. Start a new session in the **Code** tab.
2. Type `/` in the message box. The list includes
   `/langslice:register-brain`.
3. Send this message:

   ```text
   Call LangSlice's start_job tool with no arguments and tell me what it says.
   ```

Claude reports LangSlice's answer: no job is named yet, and LangSlice asks for
an image folder (`start_job(image_folder=...)`). That answer comes from
LangSlice running on your computer, so the connection works.

:::{admonition} Screenshot to come
The reply about naming an image folder after the test message.
:::

If Claude says it has no LangSlice tools, or the server failed to start, see
[Troubleshooting](../troubleshooting.md#the-app-cannot-start-langslice).

## Next

Your app can now use LangSlice. To register sections, start a new session
and ask Claude to register the brain in your image folder. It follows the
`register-brain` skill and asks you what it needs.
