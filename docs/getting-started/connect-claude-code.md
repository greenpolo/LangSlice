# Connect Claude Code (Claude)

Claude Code is the coding agent in the **Code** tab of the Claude desktop app.
On this page you install the app and add LangSlice's plugin to it. A
**plugin** is a package of add-ons that an app installs in one go.

**Before you start:**

- [Install LangSlice](install.md), including
  [step 6, which lets your AI app find LangSlice](install.md#step-6-let-your-ai-app-find-langslice).
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
- agent `langslice:register`: registers one brain through LangSlice's tools
  only, with no terminal, no file writing and no web.
- agent `langslice:register-scripting`: registers one brain with LangSlice's
  command-line tool and scripts of its own.
- MCP server `langslice`: LangSlice's tools, started with the session as
  `langslice mcp`.
- hook `SubagentStop`: when a registration agent finishes, copies its
  transcript into the job folder's `trace/` folder.

</details>

## Step 3: Test the connection

1. Start a new session in the **Code** tab.
2. Type `/` in the message box. The list includes
   `/langslice:register-brain`.
3. Send this message:

   ```text
   Call LangSlice's start_job tool with no arguments and tell me what it says.
   ```

4. Claude asks for permission before it uses a LangSlice tool. Allow it.

Claude reports LangSlice's answer: no job is named yet, and LangSlice asks for
an image folder (`start_job(image_folder=...)`). That answer comes from
LangSlice running on your computer, so the connection works.

:::{admonition} Screenshot to come
Claude's permission request for `start_job`, then the reply about naming an
image folder.
:::

If Claude says it has no LangSlice tools, or the server failed to start, see
[Troubleshooting](../troubleshooting.md#the-app-cannot-start-langslice).

<details>
<summary>Stop Claude asking before each LangSlice tool</summary>

Claude Code asks before each call to an MCP tool unless a permission rule
allows it. This rule, in the `permissions` of a Claude Code settings file
(`.claude/settings.json` in your home folder for every folder), allows all of
LangSlice's tools:

```json
{
  "permissions": {
    "allow": ["mcp__plugin_langslice_langslice"]
  }
}
```

</details>

## Next

Your app can now use LangSlice. To register sections, start a new session
and ask Claude to register the brain in your image folder. It follows the
`register-brain` skill and asks you what it needs.
