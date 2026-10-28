# Connect Claude Code (Claude)

Claude Code is the coding agent in the **Code** tab of the Claude desktop app.
On this page you install the app, make LangSlice findable by it, and install
LangSlice's plugin.

**Before you start:**

- [Install LangSlice](install.md).
- Have a paid Claude plan. Claude Code is not part of the free Claude plan.
  On the free plan, use [Codex](connect-codex.md) with a free ChatGPT account
  instead.

```{note}
The steps on this page follow LangSlice's Claude Code plugin, which is tested
in Claude Code's command-line version on Linux. They are untested in the
desktop app on Windows and macOS.
```

## Step 1: Install the Claude desktop app

1. Download the Claude desktop app from <https://claude.com/download> and
   install it.
2. Open it and sign in with your Claude account.
3. Click the **Code** tab.
4. Start a session on your computer in a folder of your choice, such as the
   folder that holds your section images.

:::{admonition} Screenshot to come
The Claude desktop app on the **Code** tab, with a local session open.
:::

## Step 2: Let Claude find LangSlice

Claude Code starts LangSlice by the name `langslice`, so that name has to work
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
environment, and the Claude app reads the same setting when it starts.

On every system, quit the Claude app completely and open it again so it picks
up the change.

## Step 3: Install the LangSlice plugin

In the **Code** tab, type these two commands into the message box, one at a
time, and press **Enter** after each:

```text
/plugin marketplace add greenpolo/LangSlice
```

```text
/plugin install langslice@langslice
```

The first adds LangSlice's plugin list from GitHub. The second installs the
plugin. When Claude asks where to install it, choose **Install for you (user
scope)**, so LangSlice is there in every folder.

:::{admonition} Screenshot to come
The plugin's details with the install choices, **Install for you (user scope)**
marked.
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

## Step 4: Test the connection

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
