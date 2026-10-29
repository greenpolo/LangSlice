# Connect Codex (ChatGPT)

Codex is the coding agent inside the ChatGPT desktop app. On this page you
install the app, add LangSlice's plugin to Codex, and ask Codex to finish
setting LangSlice up. A **plugin** is a package of add-ons that an app
installs in one go.

**Before you start:** [install LangSlice](install.md), including
[step 6, which lets your AI app find LangSlice](install.md#step-6-let-your-ai-app-find-langslice).

```{note}
The pictures on this page are from the app on Linux. The steps are untested
on Windows and macOS.
```

## Step 1: Install the ChatGPT desktop app

1. Download the ChatGPT desktop app from <https://chatgpt.com/download> and
   install it.
2. Open it and sign in with your ChatGPT account.
3. Choose **Codex**, then **New chat**.

```{figure} img/codex-new-chat.png
:alt: Codex with New chat marked at the top of the left-hand column, and the message box at the bottom of the window.
:width: 80%

Codex, ready for a new chat. You type your messages in the box at the bottom.
```

## Step 2: Add LangSlice's plugin

LangSlice's plugin comes from a **marketplace**: a list of plugins that
someone publishes, here LangSlice's own, on GitHub.

1. In the narrow column of icons on the left, click the plugins icon, marked
   below. The **Plugins** page opens.
2. Click **Add**, then **Add a marketplace**.

   ```{figure} img/codex-plugins-add-menu.png
   :alt: The Plugins page with the plugins icon, the Add button and the menu item Add a marketplace marked.

   The plugins icon, the **Add** button and **Add a marketplace**.
   ```

3. In **Source**, type `greenpolo/LangSlice`. Leave the other two fields as
   they are, and click **Add marketplace**.

   ```{figure} img/codex-add-marketplace-form.png
   :alt: The Add plugin marketplace form, with the Source field marked; Git ref and Sparse paths below it are left as they are.
   :width: 70%

   The **Add plugin marketplace** form. Only **Source** needs filling in.
   ```

4. In **Search plugins**, type `LangSlice`, and install the LangSlice plugin
   from the list.

:::{admonition} Screenshot to come
LangSlice in the plugin list, with its install button.
:::

## Step 3: Ask Codex to set up LangSlice

The plugin brings LangSlice's instructions for Codex. Codex also needs
LangSlice's registration agents and settings in its own folder. A skill in
the plugin, named `setup`, puts them there.

1. Click **New chat**.
2. Send this message:

   ```text
   Set up LangSlice.
   ```

Codex follows the `setup` skill. It finds `langslice` on your computer and
copies LangSlice's two registration agents and its rules file into Codex's
folders. It then adds LangSlice's MCP server to Codex's settings, and turns
on one setting LangSlice needs. When it is done, it tells you to restart the
app.

Codex may ask permission to write files outside the project, the folder the
chat works in. Allow it. The files go into the `.codex` folder in your home
folder, where Codex keeps its own settings.

:::{admonition} Screenshot to come
The setup chat: Codex asking to write outside the project, then reporting
what it set up.
:::

<details>
<summary>What setup changed</summary>

In the `.codex` folder in your home folder:

- `agents/register.toml` is the registration agent that works only through
  LangSlice's tools, with no terminal.
- `agents/register_scripting.toml` is the registration agent that can also
  write and run its own scripts.
- `rules/langslice.rules` lets `langslice-job` commands run without asking
  you each time.
- `config.toml`, Codex's settings file, gets LangSlice's MCP server:

  ```toml
  [mcp_servers.langslice]
  command = "/full/location/of/langslice"
  args = ["mcp"]
  tool_timeout_sec = 3600
  default_tools_approval_mode = "approve"

  [features]
  mcp_tool_list_changed = true
  ```

  - `command` and `args` start LangSlice's MCP server, `langslice mcp`.
    `command` holds the full location of `langslice` that setup found.
  - `tool_timeout_sec = 3600` lets one LangSlice tool run for up to an hour.
    Codex's own limit is 60 seconds, shorter than a deformation fit.
  - `default_tools_approval_mode = "approve"` lets LangSlice's tools run
    without asking you before each call.
  - `mcp_tool_list_changed = true` lets Codex pick up the tools a
    registration needs. LangSlice shows them only once the agent has opened
    the job; without this setting the agent sees only `start_job`.

The plugin itself holds the skills `register-brain`, which the main agent
follows to prepare a registration and check it, and `abba`, for working with
ABBA.

</details>

## Step 4: Restart the app

Quit the ChatGPT desktop app completely and open it again. Codex reads its
agents and settings when it starts.

## Step 5: Test the connection

1. Start a new Codex chat.
2. Send this message:

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
