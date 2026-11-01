# Installation

LangSlice runs on Windows, macOS and Linux with Python 3.11. It takes about
15 GB of disk space, with the atlas it downloads on first use. An NVIDIA GPU
is optional: on Linux it makes deformable fits faster.

[Let your AI app install it](#for-agents), or install it [yourself](#for-humans).

## For agents

Send this to your agent of choice:

```{code-block} text
:class: ls-copy-box

I want to install LangSlice. Read this and follow the steps: https://raw.githubusercontent.com/greenpolo/LangSlice/main/docs/agents/install.md
```

## For humans

### 1. Install LangSlice

With conda ([Miniforge](https://conda-forge.org/download/); on Windows, in
Miniforge Prompt):

```bash
conda create -n langslice python=3.11
conda activate langslice
pip install "langslice[registration,mcp]"
```

Check it with `langslice version`. To update later, run the `pip install` line
again with `-U`.

For the version on GitHub, install from the source instead:

```bash
git clone https://github.com/greenpolo/LangSlice.git
cd LangSlice
conda env create -f environment.yml
```

### 2. Connect your AI app

::::{tab-set}

:::{tab-item} Codex
Codex is the coding agent in the ChatGPT desktop app. The free plan works.

1. Install the ChatGPT desktop app from <https://chatgpt.com/download>, sign
   in, and choose **Codex**, then **New chat**.

   ```{image} img/codex-new-chat.png
   :alt: Codex with New chat marked at the top of the left-hand column, and the message box at the bottom.
   :class: ls-shot
   ```

2. Click the plugins icon in the column of icons on the left, then **Add** and
   **Add a marketplace**.

   ```{image} img/codex-plugins-add-menu.png
   :alt: The Plugins page with the plugins icon, the Add button and the menu item Add a marketplace marked.
   :class: ls-shot
   ```

3. In **Source**, type `greenpolo/LangSlice`, leave the other fields as they
   are, and click **Add marketplace**.

   ```{image} img/codex-add-marketplace-form.png
   :alt: The Add plugin marketplace form with the Source field marked; Git ref and Sparse paths below it are left as they are.
   :class: ls-shot
   ```

4. Search the plugins for `LangSlice` and install it.
5. In a new chat, send **Set up LangSlice.** and let Codex write outside the
   project when it asks.
6. Quit the app completely and open it again.

Or, in a terminal, instead of steps 2 to 4:

```bash
codex plugin marketplace add greenpolo/LangSlice
codex plugin add langslice@langslice
```
:::

:::{tab-item} Claude Code
Claude Code is the **Code** tab of the Claude desktop app. It needs a paid
Claude plan. On Windows, install [Git for Windows](https://git-scm.com/downloads/win)
first.

1. Install the Claude desktop app from <https://claude.com/download> and sign
   in. Click the **Code** tab (`</>`), then, above the message box, **Local**
   and **Select folder…**, and pick a folder to work in.

   ```{image} img/claude-code-tab.png
   :alt: The Claude app on the Code tab, with the Code button at the top left and the Local and Select folder buttons above the message box marked.
   :class: ls-shot
   ```

2. Click **Customize**, then **Plugins**, **Add** and **Add marketplace**.

   ```{image} img/claude-plugins-add-menu.png
   :alt: The Customize page with Customize, the Plugins tab, the Add button and the menu item Add marketplace marked.
   :class: ls-shot
   ```

3. Click **Add from a repository**.

   ```{image} img/claude-add-marketplace.png
   :alt: The Add marketplace window with two choices; Add from a repository is marked.
   :class: ls-shot
   ```

4. In **URL**, type `greenpolo/LangSlice` and click **Sync**.

   ```{image} img/claude-marketplace-url.png
   :alt: The Add marketplace window with a warning to add only trusted marketplaces, and the URL field marked.
   :class: ls-shot
   ```

5. Find **LangSlice** in the list of plugins and install it.
6. In a new session, send **Set up LangSlice.** and allow each change it asks
   for. Until setup is done, Claude may say that LangSlice's server failed to
   start.
7. Quit the app completely and open it again.

Or, in a terminal, instead of steps 2 to 5:

```bash
claude plugin marketplace add greenpolo/LangSlice
claude plugin install langslice@langslice
```
:::

:::{tab-item} Antigravity
Antigravity is Google's app for AI agents, with Gemini models. The free tier
works.

1. Install Antigravity from <https://antigravity.google> and sign in with your
   Google account.
2. Click **New Conversation**, type `/plugin` in the message box, and pick
   **plugin** from the list.

   ```{image} img/antigravity-plugin-skill.png
   :alt: The message box with /plugin typed and the list of matching skills above it; the plugin skill is marked.
   :class: ls-shot
   ```

3. After **plugin**, type this message and send it:

   ```text
   Please install this plugin: https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
   ```

   ```{image} img/antigravity-install-message.png
   :alt: The message box holding the plugin skill followed by the install request with LangSlice's GitHub address.
   :class: ls-shot
   ```

4. When Antigravity says the plugin is installed, quit it completely and open
   it again.
5. In a new conversation, send **Set up LangSlice.** and allow each change it
   asks for.
6. Quit Antigravity completely and open it again.

Or, in a terminal, instead of steps 2 and 3:

```bash
agy plugin install https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
```
:::

::::

### 3. Check the connection

In a new conversation, send:

```text
Call LangSlice's start_job tool with no arguments and tell me what it says.
```

LangSlice answers that no job is named yet and asks for an image folder. If
the app has no LangSlice tools instead, see
[Troubleshooting](../troubleshooting.md#the-app-cannot-start-langslice).
