# Install LangSlice: instructions for an AI agent

The user sent you here to install LangSlice on their computer and connect it
to the app you run in. LangSlice registers brain-section images to brain
atlases. It is a Python package that runs in a conda environment named
`langslice`; the app reaches it through LangSlice's plugin.

Do the steps in order and run every check. When a check fails, show the user
the error, then fix it or stop. Ask the user before you install Miniforge or
remove anything. Several steps download from the internet and write outside
your workspace: ask for approval when your sandbox refuses. Tell the user what
each long step is doing; the install downloads several gigabytes.

## 1. Look for an existing install

Run `langslice version` (find it first with `command -v langslice` on macOS and
Linux, `where langslice` on Windows). If it prints a version, LangSlice is
installed: tell the user, offer the update below, and go to step 6.

## 2. Find conda

LangSlice needs Python 3.11 exactly. The conda environment brings its own
Python, so other Pythons on the computer do not matter.

Find a conda (Miniforge, Miniconda, Anaconda or Mambaforge). An app's shell
often has not loaded conda, so when `conda --version` fails, look further:

- macOS and Linux: ask the user's own shell, `"$SHELL" -ilc 'echo $CONDA_EXE'`;
  else look for `bin/conda` under `~/miniforge3`, `~/mambaforge`,
  `~/miniconda3`, `~/anaconda3`, `/opt/conda`,
  `/opt/homebrew/Caskroom/miniforge/base` and
  `/usr/local/Caskroom/miniforge/base`.
- Windows: `Scripts\conda.exe` under `miniforge3`, `mambaforge`, `miniconda3`
  and `anaconda3`, in `%USERPROFILE%`, `%LOCALAPPDATA%` and `%ProgramData%`.

Use its full path wherever `conda` appears below.

If there is none, ask the user where their conda is, if they have one, or
whether you may install Miniforge, the free conda installer of the
conda-forge community, into their home folder. It needs no administrator
rights. With their yes:

- macOS and Linux:

  ```bash
  curl -fsSLo Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
  bash Miniforge3.sh -b -p "$HOME/miniforge3"
  ```

  conda is then `~/miniforge3/bin/conda`.
- Windows: download
  <https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe>
  and run it from `cmd`:

  ```bat
  start /wait "" Miniforge3-Windows-x86_64.exe /InstallationType=JustMe /RegisterPython=0 /S /D=%UserProfile%\miniforge3
  ```

  conda is then `%USERPROFILE%\miniforge3\Scripts\conda.exe`. Miniforge fails
  in a folder whose path holds spaces or non-ASCII letters: if the home
  folder's does, ask the user for another folder.

Delete the installer afterwards. Do not run `conda init`: LangSlice does not
need it, and the setup skill of step 6 adds only LangSlice's own folder to
`PATH`.

## 3. Make the environment

If `conda env list` already shows `langslice`, ask the user whether to update
it (below) or to remove it (`conda env remove -n langslice -y`) and make it
again.

```bash
conda create -n langslice python=3.11 -y
conda run -n langslice python -c "import sys; print(sys.executable)"
```

The second command prints the environment's Python, `PY` below.

## 4. Install LangSlice

On Windows with an NVIDIA GPU (`nvidia-smi` prints a table), first install
PyTorch's CUDA build, because the PyTorch that pip installs on Windows runs on
the processor only. Take the highest `cuXXX` that
<https://download.pytorch.org/whl/torch/> offers for this Python and that is
not above the CUDA version `nvidia-smi` shows:

```bash
PY -m pip install torch --index-url https://download.pytorch.org/whl/cuXXX
```

Then, on every system:

```bash
PY -m pip install "langslice[registration,mcp]"
```

It downloads several gigabytes, PyTorch among them, and can take half an
hour: run it in the background or with your longest time limit, and check
that it finished.

If pip finds no `langslice`, or the user wants the version on GitHub, install
from the source instead. Put it in a folder the user keeps, such as
`~/LangSlice`: `git clone https://github.com/greenpolo/LangSlice.git`, or
download and unzip
<https://github.com/greenpolo/LangSlice/archive/refs/heads/main.zip>. Then:

```bash
PY -m pip install "FOLDER[registration,mcp]"
```

## 5. Check

`langslice` and `langslice-job` sit beside `PY` (on Windows, in the `Scripts`
folder beside it). Run both by their full path:

```bash
langslice version
langslice-job ops
```

The first prints `langslice 0.2.0` or a later version. The second prints one
JSON object with `"ok": true`.

## 6. Connect the app

Install LangSlice's plugin in the app you run in:

| App | Commands |
|---|---|
| Codex (the ChatGPT desktop app, or the `codex` command) | `codex plugin marketplace add greenpolo/LangSlice`, then `codex plugin add langslice@langslice` |
| Claude Code (the Claude desktop app's Code tab, or the `claude` command) | `claude plugin marketplace add greenpolo/LangSlice`, then `claude plugin install langslice@langslice` |
| Antigravity | `agy plugin install https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity` |

If the command is not found, ask the user to install the plugin from the
app's menus, as the Installation page shows for their app
(<https://langslice.readthedocs.io/en/latest/getting-started/install.html#connect-your-ai-app>),
and wait until they say it is done. LangSlice's plugins are for these three
apps: in another app, tell the user so and stop here.

Then set LangSlice up yourself, in this conversation. The app loads a new
plugin only when it starts, so its skills are not yours yet: read the
plugin's setup skill, `skills/setup/SKILL.md` in the installed plugin, and do
its steps. The plugin is under `~/.codex/plugins/cache/langslice/`,
`~/.claude/plugins/cache/langslice/` or `~/.gemini/config/plugins/langslice/`
(on Windows, in `%USERPROFILE%`). The same skill is on GitHub at
`https://raw.githubusercontent.com/greenpolo/LangSlice/main/connectors/APP/skills/setup/SKILL.md`,
with `codex`, `claude-code` or `antigravity` for `APP`.

The skill ends by telling the user to quit and reopen the app and how to test
the connection. Tell them that too, with what you installed and where.

## Update

```bash
PY -m pip install -U "langslice[registration,mcp]"
```

From the source: update the folder (`git pull`, or a new download), then run
step 4's source install again. Then update the plugin and restart the app:

- Codex: `codex plugin marketplace upgrade`, then the setup skill again,
  which copies the plugin's agents and rules.
- Claude Code: `claude plugin marketplace update langslice`, then
  `claude plugin update langslice@langslice`.
- Antigravity: the `agy plugin install` command of step 6 again.
