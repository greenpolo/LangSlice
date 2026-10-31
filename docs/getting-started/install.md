# Install LangSlice

You will install two things:

1. **Miniforge**, a free installer for scientific Python software. Many
   bioinformatics tools install the same way.
2. **The LangSlice environment**, a self-contained folder that holds
   LangSlice and everything it runs on. It does not change other software on
   your computer.

You need about 15 GB of free disk space and an internet connection. The
download is several gigabytes, so the install takes a while on a slow
connection.

## Step 1: Install Miniforge

Skip this step if Miniforge, Miniconda or Anaconda is already installed: any
of them provides the `conda` command used below.

### Windows

1. Download the Windows installer, `Miniforge3-Windows-x86_64.exe`, from
   <https://conda-forge.org/download/>.
2. Open it and accept the default choices: install for **Just Me**, in the
   suggested folder.
3. When it finishes, open the Start menu and look for **Miniforge Prompt**.
   This is the terminal you will use for LangSlice on Windows.

:::{admonition} Screenshot to come
The Start menu with **Miniforge Prompt** highlighted.
:::

### macOS and Linux

1. Open the **Terminal** app. On macOS it is in **Applications ‣ Utilities**.
2. Copy these two lines, paste them into Terminal and press **Enter**:

   ```bash
   curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
   bash Miniforge3-$(uname)-$(uname -m).sh
   ```

3. The installer shows a licence. Press **Enter** to read through it, then type
   `yes` to accept it.
4. Press **Enter** to install in the suggested folder.
5. When it asks whether to update your shell profile to initialize conda,
   type `yes`.
6. Close Terminal and open it again. The line where you type now starts with
   `(base)`.

:::{admonition} Screenshot to come
Terminal after the reopen, with `(base)` at the start of the line.
:::

## Step 2: Install LangSlice

In the terminal (**Miniforge Prompt** on Windows, **Terminal** on macOS and
Linux), run these three commands, one at a time:

```bash
conda create -n langslice python=3.11 -y
conda activate langslice
pip install "langslice[registration,mcp]"
```

The first makes the LangSlice environment, the second switches to it, and the
third downloads LangSlice and every package it needs. Text scrolls past for
several minutes.

## Step 3: Check that LangSlice runs

In the same terminal, run:

```bash
langslice version
```

The line where you type starts with `(langslice)`, and LangSlice prints its
version:

```text
langslice 0.1.0
```

Run `conda activate langslice` each time you open a new terminal for
LangSlice. Your AI app does not need it: when you connect the app, its setup
skill finds LangSlice and lets the app start it.

:::{admonition} Screenshot to come
Miniforge Prompt (Windows) and Terminal (macOS) after `langslice version`,
showing `(langslice)` and `langslice 0.1.0`.
:::

<details>
<summary>Under the hood: how the apps find LangSlice</summary>

Claude Desktop and Antigravity start the command `langslice mcp` by name, and
LangSlice's agents run `langslice` and `langslice-job` by name. Each looks
for them in the folders listed in the `PATH` setting. On macOS and Linux, the
apps read `PATH` from your login shell when they start; on Windows, from your
user and system environment variables. The setup skill adds the LangSlice
environment's `bin` folder (`Scripts` on Windows) to the end of `PATH`: in
your shell's startup files on macOS and Linux, and in your user environment
variables on Windows. It does not activate the environment. Codex starts
LangSlice's MCP server from its full location, which its setup writes into
Codex's settings.

</details>

## The atlas

LangSlice downloads an atlas the first time a registration uses it, into a
folder named `.brainglobe` in your home folder. The default atlas, the Allen
mouse brain atlas at 25 µm, is about 600 MB. Later registrations reuse it.

<details>
<summary>Under the hood: what the environment holds</summary>

The environment is a conda environment named `langslice` with Python 3.11.
pip installs LangSlice into it with its optional parts for registration (HDF5
reading) and MCP (the connection AI apps use). It includes the registration
engines: Elastix for in-plane fits and FireANTs, on PyTorch, for deformations.

On Linux, about 9 GB of the environment is PyTorch and its graphics-card
libraries. FireANTs runs on an NVIDIA graphics card when PyTorch finds one,
and on the processor otherwise. The PyTorch that installs on Windows and
macOS runs on the processor.

</details>

## Install from the source code instead

To install the version on GitHub, for example to try a change before it is
released:

1. Go to <https://github.com/greenpolo/LangSlice>, click the green **Code**
   button, then **Download ZIP**, and unzip it. You get a folder named
   `LangSlice-main`. Move it somewhere it can stay.
2. In a terminal, type `cd` and a space, drag the `LangSlice-main` folder onto
   the terminal window, and press **Enter**.
3. Run `conda env create -f environment.yml`. It makes the same `langslice`
   environment from the folder.

Then check it as in step 3.

## Next

[Connect your AI app](connect.md).
