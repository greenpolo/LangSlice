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

## Step 2: Download LangSlice

1. Go to <https://github.com/greenpolo/LangSlice>.
2. Click the green **Code** button, then **Download ZIP**.
3. Unzip the file. You get a folder named `LangSlice-main`.
4. Move that folder somewhere it can stay, such as your **Documents**
   folder. Later steps copy files from it.

:::{admonition} Screenshot to come
The GitHub page with the **Code** menu open and **Download ZIP** marked.
:::

## Step 3: Open a terminal in the LangSlice folder

1. Open a terminal, the window where you type commands: **Miniforge
   Prompt** on Windows, **Terminal** on macOS and Linux.
2. Type `cd` followed by a space. Do not press **Enter** yet.
3. Drag the `LangSlice-main` folder from your file browser onto the terminal
   window. Its location appears after `cd `.
4. Press **Enter**.

To check that you are in the right folder, type `dir` (Windows) or `ls`
(macOS, Linux) and press **Enter**. The list includes `environment.yml`.

## Step 4: Create the LangSlice environment

In the same terminal, run:

```bash
conda env create -f environment.yml
```

Conda downloads Python and every package LangSlice needs, then installs
LangSlice. Text scrolls past for several minutes. When it is done, the last
lines are:

```text
# To activate this environment, use
#
#     $ conda activate langslice
```

## Step 5: Check that LangSlice runs

Run these two commands:

```bash
conda activate langslice
langslice version
```

The line where you type now starts with `(langslice)`, and LangSlice prints
its version:

```text
langslice 0.1.0
```

Run `conda activate langslice` each time you open a new terminal for
LangSlice. Your AI app does not need it; the next page shows how the app
finds LangSlice.

:::{admonition} Screenshot to come
Miniforge Prompt (Windows) and Terminal (macOS) after `langslice version`,
showing `(langslice)` and `langslice 0.1.0`.
:::

## Step 6: Note where LangSlice is installed

Connecting an AI app needs the full location of the `langslice` program. With
the environment active, run:

On Windows:

```bash
where langslice
```

On macOS and Linux:

```bash
which langslice
```

It prints one line, for example
`C:\Users\you\miniforge3\envs\langslice\Scripts\langslice.exe` on Windows or
`/Users/you/miniforge3/envs/langslice/bin/langslice` on macOS. Keep this
terminal open, or copy the line somewhere: you need it on the next page.

## The atlas

LangSlice downloads an atlas the first time a registration uses it, into a
folder named `.brainglobe` in your home folder. The default atlas, the Allen
mouse brain atlas at 25 µm, is about 600 MB. Later registrations reuse it.

<details>
<summary>Under the hood: what the environment holds</summary>

`environment.yml` creates a conda environment named `langslice` with
Python 3.11. It then installs LangSlice from the downloaded folder with pip.
The install includes LangSlice's optional parts for registration (HDF5
reading), MCP (the connection AI apps use) and ABBA (starting ABBA from
Python). The
environment includes the registration engines: Elastix for in-plane fits and
FireANTs, on PyTorch, for deformations.

On Linux, about 9 GB of the environment is PyTorch and its graphics-card
libraries. FireANTs runs on an NVIDIA graphics card when PyTorch finds one,
and on the processor otherwise. The PyTorch that installs on Windows and
macOS runs on the processor.

</details>

## Next

[Connect your AI app](connect.md).
