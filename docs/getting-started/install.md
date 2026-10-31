# Installation

LangSlice runs on Windows, macOS and Linux with Python 3.11 and takes about
15 GB of disk space. An NVIDIA GPU is optional; it makes deformable fits
faster.

## 1. Install LangSlice

With conda ([Miniforge](https://conda-forge.org/download/); on Windows, in
Miniforge Prompt):

```bash
conda create -n langslice python=3.11
conda activate langslice
pip install "langslice[registration,mcp]"
```

Check it with `langslice version`. To update later, run the `pip install` line
again with `-U`.

## 2. Connect your AI app

Install LangSlice's plugin in the AI app you use:

- **Codex** (ChatGPT desktop app; the free plan works): **Plugins ‣ Add ‣ Add a
  marketplace**, source `greenpolo/LangSlice`, then install **LangSlice**.
- **Claude Code** (the Claude desktop app's **Code** tab; a paid plan):
  **Customize ‣ Plugins ‣ Add ‣ Add marketplace ‣ Add from a repository**,
  `greenpolo/LangSlice`, then install **LangSlice**. On Windows this needs
  [Git for Windows](https://git-scm.com/downloads/win).
- **Antigravity** (Gemini; the free tier works): type `/plugin`, pick
  **plugin**, send `Please install this plugin:` and the address below, then
  restart Antigravity.

Or from a terminal:

```bash
# Codex
codex plugin marketplace add greenpolo/LangSlice
codex plugin add langslice@langslice
# Claude Code
claude plugin marketplace add greenpolo/LangSlice
claude plugin install langslice@langslice
# Antigravity
agy plugin install https://github.com/greenpolo/LangSlice/tree/main/connectors/antigravity
```

Then send **Set up LangSlice.** in a new conversation, allow the changes it
asks for, and restart the app. Setup finds LangSlice, adds it to your `PATH`
and lets the app run LangSlice's tools.

To check the connection, ask the app to call LangSlice's `start_job` tool with
no arguments. LangSlice answers that no job is named yet.

## From source

```bash
git clone https://github.com/greenpolo/LangSlice.git
cd LangSlice
conda env create -f environment.yml
```

The atlas downloads the first time a registration uses it, into `.brainglobe`
in your home folder: about 600 MB for the Allen mouse atlas at 25 µm.
