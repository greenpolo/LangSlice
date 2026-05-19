# LangSlice — Install (v0.1 developer preview)

The v0.1 Tauri desktop app is a **developer preview**: it wraps the
Python `langslice-harness` CLI and needs Python 3.10+ available on
`PATH`. The estimation, registration, and quick-affine features all
shell out to Python at runtime — the installer alone is not enough.

For the no-prereq option, use the browser version at
[greenpolo.github.io/LangSlice](https://greenpolo.github.io/LangSlice).
The two are deliberately different products: the web demo is
intentionally limited, the Tauri app is the feature-rich path.

## Prereqs (one-time setup)

### 1. Python 3.10+

Any standard Windows install works. Confirm:

```powershell
python --version
```

If the command isn't found, install from
[python.org/downloads](https://www.python.org/downloads/windows/) and
make sure the installer's "Add python.exe to PATH" option is checked.

### 2. `langslice-harness` package

Clone the repo and install it in editable mode:

```powershell
git clone https://github.com/greenpolo/LangSlice.git
cd LangSlice
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

This pulls in the core deps (`brainglobe-atlasapi`, `numpy`, `scipy`,
`pillow`, `tifffile`, `itk-elastix`, `google-genai`, …).

### 3. A BrainGlobe atlas

The Tauri app's first launch lets you pick + download an atlas through
its UI, but you can pre-warm one from the command line:

```powershell
python -c "from brainglobe_atlasapi import BrainGlobeAtlas; BrainGlobeAtlas('allen_mouse_25um')"
```

This downloads to `~/.brainglobe/` and is shared across all tools.

### 4. (Optional) Gemini API key

For position estimation via cloud models, get a key from
[aistudio.google.com](https://aistudio.google.com/app/apikey) and paste
it into the app's **API Keys** tab.

## Install the Tauri app

1. Download `LangSlice_0.1.0_x64-setup.exe` (NSIS, ~40 MB) from the
   [Releases page](https://github.com/greenpolo/LangSlice/releases).
2. Double-click to install. Windows SmartScreen may warn that the
   installer is unsigned — click **More info → Run anyway**.
3. The app installs to `Program Files\LangSlice\`. Uninstall via the
   standard **Add or Remove Programs**.
4. Launch from the Start menu (`LangSlice`).

The `.msi` variant (`LangSlice_0.1.0_x64_en-US.msi`) is also available
for IT-managed deployments.

## Verify it works

On launch, the app's status row should show:

- **Atlas**: `allen_mouse_25um` loaded (or pick another from the Atlas
  dropdown if you downloaded one).
- **API Keys** tab: at minimum a Gemini API key for cloud estimation,
  or set up a local model via the **Local Models** tab.

Try the **Load Demo Brain** button and run an estimate on one slice. If
the estimate completes, the Python pipeline is wired up correctly.

## Troubleshooting

- **"Failed to spawn python"** — `python` isn't on PATH for the user
  account that launched the app. Open a fresh PowerShell, run `python
  --version`. If it works there but not from the app, log out and back
  in (PATH changes don't always propagate to running sessions).
- **"No module named langslice_harness"** — the `python` on PATH isn't
  the one with `langslice-harness` installed. Either install into the
  global Python, or update PATH so your venv's `python.exe` wins.
- **Atlas operations hang on first run** — BrainGlobe is downloading
  the atlas (~500 MB for `allen_mouse_25um`). Check
  `~/.brainglobe/` for progress.

## What ships in the installer

- The Tauri shell + UI (~30 MB).

The `litert-lm` sidecar for fully-offline Gemma 4 inference is **not**
bundled in v0.1 — it's platform-specific and large. Download it
separately from the [LiteRT-LM releases](https://github.com/google-ai-edge/LiteRT-LM/releases/latest)
and run `litert-lm serve --port 8765` before launching LangSlice; the
**Local Models** tab will auto-detect it on the standard port.

## Roadmap

A future v0.2 will bundle Python + `langslice-harness` so the installer
is fully self-contained and no manual setup is needed. Track progress
on the [v0.2 milestone](https://github.com/greenpolo/LangSlice/milestones).
