# LangSlice

LangSlice is a Python desktop app for VLM-assisted histological brain slice registration against BrainGlobe atlases.
It uses Gemini to estimate anterior-posterior position, runs matrix-first in-plane affine registration, and exports QUINT/ABBA-compatible JSON.

## Current Workflow

1. Load a histology image in the GUI.
2. Pixel size is auto-detected from TIFF metadata when available; otherwise the current manual value is used.
3. Choose a BrainGlobe atlas.
4. Run AP estimation.
5. Run affine estimation from the full agent pipeline or use standalone `Run ANTs Affine` for isolated affine debugging.
6. Review the result in single, split, or overlay view.
7. Export QUINT/ABBA-compatible JSON.

## Setup

1. Create the conda environment:
   `conda env create -f environment.yml`
2. Activate it:
   `conda activate langslice`
3. Install in editable mode:
   `pip install -e .`
4. Configure authentication in `.env` using `.env.example`.
5. Launch the GUI:
   `langslice gui`

## Authentication Modes

LangSlice supports the backends implemented in `langslice/vlm/config.py`:

- `ai_studio`
- `vertex_api_key`
- `vertex_adc`

Relevant environment variables are documented in `.env.example`.

## CLI

- `langslice gui` - launch the desktop application
- `langslice version` - print the package version

## Debug Traces

Set `LANGSLICE_VLM_DEBUG_DIR` to save per-run AP estimation artifacts, including the target image, atlas slices fetched by the model, and a reasoning log.
The GUI exposes `Mark Success` and `Mark Failure` buttons to help sort those traces after a run completes.
When classifying a run, the GUI can also save optional evaluation metadata such as the ground-truth AP position, computed AP error, and freeform notes into `classification.json` inside the trace folder.

## Current Limitations

- Overlay preview now follows the same coronal frame geometry contract as export, but full physical-space calibration from per-image pixel size is not complete yet.
- Pixel-size metadata auto-detection currently focuses on TIFF metadata; non-TIFF inputs usually require manual pixel size.
- Affine verification in the GUI is currently visual; there are no built-in landmark or overlap-error metrics yet.
- Export targets QUINT/ABBA-compatible JSON only; LangSlice does not depend on ABBA, Fiji, or a JVM at runtime.

## Repository Layout

- `langslice/` - main package source
- `tests/` - script-style verification checks
- `docs/` - current architecture and workflow docs
- `references/` - external reference code kept for research
- `archive/` - legacy prototype and preserved artifacts

See `docs/index.md` for the maintained docs set and `REPO_MAP.md` for a concise navigation map.
