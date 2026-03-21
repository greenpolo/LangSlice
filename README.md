# LangSlice

LangSlice is a PySide6 desktop application for registering histology slice images to BrainGlobe atlases.
The current implementation uses Gemini for AP estimation and landmark correspondence finding, derives affine and TPS outputs in local Python code, and exports QUINT/ABBA-compatible JSON.

## What The App Does Today

1. Load an image file (`.png`, `.jpg`, `.jpeg`, `.tif`, `.tiff`).
2. Normalize the image to 8-bit RGB and try to detect pixel size from TIFF or OME metadata.
3. Let the user choose a BrainGlobe atlas, Gemini model, landmark target count, and registration thinking budget.
4. Let the user adjust the image that will be shown to the model: channel toggles, exposure, brightness, contrast, and atlas-border visibility.
5. Run either:
   - `Run Agent`: AP estimation plus registration, or
   - `Run Registration at Manual Position`: registration only, using the AP slider value.
6. Show the result in single, split, or overlay view.
7. Export a single-slice QUINT/ABBA-compatible JSON file.

## Setup

1. Create the conda environment:
   `conda env create -f environment.yml`
2. Activate it:
   `conda activate langslice`
3. Install the package in editable mode:
   `pip install -e .`
4. Configure authentication in `.env` using `.env.example`.
5. Launch the GUI:
   `langslice gui`

## Authentication Backends

`langslice/ai/config.py` currently supports three backend modes:

- `ai_studio`
- `vertex_api_key`
- `vertex_adc`

The GUI settings dialog writes the selected backend and related credentials to the repo-root `.env` file.

## Optional Gemini Flags

The AP estimator has opt-in rollout flags in `langslice/ai/config.py`:

- `LANGSLICE_GENAI_COUNT_TOKENS=true`
- `LANGSLICE_GENAI_AP_USE_FILE_API=true`
- `LANGSLICE_GENAI_AP_USE_CONTEXT_CACHE=true`
- `LANGSLICE_GENAI_AP_USE_INTERACTIONS=true`
- `LANGSLICE_GENAI_AP_CACHE_TTL=3600s`
- `LANGSLICE_GENAI_FILE_POLL_TIMEOUT_S=10.0`

The offline Batch API helper in `langslice.ai.batch_eval` is currently guarded to `vertex_adc` mode.

## CLI

- `langslice gui` - launch the desktop application
- `langslice version` - print the package version

## Debug Traces

Set `LANGSLICE_VLM_DEBUG_DIR` to save per-run artifacts.

Current behavior:

- AP estimation writes the prepared target image, `reasoning.txt`, and `telemetry.json`.
- Registration writes a `registration/` subdirectory with `slice.png`, `atlas.png`, landmark overlays, and `registration.json`.
- The GUI can move a completed run into `success/` or `failure/` and write `classification.json` with optional ground-truth AP, AP error, and notes.

## Current Constraints

- Atlas coordinate helpers and export currently assume coronal layout with AP/DV/ML on axes `0/1/2`.
- The registration runtime computes both affine and TPS outputs, but export uses the affine result only.
- `pixel_size_um` is tracked through the GUI and image-prep pipeline, but the registration runtime currently accepts it without using it.
- The overlay viewer accepts pixel-size input for API compatibility, but does not apply physical calibration from it.

## Repository Layout

- `langslice/` - installable package source
- `tests/` - pytest suite
- `docs/` - maintained project docs
- `references/` - external reference code
- `archive/` - preserved legacy material

See `docs/index.md` for the maintained docs set and `REPO_MAP.md` for the short navigation map.
