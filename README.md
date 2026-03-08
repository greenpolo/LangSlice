# LangSlice

LangSlice is a Python desktop app for VLM-assisted histological brain slice registration against BrainGlobe atlases.
It uses Gemini to estimate anterior-posterior position, then runs matrix-based in-plane affine registration with ANTsPyX as the primary backend and Gemini as a fallback, and finally exports QUINT/ABBA-compatible JSON.

## Current Workflow

1. Load a histology image in the GUI.
2. Choose a BrainGlobe atlas.
3. Run AP estimation.
4. Run affine estimation.
5. Review the result in single, split, or overlay view.
6. Export QUINT/ABBA-compatible JSON.

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

- Atlas matching in the GUI and affine prompt is visually scaled, not full ABBA-style physical-space calibration.
- Pixel size is collected in the GUI but is not yet used to physically scale atlas overlays.
- Affine verification in the GUI is currently visual; there are no built-in landmark or overlap-error metrics yet.
- Export targets QUINT/ABBA-compatible JSON only; LangSlice does not depend on ABBA, Fiji, or a JVM at runtime.

## Repository Layout

- `langslice/` - main package source
- `tests/` - script-style verification checks
- `docs/` - current architecture and workflow docs
- `references/` - external reference code kept for research
- `archive/` - legacy prototype and preserved artifacts

See `docs/index.md` for the maintained docs set and `REPO_MAP.md` for a concise navigation map.
