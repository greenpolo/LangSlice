<p align="center">
  <img alt="LangSlice" src="assets/LangSlice_dark.png">
</p>

<p align="center">
  <a href="https://github.com/greenpolo/LangSlice/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://langslice.readthedocs.io"><img alt="Documentation Status" src="https://readthedocs.org/projects/langslice/badge/?version=latest"></a>
  <a href="https://huggingface.co/greenpolo/langslice-gemma-4-E4B"><img alt="Hugging Face Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-langslice--gemma--4--E4B-yellow"></a>
</p>

<p align="center">
  <em>Register histological brain sections to BrainGlobe atlases.<br>
  A VLM estimates position, an image-gen model lays down atlas colors, Elastix warps the rest.</em>
</p>

<p align="center">
  <img alt="Three histology slices registered to the Allen mouse atlas" src="assets/promo_registration_combined.png" width="780">
</p>

## How it works

<p align="center">
  <img alt="Agent loop: inspect, explore atlas candidates, submit AP" src="assets/agent_loop.png" width="900">
</p>

A VLM agent (default [`langslice-gemma-4-E4B`](https://huggingface.co/greenpolo/langslice-gemma-4-E4B))
inspects the slice, explores candidate atlas planes through tool calls, and
submits an AP coordinate. Image generation then produces an atlas-colored
target from the histology, and itk-elastix recovers a dense B-spline
deformation. Results export to VisuAlign-compatible JSON for QUINT / ABBA.

<p align="center">
  <img alt="LangSlice registration pipeline: histology slice to atlas-colored target, dense warp, and overlay" src="assets/registration_pipeline_square.png" width="780">
</p>

## Quick start

```bash
conda env create -f environment.yml
conda activate langslice
pip install -e .
cp .env.example .env  # add AI Studio / Vertex / OpenAI keys
```

```bash
# Position estimation
langslice estimate slice.png

# End-to-end registration at a known atlas position
langslice register slice.png --position 3.9
```

Full CLI: `langslice --help`. Pipeline detail: [`docs/index.md`](./docs/index.md).

## Model Hub Transition

LangSlice is migrating training/data helpers into shared packages under `models/`:

- `models/langslice-traces/langslice_traces`
- `models/synthdata/synthdata`
- `models/training-core/langslice_training`
- `models/data/langslice_data`

Training entrypoints are exposed as small launchers:
`langslice-single-turn-rl`, `langslice-isft`, and `langslice-sft-train`.

Public model-card metadata for the released variant is in
`models/langslice-gemma-4/variants/langslice-gemma-4-e4b/README.md`.

## Links

- [**Documentation**](https://langslice.readthedocs.io) — full pipeline + harness internals
- [**langslice-gemma-4-E4B**](https://huggingface.co/greenpolo/langslice-gemma-4-E4B) — the v1.0 fine-tune
- [**SliceBench**](./slicebench) — self-contained position-estimation benchmark
- [`tauri-gui/`](./tauri-gui) — desktop app

## Citation

If you use LangSlice in your work, please cite the project metadata in
[`CITATION.cff`](./CITATION.cff). GitHub renders a "Cite this repository"
button in the right sidebar from that file.

## License

BSD-3-Clause. See [`LICENSE`](./LICENSE).
