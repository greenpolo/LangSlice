<p align="center">
  <img alt="LangSlice" src="assets/logo/banner_black.svg" width="780">
</p>

<p align="center">
  <a href="https://github.com/greenpolo/LangSlice/blob/main/LICENSE"><img alt="License: BSD-3-Clause" src="https://img.shields.io/badge/License-BSD%203--Clause-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
  <a href="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/greenpolo/LangSlice/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://langslice.readthedocs.io"><img alt="Documentation Status" src="https://readthedocs.org/projects/langslice/badge/?version=latest"></a>
</p>

<p align="center">
  <em>Register histological brain sections to BrainGlobe atlases using vision-language &amp; image-generation models.</em>
</p>

<p align="center">
  <img alt="Three histology slices registered to the Allen mouse atlas" src="assets/promo_registration_combined.png" width="780">
</p>

LangSlice puts a stack of histology sections on a [BrainGlobe](https://brainglobe.info)
atlas. A vision-language model agent orders the sections, places each at its
atlas position (with cutting angles) and aligns it in-plane against the atlas at
true scale. Optionally an image model corrects the atlas borders onto the
tissue and a deformable fit turns them into a per-section deformation. Results
are per-section coordinate and label maps, `registration.json`, and
QuickNII / VisuAlign JSON. LangSlice complements registration software such as
[ABBA](https://abba-documentation.readthedocs.io) and QUINT; it has no manual
registration tools. Coronal sections of the Allen mouse atlas are the most
tested case.

## Install

Download and extract the source, open a terminal in the folder with
`environment.yml`, then:

```bash
conda env create -f environment.yml
conda activate langslice
langslice login        # ChatGPT sign-in; or set OPENAI_API_KEY / GEMINI_API_KEY (see .env.example)
```

On Windows, run the same `conda env create` and `conda activate` commands in
Miniforge Prompt or PowerShell with conda initialized. Python 3.11 on Windows
is tested in CI. Atlases are downloaded by BrainGlobe on first use into the
user's `.brainglobe` folder. Python 3.10 or newer is required; the environment
includes the ANTs deformable engine, the Claude Desktop server and the ABBA
launcher.

## Ways to use it

- **ABBA (Fiji)**: the connector adds **Register > LangSlice Registration...** to an existing
  ABBA 0.24 and applies every change as an undoable ABBA step. Setup and sign-in
  are inside ABBA: [install and use](docs/abba_installation.md),
  [connector](connectors/fiji/README.md). `langslice abba` starts ABBA from
  Python with the connector, an agent viewer and an agent log.
- **Claude Desktop**: `langslice mcp` serves the tools to Claude's own app, so the
  model runs on your Claude subscription: [setup](connectors/claude-desktop/README.md).
- **Claude Code and Codex**: skills and one-brain registration agents over the agent
  CLI: [Claude Code](connectors/claude-code/README.md), [Codex](connectors/codex/README.md);
  the CLI itself, `langslice job FOLDER VERB`, is in [docs/agent_cli.md](docs/agent_cli.md).
- **Command line agent**: `langslice linear run sections/` runs the built-in agent
  ([docs/cli.md](docs/cli.md)); add `--tasks nonlinear` for the border correction
  and deformation on a saved linear placement.
- **Python library**: `langslice.open_job`, `create_job`, `register_section`,
  `register_job` for a scripted pipeline: [docs/library.md](docs/library.md).

```python
import langslice
job = langslice.open_job("sections/")
job.status()
```

Output files: [docs/file_formats.md](docs/file_formats.md). How the pieces fit:
[docs/architecture_overview.md](docs/architecture_overview.md). Full documentation:
[langslice.readthedocs.io](https://langslice.readthedocs.io).

## Related repositories

- **LangSlice-Training**: training infrastructure for local models.
- **SliceBench**: a position-estimation benchmark.

Both depend on LangSlice; neither is needed to run it.

## Citation and license

Cite the project metadata in [`CITATION.cff`](./CITATION.cff) (GitHub's "Cite this
repository" button). BSD-3-Clause; see [`LICENSE`](./LICENSE).
