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
  <em>Agent-driven registration of histological brain sections to BrainGlobe atlases.</em>
</p>

<p align="center">
  <img alt="Three histology slices registered to the Allen mouse atlas" src="assets/promo_registration_combined.png" width="780">
</p>

LangSlice gives a vision-language agent tools to register a stack of histology
sections to a [BrainGlobe](https://brainglobe.info) atlas. The agent inspects
the anatomy, sets positions and cutting angles, aligns sections in-plane,
and optionally fits ANTs deformations. It chooses regions and settings and
can use an image model to trace borders where useful.

Source images stay unchanged. Work is checkpointed and undoable, with
coordinate and label maps, `registration.json`, and QuickNII / VisuAlign
exports. Coronal sections of the Allen mouse atlas are the most tested case.

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

## Use LangSlice

- **[Agents, CLI and MCP](docs/agents.md):** Claude Code or Codex through
  `langslice-job`, Claude Desktop through MCP, or the built-in agent with
  `langslice linear run sections/`.
- **[ABBA](docs/abba.md):** register selected sections through the Fiji
  connector and review changes in ABBA.
- **[Python](docs/library.md):** coding agents can open a job and write their
  own workflows using the same tools.

[Registration tools](docs/registration.md) ·
[File formats](docs/file_formats.md) ·
[Architecture](docs/architecture.md) ·
[Full documentation](https://langslice.readthedocs.io)

## Citation and license

Cite the project metadata in [`CITATION.cff`](./CITATION.cff) (GitHub's "Cite this
repository" button). BSD-3-Clause; see [`LICENSE`](./LICENSE).
