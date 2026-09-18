# Worker distribution

The Fiji connector starts the separately installed LangSlice Python worker.
The worker does not require Java, Maven, or `abba-python`. The optional
`langslice[abba]` extra remains available for the older Python-starts-ABBA route.

## Install from source today

Download and extract the repository source archive, open a terminal **in its
root folder**, and run:

```bash
conda env create -f environment.yml
```

The environment installs Python 3.11 and then installs LangSlice and all its
Python dependencies from `pyproject.toml`. It is a regular installation, not an
editable development checkout. Authentication is performed from Fiji's setup
window after selecting this environment; command-line login is also available.

`environment.yml` is a source installation recipe, not a conda package. It uses
pip inside an isolated conda environment because the full native dependency
chain is not currently available from conda-forge.

## Conda package publication blocker

As checked against the public conda-forge Anaconda API on 2026-09-16,
`google-adk`, `google-genai`, `litellm`, `openai`, `brainglobe-atlasapi`, and
`brainglobe-space` are published there. **`itk-elastix` is not**: the package
endpoint returns HTTP 404. This Python extension is a required LangSlice
dependency and is different from the standalone Elastix executable.

Before advertising `conda install -c conda-forge langslice`, package
`itk-elastix` and any missing native dependencies, verify the full dependency
solve on supported platforms, and submit a LangSlice recipe through
[conda-forge's staging process](https://conda-forge.org/docs/maintainer/adding_pkgs/).
No conda package or update site is published by the source changes in this
branch. We deliberately do not supply a recipe that claims a working solve
against a nonexistent dependency or installs hidden pip packages in a conda
post-link script.

Registry checks:

- [itk-elastix package endpoint](https://api.anaconda.org/package/conda-forge/itk-elastix)
- [google-adk package endpoint](https://api.anaconda.org/package/conda-forge/google-adk)
- [brainglobe-atlasapi package endpoint](https://api.anaconda.org/package/conda-forge/brainglobe-atlasapi)

## Build and verify the Python wheel

```bash
python -m pip install build
python -m build --wheel
python packaging/check_wheel.py dist/langslice-0.1.0-py3-none-any.whl
python -m pip install dist/langslice-0.1.0-py3-none-any.whl
langslice version
```

The package workflow builds and audits the wheel, installs it in an isolated
Python environment on Linux, Windows, and macOS, and runs the worker handshake
outside the repository. It uploads the wheel as a CI artifact; it never publishes
to a package index. The audit checks package boundaries, required worker modules
and browser assets, and the console entry point. It does not substitute for a
real Fiji registration and save/reload acceptance run.
