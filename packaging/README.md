# Worker distribution

The Fiji connector starts the separately installed LangSlice Python worker.
The worker does not require Java, Maven, or `abba-python`. The optional
`langslice[abba]` extra (`abba-python`) is only for `langslice abba`, which starts
ABBA from Python so that the connector's runs can be followed in LangSlice's agent
viewer and log.

## Install from source

Download and extract the repository source archive, open a terminal **in its
root folder**, and run:

```bash
conda env create -f environment.yml
```

The environment installs Python 3.11 and then installs LangSlice and all its
Python dependencies from `pyproject.toml`. It is a regular installation, not an
editable development checkout. Authentication is performed from Fiji's setup
window after selecting this environment; command-line login is also available.

`environment.yml` is a source installation recipe. It creates an isolated
conda environment and installs the Python package and its dependencies with
pip.

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
