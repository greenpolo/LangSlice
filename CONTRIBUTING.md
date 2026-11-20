# Contributing to LangSlice

Bug reports, feature suggestions, and pull requests are welcome.

## Reporting issues

Open a [GitHub issue](https://github.com/greenpolo/LangSlice/issues) with:

- LangSlice version (`langslice version`)
- Python and OS version
- A minimal reproduction (command + slice image dimensions are usually enough)
- The full error or unexpected output

For pipeline questions and discussion with the broader community, the
[image.sc forum](https://forum.image.sc/) is a good place to start.

## Development setup

```bash
git clone https://github.com/greenpolo/LangSlice.git
cd LangSlice
uv venv --python 3.11 .venv
source .venv/bin/activate       # macOS / Linux
# .venv\Scripts\Activate.ps1    # Windows PowerShell (or activate.bat in cmd.exe)
uv pip install -e ".[dev,registration]"
```

Plain `python -m venv .venv` + `pip install -e ".[dev,registration]"` works too.

## Verifying changes

Before opening a PR, please run:

```bash
python -m ruff check .
python -m ruff format --check .
python -m basedpyright
lint-imports
python -m pytest
python -m langslice version
```

`pytest` runs in a few seconds: one registration from start to export on a
small synthetic brain (`tests/test_smoke.py`), the state records, the import
graph and layers, the comment and waiver rule
(`tests/test_no_comments_or_waivers.py`), the Markdown file kinds and the
connectors' skill copies (`tests/test_connector_skills.py`). A change to pictures is judged by running a real
brain and looking at the pictures.

## The style

`STYLE.md` is the code style. ruff (100 columns), basedpyright (standard
mode) and import-linter check what a program can decide, with no baseline
and no waiver: a finding is fixed in the code.

## Pull requests

- Branch from `main`, open the PR against `main`.
- Keep PRs focused — one logical change per PR is easier to review.
- The app plugins' manifests (`connectors/claude-code/.claude-plugin/plugin.json`, `connectors/codex/.codex-plugin/plugin.json`) carry the package's version: a version change in `pyproject.toml` changes them too because Claude Code and Codex update an installed plugin only when its version changes.
- CI runs `ruff`, `basedpyright`, import-linter and the test suite on Linux and Windows, Python 3.11. The checks run in the locked environment (`uv sync --locked --extra dev --extra registration`), since basedpyright's findings change with the libraries' type stubs; the tests run on the newest versions `pip` installs.

## License

By contributing, you agree that your contributions will be licensed
under the project's BSD-3-Clause license.
