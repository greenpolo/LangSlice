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
python -m basedpyright
lint-imports
python -m pytest
python -m langslice version
```

## Pull requests

- Branch from `main`, open the PR against `main`.
- Keep PRs focused — one logical change per PR is easier to review.
- CI runs `ruff`, `basedpyright` and the test suite on Linux Python 3.10 / 3.11 / 3.12 and Windows Python 3.11. The Fiji connector builds and runs its discovery, menu and geometry checks on Windows; its shebang-based fake-worker subprocess check runs on Linux.

## License

By contributing, you agree that your contributions will be licensed
under the project's BSD-3-Clause license.
