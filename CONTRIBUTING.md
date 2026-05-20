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
python -m venv .venv
.venv/Scripts/activate          # Windows
# source .venv/bin/activate     # macOS / Linux
pip install -e ".[dev]"
```

## Verifying changes

Before opening a PR, please run:

```bash
python -m ruff check .
python -m basedpyright
python -m pytest
python -m langslice_harness version
```

If you change the desktop GUI, also run `pnpm build` from `tauri-gui/`
and `cargo check` from `tauri-gui/src-tauri/`.

## Pull requests

- Branch from `main`, open the PR against `main`.
- Keep PRs focused — one logical change per PR is easier to review.
- The CI workflow runs `ruff`, `basedpyright`, and a harness test
  subset on Python 3.10 / 3.11 / 3.12.

## License

By contributing, you agree that your contributions will be licensed
under the project's BSD-3-Clause license.
