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
python -m tools.stylecheck
python -m basedpyright
lint-imports
python -m pytest
python -m langslice version
```

`pytest` runs in parallel (pytest-xdist) and deselects the `slow` tests; run those with `python -m pytest -m slow`. A test that fails only in parallel is re-run with `-n 0`.

## The style checks

`STYLE.md` is the code style. The rules a program can decide are checked
by `tools/stylecheck`, which also runs ruff's style rules, `ruff format`
and numpydoc's docstring validation. Run it from the repository root:

```bash
python -m tools.stylecheck                      # everything
python -m tools.stylecheck --paths src/langslice/core/space.py
python -m tools.stylecheck --rule cognitive-complexity --rule ruff:E501
python -m tools.stylecheck --explain cognitive-complexity
```

The report lists each error under its file: the line, the rule, the
enclosing function or class, and a sentence giving the limit. `--explain`
prints what a rule flags; for the checker's own checks that includes the
section of `STYLE.md` it enforces, and for a ruff or numpydoc code it is
that tool's description. Warnings (such as long paragraphs) are counted at
the end; `--show-warnings` lists them. They never fail a run. A `--paths`
or `--rule` run still parses the whole package, since some checks compare
modules: about 1.5 s for one of the checker's own checks, 15 s with ruff
and numpydoc.

### The baseline only shrinks

Code written before the checker has errors, listed in
`tools/stylecheck/baseline.json` as a count per rule, file and function.
A run fails when a function has more errors of a rule than its entry
allows, or any error without an entry. A failing check is fixed in the
code, never by adding an entry. When you fix errors the report says the
baseline can shrink; run `python -m tools.stylecheck --prune` and commit
the smaller file with the fixes. Prune only in the commit that carries
the fixes, and not while another session is editing the same checkout:
`--prune` records the working tree as it is, so a function that is
half-moved loses its entries for good. Two branches that each pruned may
conflict in the file; the right resolution is the smaller count of each
entry, which `--prune` after the merge writes.

CI runs the checker with `--against` the previous commit (the pull
request's base), and `tests/test_style.py` runs it against `HEAD`. It
fails when an entry is new or larger than at that commit, with two
exceptions: the errors of a function whose file was moved or renamed (the
same rule and function, in a file that no longer exists, at most its old
count), and the errors of a rule being adopted (below). A function moved
between files that both still exist counts as new unless the commit
declares the move in `tools/stylecheck/moves.json` (each entry an old
`path` and `symbol` and a `new_path` and `new_symbol`; generate it, never
type it): its errors are then accepted under the new name up to what the
old entry lost (its whole count once it is gone; a split module lists its
module level once per part), and the settings that name modules are read
through the same map, so a moved module is not a loosened setting.

The file also records the checks that ran and every setting that decides
what they report: `[tool.stylecheck]`, the checker's ruff rules in
`tools/stylecheck/external.py`, and the `[tool.ruff]`,
`[tool.numpydoc_validation]` and `[tool.basedpyright]` tables.
`--prune` and `--write-baseline` write that record; `--against` fails
when it differs from the run's.

### Adopting a rule, and loosening one

A new check, a newly selected ruff or numpydoc code, or a stricter limit
is adopted in one commit: change the code or the setting, then run
`python -m tools.stylecheck --write-baseline --replace-baseline`.
`--against` accepts the new entries of a rule the previous commit's record
did not run, and, for a stricter setting of the checker, the errors that
the tightening itself adds (it runs the checks with both settings on the
new code, so code added in the same commit is held to the old limit).

`--against` refuses any setting looser than at the previous commit: a
higher limit, a module added to an exemption list
(`json_boundary_modules`, `cli_modules`, `heavy_modules`), a word removed
from a banned list, a code added to an ignore list or removed from a
selection, a check removed, or any change to a setting without a
direction (such as `docs_renderer_module`). Loosening is the designer's
decision, never a fix: the designer makes it in a commit of its own and
checks that commit with `--against REF --allow-loosening`, which reports
the loosening and still applies every other check. CI never passes that
flag, so its lint job fails on that one commit by design. The checks
themselves, `tools/stylecheck/rules/`, and their tests are reviewed by a
person in every change; a check made more lenient is a loosening too,
which no program detects.

### basedpyright

basedpyright runs in strict mode. The errors that predate it are in
`.basedpyright/baseline.json`, and `python -m basedpyright` reports only
new ones. A local run also removes the entries of fixed errors from the
file by itself, so commit the smaller file with the fixes; in CI
basedpyright leaves the file as it is. Never run
`basedpyright --writebaseline`, which lists every current error: the
checker's `--against` holds this file to "only shrinks" as well (the
entries per file and rule may not grow, except for a moved file, an
entry that left a file `moves.json` moves definitions out of, and a rule
whose report level changed).

## Pull requests

- Branch from `main`, open the PR against `main`.
- Keep PRs focused — one logical change per PR is easier to review.
- CI runs `ruff`, the style checker (against the base, with both baselines), `basedpyright` and the test suite on Linux Python 3.11 / 3.12 and Windows Python 3.11. The Fiji connector builds and runs discovery, menu and geometry checks on Windows; its shebang-based fake-worker subprocess check runs on Linux.

## License

By contributing, you agree that your contributions will be licensed
under the project's BSD-3-Clause license.
