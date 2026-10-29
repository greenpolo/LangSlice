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

`pytest` runs in parallel (pytest-xdist) and deselects the `slow` and `golden` tests; run those with `python -m pytest -m slow` and `python -m pytest -m golden` (the [golden outputs](docs/developers/architecture.md#verification-and-diagnostics), one test per door: the library, the agent CLI and MCP). A test that fails only in parallel is re-run with `-n 0`.

A change meant to alter no output passes the golden test unchanged. The goldens are per-checkout and not under version control, so compare a branch with goldens generated on `main`: `LANGSLICE_UPDATE_GOLDEN=1 python -m pytest -m golden` regenerates them only on `main` (with `LANGSLICE_GOLDEN_FORCE=1` as well, anywhere), records the commit in `tests/golden/data/expected/<door>/provenance.json`, and fails once with the diff for review.

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
fails when an entry is new or larger than at that commit, unless the
errors belong to a rule being adopted (below) or followed their code
through the move map. It also fails while the baseline holds more than
the code has: an entry the report says can shrink is pruned in the same
commit, so a fixed error cannot come back later unseen.

### Moving or renaming code that has baseline entries

The one way to move or rename baselined code is to declare it in
`tools/stylecheck/moves.json`, which is reviewed in the diff like the
code. Each row names an old `path` and `symbol` and a `new_path` and
`new_symbol` (a definition's qualified name, such as `Job.save`, or
`<module>` for module-level code); a rename inside a file is a row with
the same `path` and `new_path`. Generate the rows, never type them. An
error in a file or under a name no row leads to counts as new, even when
an error of the same rule left a file that is gone.

A row is followed only when, at the reference, its `path` held its
`symbol` and, now, its `new_path` holds its `new_symbol`; `--against`
lists the rows whose new definition is missing. Its new place may then have up to
what the old entry lost since the reference (its whole count once it is
gone). A definition moved to several places, such as the module level of
a module split into parts, lists one row per place, and the places share
what it lost: together they may have no more than the old entry had. A
rule that reports at most one error per file (`format`, `module-size`,
`module-name`; declared with `per_file` where the check is registered) is
the state of the whole file, so each part of a split file may inherit its
error, once the old file has lost it. The
settings that name modules are read at the reference through the same
map, a package's name standing for every module moved out of it, so a
moved module is not a loosened setting; each exemption list names every
module its old entries' code went to. A row whose old definition is not
at the reference (its `path` absent there, or without the `symbol`) is
inert, as all rows become once the move has landed;
`python -m tools.stylecheck --prune --against REF` removes the rows inert
at `REF`, and an empty map is fine.

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
entries per file and rule may not grow, except for an entry that left a
file `moves.json` moves definitions out of, which may reappear once in a
file it moves them into, and a rule whose report level changed). A new
`report` key is a tightening when its level is at least the checking
mode's default for that rule, which the checker reads from the installed
basedpyright.

## Pull requests

- Branch from `main`, open the PR against `main`.
- Keep PRs focused — one logical change per PR is easier to review.
- CI runs `ruff`, the style checker (against the base, with both baselines), `basedpyright` and the test suite on Linux Python 3.11 / 3.12 and Windows Python 3.11. The Fiji connector builds and runs discovery, menu and geometry checks on Windows; its shebang-based fake-worker subprocess check runs on Linux.

## License

By contributing, you agree that your contributions will be licensed
under the project's BSD-3-Clause license.
