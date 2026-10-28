"""The external tools as the checker runs them: ruff, the formatter, numpydoc.

Each test writes a small package with its own ``pyproject.toml`` (ruff at
79 columns, the numpy convention, the repository's numpydoc checks) and
runs one tool on it.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from tools.stylecheck.core import Config, Project, Violation, load_project
from tools.stylecheck.external import (
    format_violations,
    ruff_explanation,
    ruff_violations,
)
from tools.stylecheck.numpydoc_hook import (
    NumpydocSettings,
    numpydoc_explanation,
    numpydoc_violations,
)
from tools.stylecheck.run import explain

REPO = Path(__file__).resolve().parents[2]


def _pyproject() -> str:
    """Return the fixture's project file: the repository's tool tables."""
    settings = NumpydocSettings.from_pyproject(REPO / "pyproject.toml")
    checks = ", ".join(f'"{code}"' for code in sorted(settings.checks))
    exclude = ", ".join(f"'{pattern}'" for pattern in settings.exclude)
    return textwrap.dedent(
        f"""\
        [tool.stylecheck]
        paths = ["pkg"]
        docs_renderer_module = "pkg.declarations"

        [tool.ruff]
        line-length = 79

        [tool.ruff.lint.pydocstyle]
        convention = "numpy"

        [tool.numpydoc_validation]
        checks = [{checks}]
        exclude = [{exclude}]
        """
    )


def _project(tmp_path: Path, files: dict[str, str]) -> tuple[Project, Config]:
    (tmp_path / "pyproject.toml").write_text(_pyproject())
    for name, text in files.items():
        path = tmp_path / "pkg" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text))
    config = Config.load(tmp_path / "pyproject.toml")
    return load_project(tmp_path, config), config


def _rules(violations: list[Violation]) -> list[tuple[str, str]]:
    return sorted((item.rule, item.symbol) for item in violations)


# --- ruff ----------------------------------------------------------------

LONG = "x" * 80


def test_ruff_reports_long_lines_and_private_access_at_their_definition(
    tmp_path: Path,
) -> None:
    project, config = _project(
        tmp_path,
        {
            "a.py": f'''\
                """Module."""


                class Box:
                    """Box."""

                    def size(self, other):
                        """Return the size."""
                        return other._size


                def call():
                    """Call."""
                    return "{LONG}"
                ''',
        },
    )
    violations = ruff_violations(project, config)
    assert ("ruff:SLF001", "Box.size") in _rules(violations)
    assert ("ruff:E501", "call") in _rules(violations)
    long_line = next(item for item in violations if item.rule == "ruff:E501")
    assert long_line.path == "pkg/a.py"
    assert long_line.message.endswith(".")


def test_ruff_leaves_out_the_rules_another_tool_owns(tmp_path: Path) -> None:
    project, config = _project(
        tmp_path,
        {
            "a.py": '''\
                """Module."""


                def run(A, B):
                    """Runs the thing"""
                    R = A @ B
                    return R
                ''',
        },
    )
    codes = {item.rule for item in ruff_violations(project, config)}
    # summary rules belong to numpydoc, capital maths names are allowed
    for owned in ("ruff:D400", "ruff:D401", "ruff:N803", "ruff:N806"):
        assert owned not in codes


def test_ruff_runs_the_import_conventions_and_blind_except(
    tmp_path: Path,
) -> None:
    project, config = _project(
        tmp_path,
        {
            "a.py": '''\
                """Module."""

                import numpy


                def run():
                    """Run."""
                    try:
                        return numpy.zeros(1)
                    except Exception:
                        pass
                ''',
        },
    )
    codes = {item.rule for item in ruff_violations(project, config)}
    assert {"ruff:ICN001", "ruff:BLE001", "ruff:S110"} <= codes


def test_ruff_explanation_is_ruff_s_own_text() -> None:
    assert "line-too-long" in ruff_explanation("E501", REPO)
    assert "E501" in explain("ruff", REPO)


# --- the formatter -------------------------------------------------------


def test_format_reports_each_unformatted_file_once(tmp_path: Path) -> None:
    project, config = _project(
        tmp_path,
        {
            "messy.py": "x = {  'a':1,\n'b' : 2 }\ny=[1,2,\n3]\n",
            "tidy.py": 'x = {"a": 1, "b": 2}\n',
        },
    )
    violations = format_violations(project, config)
    assert [(item.rule, item.path, item.symbol) for item in violations] == [
        ("format", "pkg/messy.py", "<module>")
    ]


def test_format_wraps_at_79_columns(tmp_path: Path) -> None:
    call = "result = compute(" + ", ".join(f"arg{i}" for i in range(12)) + ")"
    assert 79 < len(call) <= 100
    project, config = _project(tmp_path, {"a.py": call + "\n"})
    assert len(format_violations(project, config)) == 1


# --- numpydoc ------------------------------------------------------------


def _numpydoc(tmp_path: Path, source: str) -> list[tuple[str, str]]:
    project, config = _project(tmp_path, {"a.py": source})
    return _rules(numpydoc_violations(project, config))


def test_numpydoc_requires_public_docstrings_only(tmp_path: Path) -> None:
    violations = _numpydoc(
        tmp_path,
        '''\
        """Do one thing."""


        def public(x):
            return x


        def _private(x):
            return x


        class _Hidden:
            def method(self):
                return 1
        ''',
    )
    assert ("numpydoc:GL08", "public") in violations
    assert all("_private" not in symbol for _, symbol in violations)
    assert all("_Hidden" not in symbol for _, symbol in violations)


def test_numpydoc_checks_parameters_and_returns(tmp_path: Path) -> None:
    violations = _numpydoc(
        tmp_path,
        '''\
        """Do one thing."""


        def scale(image, factor):
            """Scale an image.

            Parameters
            ----------
            image
                The picture to scale, as an array.
            size
                A parameter the function does not have.
            """
            return image * factor
        ''',
    )
    assert ("numpydoc:PR01", "scale") in violations
    assert ("numpydoc:PR02", "scale") in violations
    assert ("numpydoc:RT01", "scale") in violations


def test_numpydoc_checks_the_summary(tmp_path: Path) -> None:
    violations = _numpydoc(
        tmp_path,
        '''\
        """Do one thing."""


        def first():
            """Returns the first item"""


        def second():
            """Return the second item of the list, which is a summary that
            runs over two lines."""
        ''',
    )
    assert ("numpydoc:SS03", "first") in violations
    assert ("numpydoc:SS05", "first") in violations
    assert ("numpydoc:SS06", "second") in violations


def test_numpydoc_skips_nested_functions_and_properties(
    tmp_path: Path,
) -> None:
    violations = _numpydoc(
        tmp_path,
        '''\
        """Do one thing."""


        class Frame:
            """Hold a frame."""

            @property
            def width_px(self) -> int:
                """Width of the frame in pixels."""
                return 3


        def outer():
            """Run the outer step."""

            def inner(value):
                return value

            inner(1)
        ''',
    )
    assert violations == []


def test_numpydoc_compares_a_documented_dataclass_with_its_fields(
    tmp_path: Path,
) -> None:
    violations = _numpydoc(
        tmp_path,
        '''\
        """Do one thing."""

        from dataclasses import dataclass, field
        from typing import ClassVar


        @dataclass(frozen=True)
        class Point:
            """Hold a point.

            Parameters
            ----------
            x_mm
                Distance along the first axis, in millimetres.
            y_mm
                Distance along the second axis, in millimetres.
            """

            x_mm: float
            y_mm: float
            cache: list[int] = field(init=False)
            UNIT: ClassVar[str] = "mm"


        @dataclass(frozen=True)
        class Short:
            """Hold a pair.

            Parameters
            ----------
            first
                The first value of the pair.
            """

            first: int
            second: int


        @dataclass(frozen=True)
        class Bare:
            """Hold a value without listing it."""

            value: int
        ''',
    )
    assert violations == [("numpydoc:PR01", "Short")]


def test_numpydoc_ignore_comments_are_not_honoured(tmp_path: Path) -> None:
    violations = _numpydoc(
        tmp_path,
        '''\
        """Do one thing."""


        def public(x):  # numpydoc ignore=GL08
            return x
        ''',
    )
    assert violations == [("numpydoc:GL08", "public")]


def test_numpydoc_checks_the_module_docstring(tmp_path: Path) -> None:
    violations = _numpydoc(tmp_path, "x = 1\n")
    assert violations == [("numpydoc:GL08", "<module>")]


def test_numpydoc_reads_the_declarations_module_too(tmp_path: Path) -> None:
    # the runner drops the findings on tool stubs (test_model_text.py)
    project, config = _project(
        tmp_path,
        {
            "declarations.py": '''\
                def look(mode: str = "atlas") -> dict:
                    """Look at the stack.

                    Args:
                        mode: what to draw.
                    """
                ''',
        },
    )
    rules = _rules(numpydoc_violations(project, config))
    assert ("numpydoc:GL08", "<module>") in rules
    assert ("numpydoc:PR01", "look") in rules


def test_numpydoc_explanation_names_the_code() -> None:
    assert "not documented" in numpydoc_explanation("PR01", REPO)
    assert "no check" in numpydoc_explanation("XX99", REPO)
    assert "GL08" in explain("numpydoc", REPO)


def test_ruff_keeps_the_shared_ignores_and_drops_the_main_runs(
    tmp_path: Path,
) -> None:
    long_function = f'''\
        """Module."""


        def call():
            """Call."""
            return "{LONG}"
        '''
    project, config = _project(
        tmp_path, {"a.py": long_function, "b.py": long_function}
    )
    with (tmp_path / "pyproject.toml").open("a") as pyproject:
        pyproject.write(
            textwrap.dedent(
                """
                [tool.ruff.lint.pycodestyle]
                max-line-length = 100

                [tool.ruff.lint.per-file-ignores]
                "pkg/**" = ["E501"]

                [tool.ruff.lint.extend-per-file-ignores]
                "pkg/b.py" = ["E501"]
                """
            )
        )
    long_lines = {
        item.path
        for item in ruff_violations(project, config)
        if item.rule == "ruff:E501"
    }
    # 79 columns, not the main run's 100; its own ignores are left out
    assert long_lines == {"pkg/a.py"}
