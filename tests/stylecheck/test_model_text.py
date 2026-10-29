"""Tests of the tool stubs, whose docstrings the docstring rules leave alone.

A stub is a function passed as the second positional argument of
``ToolSpec(...)``, or a function named in the ``STUBS`` table of the
module that renders tool descriptions. Its docstring findings are dropped; every other finding on
it, and every finding on other functions, is kept.
"""

from __future__ import annotations

import dataclasses

from tools.stylecheck.core import Config, Severity, Violation, load_project
from tools.stylecheck.model_text import (
    Stub,
    drop_stub_docstring_findings,
    find_stubs,
)
from tools.stylecheck.testing import REPOSITORY, snippet_project

CONFIG = Config.load(REPOSITORY / "pyproject.toml")

PLUGIN = "src/langslice/engines/example/plugin.py"

#: A plugin module: a stub handed to ``ToolSpec`` by position, one handed
#: through the module attribute, and functions that are not stubs.
PLUGIN_SOURCE = '''
def fit_stub(job, sections):
    """Fit the sections.

    sections: the sections to fit, by name.
    """


def plane_stub(job):
    """Search the plane."""


def body(job, sections):
    """Run the fit."""
    return job


def by_keyword(job):
    """Never passed by position."""


def tools(meta):
    return [
        ToolSpec("fit", fit_stub, body, kind="write"),
        plugin_api.ToolSpec("plane", plane_stub, body),
        ToolSpec(name="other", stub=by_keyword, body=body),
        ToolSpec("short"),
    ]
'''

#: Where each docstring of the plugin module starts and ends.
FIT_STUB_DOCSTRING = (3, 6)
BODY_DEF_LINE = 13

#: Every rule that judges a docstring's form or wording, by family.
DOCSTRING_RULES = (
    "numpydoc:GL08",
    "numpydoc:PR01",
    "ruff:D205",
    "ruff:D407",
    "summary-first-word",
    "parameter-description",
    "docstring-references",
    "docstring-layer",
)


def _violation(rule: str, symbol: str, line: int) -> Violation:
    return Violation(rule, PLUGIN, symbol, line, "Message.", Severity.ERROR)


def test_a_function_passed_second_to_toolspec_is_a_stub() -> None:
    """Find the stubs of ``ToolSpec`` calls, by name or by attribute."""
    stubs = find_stubs(snippet_project({PLUGIN: PLUGIN_SOURCE}), CONFIG)
    assert stubs == [
        Stub(PLUGIN, "fit_stub", *FIT_STUB_DOCSTRING),
        Stub(PLUGIN, "plane_stub", 10, 10),
    ]


def test_the_renderer_stubs_are_the_functions_of_its_table() -> None:
    """Treat only the functions listed in ``STUBS`` as stubs.

    The gaming case of the review: a helper put in the renderer module
    to escape the docstring rules is checked like any code.
    """
    relpath = "src/langslice/doors/declarations.py"
    source = '''
    def place(job):
        """Place the sections."""

    def look(job):
        """Look at a section."""

    def helper(name):
        """helper with no proper docstring"""

    def undocumented(job):
        return job

    STUBS: dict[str, object] = {
        stub.__name__: stub for stub in (place, look, undocumented)
    }
    FULL = helper("x")
    '''
    config = dataclasses.replace(
        CONFIG, docs_renderer_module="langslice.doors.declarations"
    )
    stubs = find_stubs(snippet_project({relpath: source}), config)
    assert [stub.symbol for stub in stubs] == ["place", "look"]


def test_the_renderer_without_a_table_has_no_stubs() -> None:
    """Find no stub in a renderer module that lists none."""
    relpath = "src/langslice/doors/declarations.py"
    source = '''
    def place(job):
        """Place the sections."""
    '''
    config = dataclasses.replace(
        CONFIG, docs_renderer_module="langslice.doors.declarations"
    )
    assert find_stubs(snippet_project({relpath: source}), config) == []


def test_the_real_declarations_stubs_are_the_tools() -> None:
    """Find the declared verbs in the real renderer, and no helper."""
    project = load_project(REPOSITORY, CONFIG)
    stubs = find_stubs(project, CONFIG)
    declared = {
        stub.symbol
        for stub in stubs
        if stub.path == "src/langslice/tools/descriptions/render.py"
    }
    assert {"look", "trace_borders", "submit"} <= declared
    assert not declared & {"base_prompt", "model_doc", "declare", "summary"}


def test_the_docstring_findings_of_a_stub_are_dropped() -> None:
    """Drop every docstring rule, at the def line or inside the docstring."""
    stubs = find_stubs(snippet_project({PLUGIN: PLUGIN_SOURCE}), CONFIG)
    first, last = FIT_STUB_DOCSTRING
    violations = [_violation(rule, "fit_stub", 2) for rule in DOCSTRING_RULES]
    violations.append(_violation("ruff:E501", "fit_stub", first + 2))
    violations.append(_violation("history-comment", "fit_stub", last))
    assert drop_stub_docstring_findings(violations, stubs) == []


def test_the_other_findings_of_a_stub_are_kept() -> None:
    """Keep complexity, names, and line rules outside the docstring."""
    stubs = find_stubs(snippet_project({PLUGIN: PLUGIN_SOURCE}), CONFIG)
    _, last = FIT_STUB_DOCSTRING
    violations = [
        _violation("cognitive-complexity", "fit_stub", 2),
        _violation("banned-name", "fit_stub", 2),
        _violation("parameters", "fit_stub", 2),
        _violation("ruff:E501", "fit_stub", 2),
        _violation("ruff:E501", "fit_stub", last + 1),
        _violation("history-comment", "fit_stub", last + 1),
    ]
    assert drop_stub_docstring_findings(violations, stubs) == violations


def test_the_docstring_findings_of_other_functions_are_kept() -> None:
    """Keep docstring findings on a body or on a stub passed by keyword."""
    stubs = find_stubs(snippet_project({PLUGIN: PLUGIN_SOURCE}), CONFIG)
    violations = [
        _violation(rule, "body", BODY_DEF_LINE) for rule in DOCSTRING_RULES
    ]
    violations.append(_violation("numpydoc:GL08", "by_keyword", 18))
    elsewhere = dataclasses.replace(
        _violation("numpydoc:GL08", "fit_stub", 2), path="src/other.py"
    )
    violations.append(elsewhere)
    assert drop_stub_docstring_findings(violations, stubs) == violations


def test_the_engine_plugins_tool_stubs_are_found() -> None:
    """Find the five engine plugin stubs in the real package."""
    project = load_project(REPOSITORY, CONFIG)
    stubs = find_stubs(project, CONFIG)
    engine_stubs = sorted(
        stub.symbol
        for stub in stubs
        if stub.path.startswith("src/langslice/engines/")
    )
    assert engine_stubs == [
        "emlddmm_plane",
        "emlddmm_stack",
        "spomialign",
        "stalign_lddmm",
        "stalign_plane_search",
    ]
