"""Tests of the docstring checks and the docstring layer check."""

from __future__ import annotations

from tools.stylecheck.testing import check_source, flagged_lines

#: Two parameters described by their own names, two described well and one
#: left without a description.
DESCRIPTIONS = '''
def fit(mode, section_mode, metric, plane, settings):
    """Fit something.

    Parameters
    ----------
    mode : str
        The mode.
    section_mode
        The section mode value.
    metric : {"mi", "ncc"}
        Similarity measure the fit optimises.
    plane : array, shape (H, W)
        The atlas plane the section is fitted to.
    settings
    """
'''


def test_short_and_tautological_descriptions_are_flagged() -> None:
    hits = check_source(DESCRIPTIONS, "parameter-description")
    assert [violation.message.split("'")[1] for violation in hits] == [
        "mode",
        "section_mode",
    ]


def test_description_of_plural_name_counts_as_its_name() -> None:
    source = '''
    def fit(sections):
        """Fit something.

        Parameters
        ----------
        sections
            The given sections values.
        """
    '''
    assert len(check_source(source, "parameter-description")) == 1


def test_summary_opening_with_a_banned_word_is_flagged() -> None:
    source = '''
    def a():
        """The affine fit."""

    def b():
        """Whether the fit converged."""

    def c():
        """Fit an affine."""

    class D:
        """A record of a fit."""

        @property
        def size(self):
            """The size in pixels."""
    '''
    assert flagged_lines(source, "summary-first-word") == [3, 6]


def test_references_are_counted_outside_see_also() -> None:
    source = '''
    def fit(job, plane):
        """Fit with :func:`a`, :class:`B`, ``np.ndarray`` and ``c.d``.

        Uses ``job.layout.folder`` and ``plane.shape``, writes
        ``state.json``, and reads ``e.f``.

        See Also
        --------
        :func:`g`, :func:`h`, ``i.j``
        """

    def gamed():
        """Fit with :func:`a` and :func:`b`.

        Notes
        -----
        Moved into Notes: :func:`c`, :func:`d`, :func:`e`.
        """
    '''
    assert flagged_lines(source, "docstring-references") == [3, 14]


def test_repeated_reference_counts_once() -> None:
    source = '''
    def fit():
        """Call :func:`a`, :func:`a`, :func:`a`, :func:`a`, :func:`a`."""
    '''
    assert flagged_lines(source, "docstring-references") == []


HIGHER = {
    "src/langslice/doors/library.py": """
    class ToolBox:
        pass

    def build_tools():
        pass

    def save():
        pass
    """,
    "src/langslice/core/space.py": """
    def save():
        pass
    """,
}


def test_lower_layer_docstring_naming_a_higher_layer_is_flagged() -> None:
    source = '''
    """Read the atlas; see langslice.doors.library for the tools."""

    def fit(job):
        """Fit with ``ToolBox`` and :func:`build_tools`.

        Also ``doors.cli``, ``ops/registry.py`` and ``job.folder``.
        """
    '''
    hits = check_source(source, "docstring-layer", others=HIGHER)
    tokens = [violation.message.split("'")[1] for violation in hits]
    assert tokens == [
        "langslice.doors",
        "build_tools",
        "ToolBox",
        "doors.cli",
        "ops/registry.py",
    ]


def test_tool_names_are_vocabulary_not_upward_references() -> None:
    """A tool's name is not a reference to the ops verb of that name."""
    others = {
        **HIGHER,
        "src/langslice/core/toolset.py": """
        TOOLS = {tool.name: tool for tool in (
            ToolDef("trace_borders", "write", "Nonlinear"),
            ToolDef("position_sections", "write", "Positioning"),
        )}
        """,
        "src/langslice/ops/verbs.py": """
        def trace_borders():
            pass

        def position_sections():
            pass

        def fit_deformable():
            pass
        """,
    }
    source = '''
    def fit():
        """Run before ``trace_borders`` and :func:`position_sections`.

        Not ``fit_deformable``, nor ``ops.verbs.trace_borders``.
        """
    '''
    hits = check_source(source, "docstring-layer", others=others)
    tokens = [violation.message.split("'")[1] for violation in hits]
    assert tokens == ["fit_deformable", "ops.verbs.trace_borders"]


def test_plain_words_and_lower_layers_are_not_flagged() -> None:
    source = '''
    def fit():
        """Call ``save`` and ``langslice.core.space`` from the doors."""
    '''
    assert flagged_lines(source, "docstring-layer", others=HIGHER) == []
    relpath = "src/langslice/hosts/run.py"
    gamed = '''
    def fit():
        """Call ``ToolBox`` from ``langslice.doors``."""
    '''
    assert flagged_lines(gamed, "docstring-layer", relpath=relpath) == []
