"""Tests of the comment checks: history and line-length waivers."""

from __future__ import annotations

from tools.stylecheck.testing import flagged_lines


def test_history_in_comments_is_flagged() -> None:
    source = """
    a = 1  # changed on 2026-10-14
    b = 2  # since Oct 2026
    c = 3  # this used to be a list
    d = 4  # the legacy reader
    e = 5  # no longer needed
    f = 6  # previously fitted
    g = 7  # the mask is used to crop
    h = 8  # plane by plane: the 3D version needs too much memory
    """
    assert flagged_lines(source, "history-comment") == [2, 3, 4, 5, 6, 7]


def test_history_in_docstrings_uses_the_shorter_list() -> None:
    source = '''
    def a():
        """Read a legacy folder, keeping the previously fitted affine."""

    def b():
        """Skip a section no longer in the stack."""

    def c():
        """Read what an earlier version wrote, as of 2026-10-14."""

    def d():
        """Return the value; it used to be a list."""
    '''
    assert flagged_lines(source, "history-comment") == [9, 12]


def test_line_length_waiver_needs_a_url_or_a_table() -> None:
    source = """
    a = "x"  # noqa: E501
    b = "https://example.org/a/very/long/address"  # noqa: E501
    TABLE = [
        ("a", 1.0, -2),  # noqa: E501
        ("b", 2.0, 3),
    ]
    c = [
        call(),  # noqa: E501
        2,
    ]
    d = 1  # noqa: F401
    """
    assert flagged_lines(source, "noqa-line-length") == [2, 9]
