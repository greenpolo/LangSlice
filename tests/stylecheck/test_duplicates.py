"""Tests of the duplicate function body check."""

from __future__ import annotations

from tools.stylecheck.testing import check_source

FIRST = '''
def resolve_sections(job, sections):
    """Return the records named, in order."""
    state = job.state
    found = []
    for ref in sections:
        record = state.resolve(ref)
        if record is not None:
            found.append(record)
    return found
'''


def test_duplicate_body_flags_a_copy_with_renamed_variables() -> None:
    # the game: rename the variables of a copied helper
    copy = """
    def _records(job, refs):
        current = job.state
        out = []
        for name in refs:
            item = current.resolve(name)
            if item is not None:
                out.append(item)
        return out
    """

    violations = check_source(
        copy,
        "duplicate-body",
        relpath="src/langslice/ops/masks.py",
        others={"src/langslice/ops/images.py": FIRST},
    )
    assert [violation.symbol for violation in violations] == ["_records"]
    assert (
        "src/langslice/ops/images.py:resolve_sections" in violations[0].message
    )


def test_duplicate_body_keeps_attribute_and_callee_names() -> None:
    different = """
    def _records(job, refs):
        current = job.layout
        out = []
        for name in refs:
            item = current.resolve(name)
            if item is not None:
                out.append(item)
        return out

    def _sizes(job, refs):
        current = job.state
        out = []
        for name in refs:
            item = measure(current, name)
            if item is not None:
                out.append(item)
        return out
    """

    violations = check_source(
        different,
        "duplicate-body",
        relpath="src/langslice/ops/masks.py",
        others={"src/langslice/ops/images.py": FIRST},
    )
    assert violations == []


def test_duplicate_body_ignores_short_bodies() -> None:
    source = """
    def first(job):
        return job.state.in_order()

    def second(job):
        return job.state.in_order()
    """
    assert check_source(source, "duplicate-body") == []


def test_duplicate_body_reports_only_copies_after_the_first() -> None:
    violations = check_source(
        FIRST.replace("resolve_sections", "again"),
        "duplicate-body",
        relpath="src/langslice/ops/images.py",
        others={"src/langslice/ops/masks.py": FIRST},
    )
    assert violations == []
