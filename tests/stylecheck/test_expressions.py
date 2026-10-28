"""Tests of the one-idea-per-line expression checks."""

from __future__ import annotations

from tools.stylecheck.testing import check_source, flagged_lines


def test_nested_conditional_flags_a_chain() -> None:
    source = """
    def caption(drawn, warp):
        return "warped" if drawn else "held" if warp else "none"
    """
    violations = check_source(source, "nested-conditional")
    assert len(violations) == 1


def test_nested_conditional_accepts_one_level() -> None:
    source = """
    def caption(drawn):
        return "warped" if drawn else "none"
    """
    assert check_source(source, "nested-conditional") == []


def test_conditional_placement_accepts_whole_values() -> None:
    source = """
    def caption(drawn, total):
        label = "warped" if drawn else "none"
        total += 1 if drawn else 0
        width: int = 2 if drawn else 1
        order = sorted(total, key=lambda x: x.a if drawn else x.b)
        return label if width else None
    """
    assert check_source(source, "conditional-placement") == []


def test_conditional_placement_flags_one_inside_an_expression() -> None:
    source = """
    def caption(drawn, items):
        draw(color="red" if drawn else "blue")
        names = [x if drawn else None for x in items]
        text = f"{'a' if drawn else 'b'}"
        for item in items if drawn else []:
            pass
        return 1 + (2 if drawn else 3)
    """
    assert len(check_source(source, "conditional-placement")) == 5


def test_conditional_placement_leaves_a_nested_one_to_its_check() -> None:
    source = """
    def caption(a, b):
        return 1 if a else 2 if b else 3
    """
    assert check_source(source, "conditional-placement") == []


def test_conditional_placement_flags_the_disguised_conditionals() -> None:
    """The gaming snippets of the review: a pair indexed, and and-or."""
    source = """
    def caption(flag, record, items):
        draw(("no", "yes")[flag])
        label = ["linear", "warped"][record.deformation is not None]
        draw(flag and "yes" or "no")
        name = record and record.name or "unnamed"
        text = f"{flag and 'a' or 'b'}"
        return flag and len(items) or 0
    """
    lines = flagged_lines(source, "conditional-placement")
    assert lines == [3, 4, 5, 6, 7, 8]


def test_conditional_placement_leaves_boolean_logic_and_indexing() -> None:
    source = """
    def caption(a, b, c, record, pairs, i):
        if a and b > 1 or c:
            pass
        while a and record.ready() or not b:
            pass
        assert a and b or isinstance(c, int)
        ok = a and b or c
        done = a and b > 0 or c < 1
        keep = [x for x in pairs if x and x.name or c]
        shown = bool(a and "x" or "")
        first = (a, b)[0]
        third = (a, b, c)[i]
        row = pairs[i]
        return a and record.is_open() or b
    """
    assert flagged_lines(source, "conditional-placement") == []


def test_nested_comprehension_flags_one_inside_another() -> None:
    source = """
    def table(rows):
        grid = [[cell for cell in row] for row in rows]
        some = [row for row in rows if any(cell for cell in row)]
        return grid, some
    """
    assert len(check_source(source, "nested-comprehension")) == 2


def test_nested_comprehension_accepts_a_phrase() -> None:
    source = """
    def names(sections):
        return [section.name for section in sections if section.done]
    """
    assert check_source(source, "nested-comprehension") == []


def test_comprehension_clauses_flags_two_fors_or_two_ifs() -> None:
    source = """
    def flat(rows):
        cells = [cell for row in rows for cell in row]
        kept = [row for row in rows if row if row.ok]
        both = [row for row in rows if row and row.ok]
        return cells, kept, both
    """
    violations = check_source(source, "comprehension-clauses")
    assert [violation.line for violation in violations] == [3, 4]


def test_walrus_in_comprehension_is_flagged() -> None:
    # the game: dodge the nested comprehension with an assignment
    source = """
    def matches(paths):
        found = [m for path in paths if (m := PATTERN.match(path))]
        if (count := len(found)) > 1:
            return count
        return found
    """
    violations = check_source(source, "walrus-in-comprehension")
    assert [violation.line for violation in violations] == [3]
