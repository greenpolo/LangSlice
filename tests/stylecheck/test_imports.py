"""Tests of the local-import check."""

from __future__ import annotations

from tools.stylecheck.testing import check_source

RELPATH = "src/langslice/ops/fit.py"

CYCLE = {
    "src/langslice/ops/views.py": "from langslice.ops.fit import fit\n",
    "src/langslice/ops/other.py": "import numpy\n",
    "src/langslice/ops/chain.py": "from langslice.ops import views\n",
}


def _messages(source: str) -> list[str]:
    hits = check_source(source, "local-import", relpath=RELPATH, others=CYCLE)
    return [violation.message for violation in hits]


def test_import_inside_a_function_is_flagged() -> None:
    source = """
    def fit():
        import json
        from pathlib import Path
    """
    assert len(_messages(source)) == 2


def test_heavy_module_needs_the_local_comment() -> None:
    source = """
    def fit():
        import torch  # local: slow to import
        import SimpleITK as sitk
        from fireants.registration import (  # local: needed for a fit
            Affine,
        )
    """
    messages = _messages(source)
    assert len(messages) == 1
    assert "slow dependency" in messages[0]


def test_cycle_comment_is_checked_against_the_import_graph() -> None:
    source = """
    def fit():
        from langslice.ops.views import a  # local: cycle with ops.views
        from langslice.ops.chain import b  # local: cycle with ops.chain
        from langslice.ops import views  # local: cycle with views
        from langslice.ops.other import c  # local: cycle with ops.other
        from langslice.ops.views import d  # local: cycle with ops.nowhere
        import json  # local: cycle
        import json  # local: slow
    """
    messages = _messages(source)
    assert len(messages) == 4
    assert "does not import this module" in messages[0]
    assert "name the imported module" in messages[1]


def test_cycle_comment_must_name_the_imported_module() -> None:
    """A real cycle elsewhere does not excuse an unrelated import."""
    source = """
    def fit():
        from langslice.ops.other import a  # local: cycle with ops.views
        from langslice.ops.other import b  # local: cycle with langslice
        from langslice.ops.other import c  # local: cycle with ops
        from langslice.ops.views import d  # local: cycle with ops
    """
    messages = _messages(source)
    assert len(messages) == 4
    assert all("name the imported module" in text for text in messages)


def test_cycle_must_run_from_the_imported_module() -> None:
    """The gaming snippet of the review: a core module names a door.

    Every higher layer imports ``core``, so the named module importing
    this one proves nothing about the module actually imported.
    """
    others = {
        "src/langslice/doors/library.py": (
            "from langslice.core.space import axes\n"
        ),
        "src/langslice/core/atlas.py": "import numpy\n",
    }
    source = """
    def axes():
        from langslice.core.atlas import a  # local: cycle with doors.library
        from langslice.core.atlas import b  # local: cycle with core
        from langslice.core.atlas import c  # local: cycle with core.atlas
    """
    hits = check_source(
        source,
        "local-import",
        relpath="src/langslice/core/space.py",
        others=others,
    )
    messages = [violation.message for violation in hits]
    assert len(messages) == 3
    assert "does not import this module" in messages[2]


def test_cycle_through_a_function_or_type_check_does_not_count() -> None:
    """Only imports that run when the imported module loads close a cycle."""
    others = {
        "src/langslice/ops/lazy.py": """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from langslice.ops.fit import fit


            def later():
                from langslice.ops.fit import fit  # local: cycle with fit
            """,
        "src/langslice/ops/eager.py": """
            try:
                from langslice.ops.fit import fit
            except ImportError:
                fit = None
            """,
    }
    source = """
    def fit():
        from langslice.ops.lazy import a  # local: cycle with ops.lazy
        from langslice.ops.eager import b  # local: cycle with ops.eager
    """
    hits = check_source(source, "local-import", relpath=RELPATH, others=others)
    assert [violation.line for violation in hits] == [3]


def test_module_level_imports_are_not_flagged() -> None:
    source = """
    import json

    if json:
        import pathlib
    """
    assert _messages(source) == []
