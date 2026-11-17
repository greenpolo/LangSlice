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


def test_a_slow_dependency_is_imported_where_it_is_used() -> None:
    source = """
    def fit():
        import torch
        import SimpleITK as sitk
        from fireants.registration import (
            Affine,
        )
    """
    assert _messages(source) == []


def test_a_cycle_is_read_from_the_import_graph() -> None:
    source = """
    def fit():
        from langslice.ops.views import a
        from langslice.ops.chain import b
        from langslice.ops import views
        from langslice.ops.other import c
        import json
    """
    hits = check_source(source, "local-import", relpath=RELPATH, others=CYCLE)
    assert [violation.line for violation in hits] == [6, 7]
    assert "Move the import to the top" in hits[0].message


def test_cycle_must_run_from_the_imported_module() -> None:
    """A door importing this core module does not excuse another import."""
    others = {
        "src/langslice/doors/library.py": (
            "from langslice.core.space import axes\n"
        ),
        "src/langslice/core/atlas.py": "import numpy\n",
    }
    source = """
    def axes():
        from langslice.core.atlas import a
    """
    hits = check_source(
        source,
        "local-import",
        relpath="src/langslice/core/space.py",
        others=others,
    )
    assert [violation.line for violation in hits] == [3]


def test_cycle_through_a_function_or_type_check_does_not_count() -> None:
    """Only imports that run when the imported module loads close a cycle."""
    others = {
        "src/langslice/ops/lazy.py": """
            from typing import TYPE_CHECKING

            if TYPE_CHECKING:
                from langslice.ops.fit import fit


            def later():
                from langslice.ops.fit import fit
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
        from langslice.ops.lazy import a
        from langslice.ops.eager import b
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
