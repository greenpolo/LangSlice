"""The in-tree engines on every platform: the GPU queue they take through the plugin API, no
machine folder in their defaults, and the job paths a reply names.

No engine is run here.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

from langslice.engines import _blocks

ENGINES = Path(_blocks.__file__).resolve().parent


# --- the GPU queue ------------------------------------------------------------------------------


def _imports_from(name: str, module: str) -> bool:
    tree = ast.parse((ENGINES / name / "engine.py").read_text(encoding="utf-8"))
    return any(isinstance(node, ast.ImportFrom) and node.module == module
               and any(alias.name == "gpu_turn" for alias in node.names)
               for node in ast.walk(tree))


def test_every_gpu_engine_takes_the_shared_queue_and_none_imports_fcntl():
    for name in ("stalign", "spomialign", "emlddmm"):
        tree = ast.parse((ENGINES / name / "engine.py").read_text(encoding="utf-8"))
        imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import)
                    for alias in node.names}
        assert "fcntl" not in imported, name
        # in LangSlice's process, through the plugin API (the core's gpu_turn)
        assert _imports_from(name, "langslice.plugin_api"), name


def test_no_engine_default_names_a_machine_folder():
    from langslice.engines.emlddmm import plugin as emlddmm
    from langslice.engines.spomialign import plugin as spomialign
    from langslice.engines.stalign import plugin as stalign

    for plugin in (emlddmm, spomialign, stalign):
        for key, value in plugin.DEFAULTS.items():
            assert "Software" not in str(value), (plugin.NAME, key)


# --- paths in replies ---------------------------------------------------------------------------


def test_a_reply_names_a_job_path_with_forward_slashes(tmp_path: Path):
    context = SimpleNamespace(job_dir=tmp_path)
    folder = tmp_path / "plugins" / "stalign_lddmm" / "calls" / "0001_s0"
    relative = _blocks.job_path(context, folder)  # pyright: ignore[reportArgumentType]
    assert relative == "plugins/stalign_lddmm/calls/0001_s0"
    outside = tmp_path.parent / "other"
    assert _blocks.job_path(context, outside) == str(outside)  # pyright: ignore[reportArgumentType]
