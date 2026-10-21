"""The in-tree engines (``langslice.engines``): the three load together in one job, their tools
do not clash, a missing environment is refused at call time with ``ENGINE_UNAVAILABLE``, and
nothing of the engine environments is imported into LangSlice's process.

The engines themselves run in their own conda environments; a test that runs one is marked
``slow`` and skips where the environment is missing.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest

from langslice.core import plugins as core_plugins
from langslice.core.spec import JobSpec
from langslice.doors.tools.toolbox import build_tools
from langslice.ops.registry import VERBS, verbs_for
from tests.linear_tool_helpers import ToolContext, stack, tool_named

REFS = [
    "langslice.engines.stalign.plugin:tools",
    "langslice.engines.spomialign.plugin:tools",
    "langslice.engines.emlddmm.plugin:tools",
]
TOOLS = {
    "stalign": ["stalign_lddmm", "stalign_sorting", "stalign_plane_search"],
    "spomialign": ["spomialign"],
    "emlddmm": ["emlddmm_stack", "emlddmm_plane"],
}
#: The first argument set that gets each tool past its argument checks.
CALLS = {
    "stalign_lddmm": {"sections": ["s0"]},
    "stalign_sorting": {"sections": ["s0"]},
    "stalign_plane_search": {"sections": ["s0"]},
    "spomialign": {"sections": ["s0"], "transform": "affine"},
    "emlddmm_stack": {"sections": ["s0", "s1"], "slot_um": 100.0},
    "emlddmm_plane": {"sections": ["s0", "s1"], "slot_um": 100.0},
}
ENGINES_PYTHON = {
    "stalign": "stalign2",
    "spomialign": "rigid_spomialign",
    "emlddmm": "rigid_emlddmm",
}


@pytest.fixture(autouse=True)
def clean_registry():
    held = dict(core_plugins._LOADED)
    yield
    core_plugins._LOADED.clear()
    core_plugins._LOADED.update(held)


def test_the_three_refs_load_together_without_a_name_clash(tmp_path: Path):
    spec = JobSpec(image_folder=str(tmp_path), plugins=REFS)
    verbs = verbs_for(spec)
    added = [name for name in verbs if name not in VERBS]
    assert sorted(added) == sorted(name for names in TOOLS.values() for name in names)
    assert len(added) == len(set(added))


def test_each_tool_is_refused_when_its_environment_is_missing(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    gone = str(tmp_path / "no" / "python")
    config = {ref: {"python": gone} for ref in REFS}
    _state, ctx, spec = stack(images, 3, placed=True, plugins=REFS, plugin_config=config,
                              tasks=["position", "transform", "nonlinear"])
    box = build_tools(_state, ctx, spec)
    for names in TOOLS.values():
        for name in names:
            tool = tool_named(box, name)
            arguments = dict(CALLS[name])
            if "tool_context" in inspect.signature(tool).parameters:
                arguments["tool_context"] = ToolContext()
            reply = tool(**arguments)
            assert reply.get("code") == "ENGINE_UNAVAILABLE" or "ENGINE_UNAVAILABLE" in str(
                reply), (name, reply)
            assert gone in str(reply)


def test_nothing_of_the_engine_environments_is_imported():
    code = (
        "import sys\n"
        "import langslice.engines.stalign.plugin, langslice.engines.spomialign.plugin, "
        "langslice.engines.emlddmm.plugin\n"
        "banned = ('STalign', 'emlddmm', 'roma', 'SPOmiAlign', 'torch', 'SimpleITK_x')\n"
        "bad = sorted(m for m in sys.modules if m.split('.')[0] in banned)\n"
        "print(bad)\nsys.exit(1 if bad else 0)\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=False)
    assert done.returncode == 0, done.stdout + done.stderr
