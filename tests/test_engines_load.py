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
    "stalign": ["stalign_lddmm", "stalign_plane_search"],
    "spomialign": ["spomialign"],
    "emlddmm": ["emlddmm_stack", "emlddmm_plane"],
}
#: The first argument set that gets each tool past its argument checks.
CALLS = {
    "stalign_lddmm": {"sections": ["s0"]},
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


# --- the contract: argument order, the mask rules, the blocks ----------------------------------

ARGUMENTS = {
    "stalign_lddmm": ["sections", "section_image", "atlas_image", "mask", "smoothness_um",
                      "affine", "displacement_um", "sorting", "iterations", "view"],
    "stalign_plane_search": ["sections", "section_image", "atlas_image", "view"],
    "spomialign": ["sections", "transform", "section_image", "atlas_image", "mask",
                   "smoothness_um", "matches", "view"],
    "emlddmm_stack": ["sections", "slot_um", "section_image", "atlas_image", "mask",
                      "smoothness_um", "rigid", "warp", "warp_freedom", "iterations", "sorting",
                      "view"],
    "emlddmm_plane": ["sections", "slot_um", "section_image", "atlas_image", "iterations",
                      "view"],
}


def test_the_declared_arguments_follow_the_interface_note(tmp_path: Path):
    from langslice.core import plugins

    specs = plugins.tool_specs(JobSpec(image_folder=str(tmp_path), plugins=REFS))
    for name, wanted in ARGUMENTS.items():
        _plugin, tool = specs[name]
        assert list(inspect.signature(tool.handler).parameters)[1:] == wanted, name
        assert list(inspect.signature(tool.stub).parameters) == wanted, name
    assert "stalign_sorting" not in specs


def test_stalign_refuses_a_section_mask_by_name_before_anything_else(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    gone = str(tmp_path / "no" / "python")
    config = {REFS[0]: {"python": gone}}
    state, ctx, spec = stack(images, 2, placed=True, plugins=[REFS[0]], plugin_config=config,
                             tasks=["position", "transform", "nonlinear"])
    reply = tool_named(build_tools(state, ctx, spec), "stalign_lddmm")(sections=["s0"], mask="m")
    assert reply.get("error") == "MASK_UNSUPPORTED" or "MASK_UNSUPPORTED" in str(reply)
    for engine in ("fireants_syn", "elastix_affine", "emlddmm_stack", "spomialign"):
        assert engine in str(reply)


def _fake_frame(h: int = 40, w: int = 60):
    from types import SimpleNamespace

    import numpy as np

    annotation = np.zeros((h, w), np.int32)
    annotation[5:35, 8:52] = 1
    return SimpleNamespace(
        id="s0.png", atlas={"annotation": annotation, "template": np.full((h, w), 100.0)},
        section_image=np.full((h, w, 3), 200, np.uint8), plane={"px_um": (25.0, 25.0)},
        placed_atlas_to_section=np.eye(3), excluded=None, damage_zone=None, applied_map=None)


def test_the_atlas_image_resolves_and_the_reply_blocks_name_what_was_used():
    import numpy as np

    from langslice.engines import _blocks

    frame = _fake_frame()
    plane, resolved = _blocks.atlas_plane(frame, "nissl")
    assert resolved == "template" and plane is frame.atlas["template"]
    frame.atlas["ccfv3augmented_nissl"] = np.zeros((40, 60))
    assert _blocks.atlas_plane(frame, "nissl")[1] == "ccfv3augmented_nissl"
    with pytest.raises(Exception, match="no image"):
        _blocks.atlas_plane(frame, "nothing")
    weights = np.ones((40, 60), np.float32)
    weights[:, :30] = 0.0
    inside = frame.atlas["annotation"] > 0
    assert _blocks.estimated_block(None, inside) is None
    estimated = _blocks.estimated_block(weights, inside)
    assert estimated is not None and 0.4 < estimated["left_out_fraction"] < 0.6
    assert "make_mask source fit" in estimated["sorting"]


def test_spomialign_drops_matches_by_the_section_mask_on_its_canvas():
    import numpy as np

    from langslice.engines.spomialign import plugin

    frame = _fake_frame()
    weights = np.ones((40, 60), np.float32)
    weights[:, :20] = 0.0
    pair = plugin._inputs(frame, frame.atlas["template"], False, weights)
    keep = pair["section_keep"]
    assert keep.shape == pair["target"].shape[:2]
    assert not keep[:, : keep.shape[1] // 6].any() and keep[:, keep.shape[1] // 2 :].all()
    assert plugin._inputs(frame, frame.atlas["template"], False)["section_keep"] is None
    assert pair["work_um"] > 0
