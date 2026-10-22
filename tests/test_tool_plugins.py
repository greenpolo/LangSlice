"""Tool plugins: loading, gating, writes, layers, doors, provenance, invariance,
background work and the SECTION_BUSY refusal.

The example plugin (``tests/fixtures/example_plugin.py``: a read tool and a
write tool) and small plugins written per test go through the same path a
real one does: a ``module:factory`` reference in ``JobSpec.plugins``, the tool
door's wrappers, the operations behind the context's writes.
"""

from __future__ import annotations

import inspect
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.core import plugins as core_plugins
from langslice.core.plugins import Layer, PluginError
from langslice.core.spec import JobSpec, TransformSpec
from langslice.core.toolset import tools_for, valid_names
from langslice.doors.declarations import FULL, Variant, declaration
from langslice.doors.tools.toolbox import build_tools
from langslice.ops.registry import enabled, verbs_for
from tests.linear_tool_helpers import ToolContext, stack, tool_named

EXAMPLE = "tests.fixtures.example_plugin:tools"

HELLO = '''
from langslice.plugin_api import ToolResult, ToolSpec

def hello_stub():
    """Say hello. Writes nothing."""
    ...

def hello(ctx):
    return ToolResult({"hello": True})

def tools():
    return [ToolSpec("hello", hello_stub, hello, kind="read", group="Common")]
'''


@pytest.fixture(autouse=True)
def clean_registry():
    """Plugins loaded by one test are not loaded for the next."""
    held = dict(core_plugins._LOADED)
    yield
    core_plugins._LOADED.clear()
    core_plugins._LOADED.update(held)


@pytest.fixture
def make(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``make(name, source)``: a plugin module on the path; its reference."""
    folder = tmp_path / "plugin_modules"
    folder.mkdir()
    monkeypatch.syspath_prepend(str(folder))

    def write(name: str, source: str, factory: str = "tools") -> str:
        (folder / f"{name}.py").write_text(textwrap.dedent(source), encoding="utf-8")
        sys.modules.pop(name, None)
        return f"{name}:{factory}"

    return write


@pytest.fixture
def images(tmp_path: Path) -> Path:
    folder = tmp_path / "images"
    folder.mkdir()
    return folder


def box_of(images: Path, *plugins: str, n: int = 3, **spec: Any):
    spec.setdefault("tasks", ["position", "transform"])
    state, ctx, jobspec = stack(images, n, placed=True, plugins=list(plugins), **spec)
    return state, ctx, build_tools(state, ctx, jobspec)


def call(toolbox, name: str, /, **arguments: Any) -> dict[str, Any]:
    """Call a tool as ADK does: with its tool context where it takes one."""
    tool = tool_named(toolbox, name)
    if "tool_context" in inspect.signature(tool).parameters:
        arguments["tool_context"] = ToolContext()
    return tool(**arguments)


# --- loading ------------------------------------------------------------------------------------


def test_a_valid_plugin_adds_its_tools_after_the_native_ones(images: Path):
    state, ctx, box = box_of(images, EXAMPLE)
    names = box.names
    assert names[-2:] == ["section_brightness", "nudge_sections"]
    assert names[:-2] == enabled(box.job.spec)[:-2]
    assert tools_for(box.job.spec)["nudge_sections"].writes == ("affine",)
    assert verbs_for(box.job.spec)["nudge_sections"].limits == {"sections": 4}
    assert "section_brightness" in valid_names(box.job.spec)
    assert "section_brightness" not in valid_names()


def test_the_spec_saves_its_plugins_and_leaves_the_key_out_without_any(make, images: Path):
    ref = make("configurable", HELLO.replace("def tools():", "def tools(config=None):"))
    spec = JobSpec(image_folder=str(images), plugins=[EXAMPLE, ref],
                   plugin_config={ref: {"note": "x"}})
    held = spec.to_dict()
    assert held["plugins"] == [EXAMPLE, ref] and held["plugin_config"] == {ref: {"note": "x"}}
    assert JobSpec.from_dict(held).plugins == [EXAMPLE, ref]
    plain = JobSpec(image_folder=str(images)).to_dict()
    assert "plugins" not in plain and "plugin_config" not in plain


SOURCES = {
    "factory missing": "x = 1\n",
    "factory raises": "def tools():\n    raise RuntimeError('boom')\n",
    "factory returns a number": "def tools():\n    return 3\n",
    "not a ToolSpec": "def tools():\n    return [1]\n",
}


def _decl(name: str = '"thing"', kind: str = '"read"', group: str = '"Common"',
          **extra: str) -> str:
    """A plugin module with one tool declared with the given keyword values."""
    more = "".join(f", {key}={value}" for key, value in extra.items())
    return textwrap.dedent(f'''
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            """Do a thing."""
        def handler(ctx):
            return ToolResult({{}})
        def tools():
            return [ToolSpec({name}, stub, handler, kind={kind}, group={group}{more})]
    ''')


DECLARATIONS = {
    "native name": {"name": '"look"'},
    "not an identifier": {"name": '"2bad"'},
    "unknown task": {"task": '"bogus"'},
    "write without properties": {"kind": '"write"', "writes": '("wings",)'},
    "read that writes": {"writes": '("affine",)'},
    "other api version": {"api_version": "2"},
    "unknown group": {"group": '"Elsewhere"'},
    "zero limit": {"limits": '{"sections": 0}'},
}


@pytest.mark.parametrize("case", [
    "no such module", "module api version", "same tool twice", "config of no plugin",
    *SOURCES, *DECLARATIONS])
def test_a_bad_plugin_is_refused_in_one_line_when_the_spec_is_made(case: str, make, images: Path):
    config: dict[str, Any] = {}
    if case == "no such module":
        plugins = ["no_such_module_anywhere:tools"]
    elif case == "module api version":
        plugins = [make("old_api", HELLO + "\nPLUGIN_API_VERSION = 0\n")]
    elif case == "same tool twice":
        plugins = [make("twin_a", HELLO), make("twin_b", HELLO)]
    elif case == "config of no plugin":
        plugins, config = [], {"a:b": {}}
    elif case in SOURCES:
        plugins = [make("bad_source", SOURCES[case])]
    else:
        plugins = [make("bad_decl", _decl(**DECLARATIONS[case]))]
    with pytest.raises(ValueError) as caught:
        JobSpec(image_folder=str(images), plugins=plugins, plugin_config=config)
    assert "\n" not in str(caught.value)


def test_a_disabled_name_must_exist_and_a_plugin_tool_can_be_disabled(images: Path):
    with pytest.raises(ValueError, match="unknown name 'nudge_sections'"):
        JobSpec(image_folder=str(images), disabled=["nudge_sections"])
    spec = JobSpec(image_folder=str(images), plugins=[EXAMPLE], disabled=["nudge_sections"])
    assert "nudge_sections" not in enabled(spec)
    assert "section_brightness" in enabled(spec)


# --- gating -------------------------------------------------------------------------------------


def test_a_tool_is_absent_while_its_task_is_off(images: Path):
    _s, _c, box = box_of(images, EXAMPLE, tasks=["position"])
    assert "nudge_sections" not in box.names and "section_brightness" in box.names


def test_a_tool_that_writes_only_what_is_frozen_everywhere_is_not_offered(images: Path):
    _s, _c, box = box_of(images, EXAMPLE, frozen={"*": ["affine"]})
    assert "nudge_sections" not in box.names


def test_a_frozen_property_refuses_the_write_as_that_sections_row(images: Path):
    state, _c, box = box_of(images, EXAMPLE, frozen={"s1.png": ["affine"]})
    reply = call(box, "nudge_sections", sections=["s0.png", "s1.png"], dx=0.05)
    rows = {row["id"]: row for row in reply["rows"]}
    assert rows["s0.png"]["status"] == "ok"
    assert rows["s1.png"]["error"] == "FROZEN" and "affine" in rows["s1.png"]["properties"]
    assert reply["written"] == ["s0.png"]
    assert state.by_id("s1.png").transform is None and state.by_id("s0.png").transform


def test_a_per_call_limit_refuses_a_longer_call(images: Path):
    _s, _c, box = box_of(images, EXAMPLE, n=6)
    reply = call(box, "nudge_sections", sections=[f"s{i}.png" for i in range(5)])
    assert reply["error"] == "TOO_MANY_SECTIONS" and reply["max_sections"] == 4


# --- writes -------------------------------------------------------------------------------------


def test_a_context_write_is_one_undo_step_that_undo_reverts(images: Path):
    state, _c, box = box_of(images, EXAMPLE)
    job = box.job
    depth = len(job.undo_stack)
    reply = call(box, "nudge_sections", sections=["s0.png", "s1.png", "s2.png"], dx=0.05)
    assert reply["status"] == "ok" and len(reply["written"]) == 3
    assert len(job.undo_stack) == depth + 1
    registration = json.loads((job.layout.folder / "registration.json").read_text())
    shifts = [item["parameters"]["affine"]["params"][2] for item in registration["sections"]]
    assert shifts == [0.05, 0.05, 0.05]
    assert all(state.by_id(f"s{i}.png").transform["kind"] == "plugin:nudge_sections"
               for i in range(3))
    assert call(box, "undo")["status"] == "ok"
    assert all(state.by_id(f"s{i}.png").transform is None for i in range(3))
    registration = json.loads((job.layout.folder / "registration.json").read_text())
    assert all((item["parameters"].get("affine") or {}).get("kind") != "plugin:nudge_sections"
               for item in registration["sections"])


def test_a_read_tool_cannot_write_and_a_write_only_what_it_declared(make, images: Path):
    ref = make("sneaky", '''
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub(kind: str):
            """Try to write."""
        def handler(ctx, kind):
            rows = [{"id": "s0.png", "params": [1, 0, 0.1, 0, 1, 0]}]
            if kind == "positions":
                return ToolResult(ctx.set_positions([{"id": "s0.png", "position_mm": 4}]).reply())
            return ToolResult(ctx.set_transforms(rows).reply())
        def tools():
            return [ToolSpec("reader", stub, handler, kind="read", group="Common"),
                    ToolSpec("writer", stub, handler, kind="write", group="Common",
                             writes=("affine",))]
    ''')
    state, _c, box = box_of(images, ref)
    assert call(box, "reader", kind="transform")["error"] == "READ_TOOL_WRITES"
    assert call(box, "writer", kind="positions")["error"] == "WRITE_NOT_DECLARED"
    assert state.by_id("s0.png").transform is None and state.by_id("s0.png").position_mm == 2.0


LONG = '''
from langslice.plugin_api import ToolResult, ToolSpec
HOOK = lambda: None

def stub(sections: list[str]):
    """Fit slowly."""

def handler(ctx, sections):
    expect = {name: ctx.section_inputs(name) for name in sections}
    HOOK()
    rows = [{"id": name, "params": [1, 0, 0.1, 0, 1, 0]} for name in sections]
    return ToolResult(ctx.set_transforms(rows, expect=expect).reply())

def tools():
    return [ToolSpec("slow_fit", stub, handler, kind="write", group="Linear", task="transform",
                     writes=("affine",), long=True)]
'''


def test_a_long_tool_computes_outside_the_lock_and_a_moved_section_is_refused(make, images: Path):
    ref = make("slow_plugin", LONG)
    state, ctx, box = box_of(images, ref)
    module = sys.modules["slow_plugin"]
    from langslice.ops import positions

    module.HOOK = lambda: positions.position_sections(
        box.job, ctx, [{"id": "s1.png", "position_mm": 6.0}])
    reply = call(box, "slow_fit", sections=["s0.png", "s1.png"])
    rows = {row["id"]: row for row in reply["rows"]}
    assert rows["s0.png"]["status"] == "ok"
    assert rows["s1.png"]["error"] == "STALE_INPUT"
    assert state.by_id("s0.png").transform is not None and state.by_id("s1.png").transform is None


def test_positions_are_written_in_one_step_and_a_frozen_position_is_that_rows_error(
        make, images: Path):
    ref = make("placer", '''
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub(sections: list[str]):
            """Place sections."""
        def handler(ctx, sections):
            rows = [{"id": name, "position_mm": 7.0 + index} for index, name in enumerate(sections)]
            return ToolResult(ctx.set_positions(rows).reply())
        def tools():
            return [ToolSpec("place", stub, handler, kind="write", group="Positioning",
                             task="position", writes=("position",))]
    ''')
    state, _c, box = box_of(images, ref, frozen={"s2.png": ["position"]},
                            inputs={"positions": {"s2.png": 3.0}})
    depth = len(box.job.undo_stack)
    reply = call(box, "place", sections=["s0.png", "s1.png", "s2.png"])
    rows = {row["id"]: row for row in reply["rows"]}
    assert rows["s2.png"]["error"] == "FROZEN"
    assert reply["written"] == ["s0.png", "s1.png"]
    assert (state.by_id("s0.png").position_mm, state.by_id("s1.png").position_mm) == (7.0, 8.0)
    assert state.by_id("s2.png").position_mm == 3.0
    assert len(box.job.undo_stack) == depth + 1
    assert call(box, "undo")["status"] == "ok"
    assert state.by_id("s0.png").position_mm == 2.0


def test_several_context_writes_in_one_block_are_one_undo_step(make, images: Path):
    ref = make("grouped", '''
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            """Place and mark."""
        def handler(ctx):
            with ctx.writing():
                ctx.set_positions([{"id": "s0.png", "position_mm": 1.5}])
                ctx.set_transforms([{"id": "s0.png", "params": [1, 0, 0.1, 0, 1, 0]}])
                ctx.mark_damage([{"id": "s0.png", "regions": ["L"], "note": "tear"}])
            return ToolResult({})
        def tools():
            return [ToolSpec("both", stub, handler, kind="write", group="Common",
                             writes=("position", "affine"))]
    ''')
    from tests.test_linear_fit_affine_regions import HalvesAtlas

    state, _c, box = box_of(images, ref, atlas=HalvesAtlas())
    depth = len(box.job.undo_stack)
    reply = call(box, "both")
    assert reply["status"] == "ok", reply
    record = state.by_id("s0.png")
    assert record.position_mm == 1.5 and record.transform and record.damage_note == "tear"
    assert len(box.job.undo_stack) == depth + 1
    call(box, "undo")
    undone = state.by_id("s0.png")
    assert undone.position_mm == 2.0 and undone.transform is None and not undone.damage_note


def test_a_long_calls_writes_are_one_undo_step_unless_another_write_came_between(
        make, images: Path):
    ref = make("long_writer", '''
        from langslice.plugin_api import ToolResult, ToolSpec
        HOOK = None
        def stub():
            """Place twice."""
        def handler(ctx):
            ctx.set_positions([{"id": "s0.png", "position_mm": 1.5}])
            if HOOK is not None:
                HOOK()
            ctx.set_positions([{"id": "s1.png", "position_mm": 6.5}])
            ctx.mark_damage([{"id": "s0.png", "regions": ["L"], "note": "tear"}])
            return ToolResult({})
        def tools():
            return [ToolSpec("place_twice", stub, handler, kind="write", group="Common",
                             writes=("position",), long=True)]
    ''')
    state, ctx, box = box_of(images, ref)
    module = sys.modules["long_writer"]
    start = {record.id: record.position_mm for record in state.slices}
    depth = len(box.job.undo_stack)
    assert call(box, "place_twice")["status"] == "ok"
    assert len(box.job.undo_stack) == depth + 1  # three writes, no ctx.writing(): one step
    call(box, "undo")
    assert {record.id: record.position_mm for record in state.slices} == start
    assert not state.by_id("s0.png").damage_note

    from langslice.ops import positions

    module.HOOK = lambda: positions.position_sections(
        box.job, ctx, [{"id": "s2.png", "position_mm": 9.0}])
    depth = len(box.job.undo_stack)
    assert call(box, "place_twice")["status"] == "ok"
    # The other write sits between the call's: three steps; undo takes back the call's last.
    assert len(box.job.undo_stack) == depth + 3
    call(box, "undo")
    assert state.by_id("s1.png").position_mm == start["s1.png"]
    assert state.by_id("s2.png").position_mm == 9.0


def test_a_handler_that_raises_is_a_reply_not_a_crash(make, images: Path):
    ref = make("crashing", '''
        from langslice.plugin_api import ToolError, ToolResult, ToolSpec
        def stub(mode: str):
            """Fail."""
        def handler(ctx, mode):
            if mode == "refuse":
                raise ToolError("NOT_NOW", "not now", why="test")
            raise RuntimeError("bug")
        def tools():
            return [ToolSpec("fails", stub, handler, kind="read", group="Common")]
    ''')
    _s, _c, box = box_of(images, ref)
    assert call(box, "fails", mode="refuse") == {
        "status": "error", "error": "NOT_NOW", "message": "not now", "why": "test"}
    crashed = call(box, "fails", mode="bug")
    assert crashed["error"] == "PLUGIN_FAILED" and "RuntimeError: bug" in crashed["message"]


def test_the_fit_frame_carries_the_marked_damage_on_the_native_plane(make, images: Path):
    ref = make("damage_reader", '''
        import numpy as np
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            """Read the damage."""
        def handler(ctx):
            frame = ctx.fit_frame("s0.png")
            if frame.excluded is None:
                return ToolResult({"excluded": None})
            labels = frame.atlas["annotation"]
            return ToolResult({"excluded": int(frame.excluded.sum()),
                               "same_grid": list(frame.excluded.shape) == list(labels.shape),
                               "zone": bool((frame.damage_zone >= frame.excluded).all()
                                            and frame.damage_zone.sum() > frame.excluded.sum()),
                               "inside": bool((labels[frame.excluded] != 0).all())})
        def tools():
            return [ToolSpec("damage_of", stub, handler, kind="read", group="Common")]
    ''')
    from tests.test_linear_fit_affine_regions import HalvesAtlas

    state, _c, box = box_of(images, ref, atlas=HalvesAtlas())
    state.by_id("s0.png").position_mm = 1.0
    assert call(box, "damage_of")["excluded"] is None
    assert call(box, "mark_damage", section="s0.png", regions=["L"])["status"] == "ok"
    reply = call(box, "damage_of")
    assert reply["excluded"] > 0 and reply["same_grid"] and reply["inside"] and reply["zone"]


# --- the context's reads and conversions --------------------------------------------------------


def test_the_context_reads_sections_images_and_the_atlas(make, images: Path):
    ref = make("reader_plugin", '''
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            """Read."""
        def handler(ctx):
            view = ctx.section("s1")
            image = ctx.section_image("s1.png", level=32)
            atlas = ctx.atlas()
            frame = ctx.fit_frame("s1.png", stain=True)
            return ToolResult({
                "stain": list(frame.stain.shape) == list(frame.image.shape),
                "id": view.id, "position": view.position_mm, "size": list(view.file_size),
                "shape": list(image.shape), "atlas": atlas.name, "plane": atlas.plane,
                "planes": sorted(frame.atlas), "matrix": frame.atlas_to_section.shape[0],
                "placed": frame.placed_atlas_to_section is None,
                "work": str(ctx.work_dir()),
                "spec_tasks": list(ctx.spec["tasks"]),
                "layers": [r.name for r in ctx.layers("s1")]})
        def tools():
            return [ToolSpec("read_it", stub, handler, kind="read", group="Common")]
    ''')
    _s, ctx, box = box_of(images, ref)
    from tests.fakes import ellipse_section

    ellipse_section((40, 30), (14, 9)).save(images / "s1.png")
    ctx.source_cache.clear()
    ctx.render_cache.clear()
    reply = call(box, "read_it")
    assert reply["id"] == "s1.png" and reply["position"] == 2.5 and reply["size"] == [40, 30]
    assert reply["shape"][2] == 3 and reply["matrix"] == 3 and "annotation" in reply["planes"]
    assert reply["placed"]  # no transform yet: no current placement to start from
    assert reply["stain"]  # the preprocessed channel on the fit grid
    assert Path(reply["work"]).parts[-2:] == ("plugins", "read_it") and Path(reply["work"]).is_dir()
    assert reply["spec_tasks"] == ["position", "transform"] and reply["layers"] == []


def test_a_matrix_becomes_the_transform_an_elastix_fit_would_store(make, images: Path):
    ref = make("converter", '''
        import numpy as np
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            """Convert."""
        def handler(ctx):
            frame = ctx.fit_frame("s0.png")
            # The map that leaves the placement as it is: the inverse of the one
            # that puts the atlas plane on the section's grid.
            same = ctx.transform_from_matrix(
                "s0.png", np.linalg.inv(frame.atlas_to_section)[:2])
            moved = ctx.transform_from_params("s0.png", [1, 0, 0.1, 0, 1, 0])
            return ToolResult({"same": same.record, "moved": moved.record,
                               "row": moved.row()["id"]})
        def tools():
            return [ToolSpec("convert", stub, handler, kind="read", group="Common")]
    ''')
    _s, ctx, box = box_of(images, ref)
    from tests.fakes import ellipse_section

    ellipse_section((40, 30), (14, 9)).save(images / "s0.png")
    ctx.source_cache.clear()
    ctx.render_cache.clear()
    reply = call(box, "convert")
    np.testing.assert_allclose(reply["same"]["params"], [1, 0, 0, 0, 1, 0], atol=1e-9)
    assert reply["moved"]["params"][2] == 0.1 and reply["moved"]["physical"]["translate_x_mm"]
    assert set(reply["moved"]) >= {"kind", "params", "physical", "calibration", "mirrored"}
    assert reply["row"] == "s0.png"


# --- layers -------------------------------------------------------------------------------------


MARKS = '''
import numpy as np
from langslice.plugin_api import Layer, ToolResult, ToolSpec

def stub(sections: list[str]):
    """Leave marks on sections."""

def handler(ctx, sections):
    refs = []
    for name in sections:
        ref = ctx.add_layer(name, Layer.points([[10, 8], [20, 12]], labels=["a", "b"],
                                               name="marks"))
        ctx.add_layer(name, Layer.lines([[[5, 5], [30, 5], [30, 20]]], closed=True, name="box"))
        ctx.add_layer(name, Layer.labels(np.pad(np.ones((6, 8), int), 3),
                                         frame="section_px", name="regions"))
        ctx.add_layer(name, Layer.scalar(np.linspace(0, 1, 30).reshape(5, 6),
                                         frame="section_px", name="map"))
        ctx.add_layer(name, Layer.picture(np.zeros((8, 8, 3), np.uint8), name="shot"))
        refs.append(ref.id)
    return ToolResult({"layers": refs})

def tools():
    return [ToolSpec("mark", stub, handler, kind="write", group="Common")]
'''


def test_layers_are_state_listed_by_status_drawn_by_look_and_hidden_by_undo(make, images: Path):
    ref = make("marker_plugin", MARKS)
    state, _c, box = box_of(images, ref)
    depth = len(box.job.undo_stack)
    assert call(box, "mark", sections=["s0.png"])["status"] == "ok"
    assert len(box.job.undo_stack) == depth + 1  # the call, not each layer
    record = state.by_id("s0.png")
    assert [item["name"] for item in record.layers] == ["marks", "box", "regions", "map", "shot"]
    assert record.layers[0]["file"].startswith("plugins/mark/layers/s0_")
    manifest = json.loads((box.job.layout.folder / "plugins/mark/layers.json").read_text())
    assert [item["name"] for item in manifest["layers"]][:2] == ["marks", "box"]
    assert {item["tool"] for item in manifest["layers"]} == {"mark"}
    row = call(box, "status")["rows"][0]
    assert [item["name"] for item in row["layers"]] == ["marks", "box", "regions", "map", "shot"]
    assert "layers" not in call(box, "status")["rows"][1]

    plain = call(box, "look", mode="section", sections=["s0.png"])
    drawn = call(box, "look", mode="section", sections=["s0.png"], layers=["all"])
    notes = {item["name"]: item for item in drawn["layers"]}
    assert [notes[name]["drawn"] for name in ("marks", "box", "regions", "map")] == [True] * 4
    assert notes["shot"]["drawn"] is False and "finished picture" in notes["shot"]["reason"]
    before, after = (np.asarray(reply["images"][0]) for reply in (plain, drawn))
    assert before.shape == after.shape and (before != after).any()
    assert "layers" not in plain

    overlay = call(box, "look", mode="overlay", sections=["s0.png"], layers=["marks", "box"])
    assert [item["drawn"] for item in overlay["layers"]] == [True, True]

    undone = call(box, "undo")
    assert undone["status"] == "ok" and state.by_id("s0.png").layers == []
    assert call(box, "look", mode="section", layers=["marks"])["error"] == "UNKNOWN_PLUGIN_LAYER"
    assert call(box, "redo")["status"] == "ok"
    assert len(state.by_id("s0.png").layers) == 5
    first = [item["id"] for item in state.by_id("s0.png").layers]
    call(box, "mark", sections=["s0.png"])  # a layer of the same name replaces the earlier one
    second = [item["id"] for item in state.by_id("s0.png").layers]
    assert len(second) == 5 and not set(first) & set(second)
    assert call(box, "look", mode="positioning", layers=["marks"])["error"] == "BAD_ARGS"
    refused = call(box, "look", mode="section", layers=["nope"])
    assert refused["error"] == "UNKNOWN_PLUGIN_LAYER" and refused["layers"] == [
        "marks", "box", "regions", "map", "shot"]


def test_layer_files_round_trip_in_their_formats(tmp_path: Path):
    from langslice.core.plugin_layers import load_layer, save_layer

    cases = [
        Layer.points([[1.5, 2.5], [3, 4]], labels=["a", ""], values=[0.5, 2], name="p"),
        Layer.points([[1, 2, 3]], frame="atlas_um", name="q"),
        Layer.lines([[[0, 0], [1, 1], [2, 0]], [[5, 5], [6, 6]]], closed=[True, False],
                    labels=["tri", "seg"], name="l"),
        Layer.labels(np.arange(12).reshape(3, 4), frame="section_px", name="r"),
        Layer.labels(np.array([[0, 70000]]), frame="section_px", name="big"),
        Layer.scalar(np.array([[0.1, 0.9]]), frame="section_px", level=0.4, name="s"),
        Layer.picture(Image.new("RGB", (4, 3), "red"), name="pic"),
    ]
    for layer in cases:
        path = save_layer(layer, tmp_path / layer.name)
        back = load_layer(path, {"kind": layer.kind, "frame": layer.frame, "name": layer.name})
        assert back.kind == layer.kind and back.frame == layer.frame
        if layer.kind == "points":
            np.testing.assert_allclose(back.xy, layer.xy)
            assert back.tags == layer.tags and back.values == layer.values
        elif layer.kind == "lines":
            assert [closed for _p, closed in back.paths] == [closed for _p, closed in layer.paths]
            np.testing.assert_allclose(back.paths[0][0], layer.paths[0][0])
            assert back.tags == layer.tags
        elif layer.kind in ("labels", "scalar"):
            np.testing.assert_allclose(back.raster, layer.raster)
        if layer.kind == "scalar":
            assert back.level == pytest.approx(0.4)
        if layer.kind == "picture":
            assert back.image.size == (4, 3)
    assert (tmp_path / "pic.png").suffix == ".png" and (tmp_path / "p.csv").read_text(
        ).splitlines()[0] == "x,y,label,value"
    assert (tmp_path / "big.npz").exists() and (tmp_path / "r.png").exists()


@pytest.mark.parametrize("make_layer", [
    lambda: Layer.points([[1, 2, 3]]),
    lambda: Layer.points([[1, 2]], frame="atlas_um"),
    lambda: Layer.points([[float("nan"), 1]]),
    lambda: Layer.points([[1, 2]], labels=["a", "b"]),
    lambda: Layer.lines([[[0, 0]]]),
    lambda: Layer.labels(np.zeros((2, 2), float), frame="section_px"),
    lambda: Layer.labels(np.array([[-1]]), frame="section_px"),
    lambda: Layer.scalar(np.zeros(3), frame="section_px"),
    lambda: Layer.picture(None),
    lambda: Layer("points", "nowhere").check(),
])
def test_a_malformed_layer_is_refused_when_it_is_made(make_layer):
    from langslice.core.plugins import ToolError

    with pytest.raises(ToolError) as caught:
        make_layer()
    assert caught.value.code == "BAD_LAYER"


def test_export_maps_copies_the_layers_with_their_manifest(make, images: Path):
    ref = make("marker_export", MARKS)
    state, ctx, jobspec = stack(images, 2, placed=True, plugins=[ref],
                                tasks=["position", "transform"])
    box = build_tools(state, ctx, jobspec, scripting=True)
    call(box, "mark", sections=["s0.png"])
    from langslice.ops import exports

    exported = exports.export_maps(box.job, ctx, ids=["s0.png"])
    target = box.job.layout.exports_dir / "plugins" / "mark"
    assert (target / "layers.json").is_file()
    listed_files = json.loads((target / "layers.json").read_text())["layers"]
    assert {item["name"] for item in listed_files} == {"marks", "box", "regions", "map", "shot"}
    assert all((target / item["file"]).is_file() for item in listed_files)
    assert any(kind == "plugin_layer" for _path, kind in exported.files)


# --- deformations -------------------------------------------------------------------------------


NONLINEAR = {"tasks": ["position", "transform", "nonlinear"]}


def warp_box(images: Path, *plugins: str, transforms: tuple[str, ...] = ("s0.png", "s1.png"),
             **spec: Any):
    """A box whose stack can take deformations; *transforms* get an in-plane shift."""
    from langslice.core.spec import NonlinearSpec

    state, ctx, box = box_of(images, EXAMPLE, *plugins, nonlinear=NonlinearSpec(provider="none"),
                             **{**NONLINEAR, **spec})
    if transforms:
        assert call(box, "nudge_sections", sections=list(transforms), dx=0.05)["status"] == "ok"
    return state, ctx, box


def test_a_written_map_comes_back_through_the_maps_and_the_coordinate_export(images: Path):
    from langslice.core.maps import native_points, section_frame
    from langslice.ops.exports import export_maps
    from tests.fixtures.example_plugin import ripple_map

    state, ctx, box = warp_box(images)
    job = box.job
    reply = call(box, "ripple_sections", sections=["s0.png"], amplitude_px=1.5)
    assert reply["written"] == ["s0.png"] and reply["rows"][0]["status"] == "ok", reply
    assert reply["rows"][0]["displacement_mm"]["max"] > 0.5
    record = state.by_id("s0.png")
    held = record.deformation
    assert held["steps"] == [{"method": "plugin:ripple_sections", "start": "linear",
                              "note": "ripple 1.5 px"}]
    warp = job.deformations.current(state, record)
    assert warp is not None and warp.engine["name"] == "plugin:ripple_sections"
    assert warp.provenance["plugin"]["tool"] == "ripple_sections"
    assert warp.provenance["plugin"]["note"] == "ripple 1.5 px"
    # The map as the plugin computed it: the ripple on the section's current
    # placement (a shift), ctx.fit_frame's placed_atlas_to_section.
    from types import SimpleNamespace

    from langslice.core import deformation

    grid = deformation.fit_grid(state, ctx, record)
    expected = ripple_map(SimpleNamespace(
        image=np.asarray(grid.image), atlas_to_section=None,
        placed_atlas_to_section=grid.placement.atlas_to_section), 1.5)
    assert record.transform["params"][2] == 0.05
    # The ripple alone: 1.5 fit-grid pixels at most, nothing of the placement.
    assert reply["rows"][0]["displacement_mm"]["max"] <= 1.5 * warp.mm_per_px + 1e-3
    frame = section_frame(state, ctx, record)
    assert tuple(frame.file_size) == (40, 30) and frame.rotation_deg == 0 and not frame.flip
    gy, gx = np.indices((30, 40), dtype=np.float64)
    nx, ny = native_points(frame, warp, gx, gy)
    # Within the record's float16 storage of the field (well under 1e-3 native px).
    np.testing.assert_allclose(np.stack([nx, ny], -1), expected, atol=1e-3)
    # The exported coordinate map is those native points in atlas micrometres.
    exported = export_maps(job, ctx, ids=["s0.png"])
    assert "s0.png" in exported.sections, exported
    import tifffile

    coords = np.moveaxis(tifffile.imread(job.layout.section_dir("s0.png") / "coords.tif"), 0, -1)
    native_um = (frame.native_to_um[:, 0] * expected[..., 1, None]
                 + frame.native_to_um[:, 1] * expected[..., 0, None] + frame.native_to_um[:, 2])
    inside = np.isfinite(coords).all(axis=-1)
    assert inside.mean() > 0.5
    # Within the field's float16 storage (under half a micrometre here).
    np.testing.assert_allclose(coords[inside], native_um[inside], rtol=1e-5, atol=0.5)
    registration = json.loads((job.layout.folder / "registration.json").read_text())
    entry = next(item for item in registration["sections"] if item["id"] == "s0.png")
    assert entry["parameters"]["deformation"]["steps"][0]["method"] == "plugin:ripple_sections"


def test_a_deformation_write_is_one_undo_step_and_the_overlay_shows_it(images: Path):
    state, _c, box = warp_box(images)
    job = box.job
    depth = len(job.undo_stack)
    reply = call(box, "ripple_sections", sections=["s0.png", "s1.png"])
    assert reply["written"] == ["s0.png", "s1.png"]
    assert len(job.undo_stack) == depth + 1
    assert "deformation drawn" in reply["pictures"][0]["caption"]
    rows = {row["id"]: row for row in call(box, "status")["rows"]}
    assert rows["s0.png"]["deformation"]["steps"] == ["plugin:ripple_sections"]
    key = state.by_id("s0.png").deformation["key"]
    assert call(box, "undo")["status"] == "ok"
    assert state.by_id("s0.png").deformation is None and state.by_id("s1.png").deformation is None
    registration = json.loads((job.layout.folder / "registration.json").read_text())
    assert all(item["parameters"]["deformation"] is None for item in registration["sections"])
    assert call(box, "redo")["status"] == "ok"
    assert state.by_id("s0.png").deformation["key"] == key


def test_a_frozen_deformation_is_that_sections_row_and_the_others_are_written(images: Path):
    state, _c, box = warp_box(images, frozen={"s1.png": ["deformation"]})
    reply = call(box, "ripple_sections", sections=["s0.png", "s1.png"])
    rows = {row["id"]: row for row in reply["rows"]}
    assert rows["s1.png"]["error"] == "FROZEN" and rows["s0.png"]["status"] == "ok"
    assert reply["written"] == ["s0.png"] and state.by_id("s1.png").deformation is None


def test_a_section_without_a_transform_is_refused_as_a_native_fit_refuses_it(images: Path):
    state, _c, box = warp_box(images, transforms=("s0.png",))
    reply = call(box, "ripple_sections", sections=["s0.png", "s1.png"])
    rows = {row["id"]: row for row in reply["rows"]}
    assert rows["s1.png"]["error"] == "INVALID_LINEAR_PLACEMENT"
    assert reply["written"] == ["s0.png"] and state.by_id("s1.png").deformation is None


def test_a_placement_change_clears_the_deformation_and_undo_brings_both_back(images: Path):
    state, _c, box = warp_box(images)
    call(box, "ripple_sections", sections=["s0.png"])
    key = state.by_id("s0.png").deformation["key"]
    reply = call(box, "nudge_sections", sections=["s0.png"], dx=0.08)
    assert reply["deformation_cleared"] == ["s0.png"]
    assert state.by_id("s0.png").deformation is None
    call(box, "undo")
    assert state.by_id("s0.png").deformation["key"] == key


FIELDS = '''
import numpy as np
from langslice.plugin_api import ToolResult, ToolSpec
HOOK = lambda: None

def stub(sections: list[str], mode: str = "good"):
    """Write deformations."""

def handler(ctx, sections, mode="good"):
    from tests.fixtures.example_plugin import ripple_map
    rows = []
    for name in sections:
        field = ripple_map(ctx.fit_frame(name), 1.0)
        if mode == "shape":
            field = field[1:]
        elif mode == "nan":
            field[3, 4, 0] = np.nan
        elif mode == "int":
            field = field.astype(np.int32)
        rows.append({"id": name, "map": field, "meta": {"engine": "synthetic"}})
    if mode == "long":
        expect = {name: ctx.section_inputs(name) for name in sections}
        proposals = [ctx.deformation_from_map(row["id"], row["map"]).row() for row in rows]
        HOOK()
        return ToolResult(ctx.set_deformations(proposals, expect=expect).reply())
    if mode == "proposal":
        proposal = ctx.deformation_from_map(sections[0], rows[0]["map"])
        picture = ctx.overlay_picture(sections[0], proposal={"deformation": proposal})
        return ToolResult({"summary": proposal.summary,
                           "inverse": proposal.inverse_consistency_mm}, [picture])
    return ToolResult(ctx.set_deformations(rows).reply())

def tools():
    return [ToolSpec("warp_it", stub, handler, kind="write", group="Nonlinear",
                     task="nonlinear", writes=("deformation",), long=True),
            ToolSpec("warp_undeclared", stub, handler, kind="write", group="Nonlinear",
                     task="nonlinear", writes=("affine",))]
'''


@pytest.mark.parametrize("mode", ["shape", "nan", "int"])
def test_a_map_that_cannot_be_stored_is_that_sections_row_bad_field(make, images: Path,
                                                                       mode: str):
    state, _c, box = warp_box(images, make("fields_plugin", FIELDS))
    depth = len(box.job.undo_stack)
    reply = call(box, "warp_it", sections=["s0.png"], mode=mode)
    assert reply["rows"][0]["error"] == "BAD_FIELD" and reply["written"] == [], reply
    assert state.by_id("s0.png").deformation is None and len(box.job.undo_stack) == depth


def test_a_tool_that_does_not_declare_deformation_cannot_write_one(make, images: Path):
    state, _c, box = warp_box(images, make("fields_plugin", FIELDS))
    reply = call(box, "warp_undeclared", sections=["s0.png"])
    assert reply["error"] == "WRITE_NOT_DECLARED"
    assert state.by_id("s0.png").deformation is None


def test_a_long_tool_writes_its_proposals_and_a_moved_section_is_stale(make, images: Path):
    ref = make("fields_plugin", FIELDS)
    state, ctx, box = warp_box(images, ref)
    from langslice.ops import positions

    sys.modules["fields_plugin"].HOOK = lambda: positions.position_sections(
        box.job, ctx, [{"id": "s1.png", "position_mm": 2.6}])
    reply = call(box, "warp_it", sections=["s0.png", "s1.png"], mode="long")
    rows = {row["id"]: row for row in reply["rows"]}
    assert rows["s0.png"]["status"] == "ok" and rows["s1.png"]["error"] == "STALE_INPUT"
    assert state.by_id("s0.png").deformation and state.by_id("s1.png").deformation is None


CHAINED = '''
import numpy as np
from langslice.plugin_api import ToolResult, ToolSpec

def stub(sections: list[str], amplitude_px: float = 1.0):
    """Ripple sections on top of the deformation each holds."""

def handler(ctx, sections, amplitude_px=1.0):
    import cv2
    from tests.fixtures.example_plugin import ripple_map
    rows = []
    for name in sections:
        frame = ctx.fit_frame(name)
        held = frame.applied_map
        field = ripple_map(frame, amplitude_px)
        start = "linear"
        if held is not None:  # the ripple read through the deformation the section holds
            placed = np.linalg.inv(frame.placed_atlas_to_section)
            grid = np.einsum("ij,hwj->hwi", np.linalg.inv(placed)[:2],
                             np.concatenate([field, np.ones(field.shape[:2] + (1,))], -1))
            applied = np.asarray(held, np.float32)
            field = np.stack([cv2.remap(np.ascontiguousarray(applied[..., c]),
                                        grid[..., 0].astype(np.float32),
                                        grid[..., 1].astype(np.float32), cv2.INTER_LINEAR,
                                        borderMode=cv2.BORDER_REPLICATE)
                              for c in range(2)], -1).astype(np.float64)
            start = "current"
        rows.append(ctx.deformation_from_map(
            name, field, note=f"ripple {amplitude_px} on {start}", start=start,
            settings={"amplitude_px": amplitude_px},
            engine_settings={"wavelength_px": "grid height"},
            engine_version="ripple 1").row())
    return ToolResult(ctx.set_deformations(rows).reply())

def tools():
    return [ToolSpec("ripple_chain", stub, handler, kind="write", group="Nonlinear",
                     task="nonlinear", writes=("deformation",), long=True)]
'''


def test_a_warp_built_on_the_held_deformation_keeps_its_chain(make, images: Path):
    """A plugin's warp made on top of the deformation a section holds is that
    deformation's child: the section's steps stay before its own, its record
    names the held record as its parent, and its settings and the engine's
    numbers are kept as a FireANTs record keeps them."""
    state, _c, box = warp_box(images, make("chain_plugin", CHAINED))
    job = box.job
    first = call(box, "ripple_chain", sections=["s0.png"], amplitude_px=1.0)
    assert first["written"] == ["s0.png"], first
    held = state.by_id("s0.png").deformation
    assert [step["start"] for step in held["steps"]] == ["linear"]
    second = call(box, "ripple_chain", sections=["s0.png"], amplitude_px=0.5)
    assert second["written"] == ["s0.png"], second
    chained = state.by_id("s0.png").deformation
    assert [(step["method"], step["start"]) for step in chained["steps"]] == [
        ("plugin:ripple_chain", "linear"), ("plugin:ripple_chain", "current")]
    assert chained["steps"][1]["settings"] == {"amplitude_px": 0.5}
    record = job.deformations.current(state, state.by_id("s0.png"))
    assert record is not None and record.parent is not None and record.step == 1
    assert record.parent.saved_as == held["record"].rsplit("/", 1)[-1]
    assert record.engine["native_parameters"] == {"wavelength_px": "grid height"}
    assert record.engine["version"] == "ripple 1"
    assert record.provenance["start"] == "current"
    assert record.provenance["settings"] == {"amplitude_px": 0.5}
    # The record on disk names its parent, and reads back with it.
    saved = json.loads((job.layout.resolve(chained["record"]) / "record.json").read_text())
    assert saved["parent"] == held["record"].rsplit("/", 1)[-1]
    rows = {row["id"]: row for row in call(box, "status")["rows"]}
    assert rows["s0.png"]["deformation"] == {
        "method": "plugin:ripple_chain", "steps": ["plugin:ripple_chain", "plugin:ripple_chain"],
        "settings": {"amplitude_px": 0.5}}
    # Undo takes the second step back to the first.
    call(box, "undo")
    assert state.by_id("s0.png").deformation["key"] == held["key"]


PICTURED = '''
import numpy as np
from PIL import Image
from langslice.plugin_api import ToolResult, ToolSpec

def stub(sections: list[str]):
    """Ripple sections and show them."""

def handler(ctx, sections):
    from tests.fixtures.example_plugin import ripple_map
    rows = [{"id": name, "map": ripple_map(ctx.fit_frame(name), 1.0)} for name in sections]
    written = ctx.set_deformations(rows)
    name = sections[0]
    proposal = ctx.deformation_from_map(name, ripple_map(ctx.fit_frame(name), 2.0))
    pictures = [ctx.caption(ctx.overlay_picture(name), f"{name}: rippled"),
                ctx.caption(ctx.overlay_picture(name, proposal={"deformation": proposal}),
                            f"{name}: a ripple not written"),
                ctx.caption(Image.fromarray(np.zeros((20, 30, 3), np.uint8)),
                            f"{name}: a picture of my own",
                            sections=[name])]
    return ToolResult(written.reply(), pictures)

def tools():
    return [ToolSpec("ripple_shown", stub, handler, kind="write", group="Nonlinear",
                     task="nonlinear", writes=("deformation",))]
'''


def test_a_plugin_s_pictures_are_listed_with_the_registration_its_write_left(make,
                                                                             images: Path):
    """A plugin's picture of a section (an overlay, or one of its own named by
    ``ctx.caption(sections=)``) records the registration the section holds,
    so the registration list names it; a picture of a proposal not written
    records none."""
    state, _c, box = warp_box(images, make("shown_plugin", PICTURED))
    reply = call(box, "ripple_shown", sections=["s0.png"])
    assert reply["written"] == ["s0.png"], reply
    overlay, proposed, own = (entry["id"] for entry in reply["pictures"])
    number = reply["registration"]["s0.png"]
    listed = call(box, "restore_registration", section="s0.png")["registrations"]
    row = next(item for item in listed if item["registration"] == number)
    assert row["by"] == "ripple_shown" and row["pictures"] == [overlay, own]
    assert box.job.views.lookup(proposed).registrations == {}
    assert box.job.views.lookup(own).sections == ("s0.png",)


def test_a_proposed_deformation_is_drawn_and_not_written(make, images: Path):
    state, _c, box = warp_box(images, make("fields_plugin", FIELDS))
    depth = len(box.job.undo_stack)
    reply = call(box, "warp_it", sections=["s0.png"], mode="proposal")
    assert reply["status"] == "ok" and reply["summary"]["displacement_mm"]["max"] > 0
    assert reply["inverse"]["max"] < 0.01
    assert "deformation drawn" in reply["pictures"][0]["caption"]
    assert state.by_id("s0.png").deformation is None and len(box.job.undo_stack) == depth


# --- the doors ----------------------------------------------------------------------------------


def _library_job(images: Path, plugins: Any = (EXAMPLE,), **settings: Any):
    import langslice

    for index in range(2):
        Image.fromarray(np.full((30, 40, 3), 60 + 20 * index, np.uint8)).save(
            images / f"s{index}.png")
    return langslice.create_job(
        images, tasks=["position", "transform"], plugins=list(plugins), preprocess="none",
        positions={"s0.png": 2.0, "s1.png": 2.5}, atlas_loader=lambda _n: _slab(), **settings)


def _slab():
    from tests.linear_tool_helpers import SLAB

    return SLAB


def test_the_library_gets_the_tools_as_methods_and_reopens_them(images: Path):
    import langslice

    job = _library_job(images)
    assert job.verbs[-2:] == ["section_brightness", "nudge_sections"]
    reply = job.section_brightness()
    assert [row["mean"] for row in reply["rows"]] == [60.0, 80.0]
    written = job.nudge_sections(sections=["s0.png"], dx=0.03)
    assert written["written"] == ["s0.png"] and written["artifacts"]
    folder = job.folder
    job.close()
    core_plugins.forget(EXAMPLE)  # a new process starts without it
    again = langslice.open_job(folder, atlas_loader=lambda _n: _slab())
    assert again.verbs[-2:] == ["section_brightness", "nudge_sections"]
    assert again.state.by_id("s0.png").transform["params"][2] == 0.03
    again.close()


def test_the_library_takes_a_factory_function_and_tool_objects(make, images: Path):
    import langslice
    from tests.fixtures import example_plugin

    job = _library_job(images, plugins=[example_plugin.tools])
    assert job.job.spec.plugins == [EXAMPLE]  # a module-level factory is named by its path
    assert "section_brightness" in job.verbs
    job.close()
    second = images.parent / "second"
    second.mkdir()
    again = _library_job(second, plugins=example_plugin.tools())  # the tools themselves
    assert again.job.spec.plugins[0].startswith("inline:")
    assert "nudge_sections" in again.verbs
    again.close()
    with pytest.raises(PluginError, match="module level"):
        from langslice.doors.plugins import as_refs

        as_refs(lambda: [])
    assert langslice.open_job


def test_the_agent_cli_calls_a_plugin_tool_by_name(images: Path, capsys, monkeypatch):
    from langslice.doors.cli.jobcli import main

    job = _library_job(images)
    folder = str(job.folder)
    job.close()
    monkeypatch.setattr("langslice.core.atlas.core.load_atlas", lambda _n: _slab())
    from tests.cli_child import install

    install(lambda _n: _slab(), monkeypatch.setattr)
    code = main([folder, "section_brightness"])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and out["ok"] is True, out
    assert [row["mean"] for row in out["result"]["rows"]] == [60.0, 80.0]
    main([folder, "nudge_sections", "--args", json.dumps({"sections": ["s0.png"]})])
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result["ok"] and result["result"]["written"] == ["s0.png"], result
    main([folder, "no_such_tool"])
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]["code"] == \
        "UNKNOWN_VERB"


def test_ops_and_schema_list_the_plugin_tools_of_the_job_in_the_folder(
        images: Path, capsys, monkeypatch):
    from langslice.doors.cli.jobcli import main

    job = _library_job(images)
    folder = job.folder
    job.close()
    install_atlas = __import__("tests.cli_child", fromlist=["install"]).install
    install_atlas(lambda _n: _slab(), monkeypatch.setattr)
    monkeypatch.chdir(folder)
    main(["ops"])
    verbs = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]["verbs"]
    assert [item["name"] for item in verbs][-3:] == ["section_brightness", "nudge_sections",
                                                     "ripple_sections"]
    assert verbs[-2]["kind"] == "write" and verbs[-2]["group"] == "Linear"
    assert verbs[-1]["kind"] == "write" and verbs[-1]["group"] == "Nonlinear"
    main(["schema", "nudge_sections"])
    schema = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]
    assert set(schema["arguments"]["properties"]) == {"sections", "dx"}
    monkeypatch.chdir(images.parent)  # a folder with no job: the native listing
    main(["ops"])
    plain = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]["verbs"]
    assert [item["name"] for item in plain] == list(verbs_for())
    # From elsewhere, the job's folder names the job: its plugin verbs with
    # native ones, several at once.
    main([str(folder), "schema", "nudge_sections", "look"])
    both = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]["verbs"]
    assert list(both) == ["nudge_sections", "look"]
    assert set(both["nudge_sections"]["arguments"]["properties"]) == {"sections", "dx"}
    main(["schema", "nudge_sections"])  # no job here: how to name one
    refused = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["error"]
    assert refused["code"] == "UNKNOWN_VERB"
    assert "langslice-job FOLDER schema nudge_sections" in refused["message"]
    card = (Path(folder) / "AGENTS.md").read_text()
    assert "`nudge_sections` (write, Linear" in card and "`section_brightness` (read" in card


def test_the_cli_flag_names_plugins_and_their_options(make, images: Path):
    from langslice.doors.cli import build_parser
    from langslice.doors.cli.linear import build_linear_spec, with_tool_plugins

    parser = build_parser()
    configurable = make("configurable_flag", HELLO.replace("def tools():", "def tools(config):"))
    args = parser.parse_args(["linear", "run", str(images), "--tool-plugin", configurable,
                              "--tool-plugin-config", f'{configurable}={{"a": 1}}'])
    spec = with_tool_plugins(build_linear_spec(args, str(images)), args)
    assert spec.plugins == [configurable] and spec.plugin_config == {configurable: {"a": 1}}
    bare = parser.parse_args(["linear", "run", str(images)])
    assert with_tool_plugins(build_linear_spec(bare, str(images)), bare).plugins == []
    stray = parser.parse_args(["linear", "run", str(images), "--tool-plugin-config", "x=1"])
    with pytest.raises(ValueError, match="needs the --tool-plugin"):
        with_tool_plugins(build_linear_spec(stray, str(images)), stray)
    broken = parser.parse_args(["linear", "run", str(images), "--tool-plugin", "no_module:x"])
    with pytest.raises(ValueError, match="cannot import"):
        with_tool_plugins(build_linear_spec(broken, str(images)), broken)
    assert "--tool-plugin" not in parser.parse_args(["mcp"]).__dict__


def test_the_mcp_server_and_the_abba_worker_load_a_jobs_plugins(
        make, images: Path, tmp_path: Path):
    from langslice.doors.api.abba_worker import prepare_linear
    from langslice.doors.mcp.server import open_folder

    ref = make("greeter", """
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            \"\"\"Hi.\"\"\"
        def tools():
            return [ToolSpec("hi", stub, lambda ctx: ToolResult({}), kind="read",
                             group="Common")]
    """)
    job = _library_job(images, plugins=[ref])
    folder = job.folder
    job.close()
    core_plugins.forget(ref)
    session = open_folder(str(folder), lambda _p: JobSpec(image_folder=str(images)),
                          lambda _n: _slab())
    try:
        assert "hi" in session.box.names
    finally:
        session.close()
    Image.new("L", (80, 40)).save(tmp_path / "section_0001.tif")
    params = {"image_folder": str(tmp_path), "pixel_size_um": 25,
              "positions_mm": {"section_0001.tif": 4.0},
              "angles_deg": {"pitch_deg": 0.0, "yaw_deg": 0.0},
              "spec": {"tasks": ["position"], "plugins": [ref]}}
    assert prepare_linear(params).spec.plugins == [ref]


def test_a_cost_estimate_of_a_spec_with_plugins_loads_nothing(make, tmp_path: Path):
    from langslice.agent.cost import estimate

    marker = tmp_path / "ran.txt"
    ref = make("costly", f'''
        from pathlib import Path
        Path({str(marker)!r}).write_text("ran")
        def tools():
            return []
    ''')
    try:
        estimate({"plugins": [ref], "tasks": ["position"]}, 3, model="gpt-5.6-luna")
    except Exception:  # noqa: BLE001 - whatever the estimate says, no code ran
        pass
    assert not marker.exists()


# --- provenance ---------------------------------------------------------------------------------


def test_job_json_carries_the_plugins_provenance_and_a_changed_file_is_a_new_digest(
        make, images: Path):
    import hashlib

    from langslice.job.layout import read_job_file

    _s, _c, box = box_of(images, EXAMPLE)
    held = read_job_file(box.job.layout)["plugins"]
    assert len(held) == 1 and held[0] == box.plugins[0]
    record = held[0]
    assert record["ref"] == EXAMPLE and record["module"] == "tests.fixtures.example_plugin"
    assert record["factory"] == "tools" and record["api_version"] == 1
    path = Path(next(iter(record["sources"])))
    assert record["sources"][str(path)] == hashlib.sha256(path.read_bytes()).hexdigest()
    nudge = {tool["name"]: tool for tool in record["tools"]}["nudge_sections"]
    assert nudge["kind"] == "write" and nudge["task"] == "transform"
    assert nudge["writes"] == ["affine"] and nudge["limits"] == {"sections": 4}

    first = make("evolving", HELLO)
    spec = JobSpec(image_folder=str(images), plugins=[first])
    digest = next(iter(core_plugins.of_spec(spec)[0].provenance["sources"].values()))
    core_plugins.forget(first)
    make("evolving", HELLO + "\n# changed\n")
    changed = next(iter(core_plugins.of_spec(spec)[0].provenance["sources"].values()))
    assert digest != changed


def test_a_job_without_plugins_shows_no_trace_of_the_seam(images: Path):
    from langslice.job.layout import read_job_file

    state, ctx, spec = stack(images, 2, placed=True)
    box = build_tools(state, ctx, spec)
    assert box.plugins == [] and "plugins" not in (read_job_file(box.job.layout) or {})
    assert "layers" not in state.to_dict()["slices"][0] and "plugins" not in spec.to_dict()
    assert "layers" not in call(box, "status")["rows"][0]
    # look's layers draws what plugins and scripts leave: on every job.
    assert "layers" in inspect.signature(tool_named(box, "look")).parameters
    assert "layers" in declaration("look", FULL).signature.parameters
    with_plugins = JobSpec(image_folder=str(images), plugins=[EXAMPLE])
    declared = declaration("look", Variant.of(with_plugins, auto=True))
    assert "layers" in declared.signature.parameters


# --- frames: array pixels and section_px ---------------------------------------------------------


def test_many_points_through_a_warp_are_sampled_in_rows_not_one_column():
    from types import SimpleNamespace

    from langslice.core.plugin_layers import picture_to_fractions

    height, width = 48, 64
    field = np.zeros((height, width, 2))
    field[..., 0] = 0.005  # half a pixel of shift at 0.01 mm per pixel
    warp = SimpleNamespace(mm_per_px=0.01, section_size=(width, height), field_mm=field,
                           inverse_field_mm=field)
    panel = {"role": "section", "render_size": [width, height], "orientation": np.eye(3).tolist(),
             "pixel_to_render": np.eye(3).tolist()}
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    xy = np.stack([xs.ravel(), ys.ravel()], axis=1).astype(float)
    xy = np.tile(xy, (40, 1))  # 120 thousand points: past what one cv2 map column takes
    assert len(xy) > 100_000
    fractions = picture_to_fractions(panel, xy, warp)
    expected = (xy + np.array([0.5, 0.0]) + 0.5) / np.array([width, height])
    np.testing.assert_allclose(fractions, expected, atol=1e-6)


@pytest.mark.parametrize("rotation, flip", [(0, False), (0, True), (90, False), (270, True),
                                            (180, True)])
def test_section_raster_converts_pixels_both_ways(make, images: Path, rotation: int, flip: bool):
    ref = make(f"frames_{rotation}_{int(flip)}", '''
        import numpy as np
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub():
            """Convert."""
        def handler(ctx):
            raster = ctx.section_raster("s0.png", level=50)
            points = np.array([[3.0, 2.0], [20.0, 10.0]])
            back = raster.from_section_px(raster.to_section_px(points))
            return ToolResult({"back": back.tolist(), "shape": list(raster.array.shape),
                               "file": list(raster.file_size)})
        def tools():
            return [ToolSpec("convert_px", stub, handler, kind="read", group="Common")]
    ''')
    state, ctx, box = box_of(images, ref)
    Image.new("RGB", (200, 100)).save(images / "s0.png")
    record = state.by_id("s0.png")
    record.rotation_deg, record.flip = rotation, flip
    ctx.source_cache.clear()
    ctx.render_cache.clear()
    reply = call(box, "convert_px")
    np.testing.assert_allclose(reply["back"], [[3, 2], [20, 10]], atol=1e-6)
    assert reply["file"] == [200, 100] and max(reply["shape"][:2]) == 50


FEATURE = '''
import numpy as np
from langslice.plugin_api import Layer, ToolResult, ToolSpec

def stub():
    """Mark the red feature."""

def handler(ctx):
    raster = ctx.section_raster("s0.png", level=LEVEL, images=["red", "green", "blue"])
    red = (raster.array[..., 0] > 200) & (raster.array[..., 1] < 80)
    ys, xs = np.nonzero(red)
    centre = raster.to_section_px([[xs.mean(), ys.mean()]])
    ctx.add_layer("s0.png", Layer.points(centre, name="mark"))
    ctx.add_layer("s0.png", Layer.labels(red.astype(int), frame="oriented", name="mask"))
    return ToolResult({"centre": centre.tolist()})

def tools():
    return [ToolSpec("find_feature", stub, handler, kind="write", group="Common")]
'''


@pytest.mark.parametrize("level, rotation, flip", [(48, 0, True), (480, 0, False),
                                                   (64, 90, True), (96, 270, False)])
def test_a_mark_made_from_section_image_pixels_is_drawn_on_the_feature(
        make, images: Path, level: int, rotation: int, flip: bool):
    from tests.fakes import ellipse_section

    ref = make(f"feature_{level}_{rotation}_{int(flip)}", FEATURE.replace("LEVEL", str(level)))
    state, ctx, box = box_of(images, ref)
    picture = ellipse_section((400, 240), (150, 80))
    pixels = np.asarray(picture).copy()
    pixels[60:100, 270:330] = (230, 20, 20)  # the feature, right of centre and above
    Image.fromarray(pixels).save(images / "s0.png")
    record = state.by_id("s0.png")
    record.rotation_deg, record.flip = rotation, flip
    ctx.source_cache.clear()
    ctx.render_cache.clear()
    assert call(box, "find_feature")["status"] == "ok"

    def picture_of(**kwargs: Any) -> np.ndarray:
        return np.asarray(call(box, "look", mode="section", sections=["s0.png"],
                               images=["red", "green", "blue"], **kwargs)["images"][0])

    plain, drawn = picture_of(), picture_of(layers=["mark", "mask"])

    def centre(image: np.ndarray, colour: tuple[int, int, int], tolerance: int = 6) -> np.ndarray:
        hit = (np.abs(image.astype(int) - np.array(colour)).max(axis=-1) <= tolerance)
        ys, xs = np.nonzero(hit)
        assert len(xs) > 8, colour
        return np.array([xs.mean(), ys.mean()])

    feature = centre(plain, (230, 20, 20), 25)
    from langslice.core.plugin_layers import PALETTE

    def rgb(index: int) -> tuple[int, int, int]:
        value = PALETTE[index].lstrip("#")
        return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)

    size = max(plain.shape[:2])
    assert np.hypot(*(centre(drawn, rgb(0)) - feature)) < 0.03 * size  # the point
    assert np.hypot(*(centre(drawn, rgb(2)) - feature)) < 0.03 * size  # the mask's outline


# --- background work, SECTION_BUSY, the applied map, planes, the angles task ------------------


LATER = '''
import threading
from langslice.plugin_api import ToolResult, ToolSpec
from tests.fixtures.example_plugin import ripple_map

GATE = threading.Event()
SEEN = {}

def stub(sections: list[str]) -> dict:
    """Ripple sections in the background.

    Args:
        sections: Filenames.
    """

def handler(ctx, sections):
    expect = {name: ctx.section_inputs(name, deformation=True) for name in sections}
    SEEN["applied"] = {name: ctx.fit_frame(name).applied_map is not None for name in sections}

    def land(lctx):
        GATE.wait(10)
        rows = [{"id": name, "map": ripple_map(lctx.fit_frame(name), 1.0)} for name in sections]
        done = lctx.set_deformations(rows, expect=expect)
        image = lctx.overlay_picture(sections[0])
        return ToolResult({"notice": f"rippled {', '.join(done.written)}.",
                           "written": done.written}, [image])

    return ToolResult({"work": ctx.start_work(sections, land)})

def tools():
    return [ToolSpec("ripple_later", stub, handler, kind="write", group="Nonlinear",
                     task="nonlinear", writes=("deformation",), long=True)]
'''


@pytest.fixture
def later(make, images: Path):
    import sys

    ref = make("ripple_later_plugin", LATER)
    state, ctx, box = warp_box(images, ref)
    module = sys.modules["ripple_later_plugin"]
    module.GATE.clear()
    yield state, box, module
    module.GATE.set()
    box.job.background.wait_all()


def test_plugin_work_starts_at_once_lands_as_its_own_step_and_its_notice_comes_next(later):
    state, box, module = later
    job = box.job
    depth = len(job.undo_stack)
    reply = call(box, "ripple_later", sections=["s0.png"])
    assert reply["status"] == "ok" and reply["work"] == "w1"
    assert job.background.get("w1").status == "running"
    assert state.by_id("s0.png").deformation is None
    assert call(box, "status")["background_running"][0]["kind"] == "ripple_later"
    module.GATE.set()
    job.background.wait_all()
    assert state.by_id("s0.png").deformation is not None
    assert len(job.undo_stack) == depth + 1
    after = call(box, "status")
    notice = after["background"][0]
    assert notice.startswith("w1 ripple_later of s0.png finished: rippled s0.png.")
    assert any(item.get("work") == "w1" for item in after.get("pictures", []))
    call(box, "undo")
    assert state.by_id("s0.png").deformation is None


def test_a_write_to_a_section_with_work_running_is_refused_section_busy(later):
    state, box, module = later
    call(box, "ripple_later", sections=["s0.png"])
    busy = call(box, "mark_damage", section="s0.png", regions=[])
    assert busy["error"] == "SECTION_BUSY" and busy["work"] == ["w1"]
    assert "ripple_later is running on s0.png (work w1)" in busy["message"]
    nudged = call(box, "nudge_sections", sections=["s0.png", "s1.png"], dx=0.01)
    rows = {row["id"]: row for row in nudged["rows"]}
    assert rows["s0.png"]["error"] == "SECTION_BUSY" and rows["s1.png"]["status"] == "ok"
    again = call(box, "ripple_later", sections=["s0.png"])
    assert again["error"] == "SECTION_BUSY"
    angles = call(box, "position_sections", cutting_angles={"pitch_deg": 1.0, "yaw_deg": 0.0})
    assert angles["error"] == "SECTION_BUSY"  # one plane for the stack, s0 among it
    module.GATE.set()
    box.job.background.wait_all()
    assert state.by_id("s0.png").deformation is not None  # its own landing is not refused
    assert call(box, "mark_damage", section="s0.png", regions=[])["status"] == "ok"


PROBE = '''
import numpy as np
from langslice.plugin_api import ToolResult, ToolSpec
FRAMES = {}
def stub(section: str) -> dict:
    """Read a section's frame.

    Args:
        section: Filename.
    """
def handler(ctx, section):
    frame = FRAMES[section] = ctx.fit_frame(section)
    plane = frame.plane
    found = ctx.angles_from_plane(np.cross(plane["dx"], plane["dy"]), plane["center_um"])
    return ToolResult({"applied": frame.applied_map is not None, **found})
def tools():
    return [ToolSpec("frame_probe", stub, handler, kind="read", group="Common")]
'''


def test_the_fit_frame_carries_the_applied_deformation_as_a_native_map(make, images: Path):
    import sys

    from tests.fixtures.example_plugin import ripple_map

    ref = make("frame_probe_plugin", PROBE)
    _state, _c, box = warp_box(images, ref)
    frames = sys.modules["frame_probe_plugin"].FRAMES
    assert call(box, "frame_probe", section="s0.png")["applied"] is False
    call(box, "ripple_sections", sections=["s0.png"], amplitude_px=1.5)
    assert call(box, "frame_probe", section="s0.png")["applied"] is True
    held = frames["s0.png"]
    expected = ripple_map(held, 1.5)
    assert held.applied_map.shape == expected.shape
    assert np.abs(held.applied_map - expected).max() < 0.05


def test_a_plane_from_the_fit_frame_gives_back_the_stacks_placement(make, images: Path):
    ref = make("frame_probe_plugin", PROBE)
    state, _c, box = warp_box(images, ref)
    assert call(box, "position_sections",
                cutting_angles={"pitch_deg": 4.0, "yaw_deg": -3.0})["status"] == "ok"
    found = call(box, "frame_probe", section="s1.png")
    assert found["pitch_deg"] == pytest.approx(4.0, abs=1e-6)
    assert found["yaw_deg"] == pytest.approx(-3.0, abs=1e-6)
    assert found["position_mm"] == pytest.approx(state.by_id("s1.png").position_mm, abs=1e-6)


ANGLE_TOOL = '''
from langslice.plugin_api import ToolResult, ToolSpec
def stub():
    """Look at the angles. Writes nothing."""
def handler(ctx):
    return ToolResult({})
def tools():
    return [ToolSpec("angle_look", stub, handler, kind="read", group="Positioning",
                     task="angles")]
'''


def test_an_angles_tool_exists_whenever_the_agent_may_write_the_cutting_angles(make,
                                                                               images: Path):
    ref = make("angle_look_plugin", ANGLE_TOOL)
    _s, _c, box = box_of(images, ref)
    assert "angle_look" in enabled(box.job.spec)
    _s, _c, only_transform = box_of(images, ref, tasks=["transform"])
    assert "angle_look" not in enabled(only_transform.job.spec)
    _s, _c, angles_free = box_of(images, ref, tasks=["transform"],
                                 transform=TransformSpec(angles=True))
    assert "angle_look" in enabled(angles_free.job.spec)


MARK_LATER = '''
from langslice.plugin_api import ToolResult, ToolSpec
def stub(section: str) -> dict:
    """Move a section to 3.3 mm in the background.

    Args:
        section: Filename.
    """
def handler(ctx, section):
    def land(lctx):
        lctx.set_positions([{"id": section, "position_mm": 3.3}])
        return ToolResult({"notice": "moved."})
    return ToolResult({"work": ctx.start_work([section], land)})
def tools():
    return [ToolSpec("mark_later", stub, handler, kind="write", group="Common", long=True,
                     writes=("position",))]
'''


def test_the_agent_cli_waits_for_work_a_plugin_verb_started_and_brings_its_notice(
        make, images: Path, capsys, monkeypatch):
    import json

    from langslice.doors.cli.jobcli import main
    from tests.cli_child import install

    ref = make("mark_later_plugin", MARK_LATER)
    job = _library_job(images, plugins=[ref])
    folder = str(job.folder)
    job.close()
    monkeypatch.setattr("langslice.core.atlas.core.load_atlas", lambda _n: _slab())
    install(lambda _n: _slab(), monkeypatch.setattr)
    code = main([folder, "mark_later", "--args", json.dumps({"section": "s0.png"})])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert code == 0 and out["ok"] is True, out
    assert out["result"]["background"][0].endswith("mark_later of s0.png finished: moved.")
    import langslice

    again = langslice.open_job(folder, atlas_loader=lambda _n: _slab())
    assert again.state.by_id("s0.png").position_mm == pytest.approx(3.3)
    again.close()


def test_the_library_waits_for_work_a_plugin_verb_started_and_returns_its_landing(
        make, images: Path):
    ref = make("mark_later_plugin", MARK_LATER)
    job = _library_job(images, plugins=[ref])
    started = job.mark_later(section="s0.png")
    assert started["work"] == "w1"
    landed = job.wait()
    assert landed["status"] == "ok" and landed["running"] == []
    assert landed["background"][0].endswith("mark_later of s0.png finished: moved.")
    assert landed["work"][0]["id"] == "w1" and landed["work"][0]["status"] == "done"
    assert job.state.by_id("s0.png").position_mm == pytest.approx(3.3)
    assert job.wait()["background"] == []  # a notice is handed out once
    job.close()


def test_the_agent_cli_dry_run_of_a_plugin_tool_checks_and_runs_nothing(
        make, images: Path, capsys, monkeypatch):
    import json

    from langslice.doors.cli.jobcli import main
    from tests.cli_child import install

    ref = make("mark_later_plugin", MARK_LATER)
    job = _library_job(images, plugins=[ref])
    folder = str(job.folder)
    job.close()
    monkeypatch.setattr("langslice.core.atlas.core.load_atlas", lambda _n: _slab())
    install(lambda _n: _slab(), monkeypatch.setattr)
    main([folder, "mark_later", "--dry-run", "--args", json.dumps({"section": "s0.png"})])
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["ok"] is True and out["result"]["simulated"] is False, out
    assert "background" not in out["result"]
    import langslice

    again = langslice.open_job(folder, atlas_loader=lambda _n: _slab())
    assert again.state.by_id("s0.png").position_mm == pytest.approx(2.0)
    assert again.job.background.all() == []
    again.close()


# --- offered by atlas and plane -----------------------------------------------------------------

BY_ATLAS = '''
from langslice.plugin_api import ToolResult, ToolSpec

def mouse_stub():
    """Place from the mouse model. [[?rat_only Not for rats: use `rat_only`.]]"""

def rat_stub():
    """Place from the rat model."""

def plain_stub():
    """Look around. [[?mouse_only Try `mouse_only` first.]][[?rat_only Try `rat_only` too.]]"""

def handler(ctx):
    return ToolResult({})

def tools():
    return [
        ToolSpec("mouse_only", mouse_stub, handler, kind="read", group="Common",
                 atlases=("allen_mouse_*um", "ccfv3augmented_mouse_25um"), planes=("coronal",)),
        ToolSpec("rat_only", rat_stub, handler, kind="read", group="Common",
                 atlases=("whs_sd_rat_39um",)),
        ToolSpec("plain", plain_stub, handler, kind="read", group="Common"),
    ]
'''


@pytest.mark.parametrize("atlas, plane, expected", [
    ("allen_mouse_25um", "coronal", {"mouse_only", "plain"}),
    ("allen_mouse_10um", "coronal", {"mouse_only", "plain"}),
    ("ccfv3augmented_mouse_25um", "coronal", {"mouse_only", "plain"}),
    ("allen_mouse_25um", "sagittal", {"plain"}),
    ("whs_sd_rat_39um", "coronal", {"rat_only", "plain"}),
    ("kim_unified_25um", "coronal", {"plain"}),
])
def test_a_tool_is_offered_only_on_its_atlases_and_planes_and_no_text_names_it(
        make, images: Path, atlas: str, plane: str, expected: set[str]):
    ref = make("by_atlas", BY_ATLAS)
    spec = JobSpec(image_folder=str(images), plugins=[ref], atlas=atlas, plane=plane)
    names = {"mouse_only", "rat_only", "plain"}
    assert names & set(enabled(spec)) == expected
    assert names & set(tools_for(spec)) == names  # defined everywhere, on only where offered
    assert set(spec.switched_off) >= names - expected
    from langslice.core.needs import render

    for name in expected:
        text = declaration(name, _variant(spec)).doc
        for absent in names - expected:
            assert f"`{absent}`" not in text, (name, absent, text)
    plain_text = declaration("plain", _variant(spec)).doc
    for present in expected - {"plain"}:
        assert f"`{present}`" in plain_text  # the sentence is there where the tool is
    assert render("[[?mouse_only `mouse_only`]]", spec.switched_off) == (
        "`mouse_only`" if "mouse_only" in expected else "")


def _variant(spec: JobSpec) -> Variant:
    return Variant.of(spec, auto=False)


TASKED = '''
from langslice.plugin_api import ToolResult, ToolSpec

def tasked_stub():
    """Say hello.[[?elastix_affine  It also fits transforms.]] Done."""

def tasked(ctx):
    return ToolResult({})

def tools():
    return [ToolSpec("tasked", tasked_stub, tasked, kind="read", group="Common")]
'''


def test_a_plugin_sentence_naming_a_tool_of_a_task_that_is_off_is_dropped(make, images: Path):
    ref = make("tasked_plugin", TASKED)
    for tasks, disabled, named in ((["position", "transform"], [], True),
                                   (["position"], [], False),
                                   (["position", "transform"], ["elastix_affine"], False)):
        spec = JobSpec(image_folder=str(images), plugins=[ref], tasks=tasks, disabled=disabled)
        text = declaration("tasked", _variant(spec)).doc
        assert ("It also fits transforms." in text) is named, (tasks, disabled, text)
    # the native declarations keep their pinned wording whatever the tasks
    spec = JobSpec(image_folder=str(images), plugins=[ref], tasks=["position"])
    assert "elastix_affine" in declaration("mark_damage", _variant(spec)).doc


def test_atlas_and_plane_patterns_are_checked_and_recorded(make, images: Path):
    bad = make("bad_atlases", BY_ATLAS.replace('planes=("coronal",)', 'planes=("axial",)'))
    with pytest.raises(ValueError, match="planes are among"):
        JobSpec(image_folder=str(images), plugins=[bad])
    ref = make("by_atlas_prov", BY_ATLAS)
    spec = JobSpec(image_folder=str(images), plugins=[ref])
    from langslice.doors.plugins.provenance import records

    tools = {item["name"]: item for item in records(_variant(spec))[0]["tools"]}
    assert tools["mouse_only"]["atlases"] == ["allen_mouse_*um", "ccfv3augmented_mouse_25um"]
    assert tools["mouse_only"]["planes"] == ["coronal"]
    assert "atlases" not in tools["plain"] and "planes" not in tools["plain"]


# --- the positioning picture, the raw section, a registration read ----------------------------

PICTURES = '''
import numpy as np
from langslice.plugin_api import ToolResult, ToolSpec

def stub(proposed: bool = False, part: int | None = None):
    """Draw."""

def handler(ctx, proposed=False, part=None):
    ids = ["s0.png", "s1.png", "s2.png"]
    proposal = {"s0.png": {"position_mm": 6.0, "angles": (4.0, -3.0)}} if proposed else None
    images = ctx.positioning_picture(ids, proposal=proposal, part=part)
    return ToolResult({"count": len(images), "sizes": [list(im.size) for im in images]}, images)

def tools():
    return [ToolSpec("draw", stub, handler, kind="read", group="Common")]
'''


def test_positioning_picture_draws_the_stack_and_a_proposal_without_writing(make, images: Path):
    ref = make("pictures", PICTURES)
    state, _c, box = box_of(images, ref)
    before = state.to_dict()
    depth = len(box.job.undo_stack)
    plain = call(box, "draw")
    drawn = call(box, "draw", proposed=True)
    assert plain["status"] == drawn["status"] == "ok" and plain["count"] == drawn["count"] == 1
    assert state.to_dict() == before and len(box.job.undo_stack) == depth
    # the proposal's picture carries one more caption band
    assert drawn["sizes"][0][0] == plain["sizes"][0][0]
    assert drawn["sizes"][0][1] > plain["sizes"][0][1]


def test_positioning_picture_is_refused_where_the_job_disables_it(make, images: Path):
    ref = make("pictures_off", PICTURES)
    _s, _c, box = box_of(images, ref, disabled=["look:positioning"])
    assert call(box, "draw")["error"] == "POSITIONING_OFF"


RAW = '''
from langslice.plugin_api import ToolResult, ToolSpec

def stub(section: str, channel: str = ""):
    """Read."""

def handler(ctx, section, channel=""):
    names = [channel] if channel else None
    default = ctx.section_raster(section, level=64).array
    raw = ctx.section_raster(section, level=64, images=names, raw=True).array
    tinted = ctx.section_raster(section, level=64, images=names).array if names else default
    return ToolResult({"default": default.tolist(), "raw": raw.tolist(),
                       "tinted": tinted.tolist()})

def tools():
    return [ToolSpec("rasters", stub, handler, kind="read", group="Common")]
'''


def test_raw_section_raster_is_the_oriented_file_not_the_preprocessed_blend(make, images: Path):
    ref = make("raw_plugin", RAW)
    rng = np.random.default_rng(3)
    delivered = rng.integers(0, 120, size=(30, 40, 3), dtype=np.uint8)  # dim, like a stain
    Image.fromarray(delivered).save(images / "s0.png")
    state, ctx, box = box_of(images, ref, n=0)
    ctx.spec.preprocess = "auto"
    record = state.by_id("s0.png")
    for rotation, flip in ((0, False), (0, True), (90, True)):
        record.rotation_deg, record.flip = rotation, flip
        ctx.source_cache.clear()
        ctx.render_cache.clear()
        reply = call(box, "rasters", section="s0.png")
        expected = np.rot90(delivered, k=rotation // 90)  # PIL's ROTATE_90 turns counter-clockwise
        expected = expected[:, ::-1] if flip else expected
        np.testing.assert_array_equal(np.array(reply["raw"], dtype=np.uint8), expected)
    # the default appearance preprocesses (CLAHE, blend): not the file
    assert not np.array_equal(np.array(reply["default"], dtype=np.uint8), expected)


def test_raw_section_raster_of_one_channel_is_that_page_as_gray(make, images: Path):
    import tifffile

    ref = make("raw_channel_plugin", RAW)
    rng = np.random.default_rng(4)
    pages = rng.integers(0, 255, size=(2, 30, 40), dtype=np.uint8)
    tifffile.imwrite(images / "s0.tif", pages)
    state, ctx, box = box_of(images, ref, n=0, inputs={"channel_names": ["DAPI", "GFP"]})
    ctx.spec.preprocess = "auto"
    record = state.by_id("s0.tif")
    record.flip = True
    ctx.source_cache.clear()
    ctx.render_cache.clear()
    reply = call(box, "rasters", section="s0.tif", channel="GFP")
    raw = np.array(reply["raw"], dtype=np.uint8)
    plane = np.array(pages[1][:, ::-1])
    assert (raw[..., 0] == raw[..., 1]).all() and (raw[..., 1] == raw[..., 2]).all()
    np.testing.assert_array_equal(raw[..., 0], plane)
    # without raw one raw channel is the same section image, as the file holds it
    np.testing.assert_array_equal(np.array(reply["tinted"], dtype=np.uint8), raw)
    # a name the section has no image of is refused
    again = call(box, "rasters", section="s0.tif", channel="nope")
    assert again["error"] == "UNKNOWN_IMAGE"
    # the preprocessed image is no raw read
    made = call(box, "rasters", section="s0.tif", channel="preprocessed")
    assert made["error"] == "BAD_ARGS"


CHANNELS = '''
from langslice.plugin_api import ToolResult, ToolSpec

def stub(section: str):
    """Read."""

def handler(ctx, section):
    return ToolResult({"names": list(ctx.channels(section))})

def tools():
    return [ToolSpec("channel_names", stub, handler, kind="read", group="Common")]
'''


def test_channels_names_the_raw_channels_section_raster_takes(make, images: Path):
    import tifffile

    ref = make("channels_plugin", CHANNELS)
    Image.fromarray(np.zeros((30, 40, 3), np.uint8)).save(images / "s0.png")
    tifffile.imwrite(images / "s1.tif", np.zeros((2, 30, 40), np.uint8))
    _state, _ctx, box = box_of(images, ref, n=0, inputs={"channel_names": ["DAPI", "GFP"]})
    assert call(box, "channel_names", section="s1.tif")["names"] == ["DAPI", "GFP"]
    assert call(box, "channel_names", section="s0.png")["names"] == ["gray"]
    assert call(box, "channel_names", section="nope.png")["error"] == "UNKNOWN_SLICE_IDS"


PLACER = '''
from langslice.plugin_api import ToolResult, ToolSpec

def stub(path: str, at_stack: bool = False):
    """Place from a registration file."""

def handler(ctx, path, at_stack=False):
    first = ctx.read_registration(path)
    rows = first["placements"]
    stack = (sum(r["pitch_deg"] for r in rows) / len(rows),
             sum(r["yaw_deg"] for r in rows) / len(rows))
    read, angles = first, None
    if at_stack:
        at = {r["id"]: {"position_mm": r["position_mm"], "angles": stack} for r in rows}
        read = ctx.read_registration(path, at=at)
        angles = {"pitch_deg": stack[0], "yaw_deg": stack[1]}
    with ctx.writing():
        a = ctx.set_positions(
            [{"id": r["id"], "position_mm": r["position_mm"]} for r in read["placements"]],
            cutting_angles=angles)
        b = ctx.set_transforms([{"id": r["id"], "params": r["params"], "note": "from file"}
                                for r in read["placements"]])
    return ToolResult({"read": read, "positions": a.reply(), "transforms": b.reply()})

def tools():
    return [ToolSpec("place_from_file", stub, handler, kind="write", group="Linear",
                     task="transform", writes=("position", "angles", "affine"))]
'''


def test_a_registration_is_read_in_the_current_orientation_and_written_as_one_step(
        make, images: Path, tmp_path: Path):
    from langslice.core.layers import atlas_facts
    from langslice.job.quint import job_export
    from tests.test_import_registrations import (
        DeepAtlas,
        Given,
        export_rows,
        knobs,
        params_for,
    )

    ref = make("place_file", PLACER)
    atlas = DeepAtlas()
    given = [
        Given("s0.png", 0.62, 3.5, -2.0, 0, False, knobs(12.0, 1.1, 0.92, 0.2, -0.1, 0.08)),
        Given("s1.png", 1.0, 3.5, -2.0, 0, True, knobs(-8.0, 1.0, 1.0, 0.0, 0.1, 0.0)),
        Given("s2.png", 1.33, 3.5, -2.0, 90, False, knobs(5.0, 0.95, 1.05, 0.0, 0.0, 0.0)),
    ]
    state, ctx, box = box_of(images, ref, atlas=atlas, inputs={"pixel_size_um": 25.0})
    # the agent's orientation: s1 flipped, s2 turned a quarter
    state.by_id("s1.png").flip = True
    state.by_id("s2.png").rotation_deg = 90
    ctx.render_cache.clear()
    file = tmp_path / "deepslice.json"
    file.write_text(json.dumps(job_export(export_rows(ctx, given), atlas_facts(atlas))))

    depth = len(box.job.undo_stack)
    reply = call(box, "place_from_file", path=str(file))
    assert reply["status"] == "ok", reply
    rows = {row["id"]: row for row in reply["read"]["placements"]}
    assert [(rows[g.section_id]["rotation_deg"], rows[g.section_id]["flip"]) for g in given] \
        == [(0, False), (0, True), (90, False)]
    for g in given:  # the six numbers are for the oriented render
        np.testing.assert_allclose(rows[g.section_id]["params"], params_for(ctx, g), atol=1e-6)
        assert rows[g.section_id]["out_of_plane_um"] < 1e-3
    assert len(box.job.undo_stack) == depth + 1  # positions + transforms: one undo step
    held = state.by_id("s1.png")
    assert held.position_mm == pytest.approx(1.0, abs=1e-4)
    assert held.transform is not None and held.transform["note"] == "from file"
    assert call(box, "undo")["status"] == "ok"
    assert state.by_id("s1.png").transform is None and state.by_id("s1.png").position_mm == 2.5


READER = '''
from langslice.plugin_api import ToolResult, ToolSpec

def stub(path: str, on: str = "file"):
    """Read a registration."""

def handler(ctx, path, on="file"):
    return ToolResult(ctx.read_registration(path, registered_on=on))

def tools():
    return [ToolSpec("read_file_registration", stub, handler, kind="read", group="Common")]
'''


def test_a_registration_made_on_the_oriented_section_is_read_for_that_orientation(
        make, images: Path, tmp_path: Path):
    from langslice.core.layers import atlas_facts
    from langslice.core.maps import SWAP, orientation_matrix, render_sizes
    from langslice.job.quint import job_export
    from tests.test_import_registrations import (
        DeepAtlas,
        Given,
        knobs,
        params_for,
        pixel_map,
    )

    ref = make("read_oriented", READER)
    atlas = DeepAtlas()
    given = [
        Given("s0.png", 0.62, 3.5, -2.0, 0, False, knobs(12.0, 1.1, 0.92, 0.2, -0.1, 0.08)),
        Given("s1.png", 1.0, 3.5, -2.0, 0, True, knobs(-8.0, 1.0, 1.0, 0.0, 0.1, 0.0)),
        Given("s2.png", 1.33, 3.5, -2.0, 90, False, knobs(5.0, 0.95, 1.05, 0.0, 0.0, 0.0)),
        Given("s3.png", 1.5, 3.5, -2.0, 270, True, knobs(-5.0, 1.05, 0.95, 0.1, 0.0, 0.0)),
    ]
    state, ctx, box = box_of(images, ref, n=4, atlas=atlas, inputs={"pixel_size_um": 25.0})
    for g in given:
        record = state.by_id(g.section_id)
        record.flip, record.rotation_deg = g.flip, g.rotation_deg
    ctx.render_cache.clear()
    rows = []
    for index, g in enumerate(given):  # the registration was made on the ORIENTED image
        size = render_sizes(ctx, g.section_id).file_size
        turned = (size[1], size[0]) if g.rotation_deg % 180 == 90 else size
        to_oriented = SWAP @ orientation_matrix(size, g.rotation_deg, g.flip) @ SWAP
        rows.append({"filename": g.section_id, "width": turned[0], "height": turned[1],
                     "nr": index + 1, "markers": [],
                     "pixel_to_atlas_um": pixel_map(ctx, g) @ np.linalg.inv(to_oriented)})
    file = tmp_path / "oriented.json"
    file.write_text(json.dumps(job_export(rows, atlas_facts(atlas))))
    reply = call(box, "read_file_registration", path=str(file), on="oriented")
    assert reply["status"] == "ok" and not reply["refused"], reply
    found = {row["id"]: row for row in reply["placements"]}
    for g in given:  # the placement and numbers a file made on the stored file would give
        row = found[g.section_id]
        assert (row["rotation_deg"], row["flip"]) == (g.rotation_deg, g.flip)
        assert row["position_mm"] == pytest.approx(g.position_mm, abs=1e-3)
        np.testing.assert_allclose(row["params"], params_for(ctx, g), atol=1e-6)
    # read as a file-frame registration the same entries put a mirror or a turn in the numbers
    as_file = call(box, "read_file_registration", path=str(file))["placements"]
    wrong = {row["id"]: row for row in as_file}
    assert not np.allclose(wrong["s1.png"]["params"], params_for(ctx, given[1]), atol=1e-3)


def test_a_registration_read_at_the_stack_plane_projects_onto_it(
        make, images: Path, tmp_path: Path):
    from langslice.core.layers import atlas_facts
    from langslice.job.quint import job_export
    from tests.test_import_registrations import DeepAtlas, Given, export_rows, knobs

    ref = make("place_file_at", PLACER)
    atlas = DeepAtlas()
    given = [Given(f"s{i}.png", 1.0 + 0.3 * i, 3.0 + i, -2.0 + 0.5 * i, 0, False,
                   knobs(2.0 * i, 1.0, 1.0, 0.0, 0.0, 0.0)) for i in range(3)]
    state, ctx, box = box_of(images, ref, atlas=atlas, inputs={"pixel_size_um": 25.0})
    file = tmp_path / "deepslice.json"
    file.write_text(json.dumps(job_export(export_rows(ctx, given), atlas_facts(atlas))))
    depth = len(box.job.undo_stack)
    reply = call(box, "place_from_file", path=str(file), at_stack=True)
    assert reply["status"] == "ok", reply
    rows = reply["read"]["placements"]
    # drawn at the stack's mean angles, not at the sections' own
    assert {round(r["pitch_deg"], 3) for r in rows} == {4.0}
    assert [r["out_of_plane_um"] > 1 for r in rows] == [True, False, True]
    assert all(r["out_of_plane_um"] < 500 for r in rows)
    assert len(box.job.undo_stack) == depth + 1
    assert state.by_id("s0.png").pitch_deg == pytest.approx(4.0)
    assert state.by_id("s0.png").transform["params"] == pytest.approx(rows[0]["params"])


def test_a_registration_read_without_a_pixel_size_gives_no_numbers(
        make, images: Path, tmp_path: Path):
    from langslice.core.layers import atlas_facts
    from langslice.job.quint import job_export
    from tests.test_import_registrations import (
        DeepAtlas,
        Given,
        export_rows,
        knobs,
        make_workspace,
    )

    ref = make("read_file_uncal", '''
        from langslice.plugin_api import ToolResult, ToolSpec
        def stub(path: str):
            """Read a registration."""
        def handler(ctx, path):
            return ToolResult({"read": ctx.read_registration(path)})
        def tools():
            return [ToolSpec("read_registration_file", stub, handler, kind="read", group="Common")]
    ''')
    atlas = DeepAtlas()
    given = [Given("s0.png", 1.0, 0.0, 0.0, 0, False, knobs(0.0, 1.0, 1.0, 0.0, 0.0, 0.0))]
    state, _ctx, box = box_of(images, ref, atlas=atlas)  # no pixel size
    file = tmp_path / "x.json"
    file.write_text(json.dumps(job_export(
        export_rows(make_workspace(images, atlas), given), atlas_facts(atlas))))
    reply = call(box, "read_registration_file", path=str(file))
    row = reply["read"]["placements"][0]
    assert row["pixel_size_source"] == "imported"
    assert row["params"] is None and "no pixel size" in row["params_note"]
