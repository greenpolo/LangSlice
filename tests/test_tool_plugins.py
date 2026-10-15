"""Tool plugins: loading, gating, writes, layers, doors, provenance, invariance.

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
from langslice.core.spec import JobSpec
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
            frame = ctx.fit_frame("s1.png")
            return ToolResult({
                "id": view.id, "position": view.position_mm, "size": list(view.file_size),
                "shape": list(image.shape), "atlas": atlas.name, "plane": atlas.plane,
                "planes": sorted(frame.atlas), "matrix": frame.atlas_to_section.shape[0],
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
    assert [item["name"] for item in verbs][-2:] == ["section_brightness", "nudge_sections"]
    assert verbs[-1]["kind"] == "write" and verbs[-1]["group"] == "Linear"
    main(["schema", "nudge_sections"])
    schema = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]
    assert set(schema["arguments"]["properties"]) == {"sections", "dx"}
    monkeypatch.chdir(images.parent)  # a folder with no job: the native listing
    main(["ops"])
    plain = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["result"]["verbs"]
    assert [item["name"] for item in plain] == list(verbs_for())
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
    assert "layers" not in inspect.signature(tool_named(box, "look")).parameters
    assert "layers" not in declaration("look", FULL).signature.parameters
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
    raster = ctx.section_raster("s0.png", level=LEVEL)
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
                               **kwargs)["images"][0])

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
