"""The tool surface: preprocess, shared display options, atlas images, renames."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.adk import TOOL_MEDIA_PARTS_KEY
from langslice.linear import appearance as looks
from langslice.linear.display import DISPLAY_DOC
from langslice.linear.engine import build_context, ingest
from langslice.linear.prompt import TOOL_LINES
from langslice.linear.render import render_slice
from langslice.linear.spec import JobSpec, PositionSpec
from langslice.linear.toolbox import build_tools
from tests.fakes import SlabAtlas

DISPLAY_ARGS = (
    "mode", "zoom", "section_image", "atlas_image", "atlas_opacity", "regions",
    "outlines", "border_color", "border_thickness",
)
PICTURE_TOOLS = (
    "view_slices", "view_atlas", "view_placement", "view_stack", "set_positions",
    "orient_slices", "fit_affine", "preprocess",
)
_REGIONS = [
    (1, "root", "root", [1]),
    (2, "CTX", "Cerebral cortex", [1, 2]),
    (3, "HPF", "Hippocampal formation", [1, 2, 3]),
    (4, "CA1", "Field CA1", [1, 2, 3, 4]),
    (6, "TH", "Thalamus", [1, 6]),
]


def _atlas() -> SlabAtlas:
    atlas = SlabAtlas(height=48, width=64)
    ann = np.zeros((20, 48, 64), dtype=np.int32)
    ann[:, 6:42, 6:58] = 1
    ann[:10, 12:24, 12:26] = 4  # CA1 (HPF) only in the anterior half
    ann[:, 26:38, 30:50] = 6
    atlas.annotation = ann
    atlas.template = (ann * 40).astype(np.uint8)
    atlas.structures = {  # type: ignore[attr-defined]
        i: {"id": i, "acronym": a, "name": n, "structure_id_path": p}
        for i, a, n, p in _REGIONS
    }
    return atlas


def _section(seed: int) -> np.ndarray:
    """Three distinct channels: a broad 'nuclear' red, a sparse green, a striped blue."""
    yy, xx = np.mgrid[:120, :160]
    body = ((yy - 60) / 45.0) ** 2 + ((xx - 80) / 65.0) ** 2 < 1
    red = np.where(body, 90 + (xx % 7) * 10, 0)
    green = np.zeros_like(red)
    green[40 + seed:55 + seed, 60:75] = 250
    blue = np.where(body & (yy % 10 < 3), 160, 0)
    return np.stack([red, green, blue], axis=-1).astype(np.uint8)


def _setup(tmp_path: Path, **spec_kwargs: Any):
    for index in range(2):
        Image.fromarray(_section(index)).save(tmp_path / f"s{index}.png")
    spec = JobSpec(image_folder=str(tmp_path), model="fake-model", **spec_kwargs)
    atlas = _atlas()
    ctx = build_context(spec, emit=lambda _m: None, atlas_loader=lambda _n: atlas)
    state = ingest(spec, ctx)
    for index, record in enumerate(state.in_order()):
        record.position_mm = 5.0 + index
    box = build_tools(state, ctx, spec)
    return state, ctx, spec, box


def _tool(box: Any, name: str) -> Any:
    return next(tool for tool in box.tools if tool.__name__ == name)


def _bytes(result: dict[str, Any]) -> list[bytes]:
    return [part.inline_data.data for part in result.get(TOOL_MEDIA_PARTS_KEY, [])]


# --- preprocess ------------------------------------------------------------


def test_preprocess_is_built_only_when_the_host_lets_the_agent_drive_it(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    assert "preprocess" not in box.names
    _, _, _, gated = _setup(tmp_path, agent_preprocessing=True)
    assert "preprocess" in gated.names
    with pytest.raises(ValueError, match="agent_preprocessing"):
        JobSpec(image_folder=str(tmp_path), agent_preprocessing="yes")  # type: ignore[arg-type]


def test_view_and_fit_appearances_are_set_independently(tmp_path: Path):
    state, ctx, _, box = _setup(tmp_path, agent_preprocessing=True)
    record = state.by_id("s0.png")
    baseline = np.asarray(render_slice(ctx, record, long_edge=512))
    result = _tool(box, "preprocess")(target="view", channel_weights=[0, 0, 1])
    assert result["status"] == "ok", result
    assert result["channels"] == ["red", "green", "blue"]
    assert result[TOOL_MEDIA_PARTS_KEY]
    assert looks.section_settings(state, "view", "s0.png")["channel_weights"] == [0, 0, 1]
    assert looks.section_settings(state, "fit", "s0.png") is None
    # The fit still reads the default appearance; the view does not.
    assert np.array_equal(np.asarray(looks.fit_image(ctx, state, record, long_edge=512)), baseline)
    viewed = render_slice(ctx, record, long_edge=512, look=looks.view_look(state, record))
    assert not np.array_equal(np.asarray(viewed), baseline)

    _tool(box, "preprocess")(target="fit", channel_weights=[0, 1, 0], clahe_clip=0)
    assert looks.section_settings(state, "fit", "s0.png")["channel_weights"] == [0, 1, 0]
    assert looks.section_settings(state, "view", "s0.png")["channel_weights"] == [0, 0, 1]
    assert not np.array_equal(
        np.asarray(looks.fit_image(ctx, state, record, long_edge=512)), np.asarray(viewed),
    )


def test_section_overrides_win_and_a_reset_returns_to_the_stack(tmp_path: Path):
    state, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    tool = _tool(box, "preprocess")
    assert tool(channel_weights=[1, 0, 0])["status"] == "ok"
    assert tool(sections=["s1.png"], channel_weights=[0, 1, 0], clahe_tiles=4)["scope"] == [
        "s1.png"
    ]
    for target in looks.TARGETS:
        assert looks.section_settings(state, target, "s0.png")["channel_weights"] == [1, 0, 0]
        assert looks.section_settings(state, target, "s1.png")["clahe_tiles"] == 4
    tool(sections=["s1.png"], target="view", reset=True)
    assert looks.section_settings(state, "view", "s1.png")["channel_weights"] == [1, 0, 0]
    assert looks.section_settings(state, "fit", "s1.png")["channel_weights"] == [0, 1, 0]
    tool(reset=True)
    assert looks.section_settings(state, "view", "s0.png") is None


def test_preprocess_is_one_undoable_checkpointed_write(tmp_path: Path):
    state, ctx, _, box = _setup(tmp_path, agent_preprocessing=True)
    _tool(box, "preprocess")(channel_weights=[0, 1, 0])
    from langslice.linear.checkpoint import load_checkpoint

    saved = load_checkpoint(ctx.checkpoint_path)
    assert saved is not None and saved.appearance["view"]["stack"]["channel_weights"] == [0, 1, 0]
    assert _tool(box, "undo")()["status"] == "ok"
    assert state.appearance == {}
    assert _tool(box, "redo")()["status"] == "ok"
    assert state.appearance["fit"]["stack"]["channel_weights"] == [0, 1, 0]


def test_preprocess_refuses_bad_settings_before_writing(tmp_path: Path):
    state, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    tool = _tool(box, "preprocess")
    assert tool(channel_weights=[1, 1])["error"] == "CHANNEL_COUNT_MISMATCH"
    assert tool(clahe_clip=-1)["error"] == "BAD_ARGS"
    assert tool(target="atlas")["error"] == "BAD_TARGET"
    assert tool(sections=["nope.png"])["error"] == "UNKNOWN_SLICE_IDS"
    assert state.appearance == {} and not box.undo_stack


def test_ants_steps_run_when_the_extra_is_installed(tmp_path: Path):
    pytest.importorskip("ants")
    state, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    result = _tool(box, "preprocess")(sections=["s0.png"], n4=True, denoise=True)
    assert result["status"] == "ok", result
    assert looks.section_settings(state, "view", "s0.png")["n4"] is True


def test_the_image_model_input_ignores_the_agents_appearance(tmp_path: Path):
    state, ctx, _, box = _setup(tmp_path, agent_preprocessing=True)
    record = state.by_id("s0.png")
    before = np.asarray(render_slice(ctx, record, long_edge=2048, frame=False)).copy()
    _tool(box, "preprocess")(channel_weights=[0, 0, 1])
    ctx.render_cache.clear()
    after = np.asarray(render_slice(ctx, record, long_edge=2048, frame=False))
    assert np.array_equal(before, after)


# --- display options -------------------------------------------------------


def test_every_picture_tool_takes_the_same_display_options(tmp_path: Path):
    _, _, _, box = _setup(
        tmp_path, agent_preprocessing=True, position=PositionSpec(bayesian=True),
    )
    for name in PICTURE_TOOLS:
        tool = _tool(box, name)
        assert set(DISPLAY_ARGS) <= set(inspect.signature(tool).parameters), name
        assert "regions:" in (tool.__doc__ or ""), name
    assert "regions:" in (_tool(box, "adjust_transforms").__doc__ or "")
    assert DISPLAY_DOC.strip().splitlines()[0] in (_tool(box, "view_atlas").__doc__ or "")
    result = _tool(box, "adjust_transforms")([{
        "id": "s0.png", "rotation_deg": 0, "scale_x": 1, "scale_y": 1,
        "translate_x_mm": 0, "translate_y_mm": 0, "mode": "overlay", "zoom": [],
        "section_image": "blue", "atlas_image": "borders", "atlas_opacity": 0.4,
        "regions": ["TH"], "outlines": "outer", "border_color": "#00ff00",
        "border_thickness": 1.0,
    }])
    row = result["results"][0]
    assert row["status"] == "ok", row
    assert row["view"]["regions"] == ["TH"] and row["view"]["atlas_image"] == "borders"


def test_per_call_options_never_change_the_stored_defaults(tmp_path: Path):
    state, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    view = _tool(box, "view_slices")
    default = _bytes(view(["s0.png"]))
    red = view(["s0.png"], section_image="red")
    assert red["view"]["section_image"] == "red"
    assert _bytes(red) != default
    assert state.appearance == {}
    assert _bytes(view(["s0.png"])) == default
    assert view(["s0.png"], section_image="dapi")["error"] == "UNKNOWN_CHANNEL"


def test_regions_highlight_only_their_borders(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    show = _tool(box, "view_placement")
    plain = show([{"id": "s0.png"}], mode="overlay")
    highlighted = show([{"id": "s0.png"}], mode="overlay", regions=["HPF"])
    assert highlighted["status"] == "ok", highlighted
    assert highlighted["view"]["regions"] == ["HPF"]
    assert _bytes(highlighted) != _bytes(plain)
    assert "regions_not_in_plane" not in highlighted["compared"][0]
    # HPF (here only its descendant CA1) is not in the posterior half.
    later = show([{"id": "s0.png", "positions_mm": [15.0]}], mode="template", regions=["HPF"])
    assert later["compared"][0]["regions_not_in_plane"] == ["HPF"]
    assert show([{"id": "s0.png"}], regions=["XYZ"])["error"] == "UNKNOWN_REGIONS"
    atlas_view = _tool(box, "view_atlas")([5.0], regions=["CA1"], outlines="all")
    assert atlas_view["status"] == "ok" and atlas_view[TOOL_MEDIA_PARTS_KEY]


def test_atlas_image_choice_and_the_host_without_nissl(tmp_path: Path):
    _, ctx, _, box = _setup(tmp_path)
    show = _tool(box, "view_placement")
    ara = _bytes(show([{"id": "s0.png"}], mode="template"))
    borders = show([{"id": "s0.png"}], mode="template", atlas_image="borders")
    assert borders["status"] == "ok" and _bytes(borders) != ara
    ctx._abba, ctx._abba_checked = None, True
    refused = show([{"id": "s0.png"}], atlas_image="nissl")
    assert refused["error"] == "ATLAS_IMAGE_UNAVAILABLE"
    assert refused["available"] == ["ara", "borders"]
    assert _tool(box, "view_atlas")([5.0], atlas_image="nissl")["error"] == (
        "ATLAS_IMAGE_UNAVAILABLE"
    )

    class _Nissl:
        def sample_plane(self, channel, atlas, position_mm, plane, pitch, yaw):
            assert channel == "NISSL"
            return np.linspace(0, 500, 48 * 64, dtype=np.float32).reshape(48, 64)

    ctx._abba = _Nissl()
    nissl = show([{"id": "s0.png"}], mode="template", atlas_image="nissl")
    assert nissl["status"] == "ok" and _bytes(nissl) not in (ara, _bytes(borders))


def test_view_placement_shows_the_stored_transform(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    show = _tool(box, "view_placement")
    before = show([{"id": "s0.png"}], mode="overlay")
    assert before["compared"][0]["transform"] == "identity"
    _tool(box, "adjust_transforms")([{
        "id": "s0.png", "rotation_deg": 20, "scale_x": 1, "scale_y": 1,
        "translate_x_mm": 3, "translate_y_mm": 0,
    }])
    after = show([{"id": "s0.png"}], mode="overlay")
    assert after["compared"][0]["transform"] == "interactive"
    assert _bytes(after) != _bytes(before)


def test_set_positions_pictures_take_the_placement_modes(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    result = _tool(box, "set_positions")(
        [{"id": "s0.png", "position_mm": 4.0}], mode="overlay", regions=["TH"],
    )
    assert result["status"] == "ok", result
    assert result["view"]["mode"] == "overlay" and len(_bytes(result)) == 1
    refused = _tool(box, "set_positions")(
        [{"id": "s1.png", "position_mm": 4.5}], zoom=[0, 0, 0.5, 0.5],
    )
    assert refused["error"] == "ZOOM_UNSUPPORTED"


# --- renames ---------------------------------------------------------------


def test_renamed_tools_replace_the_old_names(tmp_path: Path):
    _, _, _, box = _setup(tmp_path, position=PositionSpec(bayesian=True))
    names = set(box.names)
    assert {"view_atlas", "view_placement", "search_position"} <= names
    assert not names & {"fetch_atlas", "compare_placement", "fit_position"}
    lines = " ".join(TOOL_LINES.values()) + " ".join(TOOL_LINES)
    for old in ("fetch_atlas", "compare_placement", "fit_position", "template_opacity"):
        assert old not in lines
