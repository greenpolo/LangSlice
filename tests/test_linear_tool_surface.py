"""The tool surface: preprocess, shared display options, atlas images, renames."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from langslice.agent.engine import build_context
from langslice.agent.prompt import TOOL_LINES
from langslice.core import appearance as looks
from langslice.core.sections import render_slice
from langslice.core.spec import JobSpec, NonlinearSpec, PositionSpec
from langslice.doors.tools import TOOL_MEDIA_PARTS_KEY
from langslice.doors.tools.media import package_result
from langslice.doors.tools.toolbox import build_tools
from langslice.job.job import ingest
from tests.fakes import SlabAtlas
from tests.linear_tool_helpers import tool_named as _tool

PICTURE_TOOLS = (
    "view_slices", "view_atlas", "view_placement", "view_stack", "set_positions",
    "orient_slices", "fit_affine", "adjust_transforms", "preprocess",
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


def _bytes(result: dict[str, Any]) -> list[bytes]:
    # What the ADK agent receives: the door's JPEG parts.
    return [part.inline_data.data
            for part in package_result(result).get(TOOL_MEDIA_PARTS_KEY, [])]


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
    assert tool(slices=["s1.png"], channel_weights=[0, 1, 0], clahe_tiles=4)["scope"] == [
        "s1.png"
    ]
    for target in looks.TARGETS:
        assert looks.section_settings(state, target, "s0.png")["channel_weights"] == [1, 0, 0]
        assert looks.section_settings(state, target, "s1.png")["clahe_tiles"] == 4
    tool(slices=["s1.png"], target="view", reset=True)
    assert looks.section_settings(state, "view", "s1.png")["channel_weights"] == [1, 0, 0]
    assert looks.section_settings(state, "fit", "s1.png")["channel_weights"] == [0, 1, 0]
    tool(reset=True)
    assert looks.section_settings(state, "view", "s0.png") is None


def test_preprocess_is_one_undoable_checkpointed_write(tmp_path: Path):
    state, ctx, _, box = _setup(tmp_path, agent_preprocessing=True)
    _tool(box, "preprocess")(channel_weights=[0, 1, 0])
    from langslice.job.checkpoint import load_checkpoint

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
    assert tool(slices=["nope.png"])["error"] == "UNKNOWN_SLICE_IDS"
    assert state.appearance == {} and not box.job.undo_stack


def test_ants_steps_run_when_the_extra_is_installed(tmp_path: Path):
    pytest.importorskip("ants")
    state, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    result = _tool(box, "preprocess")(slices=["s0.png"], n4=True, denoise=True)
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




# --- one `view` argument ---------------------------------------------------


def _pixels(result: dict[str, Any], index: int = 0) -> np.ndarray:
    import io

    return np.asarray(Image.open(io.BytesIO(_bytes(result)[index])).convert("RGB")).astype(int)


def test_every_picture_tool_takes_one_view_argument(tmp_path: Path):
    from google.adk.tools import FunctionTool

    from langslice.providers.openai_oauth import _json_schema_dict

    _, _, _, box = _setup(
        tmp_path, agent_preprocessing=True, position=PositionSpec(bayesian=True),
    )
    old = {"mode", "zoom", "section_image", "atlas_image", "atlas_opacity", "regions",
           "outlines", "border_color", "border_thickness", "resolution", "slice_ids",
           "sections"}
    for name in PICTURE_TOOLS:
        tool = _tool(box, name)
        parameters = set(inspect.signature(tool).parameters)
        assert "view" in parameters and not parameters & old, name
        # The model is sent every view key with its type, not an opaque object.
        schema = _json_schema_dict(FunctionTool(tool)._get_declaration())
        view = schema["properties"]["view"]
        assert view["type"] == "object" and view.get("additionalProperties") is False, name
        assert view["properties"]["channels"]["type"] == "array", name
        assert view["properties"]["atlas_opacity"]["type"] == "number", name
        assert "$ref" not in str(schema), name
        assert "described once in the job statement" in (tool.__doc__ or ""), name


def test_unknown_and_misplaced_arguments_are_refused(tmp_path: Path):
    import asyncio

    from google.adk.tools import FunctionTool

    from langslice.agent.plugins import StrictArgumentsPlugin

    state, _, _, box = _setup(tmp_path)
    before = state.to_dict()
    view = _tool(box, "view_slices")
    top = view(["s0.png"], mode="section")
    assert top["error"] == "UNKNOWN_ARGUMENTS"
    assert top["problems"][0] == {"argument": "(top level)", "unknown": ["mode"],
                                  "accepted": ["slices", "view"],
                                  "notes": ["`mode` belongs inside `view`."]}
    nested = view(["s0.png"], view={"mode": "section", "colour": "red"})
    assert nested["problems"][0]["argument"] == "view"
    assert nested["problems"][0]["unknown"] == ["colour"]
    entry = _tool(box, "set_positions")([{"id": "s0.png", "position": 4.0}])
    assert entry["error"] == "UNKNOWN_ARGUMENTS"
    assert entry["problems"][0]["argument"] == "entries[0]"
    assert entry["problems"][0]["accepted"] == ["id", "position_mm"]
    assert state.to_dict() == before and not box.job.undo_stack
    # ADK drops unknown top-level arguments before a tool runs; the plugin
    # answers the call first.
    plugin = StrictArgumentsPlugin()
    answer = asyncio.run(plugin.before_tool_callback(
        tool=FunctionTool(view), tool_args={"slices": ["s0.png"], "zoom": [0, 0, 1, 1]},
        tool_context=None))
    assert answer is not None and answer["error"] == "UNKNOWN_ARGUMENTS"
    assert asyncio.run(plugin.before_tool_callback(
        tool=FunctionTool(view), tool_args={"slices": ["s0.png"]}, tool_context=None)) is None


def test_view_keys_that_mean_nothing_are_refused_with_the_reason(tmp_path: Path):
    _, _, _, box = _setup(tmp_path, tasks=["position", "transform", "nonlinear"],
                          nonlinear=NonlinearSpec(provider="none"))
    show = _tool(box, "view_placement")

    def unused(result: dict[str, Any]) -> list[str]:
        assert result["error"] == "VIEW_KEY_UNUSED", result
        return [item["key"] for item in result["unused"]]

    assert unused(_tool(box, "view_slices")(["s0.png"], view={"atlas_channels": ["template"]})) \
        == ["atlas_channels"]
    assert unused(_tool(box, "view_atlas")([5.0], view={"channels": ["red"]})) == ["channels"]
    assert unused(show([{"id": "s0.png"}], view={"mode": "template", "channels": ["red"]})) \
        == ["channels"]
    assert unused(show([{"id": "s0.png"}], view={"mode": "overlay", "outlines": "outer",
                                                  "atlas_channels": ["template"]})) == ["outlines"]
    assert unused(show([{"id": "s0.png"}], view={"mode": "overlay", "atlas_opacity": 0.5})) \
        == ["atlas_opacity"]
    assert unused(show([{"id": "s0.png"}], view={"mode": "stacked", "deformation": "none"})) \
        == ["deformation"]
    assert unused(_tool(box, "fit_affine")(["s0.png"], view={"deformation": "none"})) \
        == ["deformation"]
    assert show([{"id": "s0.png"}], view={"mode": "overlay", "atlas_channels": ["template"],
                                          "outlines": "none"})["error"] == "VIEW_KEY_UNUSED"
    assert show([{"id": "s0.png"}], view={"mode": "checkerboard",
                                          "atlas_channels": ["borders"]})[
        "error"] == "BAD_ATLAS_CHANNELS"


def test_raw_channels_one_in_gray_several_overlaid_in_colour(tmp_path: Path):
    state, _, _, box = _setup(tmp_path)
    view = _tool(box, "view_slices")
    default = view(["s0.png"])
    assert default["view"]["channels"] == ["view"]
    green = view(["s0.png"], view={"channels": ["green"]})
    pixels = _pixels(green)[:-45]  # above the caption
    assert np.abs(pixels[..., 0] - pixels[..., 1]).max() <= 12  # grayscale
    both = view(["s0.png"], view={"channels": ["red", "blue"]})
    assert both["view"]["channel_colors"] == {"red": "red", "blue": "blue"}
    mixed = _pixels(both)[:-45][-40:]  # tissue rows, clear of the white caption text
    # Both present; each is dimmed by its fine detail relative to the other.
    assert mixed[..., 0].max() > 200 and mixed[..., 2].max() > 40
    # No green channel in a red + blue overlay (JPEG chroma leaves a little at edges).
    assert np.percentile(mixed[..., 1], 99) < 40 and mixed[..., 1].mean() < 10
    assert view(["s0.png"], view={"channels": ["dapi"]})["error"] == "UNKNOWN_CHANNEL"
    assert view(["s0.png"], view={"channels": ["red", "fit"]})["error"] == "BAD_CHANNELS"
    assert state.appearance == {}


def test_a_flat_channel_is_dimmed_in_an_overlay():
    from langslice.core.sections import fine_detail

    rng = np.random.default_rng(0)
    textured = rng.uniform(0.2, 1.0, (64, 64)).astype(np.float32)
    flat = np.full((64, 64), 0.9, dtype=np.float32)
    assert fine_detail(flat) < 0.01 < fine_detail(textured)


def test_the_fit_version_shows_what_registration_reads(tmp_path: Path):
    _, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    _tool(box, "preprocess")(target="fit", channel_weights=[0, 1, 0], clahe_clip=0)
    view = _tool(box, "view_slices")
    seen = view(["s0.png"], view={"channels": ["view"]})
    fit = view(["s0.png"], view={"channels": ["fit"]})
    assert fit["view"]["channels"] == ["fit"] and _bytes(fit) != _bytes(seen)
    assert _bytes(seen) == _bytes(view(["s0.png"]))


def test_view_slices_channels_mode_shows_every_raw_channel(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    result = _tool(box, "view_slices")(["s0.png", "s1.png"], view={"mode": "channels"})
    assert result["status"] == "ok", result
    assert result["channels"] == {"s0.png": ["red", "green", "blue"],
                                  "s1.png": ["red", "green", "blue"]}
    assert len(_bytes(result)) == 2  # one strip per section
    strip = _pixels(result)
    single = _pixels(_tool(box, "view_slices")(["s0.png"]))
    assert strip.shape[1] > 2 * strip.shape[0]  # three tiles side by side
    assert strip.shape[1] > single.shape[1]


def test_preprocess_returns_before_and_after(tmp_path: Path):
    _, _, _, box = _setup(tmp_path, agent_preprocessing=True)
    result = _tool(box, "preprocess")(slices=["s0.png"], target="view",
                                      channel_weights=[0, 0, 1])
    assert result["status"] == "ok", result
    assert result["image_indexes"] == {"s0.png": {"before": 0, "after": 1}}
    before, after = _bytes(result)
    assert before != after
    assert "BEFORE" in result["description"] and "AFTER" in result["description"]
    assert _tool(box, "preprocess")(view={"channels": ["red"]})["error"] == "VIEW_KEY_UNUSED"


def test_atlas_channels_overlay_and_the_host_without_nissl(tmp_path: Path):
    _, ctx, _, box = _setup(tmp_path)
    show = _tool(box, "view_placement")
    template = show([{"id": "s0.png"}], view={"mode": "template"})
    assert template["view"]["atlas_channels"] == ["template"]
    lined = show([{"id": "s0.png"}], view={"mode": "template",
                                           "atlas_channels": ["template", "borders"]})
    lines_only = show([{"id": "s0.png"}], view={"mode": "template",
                                                "atlas_channels": ["borders"]})
    assert len({_bytes(template)[0], _bytes(lined)[0], _bytes(lines_only)[0]}) == 3
    under = show([{"id": "s0.png"}], view={"mode": "overlay",
                                           "atlas_channels": ["template", "borders"],
                                           "atlas_opacity": 0.6})
    assert under["view"]["atlas_opacity"] == 0.6
    ctx._nissl, ctx._nissl_checked = None, True
    refused = show([{"id": "s0.png"}], view={"mode": "template", "atlas_channels": ["nissl"]})
    assert refused["error"] == "ATLAS_CHANNEL_UNAVAILABLE"
    assert refused["available"] == ["template", "borders"]

    class _Nissl:
        def sample_plane(self, atlas, position_mm, plane, pitch, yaw):
            return np.linspace(0, 500, 48 * 64, dtype=np.float32).reshape(48, 64)

    ctx._nissl = _Nissl()
    both = show([{"id": "s0.png"}], view={"mode": "template",
                                          "atlas_channels": ["template", "nissl"]})
    assert both["status"] == "ok", both
    assert both["view"]["atlas_colors"] == {"template": "green", "nissl": "magenta"}
    assert _bytes(both) != _bytes(template)


def test_regions_highlight_only_their_borders(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    show = _tool(box, "view_placement")
    plain = show([{"id": "s0.png"}], view={"mode": "overlay"})
    highlighted = show([{"id": "s0.png"}], view={"mode": "overlay", "regions": ["HPF"]})
    assert highlighted["status"] == "ok", highlighted
    assert highlighted["view"]["regions"] == ["HPF"]
    assert _bytes(highlighted) != _bytes(plain)
    assert "regions_not_in_plane" not in highlighted["compared"][0]
    # HPF (here only its descendant CA1) is not in the posterior half.
    later = show([{"id": "s0.png", "positions_mm": [15.0]}],
                 view={"mode": "template", "regions": ["HPF"]})
    assert later["compared"][0]["regions_not_in_plane"] == ["HPF"]
    assert show([{"id": "s0.png"}], view={"regions": ["XYZ"]})["error"] == "UNKNOWN_REGIONS"
    atlas_view = _tool(box, "view_atlas")([5.0], view={"regions": ["CA1"],
                                                       "atlas_channels": ["template", "borders"]})
    assert atlas_view["status"] == "ok" and atlas_view[TOOL_MEDIA_PARTS_KEY]


def test_view_placement_shows_the_stored_transform(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    show = _tool(box, "view_placement")
    before = show([{"id": "s0.png"}], view={"mode": "overlay"})
    assert before["compared"][0]["transform"] == "identity"
    _tool(box, "adjust_transforms")([{
        "id": "s0.png", "rotation_deg": 20, "scale_x": 1, "scale_y": 1,
        "translate_x_mm": 3, "translate_y_mm": 0,
    }])
    after = show([{"id": "s0.png"}], view={"mode": "overlay"})
    assert after["compared"][0]["transform"] == "interactive"
    assert _bytes(after) != _bytes(before)


def test_set_positions_pictures_take_the_placement_modes(tmp_path: Path):
    _, _, _, box = _setup(tmp_path)
    result = _tool(box, "set_positions")(
        [{"id": "s0.png", "position_mm": 4.0}], view={"mode": "overlay", "regions": ["TH"]},
    )
    assert result["status"] == "ok", result
    assert result["view"]["mode"] == "overlay" and len(_bytes(result)) == 1
    refused = _tool(box, "set_positions")(
        [{"id": "s1.png", "position_mm": 4.5}], view={"zoom": [0, 0, 0.5, 0.5]},
    )
    assert refused["error"] == "ZOOM_UNSUPPORTED"


def test_the_job_statement_describes_view_once_with_the_channels():
    from langslice.agent.prompt import PICTURE_TOOLS as PROMPT_PICTURE_TOOLS
    from langslice.agent.prompt import display_lines

    assert "fit_deformable" in PROMPT_PICTURE_TOOLS
    lines = display_lines(["view_slices"], channels=["red", "green", "blue"],
                          atlas_channels=("template", "nissl", "borders"))
    text = "\n".join(lines)
    assert text.count("ONE argument, `view`") == 1
    for key in ("`channels`", "`atlas_channels`", "`atlas_opacity`", "`regions`",
                "`outlines`", "`border_color`", "`border_thickness`", "`zoom`",
                "`deformation`"):
        assert key in text, key
    assert "may not say which is the stain" in text and "mode channels" in text
    assert "Raw image channels of every section" in text and "red, green, blue." in text
    assert "nissl (a Nissl-stained reference" in text and "borders (the atlas regions" in text
    assert "resolution" not in text
    assert display_lines(["status"]) == []


# --- renames ---------------------------------------------------------------


def test_renamed_tools_and_arguments_replace_the_old_names(tmp_path: Path):
    _, _, _, box = _setup(tmp_path, position=PositionSpec(bayesian=True))
    names = set(box.names)
    assert {"view_atlas", "view_placement", "search_position"} <= names
    assert not names & {"fetch_atlas", "compare_placement", "fit_position"}
    lines = " ".join(TOOL_LINES.values()) + " ".join(TOOL_LINES)
    for old in ("fetch_atlas", "compare_placement", "fit_position", "template_opacity",
                "new_order", "section_image", "atlas_image", "slice_ids"):
        assert old not in lines, old
    for name in ("reorder_slices", "fit_affine", "view_slices"):
        assert "slices" in inspect.signature(_tool(box, name)).parameters, name
    assert "id" in inspect.signature(_tool(box, "search_position")).parameters


def test_the_view_echo_names_channels_only_where_they_choose_the_picture(tmp_path: Path):
    """Handed-over bug 3: preprocess (a target's appearance) and fit_deformable
    (the image the fit read) echoed channels ["view"], which they never drew."""
    _state, _ctx, _spec, box = _setup(tmp_path, agent_preprocessing=True)
    shown = _tool(box, "view_slices")(["s0.png"])
    assert shown["view"]["channels"] == ["view"]
    picked = _tool(box, "view_slices")(["s0.png"], view={"channels": ["fit"]})
    assert picked["view"]["channels"] == ["fit"]
    preprocessed = _tool(box, "preprocess")(target="fit", clahe_clip=2.0)
    assert preprocessed["status"] == "ok" and "channels" not in preprocessed["view"]



def test_old_ara_name_is_still_accepted(tmp_path: Path):
    """The atlas template was once called ``ara``: call arguments and saved
    settings that use it load as ``template``."""
    from dataclasses import asdict

    from langslice.core.deformable.settings import FitSettings

    _, ctx, _, box = _setup(tmp_path)
    shown = _tool(box, "view_placement")([{"id": "s0.png"}], view={
        "mode": "template", "atlas_channels": ["ara", "borders"]})
    assert shown["view"]["atlas_channels"] == ["template", "borders"]

    saved = asdict(FitSettings(section_image="stain", atlas_image="template"))
    saved["atlas_image"] = "ara"  # as an older job folder recorded it
    assert FitSettings.from_dict(saved).atlas_image == "template"
