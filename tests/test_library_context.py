"""A script's own method through the library: ``job.context``.

The plugin ``ToolContext`` handed to a script (``docs/library.md``, "Your own
method"): its writes recorded as ``script``, one undo step each, the job's
freeze and staleness rules, the previews saved as numbered pictures, the
engine I/O bridge, points through the registration, atlas planes anywhere,
the native engines with settings of the script's own and the script's layers.
Everything runs on the shared synthetic stack and atlas.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.cli_child import install
from tests.synthetic_stack import ID0, ID1, ID2, PIXEL_SIZE_UM, atlas_loader, write_sections

POSITIONS = {ID0: 0.1, ID1: 0.15, ID2: 0.2}
TRANSFORMS = {ID0: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], ID1: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              ID2: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]}


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    images = tmp_path_factory.mktemp("context") / "stack"
    write_sections(images)
    return images


@pytest.fixture
def images(stack: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    install(atlas_loader(), monkeypatch.setattr)
    folder = tmp_path / "stack"
    shutil.copytree(stack, folder)
    return folder


def make(images: Path, **settings: Any) -> Any:
    import langslice

    settings.setdefault("tasks", ["transform", "nonlinear"])
    return langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                                pixel_size_um=PIXEL_SIZE_UM, preprocess="none", **settings)


def events(job: Any) -> list[dict[str, Any]]:
    path = Path(job.folder) / "logs" / "events.jsonl"
    lines = path.read_text().splitlines() if path.exists() else []
    return [entry for entry in map(json.loads, lines) if entry["kind"] == "script_write"]


def shift(frame: Any, dx: float) -> np.ndarray:
    """A 2x3 fit-grid -> native map: the current placement, then *dx* fit-grid pixels."""
    placed = frame.placed_atlas_to_section
    move = np.array([[1.0, 0.0, dx], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    return (np.linalg.inv(placed) @ move)[:2]


# --- the context ----------------------------------------------------------------------


def test_the_context_is_one_per_handle_and_refuses_what_a_script_does_itself(images: Path):
    from langslice.core.plugins import Layer, ToolError
    from langslice.plugin_api import ToolContext

    with make(images) as job:
        c = job.context
        assert c is job.context and isinstance(c, ToolContext) and c.tool == "script"
        assert c.__doc__ and "script" in c.__doc__
        for call in (c.work_dir, lambda: c.start_work([ID0], lambda _ctx: None)):
            with pytest.raises(ToolError) as refused:
                call()
            assert refused.value.code == "NOT_FOR_SCRIPTS"
        ref = c.add_layer(ID0, Layer.points([[10.0, 12.0]], frame="section_px", name="cells"))
        assert ref.name == "cells" and ref.file.startswith("plugins/script/")


def test_an_affine_is_recorded_as_the_scripts_and_undo_reverts_it(images: Path):
    with make(images) as job:
        c = job.context
        before = job.state.by_id(ID0).transform
        depth = len(job.job.undo_stack)
        proposal = c.transform_from_matrix(ID0, shift(c.fit_frame(ID0), 6.0), note="shifted")
        done = c.set_transforms([proposal.row()])
        assert done.written == [ID0] and len(job.job.undo_stack) == depth + 1
        registration = json.loads((Path(job.folder) / "registration.json").read_text())
        entry = next(item for item in registration["sections"] if item["id"] == ID0)
        assert entry["parameters"]["affine"]["kind"] == "script"
        assert job.state.by_id(ID0).transform["note"] == "shifted"
        rows = {row["id"]: row for row in job.status()["rows"]}
        assert rows[ID0]["transform"] == "script"
        job.undo()
        assert job.state.by_id(ID0).transform == before
        written = events(job)
        assert len(written) == 1 and written[0]["property"] == ["affine"]
        assert written[0]["sections"] == [ID0] and written[0]["notes"] == {ID0: "shifted"}


def test_a_deformation_is_recorded_as_the_scripts_with_its_provenance(images: Path):
    from tests.fixtures.example_plugin import ripple_map

    with make(images) as job:
        c = job.context
        digest = c.section_inputs(ID0)
        native = ripple_map(c.fit_frame(ID0), 1.0)
        done = c.set_deformations([{"id": ID0, "map": native, "note": "ripple"}],
                                  expect={ID0: digest})
        assert done.written == [ID0], done.rows
        held = job.state.by_id(ID0).deformation
        assert held["steps"][-1] == {"method": "script", "start": "linear", "note": "ripple"}
        record = job.job.deformations.current(job.state, job.state.by_id(ID0))
        assert record.engine["name"] == "script"
        assert record.provenance["script"]["note"] == "ripple"
        assert record.provenance["script"]["argv0"] and record.provenance["script"]["call"]
        assert job.section_maps(ID0).residual is not None
        assert job.look(mode="overlay", sections=[ID0])["pictures"]
        job.undo()
        assert job.state.by_id(ID0).deformation is None


def test_frozen_properties_and_tasks_that_are_off_refuse_a_scripts_write(images: Path):
    import langslice
    from tests.fixtures.example_plugin import ripple_map

    with make(images, frozen={"*": ["deformation"], ID1: ["affine"]}) as job:
        c = job.context
        done = c.set_deformations([{"id": ID0, "map": ripple_map(c.fit_frame(ID0), 1.0)}])
        assert done.rows[0]["error"] == "FROZEN" and not done.written
        moved = c.set_transforms([c.transform_from_matrix(ID1, shift(c.fit_frame(ID1), 3)).row()])
        assert moved.rows[0]["error"] == "FROZEN" and not moved.written
    with langslice.create_job(images, positions=POSITIONS, transforms=TRANSFORMS,
                              pixel_size_um=PIXEL_SIZE_UM, preprocess="none",
                              tasks=["transform"], fresh=True) as job:
        c = job.context
        done = c.set_deformations([{"id": ID0, "map": ripple_map(c.fit_frame(ID0), 1.0)}])
        assert done.rows[0]["error"] == "FROZEN"
    # Switching a tool off does not gate a script's write of what it writes.
    with make(images, disabled=["fireants_syn"], fresh=True) as job:
        c = job.context
        done = c.set_deformations([{"id": ID0, "map": ripple_map(c.fit_frame(ID0), 1.0)}])
        assert done.written == [ID0]


def test_a_section_moved_since_the_digest_is_stale_and_nothing_is_written(images: Path):
    from tests.fixtures.example_plugin import ripple_map

    with make(images, tasks=["position", "transform", "nonlinear"]) as job:
        c = job.context
        digest = c.section_inputs(ID0)
        job.position_sections(sections=[{"id": ID0, "position_mm": 0.12}])
        late = c.set_deformations([{"id": ID0, "map": ripple_map(c.fit_frame(ID0), 1.0)}],
                                  expect={ID0: digest})
        assert late.rows[0]["error"] == "STALE_INPUT" and not late.written
        assert job.state.by_id(ID0).deformation is None
        landed = c.set_deformations([{"id": ID0, "map": ripple_map(c.fit_frame(ID0), 1.0)}])
        assert landed.written == [ID0]  # without expect= the script chose not to check
        assert not [entry for entry in events(job) if not entry["sections"]]
        assert len(events(job)) == 1


def test_another_handles_write_is_seen_by_the_next_read_and_refused_by_expect(images: Path):
    import langslice

    make(images).close()
    with langslice.open_job(images) as script, langslice.open_job(images) as agent:
        c = script.context
        digest = c.section_inputs(ID1)
        agent.interactive_transform(sections=[{"id": ID1, "translate_x_mm": 0.05}])
        assert c.section(ID1).affine is not None and c.section(ID1).affine.kind == "interactive"
        frame = c.fit_frame(ID1)
        done = c.set_transforms([c.transform_from_matrix(ID1, shift(frame, 2.0)).row()],
                                expect={ID1: digest})
        assert done.rows[0]["error"] == "STALE_INPUT"


def test_each_outermost_write_is_one_step_and_a_writing_block_is_one(images: Path):
    with make(images) as job:
        c = job.context
        undo = job.job.undo_stack
        depth = len(undo)
        for dx in (2.0, 4.0):
            c.set_transforms([c.transform_from_matrix(ID0, shift(c.fit_frame(ID0), dx)).row()])
        assert len(undo) == depth + 2
        with c.writing():
            for dx in (2.0, 4.0):
                c.set_transforms([c.transform_from_matrix(ID1, shift(c.fit_frame(ID1),
                                                                     dx)).row()])
        assert len(undo) == depth + 3
        calls = [entry["call"] for entry in events(job)]
        assert len(calls) == 4 and len(set(calls)) == 3 and calls[2] == calls[3]


def test_a_preview_is_saved_numbered_and_writes_nothing(images: Path):
    from PIL import Image

    from tests.fixtures.example_plugin import ripple_map

    with make(images, tasks=["position", "transform", "nonlinear"]) as job:
        c = job.context
        stamp = (Path(job.folder) / "state.json").read_bytes()
        saved = c.preview_overlay(ID0, {"deformation": ripple_map(c.fit_frame(ID0), 2.0)})
        assert saved.path.is_file() and saved.path.parent.name.endswith("_script_overlay")
        assert saved.path.parent.parent.name == "views" and "NOT written" in saved.caption
        with Image.open(saved.path) as opened:
            assert opened.size == saved.image.size
        view = json.loads((saved.path.parent / "view.json").read_text())
        assert view["recipe"]["renderer"] == "transport_snapshot" and view["frame"] is None
        linear = c.preview_overlay(ID0, {"position_mm": 0.12})
        assert json.loads((linear.path.parent / "view.json").read_text())["frame"] is not None
        assert (linear.path.parent / "labels.tif").is_file()
        zoomed = job.zoom(picture=saved.index, box=[10, 10, 120, 90])
        assert zoomed["redrawn"] is False
        positioned = c.preview_positioning({ID1: {"position_mm": 0.18}})
        assert positioned and all(item.path.is_file() for item in positioned)
        assert positioned[0].path.parent.name.endswith("_script_positioning")
        assert (Path(job.folder) / "state.json").read_bytes() == stamp
        assert not events(job)


# --- the engine I/O bridge, points, atlas planes ------------------------------------------


def test_an_outside_engines_field_and_affine_come_back_through_the_bridge(images: Path,
                                                                          tmp_path: Path):
    import SimpleITK as sitk

    with make(images) as job:
        c = job.context
        files = c.write_frame(ID0, tmp_path / "frame", format="nii.gz")
        frame = json.loads(files["frame"].read_text())
        width, height = frame["fit_size"]
        assert sitk.ReadImage(str(files["stain"])).GetSize() == (width, height)
        assert sitk.ReadImage(str(files["placed_template"])).GetSize() == (width, height)
        # An engine with the section fixed answering "move 3 px right": an ITK field file.
        field = np.zeros((height, width, 2))
        field[..., 0] = 3.0
        image = sitk.GetImageFromArray(field, isVector=True)
        sitk.WriteImage(image, str(tmp_path / "warp.nii.gz"))
        native = c.map_from_itk(ID0, tmp_path / "warp.nii.gz")
        placed = np.asarray(frame["placed_atlas_to_section"])
        back = np.einsum("ij,hwj->hwi", placed[:2],
                         np.dstack([native, np.ones((height, width))]))
        yy, xx = np.indices((height, width))
        np.testing.assert_allclose(back[..., 0], xx + 3.0, atol=1e-6)
        np.testing.assert_allclose(back[..., 1], yy, atol=1e-6)
        # The same shift with the atlas fixed: inverted on the fit grid.
        flipped = c.map_from_itk(ID0, -field, fixed="atlas")
        np.testing.assert_allclose(flipped[5:-5, 5:-5], native[5:-5, 5:-5], atol=0.05)
        # An affine (ANTs writes .mat; any ITK transform file reads the same).
        affine = sitk.TranslationTransform(2, (3.0, 0.0))
        sitk.WriteTransform(affine, str(tmp_path / "affine.tfm"))
        proposal = c.transform_from_itk(ID0, tmp_path / "affine.tfm", note="itk")
        expected = c.transform_from_matrix(ID0, shift(c.fit_frame(ID0), 3.0))
        np.testing.assert_allclose(proposal.params, expected.params, atol=1e-9)
        from langslice.core.plugins import ToolError

        bumpy = sitk.DisplacementFieldTransform(sitk.GetImageFromArray(
            np.random.default_rng(0).normal(size=(height, width, 2)), isVector=True))
        with pytest.raises(ToolError) as refused:
            c.transform_from_itk(ID0, bumpy)
        assert refused.value.code == "BAD_MATRIX"


def test_points_go_to_the_atlas_and_back_through_the_deformation(images: Path):
    from tests.fixtures.example_plugin import ripple_map

    with make(images) as job:
        c = job.context
        c.set_deformations([{"id": ID0, "map": ripple_map(c.fit_frame(ID0), 3.0)}])
        xy = np.array([[120.0, 100.0], [200.0, 140.0], [170.5, 90.25]])
        um = c.to_atlas_um(ID0, xy)
        maps = job.section_maps(ID0, full_resolution=True)
        np.testing.assert_allclose(um[0], maps.coords[100, 120], atol=0.5)
        np.testing.assert_allclose(c.from_atlas_um(ID0, um), xy, atol=0.3)
        assert np.all(c.plane_distance_um(ID0, um) < 1e-3)
        regions = c.region_at(ID0, xy)
        assert regions[0]["id"] == int(maps.labels[100, 120])
        assert set(regions[0]) == {"id", "acronym", "name"}
        fit_xy = c.from_atlas_um(ID0, um, frame="fit")
        np.testing.assert_allclose(c.to_atlas_um(ID0, fit_xy, frame="fit"), um, atol=0.5)


def test_an_atlas_plane_anywhere_matches_the_sections_own(images: Path):
    with make(images) as job:
        c = job.context
        frame = c.fit_frame(ID0)
        plane = c.atlas_plane(frame.position_mm, pitch_deg=frame.angles[0],
                              yaw_deg=frame.angles[1])
        np.testing.assert_array_equal(plane.atlas["annotation"], frame.atlas["annotation"])
        np.testing.assert_allclose(plane.atlas["template"], frame.atlas["template"])
        assert plane.plane["native_to_um"] == frame.plane["native_to_um"]
        assert c.atlas_volume("annotation").ndim == 3
        assert c.atlas_volume().shape == c.atlas_volume("annotation").shape


# --- the native engines with settings of the script's own ----------------------------------


def test_fireants_with_settings_and_a_mask_is_a_proposal_until_written(images: Path):
    with make(images) as job:
        c = job.context
        from langslice.core.plugins import ToolError
        from langslice.ops import masks as ops_masks

        frame = c.fit_frame(ID0)
        with pytest.raises(ToolError) as array_refused:  # a mask is a name, not an array
            c.fit_fireants(ID0, mask=np.ones(frame.image.shape[:2], dtype=bool))
        assert array_refused.value.code == "BAD_ARGS" and "make_mask" in str(array_refused.value)
        with pytest.raises(ToolError) as unknown:
            c.fit_fireants(ID0, mask="nope")
        assert unknown.value.code == "UNKNOWN_MASK"
        ops_masks.make_mask(job.job, job.workspace, "tissue", [ID0], source="em")
        assert c.mask(ID0, "tissue").kind == "em"
        proposal = c.fit_fireants(ID0, settings={"smoothness_um": 50.0, "detail": "coarse"},
                                  mask="tissue", note="soft, em mask")
        assert job.state.by_id(ID0).deformation is None
        done = c.set_deformations([proposal.row()])
        assert done.written == [ID0], done.rows
        record = job.job.deformations.current(job.state, job.state.by_id(ID0))
        meta = record.provenance["script"]["meta"]
        assert meta["settings"]["smoothness_um"] == 50.0 and meta["engine"] == "fireants"
        assert meta["mask"] == "tissue"
        assert record.provenance["inputs"]["mask"]["name"] == "tissue"
        assert record.provenance["inputs"]["atlas_image"] == "nissl"
        assert job.state.by_id(ID0).deformation["steps"][-1]["method"] == "script"
        from langslice.core.plugins import ToolError

        with pytest.raises(ToolError) as refused:
            c.fit_fireants(ID0, settings={"no_such": 1})
        assert refused.value.code == "BAD_ARGS"


def test_elastix_with_a_mask_is_a_proposal_recorded_with_its_settings(images: Path):
    with make(images) as job:
        c = job.context
        frame = c.fit_frame(ID1)
        from langslice.core.plugins import ToolError

        with pytest.raises(ToolError) as array_refused:
            c.fit_elastix(ID1, mask=np.ones(frame.image.shape[:2], dtype=bool))
        assert array_refused.value.code == "BAD_ARGS"
        proposal = c.fit_elastix(ID1)
        assert proposal.record["kind"] == "script"
        assert proposal.record["fit"]["engine"] == "elastix"
        assert proposal.record["fit"]["mask"] == "none"
        assert proposal.record["fit"]["inputs"]["atlas_image"] == "nissl"
        assert job.state.by_id(ID1).transform["params"] == TRANSFORMS[ID1]
        assert c.set_transforms([proposal.row()]).written == [ID1]


# --- the script's layers on look ---------------------------------------------------------


def test_a_scripts_layer_is_drawn_by_look_on_a_job_without_plugins(images: Path):
    from langslice.core.plugins import Layer

    with make(images) as job:
        assert "plugins" not in json.loads((Path(job.folder) / "job.json").read_text())["spec"]
        job.context.add_layer(ID0, Layer.points([[100.0, 80.0], [150.0, 120.0]],
                                                frame="section_px", name="cells"))
        rows = {row["id"]: row for row in job.status()["rows"]}
        assert rows[ID0]["layers"][0]["name"] == "cells"
        reply = job.look(mode="overlay", sections=[ID0], layers=["cells"])
        drawn = reply["layers"][0]
        assert drawn["name"] == "cells" and drawn["drawn"] is True and drawn["tool"] == "script"
        job.undo()
        assert not job.state.by_id(ID0).layers
