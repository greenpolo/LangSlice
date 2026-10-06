"""Cutting angles per section (state format 3).

A job keeps each section's own cutting angles, so a registration made
elsewhere (QuickNII, VisuAlign, DeepSlice) keeps every section's plane as it
was given. Everything LangSlice angles itself stays one plane for the whole
stack: ``set_cutting_angles`` sets every section, and a single-angle job
reads, draws and writes exactly as before (the goldens are unchanged). The
end-to-end tests run on the golden recorder's synthetic stack through the
library and the CLI, with the stub image model (``tests/test_combinations.py``).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from langslice.core.spec import JobSpec, NonlinearSpec, TransformSpec, supplied_angles
from langslice.core.state import MixedAngles, SliceState, StackState
from tests import test_combinations as combinations
from tests.golden.record import ID0, ID1, ID2, PIXEL_SIZE_UM
from tests.test_combinations import (
    EXTERNAL_TRANSFORMS,
    IDS,
    POSITIONS,
    STEMS,
    assert_exported,
    cli,
    create,
    init,
    spec_for,
    transforms_file,
)

# The combination tests' fixtures: the synthetic stack, and a fresh copy of it
# with the synthetic atlas and the stub image model wherever a door opens a job.
stack = combinations.stack
images = combinations.images

#: A registration made elsewhere: every section on its own plane.
SECTION_ANGLES = {
    ID0: {"pitch": 1.0, "yaw": -0.5},
    ID1: {"pitch": 2.5, "yaw": 1.0},
    ID2: {"pitch": -1.5, "yaw": 0.5},
}


def _stack(angles: list[tuple[float, float]] | None = None) -> StackState:
    slices = [SliceState(id=f"s{i}.png", index_original=i, index_corrected=i) for i in range(3)]
    for record, (pitch, yaw) in zip(slices, angles or [], strict=False):
        record.cutting_angles_deg = {"pitch": pitch, "yaw": yaw}
    return StackState(atlas="allen_mouse_25um", slices=slices)


def _external(**extra: Any) -> dict[str, Any]:
    return {"positions": dict(POSITIONS), "transforms": copy.deepcopy(EXTERNAL_TRANSFORMS),
            "angles": copy.deepcopy(SECTION_ANGLES), **extra}


# --- the state ----------------------------------------------------------------------------


def test_a_single_angle_state_serializes_as_before():
    state = _stack()
    state.cutting_angles_deg = {"pitch": 3.0, "yaw": -1.0}
    data = state.to_dict()
    assert data["cutting_angles_deg"] == {"pitch": 3.0, "yaw": -1.0}
    assert all("cutting_angles_deg" not in row for row in data["slices"])
    assert not state.mixed_angles and state.stack_angles == (3.0, -1.0)
    assert StackState.from_dict(data).to_dict() == data


def test_a_mixed_state_keeps_each_sections_angles_and_round_trips():
    state = _stack([(1.0, 0.0), (2.0, -1.0), (1.0, 0.0)])
    assert state.mixed_angles and state.is_oblique
    data = state.to_dict()
    assert data["cutting_angles_deg"] is None
    assert [row["cutting_angles_deg"] for row in data["slices"]] == [
        {"pitch": 1.0, "yaw": 0.0}, {"pitch": 2.0, "yaw": -1.0}, {"pitch": 1.0, "yaw": 0.0}]
    again = StackState.from_dict(json.loads(json.dumps(data)))
    assert [record.angles for record in again.slices] == [(1.0, 0.0), (2.0, -1.0), (1.0, 0.0)]
    assert again.to_dict() == data
    # No stack-wide angle to read: a reader that should use the section's own fails loudly.
    with pytest.raises(MixedAngles):
        _ = state.pitch_deg
    with pytest.raises(MixedAngles):
        _ = state.cutting_angles_deg
    # A picture without a section is drawn at the median of the sections' angles.
    assert state.view_angles == (1.0, 0.0)


def test_undo_restores_each_sections_angles():
    state = _stack([(1.0, 0.0), (2.0, -1.0), (0.0, 0.5)])
    snapshot = state.to_dict()
    state.cutting_angles_deg = {"pitch": 4.0, "yaw": 0.0}
    assert not state.mixed_angles and state.stack_angles == (4.0, 0.0)
    state.restore(snapshot)
    assert [record.angles for record in state.slices] == [(1.0, 0.0), (2.0, -1.0), (0.0, 0.5)]


def test_a_shared_angle_is_stored_on_the_stack_and_read_onto_every_section(tmp_path):
    from langslice.job.checkpoint import (
        FORMAT_KEY,
        STATE_FORMAT_VERSION,
        load_checkpoint,
        write_checkpoint,
    )

    stored = {FORMAT_KEY: STATE_FORMAT_VERSION, "atlas": "allen_mouse_25um",
              "plane": "coronal", "cutting_angles_deg": {"pitch": 2.0, "yaw": -1.0},
              "slices": [{"id": f"s{i}.png", "index_original": i, "index_corrected": i}
                         for i in range(3)]}
    path = tmp_path / "state.json"
    path.write_text(json.dumps(stored))
    state = load_checkpoint(str(path))
    assert state is not None
    assert [record.angles for record in state.slices] == [(2.0, -1.0)] * 3
    assert not state.mixed_angles
    assert state.to_dict()["cutting_angles_deg"] == {"pitch": 2.0, "yaw": -1.0}
    write_checkpoint(state, str(path))
    written = json.loads(path.read_text())
    assert written[FORMAT_KEY] == STATE_FORMAT_VERSION
    assert written["cutting_angles_deg"] == {"pitch": 2.0, "yaw": -1.0}
    assert all("cutting_angles_deg" not in row for row in written["slices"])


# --- the supplied inputs ------------------------------------------------------------------


def test_supplied_angles_reads_both_forms():
    assert supplied_angles(None) == (None, {})
    assert supplied_angles({}) == (None, {})
    assert supplied_angles({"pitch": 2, "yaw": -1}) == ((2.0, -1.0), {})
    assert supplied_angles({"pitch": 2}) == ((2.0, 0.0), {})
    assert supplied_angles({"a.png": {"pitch": 1}, "b.png": {"pitch": 2, "yaw": 3}}) == (
        None, {"a.png": (1.0, 0.0), "b.png": (2.0, 3.0)})


@pytest.mark.parametrize(("value", "words"), [
    ({"pitch": 1.0, "a.png": {"pitch": 2.0}}, "mixes"),
    ({"pitch": 1.0, "roll": 2.0}, "only pitch and yaw"),
    ({"a.png": {"pitch": 1.0, "roll": 0.0}}, "only pitch and yaw"),
    ({"a.png": {"pitch": "steep"}}, "number"),
    ({"pitch": float("nan")}, "finite"),
    ({"a.png": {"yaw": True}}, "number"),
    ([1.0, 2.0], "per section"),
])
def test_a_bad_or_ambiguous_angles_input_is_refused(value, words):
    with pytest.raises(ValueError, match=words):
        supplied_angles(value)
    with pytest.raises(ValueError, match=words):
        JobSpec(image_folder="unused", inputs={"angles": value})


def test_an_unknown_section_in_the_angles_is_refused(images):
    with pytest.raises(ValueError, match="unknown section"):
        create(spec_for(images, ["nonlinear"], **_external(
            angles={**SECTION_ANGLES, "missing.png": {"pitch": 1.0}})))


def test_a_section_not_named_keeps_the_flat_plane(images):
    import langslice

    create(spec_for(images, ["nonlinear"], **_external(angles={ID1: {"pitch": 2.0}})))
    with langslice.open_job(images) as job:
        assert [job.state.by_id(name).angles for name in IDS] == [
            (0.0, 0.0), (2.0, 0.0), (0.0, 0.0)]


# --- kept verbatim from the input to the exports ------------------------------------------


def _uniform_copy(state: StackState, angles: tuple[float, float]) -> StackState:
    copied = StackState.from_dict(state.to_dict())
    copied.cutting_angles_deg = {"pitch": angles[0], "yaw": angles[1]}
    return copied


def _single_angle_frame(job: Any, record_id: str) -> Any:
    """The section's frame in a copy of the stack where EVERY section has its
    angle (a single-angle job), computed afresh."""
    from langslice.core.maps import section_frame

    state = job.state
    record = state.by_id(record_id)
    single = _uniform_copy(state, record.angles)
    job.workspace.frame_cache.clear()
    return section_frame(single, job.workspace, single.by_id(record_id))


def assert_planes_as_supplied(job_folder: Path, job: Any) -> None:
    """``registration.json`` and ``quicknii.json``: each section on its own
    plane, as a single-angle job at that angle would place it."""
    from langslice.core.layers import atlas_facts
    from langslice.job.quint import anchoring_from_pixel_map, to_target_grid

    document = json.loads((job_folder / "registration.json").read_text())
    assert document["cutting_angles_deg"] is None
    quicknii = json.loads((job_folder / "exports" / "quicknii.json").read_text())
    facts = atlas_facts(job.workspace.atlas)
    anchorings = {entry["filename"]: entry["anchoring"] for entry in quicknii["slices"]}
    normals = []
    for entry in document["sections"]:
        supplied = SECTION_ANGLES[entry["id"]]
        plane = entry["parameters"]["plane"]
        assert (plane["pitch_deg"], plane["yaw_deg"]) == (supplied["pitch"], supplied["yaw"])
        frame = _single_angle_frame(job, entry["id"])
        expected = frame.pixel_to_atlas_um()
        np.testing.assert_allclose(entry["pixel_to_atlas_um"], expected, rtol=0, atol=1e-6)
        anchoring = to_target_grid(
            anchoring_from_pixel_map(expected, frame.file_size[0], frame.file_size[1], facts),
            str(facts["name"]), facts["resolution_um"]).to_list()
        np.testing.assert_allclose(anchorings[entry["id"]], anchoring, rtol=0, atol=1e-5)
        u, v = np.asarray(anchoring[3:6]), np.asarray(anchoring[6:9])
        normal = np.cross(u, v)
        normals.append(normal / np.linalg.norm(normal))
    # Three different planes: no two sections share a normal.
    for i in range(3):
        for j in range(i + 1, 3):
            assert abs(float(np.dot(normals[i], normals[j]))) < 1 - 1e-6


def test_per_section_angles_are_kept_through_trace_fit_submit_and_export(images):
    import langslice
    from langslice.core.handoff import prepare_linear_registration

    create(spec_for(images, ["nonlinear"], provider="openai-oauth", **_external()))
    with langslice.open_job(images) as job:
        state = job.state
        assert state.mixed_angles
        assert {name: state.by_id(name).cutting_angles_deg for name in IDS} == SECTION_ANGLES
        status = job.status()
        assert status["cutting_angles_deg"] == "per section"
        assert {row["id"]: row["cutting_angles_deg"] for row in status["rows"]} == (
            SECTION_ANGLES)
        for name in IDS:
            handoff = prepare_linear_registration(state, job.workspace, name)
            assert (handoff.pitch_deg, handoff.yaw_deg) == state.by_id(name).angles
            assert job.trace_borders(id=name)["status"] in ("running", "ok")
        traced = job.fit_deformable(slices=[ID1], fit_section="traced_lines", engine="elastix")
        assert traced["status"] == "ok" and traced["results"][0]["written"] is True, traced
        fit = job.fit_deformable(slices=[ID0], engine="elastix")
        assert fit["status"] == "ok" and fit["results"][0]["written"] is True, fit
        assert job.fit_deformable(slices=[ID2], keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        assert {name: job.state.by_id(name).cutting_angles_deg for name in IDS} == (
            SECTION_ANGLES)
        folder = Path(job.folder)
        # Each fitted record was fitted on its section's own plane.
        for name, stem in ((ID0, "s0"), (ID1, "s1")):
            records = [json.loads(path.read_text()) for path in
                       (folder / "sections" / stem / "deformable").rglob("*.json")]
            placements = [record["placement"] for record in records if "placement" in record]
            assert placements, stem
            assert {(p["pitch_deg"], p["yaw_deg"]) for p in placements} == {
                (SECTION_ANGLES[name]["pitch"], SECTION_ANGLES[name]["yaw"])}
        assert_exported(folder, residual=(ID0, ID1))
        assert_planes_as_supplied(folder, job)


def test_maps_of_a_mixed_stack_match_single_angle_maps(images):
    import tifffile

    import langslice
    from langslice.core.maps import section_maps

    create(spec_for(images, ["nonlinear"], **_external()))
    with langslice.open_job(images) as job:
        assert job.fit_deformable(slices=list(IDS), keep_linear="kept")["status"] == "ok"
        assert job.submit(summary="done", notes=[], interval_breaks=[])["status"] == "ok"
        folder = Path(job.folder)
        flat_differs = 0
        for name, stem in zip(IDS, STEMS, strict=True):
            single = section_maps(job.workspace, _single_angle_frame(job, name), None)
            labels = tifffile.imread(folder / "sections" / stem / "labels.tif")
            np.testing.assert_array_equal(labels, single.labels)
            coords = tifffile.imread(folder / "sections" / stem / "coords.tif")
            np.testing.assert_allclose(np.moveaxis(np.squeeze(coords), 0, -1), single.coords,
                                       rtol=0, atol=1e-3, equal_nan=True)
            # The plane matters: the flat plane gives other coordinates.
            from langslice.core.maps import section_frame

            flat = _uniform_copy(job.state, (0.0, 0.0))
            job.workspace.frame_cache.clear()
            flat_coords = section_maps(job.workspace, section_frame(
                flat, job.workspace, flat.by_id(name)), None).coords
            flat_differs += int(not np.allclose(flat_coords, single.coords, equal_nan=True))
        assert flat_differs == 3


def test_the_cli_takes_per_section_angles(capsys, images):
    job = init(capsys, images, "nonlinear", "--positions", json.dumps(POSITIONS),
               "--transforms", transforms_file(images),
               "--section-angles", json.dumps(SECTION_ANGLES))
    from langslice.job.checkpoint import load_checkpoint

    state = load_checkpoint(str(job / "state.json"))
    assert state is not None
    assert {name: state.by_id(name).cutting_angles_deg for name in IDS} == SECTION_ANGLES
    saved = json.loads((job / "state.json").read_text())
    assert saved["format_version"] == 3 and saved["cutting_angles_deg"] is None


def test_the_cli_refuses_section_angles_with_pitch(capsys, images):
    code, envelope = cli(capsys, str(images), "init", "--tasks", "nonlinear",
                         "--image-provider", "none", "--pitch", "1",
                         "--section-angles", json.dumps(SECTION_ANGLES))
    assert code == 2 and envelope["error"]["code"] == "BAD_ARGUMENTS", envelope
    assert "--pitch/--yaw" in envelope["error"]["message"]


# --- set_cutting_angles: one plane for the whole stack -------------------------------------


def _linear_spec(images: Path, **inputs: Any) -> JobSpec:
    return JobSpec(image_folder=str(images), model="fake-model", preprocess="none",
                   tasks=["transform"], transform=TransformSpec(angles=True),
                   nonlinear=NonlinearSpec(provider="none"), debrief=False,
                   inputs={"pixel_size_um": PIXEL_SIZE_UM, **inputs})


def test_set_cutting_angles_flattens_a_mixed_stack_and_undo_restores_it(images):
    import langslice

    create(_linear_spec(images, **_external()))
    with langslice.open_job(images) as job:
        assert job.state.mixed_angles
        reply = job.set_cutting_angles(pitch_deg=2.0, yaw_deg=-0.5)
        assert reply["status"] == "ok"
        assert reply["cutting_angles_deg"] == {"pitch": 2.0, "yaw": -0.5}
        assert [job.state.by_id(name).angles for name in IDS] == [(2.0, -0.5)] * 3
        document = json.loads((Path(job.folder) / "registration.json").read_text())
        assert document["cutting_angles_deg"] == {"pitch": 2.0, "yaw": -0.5}
        assert job.undo()["status"] == "ok"
        assert {name: job.state.by_id(name).cutting_angles_deg for name in IDS} == (
            SECTION_ANGLES)
        document = json.loads((Path(job.folder) / "registration.json").read_text())
        assert document["cutting_angles_deg"] is None
        assert job.redo()["status"] == "ok"
        assert [job.state.by_id(name).angles for name in IDS] == [(2.0, -0.5)] * 3


def test_a_mixed_stack_is_drawn_per_section_and_said_plainly(images):
    import langslice
    from langslice.agent.prompt import stack_angles_fact
    from langslice.core.status import status_text

    create(_linear_spec(images, **_external()))
    with langslice.open_job(images) as job:
        state = job.state
        assert "differ between sections" in stack_angles_fact(state)
        assert "angles=pitch 2.50 yaw 1.00" in status_text(state)
        shown = job.view_placement(entries=[{"id": ID1, "positions_mm": [POSITIONS[ID1]]}])
        assert shown["status"] == "ok", shown
        job.job.views.flush()
        frames = [json.loads(path.read_text())["frame"] for path in
                  (Path(job.folder) / "sections" / "s1" / "views").glob("*/view.json")]
        planes = {(f["plane"]["pitch_deg"], f["plane"]["yaw_deg"]) for f in frames if f}
        assert planes == {(2.5, 1.0)}  # drawn at the section's own plane
        atlas = job.view_atlas(positions_mm=[0.15])
        assert atlas["status"] == "ok"
        assert atlas["cutting_angles_deg"] == {"pitch": 1.0, "yaw": 0.5}  # the medians
        assert "median" in atlas["description"]
        # A single-angle stack says exactly what it always said.
        job.set_cutting_angles(pitch_deg=1.0, yaw_deg=0.0)
        assert stack_angles_fact(job.state) == (
            "- Stack-wide cutting angles: pitch 1.00 deg, yaw 0.00 deg.")
        assert "angles=" not in status_text(job.state)
        atlas = job.view_atlas(positions_mm=[0.15])
        assert atlas["cutting_angles_deg"] == {"pitch": 1.0, "yaw": 0.0}
        assert "median" not in atlas["description"]


@pytest.mark.parametrize("mode", ["overlay", "side_by_side", "stacked", "template"])
def test_a_sections_pictures_match_a_single_angle_stack_at_its_angle(images, mode):
    """Each section of a mixed stack is pictured exactly as a single-angle
    stack at that section's angle pictures it, pixel for pixel."""
    import langslice
    from langslice.core.display import default_options
    from langslice.core.placement import placement_pictures

    create(spec_for(images, ["nonlinear"], **_external()))
    with langslice.open_job(images) as job:
        workspace, state = job.workspace, job.state
        options = default_options(mode)
        for name in IDS:
            record = state.by_id(name)
            single = _uniform_copy(state, record.angles)
            workspace.picture_cache.clear()
            mixed = placement_pictures(workspace, state, record, POSITIONS[name], options, {})
            workspace.picture_cache.clear()
            alone = placement_pictures(workspace, single, single.by_id(name), POSITIONS[name],
                                       options, {})
            assert len(mixed.images) == len(alone.images)
            for drawn, expected in zip(mixed.images, alone.images, strict=True):
                assert drawn.tobytes() == expected.tobytes(), (name, mode)
