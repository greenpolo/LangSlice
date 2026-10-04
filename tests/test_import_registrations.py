"""Importing a linear registration made elsewhere (QuickNII, VisuAlign,
DeepSlice, an earlier LangSlice job) as LangSlice placements.

Round trips: sections are given varied placements (positions, quarter turns,
flips, rotations, anisotropic scale, shear, cutting angles, per section when
the file can carry them), drawn through the job's own geometry
(``core.maps.section_frame``), exported by the job's own writers
(``job.quint.job_export``, ``job.formats.registration_document``), read back
by ``job.imports`` and inverted by ``core.import_geometry``; the recovered
placement must be the one given. The DeepSlice files under
``tests/fixtures/deepslice/`` were written by DeepSlice 1.2.8's own writers
(``write_QUINT_JSON``, ``write_QuickNII_XML``, and ``save_predictions``'s
``DataFrame.to_csv``) from this module's :func:`deepslice_case` export, by
``tests/fixtures/deepslice/convert_with_deepslice.py`` run in DeepSlice's
environment (no network, no prediction: only its writers).
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from langslice.core.affine import normalized_physical_affine, pixel_center_map
from langslice.core.import_geometry import (
    ANGLE_EPSILON_DEG,
    implied_pixel_size_um,
    placement_from_pixel_map,
    plane_angles,
)
from langslice.core.layers import atlas_facts
from langslice.core.maps import SWAP, render_sizes, section_frame
from langslice.core.oblique import build_rotation_matrix, plane_axes
from langslice.core.spec import JobSpec
from langslice.core.state import SliceState, StackState
from langslice.core.workspace import Workspace
from langslice.job import imports
from langslice.job.formats import registration_document
from langslice.job.layout import JobLayout
from langslice.job.quint import job_export
from tests.deformable_synthetic import CTX, STR, SyntheticAtlas
from tests.golden.record import ID0, ID1, ID2, PIXEL_SIZE_UM, write_sections

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "deepslice"
#: Recovered six numbers and angles agree with the given ones to this.
TOL_PARAMS = 1e-9
TOL_DEG = 1e-7
TOL_MM = 1e-9
#: Through a QuickNII file: its anchorings are rounded to 1e-6 voxels.
ROUNDED = 1000.0


class DeepAtlas(SyntheticAtlas):
    """The synthetic atlas, 40 sections deep (2 mm at 50 um) with anatomy
    that changes along AP, so positions and angles change what is drawn."""

    def __init__(self, name: str = "synthetic_deep_50um") -> None:
        super().__init__()
        self.atlas_name = name
        depth = 40
        annotation = np.zeros((depth, 120, 160), dtype=np.int32)
        for index in range(depth):
            plane = annotation[index]
            cv2.ellipse(plane, (80, 62), (min(75, 40 + index), 30 + index // 2), 0, 0, 360,
                        CTX, -1)
            cv2.ellipse(plane, (52 + index // 4, 58), (14, 10), 0, 0, 360, STR, -1)
        self.annotation = annotation
        self.template = (annotation > 0).astype(np.float32) * 100.0
        self.reference = self.template


@dataclass(frozen=True)
class Given:
    """One section's placement as given."""

    section_id: str
    position_mm: float
    pitch_deg: float
    yaw_deg: float
    rotation_deg: int
    flip: bool
    knobs: dict[str, float]


def make_workspace(folder: Path, atlas: Any, *, plane: str = "coronal",
                   pixel_size: float | None = PIXEL_SIZE_UM) -> Workspace:
    inputs = {} if pixel_size is None else {"pixel_size_um": pixel_size}
    spec = JobSpec(image_folder=str(folder), atlas=atlas.atlas_name, plane=plane,
                   inputs=inputs, preprocess="none")
    return Workspace(spec=spec, image_folder=str(folder), atlas_loader=lambda _name: atlas)


def params_for(workspace: Workspace, given: Given) -> list[float]:
    sizes = render_sizes(workspace, given.section_id)
    unturned = sizes.unturned_render
    render = (unturned[1], unturned[0]) if given.rotation_deg % 180 == 90 else unturned
    file_um, _source = workspace.calibration(given.section_id)
    assert file_um is not None
    return normalized_physical_affine(size=render, um_per_px=file_um * sizes.render_scale,
                                      **{**knobs(0.0, 1.0, 1.0, 0.0, 0.0, 0.0), **given.knobs})


def state_of(workspace: Workspace, sections: list[Given]) -> StackState:
    """A state holding *sections* as given, each at its own cutting angles."""
    records = [
        SliceState(id=g.section_id, index_original=i, index_corrected=i, flip=g.flip,
                   rotation_deg=g.rotation_deg, position_mm=g.position_mm,
                   transform={"kind": "interactive", "params": params_for(workspace, g)},
                   cutting_angles_deg={"pitch": g.pitch_deg, "yaw": g.yaw_deg})
        for i, g in enumerate(sections)
    ]
    return StackState(image_folder=workspace.image_folder, atlas=workspace.spec.atlas,
                      plane=workspace.spec.plane, slices=records)


def pixel_map(workspace: Workspace, given: Given) -> np.ndarray:
    """The section's file pixel -> atlas map as the job draws it, at its own
    cutting angles."""
    state = state_of(workspace, [given])
    return section_frame(state, workspace, state.slices[0]).pixel_to_atlas_um()


def export_rows(workspace: Workspace, sections: list[Given],
                markers: dict[str, list[list[float]]] | None = None) -> list[dict[str, Any]]:
    rows = []
    for index, given in enumerate(sections):
        size = render_sizes(workspace, given.section_id).file_size
        rows.append({"filename": given.section_id, "width": size[0], "height": size[1],
                     "nr": index + 1, "pixel_to_atlas_um": pixel_map(workspace, given),
                     "markers": (markers or {}).get(given.section_id, [])})
    return rows


def assert_recovered(found: Any, given: Given, expected_params: list[float],
                     loose: float = 1.0) -> None:
    assert (found.rotation_deg, found.flip) == (given.rotation_deg, given.flip)
    assert found.position_mm == pytest.approx(given.position_mm, abs=TOL_MM * loose)
    assert found.pitch_deg == pytest.approx(given.pitch_deg, abs=TOL_DEG * loose)
    assert found.yaw_deg == pytest.approx(given.yaw_deg, abs=TOL_DEG * loose)
    assert np.allclose(found.params, expected_params, atol=TOL_PARAMS * loose, rtol=0)


def knobs(rotation: float, sx: float, sy: float, tx: float, ty: float,
          shear: float) -> dict[str, float]:
    return {"rotation_deg": rotation, "scale_x": sx, "scale_y": sy, "translate_x_mm": tx,
            "translate_y_mm": ty, "shear": shear}


#: Three sections, every orientation knob and an affine of each kind, with
#: DIFFERENT cutting angles per section (QuickNII files carry them per section).
PER_SECTION = [
    Given(ID0, 0.62, 3.5, -2.0, 0, False, knobs(12.0, 1.1, 0.92, 0.2, -0.1, 0.08)),
    Given(ID1, 0.91, -1.25, 4.75, 90, True, knobs(-30.0, 0.85, 1.2, -0.3, 0.25, -0.15)),
    Given(ID2, 1.33, 0.4, 0.0, 270, False, knobs(41.0, 1.0, 1.0, 0.0, 0.0, 0.0)),
]


@pytest.fixture(scope="module")
def stack(tmp_path_factory: pytest.TempPathFactory) -> Path:
    folder = tmp_path_factory.mktemp("imports") / "stack"
    write_sections(folder)
    return folder


# --- the core inversion --------------------------------------------------------------


def test_angle_convention_inverts_the_rotation_builder():
    atlas = DeepAtlas()
    for plane in ("coronal", "sagittal", "horizontal"):
        normal_axis, row_axis, col_axis = plane_axes(atlas, plane)  # type: ignore[arg-type]
        for pitch, yaw in ((0.0, 0.0), (7.5, 0.0), (0.0, -9.0), (-12.0, 21.0), (30.0, 30.0)):
            rotation = build_rotation_matrix(pitch, yaw, row_axis=row_axis, col_axis=col_axis)
            for sign in (1.0, -1.0):
                found = plane_angles(sign * rotation[:, normal_axis], atlas,
                                     plane)  # type: ignore[arg-type]
                assert found == pytest.approx((pitch, yaw), abs=1e-9)


@pytest.mark.parametrize("plane", ["coronal", "sagittal", "horizontal"])
def test_core_round_trip_is_exact(stack: Path, plane: str):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas, plane=plane)
    rng = np.random.default_rng({"coronal": 1, "sagittal": 2, "horizontal": 3}[plane])
    for trial in range(24):
        flat = trial % 4 == 0
        given = Given(
            section_id=(ID0, ID1, ID2)[trial % 3],
            # A flat plane is drawn on a voxel: give one that lies on it.
            position_mm=(0.05 * int(rng.integers(8, 30)) if flat
                         else float(rng.uniform(0.4, 1.5))),
            pitch_deg=0.0 if flat else float(rng.uniform(-6, 6)),
            yaw_deg=0.0 if flat else float(rng.uniform(-6, 6)),
            rotation_deg=int(rng.choice([0, 90, 180, 270])), flip=bool(rng.integers(2)),
            knobs=knobs(float(rng.uniform(-40, 40)), float(rng.uniform(0.8, 1.25)),
                        float(rng.uniform(0.8, 1.25)), float(rng.uniform(-0.5, 0.5)),
                        float(rng.uniform(-0.5, 0.5)), float(rng.uniform(-0.2, 0.2))))
        found = placement_from_pixel_map(
            pixel_map(workspace, given), atlas=atlas, plane=plane,  # type: ignore[arg-type]
            sizes=render_sizes(workspace, given.section_id), file_um_per_px=PIXEL_SIZE_UM)
        assert_recovered(found, given, params_for(workspace, given))
        assert found.out_of_plane_um < 1e-6 and found.in_plane_error_px < 1e-6


def test_a_fixed_orientation_still_reproduces_the_map(stack: Path):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    given = PER_SECTION[1]
    matrix = pixel_map(workspace, given)
    sizes = render_sizes(workspace, given.section_id)
    found = placement_from_pixel_map(matrix, atlas=atlas, plane="coronal", sizes=sizes,
                                     file_um_per_px=PIXEL_SIZE_UM, orientation=(0, False))
    assert (found.rotation_deg, found.flip) == (0, False)
    assert found.in_plane["mirrored"]  # the flip is carried by the six numbers
    again = Given(given.section_id, found.position_mm, found.pitch_deg, found.yaw_deg, 0,
                  False, {})
    state = state_of(workspace, [again])
    state.slices[0].transform = {"params": list(found.params)}
    redrawn = section_frame(state, workspace, state.slices[0]).pixel_to_atlas_um()
    assert np.allclose(redrawn, matrix, atol=1e-7, rtol=0)


def test_another_pixel_size_moves_the_scale_into_the_six_numbers(stack: Path):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    given = PER_SECTION[0]
    matrix = pixel_map(workspace, given)
    found = placement_from_pixel_map(matrix, atlas=atlas, plane="coronal",
                                     sizes=render_sizes(workspace, given.section_id),
                                     file_um_per_px=2.0 * PIXEL_SIZE_UM)
    redrawn_workspace = make_workspace(stack, atlas, pixel_size=2.0 * PIXEL_SIZE_UM)
    state = state_of(redrawn_workspace, [Given(given.section_id, found.position_mm,
                                               found.pitch_deg, found.yaw_deg,
                                               found.rotation_deg, found.flip, {})])
    state.slices[0].transform = {"params": list(found.params)}
    redrawn = section_frame(state, redrawn_workspace, state.slices[0]).pixel_to_atlas_um()
    assert np.allclose(redrawn, matrix, atol=1e-7, rtol=0)
    assert found.in_plane["scale_x"] == pytest.approx(
        0.5 * params_scale(workspace, given), rel=1e-3)


def params_scale(workspace: Workspace, given: Given) -> float:
    from langslice.core.affine import decompose_affine

    sizes = render_sizes(workspace, given.section_id)
    render = sizes.unturned_render
    return float(decompose_affine(params_for(workspace, given), render)["scale_x"])


def test_a_flat_plane_between_voxels_is_drawn_on_the_nearest(stack: Path):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    given = Given(ID0, 0.5, 0.0, 0.0, 0, False, knobs(5.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    matrix = pixel_map(workspace, given).copy()
    matrix[0, 2] += 12.0  # 12 um further along AP: between voxels (50 um apart)
    found = placement_from_pixel_map(matrix, atlas=atlas, plane="coronal",
                                     sizes=render_sizes(workspace, ID0),
                                     file_um_per_px=PIXEL_SIZE_UM)
    assert (found.pitch_deg, found.yaw_deg) == (0.0, 0.0)
    assert found.position_mm == pytest.approx(0.512, abs=1e-9)
    assert found.out_of_plane_um == pytest.approx(12.0, abs=1e-6)
    assert np.allclose(found.params, params_for(workspace, given), atol=1e-9)


def test_tiny_angles_are_rounding_and_other_planes_are_refused(stack: Path):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    sizes = render_sizes(workspace, ID0)
    given = Given(ID0, 0.75, ANGLE_EPSILON_DEG / 10, 0.0, 0, False,
                  knobs(0.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    found = placement_from_pixel_map(pixel_map(workspace, given), atlas=atlas,
                                     plane="coronal", sizes=sizes,
                                     file_um_per_px=PIXEL_SIZE_UM)
    assert (found.pitch_deg, found.yaw_deg) == (0.0, 0.0)
    sagittal = pixel_map(make_workspace(stack, atlas, plane="sagittal"),
                         Given(ID0, 4.0, 0.0, 0.0, 0, False, {}))
    with pytest.raises(ValueError, match="tilted"):
        placement_from_pixel_map(sagittal, atlas=atlas, plane="coronal", sizes=sizes,
                                 file_um_per_px=PIXEL_SIZE_UM)
    with pytest.raises(ValueError, match="degenerate"):
        placement_from_pixel_map(np.zeros((3, 3)), atlas=atlas, plane="coronal", sizes=sizes,
                                 file_um_per_px=PIXEL_SIZE_UM)


# --- through the job's own exports ---------------------------------------------------


def write_export(path: Path, document: dict[str, Any]) -> Path:
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def test_quicknii_json_round_trip_with_per_section_angles(stack: Path, tmp_path: Path):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    document = job_export(export_rows(workspace, PER_SECTION), atlas_facts(atlas))
    result = imports.import_placements(write_export(tmp_path / "quicknii.json", document),
                                       workspace)
    assert result.source.format == "quicknii-json"
    assert (result.unmatched, result.missing, result.refused) == ([], [], {})
    assert result.pixel_size_source == "host" and result.pixel_size_um == PIXEL_SIZE_UM
    found = result.by_section()
    for given in PER_SECTION:
        placement = found[given.section_id]
        assert placement.match == "exact" and placement.problems == []
        assert_recovered(placement, given, params_for(workspace, given), ROUNDED)
    # Per section, not one stack angle: the reader keeps each section's own.
    assert len({(p.pitch_deg, p.yaw_deg) for p in result.placements}) == 3
    transform = found[ID1].transform()
    assert transform["kind"] == "imported" and transform["mirrored"] is False
    assert transform["calibration"]["source"] == "host"


def test_quicknii_on_a_downsampled_copy_carries_over(stack: Path, tmp_path: Path):
    """A registration made on half-size PNG copies (fractions of the image)
    places the job's full-size files identically."""
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    rows = export_rows(workspace, PER_SECTION)
    for row in rows:
        size = (row["width"], row["height"])
        small = (size[0] // 2, size[1] // 2)
        row["pixel_to_atlas_um"] = row["pixel_to_atlas_um"] @ (
            SWAP @ pixel_center_map(small, size) @ SWAP)
        row["width"], row["height"] = small
        row["filename"] = "copies/" + row["filename"].replace(".png", ".PNG")
    path = write_export(tmp_path / "small.json", job_export(rows, atlas_facts(atlas)))
    result = imports.import_placements(path, workspace)
    for given in PER_SECTION:
        placement = result.by_section()[given.section_id]
        assert placement.match == "stem"
        # half sizes are floored: a slightly different aspect, said, same fractions
        assert all("fractions" in problem for problem in placement.problems)
        assert (placement.rotation_deg, placement.flip) == (given.rotation_deg, given.flip)
        assert placement.position_mm == pytest.approx(given.position_mm, abs=1e-6)
        assert placement.pitch_deg == pytest.approx(given.pitch_deg, abs=1e-6)


def test_allen_target_grid_round_trip(stack: Path, tmp_path: Path):
    """An Allen atlas at another resolution exports to the 25 um target;
    reading it back undoes the grid change."""
    atlas = DeepAtlas("allen_mouse_50um")
    workspace = make_workspace(stack, atlas)
    document = job_export(export_rows(workspace, PER_SECTION), atlas_facts(atlas))
    assert document["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
    result = imports.import_placements(write_export(tmp_path / "allen.json", document),
                                       workspace)
    for given in PER_SECTION:
        assert_recovered(result.by_section()[given.section_id], given,
                         params_for(workspace, given), ROUNDED)
    # Another target for this atlas is refused, section by section.
    document["target"] = "WHS_Rat_v4_39um.cutlas"
    refused = imports.import_placements(write_export(tmp_path / "rat.json", document),
                                        workspace)
    assert refused.placements == [] and set(refused.refused) == {ID0, ID1, ID2}
    assert "WHS_Rat_v4_39um.cutlas" in refused.refused[ID0]


def test_visualign_markers_come_back_on_the_file(stack: Path, tmp_path: Path):
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    markers = {ID0: [[10.5, 20.5, 12.0, 19.0], [100.0, 80.0, 98.5, 83.25]]}
    document = job_export(export_rows(workspace, PER_SECTION, markers), atlas_facts(atlas))
    result = imports.import_placements(write_export(tmp_path / "visualign.json", document),
                                       workspace)
    assert result.source.format == "visualign-json"
    found = result.by_section()
    assert found[ID0].markers == markers[ID0] and found[ID0].markers_raw == markers[ID0]
    assert found[ID1].markers is None and found[ID1].markers_raw == []
    assert_recovered(found[ID0], PER_SECTION[0], params_for(workspace, PER_SECTION[0]),
                     ROUNDED)


def test_registration_json_seeds_a_new_job(stack: Path, tmp_path: Path):
    """A finished job's registration.json gives back every section's
    placement, whether the stack shares one cutting angle or each section
    has its own."""
    atlas = DeepAtlas()
    workspace = make_workspace(stack, atlas)
    shared = [Given(g.section_id, g.position_mm, 2.25, -1.5, g.rotation_deg, g.flip, g.knobs)
              for g in PER_SECTION]
    for index, sections in enumerate((shared, PER_SECTION)):
        state = state_of(workspace, sections)
        assert state.mixed_angles is (sections is PER_SECTION)
        document = registration_document(state, workspace,
                                         JobLayout(tmp_path / f"job{index}", stack))
        assert all(entry["problem"] is None for entry in document["sections"])
        path = write_export(tmp_path / f"registration{index}.json", document)
        result = imports.import_placements(path, workspace)
        assert result.source.format == "langslice-registration"
        for given in sections:
            placement = result.by_section()[given.section_id]
            assert_recovered(placement, given, params_for(workspace, given))
            record = state.by_id(given.section_id)
            assert record is not None and record.transform is not None
            assert np.allclose(placement.params, record.transform["params"], atol=TOL_PARAMS)
    # The same registration into a job on another atlas of the same target.
    other = DeepAtlas("allen_mouse_50um")
    elsewhere = make_workspace(stack, other)
    document_allen = registration_document(state_of(elsewhere, shared), elsewhere,
                                           JobLayout(tmp_path / "job2", stack))
    refused = imports.import_placements(
        write_export(tmp_path / "allen_registration.json", document_allen), workspace)
    assert set(refused.refused) == {ID0, ID1, ID2}
    assert "no common QuickNII target" in refused.refused[ID0]


def test_without_a_pixel_size_the_imported_one_is_used(stack: Path, tmp_path: Path):
    atlas = DeepAtlas()
    calibrated = make_workspace(stack, atlas)
    document = job_export(export_rows(calibrated, PER_SECTION), atlas_facts(atlas))
    path = write_export(tmp_path / "quicknii.json", document)
    bare = make_workspace(stack, atlas, pixel_size=None)
    result = imports.import_placements(path, bare)
    assert result.pixel_size_source == "imported"
    implied = [implied_pixel_size_um(p.pixel_to_atlas_um) for p in result.placements]
    assert result.pixel_size_um == pytest.approx(float(np.median(implied)))
    # Given that pixel size, the job draws exactly the imported maps.
    assert result.pixel_size_um is not None
    redrawn = make_workspace(stack, atlas, pixel_size=result.pixel_size_um)
    for placement in result.placements:
        assert placement.pixel_size_source == "imported"
        given = Given(placement.section_id, placement.position_mm, placement.pitch_deg,
                      placement.yaw_deg, placement.rotation_deg, placement.flip, {})
        state = state_of(redrawn, [given])
        state.slices[0].transform = {"params": list(placement.params)}
        frame = section_frame(state, redrawn, state.slices[0])
        assert np.allclose(frame.pixel_to_atlas_um(), placement.pixel_to_atlas_um,
                           atol=1e-6, rtol=0)


# --- DeepSlice's own files ---------------------------------------------------------------


#: The DeepSlice case: the stack's files renamed with QuickNII section
#: numbers, the registration made on files named otherwise (DeepSlice keeps
#: the folder it read them from), an Allen atlas (its 25 um target).
DEEPSLICE_NAMES = {ID0: "M1_s001.png", ID1: "M1_s002.png", ID2: "M1_s003.png"}
DEEPSLICE_SECTIONS = [
    Given(DEEPSLICE_NAMES[g.section_id], g.position_mm, g.pitch_deg, g.yaw_deg,
          g.rotation_deg, g.flip, g.knobs) for g in PER_SECTION
]


def deepslice_stack(stack: Path, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    for old, new in DEEPSLICE_NAMES.items():
        shutil.copy2(stack / old, folder / new)
    return folder


def deepslice_case(stack: Path, folder: Path) -> tuple[Workspace, dict[str, Any]]:
    """The workspace and the LangSlice QuickNII export the DeepSlice fixtures
    were converted from (entry names as DeepSlice writes them)."""
    atlas = DeepAtlas("allen_mouse_50um")
    workspace = make_workspace(deepslice_stack(stack, folder), atlas)
    rows = export_rows(workspace, DEEPSLICE_SECTIONS)
    for row in rows:
        row["filename"] = row["filename"].replace(".png", "_10x_A.png")
    return workspace, job_export(rows, atlas_facts(atlas))


@pytest.mark.parametrize("name", ["deepslice.json", "deepslice.xml", "deepslice.csv"])
def test_deepslice_files_round_trip(stack: Path, tmp_path: Path, name: str):
    workspace, _document = deepslice_case(stack, tmp_path / "M1")
    result = imports.import_placements(FIXTURES / name, workspace)
    assert result.source.format == {"deepslice.json": "quicknii-json",
                                    "deepslice.xml": "quicknii-xml",
                                    "deepslice.csv": "deepslice-csv"}[name]
    assert (result.unmatched, result.missing, result.refused) == ([], [], {})
    assert result.source.aligner == (None if name.endswith(".csv") else "prerelease_1.0.0")
    for given in DEEPSLICE_SECTIONS:
        placement = result.by_section()[given.section_id]
        assert placement.match == "section number"
        # The export rounds anchorings to 1e-6 voxels.
        assert (placement.rotation_deg, placement.flip) == (given.rotation_deg, given.flip)
        assert placement.position_mm == pytest.approx(given.position_mm, abs=1e-6)
        assert placement.pitch_deg == pytest.approx(given.pitch_deg, abs=1e-5)
        assert placement.yaw_deg == pytest.approx(given.yaw_deg, abs=1e-5)
        assert np.allclose(placement.params, params_for(workspace, given), atol=1e-6)


def test_the_fixtures_are_this_cases_export(stack: Path, tmp_path: Path):
    """The fixture anchorings are this module's export, unchanged by DeepSlice."""
    _workspace, document = deepslice_case(stack, tmp_path / "M1")
    written = json.loads((FIXTURES / "deepslice.json").read_text())
    assert written["target"] == document["target"]
    for ours, theirs in zip(document["slices"], written["slices"], strict=True):
        assert theirs["filename"].endswith(ours["filename"])
        assert np.allclose(theirs["anchoring"], ours["anchoring"], atol=1e-9)


def test_an_old_deepslice_xml_parses(tmp_path: Path):
    """DeepSlice's earlier releases: bare ampersands, width and height -999,
    nr as a float, a folder in the file name."""
    raw = ('<series aligner="DeepSlice_ver_3.0_python" first="[ 4. 14.]" last="[ 4. 14.]" '
           'name="brain/images/x_s014.png">\n'
           '  <slice anchoring="ox=482.9&oy=446.7&oz=327.6&ux=-483.4&uy=-5.0&uz=-13.7'
           '&vx=1.7&vy=-7.3&vz=-356.9" filename="brain/images/x_s004_10x_A.png" '
           'height="-999" nr="4.0" width="-999" />\n</series>\n')
    file = tmp_path / "old.xml"
    file.write_text(raw)
    read = imports.read_registration(file)
    assert read.format == "quicknii-xml" and read.aligner == "DeepSlice_ver_3.0_python"
    (entry,) = read.entries
    assert entry.nr == 4 and entry.size is None and entry.problem is None
    assert entry.anchoring == (482.9, 446.7, 327.6, -483.4, -5.0, -13.7, 1.7, -7.3, -356.9)
    assert imports.match_sections([entry.filename], ["x_s004.tif", "x_s014.tif"]) == {
        0: ("x_s004.tif", "section number")}


# --- matching ---------------------------------------------------------------------------


def test_matching_rules_in_order():
    ids = ["a.tif", "B_s002.tif", "c_s010.png", "c_s010_copy.png", "d_s003.png"]
    found = imports.match_sections(
        ["a.tif", "dir\\b_S002.PNG", "scan_s003_10x.png", "x/c_s010_copy.jpg", "nothing.png"],
        ids)
    assert found == {0: ("a.tif", "exact"), 1: ("B_s002.tif", "stem"),
                     2: ("d_s003.png", "section number"), 3: ("c_s010_copy.png", "stem")}
    # The exact name wins over a stem two files share.
    assert imports.match_sections(["a.png"], ["a.tif", "a.png"]) == {0: ("a.png", "exact")}
    # "other_s010" names two sections by number: refused.
    with pytest.raises(imports.AmbiguousMatch, match="several sections"):
        imports.match_sections(["other_s010.png"], ids)
    with pytest.raises(imports.AmbiguousMatch, match="several entries"):
        imports.match_sections(["a.png", "a.jpg"], ids)
    # Two section numbers in one name: no number at all.
    assert imports.match_sections(["x_s1_s2.png"], ["y_s1.png"]) == {}
    assert imports.section_number("folder/name_s0042_10x.png") == 42
