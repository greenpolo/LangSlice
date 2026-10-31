"""The QUINT export: its target grid, the perpendicular anchoring and the
VisuAlign markers that carry what that anchoring drops."""

import numpy as np

from langslice.core.geometry.triangulation import visualign_mesh
from langslice.core.maps.section_maps import MARKER_MARGIN_PX, marker_grid
from langslice.job.quint import (
    AnchoringVector,
    anchoring_from_pixel_map,
    job_export,
    perpendicular_anchoring,
)


def test_finer_and_coarser_allen_atlases_export_to_the_25um_target() -> None:
    """QuickNII and VisuAlign ship the Allen CCFv3 at 25 um only
    (``ABA_Mouse_CCFv3_2017_25um.cutlas``, the target DeepSlice writes too):
    a 10, 50 or 100 um job is exported to it, its anchoring in that
    target's voxel space, the same physical plane as a 25 um job's."""

    def facts(res: float) -> dict:
        return {"name": f"allen_mouse_{int(res)}um", "orientation": "asr",
                "shape": [round(13200 / res), round(8000 / res), round(11400 / res)],
                "resolution_um": [res] * 3}

    def pixel_map(res: float) -> np.ndarray:
        # A physical (CCF, voxel-edge) placement, in each atlas's own
        # BrainGlobe micrometres (voxel i's centre at i * res: CCF - res / 2).
        physical = np.array([[0.0, 0.0, 6000.0], [20.0, 0.0, 2000.0], [0.0, 20.0, 1500.0]])
        physical[:, 2] -= res / 2.0
        return physical

    reference = job_export([{"filename": "a.png", "width": 300, "height": 200, "nr": 1,
                             "pixel_to_atlas_um": pixel_map(25.0)}], facts(25.0))
    for res in (10.0, 50.0, 100.0):
        exported = job_export([{"filename": "a.png", "width": 300, "height": 200, "nr": 1,
                                "pixel_to_atlas_um": pixel_map(res)}], facts(res))
        assert exported["target"] == "ABA_Mouse_CCFv3_2017_25um.cutlas"
        np.testing.assert_allclose(exported["slices"][0]["anchoring"],
                                   reference["slices"][0]["anchoring"], atol=1e-4)


# --- shear: the perpendicular anchoring and the markers that carry it ---------------

#: A sheared anchoring (QuickNII voxels): u and v about 80 degrees apart.
SHEARED = [100.0, 300.0, 200.0, 250.0, -20.0, 10.0, 50.0, 5.0, -300.0]
SIZE = (400, 300)
ALLEN = {"name": "allen_mouse_25um", "orientation": "asr", "shape": [528, 320, 456],
         "resolution_um": [25.0] * 3}


def place(anchoring, x, y, size=SIZE) -> np.ndarray:
    """QuickNII's anchoring of continuous pixel points, (n, 3) voxels."""
    values = np.asarray(anchoring, dtype=float)
    return (values[:3] + np.outer(np.asarray(x) / size[0], values[3:6])
            + np.outer(np.asarray(y) / size[1], values[6:9]))


def orthonormalize(anchoring) -> np.ndarray:
    """What QuickNII (``SliceData.unpack``) and VisuAlign
    (``Slice.decompose`` then ``recompose``) make of an anchoring on load:
    the centre and the lengths of u and v kept, u's direction kept, v made
    perpendicular to it."""
    values = np.asarray(anchoring, dtype=float)
    u, v = values[3:6], values[6:9]
    centre = values[:3] + (u + v) / 2
    unit_u, unit_v = u / np.linalg.norm(u), v / np.linalg.norm(v)
    across = unit_v - unit_u * (unit_u @ unit_v)
    new_u = unit_u * np.linalg.norm(u)
    new_v = across / np.linalg.norm(across) * np.linalg.norm(v)
    return np.concatenate([centre - (new_u + new_v) / 2, new_u, new_v])


def visualign_map(item: dict, x, y) -> np.ndarray:
    """A reader of visualign.json: the anchoring orthonormalized as VisuAlign
    does, the markers triangulated with VisuAlign's four fixed corners, the
    section point (x, y) taken through both; (n, 3) voxels."""
    size = (item["width"], item["height"])
    points = np.column_stack([x, y])
    markers = np.asarray(item["markers"], dtype=float).reshape(-1, 4)
    if len(markers):
        mesh = visualign_mesh(markers[:, 2:] - 0.5, markers[:, :2] - 0.5, size=size)
        points = mesh.map(points - 0.5) + 0.5
    return place(orthonormalize(item["anchoring"]), points[:, 0], points[:, 1], size)


def sheared_rows(markers=None) -> list[dict]:
    """One section on the 25 um Allen atlas whose exact map is sheared."""
    matrix = np.array([[18.0, 4.0, 6000.0], [3.0, 1.0, 2000.0], [2.0, 21.0, 1500.0]])
    row = {"filename": "a_s001.png", "width": SIZE[0], "height": SIZE[1], "nr": 1,
           "pixel_to_atlas_um": matrix}
    if markers is not None:
        row["markers"] = markers
    return [row]


def test_the_perpendicular_anchoring_is_the_nearest_in_the_same_plane():
    values = np.asarray(perpendicular_anchoring(AnchoringVector(*SHEARED)).to_list())
    old = np.asarray(SHEARED)
    u, v = values[3:6], values[6:9]
    assert abs(u @ v) < 1e-6 * (u @ u)
    # the same plane and the same image centre
    normal = np.cross(old[3:6], old[6:9])
    for axis in (u, v):
        assert abs(axis @ normal) < 1e-6 * np.linalg.norm(axis) * np.linalg.norm(normal)
    np.testing.assert_allclose(values[:3] + (u + v) / 2, old[:3] + (old[3:6] + old[6:9]) / 2,
                               atol=1e-5)

    # nearer the sheared axes than any other perpendicular pair in the plane
    def miss(new_u: np.ndarray, new_v: np.ndarray) -> float:
        return float(np.sum((new_u - old[3:6]) ** 2) + np.sum((new_v - old[6:9]) ** 2))

    e1 = old[3:6] / np.linalg.norm(old[3:6])
    e2 = np.cross(normal, e1) / np.linalg.norm(normal)
    best = miss(u, v)
    for angle in np.linspace(-0.5, 0.5, 2001):
        r1 = np.cos(angle) * e1 + np.sin(angle) * e2
        r2 = np.cos(angle) * e2 - np.sin(angle) * e1
        # (the written numbers are rounded to 1e-6 voxels)
        assert best <= miss((r1 @ old[3:6]) * r1, (r2 @ old[6:9]) * r2) + 1e-4
    # QuickNII and VisuAlign read it as written
    np.testing.assert_allclose(orthonormalize(values), values, atol=1e-4)


def test_an_anchoring_without_shear_is_kept():
    square = [10.0, 20.0, 30.0, 0.0, 0.0, -300.0, 200.0, 0.0, 0.0]
    assert perpendicular_anchoring(AnchoringVector(*square)).to_list() == square


def test_visualign_json_reproduces_a_sheared_linear_placement():
    """A linear section with shear: quicknii.json holds the perpendicular
    anchoring alone; visualign.json adds the few markers that carry the shear,
    so a reader that orthonormalizes the anchoring and applies the markers
    maps every point of the image where LangSlice does."""
    rows = sheared_rows()
    exact = anchoring_from_pixel_map(rows[0]["pixel_to_atlas_um"], *SIZE, ALLEN).to_list()
    quicknii = job_export(rows, ALLEN)["slices"][0]
    visualign = job_export(rows, ALLEN, write_markers=True)["slices"][0]
    assert quicknii["markers"] == [] and quicknii["anchoring"] == visualign["anchoring"]
    assert 0 < len(visualign["markers"]) <= 9
    x, y = np.meshgrid(np.linspace(0, SIZE[0], 23), np.linspace(0, SIZE[1], 17))
    x, y = x.ravel(), y.ravel()
    truth = place(exact, x, y)
    assert np.abs(visualign_map(visualign, x, y) - truth).max() < 1e-3
    # quicknii.json alone misses by the shear it drops
    assert np.abs(visualign_map(quicknii, x, y) - truth).max() > 1.0


def test_visualign_json_reproduces_a_deformation_on_a_sheared_placement():
    """Markers made against the exact (sheared) anchoring are carried onto the
    perpendicular one: the reader lands everywhere where the same markers on
    the exact anchoring land."""
    grid_x, grid_y = marker_grid(SIZE)
    bend_x = grid_x + 6.0 * np.sin(grid_y / 40.0)
    bend_y = grid_y + 4.0 * np.cos(grid_x / 55.0)
    markers = np.column_stack([bend_x, bend_y, grid_x, grid_y]).tolist()
    rows = sheared_rows(markers)
    exact = anchoring_from_pixel_map(rows[0]["pixel_to_atlas_um"], *SIZE, ALLEN).to_list()
    item = job_export(rows, ALLEN, write_markers=True)["slices"][0]
    assert len(item["markers"]) == len(markers)
    old = visualign_mesh(np.column_stack([grid_x, grid_y]) - 0.5,
                         np.column_stack([bend_x, bend_y]) - 0.5, size=SIZE)
    generator = np.random.default_rng(3)
    x, y = generator.uniform(0, SIZE[0], 2000), generator.uniform(0, SIZE[1], 2000)
    moved = old.map(np.column_stack([x, y]) - 0.5) + 0.5
    truth = place(exact, moved[:, 0], moved[:, 1])
    assert np.abs(visualign_map(item, x, y) - truth).max() < 1e-3


def test_a_section_without_shear_or_deformation_gets_no_markers():
    rows = sheared_rows()
    rows[0]["pixel_to_atlas_um"] = np.array([[20.0, 0.0, 6000.0], [0.0, 0.0, 2000.0],
                                             [0.0, 20.0, 1500.0]])
    assert job_export(rows, ALLEN, write_markers=True)["slices"][0]["markers"] == []


def test_the_marker_grid_surrounds_the_image_off_the_pixel_lattice():
    """The outline stands outside the image (every pixel inside the mesh), the
    points are moved off the lattice of pixel centres (a pixel centre on a
    mesh edge falls in no triangle in PyNutil's and Nutil's arithmetic), and
    the grid is the same on every call."""
    x, y = marker_grid((2048, 1389))
    assert 800 <= len(x) <= 1000
    assert x.min() == -MARKER_MARGIN_PX and x.max() == 2048 + MARKER_MARGIN_PX
    assert y.min() == -MARKER_MARGIN_PX and y.max() == 1389 + MARKER_MARGIN_PX
    inside = (x > 0) & (x < 2048) & (y > 0) & (y < 1389)
    on_lattice = np.isclose((x * 2) % 1, 0) & np.isclose((y * 2) % 1, 0)
    assert not np.any(on_lattice & inside)
    again = marker_grid((2048, 1389))
    np.testing.assert_array_equal(x, again[0])
    np.testing.assert_array_equal(y, again[1])
    # a small image: steps no shorter than the minimum, still around the image
    small_x, small_y = marker_grid((100, 60))
    assert len(small_x) <= 16 and small_x.max() == 101.0 and small_y.max() == 61.0
