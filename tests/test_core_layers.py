"""A placement picture's layers come from the frame it was drawn in."""

from __future__ import annotations

import numpy as np
from PIL import Image

from langslice.core import canvas, layers
from langslice.core.oblique import plane_index_affine, plane_index_coordinates
from tests.deformable_synthetic import SMOOTH_FIELD, SyntheticAtlas, render_section


def _draw(**kwargs) -> tuple[list[Image.Image], list[canvas.PanelFrame]]:
    atlas = SyntheticAtlas()
    section, _ = render_section(atlas, SMOOTH_FIELD(), seed=0)
    panels: list[canvas.PanelFrame] = []
    images = canvas.physical_views(
        section, 25.0, atlas, 0.15, "coronal", 0.0, 0.0,
        {"rotation_deg": 6.0, "scale_x": 1.0, "scale_y": 1.0, "translate_x_mm": 0.2,
         "translate_y_mm": 0.0}, long_edge=300, panel_frames=panels, **kwargs)
    return images, panels


def test_a_panel_frame_per_picture_on_the_pictures_grid():
    images, panels = _draw(mode="overlay", zoom=[0.1, 0.1, 0.9, 0.8])
    assert len(panels) == len(images) == 1
    for image, panel in zip(images, panels, strict=True):
        # The content starts at the top; the caption band is below it.
        assert panel.size == image.size and panel.content_box[1] == 0
        assert panel.content_box[3] < image.height
        assert layers.labels_layer(panel).shape == image.size[::-1]
        assert layers.borders_layer(panel).dtype == np.uint8


def test_a_section_pixel_lands_where_the_placement_puts_it():
    _images, panels = _draw(mode="overlay")
    panel = panels[0]
    to_picture = layers.section_to_picture(panel)
    canvas_xy = panel.section_matrix @ np.array([10.0, 20.0, 1.0])  # section x=10, y=20
    x0, y0 = panel.content_box[:2]
    expected = [(canvas_xy[1] - panel.crop_box[1]) * panel.factor + y0,
                (canvas_xy[0] - panel.crop_box[0]) * panel.factor + x0]
    np.testing.assert_allclose((to_picture @ [20.0, 10.0, 1.0])[:2], expected)


def test_the_plane_affine_is_the_sampled_plane():
    atlas = SyntheticAtlas()
    for pitch, yaw, position in ((0.0, 0.0, 0.15), (2.0, -1.5, 0.12)):
        grid = plane_index_coordinates(atlas, position, "coronal", pitch, yaw)
        matrix = plane_index_affine(atlas, position, "coronal", pitch, yaw)
        rows, cols = np.mgrid[0:grid.shape[1], 0:grid.shape[2]]
        mapped = np.tensordot(matrix[:, :2], np.stack([rows, cols]), axes=1) \
            + matrix[:, 2][:, None, None]
        np.testing.assert_allclose(mapped, grid, atol=1e-9)


#: Which picture edge a positive angle puts at the larger position, per plane:
#: the convention the angle conventions in `core.oblique` keep.
ANGLE_EDGES = {
    "coronal": {"pitch": "top", "yaw": "right"},
    "sagittal": {"pitch": "left", "yaw": "bottom"},
    "horizontal": {"pitch": "right", "yaw": "top"},
}


def test_the_cutting_angle_directions_are_the_described_ones():
    from langslice.core.oblique import plane_axes

    atlas = SyntheticAtlas()
    for plane, edges in ANGLE_EDGES.items():
        normal = plane_axes(atlas, plane)[0]
        flat = plane_index_coordinates(atlas, 0.1, plane)
        h, w = flat.shape[1:]
        # The sign that turns the normal index into position (mm).
        step = plane_index_coordinates(atlas, 0.15, plane)[normal, h // 2, w // 2]
        sign = 1.0 if step > flat[normal, h // 2, w // 2] else -1.0
        for angle, edge in edges.items():
            tilted = sign * plane_index_coordinates(
                atlas, 0.1, plane, **{f"{angle}_deg": 10.0})[normal]
            at = {"top": tilted[0, w // 2], "bottom": tilted[h - 1, w // 2],
                  "left": tilted[h // 2, 0], "right": tilted[h // 2, w - 1]}
            facing = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
            assert at[edge] > at[facing[edge]], (plane, angle, edge)
