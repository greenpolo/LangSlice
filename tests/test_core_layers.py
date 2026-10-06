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
    images, _iou = canvas.physical_views(
        section, 25.0, atlas, 0.15, "coronal", 0.0, 0.0,
        {"rotation_deg": 6.0, "scale_x": 1.0, "scale_y": 1.0, "translate_x_mm": 0.2,
         "translate_y_mm": 0.0}, long_edge=300, panel_frames=panels, **kwargs)
    return images, panels


def test_a_panel_frame_per_picture_on_the_pictures_grid():
    images, panels = _draw(mode="side_by_side", zoom=[0.1, 0.1, 0.9, 0.8])
    assert len(panels) == len(images) == 2
    for image, panel in zip(images, panels, strict=True):
        # The content starts at the top; the caption band is below it.
        assert panel.size == image.size and panel.content_box[1] == 0
        assert panel.content_box[3] < image.height
        assert layers.labels_layer(panel).shape == image.size[::-1]
        assert layers.borders_layer(panel).dtype == np.uint8


def test_the_border_layer_is_the_drawn_lines():
    images, panels = _draw(mode="outlines", outlines="all", border_color="#ff0000",
                           border_thickness=1.0)
    picture = np.asarray(images[0]).astype(int)
    borders = layers.borders_layer(panels[0])
    red = (picture[..., 0] - np.maximum(picture[..., 1], picture[..., 2])) > 60
    assert red.sum() > 200
    assert (red & (borders > 0)).sum() / red.sum() > 0.99
    assert (red & (borders >= 128)).sum() / (borders >= 128).sum() > 0.9


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


def test_notes_are_collected_per_call_and_found_by_identity():
    image = Image.new("RGB", (4, 4))
    twin = image.copy()
    layers.note(image, sections=("ignored",))  # nothing collecting: free
    with layers.collecting() as notes:
        layers.note(image, sections=("s0.png",), mode="section")
    assert layers.note_for(image, notes).sections == ("s0.png",)
    assert layers.note_for(twin, notes) is None
