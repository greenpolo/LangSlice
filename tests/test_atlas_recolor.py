"""The recolor layer: organized colors for degenerate or chaotic palettes."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from langslice.atlas.recolor import MERGE_EPS, MIN_SEPARATION, _allen_join, color_lut


def _atlas(rows: dict[int, dict], name: str | None = None) -> SimpleNamespace:
    ns = SimpleNamespace(structures=rows)
    if name is not None:
        ns.atlas_name = name
    return ns


def _row(sid, acronym, path, rgb, name=None):
    return {
        "id": sid,
        "acronym": acronym,
        "name": name if name is not None else acronym,
        "structure_id_path": list(path),
        "rgb_triplet": list(rgb),
    }


def test_healthy_native_colors_pass_through_untouched():
    rows = {
        1: _row(1, "root", [1], [255, 255, 255]),
        2: _row(2, "A", [1, 2], [255, 0, 0]),
        3: _row(3, "B", [1, 3], [0, 255, 0]),
    }
    assert color_lut(_atlas(rows))[2] == (255, 0, 0)


def test_all_white_allen_tree_recovers_real_allen_colors():
    rows = {
        997: _row(997, "root", [997], [255, 255, 255]),
        8: _row(8, "grey", [997, 8], [255, 255, 255]),
        315: _row(315, "Isocortex", [997, 8, 315], [255, 255, 255]),
        1009: _row(1009, "made-up-acronym", [997, 8, 315, 1009], [255, 255, 255]),
    }
    lut = color_lut(_atlas(rows))
    assert lut[315] == (112, 255, 113)  # Allen's true Isocortex color
    assert lut[1009] == lut[315]  # unmatched acronym inherits its ancestor


def test_foreign_terminology_still_lands_on_the_allen_counterpart():
    """A tree naming things its own way (the Waxholm rat's) joins anyway."""
    white = [255, 255, 255]
    rows = {
        1: _row(1, "root", [1], white, name="Root"),
        2: _row(2, "Mes", [1, 2], white, name="Mesencephalon"),
        3: _row(3, "OB", [1, 3], white, name="Olfactory bulb"),
        4: _row(4, "Cb-u", [1, 4], white, name="Cerebellum, unspecified"),
        5: _row(5, "V2", [1, 5], white, name="Secondary visual area"),
        6: _row(6, "wmt", [1, 6], white, name="Olfactory white matter"),
        7: _row(7, "BS-u", [1, 2, 7], white, name="Brainstem, unspecified"),
    }
    # The join itself, before color_lut pulls the palette apart (see
    # test_a_derived_palette_is_pulled_apart_past_the_merge_radius).
    lut = _allen_join(_atlas(rows))
    assert lut is not None
    assert lut[2] == (255, 100, 255)  # bridge: mesencephalon -> Midbrain
    assert lut[3] == (154, 210, 189)  # bridge: -> Main olfactory bulb
    assert lut[4] == (240, 240, 128)  # ", unspecified" dropped -> Cerebellum
    assert lut[5] == (8, 133, 140)  # variant: -> Visual areas
    assert lut[6] == (204, 204, 204)  # white matter -> fiber tracts
    assert lut[7] == lut[2]  # no counterpart: inherits its division


def test_acronym_alone_cannot_recolor_a_different_structure():
    """Allen's ``V`` is the trigeminal motor nucleus, not a ventricle."""
    white = [255, 255, 255]
    rows = {
        1: _row(1, "root", [1], white, name="Root"),
        2: _row(2, "Mes", [1, 2], white, name="Mesencephalon"),
        3: _row(3, "V", [1, 3], white, name="Ventricular system"),
        4: _row(4, "SO", [1, 2, 4], white, name="Superior olivary complex"),
    }
    lut = _allen_join(_atlas(rows))
    assert lut is not None
    assert lut[3] == (170, 170, 170)  # Allen's ventricular systems, by name
    assert lut[4] == (255, 174, 111)  # not Allen SO, the supraoptic nucleus


def test_disorganized_deep_tree_gets_hue_families():
    rng = np.random.default_rng(1)
    rows = {1: _row(1, "root", [1], [0, 0, 0])}
    for division in (10, 20):
        rows[division] = _row(division, f"d{division}", [1, division], rng.integers(0, 256, 3))
        for child in range(30):
            sid = division * 100 + child
            rows[sid] = _row(
                sid, f"s{sid}", [1, division, sid], rng.integers(0, 256, 3)
            )
    lut = color_lut(_atlas(rows))
    d10 = np.array([lut[1000 + c] for c in range(30)], float)
    d20 = np.array([lut[2000 + c] for c in range(30)], float)
    within = np.linalg.norm(d10 - d10.mean(axis=0), axis=1).mean()
    across = np.linalg.norm(d10.mean(axis=0) - d20.mean(axis=0))
    assert within < across  # one division = one color family
    assert color_lut(_atlas(rows), mode="never")[1000] == tuple(
        int(c) for c in rows[1000]["rgb_triplet"]
    )


def test_flat_tree_keeps_native_colors_however_chaotic():
    rng = np.random.default_rng(2)
    rows = {1: _row(1, "root", [1], [0, 0, 0])}
    for sid in range(2, 80):
        rows[sid] = _row(sid, f"s{sid}", [1, sid], rng.integers(0, 256, 3))
    lut = color_lut(_atlas(rows))
    assert lut[2] == tuple(int(c) for c in rows[2]["rgb_triplet"])


def _pairwise_min(lut: dict[int, tuple[int, int, int]]) -> float:
    colors = np.array(sorted(set(lut.values())), dtype=float)
    diff = colors[:, None, :] - colors[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(distances.min())


def _allen_style_tree() -> dict[int, dict]:
    """A white Allen-tree atlas: the join path, which is a DERIVED palette."""
    white = [255, 255, 255]
    rows = {
        997: _row(997, "root", [997], white, name="root"),
        8: _row(8, "grey", [997, 8], white, name="Basic cell groups and regions"),
        315: _row(315, "Isocortex", [997, 8, 315], white, name="Isocortex"),
        1089: _row(1089, "HPF", [997, 8, 1089], white, name="Hippocampal formation"),
        382: _row(382, "CA1", [997, 8, 1089, 382], white, name="Field CA1"),
        423: _row(423, "CA2", [997, 8, 1089, 423], white, name="Field CA2"),
        463: _row(463, "CA3", [997, 8, 1089, 463], white, name="Field CA3"),
        726: _row(726, "DG", [997, 8, 1089, 726], white, name="Dentate gyrus"),
        512: _row(512, "CB", [997, 8, 512], white, name="Cerebellum"),
        10707: _row(
            10707, "LINGmo", [997, 8, 512, 10707], white, name="Lingula (I), molecular layer"
        ),
        10708: _row(
            10708, "LINGgr", [997, 8, 512, 10708], white, name="Lingula (I), granular layer"
        ),
        68: _row(68, "MO1", [997, 8, 315, 68], white, name="Somatomotor areas, layer 1"),
        69: _row(69, "MO5", [997, 8, 315, 69], white, name="Somatomotor areas, layer 5"),
    }
    return rows


def test_a_derived_palette_is_pulled_apart_past_the_merge_radius():
    """Allen's leaf greens are closer than the pipeline's own merge radius."""
    lut = color_lut(_atlas(_allen_style_tree()))
    assert _pairwise_min(lut) > MERGE_EPS
    assert _pairwise_min(lut) >= MIN_SEPARATION - 1.0
