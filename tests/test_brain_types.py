from langslice_harness.whole_brain.types import (
    BrainEstimationConfig,
    BrainEstimationResult,
    BrainEstimationSummary,
    SlicePosition,
)


def test_slice_position_creation():
    sp = SlicePosition(
        filename="slice_001.tif",
        index=0,
        position_mm=1.30,
        source="anchor",
        locked=True,
    )
    assert sp.filename == "slice_001.tif"
    assert sp.position_mm == 1.30
    assert sp.source == "anchor"
    assert sp.locked is True


def test_config_interval_mm():
    cfg = BrainEstimationConfig(
        image_folder="/tmp/slices",
        atlas_name="allen_mouse_25um",
        thickness_um=50,
        interval_um=200,
        n_anchors=4,
        max_parallel=4,
        z_axis="AP",
    )
    assert cfg.interval_mm == 0.200
    assert cfg.thickness_mm == 0.050


def test_result_to_dict():
    slices = [
        SlicePosition("a.tif", 0, 1.0, "anchor", True),
        SlicePosition("b.tif", 1, 1.2, "interpolated+refined", True),
    ]
    summary = BrainEstimationSummary(
        mean_interval_mm=0.2,
        std_interval_mm=0.01,
        n_slices=2,
        n_anchors=1,
        n_refined=1,
        n_skipped=0,
    )
    result = BrainEstimationResult(
        config=BrainEstimationConfig(
            image_folder="/tmp",
            atlas_name="allen_mouse_25um",
            thickness_um=50,
            interval_um=200,
            n_anchors=1,
            max_parallel=4,
            z_axis="AP",
        ),
        slices=slices,
        summary=summary,
    )
    d = result.to_dict()
    assert d["atlas"] == "allen_mouse_25um"
    assert d["thickness_um"] == 50
    assert len(d["slices"]) == 2
    assert d["slices"][0]["filename"] == "a.tif"
    assert d["slices"][0]["source"] == "anchor"
