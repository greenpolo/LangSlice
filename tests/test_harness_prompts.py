from langslice_harness.harness.estimation.prompts import (
    build_group_prompt,
    build_single_slice_prompt,
)


def test_single_slice_prompt_coronal_mentions_ap_and_olfactory():
    p = build_single_slice_prompt(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, species="mouse",
    )
    assert "AP" in p
    assert "olfactory" in p.lower()
    assert "0.00" in p or "0.0" in p
    assert "13.2" in p


def test_single_slice_prompt_sagittal_mentions_ml_not_ap():
    p = build_single_slice_prompt(
        atlas_name="allen_mouse_25um", plane="sagittal",
        pos_lo=0.0, pos_hi=11.0, species="mouse",
    )
    assert "ML" in p
    assert "AP" not in p
    assert "olfactory" not in p.lower()


def test_single_slice_prompt_does_not_suggest_removed_tools():
    p = build_single_slice_prompt(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, species="mouse",
    )
    assert "`zoom`" not in p
    assert "`side_by_side`" not in p
    assert "`fetch_atlas`" in p


def test_group_prompt_mentions_interval_and_n_slices():
    p = build_group_prompt(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, species="mouse",
        n_slices=4, interval_mm=0.200, thickness_um=50,
    )
    assert "4" in p
    assert "0.200" in p or "200" in p  # micron or mm form
    assert "AP" in p


def test_group_prompt_does_not_suggest_removed_tools():
    p = build_group_prompt(
        atlas_name="allen_mouse_25um", plane="coronal",
        pos_lo=0.0, pos_hi=13.2, species="mouse",
        n_slices=4, interval_mm=0.200, thickness_um=50,
    )
    assert "`zoom`" not in p
    assert "`side_by_side`" not in p
    assert "`fetch_atlas`" in p
