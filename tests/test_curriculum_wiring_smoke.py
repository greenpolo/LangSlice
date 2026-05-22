"""Import-path smoke test for the canonical curriculum public API.

The trainer subclasses (``CurriculumGRPOTrainer`` / ``CurriculumSFTTrainer``)
are constructed lazily via ``__getattr__`` to avoid pulling TRL into every
consumer. We don't actually exercise them here (TRL isn't installed in the
test container) — but we do verify the import paths everyone else relies
on resolve cleanly:

    from langslice_training.adaptive.curriculum import compute_weights, read_per_bin_mae,
                                                      update_weighted_dataset, WeightedRowDataset,
                                                      CurriculumLogger, compute_section_bins

This guards against ``__init__.py`` regressions that silently break the
expert-iteration multi-round wiring.
"""

from __future__ import annotations


def test_curriculum_public_api_imports() -> None:
    """Every name listed in ``curriculum.__all__`` resolves."""
    import langslice_training.adaptive.curriculum as curriculum  # noqa: PLC0415

    expected = {
        "compute_section_bins",
        "WeightedRowDataset",
        "CurriculumGRPOTrainer",
        "CurriculumSFTTrainer",
        "CurriculumLogger",
        "compute_weights",
        "read_per_bin_mae",
        "update_weighted_dataset",
        "read_weights_json",
        "write_weights_json",
    }
    assert expected.issubset(set(curriculum.__all__))


def test_phase_c_helpers_importable() -> None:
    """Phase C public helpers import without TRL."""
    from langslice_training.adaptive.curriculum import (  # noqa: PLC0415
        compute_weights,
        read_per_bin_mae,
        read_weights_json,
        update_weighted_dataset,
        write_weights_json,
    )

    assert callable(compute_weights)
    assert callable(read_per_bin_mae)
    assert callable(update_weighted_dataset)
    assert callable(read_weights_json)
    assert callable(write_weights_json)


def test_phase_b_dataset_importable() -> None:
    """``WeightedRowDataset`` imports without TRL (it only uses RowDataset)."""
    from langslice_training.adaptive.curriculum import WeightedRowDataset  # noqa: PLC0415

    rows = [
        {
            "kind": "single",
            "_image_paths": ("nonexistent.png",),
            "_apply_clahes": (False,),
            "_atlas_long_edge": 256,
            "row_id": i,
        }
        for i in range(3)
    ]
    ds = WeightedRowDataset(rows)
    assert ds._weights == [1.0, 1.0, 1.0]
    assert len(ds) == 3


def test_curriculum_cli_module_imports() -> None:
    """The standalone weight-update CLI imports cleanly."""
    from langslice_training.adaptive.curriculum import cli  # noqa: PLC0415

    assert hasattr(cli, "main")
    assert callable(cli.main)


def test_curriculum_logger_importable() -> None:
    """``CurriculumLogger`` imports via the ``curriculum`` package root."""
    from langslice_training.adaptive.curriculum import CurriculumLogger  # noqa: PLC0415

    assert callable(CurriculumLogger)


def test_unknown_attribute_raises() -> None:
    """``__getattr__`` rejects names not in __all__."""
    import langslice_training.adaptive.curriculum as curriculum  # noqa: PLC0415

    try:
        curriculum.does_not_exist  # noqa: B018 — exercising __getattr__
    except AttributeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected AttributeError")
