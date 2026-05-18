from __future__ import annotations

from synthdata import dataset


def test_synthdata_dataset_module_exports_core_api() -> None:
    assert dataset.__version__
    assert callable(dataset.sample_spec)
    assert callable(dataset.render)
    assert callable(dataset.write_dataset)
