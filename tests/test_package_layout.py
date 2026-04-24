from __future__ import annotations

import importlib


def test_langslice_harness_exports_public_registration_runtime() -> None:
    import langslice_harness

    registration = importlib.import_module("langslice_harness.registration")

    assert langslice_harness.__version__ == "0.1.0"
    assert registration.estimate_registration_runtime.__name__ == "estimate_registration_runtime"
