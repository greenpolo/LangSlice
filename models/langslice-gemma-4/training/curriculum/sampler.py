"""Backward-compatible shim — curriculum.sampler is now adaptive.curriculum.sampler."""
from adaptive.curriculum.sampler import *  # noqa: F401,F403
from adaptive.curriculum.sampler import WeightedRowDataset  # noqa: F401
