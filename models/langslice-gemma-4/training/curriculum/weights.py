"""Backward-compatible shim — curriculum.weights is now adaptive.curriculum.weights."""
from adaptive.curriculum.weights import *  # noqa: F401,F403
from adaptive.curriculum.weights import (  # noqa: F401
    PerBinMAE,
    compute_weights,
    read_per_bin_mae,
    read_weights_json,
    update_weighted_dataset,
    write_weights_json,
)
