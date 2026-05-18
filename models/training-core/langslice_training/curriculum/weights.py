"""Backward-compatible shim to shared adaptive curriculum weights."""
from langslice_training.adaptive.curriculum.weights import *  # noqa: F401,F403
from langslice_training.adaptive.curriculum.weights import (  # noqa: F401
    PerBinMAE,
    compute_weights,
    read_per_bin_mae,
    read_weights_json,
    update_weighted_dataset,
    write_weights_json,
)
