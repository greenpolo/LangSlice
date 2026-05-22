from __future__ import annotations

import importlib
import pytest

from langslice_training.rl.common.atlas_grid import GRID_STEP_MM
from langslice_training.rl.common.rewards import normalized_bell_reward


MODULES = [
    "langslice_training.rl",
    "langslice_training.rl.common.atlas_grid",
    "langslice_training.rl.common.dataset",
    "langslice_training.rl.common.rewards",
    "langslice_training.rl.multi_turn_env.env",
    "langslice_training.rl.single_turn.adaptive_reward",
    "langslice_training.rl.single_turn.manifest_index",
    "langslice_training.rl.single_turn.rewards",
    "langslice_training.rl.single_turn.section_state",
    "langslice_training.rl.single_turn.terminal_states",
]


def test_rl_modules_import() -> None:
    for mod in MODULES:
        importlib.import_module(mod)


def test_common_helpers_basic_behavior() -> None:
    assert GRID_STEP_MM == 0.05
    assert normalized_bell_reward(0.0, axis_span_mm=10.0) == 1.0


def test_legacy_rl_core_modules_are_not_importable() -> None:
    legacy_rl = "langslice_training." + "rl_core"
    legacy_st = "langslice_training." + "single_turn_core"
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_rl)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(legacy_st)
