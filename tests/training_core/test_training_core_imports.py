from __future__ import annotations

import importlib


MODULES = [
    "langslice_training.adaptive.schedule",
    "langslice_training.adaptive.curriculum.bins",
    "langslice_training.adaptive.curriculum.log",
    "langslice_training.adaptive.curriculum.sampler",
    "langslice_training.adaptive.curriculum.weights",
    "langslice_training.embeddings.cache",
    "langslice_training.embeddings.query_cache",
    "langslice_training.embeddings.splice",
    "langslice_training.rl.common.atlas_grid",
    "langslice_training.rl.common.dataset",
    "langslice_training.rl.common.rewards",
    "langslice_training.rl.single_turn.adaptive_reward",
    "langslice_training.rl.single_turn.manifest_index",
    "langslice_training.rl.single_turn.rewards",
    "langslice_training.rl.single_turn.section_state",
    "langslice_training.rl.single_turn.terminal_states",
    "langslice_training.rl.multi_turn_env.env",
    "langslice_training.contracts",
    "langslice_training.model_io",
]


def test_training_core_modules_import() -> None:
    for mod in MODULES:
        importlib.import_module(mod)
