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
    "langslice_training.isft_core.filter",
    "langslice_training.isft_core.path_rewriter",
    "langslice_training.isft_core.rollout",
    "langslice_training.isft_core.seen_ledger",
    "langslice_training.isft_core.state",
    "langslice_training.isft_core.trace_format",
    "langslice_training.single_turn_core.adaptive_reward",
    "langslice_training.single_turn_core.manifest_index",
    "langslice_training.single_turn_core.rewards",
    "langslice_training.single_turn_core.section_state",
    "langslice_training.single_turn_core.terminal_states",
    "langslice_training.rl_core.atlas_grid",
    "langslice_training.rl_core.dataset",
    "langslice_training.rl_core.env",
    "langslice_training.rl_core.rewards",
]


def test_training_core_modules_import() -> None:
    for mod in MODULES:
        importlib.import_module(mod)
