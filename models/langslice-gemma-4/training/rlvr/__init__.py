"""RLVR (multi-turn GRPO) training for langslice-gemma-4.

Submodules:
    atlas_grid  — pre-rendered atlas slice cache shared across rollouts
    env         — LangSliceEstimateEnv exposed to GRPOTrainer.environment_factory
    rewards     — position_accuracy / format_compliance / length_budget rewards
    dataset     — HF Dataset assembly from TestImages + SFT-pipeline holdout
    train_grpo  — driver script (requires unsloth + trl runtime; not auto-imported)
"""
