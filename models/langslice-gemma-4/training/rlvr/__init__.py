"""RLVR (multi-turn GRPO) training for langslice-gemma-4.

Submodules:
    atlas_grid  — pre-rendered atlas slice cache shared across rollouts
    env         — LangSliceEstimateEnv exposed to GRPOTrainer.environment_factory,
                  with prod-shape (positions_mm, reasoning, tool_context) tools
                  returning dict responses, and a defensive done-fence
    rewards     - single gated-linear closeness reward (max(0, 1 - |err|/window))
    dataset     - HF Dataset assembly from references/TestImages with strict
                  subject-level train/eval holdout
    train_grpo  - driver script launched via python -m langslice_rlvr
                  (heavy deps imported lazily; not auto-imported)
"""
