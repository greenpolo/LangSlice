# Training Overview

LangSlice training code is public, but raw data, generated corpora, caches,
checkpoints, and private run logs are local-only.

## Layout

- `models/langslice-gemma-4/training/sft/` - SFT trainer and data contract.
- `models/training-core/langslice_training/rl/single_turn/` - active single-turn RL trainer.
- `models/training-core/langslice_training/` - shared reusable training code.
- `models/langslice-traces/langslice_traces/` - trace generation and rendering primitives.
- `models/synthdata/synthdata/` - augmentation and synthetic-data utilities.
- `models/data/langslice_data/` - public manifest/QC tooling and fixtures.

## Entrypoints

The public harness CLI remains:

```powershell
langslice version
```

Training entrypoints live under the model hub and are exposed through small
launchers:

```powershell
langslice-gemma-sft --help
langslice-gemma-rl --help
```

These launchers only validate imports and arguments when invoked with `--help`;
they do not start training unless full training arguments are provided.

iSFT is retired as a public product/pipeline and no longer has a public launcher.

## Data Policy

Tracked files may include package code, README files, small fixtures, and model
metadata. Keep the following out of the public repo:

- raw datasets and manifest rows
- generated SFT/RL corpora
- atlas/query image caches
- model checkpoints and adapters
- QC thumbnails, debug traces, and local training logs

The `.gitignore` rules reserve local paths for those materials.
