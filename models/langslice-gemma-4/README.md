# Gemma 4 31B Model Project

Fine-tuned Gemma 4 for brain atlas AP position estimation. Domain-adapted via comparison-based SFT on BrainGlobe atlases with Gemini-distilled chain-of-thought reasoning.

## Pipeline

```
1. Data Generation
   ├── Extract coronal slices from all BrainGlobe atlases
   ├── Build comparison triplets (query + 2 references)
   └── Distill CoT reasoning from Gemini for each triplet

2. Fine-Tuning (QLoRA via Unsloth)
   ├── Gemma 4 31B on RTX 5090 (32GB)
   ├── Comparison-framed SFT with anatomical reasoning
   └── Checkpoint evaluation against held-out slice benchmarks

3. Evaluation
   ├── Held-out slice sets with known AP ground truth
   ├── Compare: Gemma base → fine-tuned → Gemini API
   └── Publish weights + benchmark results
```

## Structure

```
models/langslice-gemma-4/
├── data/
│   ├── generate_atlas_slices.py    # Extract slices from BrainGlobe atlases
│   ├── build_triplets.py           # Create comparison training examples
│   ├── distill_cot.py              # Generate CoT reasoning via Gemini
│   └── dataset.py                  # Dataset loader for training
├── training/
│   ├── finetune.py                 # QLoRA fine-tuning script (Unsloth)
│   └── configs/                    # Training hyperparameter configs
├── inference/
│   └── predict.py                  # Run inference with fine-tuned model
└── README.md
```

## Hardware Requirements

- Fine-tuning: RTX 5090 (32GB VRAM) — Gemma 4 31B QLoRA ~22GB
- Inference: RTX 3090+ (24GB) — Gemma 4 31B Q4_K_M ~11GB + overhead
