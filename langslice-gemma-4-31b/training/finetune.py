"""QLoRA fine-tuning of Gemma 4 31B via Unsloth.

Usage:
    python -m langslice_gemma_4_31b.training.finetune \
        --dataset langslice-gemma-4-31b/data/triplets_with_cot.jsonl \
        --output-dir langslice-gemma-4-31b/checkpoints
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 for AP estimation")
    p.add_argument("--dataset", required=True, help="Training JSONL with CoT")
    p.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    p.add_argument("--base-model", default="unsloth/gemma-4-31b-it-bnb-4bit", help="Base model ID")
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (keep low for 32GB VRAM)")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return p.parse_args()


def finetune(args: argparse.Namespace):
    """Run QLoRA fine-tuning via Unsloth.

    Pipeline:
    1. Load base Gemma 4 31B in 4-bit via Unsloth
    2. Attach LoRA adapters to attention layers
    3. Load comparison triplet dataset with CoT targets
    4. Train with gradient checkpointing + accumulation
    5. Save LoRA adapter weights + merged GGUF for inference
    """
    # TODO: Implement fine-tuning
    # Key Unsloth setup:
    #   from unsloth import FastLanguageModel
    #   model, tokenizer = FastLanguageModel.from_pretrained(
    #       model_name=args.base_model,
    #       max_seq_length=4096,
    #       load_in_4bit=True,
    #   )
    #   model = FastLanguageModel.get_peft_model(
    #       model, r=args.lora_rank, target_modules=[...],
    #   )
    raise NotImplementedError("Fine-tuning pending Unsloth integration")


if __name__ == "__main__":
    finetune(_parse_args())
