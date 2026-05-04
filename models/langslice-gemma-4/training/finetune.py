"""QLoRA fine-tuning scaffold for LangSlice Gemma 4 E4B via Unsloth.

Usage:
    python models/langslice-gemma-4/training/finetune.py \
        --dataset models/langslice-gemma-4/data/sft_traces.jsonl \
        --output-dir models/langslice-gemma-4/checkpoints
"""

from __future__ import annotations

import argparse


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune Gemma 4 E4B for slice position estimation")
    p.add_argument(
        "--dataset",
        required=True,
        help="Training JSONL rendered from deployment traces",
    )
    p.add_argument("--output-dir", required=True, help="Checkpoint output directory")
    p.add_argument("--base-model", default="google/gemma-4-E4B-it", help="Base model ID")
    p.add_argument("--lora-rank", type=int, default=16, help="LoRA rank")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--batch-size", type=int, default=1, help="Batch size (keep low for 32GB VRAM)")
    p.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    return p.parse_args()


def finetune(args: argparse.Namespace):
    """Run QLoRA fine-tuning via Unsloth.

    Pipeline:
    1. Load base Gemma 4 E4B in 4-bit via Unsloth
    2. Attach LoRA adapters to attention layers
    3. Load deployment-shaped trace data by default
    4. Train with gradient checkpointing + accumulation
    5. Save LoRA adapter weights + merged GGUF for inference

    Gemini rationale summaries may be used for fallback experiments or auxiliary
    caption tasks, but full reasoning traces are not the default target for E4B.
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
