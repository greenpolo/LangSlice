"""Merge a LoRA adapter into the bf16 base model and save a single bf16 checkpoint
suitable for vLLM serving. Bypasses Unsloth's runtime LoRA pathway.

Usage:
    python -m sft.merge_adapter \
        --adapter-dir /workspace/LangSlice/out/sft/docker-sft-1011-trimmed-noeval \
        --output-dir /workspace/LangSlice/out/sft/docker-sft-1011-trimmed-noeval-merged-bf16
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    p = argparse.ArgumentParser(description="Merge SFT LoRA into bf16 base for serving.")
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--base-model", type=str, default="unsloth/gemma-4-E4B-it")
    args = p.parse_args(argv)

    if not (args.adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"adapter_config.json missing in {args.adapter_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading bf16 base %s ...", args.base_model)
    from unsloth import FastVisionModel  # noqa: I001 - Unsloth must patch first
    import peft  # noqa: I001
    import torch  # noqa: I001

    model, processor = FastVisionModel.from_pretrained(
        args.base_model,
        load_in_4bit=False,
        dtype=torch.bfloat16,
        max_seq_length=8192,
    )
    logger.info("Loading adapter from %s ...", args.adapter_dir)
    model = peft.PeftModel.from_pretrained(model, str(args.adapter_dir))
    logger.info("Merging adapter into base weights ...")
    model = model.merge_and_unload()
    logger.info("Saving merged bf16 model to %s ...", args.output_dir)
    model.save_pretrained(str(args.output_dir), safe_serialization=True)
    processor.save_pretrained(str(args.output_dir))
    logger.info("Done. Use this directory as the vLLM --model arg.")


if __name__ == "__main__":
    main()
