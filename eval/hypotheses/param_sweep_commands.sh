#!/bin/bash
# Parameter sweep commands for hypotheses H3-H6
# These don't require code changes - just different eval args

# H3: media_resolution sweep (requires code change to pass through)
# Current default: ultra_high
# Test: low, medium, high

# H4: Thinking level sweep (requires vlm_config override)
# Current default: HIGH
# Test: MINIMAL, LOW, MEDIUM

# H5: Atlas resolution sweep (requires --atlas-resolution passthrough)
# Current default: 1024
# Test: 512, 768, 1536, 2048

# H6: Model comparison
# Current default: gemini-3.1-pro-preview
# Test: gemini-3-flash-preview
python eval/eval_group.py \
    --images references/TestImages/M01 \
    --ground-truth references/TestImages/M01/ground_truth.json \
    --model gemini-3-flash-preview \
    --json > eval/hypotheses/h06_flash_m01.json 2>eval/hypotheses/h06_flash_m01_stderr.log
