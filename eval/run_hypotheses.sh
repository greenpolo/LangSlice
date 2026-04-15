#!/bin/bash
# Run hypothesis evaluations sequentially (one at a time to avoid rate limits)
# Each hypothesis saves results to eval/hypotheses/

set -e
EVAL="python eval/eval_group.py"
IMAGES_M01="--images references/TestImages/M01 --ground-truth references/TestImages/M01/ground_truth.json"
IMAGES_M09="--images references/TestImages/M09 --ground-truth references/TestImages/M09/ground_truth.json"

echo "=== H3: media_resolution sweep ==="
# H3a: low
echo "H3a: media_resolution=low"
$EVAL $IMAGES_M01 --media-resolution low --json > eval/hypotheses/h03a_low_m01.json 2>eval/hypotheses/h03a_low_m01_stderr.log
echo "H3a M01 done"
$EVAL $IMAGES_M09 --media-resolution low --json > eval/hypotheses/h03a_low_m09.json 2>eval/hypotheses/h03a_low_m09_stderr.log
echo "H3a M09 done"

# H3b: medium
echo "H3b: media_resolution=medium"
$EVAL $IMAGES_M01 --media-resolution medium --json > eval/hypotheses/h03b_med_m01.json 2>eval/hypotheses/h03b_med_m01_stderr.log
echo "H3b M01 done"

# H3c: high
echo "H3c: media_resolution=high"
$EVAL $IMAGES_M01 --media-resolution high --json > eval/hypotheses/h03c_high_m01.json 2>eval/hypotheses/h03c_high_m01_stderr.log
echo "H3c M01 done"

echo "=== H4: Thinking level sweep ==="
# H4a: MINIMAL
echo "H4a: thinking=MINIMAL"
$EVAL $IMAGES_M01 --thinking MINIMAL --json > eval/hypotheses/h04a_minimal_m01.json 2>eval/hypotheses/h04a_minimal_m01_stderr.log
echo "H4a M01 done"

# H4b: LOW
echo "H4b: thinking=LOW"
$EVAL $IMAGES_M01 --thinking LOW --json > eval/hypotheses/h04b_low_m01.json 2>eval/hypotheses/h04b_low_m01_stderr.log
echo "H4b M01 done"

# H4c: MEDIUM
echo "H4c: thinking=MEDIUM"
$EVAL $IMAGES_M01 --thinking MEDIUM --json > eval/hypotheses/h04c_med_m01.json 2>eval/hypotheses/h04c_med_m01_stderr.log
echo "H4c M01 done"

echo "=== H5: Atlas resolution sweep ==="
# H5a: 512
echo "H5a: atlas_resolution=512"
$EVAL $IMAGES_M01 --atlas-resolution 512 --json > eval/hypotheses/h05a_512_m01.json 2>eval/hypotheses/h05a_512_m01_stderr.log
echo "H5a M01 done"

# H5b: 2048
echo "H5b: atlas_resolution=2048"
$EVAL $IMAGES_M01 --atlas-resolution 2048 --json > eval/hypotheses/h05b_2048_m01.json 2>eval/hypotheses/h05b_2048_m01_stderr.log
echo "H5b M01 done"

echo "=== H6: Flash model ==="
echo "H6: model=gemini-3-flash-preview"
$EVAL $IMAGES_M01 --model gemini-3-flash-preview --json > eval/hypotheses/h06_flash_m01.json 2>eval/hypotheses/h06_flash_m01_stderr.log
echo "H6 M01 done"
$EVAL $IMAGES_M09 --model gemini-3-flash-preview --json > eval/hypotheses/h06_flash_m09.json 2>eval/hypotheses/h06_flash_m09_stderr.log
echo "H6 M09 done"

echo "=== All parameter sweeps complete ==="
