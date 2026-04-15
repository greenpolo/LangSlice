#!/bin/bash
# Parameter sweep hypotheses - run sequentially on M01 with Flash
set -e
EVAL="python eval/eval_group.py"
M01="--images references/TestImages/M01 --ground-truth references/TestImages/M01/ground_truth.json --model gemini-3-flash-preview"

echo "=== H3: media_resolution sweep ==="
echo "H3a: low" && $EVAL $M01 --media-resolution low --json > eval/hypotheses/h03a_low_m01.json 2>eval/hypotheses/h03a_low_m01_stderr.log && echo "done"
echo "H3b: medium" && $EVAL $M01 --media-resolution medium --json > eval/hypotheses/h03b_med_m01.json 2>eval/hypotheses/h03b_med_m01_stderr.log && echo "done"
echo "H3c: high" && $EVAL $M01 --media-resolution high --json > eval/hypotheses/h03c_high_m01.json 2>eval/hypotheses/h03c_high_m01_stderr.log && echo "done"

echo "=== H4: thinking level sweep ==="
echo "H4a: MINIMAL" && $EVAL $M01 --thinking MINIMAL --json > eval/hypotheses/h04a_minimal_m01.json 2>eval/hypotheses/h04a_minimal_m01_stderr.log && echo "done"
echo "H4b: LOW" && $EVAL $M01 --thinking LOW --json > eval/hypotheses/h04b_low_m01.json 2>eval/hypotheses/h04b_low_m01_stderr.log && echo "done"
echo "H4c: MEDIUM" && $EVAL $M01 --thinking MEDIUM --json > eval/hypotheses/h04c_med_m01.json 2>eval/hypotheses/h04c_med_m01_stderr.log && echo "done"

echo "=== H5: atlas resolution sweep ==="
echo "H5a: 512" && $EVAL $M01 --atlas-resolution 512 --json > eval/hypotheses/h05a_512_m01.json 2>eval/hypotheses/h05a_512_m01_stderr.log && echo "done"
echo "H5b: 2048" && $EVAL $M01 --atlas-resolution 2048 --json > eval/hypotheses/h05b_2048_m01.json 2>eval/hypotheses/h05b_2048_m01_stderr.log && echo "done"

echo "=== All parameter sweeps complete ==="
