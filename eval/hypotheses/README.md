# Hypothesis Implementations

Each hypothesis is implemented as a code change to `ap_multi_slice.py` (and sometimes `tool_definitions.py`).

Workflow:
1. Apply the hypothesis change
2. Run eval: `python eval/eval_group.py --images references/TestImages/M01 --ground-truth references/TestImages/M01/ground_truth.json --json > eval/hypotheses/HXX_m01.json`
3. Record results in `eval/group_research_log.json`
4. Revert: `git checkout langslice/estimation/google/ap_multi_slice.py`
