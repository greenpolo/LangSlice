# LangSlice Gemma 4 SFT Pipeline

This directory contains the consumer side of supervised fine-tuning. It does
not read raw Gemini run folders directly.

## What The Trainer Wants

The trainer expects one completed **langslice-native JSONL corpus** passed with
`--dataset`:

```powershell
cd models/langslice-gemma-4/training
python -m sft.train_sft `
  --config configs/sft_default.toml `
  --dataset ../../../models/langslice-gemma-4/data/sft_examples.jsonl `
  --output-dir ../../../out/cache_fast/sft/run0
```

So the data-side handoff should be:

1. Write a corpus builder, for example `build_sft_corpus.py`, that walks
   `runs_verified/` plus `runs_reroll_*/`.
2. Choose the best accepted run per slice id.
3. Filter to strict-accepted single-slice traces only.
4. Emit `models/langslice-gemma-4/data/sft_examples.jsonl`.
5. Put or copy every referenced query/atlas image under the JSONL parent
   directory, and write paths relative to that parent.

The trainer does not currently walk `raw_trace.json` itself. If the collected
traces live as per-run files, they need to be converted into this JSONL first.

## Row Shape

Each JSONL line is one training example:

```json
{
  "bucket": 1,
  "atlas_name": "allen_mouse_25um",
  "atlas_version": "CCFv3",
  "plane": "coronal",
  "subject_id": "M03",
  "system_prompt_kind": "single_slice",
  "query_image_paths": ["queries/M03_001_001.png"],
  "user_prompt_text": "Estimate the AP position of this slice.",
  "trace": [
    {
      "tool_call": {
        "name": "fetch_atlas",
        "args": {"positions_mm": [3.0, 5.0, 7.0]}
      },
      "tool_result": {
        "image_paths": [
          "atlas/allen_mouse_25um/coronal/3.00mm.png",
          "atlas/allen_mouse_25um/coronal/5.00mm.png",
          "atlas/allen_mouse_25um/coronal/7.00mm.png"
        ],
        "text": "Atlas at 3.00 mm | 5.00 mm | 7.00 mm"
      }
    },
    {
      "submit": {
        "name": "submit_estimate",
        "args": {
          "position_mm": 5.2,
          "reasoning": "Best match after broad and narrow atlas comparison."
        }
      }
    }
  ],
  "gemini_reasoning": "optional; ignored by v1 training"
}
```

Required details:

- `bucket` is always `1`.
- `system_prompt_kind` is always `single_slice`.
- `query_image_paths` must contain exactly one image path.
- All image paths are relative to the JSONL file's parent directory.
- Every non-final trace step is `tool_call` plus `tool_result`.
- The final trace step is always `submit.name == "submit_estimate"`.
- Final submit args require both numeric `position_mm` and non-empty
  `reasoning`.
- Do not include group-only fields such as `interval_mm` or `thickness_um`.
- Do not include HuggingFace chat-template messages, system prompts, tool
  schemas, or tool-call ids. `render.py` creates those at training time.

## Quick Validation

Before handing the corpus to training, run:

```powershell
cd models/langslice-gemma-4/training
python -m sft.train_sft `
  --config configs/sft_default.toml `
  --dataset ../../../models/langslice-gemma-4/data/sft_examples.jsonl `
  --output-dir ../../../out/sft/dryrun `
  --dry-run
```

This validates JSONL structure, required submit reasoning, subject ids, and
image path resolution without loading Gemma or starting training.

For a narrower loader-only check:

```powershell
python -c "from sft.dataset import load_examples; xs=load_examples(r'..\..\..\models\langslice-gemma-4\data\sft_examples.jsonl'); print(len(xs), 'examples')"
```

## Pipeline Ownership

Data builder owns:

- choosing strict-accepted traces;
- best-per-id/reroll selection;
- converting per-run artifacts into the JSONL shape above;
- staging referenced query and atlas images next to the corpus.

SFT trainer owns:

- validating the JSONL;
- constructing production system prompts and tool schemas;
- translating traces into Gemma chat-template messages;
- tokenization, image hydration, label masking, and LoRA training.
