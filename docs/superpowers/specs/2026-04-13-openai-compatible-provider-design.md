# OpenAI-Compatible Provider for LangSlice

**Date:** 2026-04-13
**Status:** Draft

## Purpose

Port the Google/Gemini estimation and registration workflows to use the OpenAI Python SDK (`openai`) as a universal interface for any OpenAI-compatible model server. The OpenAI SDK is not being used to access OpenAI models specifically — it is the lingua franca for open-source model serving. Primary target model: Gemma 4 via Ollama. Image generation via a separate Images API endpoint (local Flux2 Klein or any compatible server).

## Scope

### In scope

- **Estimation tool-use:** `ap_single_slice`, `ap_multi_slice`, `tool_definitions`, `common`
- **Estimation image-gen:** `ap_image_gen`
- **Registration image-gen:** `warping_image_gen`, `landmarks_image_gen`
- **Client configuration module** (`openai_config.py`)
- **CLI integration** (`--provider openai` flag)
- **Workflow routing** (update `registration/common.py` and `estimation/__init__.py`)

### Out of scope

- Registration tool-use (`landmarks_tool_use.py` — doesn't work on Gemini either, stays as stub)
- Batch eval (`batch_eval.py`)
- Whole-brain pipeline integration (future work after per-workflow validation)

## Target Serving Stack

| Server | Responses API | Images API | Notes |
|--------|:---:|:---:|-------|
| **Ollama** | Yes | No | Primary target. Gemma 4 supported. |
| **vLLM** | Yes | No | Responses API with persistence backends (in-memory, file, Redis). |
| **SGLang** | Yes | No | Gemma 4 PR merged (sgl-project/sglang#21952). |
| **OpenAI** | Yes | Yes | gpt-5.4 family for text, gpt-image-1.5 for images. |
| **Local Flux2 Klein** | N/A | Yes | Image generation only, via OpenAI-compatible Images API. |

## Client Configuration

New file: `langslice/openai_config.py`

Two independent clients, since the text/tool model and image model will typically run on different servers (e.g., Ollama for Gemma 4, a local Flux server for image gen).

### Environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `OPENAI_BASE_URL` | `http://localhost:11434/v1` | Text/tool model endpoint (Ollama default) |
| `OPENAI_API_KEY` | `ollama` | API key (dummy for local servers) |
| `OPENAI_MODEL` | `gemma4:31b` | Model name for responses |
| `OPENAI_IMAGE_BASE_URL` | *(none — falls back to `OPENAI_BASE_URL`)* | Image model endpoint |
| `OPENAI_IMAGE_API_KEY` | *(none — falls back to `OPENAI_API_KEY`)* | Image model API key |
| `OPENAI_IMAGE_MODEL` | `flux2-klein` | Model name for image generation |

### Functions

- `get_openai_client() -> openai.OpenAI` — singleton text/tool client
- `get_openai_image_client() -> openai.OpenAI` — singleton image client (separate base_url)
- `get_openai_model() -> str` — configured model name
- `get_openai_image_model() -> str` — configured image model name

Mirrors the structure of `vlm_config.py` (`get_client()`, `MODEL_NAME`). No shared base class or provider abstraction — keep it simple, two independent config modules.

## API Mapping

### Responses API (tool-use estimation)

| Gemini (`google.genai`) | OpenAI (`openai`) |
|--------------------------|-------------------|
| `client.models.generate_content(model, contents, config)` | `client.responses.create(model, input, tools, ...)` |
| `types.Content(role="user", parts=[...])` | `{"role": "user", "content": [...]}` |
| `types.Part.from_text("...")` | `{"type": "input_text", "text": "..."}` |
| `types.Part.from_bytes(data, "image/jpeg")` | `{"type": "input_image", "image_url": "data:image/jpeg;base64,..."}` |
| `types.FunctionDeclaration(name, params)` | `{"type": "function", "function": {"name": ..., "parameters": ...}}` |
| `types.Tool(function_declarations=[...])` | `tools=[{"type": "function", ...}, ...]` |
| `candidate.content.parts[i].function_call` | `response.output[i]` where `type == "function_call"` |
| `types.Part.from_function_response(name, response)` | `{"type": "function_call_output", "call_id": ..., "output": "..."}` |
| `config.system_instruction` | `instructions` parameter |
| Manual `contents: list[Content]` | Server-managed via `previous_response_id` |

### Images API (image-gen workflows)

| Gemini | OpenAI |
|--------|--------|
| `generate_content(response_modalities=["IMAGE","TEXT"])` | `client.images.generate(prompt=..., model=...)` |
| Response `part.inline_data.data` (raw bytes) | `response.data[0].b64_json` (base64-encoded) |
| Image generation + text reasoning in one call | Two separate calls: Chat Completions for reasoning, Images API for generation |
| Input image in same call | `client.images.edit(image=..., prompt=...)` for image-to-image |

## Conversation Management

Server-managed conversation state via `previous_response_id`. The agentic tool-use loop:

```python
response = client.responses.create(
    model=model_name,
    instructions=system_prompt,
    input=[
        {"role": "user", "content": [
            {"type": "input_text", "text": task_description},
            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{b64_tissue}"},
        ]},
    ],
    tools=tool_declarations,
)

while True:
    # Check for function calls in output
    function_calls = [o for o in response.output if o.type == "function_call"]
    if not function_calls:
        # Model returned text — done
        final_text = response.output_text
        break

    # Execute tool calls and feed results back
    tool_outputs = []
    for fc in function_calls:
        result = handle_tool_call(fc.name, fc.arguments)
        tool_outputs.append({
            "type": "function_call_output",
            "call_id": fc.call_id,
            "output": json.dumps(result),
        })

    response = client.responses.create(
        model=model_name,
        previous_response_id=response.id,
        input=tool_outputs,
        tools=tool_declarations,
    )
```

Key advantage over Gemini's `generate_content`: no manual conversation history management. The server tracks state via `previous_response_id`, so each turn only sends the new input (tool outputs, follow-up images, etc.).

## Image-Gen Workflow Restructure

Gemini's image-gen model does reasoning + image generation in one call. With OpenAI-compatible endpoints, these are separate. Each image-gen workflow becomes a two-step process.

### Estimation (`ap_image_gen.py`)

The Gemini version sends tissue + atlas grid images and asks the model to identify the best-matching AP position. The "image-gen" in the name refers to using Gemini's image-gen model for its multi-pass zoom approach, not because the model generates images for this workflow. The model's output is text (a position choice).

**Port approach:** Responses API with vision input. Send tissue + atlas grid images as `input_image` content parts, receive text response with position choice. Structurally identical to the tool-use workflows but without function calling — just vision + text.

### Registration (`warping_image_gen.py`)

The Gemini version sends the tissue photo and atlas reference, then the model generates a colored segmentation image that matches the tissue's anatomy. Elastix then registers the generated segmentation against the actual atlas segmentation.

**Port approach — two-step:**

1. **Reason** (Responses API): Send tissue + atlas images to Gemma 4. Ask the model to describe the visible brain regions, their boundaries, and colors in detail. The model outputs a structured description.
2. **Generate** (Images API): Pass the atlas reference image + the model's region description to `images.edit()`. The image model (Flux2 Klein) generates the colored segmentation based on the reference and description.

The Elastix registration pipeline downstream is unchanged — it receives the generated segmentation image regardless of how it was produced.

### Registration (`landmarks_image_gen.py`)

Legacy two-shot workflow. Same two-step restructure as above, but for landmark overlay images instead of colored segmentations.

## File Structure

```
langslice/
  openai_config.py                          # NEW — client config (env vars, singletons)
  estimation/
    _types.py                               # UNCHANGED — APResult, MultiSliceResult
    __init__.py                             # MODIFY — add OpenAI re-exports
    openai/
      __init__.py                           # REPLACE stub
      common.py                             # NEW — shared helpers
      tool_definitions.py                   # REPLACE stub — tool declarations + handlers
      ap_single_slice.py                    # REPLACE stub — single-slice tool-use
      ap_multi_slice.py                     # REPLACE stub — multi-slice group tool-use
      ap_image_gen.py                       # REPLACE stub — image-gen multi-pass zoom
  registration/
    common.py                               # MODIFY — route to openai/ when provider=openai
    openai/
      __init__.py                           # REPLACE stub
      warping_image_gen.py                  # REPLACE stub — colored segmentation
      landmarks_image_gen.py                # REPLACE stub — two-shot legacy
      landmarks_tool_use.py                 # UNCHANGED — stays as stub (out of scope)
  cli.py                                    # MODIFY — add --provider flag
```

### Shared helpers (`estimation/openai/common.py`)

Port of `estimation/google/common.py`, adapted for OpenAI message format:

- `_image_to_base64(image) -> str` — convert PIL/path/bytes to base64 data URI
- `_build_image_content(b64: str) -> dict` — wrap as `{"type": "input_image", ...}` content part
- `_load_atlas_lazy()`, `_fetch_atlas_slice_bytes()`, `_get_position_range_lazy()` — reuse directly (atlas logic is provider-agnostic)
- `_extract_text(response) -> str` — extract text from Responses API response (`response.output_text`)
- `_extract_usage(response) -> dict` — extract token usage metadata
- `_emit_trace(...)` — trace event emission (reuse `agent_trace` module)

### Tool definitions (`estimation/openai/tool_definitions.py`)

Port of `estimation/google/tool_definitions.py`. Same tool interface, different declaration format:

- `_tool_declarations() -> list[dict]` — returns OpenAI function tool format
- `_extract_function_calls(response) -> list` — parse function_call items from Responses API output
- `_process_ap_function_calls(...)` — execute tool calls, return function_call_output items
- `_handle_fetch_atlas(...)` — fetch atlas slice, return as base64 image
- `_build_atlas_grid(...)` — composite atlas image layout (reused from google/ — pure image logic)
- `_build_nudge_text(...)` — guidance text for stalled models (reused)

## Error Handling

Reuse existing `retry_with_backoff()` from `langslice/retry.py`. Exception mapping:

| OpenAI Exception | Action |
|-----------------|--------|
| `openai.RateLimitError` | Retry with backoff |
| `openai.APIConnectionError` | Retry (server may be starting up) |
| `openai.APITimeoutError` | Retry with backoff |
| `openai.BadRequestError` | Fail fast (prompt/format issue) |
| `openai.AuthenticationError` | Fail fast (config issue) |

## CLI Integration

New `--provider` argument on `estimate`, `estimate-group`, and `register` commands:

```bash
# Estimation with OpenAI-compatible provider
langslice estimate <image> --provider openai
langslice estimate <image> --provider openai --model gemma4:31b
langslice estimate-group <img1> <img2> --provider openai

# Registration with OpenAI-compatible provider
langslice register <image> --position 5.0 --provider openai

# Override endpoints via env vars
OPENAI_BASE_URL=http://localhost:8080/v1 langslice estimate <image> --provider openai
```

Default `--provider` remains `google`. No breaking changes to existing workflows.

The `--model` flag already exists for the Gemini provider. For `--provider openai`, `--model` overrides `OPENAI_MODEL`. If neither is set, defaults to `gemma4:31b`.

## Dependencies

Add `openai` to `environment.yml` and `pyproject.toml`:

```
openai>=1.0.0
```

The `openai` package is the only new dependency. It has no heavy transitive dependencies (just `httpx`, `pydantic`, etc. which are already present).

## Testing

No new test infrastructure. The estimation workflows output `APResult` / `MultiSliceResult` from `_types.py`, which are provider-agnostic. Registration workflows output the same correspondence data structures.

Integration tests require a running Ollama instance with Gemma 4, so they are manual — same as Gemini tests requiring an API key. Test with:

```bash
# Start Ollama with Gemma 4
ollama run gemma4:31b

# Run estimation
langslice estimate references/TestImages/M01.jpg --provider openai

# Compare against ground truth
python -m pytest tests/ -k "test_openai" # if/when unit tests are added
```

## Future Work

- **Agentic image generation:** Define image generation as a function tool in the Responses API. The model decides when to generate, calls a tool, we run Flux2 Klein locally, feed the image back via `previous_response_id` + `input_image`. Builds on the two-API-surface design naturally.
- **Multi-turn image editing:** Iterative refinement of generated images via `previous_response_id` conversation state. The Responses API already supports this — just needs workflow design.
- **Whole-brain pipeline:** Wire OpenAI provider into `whole_brain/pipeline.py` after individual workflow validation.
