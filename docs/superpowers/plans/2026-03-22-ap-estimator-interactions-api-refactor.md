# AP Estimator → Interactions API Refactor

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor the AP estimation agent to use the Gemini Interactions API as the sole API path, removing Vertex backend support and the generate_content loop.

**Architecture:** The estimator currently has 3 parallel API paths (generate_content, interactions, context caching) with 3 image transports (inline, File API, both). This refactor collapses everything to one clean path: Interactions API + File API with `resolution: "ultra_high"`. The Interactions API handles server-side conversation state, eliminating image accumulation across turns. Vertex backend support is removed entirely (AI Studio only).

**Tech Stack:** google-genai SDK (Interactions API, File API), Python 3.11+, PIL

**Reference implementation:** `langslice/registration/agents_tool_loop.py` — already migrated to Interactions API pattern in earlier session.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `langslice/ai/config.py` | Modify | Remove Vertex backends, remove env var feature flags, simplify to AI Studio only |
| `langslice/ai/estimator.py` | Major rewrite | Replace generate_content loop with Interactions API loop, remove context caching, simplify image transport |
| `langslice/ai/estimator_tools.py` | Modify | Convert tool declarations to FunctionParam dicts, update function call extraction for Interactions API only, update tool result building |
| `langslice/ai/estimator_debug.py` | Minor modify | Update debug artifact writing for new data structures (if needed) |

---

### Task 1: Simplify `config.py` — Remove Vertex and Feature Flags

**Files:**
- Modify: `langslice/ai/config.py`

This task removes all Vertex-specific code and the AP feature flag env vars. The result is a clean AI-Studio-only config.

- [ ] **Step 1: Read the current config.py fully**

Understand all exports and callers before removing anything.

- [ ] **Step 2: Remove Vertex backend constants and detection**

Remove `_BACKEND_VERTEX_API_KEY`, `_BACKEND_VERTEX_ADC`, `_VALID_BACKENDS` dict, `get_backend()` function (replace with a simple AI Studio assumption), `_vertex_project()`, `_vertex_location()`, and the Vertex branches in `get_client()` and `get_api_key()`.

- [ ] **Step 3: Remove AP feature flag env vars and functions**

Remove: `_ENV_AP_USE_FILE_API`, `_ENV_AP_USE_CONTEXT_CACHE`, `_ENV_AP_USE_INTERACTIONS`, `_ENV_AP_CACHE_TTL`, `_ENV_FILE_POLL_TIMEOUT_S`, and their corresponding functions: `ap_use_file_api()`, `ap_use_context_cache()`, `ap_use_interactions()`, `ap_cache_ttl()`, `file_poll_timeout_s()`.

Also remove: `supports_file_api()`, `supports_interactions_api()`, `supports_batch_api()`, `create_batch_client()`.

Simplify `feature_flags()` to only return what's still relevant (temperature, model, thinking level).

- [ ] **Step 4: Simplify `get_client()` to AI Studio only**

```python
def get_client() -> GenAIClientProtocol:
    global _client_instance
    if _client_instance is not None:
        return _client_instance
    genai_module = importlib.import_module("google.genai")
    client_cls = cast(Callable[..., GenAIClientProtocol], getattr(genai_module, "Client"))
    _client_instance = client_cls(api_key=get_api_key())
    return _client_instance
```

- [ ] **Step 5: Simplify `get_api_key()` to AI Studio only**

```python
def get_api_key() -> str:
    key = _env("GEMINI_API_KEY") or _env("GOOGLE_API_KEY")
    if key:
        return key
    raise RuntimeError(
        "GEMINI_API_KEY (or GOOGLE_API_KEY) is required. "
        "Get one at https://aistudio.google.com/apikey"
    )
```

- [ ] **Step 6: Clean up protocol classes**

Remove `_GenAIBatchesProtocol` and the `batches` field from `GenAIClientProtocol`. Keep `models`, `files`, `caches`, `interactions`.

- [ ] **Step 7: Verify syntax compiles**

Run: `python -c "import py_compile; py_compile.compile('langslice/ai/config.py', doraise=True)"`

- [ ] **Step 8: Run existing tests**

Run: `python -m pytest tests/ -x -q`
Fix any import errors from removed exports.

- [ ] **Step 9: Grep for removed function names across codebase**

Search for: `get_backend`, `ap_use_file_api`, `ap_use_context_cache`, `ap_use_interactions`, `supports_file_api`, `supports_interactions_api`, `supports_batch_api`, `create_batch_client`, `file_poll_timeout_s`, `ap_cache_ttl`.

Fix any callers (likely in `estimator.py` — will be handled in Task 2, but fix any in GUI/CLI now).

- [ ] **Step 10: Commit**

```
feat: simplify config.py to AI Studio only, remove Vertex backends and feature flags
```

---

### Task 2: Convert Tool Declarations to FunctionParam Dicts

**Files:**
- Modify: `langslice/ai/estimator_tools.py`

Convert the 5 tool declarations from `types.FunctionDeclaration` format to plain FunctionParam dicts for the Interactions API, matching the pattern in `agents_tool_loop.py`.

- [ ] **Step 1: Create a `_tool_dicts()` function**

Replace the existing `types.Tool(function_declarations=[...])` with a function returning `list[dict]`:

```python
def _tool_dicts() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "fetch_atlas_slice",
            "description": "Fetch a coronal brain atlas reference image at a specific AP coordinate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "position_mm": {
                        "type": "number",
                        "description": "AP position in mm from anterior edge"
                    }
                },
                "required": ["position_mm"]
            }
        },
        # ... same for fetch_multiple_atlas_slices, get_atlas_info, get_region_names, submit_estimate
    ]
```

Preserve all existing parameter schemas — just convert from `types.FunctionDeclaration` + `parameters_json_schema` to plain dicts.

- [ ] **Step 2: Update function call extraction**

Replace `_extract_generate_function_calls` and `_extract_interaction_function_calls` with a single `_extract_function_calls(interaction)` that works with Interactions API outputs:

```python
def _extract_function_calls(interaction: Any) -> list[dict[str, Any]]:
    calls = []
    for output in interaction.outputs:
        if getattr(output, "type", None) == "function_call":
            calls.append({
                "call_id": output.id,
                "name": output.name,
                "args": dict(output.arguments) if output.arguments else {},
            })
    return calls
```

- [ ] **Step 3: Update tool result building for Interactions API format**

The tool dispatch functions currently return `types.Part` objects. Add a helper to convert results to Interactions API `function_result` dicts:

```python
def _build_function_result(
    call_id: str, name: str, result_dict: dict, image_payloads: list
) -> list[dict[str, Any]]:
    """Build Interactions API content items for a function result."""
    items: list[dict[str, Any]] = [
        {"type": "function_result", "call_id": call_id, "name": name,
         "result": json.dumps(result_dict)},
    ]
    for payload in image_payloads:
        items.append({
            "type": "image", "data": payload["b64"],
            "mime_type": payload["mime_type"], "resolution": "ultra_high",
        })
    return items
```

- [ ] **Step 4: Update atlas image payload building**

The `_build_atlas_image_payload` / fetch functions currently build `_ImagePayload` with `types.Part`. Simplify to return base64 image data + metadata directly, since we only need the Interactions API format.

- [ ] **Step 5: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('langslice/ai/estimator_tools.py', doraise=True)"`

- [ ] **Step 6: Commit**

```
refactor: convert AP estimator tool declarations to Interactions API FunctionParam dicts
```

---

### Task 3: Rewrite Main Loop to Interactions API

**Files:**
- Modify: `langslice/ai/estimator.py`

This is the core task. Replace the `generate_content` loop (and the experimental interactions path) with a single clean Interactions API loop, using the `agents_tool_loop.py` pattern as reference.

- [ ] **Step 1: Remove the `_retry_generate` function**

No longer needed — we'll build retry logic directly for `interactions.create()`.

- [ ] **Step 2: Remove the `_ImagePayload` dataclass**

No longer needed — images go as base64 content items or File API URIs.

- [ ] **Step 3: Remove `_first_model_content` helper**

Only used by generate_content path.

- [ ] **Step 4: Remove context caching code**

Remove all cache creation, `cached_content` references, and cache cleanup from `estimate_position()`.

- [ ] **Step 5: Simplify target image preparation**

Upload target image via File API, get URI:

```python
def _upload_target_image(client, image_bytes: bytes) -> tuple[str, str, Any]:
    """Upload target slice via File API. Returns (uri, mime, file_obj)."""
    buf = io.BytesIO(image_bytes)
    uploaded = client.files.upload(
        file=buf,
        config=types.UploadFileConfig(mime_type="image/jpeg", display_name="target_slice"),
    )
    # Wait for ACTIVE state
    _wait_for_uploaded_file(client, file_name=uploaded.name, timeout_s=30.0)
    return uploaded.uri, "image/jpeg", uploaded
```

- [ ] **Step 6: Build the initial input**

```python
initial_input = [
    {"type": "text", "text": system_prompt},
    {"type": "text", "text": "Target histology brain slice:"},
    {"type": "image", "uri": target_uri, "mime_type": "image/jpeg", "resolution": "ultra_high"},
    {"type": "text", "text": atlas_info_text},
]
```

- [ ] **Step 7: Write the main loop**

Follow the `agents_tool_loop.py` pattern:

```python
tools = estimator_tools._tool_dicts()
prev_id = None
uploaded_files = [target_file_obj]

for iteration in range(1, max_iterations + 1):
    create_kwargs = {
        "model": model_name,
        "tools": tools,
        "system_instruction": system_instruction,
        "generation_config": {"temperature": temperature},
    }
    if prev_id is None:
        create_kwargs["input"] = initial_input
    else:
        create_kwargs["input"] = current_input
        create_kwargs["previous_interaction_id"] = prev_id

    # Call with retries
    interaction = _retry_interaction(client, create_kwargs, ...)
    prev_id = interaction.id

    # Extract function calls
    fc_outputs = estimator_tools._extract_function_calls(interaction)

    if not fc_outputs:
        # Nudge
        current_input = [{"type": "text", "text": "Continue searching..."}]
        continue

    # Process tool calls
    result_contents = []
    for fc in fc_outputs:
        result_dict, image_payloads = _execute_tool(fc, state, atlas, client, ...)
        result_contents.extend(
            estimator_tools._build_function_result(fc["call_id"], fc["name"], result_dict, image_payloads)
        )
        if state.estimate_result is not None:
            break

    current_input = result_contents

    if state.estimate_result is not None:
        # Done
        break
```

- [ ] **Step 8: Update tool execution to return base64 payloads**

The tool dispatch needs to return image data as base64 dicts instead of `types.Part`. Update `_execute_tool_call` to return `(result_dict, image_payloads)` where `image_payloads` is a list of `{"b64": str, "mime_type": str}`.

- [ ] **Step 9: Add File API cleanup**

```python
finally:
    for f in uploaded_files:
        try:
            client.files.delete(name=f.name)
        except Exception:
            logger.warning("Failed to delete %s", getattr(f, "name", "?"))
```

- [ ] **Step 10: Update `_retry_interaction` helper**

```python
def _retry_interaction(client, kwargs, *, on_progress=None, label=""):
    for attempt in range(1, 5):
        try:
            interaction = client.interactions.create(**kwargs)
            return interaction
        except Exception as exc:
            if attempt < 4:
                time.sleep(min(2 ** attempt, 8))
            else:
                raise
```

- [ ] **Step 11: Verify syntax**

Run: `python -c "import py_compile; py_compile.compile('langslice/ai/estimator.py', doraise=True)"`

- [ ] **Step 12: Commit**

```
refactor: rewrite AP estimator main loop to Interactions API
```

---

### Task 4: Update Debug Artifacts and Telemetry

**Files:**
- Modify: `langslice/ai/estimator_debug.py`
- Modify: `langslice/ai/estimator.py` (telemetry collection)

- [ ] **Step 1: Read `estimator_debug.py` fully**

Understand what debug artifacts it writes and what data structures it expects.

- [ ] **Step 2: Update telemetry collection**

The Interactions API doesn't return `usage_metadata` the same way as generate_content. Update the turn metric collection to extract what's available from the interaction response.

- [ ] **Step 3: Update debug artifact writing**

Ensure `write_debug_artifacts()` works with the new data structures. The reasoning log format may change slightly (interaction IDs instead of history indices).

- [ ] **Step 4: Test with a real run**

Run: `langslice estimate test_runs/standard_test_slice.png --atlas allen_mouse_25um`
(or whatever the CLI command is for AP estimation)

Verify debug artifacts are written correctly.

- [ ] **Step 5: Commit**

```
fix: update AP estimator debug artifacts for Interactions API
```

---

### Task 5: Update Callers (GUI, CLI) and Final Cleanup

**Files:**
- Modify: `langslice/gui/workers.py` (if it references removed config functions)
- Modify: `langslice/gui/settings_dialog.py` (if it exposes removed feature flags)
- Modify: `langslice/cli.py` (if needed)

- [ ] **Step 1: Grep for all removed function/constant references**

Search for: `ap_use_file_api`, `ap_use_context_cache`, `ap_use_interactions`, `get_backend`, `supports_file_api`, `supports_interactions_api`, `_BACKEND_VERTEX`, `create_batch_client`, `_ImagePayload`, `_retry_generate`.

- [ ] **Step 2: Fix all callers**

Remove references to feature flags from GUI settings. Remove backend selection UI if present.

- [ ] **Step 3: Run full test suite**

Run: `python -m pytest tests/ -x -q`

- [ ] **Step 4: Run ruff and basedpyright**

Run: `python -m ruff check . && python -m basedpyright`

- [ ] **Step 5: Run a real AP estimation end-to-end**

Test with the standard test image to verify the full pipeline works.

- [ ] **Step 6: Final commit**

```
chore: clean up removed Vertex/feature-flag references across codebase
```

---

## Notes for Implementation

- **Always check context7 `/googleapis/python-genai`** before writing any Gemini API code. The PreToolUse hook will remind you.
- **Image resolution:** Set `"resolution": "ultra_high"` on ALL `ImageContentParam` dicts.
- **File API images:** Use `"uri"` field (not `"data"`) for File API uploads. Base64 `"data"` for inline tool response images.
- **Function results:** Send as separate content items alongside the `function_result` dict (not inside `result.items` — that causes 400 errors after ~8 turns).
- **The `_wait_for_uploaded_file` helper** should be kept — it polls File API until the file is ACTIVE.
- **Submission guards** in `estimator_tools.py` (broad sweep, narrow sweep, neighbor bracket) should be preserved exactly as-is.
