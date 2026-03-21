# Tool Loop Registration Overhaul — Design Spec

## Goal

Rewrite the multimodal tool-loop registration workflow (`agents_tool_loop.py`) to use the official Gemini function calling protocol, enrich the prompt with domain-expert guidance, and add a confirm-after-zoom gate that enforces quality verification before locking landmark points.

## Motivation

1. **The current code simulates tool calls via structured JSON output** (`response_mime_type: "application/json"` + `response_json_schema`). Gemini has a native function calling protocol (`FunctionDeclaration` + `types.Tool`) that the model is trained and optimized for.
2. **The prompt is too terse.** The master doc (`docs/LangSlice_Master_Doc (2).md`) contains detailed neuroanatomical guidance that isn't in the current 18-line prompt.
3. **Place and lock are the same action.** There's no verification gate — the model can place a point and move on without zooming to confirm.
4. **Border vs interior points aren't distinguished.** The master doc explicitly calls for separate counts, and border points are critical for alignment quality.
5. **The code uses raw dicts instead of the SDK's typed objects.** This is fragile and diverges from official documentation.

## Constraints

- DO NOT modify `agents_image_gen.py` — it uses a different workflow entirely.
- The shared utilities in `agents.py` (retry logic, coordinate helpers, trace emission) may be adapted but their external behavior must be preserved.
- The AP estimator (`ai/estimator.py` and related) is out of scope.
- No cost optimization logic — just tracking infrastructure.

---

## Architecture

### SDK Modernization

Replace raw dict construction with official `google.genai.types` objects throughout the tool loop:

| Current (raw dicts) | New (SDK types) |
|---|---|
| `{"inline_data": {"mime_type": "image/png", "data": bytes}}` | `types.Part.from_bytes(data=bytes, mime_type="image/png")` |
| `{"role": "user", "parts": [...]}` | `types.Content(role="user", parts=[...])` |
| `{"role": "model", "parts": [{"text": json.dumps(action)}]}` | Model response content is used directly from `response.candidates[0].content` |
| `{"role": "user", "parts": tool_result_parts}` | `types.Content(role="user", parts=[types.Part(function_response=...)])` — note: role is `"user"`, not `"function"` per SDK convention |
| `response_mime_type` + `response_json_schema` in config | `tools=[types.Tool(function_declarations=[...])]` in config |

### File API for Base Images (AI Studio only)

The File API (`client.files.upload`) is only available on the AI Studio backend. On Vertex backends (`vertex_api_key`, `vertex_adc`), it raises `ValueError`. The tool loop must handle both:

- **AI Studio backend:** Upload base atlas and slice images via File API. Reference by URI in the initial message. Clean up in `finally` block.
- **Vertex backends:** Fall back to `types.Part.from_bytes()` inline data for base images.

Use `langslice.ai.config.get_backend()` to detect which path to take. The `supports_file_api()` predicate already exists in `config.py` for this purpose.

```python
from langslice.ai.config import supports_file_api

if supports_file_api():
    atlas_file = client.files.upload(
        file=atlas_bytes_io,
        config=types.UploadFileConfig(mime_type="image/png")
    )
    slice_file = client.files.upload(
        file=slice_bytes_io,
        config=types.UploadFileConfig(mime_type="image/png")
    )
    atlas_part = atlas_file  # SDK converts to Part automatically
    slice_part = slice_file
else:
    atlas_part = types.Part.from_bytes(data=atlas_bytes, mime_type="image/png")
    slice_part = types.Part.from_bytes(data=slice_bytes, mime_type="image/png")
```

Files are cleaned up in a `finally` block (only if File API was used).

### Image Resolution

The tool loop accepts a `vlm_resolution` parameter: `"1K"` or `"2K"` (default `"2K"`). Images are downscaled to this resolution before upload/inline. This controls the token cost per image.

---

## Tool Definitions

Five tools defined as `types.FunctionDeclaration`:

### `view_overview`

Returns the full atlas and slice images with all current annotations rendered.

**Parameters:** None.

### `view_zoom_pair`

Zooms into a region of both atlas and slice for detailed inspection. Existing annotations are visible in the zoomed view.

**Parameters:**
- `zoom` (number, required) — Zoom factor (e.g., 3.0 for 3x, 1.5 for 1.5x)
- `atlas_center_2d` (array of 2 integers, required) — [y, x] in 0-1000 normalized range
- `slice_center_2d` (array of 2 integers, required) — [y, x] in 0-1000 normalized range

### `place_point_pair`

Place a provisional landmark pair on atlas and slice. The point is NOT confirmed until `confirm_point` is called after zoom inspection.

**Parameters:**
- `label` (string, required) — Point label (e.g., "1", "2")
- `category` (string enum, required) — `"border"` or `"interior"`
- `feature_description` (string, required) — Rich description of the anatomical feature being matched (e.g., "the deepest point of the dorsal midline notch")
- `atlas_point_2d` (array of 2 integers) — [y, x] global coordinates in 0-1000 range
- `slice_point_2d` (array of 2 integers) — [y, x] global coordinates in 0-1000 range
- `atlas_point_2d_local` (array of 2 integers) — [y, x] local coordinates relative to last zoom view
- `slice_point_2d_local` (array of 2 integers) — [y, x] local coordinates relative to last zoom view
- `artifact_note` (string, optional) — Note about damage/artifacts at this location

Either global or local coordinate pairs are required (same logic as current code).

### `confirm_point`

Confirm a provisional point after zoom inspection. **Hard gate:** rejected if `view_zoom_pair` has not been called since the last `place_point_pair` for this label.

**Parameters:**
- `label` (string, required) — Label of the point to confirm

### `finish`

Complete landmark placement. **Hard gate:** rejected if `border_count` confirmed border points and `interior_count` confirmed interior points are not both met.

**Parameters:** None.

---

## Point Lifecycle

```
place_point_pair("1", category="border")
    → point "1" is PROVISIONAL, added to awaiting_zoom set

view_zoom_pair(zoom=3.0, ...)
    → "1" removed from awaiting_zoom (zoom happened)

confirm_point("1")
    → "1" is now CONFIRMED (counts toward border quota)

# Re-placing a confirmed point resets it:
place_point_pair("1", category="border", ...)
    → "1" is back to PROVISIONAL, added to awaiting_zoom again
```

### Session State

`RegistrationAnnotationSession` tracks:
- `border_count` / `interior_count` — quotas to meet
- `awaiting_zoom: set[str]` — labels that need zoom before confirm (stored in `session.metadata`)
- `confirmed: set[str]` — labels that have been confirmed (stored in `session.metadata`)
- Each `LandmarkAnnotation` gets a `category` field ("border" or "interior", default "border")

**Frozen dataclass note:** `LandmarkAnnotation` is `frozen=True`. To "confirm" a point, we do NOT mutate the annotation. Instead, the confirmed/awaiting_zoom state lives in `session.metadata` sets. The `_upsert_annotation` pattern (replace-by-label) handles re-placement.

**Confirm edge cases:**
- `confirm_point` for a label that was never placed → reject with "no provisional point with label X"
- `confirm_point` for a label already confirmed → no-op success (idempotent)

### Confirmed Entry Filter

`_confirmed_tool_loop_entries()` is updated to return only entries where:
1. The label exists in `session.metadata["confirmed"]`
2. Both atlas and slice annotations exist for that label

This replaces the current filter of `status != "not_visible"`. The `status` field on `LandmarkAnnotation` is no longer used by the tool loop (kept for compatibility with image_gen workflow).

### Quota Enforcement at `finish`

```python
confirmed_border = count of confirmed annotations where category == "border"
confirmed_interior = count of confirmed annotations where category == "interior"

if confirmed_border < border_count or confirmed_interior < interior_count:
    reject with specific message showing current vs required counts
```

---

## Entry Point Changes

```python
def _estimate_correspondences_tool_loop(
    client: Any,
    *,
    prepared: _PreparedRegistrationInputs,
    atlas_name: str,
    position_mm: float,
    border_count: int | None = None,      # NEW
    interior_count: int | None = None,     # NEW
    on_progress: Callable[[str], None] | None = None,
    on_trace: Callable[[dict[str, object]], None] | None = None,
) -> list[dict[str, object]]:
```

If `border_count` and `interior_count` are both `None`, default to 50/50 split of `prepared.target_count`. If `target_count` is odd, give the extra point to border.

**Relationship to `min_edge_landmarks`:** The existing `min_edge_landmarks` parameter (in `_PreparedRegistrationInputs`) is superseded by `border_count` for the tool loop workflow. `min_edge_landmarks` remains available for the image_gen workflow (which uses a different mechanism). The tool loop ignores `min_edge_landmarks` and uses `border_count`/`interior_count` exclusively.

**`vlm_resolution` mapping:** The `vlm_resolution` parameter (`"1K"` or `"2K"`, default `"2K"`) maps to the existing `max_long_edge` parameter in `prepare_image_for_vlm()`: `"1K"` → `max_long_edge=1024`, `"2K"` → `max_long_edge=2048`. This is resolved in `_prepare_registration_inputs()` before reaching the tool loop.

The caller chain (`agents.py` → `runtime.py` → `core.py` → CLI) passes through the new params. CLI gets `--border-count` and `--interior-count` flags. The existing `--landmarks` flag continues to set `target_count` (which is split 50/50 if border/interior aren't specified).

---

## Conversation Loop

### Returning images in tool responses

`FunctionResponse.response` is a `dict[str, Any]` — it cannot carry binary image data that the model can see. Tool results that include images (view_overview, view_zoom_pair, place_point_pair) must send images as a **separate `role: "user"` content message** after the function response content. This is a two-message pattern per iteration:

1. `types.Content(role="user", parts=[FunctionResponse parts])` — structured result
2. `types.Content(role="user", parts=[image parts + text context])` — annotated images the model can see

This matches how the current code works (tool results go in `role: "user"` messages with inline images), just using typed SDK objects.

### Retry logic

The loop uses `_agents._retry_generate()` (not raw `client.models.generate_content`) to preserve the existing retry handling for 429/500/502/503/504 errors and progress heartbeat callbacks. The `_retry_generate` helper is adapted to accept `types.Content` history and `types.GenerateContentConfig`.

### Trace emission

All trace events (`tool_call_event`, `tool_result_event`, `model_event`) are preserved in the new loop, using the same `_agents._emit_trace(on_trace, ...)` pattern as today.

### Pseudocode

```python
# 1. Upload base images (AI Studio) or prepare inline (Vertex)
atlas_part, slice_part, uploaded_files = _prepare_base_images(client, prepared)

# 2. Build tools
tools = types.Tool(function_declarations=[
    view_overview_decl, view_zoom_pair_decl,
    place_point_pair_decl, confirm_point_decl, finish_decl
])

# 3. Config
config = types.GenerateContentConfig(
    tools=[tools],
    thinking_config=types.ThinkingConfig(
        thinking_level=types.ThinkingLevel(prepared.thinking_level)
    ),
    temperature=prepared.temperature,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

# 4. Initial message
history = [types.Content(role="user", parts=[
    types.Part.from_text(system_prompt),
    types.Part.from_text("Atlas reference image:"),
    atlas_part,
    types.Part.from_text("Histology slice image:"),
    slice_part,
    types.Part.from_text(json.dumps(session_summary)),
])]

# 5. Loop
accumulated_usage = {}  # Token tracking

try:
    for iteration in range(max_steps):
        response = _agents._retry_generate(
            client, model=prepared.model_name,
            contents=history, config=config,
            request_label=f"Registration tool-loop step {iteration}",
            on_progress=on_progress,
        )

        # Track tokens
        _accumulate_usage(accumulated_usage, response.usage_metadata)

        # Emit model trace event
        _agents._emit_trace(on_trace, model_event(...))

        # Add model response to history
        history.append(response.candidates[0].content)

        # Process function calls
        finished = False
        function_response_parts = []
        image_parts = []

        for part in response.candidates[0].content.parts:
            if not part.function_call:
                continue

            fc = part.function_call
            _agents._emit_trace(on_trace, tool_call_event(...))

            result_dict, result_images, is_finished = _execute_tool(
                fc, session=session, atlas_image=..., slice_image=...,
                iteration=iteration, on_trace=on_trace,
            )

            function_response_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=fc.name,
                    response=result_dict,
                    id=fc.id,
                )
            ))
            image_parts.extend(result_images)

            if is_finished:
                finished = True

        # Send function responses (role="user" per SDK convention)
        if function_response_parts:
            history.append(types.Content(
                role="user", parts=function_response_parts
            ))

        # Send result images as a separate user message (if any)
        if image_parts:
            history.append(types.Content(role="user", parts=image_parts))

        if finished:
            return _confirmed_entries(session)

    raise RuntimeError("Tool-loop exceeded maximum steps")

finally:
    # Clean up File API uploads (if any)
    for f in uploaded_files:
        try:
            client.files.delete(name=f.name)
        except Exception:
            logger.warning("Failed to delete uploaded file %s", f.name)
    # Log accumulated token usage
    logger.info("Tool loop token usage: %s", accumulated_usage)
    session.metadata["token_usage"] = accumulated_usage
```

### `_execute_tool` return signature

Each tool handler returns a 3-tuple:
- `result_dict: dict` — JSON-serializable result for `FunctionResponse.response`
- `result_images: list[types.Part]` — annotated/zoomed image parts (may be empty)
- `is_finished: bool` — True only for successful `finish`

For example, `view_zoom_pair` returns:
```python
result_dict = {"status": "ok", "zoom": 3.0, "atlas_window": {...}, "slice_window": {...}}
result_images = [
    types.Part.from_bytes(data=atlas_zoom_bytes, mime_type="image/png"),
    types.Part.from_text("Zoomed atlas view at 3.0x"),
    types.Part.from_bytes(data=slice_zoom_bytes, mime_type="image/png"),
    types.Part.from_text("Zoomed slice view at 3.0x"),
]
is_finished = False
```

---

## Prompt

Replace the current 18-line prompt with rich domain guidance from the master doc. The prompt is built by `_build_tool_loop_prompt()` and includes:

**Section 1 — Role and context:**
```
You are an expert neuroanatomist placing matched landmark points between a
histology brain slice and a reference atlas.
Atlas: {atlas_name}
AP position: {position_mm:.3f} mm
```

**Section 2 — Task:**
```
Place {border_count} paired landmarks on the outermost edge/border of the
brain, then {interior_count} paired landmarks in the interior.
Label points sequentially: 1, 2, 3, ...
```

**Section 3 — Per-point workflow:**
```
For each point, follow this workflow:

1. DEFINE the target feature before placing anything.
   State the exact local anatomical/geometric feature you are matching.
   Use rich, specific descriptions such as "the deepest point of the dorsal
   midline notch" or "the lower third of the intact medial wall of the right
   lateral ventricle" — not broad labels like "in the ventricle."

2. PLACE the point pair using place_point_pair.
   Choose the same kind of local feature in both images.
   Prioritize local correspondence in depth, curvature, neighboring contours,
   and boundary context over global shape similarity.

3. ZOOM IN at 3x and inspect locally using view_zoom_pair.
   Compare the atlas and slice placements side by side.

4. ZOOM OUT to 1.5x and sanity-check in broader context.
   Verify the point still makes sense within the surrounding anatomy.
   Adjust the point (re-place) if needed before confirming.

5. HANDLE DAMAGE explicitly.
   If the slice region is torn, distorted, collapsed, bubbled, folded, or
   weakly stained, say so in the artifact_note.
   Prefer a stable intact neighboring feature over a distorted tip or
   artificial edge.

6. CONFIRM the point using confirm_point.
   You MUST zoom (step 3 or 4) between placing and confirming.
   The system will reject confirm_point if no zoom has occurred since placement.
```

**Section 4 — Coordinate system:**
```
Use [y, x] integers in the 0-1000 normalized range.
After a zoom view, prefer atlas_point_2d_local and slice_point_2d_local
so coordinates are relative to the zoomed images. The system maps them back
to the full image automatically.
From the full overview, use atlas_point_2d and slice_point_2d.
```

**Section 5 — Rules:**
```
- Do NOT place points on the black background.
- Do NOT assume left-right symmetry. Hemispheres may differ substantially.
- Do NOT rely on broad anatomical guesses without local confirmation.
- Use already confirmed points only as loose anchors, not rigid constraints.
- Do not keep moving previously confirmed points unless you conclude the
  prior feature definition was wrong.
- When all quotas are met, call finish.
```

---

## Token Tracking

Accumulate `response.usage_metadata` across all iterations:

```python
def _accumulate_usage(accumulated: dict, usage_metadata) -> None:
    if usage_metadata is None:
        return
    for field in ("prompt_token_count", "candidates_token_count",
                  "total_token_count", "cached_content_token_count",
                  "thoughts_token_count", "tool_use_prompt_token_count"):
        value = getattr(usage_metadata, field, None)
        if value is not None:
            accumulated[field] = accumulated.get(field, 0) + value
```

The accumulated dict is:
- Logged at INFO level when the loop completes
- Included in the debug artifacts if `LANGSLICE_VLM_DEBUG_DIR` is set
- Returned as part of the session metadata (available to callers)

No cost calculation or optimization — just raw counts for future use.

---

## Files Changed

### Modified
- `langslice/registration/agents_tool_loop.py` — Full rewrite: proper function calling, new tools, enriched prompt, File API upload, token tracking
- `langslice/registration/agents.py` — Update `_image_to_inline_data` to use `types.Part.from_bytes()`, pass through `border_count`/`interior_count` params, update shared helpers for SDK types where needed
- `langslice/registration/types.py` — Add `category` and `confirmed` fields to `LandmarkAnnotation`, add `border_count`/`interior_count` to `RegistrationAnnotationSession`
- `langslice/registration/runtime.py` — Pass through `border_count`/`interior_count`
- `langslice/registration/core.py` — Pass through `border_count`/`interior_count`
- `langslice/cli.py` — Add `--border-count` and `--interior-count` flags

### New
- None expected (all changes within existing files)

### Not Modified
- `langslice/registration/agents_image_gen.py` — Separate workflow, out of scope
- `langslice/ai/` — AP estimator is out of scope
- `langslice/gui/` — GUI passes `target_count` which continues to work (split to 50/50 internally)

---

## Testing

### Unit Tests (no live API)
- Tool schema: verify all 5 FunctionDeclarations are well-formed
- Session state: provisional → zoom → confirmed lifecycle
- Confirm gate: rejected without zoom, accepted after zoom
- Re-place resets confirmed point to provisional
- Quota enforcement: finish rejected when quotas not met, accepted when met
- Border/interior default split: odd target_count gives extra to border
- Token accumulation: usage_metadata aggregated correctly
- Coordinate helpers: existing tests still pass (local→global mapping)

### Integration Tests (mocked client)
- Full loop with mocked `client.models.generate_content` returning `FunctionCall` parts
- Verify conversation history uses `types.Content` with correct roles
- Verify File API upload called and cleanup in finally block
- Verify `types.FunctionResponse` sent back correctly

### Manual Verification
- Run against live Gemini with a real slice image
- Verify the model follows the zoom-then-confirm workflow
- Verify border/interior quotas are enforced
- Send output images to user for visual verification
