/**
 * Browser-side TS port of the Python single-slice position-estimation agent
 * (src/langslice_harness/harness/estimation/single_slice.py). Talks to a
 * local `litert-lm serve` endpoint via the `@google/genai` JS SDK; the SDK's
 * `httpOptions.baseUrl` redirects REST traffic to the local server, and the
 * required `apiKey` is ignored by litert-lm.
 */

import { GoogleGenAI, Type } from "@google/genai";
import type {
  Content,
  FunctionDeclaration,
  GenerateContentResponse,
  Part,
} from "@google/genai";

import { getCoronalSliceLayers } from "./browserCommands";

// --- Public types ----------------------------------------------------------

export interface AgentEvent {
  type: "model_text" | "tool_call" | "tool_result" | "error" | "done";
  text?: string;
  toolName?: string;
  toolArgs?: Record<string, unknown>;
  toolResult?: Record<string, unknown>;
  fetchedAtlasUrls?: string[];
  error?: string;
}

/** Coarse Gemini control. Mapped to thinkingBudget token counts in the
 *  generateContent config. "off" disables thinking entirely. Has no effect
 *  on local sidecar (gemma) calls — silently dropped. */
export type ThinkingPreset = "off" | "low" | "medium" | "high";

/** Coarse Gemini media-resolution dial. Maps to the SDK's
 *  MEDIA_RESOLUTION_LOW/MEDIUM/HIGH enum. Skipped for non-Gemini callers. */
export type MediaResolutionPreset = "low" | "medium" | "high";

export interface EstimateParams {
  imageUrl: string;
  atlasName: "allen_mouse_25um";
  plane: "coronal";
  posLo: number;
  posHi: number;
  species: string;
  modelId: string;
  baseUrl?: string;
  maxIterations?: number;
  signal?: AbortSignal;
  /** When true, drop the broad/narrow strategy + per-call guidance entirely
   *  and just ask for an estimate with one tool call. Used for the fine-
   *  tuned LangSlice-Gemma models where the long playbook hurts more than
   *  it helps. */
  freeMode?: boolean;
  /** Optional Gemini-only knobs. Ignored when the request is routed to the
   *  local sidecar or any non-Gemini engine. */
  thinkingPreset?: ThinkingPreset;
  mediaResolution?: MediaResolutionPreset;
}

export interface EstimateResult {
  positionMm: number;
  reasoning: string;
  iterations: number;
}

// --- Prompt builder (port of prompts.build_single_slice_prompt) -----------

const PLANE_BOILERPLATE: Record<string, string> = {
  coronal:
    "Coordinate system: 0.0 mm is the anterior edge (olfactory bulb); " +
    "larger mm moves posterior toward the cerebellum.",
  sagittal:
    "Coordinate system: 0.0 mm is at the lateral edge of one hemisphere; " +
    "larger mm moves toward the midline. The two hemispheres are mirror " +
    "images, so estimate the canonical (single-hemisphere) ML position; " +
    "your estimate should fall in the canonical hemisphere range.",
  horizontal:
    "Coordinate system: 0.0 mm is dorsal (top of brain); " +
    "larger mm moves ventral.",
};

const PLANE_AXIS_LABEL: Record<string, string> = {
  coronal: "AP",
  sagittal: "ML",
  horizontal: "DV",
};

function buildSingleSlicePrompt(args: {
  atlasName: string;
  plane: string;
  posLo: number;
  posHi: number;
  species: string;
}): string {
  const axis = PLANE_AXIS_LABEL[args.plane];
  const boilerplate = PLANE_BOILERPLATE[args.plane];
  const lo = args.posLo.toFixed(2);
  const hi = args.posHi.toFixed(2);
  return (
    `You are an expert neuroanatomist. You are given a histology brain ` +
    `slice image and must determine its ${axis} position within a ` +
    `reference atlas. ${boilerplate}\n\n` +
    `Atlas: ${args.atlasName} (${args.species}). ` +
    `Valid ${axis} range: ${lo}-${hi} mm.\n\n` +
    `You have tools to fetch atlas reference images and submit your final estimate.\n\n` +
    `STUDENT-SIZED STRATEGY:\n` +
    `1. IN YOUR FIRST RESPONSE, do two things simultaneously:\n` +
    `   a) Describe 2-3 prominent anatomical landmarks in the target ` +
    `slice (e.g., 'anterior commissure visible', 'hippocampus forming', ` +
    `'corpus callosum is thick', 'ventricles are large').\n` +
    `   b) Based on those landmarks, call \`fetch_atlas\` with broadly ` +
    `spaced positions (e.g., [2, 4, 6, 8, 10]) to find the general ` +
    `${axis} neighborhood.\n` +
    `2. COMPARE: after each fetch, briefly say which atlas section best ` +
    `matches the target and which landmark evidence drove that choice.\n` +
    `3. RECOVER: if the broad sweep is clearly wrong, call \`fetch_atlas\` ` +
    `with a different broad ${axis} range instead of narrowing in the ` +
    `wrong neighborhood.\n` +
    `4. NARROW: once one area is close, call \`fetch_atlas\` with tighter ` +
    `positions around your best candidate.\n` +
    `5. OPTIONAL: if the narrow result is still ambiguous, make one finer ` +
    `\`fetch_atlas\` call; otherwise call \`submit_estimate\` with concise ` +
    `reasoning.\n\n` +
    `If atlas images don't match the target, DO NOT keep narrowing in ` +
    `the same area — try a completely different ${axis} range.`
  );
}

/** Minimal prompt for fine-tuned LangSlice-Gemma models: no playbook, no
 *  enforced order. The tools are still available — Gemma can use them if it
 *  wants — but it's free to submit a one-shot estimate. */
function buildFreeModePrompt(args: {
  atlasName: string;
  plane: string;
  posLo: number;
  posHi: number;
  species: string;
}): string {
  const axis = PLANE_AXIS_LABEL[args.plane];
  const boilerplate = PLANE_BOILERPLATE[args.plane];
  const lo = args.posLo.toFixed(2);
  const hi = args.posHi.toFixed(2);
  return (
    `Estimate the ${axis} position of the given histology section in the ` +
    `${args.atlasName} atlas (${args.species}). ${boilerplate} ` +
    `Valid range: ${lo}-${hi} mm. ` +
    `You may optionally call \`fetch_atlas\` to compare against reference ` +
    `sections, then call \`submit_estimate\` with your final answer.`
  );
}

// --- Position clamp + dedupe (port of tools._clamp_and_dedupe_positions) --

function clampAndDedupePositions(
  positions: unknown[],
  posLo: number,
  posHi: number,
  dedupeTol = 0.02,
): number[] {
  // Defensive flatten: model occasionally emits [[1.5, 2.5]] not [1.5, 2.5].
  const flat: number[] = [];
  for (const p of positions) {
    if (Array.isArray(p)) {
      for (const q of p) {
        const n = Number(q);
        if (Number.isFinite(n)) flat.push(n);
      }
    } else {
      const n = Number(p);
      if (Number.isFinite(n)) flat.push(n);
    }
  }
  const clamped = flat.map((p) => Math.max(posLo, Math.min(posHi, p)));
  const out: number[] = [];
  for (const p of clamped) {
    if (out.some((q) => Math.abs(p - q) <= dedupeTol)) continue;
    out.push(p);
  }
  return out;
}

// --- Helpers --------------------------------------------------------------

async function fetchAsInlinePart(url: string): Promise<Part> {
  const r = await fetch(url, { cache: "force-cache" });
  if (!r.ok) throw new Error(`Fetch ${url} failed: HTTP ${r.status}`);
  const blob = await r.blob();
  const mimeType = blob.type || "image/png";
  const buf = await blob.arrayBuffer();
  const bytes = new Uint8Array(buf);
  // Chunked btoa to avoid call-stack blow-up on large PNGs.
  let bin = "";
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    bin += String.fromCharCode.apply(
      null,
      bytes.subarray(i, i + CHUNK) as unknown as number[],
    );
  }
  return { inlineData: { mimeType, data: btoa(bin) } };
}

const fetchAtlasDecl: FunctionDeclaration = {
  name: "fetch_atlas",
  description:
    "Fetch 1-8 atlas sections along the session's slicing plane. " +
    "Positions outside the valid range are clamped; duplicates within " +
    "0.02 mm are coalesced.",
  parameters: {
    type: Type.OBJECT,
    properties: {
      positions_mm: {
        type: Type.ARRAY,
        description: "List of 1-8 positions in mm along the slicing axis.",
        items: { type: Type.NUMBER },
      },
    },
    required: ["positions_mm"],
  },
};

function buildSubmitEstimateDecl(freeMode: boolean): FunctionDeclaration {
  return {
    name: "submit_estimate",
    description: freeMode
      ? "Submit the final position estimate in mm along the slicing axis."
      : "Submit the final position estimate. Only call after broad + narrow " +
        "atlas sweeps and verifying at least one neighbor on each side.",
    parameters: {
      type: Type.OBJECT,
      properties: {
        position_mm: {
          type: Type.NUMBER,
          description: "Final position in mm along the slicing axis.",
        },
        reasoning: {
          type: Type.STRING,
          description: "Concise reasoning behind the estimate.",
        },
      },
      required: ["position_mm", "reasoning"],
    },
  };
}

// Map presets to the wire-format values the Google GenAI SDK expects.
function thinkingBudgetFromPreset(preset: ThinkingPreset): number {
  switch (preset) {
    case "off":
      return 0;
    case "low":
      return 2048;
    case "medium":
      return 8192;
    case "high":
      return 24576;
  }
}
function mediaResolutionFromPreset(
  preset: MediaResolutionPreset,
): "MEDIA_RESOLUTION_LOW" | "MEDIA_RESOLUTION_MEDIUM" | "MEDIA_RESOLUTION_HIGH" {
  switch (preset) {
    case "low":
      return "MEDIA_RESOLUTION_LOW";
    case "medium":
      return "MEDIA_RESOLUTION_MEDIUM";
    case "high":
      return "MEDIA_RESOLUTION_HIGH";
  }
}

/** Build the per-call Gemini extras (thinking + media resolution). Returns
 *  an empty object for non-Gemini models so the local sidecar isn't asked
 *  to honor fields it doesn't understand. */
function buildGeminiExtras(
  modelId: string,
  thinkingPreset?: ThinkingPreset,
  mediaResolution?: MediaResolutionPreset,
): Record<string, unknown> {
  if (!/gemini/i.test(modelId)) return {};
  const extras: Record<string, unknown> = {};
  if (thinkingPreset) {
    extras.thinkingConfig = {
      thinkingBudget: thinkingBudgetFromPreset(thinkingPreset),
    };
  }
  if (mediaResolution) {
    extras.mediaResolution = mediaResolutionFromPreset(mediaResolution);
  }
  return extras;
}

// Recover one call from a literal "<|tool_call|>{json}<|tool_call|>" payload.
interface ParsedToolCall {
  name: string;
  args: Record<string, unknown>;
}
function parseLegacyToolCall(text: string): ParsedToolCall | null {
  const m = text.match(/<\|tool_call\|>([\s\S]*?)<\|tool_call\|>/);
  if (!m) return null;
  try {
    const obj = JSON.parse(m[1].trim()) as Record<string, unknown>;
    const name =
      typeof obj.name === "string"
        ? obj.name
        : typeof (obj as { tool?: unknown }).tool === "string"
          ? (obj as { tool: string }).tool
          : null;
    if (!name) return null;
    const rawArgs =
      (obj.arguments as Record<string, unknown> | undefined) ??
      (obj.args as Record<string, unknown> | undefined) ??
      (obj.parameters as Record<string, unknown> | undefined) ??
      {};
    return { name, args: rawArgs };
  } catch {
    return null;
  }
}

function extractText(resp: GenerateContentResponse): string {
  const direct = (resp as unknown as { text?: string }).text;
  if (typeof direct === "string" && direct.length > 0) return direct;
  const parts = resp.candidates?.[0]?.content?.parts ?? [];
  return parts
    .map((p) => (typeof p.text === "string" ? p.text : ""))
    .filter((s) => s.length > 0)
    .join("");
}

function extractFunctionCalls(
  resp: GenerateContentResponse,
): Array<{ name: string; args: Record<string, unknown> }> {
  const direct = (
    resp as unknown as {
      functionCalls?: Array<{ name?: string; args?: Record<string, unknown> }>;
    }
  ).functionCalls;
  if (Array.isArray(direct) && direct.length > 0) {
    return direct
      .filter((fc): fc is { name: string; args: Record<string, unknown> } =>
        typeof fc.name === "string",
      )
      .map((fc) => ({ name: fc.name, args: fc.args ?? {} }));
  }
  const out: Array<{ name: string; args: Record<string, unknown> }> = [];
  for (const p of resp.candidates?.[0]?.content?.parts ?? []) {
    const fc = (p as Part).functionCall;
    if (fc && typeof fc.name === "string") {
      out.push({
        name: fc.name,
        args: (fc.args as Record<string, unknown>) ?? {},
      });
    }
  }
  return out;
}

// --- Main entry: runEstimate ----------------------------------------------

export async function* runEstimate(
  p: EstimateParams,
): AsyncGenerator<AgentEvent, EstimateResult, void> {
  const baseUrl = p.baseUrl ?? "http://127.0.0.1:8765";
  // Default matches Python `_DEFAULT_MAX_ITERATIONS_SINGLE=20` (runner.py:48).
  const maxIterations = p.maxIterations ?? 20;

  const client = new GoogleGenAI({
    apiKey: "not-used-locally",
    httpOptions: { baseUrl },
  });

  const systemInstruction = (p.freeMode ? buildFreeModePrompt : buildSingleSlicePrompt)({
    atlasName: p.atlasName,
    plane: p.plane,
    posLo: p.posLo,
    posHi: p.posHi,
    species: p.species,
  });

  const initialImagePart = await fetchAsInlinePart(p.imageUrl);
  const contents: Content[] = [
    {
      role: "user",
      parts: [
        initialImagePart,
        { text: "Estimate the AP position of this slice." },
      ],
    },
  ];

  const submitEstimateDecl = buildSubmitEstimateDecl(p.freeMode ?? false);
  const tools = [
    { functionDeclarations: [fetchAtlasDecl, submitEstimateDecl] },
  ];
  const geminiExtras = buildGeminiExtras(
    p.modelId,
    p.thinkingPreset,
    p.mediaResolution,
  );

  for (let iteration = 1; iteration <= maxIterations; iteration++) {
    if (p.signal?.aborted) {
      yield { type: "error", error: "aborted" };
      throw new Error("aborted");
    }

    let resp: GenerateContentResponse;
    try {
      resp = await client.models.generateContent({
        model: p.modelId,
        contents,
        config: {
          systemInstruction,
          temperature: 1.0,
          maxOutputTokens: 4000,
          tools,
          ...geminiExtras,
        },
      });
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      yield { type: "error", error: msg };
      throw e;
    }

    const text = extractText(resp);
    if (text) yield { type: "model_text", text };

    let calls = extractFunctionCalls(resp);
    if (calls.length === 0 && text.includes("<|tool_call|>")) {
      const parsed = parseLegacyToolCall(text);
      if (parsed) calls = [parsed];
    }

    const modelContent = resp.candidates?.[0]?.content ?? null;

    if (calls.length === 0) {
      // No tool call: nudge the model to call one on the next turn.
      contents.push(
        modelContent ?? {
          role: "model",
          parts: text ? [{ text }] : [{ text: "" }],
        },
      );
      contents.push({
        role: "user",
        parts: [
          {
            text:
              "You must call either fetch_atlas or submit_estimate. " +
              "Make exactly one tool call now.",
          },
        ],
      });
      continue;
    }

    // Persist the model's own turn (carrying its functionCall) before the
    // matching functionResponse + image parts.
    contents.push(
      modelContent ?? {
        role: "model",
        parts: calls.map((c) => ({
          functionCall: { name: c.name, args: c.args },
        })),
      },
    );

    // Process the first call per turn (mirrors ADK single-call cadence).
    const call = calls[0];
    yield { type: "tool_call", toolName: call.name, toolArgs: call.args };

    if (call.name === "submit_estimate") {
      const positionMm = Number(
        (call.args as { position_mm?: unknown }).position_mm,
      );
      const reasoning = String(
        (call.args as { reasoning?: unknown }).reasoning ?? "",
      );
      if (!Number.isFinite(positionMm)) {
        const err = "submit_estimate: position_mm not finite";
        yield { type: "error", error: err };
        throw new Error(err);
      }
      yield {
        type: "tool_result",
        toolName: "submit_estimate",
        toolResult: { status: "ok", position_mm: positionMm },
      };
      yield { type: "done" };
      return { positionMm, reasoning, iterations: iteration };
    }

    if (call.name === "fetch_atlas") {
      const raw =
        ((call.args as { positions_mm?: unknown }).positions_mm as unknown[]) ??
        [];
      const capped = (Array.isArray(raw) ? raw : [raw]).slice(0, 8);
      const positions = clampAndDedupePositions(capped, p.posLo, p.posHi);

      if (positions.length === 0) {
        const toolResult = { status: "error", error: "EMPTY_RESULT" };
        yield {
          type: "tool_result",
          toolName: "fetch_atlas",
          toolResult,
          fetchedAtlasUrls: [],
        };
        contents.push({
          role: "user",
          parts: [
            { functionResponse: { name: "fetch_atlas", response: toolResult } },
          ],
        });
        continue;
      }

      const fetched: Array<{ pos: number; url: string; part: Part }> = [];
      for (const pos of positions) {
        if (p.signal?.aborted) {
          yield { type: "error", error: "aborted" };
          throw new Error("aborted");
        }
        try {
          const layers = await getCoronalSliceLayers(pos);
          const part = await fetchAsInlinePart(layers.sectionUrl);
          fetched.push({ pos: layers.apMm, url: layers.sectionUrl, part });
        } catch (e) {
          const msg = e instanceof Error ? e.message : String(e);
          yield { type: "error", error: `fetch_atlas(${pos.toFixed(2)}): ${msg}` };
        }
      }

      const descriptions = fetched.map((f) => `${f.pos.toFixed(2)} mm`);
      const plural = fetched.length === 1 ? "" : "s";
      const toolResult: Record<string, unknown> = {
        status: "ok",
        positions_mm: fetched.map((f) => Number(f.pos.toFixed(2))),
        description:
          `Fetched ${fetched.length} atlas section${plural}: ` +
          descriptions.join(", "),
      };

      yield {
        type: "tool_result",
        toolName: "fetch_atlas",
        toolResult,
        fetchedAtlasUrls: fetched.map((f) => f.url),
      };

      // functionResponse + caption + each image, in one user turn — mirrors
      // the Python plugin's MULTIMODAL_PARTS_RESULT_KEY surfacing.
      const responseParts: Part[] = [
        { functionResponse: { name: "fetch_atlas", response: toolResult } },
        {
          text:
            `Fetched ${fetched.length} atlas section${plural} at: ` +
            descriptions.join(", ") +
            ".",
        },
      ];
      for (const f of fetched) {
        responseParts.push({ text: `Atlas at ${f.pos.toFixed(2)} mm:` });
        responseParts.push(f.part);
      }
      contents.push({ role: "user", parts: responseParts });
      continue;
    }

    // Unknown tool: report and continue.
    const toolResult = { status: "error", error: `UNKNOWN_TOOL:${call.name}` };
    yield { type: "tool_result", toolName: call.name, toolResult };
    contents.push({
      role: "user",
      parts: [
        { functionResponse: { name: call.name, response: toolResult } },
      ],
    });
  }

  // Iteration budget exhausted without submit_estimate. Fall back to the
  // atlas midpoint (mirrors runner.py:515-527) so the demo always returns
  // a valid AP rather than crashing the pipeline.
  const midpoint = (p.posLo + p.posHi) / 2;
  yield {
    type: "error",
    error:
      `Agent exceeded maxIterations=${maxIterations} without calling ` +
      `submit_estimate; falling back to ${midpoint.toFixed(2)} mm midpoint.`,
  };
  yield { type: "done" };
  return {
    positionMm: midpoint,
    reasoning:
      "Model did not submit within iteration budget; " +
      "fell back to atlas midpoint.",
    iterations: maxIterations,
  };
}
