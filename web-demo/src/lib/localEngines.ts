/** Browser-side probe for OpenAI-compatible local LLM servers.
 *
 * Port of `tauri-gui/src-tauri/src/local_engines.rs`, minus the IPv4→IPv6
 * fallback (browsers can't bind to a specific stack — they just hit
 * 127.0.0.1). The Tauri version uses reqwest; here we use `fetch` with
 * `AbortSignal.timeout`. Concurrently probes a small registry of well-known
 * localhost ports for `/v1/models`. Engines that respond are reported with
 * their model list; everything else comes back as `reachable: false` after
 * a short timeout.
 *
 * Browser-specific caveats handled here:
 *   - Chrome 142+ Local Network Access (LNA) prompts fire on the first
 *     127.0.0.1 request after page load. A blocked probe surfaces as a
 *     TypeError from fetch — we classify it as "not running" so the user
 *     can grant LNA via the prompt and refresh.
 *   - CORS: most local engines allow cross-origin from arbitrary pages
 *     (Ollama, LM Studio, llama-server, vLLM with `--cors-allowed-origins '*'`,
 *     Jan). If a server blocks the page, the probe gets a TypeError, same
 *     "not running" classification. The custom-endpoint form is the user's
 *     escape hatch — they can point at a CORS-friendly proxy. */

/** One model returned by an engine's `/v1/models` listing. */
export interface LocalModel {
  id: string;
  engine: string;
  endpoint: string;
}

/** One row in the Local Models modal. `reachable: false` rows are still
 *  rendered with a "Not running" pill so the user knows we looked. */
export interface LocalEngineStatus {
  name: string;
  port: number;
  endpoint: string; // e.g. "http://127.0.0.1:11434/v1"
  reachable: boolean;
  models: LocalModel[];
  error: string | null;
}

/** User-saved extra endpoint (non-default port or remote server). */
export interface CustomEndpoint {
  label: string;
  url: string;
  apiKey?: string;
}

/** Engines probed by default. Order is preserved for UI rendering — keep
 *  in sync with the Rust ENGINES constant.
 *
 *  LiteRT-LM defaults to port 8765 (litert_lm::DEFAULT_PORT) so the
 *  optional sidecar shows up here without colliding with llama-server. */
const ENGINES: ReadonlyArray<readonly [string, number]> = [
  ["Ollama", 11434],
  ["LM Studio", 1234],
  ["llama-server", 8080],
  ["vLLM", 8000],
  ["Jan", 1337],
  ["LiteRT-LM", 8765],
];

const PROBE_TIMEOUT_MS = 400;

interface ModelsListResponse {
  data?: Array<{ id?: unknown }>;
}

/** Parse an OpenAI-style /v1/models body into ids. Returns [] if the shape
 *  doesn't match. Mirrors the Rust filter_map walk on body["data"][].id. */
function parseModelIds(body: unknown): string[] {
  const data = (body as ModelsListResponse | null)?.data;
  if (!Array.isArray(data)) return [];
  const out: string[] = [];
  for (const item of data) {
    if (item && typeof item.id === "string") out.push(item.id);
  }
  return out;
}

/** Probe a single endpoint URL (no trailing slash). Returns a partial result
 *  the caller wraps into a LocalEngineStatus. Never throws. */
async function probeEndpoint(
  endpoint: string,
): Promise<
  | { ok: true; ids: string[] }
  | { ok: false; error: string }
> {
  const url = `${endpoint}/models`;
  let response: Response;
  try {
    response = await fetch(url, { signal: AbortSignal.timeout(PROBE_TIMEOUT_MS) });
  } catch (err) {
    // TypeError = network error / CORS / LNA-block. TimeoutError /
    // AbortError = the 400ms deadline tripped. All look the same to the
    // user: the engine isn't responding. The custom-endpoint form is the
    // escape hatch for legitimate-but-CORS-blocked targets.
    const name = err instanceof Error ? err.name : "";
    if (name === "TimeoutError" || name === "AbortError") {
      return { ok: false, error: "not running" };
    }
    return { ok: false, error: "not running" };
  }

  if (!response.ok) {
    return { ok: false, error: `HTTP ${response.status}` };
  }

  let body: unknown;
  try {
    body = await response.json();
  } catch {
    return { ok: false, error: "invalid response" };
  }

  return { ok: true, ids: parseModelIds(body) };
}

/** Probe one registry engine on its canonical port. Always returns a status
 *  row — unreachable engines come back with `reachable: false`. */
async function probeEngine(name: string, port: number): Promise<LocalEngineStatus> {
  const endpoint = `http://127.0.0.1:${port}/v1`;
  const result = await probeEndpoint(endpoint);
  if (!result.ok) {
    return {
      name,
      port,
      endpoint,
      reachable: false,
      models: [],
      error: result.error,
    };
  }
  const models: LocalModel[] = result.ids.map((id) => ({
    id,
    engine: name,
    endpoint,
  }));
  return {
    name,
    port,
    endpoint,
    reachable: true,
    models,
    error: null,
  };
}

/** Probe every engine in the registry concurrently. Always returns one
 *  entry per engine, in registry order. */
export async function probeAllEngines(): Promise<LocalEngineStatus[]> {
  return Promise.all(ENGINES.map(([name, port]) => probeEngine(name, port)));
}

/** Probe a single user-supplied custom endpoint (e.g. a remote server or a
 *  non-standard port). Treats whatever the user typed as the canonical
 *  `endpoint` (sans trailing slash) and labels the engine after the
 *  supplied label. `port` is 0 — custom endpoints don't have a registry
 *  port and the UI doesn't surface this field. */
export async function probeCustomEndpoint(
  label: string,
  baseUrl: string,
): Promise<LocalEngineStatus> {
  const endpoint = baseUrl.trim().replace(/\/+$/, "");
  const result = await probeEndpoint(endpoint);
  if (!result.ok) {
    return {
      name: label,
      port: 0,
      endpoint,
      reachable: false,
      models: [],
      error: result.error,
    };
  }
  const models: LocalModel[] = result.ids.map((id) => ({
    id,
    engine: label,
    endpoint,
  }));
  return {
    name: label,
    port: 0,
    endpoint,
    reachable: true,
    models,
    error: null,
  };
}
