/** Runtime probe + React hook for the litert-lm-serve sidecar.
 *  Hits GET {baseUrl}/v1/models and classifies the outcome so callers can
 *  drive UI without try/catch. Never throws. */

import { useEffect, useRef, useState } from "react";

export type SidecarStatus =
  | { state: "unknown" }
  | { state: "ok"; models: string[] }
  | { state: "not-running" }
  | { state: "cors-blocked"; hint: string };

export interface SidecarProbeOptions {
  baseUrl?: string;
  timeoutMs?: number;
}

const DEFAULT_BASE_URL = "http://127.0.0.1:8765";
const DEFAULT_TIMEOUT_MS = 500;
const NO_CORS_FOLLOWUP_TIMEOUT_MS = 300;
const CORS_HINT =
  "Start litert-lm with --cors-origin '*' (or restart Chrome with --disable-web-security for testing).";

interface ModelsListResponse {
  object?: string;
  data?: Array<{ id?: unknown }>;
}

/** One-shot probe. Resolves; never rejects. */
export async function probeSidecarOnce(
  opts?: SidecarProbeOptions,
): Promise<SidecarStatus> {
  const baseUrl = opts?.baseUrl ?? DEFAULT_BASE_URL;
  const timeoutMs = opts?.timeoutMs ?? DEFAULT_TIMEOUT_MS;
  const url = `${baseUrl}/v1/models`;

  let response: Response;
  try {
    response = await fetch(url, { signal: AbortSignal.timeout(timeoutMs) });
  } catch (err) {
    return classifyFetchError(err, url);
  }

  if (!response.ok) {
    // Sidecar may be up but exposing a different shape (e.g. /v1beta only).
    // Keep the Setup overlay visible; README covers follow-up.
    return { state: "not-running" };
  }

  let parsed: ModelsListResponse;
  try {
    parsed = (await response.json()) as ModelsListResponse;
  } catch {
    return { state: "not-running" };
  }

  const data = Array.isArray(parsed.data) ? parsed.data : [];
  const models = data
    .map((m) => (typeof m?.id === "string" ? m.id : null))
    .filter((id): id is string => id !== null);
  return { state: "ok", models };
}

/** Distinguish CORS-blocked-but-reachable from network-down by retrying
 *  with mode:"no-cors". An opaque success means the socket opened. */
async function classifyFetchError(
  err: unknown,
  url: string,
): Promise<SidecarStatus> {
  const name = err instanceof Error ? err.name : "";
  if (name === "AbortError" || name === "TimeoutError") {
    return { state: "not-running" };
  }
  if (err instanceof TypeError) {
    try {
      await fetch(url, {
        mode: "no-cors",
        signal: AbortSignal.timeout(NO_CORS_FOLLOWUP_TIMEOUT_MS),
      });
      return { state: "cors-blocked", hint: CORS_HINT };
    } catch {
      return { state: "not-running" };
    }
  }
  return { state: "not-running" };
}

/** React hook. Polls every `intervalMs` while `enabled` is true and the
 *  current state is not "ok". Cleans up timers + in-flight aborts on unmount. */
export function useSidecarStatus(opts?: {
  enabled?: boolean;
  intervalMs?: number;
  baseUrl?: string;
  timeoutMs?: number;
}): SidecarStatus {
  const enabled = opts?.enabled ?? true;
  const intervalMs = opts?.intervalMs ?? 2000;
  const baseUrl = opts?.baseUrl;
  const timeoutMs = opts?.timeoutMs;

  const [status, setStatus] = useState<SidecarStatus>({ state: "unknown" });
  const statusRef = useRef<SidecarStatus>(status);
  statusRef.current = status;

  const abortRef = useRef<AbortController | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!enabled) return;

    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const runProbe = async () => {
      // Cancel any in-flight probe so re-enables don't pile up.
      abortRef.current?.abort();
      const ctrl = new AbortController();
      abortRef.current = ctrl;
      const next = await probeSidecarOnce({ baseUrl, timeoutMs });
      if (cancelled || ctrl.signal.aborted || !mountedRef.current) return;
      setStatus(next);
      if (next.state === "ok" && timer !== null) {
        clearInterval(timer);
        timer = null;
      }
    };

    void runProbe();
    timer = setInterval(() => {
      if (statusRef.current.state === "ok") {
        if (timer !== null) {
          clearInterval(timer);
          timer = null;
        }
        return;
      }
      void runProbe();
    }, intervalMs);

    return () => {
      cancelled = true;
      if (timer !== null) clearInterval(timer);
      abortRef.current?.abort();
      abortRef.current = null;
    };
  }, [enabled, intervalMs, baseUrl, timeoutMs]);

  return status;
}
