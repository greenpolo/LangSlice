import { useEffect, useMemo, useRef, useState } from "react";
import { useAppStore } from "../stores/appStore";

/** Chat — BYO-engine chat drawer.
 *
 * Slides in from the right edge as a 440px panel. Reuses the localEngines
 * the Local Models modal already probed, so the user can pick any reachable
 * model (Ollama, LM Studio, llama-server, vLLM, Jan, or a custom endpoint)
 * and talk to it directly. No engine bundling — the user brings their own.
 *
 * Visual language: serial-console / laboratory-instrument. LINK status dot,
 * monospace prompts, streaming cursor, TX/RX telemetry strip. Subtle scanline
 * overlay only inside the message area to differentiate it from the rest of
 * the app without polluting the registration workspace behind it. */
export function ChatDrawer() {
  const open = useAppStore((s) => s.chatOpen);
  const close = useAppStore((s) => s.closeChat);
  const messages = useAppStore((s) => s.chatMessages);
  const endpoint = useAppStore((s) => s.chatEndpoint);
  const modelId = useAppStore((s) => s.chatModelId);
  const streaming = useAppStore((s) => s.chatStreaming);
  const error = useAppStore((s) => s.chatError);
  const telemetry = useAppStore((s) => s.chatTelemetry);
  const setModel = useAppStore((s) => s.setChatModel);
  const send = useAppStore((s) => s.sendChatMessage);
  const cancel = useAppStore((s) => s.cancelChatStream);
  const clear = useAppStore((s) => s.clearChat);
  const localEngines = useAppStore((s) => s.localEngines);
  const probing = useAppStore((s) => s.localEnginesProbing);
  const probe = useAppStore((s) => s.probeLocalEngines);
  const openLocalModels = useAppStore((s) => s.openLocalModels);
  const selectedSliceImage = useAppStore((s) => s.selectedSliceImage);
  const selectedSliceIndex = useAppStore((s) => s.selectedSliceIndex);
  const brains = useAppStore((s) => s.brains);
  const selectedBrainId = useAppStore((s) => s.selectedBrainId);

  const [draft, setDraft] = useState("");
  const [attachSlice, setAttachSlice] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // All reachable model options flattened into a single picker. Cloud
  // models are intentionally excluded here — this drawer is for local
  // inspection of self-hosted weights. Keys are stable across renders
  // (engine + model id) for React reconciliation in the <select>.
  const modelOptions = useMemo(() => {
    return localEngines
      .filter((e) => e.reachable)
      .flatMap((e) =>
        e.models.map((m) => ({
          key: `${e.endpoint}::${m.id}`,
          endpoint: e.endpoint,
          modelId: m.id,
          engineName: e.name,
          label: `${m.id} · ${e.name}`,
        })),
      );
  }, [localEngines]);

  const selectedKey = endpoint && modelId ? `${endpoint}::${modelId}` : "";

  // Auto-pick the first available model on open if nothing is selected yet,
  // so the user doesn't land on an empty picker when they have engines up.
  useEffect(() => {
    if (open && !selectedKey && modelOptions.length > 0) {
      const first = modelOptions[0];
      setModel(first.endpoint, first.modelId, first.engineName);
    }
  }, [open, selectedKey, modelOptions, setModel]);

  // Scroll the message log to the bottom on every new token / message.
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [messages, streaming]);

  // Focus the input when the drawer opens or a stream finishes.
  useEffect(() => {
    if (open && !streaming) {
      textareaRef.current?.focus();
    }
  }, [open, streaming]);

  if (!open) return null;

  const linkState: "live" | "standby" | "down" = !endpoint
    ? "standby"
    : streaming
      ? "live"
      : "live"; // when an engine is patched in, treat as live.
  const linkColor =
    linkState === "live" ? "var(--accent)" : linkState === "standby" ? "var(--warning)" : "var(--error)";

  const onSelectModel = (key: string) => {
    if (!key) {
      setModel("", "", "");
      return;
    }
    const opt = modelOptions.find((o) => o.key === key);
    if (opt) setModel(opt.endpoint, opt.modelId, opt.engineName);
  };

  const onSubmit = async () => {
    if (streaming) {
      cancel();
      return;
    }
    if (!draft.trim() || !endpoint || !modelId) return;
    const text = draft;
    setDraft("");
    setAttachSlice(false);
    await send(text, attachSlice);
  };

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Enter to send, Shift+Enter for a newline — matches the conventional
    // chat-app contract while staying out of the way of multi-line drafts.
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit();
    }
  };

  // Slice attachment availability — only meaningful when the user is in a
  // brain-detail view with a slice selected and the image actually loaded.
  const brain = brains.find((b) => b.id === selectedBrainId);
  const slice = brain && selectedSliceIndex !== null ? brain.slices[selectedSliceIndex] : null;
  const canAttachSlice = !!(slice && selectedSliceImage);

  return (
    <aside className="chat-drawer" role="dialog" aria-label="Chat — local model">
      {/* Header: LINK status + engine pill + close ─────────────── */}
      <header className="chat-header">
        <div className="chat-title-row">
          <span className="chat-title">CHAT</span>
          <span className="chat-subtitle">// local model link</span>
        </div>
        <button
          type="button"
          className="chat-close"
          onClick={close}
          title="Close drawer (stream is cancelled)"
          aria-label="Close chat drawer"
        >
          ×
        </button>
      </header>

      {/* Connection bar ───────────────────────────────────────── */}
      <div className="chat-link">
        <span
          className={`chat-link-dot ${streaming ? "chat-link-dot-pulse" : ""}`}
          style={{ background: linkColor, boxShadow: `0 0 8px ${linkColor}` }}
          aria-hidden
        />
        <span className="chat-link-label">
          {endpoint ? "LINK" : "STANDBY"}
        </span>
        <span className="chat-link-sep">·</span>
        <select
          className="chat-link-select"
          value={selectedKey}
          onChange={(e) => onSelectModel(e.target.value)}
          disabled={streaming}
        >
          {modelOptions.length === 0 ? (
            <option value="">no engines reachable</option>
          ) : (
            <>
              <option value="">— select model —</option>
              {modelOptions.map((o) => (
                <option key={o.key} value={o.key}>
                  {o.label}
                </option>
              ))}
            </>
          )}
        </select>
        <button
          type="button"
          className="chat-link-action"
          onClick={() => probe()}
          disabled={probing}
          title="Re-probe local engines"
        >
          {probing ? "..." : "↻"}
        </button>
      </div>

      {/* Message log or empty state ───────────────────────────── */}
      <div className="chat-log" ref={logRef}>
        {messages.length === 0 ? (
          <EmptyState
            modelOptions={modelOptions.length}
            onBrowse={() => {
              openLocalModels();
            }}
          />
        ) : (
          messages.map((m, i) => (
            <MessageBlock
              key={m.id}
              role={m.role}
              content={m.content}
              images={m.images}
              streaming={streaming && i === messages.length - 1 && m.role === "assistant"}
            />
          ))
        )}
        {error && (
          <div className="chat-error" role="alert">
            <span className="chat-error-tag">ERR</span>
            <span className="chat-error-text">{error}</span>
          </div>
        )}
      </div>

      {/* Telemetry strip — only after the first exchange ──────── */}
      {(telemetry.rxTokens || telemetry.ttftMs) && (
        <div className="chat-telemetry" aria-live="polite">
          <span className="chat-tel-cell">
            <span className="chat-tel-key">rx</span>
            <span className="chat-tel-val">{telemetry.rxTokens ?? 0}</span>
          </span>
          <span className="chat-tel-cell">
            <span className="chat-tel-key">ttft</span>
            <span className="chat-tel-val">
              {telemetry.ttftMs !== undefined ? `${telemetry.ttftMs}ms` : "—"}
            </span>
          </span>
          <span className="chat-tel-cell">
            <span className="chat-tel-key">rate</span>
            <span className="chat-tel-val">
              {telemetry.tokensPerSec !== undefined ? `${telemetry.tokensPerSec} t/s` : "—"}
            </span>
          </span>
          <span className="chat-tel-cell">
            <span className="chat-tel-key">elapsed</span>
            <span className="chat-tel-val">
              {telemetry.elapsedMs !== undefined ? `${(telemetry.elapsedMs / 1000).toFixed(1)}s` : "—"}
            </span>
          </span>
        </div>
      )}

      {/* Composer ────────────────────────────────────────────── */}
      <div className="chat-composer">
        {attachSlice && canAttachSlice && selectedSliceImage && (
          <div className="chat-attach">
            <img
              className="chat-attach-thumb"
              src={`data:image/png;base64,${selectedSliceImage}`}
              alt="Slice attachment"
            />
            <span className="chat-attach-meta">
              {slice?.name ?? "slice"}
              {slice?.apMm !== undefined ? ` · ${slice.apMm.toFixed(2)}mm` : ""}
            </span>
            <button
              type="button"
              className="chat-attach-remove"
              onClick={() => setAttachSlice(false)}
              aria-label="Remove attachment"
            >
              ×
            </button>
          </div>
        )}
        <div className="chat-composer-row">
          <span className="chat-prompt-caret" aria-hidden>
            ▸
          </span>
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={onKeyDown}
            placeholder={
              !endpoint
                ? "patch into a local engine to begin transmission..."
                : "transmit packet // enter to send, shift+enter for newline"
            }
            rows={2}
            disabled={!endpoint || streaming}
          />
        </div>
        <div className="chat-composer-actions">
          {canAttachSlice && (
            <button
              type="button"
              className={`chat-attach-btn ${attachSlice ? "active" : ""}`}
              onClick={() => setAttachSlice((v) => !v)}
              title="Attach the currently selected slice as a vision input"
              disabled={streaming}
            >
              <span aria-hidden>◧</span> attach slice
            </button>
          )}
          {messages.length > 0 && (
            <button
              type="button"
              className="chat-clear-btn"
              onClick={clear}
              disabled={streaming}
              title="Clear conversation"
            >
              wipe log
            </button>
          )}
          <div className="chat-spacer" />
          <button
            type="button"
            className={`chat-send-btn ${streaming ? "streaming" : ""}`}
            onClick={onSubmit}
            disabled={!streaming && (!endpoint || !draft.trim())}
          >
            {streaming ? "cancel ▣" : "send ▸"}
          </button>
        </div>
      </div>
    </aside>
  );
}

/** Empty-state card. Renders an ASCII-frame standby card when no engines are
 * reachable, or a friendlier "ready to transmit" message when at least one
 * model is patched in. */
function EmptyState({
  modelOptions,
  onBrowse,
}: {
  modelOptions: number;
  onBrowse: () => void;
}) {
  const noEngines = modelOptions === 0;
  return (
    <div className="chat-empty">
      <pre className="chat-empty-frame" aria-hidden>
        {`┌─[ ${noEngines ? "STANDBY" : "READY  "} ]──────────────────┐`}
        {"\n│                                       │\n"}
        {noEngines
          ? "│  no local engines reachable.          │\n│  spin up ollama, lm studio, or        │\n│  llama-server and re-probe.           │\n"
          : "│  link established.                    │\n│  awaiting first transmission.         │\n│                                       │\n"}
        {"│                                       │\n"}
        {"└───────────────────────────────────────┘"}
      </pre>
      {noEngines && (
        <button type="button" className="chat-empty-cta" onClick={onBrowse}>
          browse local models →
        </button>
      )}
    </div>
  );
}

/** Single message block. User turns are a single line with a `▸` caret;
 * assistant turns get a vertical hairline rail on the left so multi-line
 * answers scan as a single block. The streaming variant appends a blinking
 * `▍` cursor at the end of the content. */
function MessageBlock({
  role,
  content,
  images,
  streaming,
}: {
  role: "user" | "assistant";
  content: string;
  images?: string[];
  streaming: boolean;
}) {
  if (role === "user") {
    return (
      <div className="chat-msg chat-msg-user">
        <span className="chat-msg-caret" aria-hidden>
          ▸
        </span>
        <div className="chat-msg-body">
          {content}
          {images && images.length > 0 && (
            <div className="chat-msg-images">
              {images.map((src, i) => (
                <img key={i} className="chat-msg-img" src={src} alt={`attachment ${i + 1}`} />
              ))}
            </div>
          )}
        </div>
      </div>
    );
  }
  return (
    <div className="chat-msg chat-msg-assistant">
      <div className="chat-msg-rail" aria-hidden />
      <div className="chat-msg-body">
        {content || (streaming ? <span className="chat-msg-waiting">awaiting tokens...</span> : null)}
        {streaming && <span className="chat-stream-cursor" aria-hidden>▍</span>}
      </div>
    </div>
  );
}
