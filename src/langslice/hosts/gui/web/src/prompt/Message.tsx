// Your message: the problems that stop it, the warnings, the message word for word as it
// is copied (the request in the window's face, the command block in the monospace face,
// its fences dim), and Copy, which says Copied for two seconds.
import { useEffect, useRef, useState } from 'react';
import type { Made } from './form';

const OPEN = '\n\n```bash\n';

export function Message({ made }: { made: Made | null }) {
  const text = made?.text ?? '';
  const [copied, setCopied] = useState(false);
  const timer = useRef(0);
  const out = useRef<HTMLPreElement>(null);
  useEffect(() => { setCopied(false); clearTimeout(timer.current); }, [text]);
  useEffect(() => () => clearTimeout(timer.current), []);

  async function copy() {
    if (!text) return;
    try { await navigator.clipboard.writeText(text); }
    catch {
      const sel = window.getSelection();
      if (out.current && sel) {
        const range = document.createRange(); range.selectNodeContents(out.current);
        sel.removeAllRanges(); sel.addRange(range);
        document.execCommand('copy'); sel.removeAllRanges();
      }
    }
    setCopied(true);
    clearTimeout(timer.current);
    timer.current = window.setTimeout(() => setCopied(false), 2000);
  }

  const at = text.indexOf(OPEN), close = text.lastIndexOf('\n```');
  const split = text && at >= 0 && close > at;
  return (
    <aside className="message" aria-label="Your message">
      <div className="message-head">
        <h2 className="message-title">Your message</h2>
        <button type="button" className={'button primary copy' + (copied ? ' copied' : '')} disabled={!text} onClick={copy}>
          <span className="swap" data-on={copied || undefined}>
            <span aria-hidden={copied}>Copy</span>
            <span aria-hidden={!copied}><svg className="copied-tick" viewBox="0 0 12 12" aria-hidden="true"><path d="M2.6 6.3 4.9 8.6 9.4 3.6" /></svg>Copied</span>
          </span>
        </button>
      </div>
      {made && made.problems.length > 0 && (
        <ul className="problems" role="alert">{made.problems.map(p => <li key={p}>{p}</li>)}</ul>
      )}
      {made && made.warnings.length > 0 && (
        <ul className="warnings">{made.warnings.map(w => <li key={w}>{w}</li>)}</ul>
      )}
      {text ? (
        <pre ref={out} className="message-text" tabIndex={0}>
          {split ? (
            <>
              <span className="request">{text.slice(0, at)}</span>{'\n\n'}
              <span className="code"><span className="fence">```bash</span>{'\n'}<span className="command">{text.slice(at + OPEN.length, close)}</span>{'\n'}<span className="fence">```</span></span>
              {text.slice(close + '\n```'.length)}
            </>
          ) : text}
        </pre>
      ) : (
        <p className="message-empty">Your message appears here.</p>
      )}
      <p className="help">Paste it into a new chat in your agent app (Claude Code, Codex or Antigravity) with LangSlice installed. The agent sets up the job, asks about anything left open, and starts the registration; follow it in the Agent viewer.</p>
    </aside>
  );
}
