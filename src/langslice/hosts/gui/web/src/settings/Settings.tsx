// The Settings tab: Connection (the ChatGPT sign-in, the API keys), Agent model and
// Registration runs in. The state comes from GET settings; each change is a POST whose
// answer is the new state (hosts/gui/settings.py). A saved key is never sent back to the
// page. Refreshed whenever the tab is opened, and every second while a sign-in runs; a
// key being typed survives every refresh.
import { useEffect, useRef, useState } from 'react';
import { Prose, Radio, Select } from '../controls';
import { go } from '../nav';
import './settings.css';

type Connection = { id: string; label: string; kind: 'signin' | 'key' | string; state: string; present: boolean };
type Agent = { model: string | null; models: string[]; default: string | null; reasoning: string | null };
export type SettingsState = {
  connections: Connection[];
  connection: string | null;
  signin: { running: boolean; url?: string | null; error?: string | null };
  credentials_error?: string | null;
  agent: Record<string, Agent>;
  reasoning: string[];
  runs_in: { enabled: boolean; value: string | null };
};
type Said = { text: string; bad?: boolean };

const RUNS = [
  ['app', 'This app', 'Your agent app (Codex, Claude Code, Antigravity) runs registration itself.'],
  ['langslice', 'LangSlice, using the connection above', 'LangSlice’s own agent runs registration on the chosen connection.'],
] as const;

async function load(): Promise<SettingsState> {
  const r = await fetch('settings', { cache: 'no-store' });
  if (!r.ok) throw new Error('unavailable');
  return r.json();
}

// The first open of the window with no connection chosen lands on Settings, unless the
// address names a tab.
export function landOnSettingsIfUnset() {
  if (new URLSearchParams(location.hash.slice(1)).get('tab')) return;
  load().then(s => { if (!s.connection) go({ tab: 'settings' }); }).catch(() => {});
}

export function Settings({ shown }: { shown: boolean }) {
  const [state, setState] = useState<SettingsState | null>(null);
  const [failed, setFailed] = useState(false);
  const [said, setSaid] = useState<Record<string, Said>>({});
  const [keys, setKeys] = useState<Record<string, string>>({});
  const polling = useRef(false);

  const say = (slot: string, text: string, bad?: boolean) => setSaid(s => ({ ...s, [slot]: { text, bad } }));

  // POST one change; the answer is the new state. `slot` is the line that says what it did.
  async function post(resource: string, body: object, slot: string) {
    say(slot, '');
    try {
      const r = await fetch(resource, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), cache: 'no-store',
      });
      const answer = await r.json().catch(() => ({ error: 'LangSlice did not answer.' }));
      if (!r.ok || answer.error) { say(slot, answer.error || 'That did not work.', true); return false; }
      setState(answer);
      return true;
    } catch { say(slot, 'LangSlice is not reachable.', true); return false; }
  }

  async function startPolling() {
    if (polling.current) return;
    polling.current = true;
    try {
      while (true) {
        await new Promise(done => setTimeout(done, 1000));
        const next = await load().catch(() => null);
        if (next) setState(next);
        if (!next || !next.signin.running) break;
      }
    } finally { polling.current = false; }
  }

  // The first load, then a refresh each time the tab is opened.
  const first = useRef(true);
  useEffect(() => {
    if (!shown && !first.current) return;
    const initial = first.current;
    first.current = false;
    load().then(next => {
      setState(next); setFailed(false);
      if (initial && next.signin.running) startPolling();
    }).catch(() => { if (initial) setFailed(true); });
  }, [shown]); // eslint-disable-line react-hooks/exhaustive-deps

  if (failed) return <div className="page settings"><p className="lead">Settings could not be loaded.</p></div>;
  if (!state) return <div className="page settings" aria-busy="true" />;

  const chosen = state.connection;
  const agent = chosen ? state.agent[chosen] : null;
  const runs = state.runs_in;
  const models = agent ? (agent.models.includes(agent.model ?? '') || !agent.model ? agent.models : [agent.model, ...agent.models]) : [];

  return (
    <div className="page settings">
      <header className="page-head">
        <h1 className="page-title">Settings</h1>
        <p className="lead">How LangSlice reaches a model, and which one it uses. Only you set these, in this window; no agent tool reads or changes them.</p>
      </header>

      <section className="part" aria-labelledby="set-connection">
        <div className="part-head">
          <h2 id="set-connection" className="part-title">Connection</h2>
          <p className="help">Choose the one LangSlice uses.</p>
        </div>
        <div className="choices" role="radiogroup" aria-labelledby="set-connection">
          {state.connections.map(row => {
            const on = chosen === row.id;
            const s = state.signin;
            const own = said[row.id];
            const message = row.kind === 'signin' && s.error ? { text: s.error, bad: true } : own?.text ? own : null;
            return (
              <div
                key={row.id}
                className={'choice conn' + (on ? ' chosen' : '')}
                onClick={e => {
                  if (!(e.target as Element).closest('label, button, input, a, form') && !on) post('settings', { connection: row.id }, 'connection');
                }}
              >
                <label className="choice-pick">
                  <Radio name="connection" value={row.id} checked={on} onPick={v => post('settings', { connection: v }, 'connection')} />
                  <span className="choice-name">{row.label}</span>
                </label>
                <span className={'tag' + (row.present ? ' present' : '')}>{row.state}</span>
                <div className="conn-action">
                  {row.kind === 'signin' ? (
                    <>
                      <button type="button" className="button" disabled={s.running}
                        onClick={() => post('settings/signin', {}, row.id).then(ok => { if (ok) startPolling(); })}>
                        {s.running ? 'Waiting for the browser…' : row.present ? 'Sign in again' : 'Sign in with ChatGPT'}
                      </button>
                      {s.running && s.url && <a className="link" href={s.url} target="_blank" rel="noopener noreferrer">Open the sign-in page</a>}
                    </>
                  ) : (
                    <form className="key" onSubmit={async e => {
                      e.preventDefault();
                      const key = keys[row.id] || '';
                      if (!key.trim()) { say(row.id, 'Paste a key first.', true); return; }
                      setKeys(k => ({ ...k, [row.id]: '' }));
                      if (await post('settings/key', { provider: row.id, key }, row.id)) say(row.id, row.label + ' saved.');
                    }}>
                      <input className="input" type="password" autoComplete="off" spellCheck={false}
                        placeholder={row.present ? 'Paste a new key' : 'Paste a key'} aria-label={row.label}
                        value={keys[row.id] || ''} onChange={e => { const v = e.target.value; setKeys(k => ({ ...k, [row.id]: v })); }} />
                      <button type="submit" className="button">Save</button>
                    </form>
                  )}
                </div>
                {message && <p className={'said' + (message.bad ? ' bad' : '')} role={message.bad ? 'alert' : 'status'}>{message.text}</p>}
              </div>
            );
          })}
        </div>
        <Said said={said.connection} />
        {state.credentials_error && <p className="said bad">{state.credentials_error}</p>}
        <Prose className="help" parts={['Keys and the sign-in are kept on this computer (', ['~/.langslice/'], '). A saved key is never shown again; save a new one to replace it.']} />
      </section>

      {chosen && agent && (
        <section className="part" aria-labelledby="set-agent">
          <div className="part-head">
            <h2 id="set-agent" className="part-title">Agent model</h2>
            <span className="part-sub">for the {state.connections.find(c => c.id === chosen)?.label}</span>
          </div>
          <div className="pairs">
            <label className="label" htmlFor="set-model">Model</label>
            <Select id="set-model" value={agent.model || ''} onChange={e => { if (e.target.value) post('settings', { model: e.target.value }, 'agent'); }}>
              {!agent.model && <option value="">Choose a model</option>}
              {models.map(name => <option key={name} value={name}>{name === agent.default ? name + ' (LangSlice default)' : name}</option>)}
            </Select>
            <label className="label" htmlFor="set-reasoning">Reasoning</label>
            <Select id="set-reasoning" value={agent.reasoning || ''} onChange={e => post('settings', { reasoning: e.target.value || null }, 'agent')}>
              <option value="">Provider default</option>
              {state.reasoning.map(effort => <option key={effort} value={effort}>{effort}</option>)}
            </Select>
          </div>
          {!agent.model && <p className="said">LangSlice has no default model for this connection; choose one.</p>}
          <Said said={said.agent} />
          <Prose className="help" parts={['LangSlice’s own agent (', ['langslice linear run'], ') uses this model when you start it without ', ['--model'], '. Reasoning applies to models that take an effort setting.']} />
        </section>
      )}

      <section className="part" aria-labelledby="set-runs">
        <div className="part-head">
          <h2 id="set-runs" className="part-title">Registration runs in</h2>
        </div>
        <div className="choices" role="radiogroup" aria-labelledby="set-runs">
          {RUNS.map(([value, text, hint]) => {
            const on = runs.enabled && runs.value === value;
            return (
              <label key={value} className={'choice' + (on ? ' chosen' : '') + (runs.enabled ? '' : ' disabled')}>
                <span className="choice-pick">
                  <Radio name="runs-in" value={value} checked={on} disabled={!runs.enabled} onPick={v => post('settings', { runs_in: v }, 'runs')} />
                  <span className="choice-words">
                    <span className="choice-name">{text}</span>
                    <span className="help">{hint}</span>
                  </span>
                </span>
              </label>
            );
          })}
        </div>
        <Said said={said.runs} />
        <p className="help">
          {runs.enabled
            ? 'Saved for later: this version of LangSlice does not act on this choice yet.'
            : chosen ? 'Available once the chosen connection is signed in or has a key.' : 'Available once a connection is chosen above.'}
        </p>
      </section>
    </div>
  );
}

function Said({ said }: { said?: Said }) {
  if (!said?.text) return null;
  return <p className={'said' + (said.bad ? ' bad' : '')} role={said.bad ? 'alert' : 'status'}>{said.text}</p>;
}
