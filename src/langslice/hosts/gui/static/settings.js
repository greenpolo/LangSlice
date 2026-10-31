'use strict';
// The Settings tab: three cards, Connection (the ChatGPT sign-in, the API keys), Agent model
// and Registration runs in. State from GET settings; changes by POST (hosts/gui/settings.py).
// A saved key is never sent back to the page. The tab is built once and updated from each
// answer, so a key being typed survives every refresh.
(() => {
  const box = document.getElementById('tab-settings');
  if (!box) return;
  const mk = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  const tabButton = document.querySelector('[data-tab="settings"]');
  let state = null, ui = null, polling = false;

  // A sentence whose [literal] parts are set in violet: `prose('set-help', 'Run ', ['x'], '.')`.
  function prose(cls, ...parts) {
    const p = mk('p', cls);
    for (const part of parts) p.append(Array.isArray(part) ? mk('code', 'set-literal', part[0]) : part);
    return p;
  }

  // POST one change; the answer is the new state. `slot` is the card's message line.
  async function post(resource, body, slot = ui.connectionMessage) {
    say(slot, '');
    try {
      const r = await fetch(resource, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), cache: 'no-store' });
      const answer = await r.json().catch(() => ({ error: 'LangSlice did not answer.' }));
      if (!r.ok || answer.error) { say(slot, answer.error || 'That did not work.', true); return false; }
      update(answer); return true;
    } catch { say(slot, 'LangSlice is not reachable.', true); return false; }
  }
  function say(slot, text, bad) { slot.textContent = text; slot.className = 'set-message' + (bad ? ' bad' : ''); }

  // A radio drawn as a square; the native input stays, for the keyboard and screen readers.
  function radio(name, value, onPick) {
    const input = mk('input', 'set-native'); input.type = 'radio'; input.name = name; input.value = value;
    input.addEventListener('change', () => { if (input.checked) onPick(value); });
    const mark = mk('span', 'set-radio'); mark.setAttribute('aria-hidden', 'true');
    return { input, mark };
  }
  function card(title) {
    const section = mk('section', 'set-card');
    const head = mk('div', 'set-card-head');
    head.append(mk('h3', 'set-card-title', title));
    section.append(head);
    return { section, head };
  }

  function build() {
    box.replaceChildren();
    const page = mk('div', 'set-page');
    page.append(mk('h2', 'set-title', 'Settings'),
      mk('p', 'set-intro', 'How LangSlice reaches a model, and which one it uses. Only you set these, in this window; no agent tool reads or changes them.'));

    // Connection: one card per way in, the chosen one marked.
    const conn = card('Connection');
    conn.section.append(mk('p', 'set-help', 'Choose the one LangSlice uses.'));
    const list = mk('div', 'set-conns');
    const rows = {};
    for (const row of state.connections) {
      const line = mk('div', 'set-conn');
      const pick = mk('label', 'set-pick');
      const { input, mark } = radio('connection', row.id, v => post('settings', { connection: v }, ui.connectionMessage));
      pick.append(input, mark, mk('span', 'set-name', row.label));
      const word = mk('span', 'set-pill');
      const action = mk('div', 'set-action');
      const message = mk('p', 'set-message');  // what this card's sign-in or key did
      const parts = { line, input, word, message };
      if (row.kind === 'signin') {
        const button = mk('button', 'set-button', 'Sign in with ChatGPT'); button.type = 'button';
        button.addEventListener('click', () => post('settings/signin', {}, message).then(ok => { if (ok) startPolling(); }));
        const link = mk('a', 'set-link', 'Open the sign-in page'); link.target = '_blank'; link.rel = 'noopener noreferrer'; link.hidden = true;
        action.append(button, link); parts.button = button; parts.link = link;
      } else {
        const form = mk('form', 'set-key');
        const field = mk('input', 'set-input'); field.type = 'password'; field.autocomplete = 'off'; field.spellcheck = false;
        field.placeholder = 'Paste a key'; field.setAttribute('aria-label', row.label);
        const save = mk('button', 'set-button', 'Save'); save.type = 'submit';
        form.addEventListener('submit', async e => {
          e.preventDefault();
          if (!field.value.trim()) { say(message, 'Paste a key first.', true); return; }
          const key = field.value; field.value = '';
          if (await post('settings/key', { provider: row.id, key }, message)) say(message, row.label + ' saved.');
        });
        form.append(field, save); action.append(form); parts.field = field;
      }
      // A click on the card's empty space chooses it, as its square does.
      line.addEventListener('click', e => {
        if (!e.target.closest('label, button, input, a, form') && !input.checked) input.click();
      });
      line.append(pick, word, action, message);
      list.append(line); rows[row.id] = parts;
    }
    const credError = mk('p', 'set-message bad');
    const connectionMessage = mk('p', 'set-message');
    conn.section.append(list, connectionMessage, credError,
      prose('set-help', 'Keys and the sign-in are kept on this computer (', ['~/.langslice/'], '). A saved key is never shown again; save a new one to replace it.'));
    page.append(conn.section);

    // The agent model of the chosen connection.
    const agent = card('Agent model');
    const agentFor = mk('span', 'set-card-sub');
    agent.head.append(agentFor);
    const modelSelect = mk('select', 'set-select'); modelSelect.id = 'set-model';
    modelSelect.addEventListener('change', () => { if (modelSelect.value) post('settings', { model: modelSelect.value }, ui.agentMessage); });
    const reasoningSelect = mk('select', 'set-select'); reasoningSelect.id = 'set-reasoning';
    reasoningSelect.append(new Option('Provider default', ''));
    for (const effort of state.reasoning) reasoningSelect.append(new Option(effort, effort));
    reasoningSelect.addEventListener('change', () => post('settings', { reasoning: reasoningSelect.value || null }, ui.agentMessage));
    const fields = mk('div', 'set-fields');
    const modelLabel = mk('label', 'set-label', 'Model'); modelLabel.htmlFor = modelSelect.id;
    const reasoningLabel = mk('label', 'set-label', 'Reasoning'); reasoningLabel.htmlFor = reasoningSelect.id;
    fields.append(modelLabel, modelSelect, reasoningLabel, reasoningSelect);
    const agentChoose = mk('p', 'set-message');
    const agentMessage = mk('p', 'set-message');
    agent.section.append(fields, agentChoose, agentMessage,
      prose('set-help', 'LangSlice’s own agent (', ['langslice linear run'], ') uses this model when you start it without ', ['--model'], '. Reasoning applies to models that take an effort setting.'));
    page.append(agent.section);

    // Where registration runs.
    const runs = card('Registration runs in');
    const choices = mk('div', 'set-choices');
    const runChoices = {};
    for (const [value, text, hint] of [
      ['app', 'This app', 'Your agent app (Codex, Claude Code, Antigravity) runs registration itself.'],
      ['langslice', 'LangSlice, using the connection above', 'LangSlice’s own agent runs registration on the chosen connection.']]) {
      const label = mk('label', 'set-choice');
      const { input, mark } = radio('runs-in', value, v => post('settings', { runs_in: v }, ui.runsMessage));
      const words = mk('span', 'set-choice-words');
      words.append(mk('span', 'set-name', text), mk('span', 'set-help', hint));
      label.append(input, mark, words);
      choices.append(label); runChoices[value] = { input, label };
    }
    const runsNote = mk('p', 'set-help');
    const runsMessage = mk('p', 'set-message');
    runs.section.append(choices, runsMessage, runsNote);
    page.append(runs.section);

    box.append(page);
    ui = { rows, credError, connectionMessage, agent: agent.section, agentFor, modelSelect, reasoningSelect,
           agentChoose, agentMessage, runChoices, runsNote, runsMessage };
  }

  function update(next) {
    state = next;
    if (!ui) build();
    // Connections.
    for (const row of state.connections) {
      const parts = ui.rows[row.id];
      const chosen = state.connection === row.id;
      parts.input.checked = chosen;
      parts.line.classList.toggle('chosen', chosen);
      parts.word.textContent = row.state;
      parts.word.className = 'set-pill' + (row.present ? ' present' : '');
      if (parts.button) {
        const s = state.signin;
        parts.button.disabled = s.running;
        parts.button.textContent = s.running ? 'Waiting for the browser…' : row.present ? 'Sign in again' : 'Sign in with ChatGPT';
        parts.link.hidden = !(s.running && s.url);
        if (s.url) parts.link.href = s.url;
        if (s.error) say(parts.message, s.error, true);
      } else {
        parts.field.placeholder = row.present ? 'Paste a new key' : 'Paste a key';
      }
    }
    ui.credError.textContent = state.credentials_error || '';
    // The chosen connection's agent model: hidden until one is chosen.
    const chosen = state.connection;
    ui.agent.hidden = !chosen;
    if (chosen) {
      const a = state.agent[chosen];
      ui.agentFor.textContent = 'for the ' + state.connections.find(c => c.id === chosen).label;
      const select = ui.modelSelect; select.replaceChildren();
      if (!a.model) select.append(new Option('Choose a model', ''));
      const names = a.models.includes(a.model) || !a.model ? a.models : [a.model, ...a.models];
      for (const name of names) select.append(new Option(name === a.default ? name + ' (LangSlice default)' : name, name));
      select.value = a.model || '';
      ui.reasoningSelect.value = a.reasoning || '';
      ui.agentChoose.textContent = a.model ? '' : 'LangSlice has no default model for this connection; choose one.';
    }
    // Where registration runs: greyed until the chosen connection works.
    const runs = state.runs_in;
    for (const [value, { input, label }] of Object.entries(ui.runChoices)) {
      input.disabled = !runs.enabled;
      input.checked = runs.enabled && runs.value === value;
      label.classList.toggle('disabled', !runs.enabled);
      label.classList.toggle('chosen', input.checked);
    }
    ui.runsNote.textContent = runs.enabled
      ? 'Saved for later: this version of LangSlice does not act on this choice yet.'
      : state.connection ? 'Available once the chosen connection is signed in or has a key.' : 'Available once a connection is chosen above.';
  }

  async function load() {
    const r = await fetch('settings', { cache: 'no-store' });
    if (!r.ok) throw new Error('unavailable');
    return r.json();
  }
  async function startPolling() {
    if (polling) return; polling = true;
    try {
      while (true) {
        await new Promise(done => setTimeout(done, 1000));
        const next = await load().catch(() => null);
        if (next) update(next);
        if (!next || !next.signin.running) break;
      }
    } finally { polling = false; }
  }

  // Refresh whenever the tab is opened (a key set elsewhere, a sign-in in a terminal).
  if (tabButton) tabButton.addEventListener('click', () => load().then(update).catch(() => {}));
  // The tab the address names (app.js's startTab), if any.
  const named = typeof startTab !== 'undefined' ? startTab : new URLSearchParams(location.hash.slice(1)).get('tab');
  load().then(first => {
    update(first);
    if (first.signin.running) startPolling();
    // First open with no connection chosen lands here, unless the address names a tab.
    if (!first.connection && !named && tabButton) tabButton.click();
  }).catch(() => { box.replaceChildren(mk('p', 'set-intro set-failed', 'Settings could not be loaded.')); });
})();
