'use strict';
// The Settings tab: where registration runs, the connections (ChatGPT sign-in,
// API keys) and the agent model. State from GET settings; changes by POST
// (hosts/gui/settings.py). A saved key is never sent back to the page.
(() => {
  const box = document.getElementById('tab-settings');
  const mk = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; };
  let state = null, ui = null, polling = false;

  async function post(resource, body) {
    ui.message.textContent = ''; ui.message.className = 'set-message';
    try {
      const r = await fetch(resource, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body), cache: 'no-store' });
      const answer = await r.json().catch(() => ({ error: 'LangSlice did not answer.' }));
      if (!r.ok || answer.error) { say(answer.error || 'That did not work.', true); return false; }
      update(answer); return true;
    } catch { say('LangSlice is not reachable.', true); return false; }
  }
  function say(text, bad) { ui.message.textContent = text; ui.message.className = 'set-message' + (bad ? ' bad' : ''); }

  function radio(name, value, onPick) {
    const input = mk('input'); input.type = 'radio'; input.name = name; input.value = value;
    input.onchange = () => { if (input.checked) onPick(value); };
    return input;
  }

  function build() {
    box.replaceChildren();
    const page = mk('div', 'set-page');
    page.append(mk('h2', 'set-title', 'Settings'));

    // Where registration runs.
    const runs = mk('section', 'set-block');
    runs.append(mk('h3', '', 'Registration runs in'));
    const runChoices = {};
    for (const [value, text, hint] of [
      ['app', 'this app', 'Your agent app (Codex, Claude Code, Antigravity) runs registration itself.'],
      ['langslice', 'LangSlice, using the connection below', "LangSlice's own agent runs registration on the chosen connection."]]) {
      const label = mk('label', 'set-choice');
      const input = radio('runs-in', value, v => post('settings', { runs_in: v }));
      label.append(input, mk('span', 'set-choice-text', text), mk('span', 'set-hint', hint));
      runs.append(label); runChoices[value] = { input, label };
    }
    const runsNote = mk('p', 'set-note');
    runs.append(runsNote);
    page.append(runs);

    // Connections.
    const conns = mk('section', 'set-block');
    conns.append(mk('h3', '', 'Connections'));
    conns.append(mk('p', 'set-note', 'Choose the one LangSlice uses.'));
    const rows = {};
    for (const row of state.connections) {
      const line = mk('div', 'set-row');
      const label = mk('label', 'set-choice');
      const input = radio('connection', row.id, v => post('settings', { connection: v }));
      label.append(input, mk('span', 'set-choice-text', row.label));
      const word = mk('span', 'set-state');
      const action = mk('span', 'set-action');
      const parts = { line, input, word, action };
      if (row.kind === 'signin') {
        const button = mk('button', 'set-button', 'Sign in with ChatGPT');
        button.onclick = () => post('settings/signin', {}).then(ok => { if (ok) startPolling(); });
        const link = mk('a', 'set-link', 'Open the sign-in page'); link.target = '_blank'; link.rel = 'noopener noreferrer'; link.hidden = true;
        action.append(button, link); parts.button = button; parts.link = link;
      } else {
        const form = mk('form', 'set-key');
        const field = mk('input'); field.type = 'password'; field.autocomplete = 'off'; field.spellcheck = false;
        field.placeholder = 'Paste a key'; field.setAttribute('aria-label', row.label);
        const save = mk('button', 'set-button', 'Save'); save.type = 'submit';
        form.onsubmit = async e => {
          e.preventDefault();
          if (!field.value.trim()) { say('Paste a key first.', true); return; }
          const key = field.value; field.value = '';
          if (await post('settings/key', { provider: row.id, key })) say(row.label + ' saved.');
        };
        form.append(field, save); action.append(form); parts.field = field;
      }
      line.append(label, word, action);
      conns.append(line); rows[row.id] = parts;
    }
    const keysNote = mk('p', 'set-note', 'Keys and the sign-in are kept on this computer (~/.langslice/). A saved key is never shown again; save a new one to replace it.');
    const credError = mk('p', 'set-note bad');
    conns.append(keysNote, credError);
    page.append(conns);

    // The agent model of the chosen connection.
    const agent = mk('section', 'set-block');
    const agentTitle = mk('h3', '', 'Agent model');
    const modelSelect = mk('select'); modelSelect.setAttribute('aria-label', 'Model');
    modelSelect.onchange = () => { if (modelSelect.value) post('settings', { model: modelSelect.value }); };
    const reasoningSelect = mk('select'); reasoningSelect.setAttribute('aria-label', 'Reasoning');
    reasoningSelect.append(new Option('Provider default', ''));
    for (const effort of state.reasoning) reasoningSelect.append(new Option(effort, effort));
    reasoningSelect.onchange = () => post('settings', { reasoning: reasoningSelect.value || null });
    const grid = mk('div', 'set-grid');
    grid.append(mk('span', 'set-field', 'Model'), modelSelect, mk('span', 'set-field', 'Reasoning'), reasoningSelect);
    const agentNote = mk('p', 'set-note');
    agent.append(agentTitle, grid, agentNote);
    page.append(agent);

    const message = mk('p', 'set-message');
    page.append(message);
    box.append(page);
    ui = { runChoices, runsNote, rows, credError, agent, agentTitle, modelSelect, reasoningSelect, agentNote, message };
  }

  function update(next) {
    state = next;
    if (!ui) build();
    // Connections.
    for (const row of state.connections) {
      const parts = ui.rows[row.id];
      parts.input.checked = state.connection === row.id;
      parts.word.textContent = row.state;
      parts.word.className = 'set-state' + (row.present ? ' good' : '');
      parts.line.classList.toggle('chosen', state.connection === row.id);
      if (parts.button) {
        const s = state.signin;
        parts.button.disabled = s.running;
        parts.button.textContent = s.running ? 'Waiting for the browser…' : row.present ? 'Sign in again' : 'Sign in with ChatGPT';
        parts.link.hidden = !(s.running && s.url);
        if (s.url) parts.link.href = s.url;
        if (s.error) say(s.error, true);
      } else {
        parts.field.placeholder = row.present ? 'Paste a new key' : 'Paste a key';
      }
    }
    ui.credError.textContent = state.credentials_error || '';
    // Where registration runs.
    const runs = state.runs_in;
    for (const [value, { input, label }] of Object.entries(ui.runChoices)) {
      input.disabled = !runs.enabled; label.classList.toggle('disabled', !runs.enabled);
      input.checked = runs.enabled && runs.value === value;
    }
    ui.runsNote.textContent = runs.enabled
      ? 'Saved for later: this version of LangSlice does not act on this choice yet.'
      : state.connection ? 'Available once the chosen connection is signed in or has a key.' : 'Available once a connection is chosen below.';
    // The chosen connection's agent model.
    const chosen = state.connection;
    ui.agent.hidden = !chosen;
    if (chosen) {
      const a = state.agent[chosen];
      const label = state.connections.find(c => c.id === chosen).label;
      ui.agentTitle.textContent = 'Agent model · ' + label;
      const select = ui.modelSelect; select.replaceChildren();
      if (!a.model) select.append(new Option('Choose a model', ''));
      const names = a.models.includes(a.model) || !a.model ? a.models : [a.model, ...a.models];
      for (const name of names) select.append(new Option(name === a.default ? name + ' (LangSlice default)' : name, name));
      select.value = a.model || '';
      ui.reasoningSelect.value = a.reasoning || '';
      ui.agentNote.textContent = (a.model ? '' : 'LangSlice has no default model for this connection; choose one. ')
        + "LangSlice's own agent (langslice linear run) uses this model when you start it without --model. Reasoning applies to models that take an effort setting.";
    }
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
  document.querySelector('[data-tab="settings"]').addEventListener('click', () => load().then(update).catch(() => {}));
  load().then(first => {
    update(first);
    if (first.signin.running) startPolling();
    // First open with no connection chosen lands here.
    if (!first.connection) document.querySelector('[data-tab="settings"]').click();
  }).catch(() => { box.replaceChildren(mk('p', 'set-note', 'Settings could not be loaded.')); });
})();
