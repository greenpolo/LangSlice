'use strict';
// The Export menu: the shown job's registration as an ABBA project, a QuickNII file or a
// VisuAlign file, written by the server under the job's exports/ folder
// (hosts/gui/export.py). The page starts an export, polls its state while it runs and then
// says where the file is, with Open folder and Copy path.
const Exporter = (() => {
  // An ABBA warp is landmark pairs: many follow LangSlice's warp closely, few are easy to
  // edit by hand in BigWarp (the accuracies measured on the 331CTX brain).
  const EXACT = 1089, EDITABLE = 100, MIN_LANDMARKS = 16;
  const FORMATS = [
    { kind: 'abba', landmarks: EXACT, name: 'ABBA project, exact (.abba)',
      about: 'Each warp as up to 1,089 landmarks, about 2.5 µm from LangSlice’s. Open with File > Load State.' },
    { kind: 'abba', landmarks: EDITABLE, name: 'ABBA project, editable (.abba)',
      about: 'Each warp as 100 landmarks, easy to edit in BigWarp; about 45 µm from LangSlice’s.' },
    { kind: 'quicknii', name: 'QuickNII (.json)',
      about: 'The linear placement of each section, with its image.' },
    { kind: 'visualign', name: 'VisuAlign (.json)',
      about: 'The placement and each warp as VisuAlign markers.' },
  ];
  const NAMES = { abba: 'ABBA project', quicknii: 'QuickNII file', visualign: 'VisuAlign file' };
  const POLL_MS = 700;
  let job = null;
  let root = null;
  let button = null;
  let menu = null;
  let note = null;
  let timer = null;
  let home = '';  // the home folder, written ~ in the path shown
  const shown = path => (home && path.toLowerCase().startsWith(home.toLowerCase())
    ? '~' + path.slice(home.length) : path);

  function el(tag, cls, text) {
    const node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  async function post(resource, body) {
    const r = await fetch(resource, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}), cache: 'no-store' });
    const answer = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(answer.error || `The server answered ${r.status}.`);
    return answer;
  }

  function closeMenu() {
    if (!menu) return;
    menu.hidden = true;
    button.setAttribute('aria-expanded', 'false');
  }
  function openMenu() {
    note.hidden = true;
    menu.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    menu.querySelector('button')?.focus();
  }

  function closeButton() {
    const close = el('button', '', 'Close');
    close.addEventListener('click', () => { note.hidden = true; });
    return close;
  }

  function show(state) {
    note.replaceChildren();
    note.className = `export-note ${state.state}`;
    const name = NAMES[state.format] || 'export';
    if (state.state === 'running') {
      note.append(el('div', 'export-note-title', `Writing the ${name}…`));
      if (state.format === 'abba') {
        note.append(el('div', 'export-note-message', 'Each warp becomes landmark pairs; this can take a few minutes.'));
      }
    } else if (state.state === 'done') {
      note.append(el('div', 'export-note-title', `${name} written`));
      note.append(el('div', 'export-note-path mono', shown(state.path)));
      const left = state.skipped ? `; ${state.skipped} left out (abba_export.json says why)` : '';
      note.append(el('div', 'export-note-message', `${state.message}${left}, in ${state.seconds} s.`));
      const actions = el('div', 'export-note-actions');
      const open = el('button', '', 'Open folder');
      open.addEventListener('click', () => post(`jobs/${job}/export/reveal`).catch(e => fail(e.message)));
      const copy = el('button', '', 'Copy path');
      copy.addEventListener('click', () => {
        navigator.clipboard?.writeText(state.path).then(() => { copy.textContent = 'Copied'; });
      });
      actions.append(open, copy, closeButton());
      note.append(actions);
    } else if (state.state === 'failed') {
      note.append(el('div', 'export-note-title', `The ${name} was not written`));
      note.append(el('div', 'export-note-message', state.message));
      const actions = el('div', 'export-note-actions');
      actions.append(closeButton());
      note.append(actions);
    }
    note.hidden = state.state === 'idle';
    button.disabled = state.state === 'running';
  }
  function fail(message) { show({ state: 'failed', format: '', message }); }

  async function poll(index) {
    clearTimeout(timer);
    if (index !== job) return;
    try {
      const r = await fetch(`jobs/${index}/export`, { cache: 'no-store' });
      if (!r.ok || index !== job) return;
      const state = await r.json();
      // An export finished before this page opened is not news.
      if (state.state !== 'running' && !note.dataset.asked) return;
      show(state);
      if (state.state === 'running') timer = setTimeout(() => poll(index), POLL_MS);
    } catch (_) {
      timer = setTimeout(() => poll(index), POLL_MS * 3);
    }
  }

  async function start(kind, landmarks) {
    closeMenu();
    const index = job;
    note.dataset.asked = '1';
    try {
      show(await post(`jobs/${index}/export`, { format: kind, landmarks: landmarks || EXACT }));
      poll(index);
    } catch (error) {
      fail(error.message);
    }
  }

  // The menu's last row: an ABBA project with a landmark count of the person's own.
  function otherCount() {
    const row = el('div', 'export-other');
    const label = el('label', 'export-other-label', 'ABBA, landmarks per warp');
    const input = el('input', 'export-other-count');
    input.type = 'number';
    input.min = String(MIN_LANDMARKS);
    input.max = String(EXACT);
    input.step = '1';
    input.value = '200';
    input.id = 'export-landmarks';
    label.htmlFor = input.id;
    const go = el('button', '', 'Export');
    go.setAttribute('role', 'menuitem');
    const send = () => {
      const count = Math.round(Number(input.value));
      if (count >= MIN_LANDMARKS && count <= EXACT) start('abba', count);
      else fail(`A warp takes ${MIN_LANDMARKS} to ${EXACT} landmarks.`);
    };
    go.addEventListener('click', send);
    input.addEventListener('keydown', event => { if (event.key === 'Enter') send(); });
    row.append(label, input, go);
    return row;
  }

  function build() {
    root = document.getElementById('export');
    if (!root) return;
    button = el('button', 'export-button', 'Export');
    button.append(el('span', 'caret', '⌄'));
    button.setAttribute('aria-haspopup', 'menu');
    button.setAttribute('aria-expanded', 'false');
    button.title = "Write this job's registration as a file for another program";
    menu = el('div', 'export-menu');
    menu.setAttribute('role', 'menu');
    menu.hidden = true;
    for (const format of FORMATS) {
      const item = el('button', 'export-item');
      item.setAttribute('role', 'menuitem');
      item.append(el('span', 'export-item-name', format.name), el('span', 'export-item-about', format.about));
      item.addEventListener('click', () => start(format.kind, format.landmarks));
      menu.append(item);
    }
    menu.append(otherCount());
    note = el('div', 'export-note');
    note.setAttribute('role', 'status');
    note.hidden = true;
    root.replaceChildren(button, menu, note);
    button.addEventListener('click', () => (menu.hidden ? openMenu() : closeMenu()));
    document.addEventListener('pointerdown', event => {
      if (!event.target.closest('#export')) closeMenu();
    });
    document.addEventListener('keydown', event => { if (event.key === 'Escape') closeMenu(); });
    if (job !== null) setJob(job, home);
  }

  // app.js names the job shown, and the home folder.
  function setJob(index, homeFolder) {
    job = index;
    home = String(homeFolder || '');
    if (!root) return;
    root.classList.toggle('off', index === null);
    note.hidden = true;
    delete note.dataset.asked;
    button.disabled = false;
    if (index !== null) poll(index);
  }

  document.addEventListener('DOMContentLoaded', build);
  return { setJob, FORMATS };
})();
