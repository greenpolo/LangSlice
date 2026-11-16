'use strict';
// The Export menu: the shown job's registration as an ABBA project, a QuickNII file or a
// VisuAlign file, written by the server under the job's exports/ folder
// (hosts/gui/export.py). The page starts an export, polls its state while it runs and then
// says where the file is, with Open folder and Copy path.
const Exporter = (() => {
  const FORMATS = [
    { kind: 'abba', name: 'ABBA project (.abba)',
      about: 'Open in ABBA with File > Load State: each section with its affine and its warp.' },
    { kind: 'quicknii', name: 'QuickNII (.json)',
      about: 'The linear placement of each section, with its image.' },
    { kind: 'visualign', name: 'VisuAlign (.json)',
      about: 'The placement and each warp as VisuAlign markers.' },
  ];
  const NAMES = Object.fromEntries(FORMATS.map(f => [f.kind, f.name]));
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

  async function start(kind) {
    closeMenu();
    const index = job;
    note.dataset.asked = '1';
    try {
      show(await post(`jobs/${index}/export`, { format: kind }));
      poll(index);
    } catch (error) {
      fail(error.message);
    }
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
      item.addEventListener('click', () => start(format.kind));
      menu.append(item);
    }
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
