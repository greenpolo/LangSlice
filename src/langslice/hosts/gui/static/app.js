'use strict';
// The LangSlice window's shell: the tabs and the address (`#tab=<tab>&job=<n>`, which
// always names what the window shows), the job picker, the polling of the local
// server (the job list every 2 s; the shown job's snapshot every 300 ms, and its
// positioning strip when its sections change), the status line and the keyboard. The
// agent viewer's parts are viewer-strip.js (Strip), viewer.js (Viewer) and
// viewer-log.js (AgentLog); they start when the page has loaded.

// The tab the address names when the window opens (`langslice gui --tab prompt`:
// `#tab=prompt&job=<n>`). settings.js reads it too, so it stays a global.
const startTab = new URLSearchParams(location.hash.slice(1)).get('tab');

// Small helpers the viewer's scripts share.
const Ui = (() => {
  const el = (tag, cls, text) => {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text !== undefined && text !== null) e.textContent = text;
    return e;
  };
  // Per-browser layout choices (divider positions); a page without storage keeps defaults.
  const store = {
    get(key) { try { return localStorage.getItem('langslice.' + key); } catch { return null; } },
    set(key, value) { try { localStorage.setItem('langslice.' + key, String(value)); } catch { /* not kept */ } },
  };
  // A draggable divider: `move(dx, dy)` from where the drag started; arrow keys nudge it.
  function drag(divider, { start, move, end }) {
    divider.addEventListener('pointerdown', event => {
      if (event.button !== 0) return;
      event.preventDefault();
      divider.setPointerCapture(event.pointerId);
      divider.classList.add('dragging');
      const x0 = event.clientX, y0 = event.clientY;
      start?.();
      const onMove = e => move(e.clientX - x0, e.clientY - y0);
      const onUp = () => {
        divider.classList.remove('dragging');
        divider.removeEventListener('pointermove', onMove);
        divider.removeEventListener('pointerup', onUp);
        divider.removeEventListener('pointercancel', onUp);
        end?.();
      };
      divider.addEventListener('pointermove', onMove);
      divider.addEventListener('pointerup', onUp);
      divider.addEventListener('pointercancel', onUp);
    });
    divider.addEventListener('keydown', event => {
      const step = event.shiftKey ? 40 : 10;
      const d = { ArrowUp: [0, -step], ArrowDown: [0, step], ArrowLeft: [-step, 0], ArrowRight: [step, 0] }[event.key];
      if (!d) return;
      event.preventDefault();
      start?.(); move(d[0], d[1]); end?.();
    });
  }
  function ago(seconds) {
    if (!seconds) return 'no files yet';
    const s = Date.now() / 1000 - seconds;
    if (s < 60) return 'just now';
    if (s < 3600) return Math.round(s / 60) + ' min ago';
    if (s < 86400) return Math.round(s / 3600) + ' h ago';
    return Math.round(s / 86400) + ' d ago';
  }
  // A long text cut in the middle (a path keeps its end, where the job's own folder is).
  function middle(text, max) {
    if (text.length <= max || max < 8) return text;
    const head = Math.ceil((max - 1) * 0.35);
    return text.slice(0, head) + '…' + text.slice(text.length - (max - 1 - head));
  }
  // A number of millimetres as the window writes it.
  function mm(value, digits = 2) {
    return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(digits) : '';
  }
  return { el, store, drag, ago, middle, mm };
})();

(() => {
  const $ = id => document.getElementById(id);
  const { el } = Ui;
  const TABS = ['viewer', 'prompt', 'settings'];
  const SNAPSHOT_MS = 300, JOBS_MS = 2000, STRIP_MS = 1500;

  let tab = TABS.includes(startTab) ? startTab : 'viewer';
  let jobs = null;       // the job list; null until the server first answers
  let current = null;    // the index of the job shown
  let chosen = false;    // the person (or the address) named the job; else the newest is shown
  let watch = null;      // the shown job's snapshot cursor
  let strip = null;      // the shown job's strip requests

  // --- the tabs and the address ------------------------------------------------------
  const addressed = () => new URLSearchParams(location.hash.slice(1));
  function addressJob() {
    const value = addressed().get('job');
    return value !== null && /^[0-9]+$/.test(value) ? Number(value) : null;
  }
  // The job the address named when the window opened, kept until the job list arrives.
  let named = addressJob();
  function writeAddress() {
    const params = new URLSearchParams();
    params.set('tab', tab);
    const job = current ?? named;
    if (job !== null) params.set('job', String(job));
    const hash = '#' + params.toString();
    if (location.hash !== hash) history.replaceState(null, '', hash);
  }
  function showTab(name) {
    tab = name;
    for (const button of document.querySelectorAll('[data-tab]')) {
      const on = button.dataset.tab === name;
      button.classList.toggle('active', on);
      if (on) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    }
    for (const box of document.querySelectorAll('.tab')) box.hidden = box.id !== 'tab-' + name;
    $('picker').classList.toggle('off', name !== 'viewer');
    closePicker();
    writeAddress();
    if (name === 'viewer') Viewer.shown();
  }
  for (const button of document.querySelectorAll('[data-tab]')) {
    button.addEventListener('click', () => showTab(button.dataset.tab));
  }
  window.addEventListener('hashchange', () => {
    const named = addressed().get('tab');
    if (TABS.includes(named) && named !== tab) showTab(named);
    const job = addressJob();
    if (job !== null && job !== current && jobs?.some(j => j.index === job)) { chosen = true; select(job); }
  });

  // --- the job picker -------------------------------------------------------------------
  const changedOf = job => Number(job?.changed_at ?? job?.changed ?? 0) || 0;
  const pathOf = job => String(job?.path ?? job?.folder ?? '');
  let active = null;  // the option the keyboard is on
  function renderPicker() {
    const job = jobs?.find(j => j.index === current);
    $('picker-name').textContent = job ? job.name : jobs ? 'No jobs' : 'Looking for jobs…';
    $('picker-dot').className = 'dot ' + (job ? job.status : 'none');
    $('picker-button').title = job ? pathOf(job) : '';
    $('picker-button').disabled = !jobs?.length;
    if (!$('picker-list').hidden) renderList();
  }
  function ordered() { return [...(jobs || [])].sort((a, b) => changedOf(b) - changedOf(a)); }
  function renderList() {
    const list = $('picker-list');
    list.replaceChildren();
    const width = Math.max(200, list.clientWidth - 2 * 12 - 20);
    const chars = Math.floor(width / 6.7);
    for (const job of ordered()) {
      const option = el('div', 'job-option');
      option.id = 'job-option-' + job.index;
      option.setAttribute('role', 'option');
      option.setAttribute('aria-selected', String(job.index === current));
      option.classList.toggle('active', job.index === active);
      option.title = pathOf(job);
      option.append(el('span', 'dot ' + job.status), el('span', 'job-option-name', job.name),
        el('span', 'job-option-word word-' + job.status, job.status),
        el('span', 'job-option-path', Ui.middle(pathOf(job), chars)),
        el('span', 'job-option-facts', [job.atlas, Ui.ago(changedOf(job))].filter(Boolean).join(' · ')));
      option.addEventListener('click', () => { choose(job.index); closePicker(true); });
      list.append(option);
    }
    if (!list.children.length) list.append(el('p', 'picker-none', 'No jobs yet.'));
    if (active !== null) list.setAttribute('aria-activedescendant', 'job-option-' + active);
    else list.removeAttribute('aria-activedescendant');
  }
  function openPicker() {
    if (!jobs?.length) return;
    active = current;
    $('picker-list').hidden = false;
    $('picker-button').setAttribute('aria-expanded', 'true');
    renderList();
    $('picker-list').focus();
    document.getElementById('job-option-' + active)?.scrollIntoView({ block: 'nearest' });
  }
  function closePicker(refocus) {
    if ($('picker-list').hidden) return;
    $('picker-list').hidden = true;
    $('picker-button').setAttribute('aria-expanded', 'false');
    if (refocus) $('picker-button').focus();
  }
  $('picker-button').addEventListener('click', () => ($('picker-list').hidden ? openPicker() : closePicker(true)));
  $('picker-list').addEventListener('keydown', event => {
    const list = ordered();
    let i = list.findIndex(j => j.index === active);
    if (event.key === 'ArrowDown') i = Math.min(list.length - 1, i + 1);
    else if (event.key === 'ArrowUp') i = Math.max(0, i - 1);
    else if (event.key === 'Home') i = 0;
    else if (event.key === 'End') i = list.length - 1;
    else if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (active !== null) choose(active);
      closePicker(true);
      return;
    } else if (event.key === 'Escape' || event.key === 'Tab') { closePicker(event.key === 'Escape'); return; }
    else return;
    event.preventDefault();
    active = list[Math.max(0, i)]?.index ?? null;
    renderList();
    document.getElementById('job-option-' + active)?.scrollIntoView({ block: 'nearest' });
  });
  document.addEventListener('pointerdown', event => {
    if (!event.target.closest('#picker')) closePicker(false);
  });

  // --- the job shown --------------------------------------------------------------------
  function choose(index) { chosen = true; select(index); }
  function select(index) {
    if (current === index && watch) return;
    current = index;
    watch = { index, gen: 0, cursor: 0, sv: -1 };
    strip = { busy: false, again: false, last: 0, timer: null, missing: false };
    Viewer.setJob(index);
    AgentLog.setJob(index);
    clearStatus();
    writeAddress();
    renderPicker();
    fetchStrip();
  }
  // The job the address names; else, until the person picks one, the most recently changed.
  function pickJob() {
    if (!jobs.length) return;
    const known = index => jobs.some(j => j.index === index);
    if (current === null && named !== null && known(named)) { chosen = true; select(named); return; }
    named = null;
    if (current !== null && known(current) && chosen) return;
    const newest = jobs.reduce((a, b) => (changedOf(b) > changedOf(a) ? b : a));
    const shown = jobs.find(j => j.index === current);
    if (!shown || (newest.index !== current && changedOf(newest) > changedOf(shown) + 5)) select(newest.index);
  }
  async function pollJobs() {
    try {
      const r = await fetch('jobs', { cache: 'no-store' });
      if (r.ok) {
        jobs = (await r.json()).jobs || [];
        pickJob();
        renderPicker();
        const none = !jobs.length;
        $('no-jobs').hidden = !none;
        $('viewer-body').hidden = none;
        $('statusline').hidden = none;
      }
    } catch { /* the snapshot poll reports the connection */ }
    setTimeout(pollJobs, JOBS_MS);
  }

  // --- the snapshot -----------------------------------------------------------------------
  async function pollSnapshot() {
    const w = watch;
    try {
      if (w) {
        const r = await fetch(`jobs/${w.index}/snapshot?since=${w.cursor}&gen=${w.gen}&sv=${w.sv}`, { cache: 'no-store' });
        if (!r.ok) throw new Error(String(r.status));
        const snap = await r.json();
        if (w === watch) take(snap);
      }
      $('connection').textContent = 'Connected locally';
    } catch {
      $('connection').textContent = 'Waiting for LangSlice…';
    }
    setTimeout(pollSnapshot, SNAPSHOT_MS);
  }
  function take(snap) {
    if (snap.reset && watch.gen !== 0) {
      // The folder's files were replaced (a new watch): start the job's view over.
      Viewer.setJob(watch.index);
      AgentLog.setJob(watch.index);
    }
    watch.gen = snap.generation;
    watch.cursor = snap.cursor;
    AgentLog.add(snap.items || []);
    const tables = snap.sections !== undefined;
    if (tables) watch.sv = snap.state_version;
    Viewer.update(tables ? snap : null, snap.focus || null);
    if (tables) fetchStrip();
    showStatus(snap);
  }
  function clearStatus() {
    $('status-dot').className = 'dot none';
    $('word').textContent = '';
    $('sentence').textContent = 'Loading the job…';
    $('facts').textContent = '';
    AgentLog.setStatus('');
  }
  function showStatus(snap) {
    const word = snap.status?.word || '';
    const h = snap.header || {};
    $('status-dot').className = 'dot ' + word;
    $('word').textContent = word;
    $('word').className = 'word word-' + word;
    $('sentence').textContent = snap.status?.sentence || '';
    $('facts').textContent = [h.atlas, h.plane, h.model, h.door && 'via ' + h.door].filter(Boolean).join(' · ');
    AgentLog.setStatus(word);
    Viewer.setHeader(h);
    const job = jobs?.find(j => j.index === watch.index);
    if (job && job.status !== word) { job.status = word; renderPicker(); }
  }

  // --- the positioning strip's data -------------------------------------------------------
  async function fetchStrip() {
    const w = watch, s = strip;
    if (!w || s.missing) return;
    if (s.busy) { s.again = true; return; }
    const wait = s.last + STRIP_MS - Date.now();
    if (wait > 0) {
      if (!s.timer) s.timer = setTimeout(() => { s.timer = null; if (s === strip) fetchStrip(); }, wait);
      return;
    }
    s.busy = true;
    s.last = Date.now();
    try {
      const r = await fetch(`jobs/${w.index}/strip`, { cache: 'no-store' });
      if (s !== strip) return;
      if (r.ok) Strip.setInfo(await r.json());
      else if (r.status === 404) { s.missing = true; Strip.setInfo(null, ''); }
      else {
        const answer = await r.json().catch(() => ({}));
        Strip.setInfo(null, answer.error || '');
      }
    } catch { /* tried again when the sections change */ }
    finally {
      s.busy = false;
      if (s.again && s === strip) { s.again = false; fetchStrip(); }
    }
  }

  // --- the keyboard -----------------------------------------------------------------------
  document.addEventListener('keydown', event => {
    if (tab !== 'viewer' || event.ctrlKey || event.metaKey || event.altKey) return;
    if (event.target.closest?.('input, textarea, select, [contenteditable], .picker-list')) return;
    if ($('lightbox').open) return;
    if (event.key === '[') Viewer.pageBy(-1);
    else if (event.key === ']') Viewer.pageBy(1);
    else if (event.key === 'f' || event.key === 'F') Viewer.toggleFollow();
    else return;
    event.preventDefault();
  });

  document.addEventListener('DOMContentLoaded', () => {
    Viewer.init();
    AgentLog.init();
    showTab(tab);
    clearStatus();
    pollJobs();
    pollSnapshot();
  });
})();
